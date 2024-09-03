import torch
from torchvision import transforms
from EdgeFace.backbones import get_model
import numpy as np
import logging
import torch.nn.functional as F
import os
import cv2
import time
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)

# Check for GPU availability
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
yolo_model.eval()

def get_embedding(image):
    # Face Detection
    detection_start = time.time()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = yolo_model(image_rgb)
    detection_time = (time.time() - detection_start) * 1000
    
    # Filter for person class (index 0 in COCO dataset)
    person_detections = results.xyxy[0][results.xyxy[0][:, -1] == 0]
    
    if len(person_detections) == 0:
        logging.warning("No person detected")
        return None, detection_time, None, None
    
    # Face Alignment (using the first detected person)
    alignment_start = time.time()
    box = person_detections[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box[:4])
    face_img = image[y1:y2, x1:x2]
    
    # Resize and prepare for the embedding model
    face_tensor = transform(Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    alignment_time = (time.time() - alignment_start) * 1000
    
    # Face Embedding
    embedding_start = time.time()
    with torch.no_grad():
        embedding = model(face_tensor)
    embedding_time = (time.time() - embedding_start) * 1000
    
    return embedding.squeeze().cpu(), detection_time, alignment_time, embedding_time

arch = "edgeface_xs_gamma_06"  # or edgeface_xs_gamma_06
model = get_model(arch)
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

checkpoint_path = f'EdgeFace/checkpoints/{arch}.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Paths for the first image
image_path1 = 'EdgeFace/checkpoints/pey1.jpg'

# Get embedding for the first image
image1 = cv2.imread(image_path1)
if image1 is None:
    logging.error(f"Failed to load image: {image_path1}")
    exit()

embedding1, _, _, _ = get_embedding(image1)
if embedding1 is None:
    logging.error("No face detected in the first image. Please use an image with a clear face.")
    exit()

embedding1 = F.normalize(embedding1, p=2, dim=0).numpy()
logging.info(f"Embedding 1 shape: {embedding1.shape}, data type: {embedding1.dtype}")

# Initialize video capture for live camera feed
cap = cv2.VideoCapture(0)

# Variables for FPS calculation
frame_count = 0
start_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to capture image from camera")
        break
    
    embedding2, detection_time, alignment_time, embedding_time = get_embedding(frame)
    
    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    
    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Det: {detection_time:.2f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    if embedding2 is None:
        cv2.putText(frame, "No face detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        embedding2 = F.normalize(embedding2, p=2, dim=0).numpy()
        
        # Compute cosine similarity between the first image and the current frame
        similarity = torch.nn.functional.cosine_similarity(torch.tensor(embedding1).unsqueeze(0), torch.tensor(embedding2).unsqueeze(0))
        similarity_score = similarity.item()
        
        # Display the similarity score on the frame
        cv2.putText(frame, f"Similarity: {similarity_score:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Optionally, you can set a threshold for face matching
        threshold = 0.7  # This is an example threshold, adjust as needed
        if similarity_score > threshold:
            cv2.putText(frame, "Match!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Match", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display alignment and embedding times
        cv2.putText(frame, f"Align: {alignment_time:.2f}ms", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Embed: {embedding_time:.2f}ms", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Live Camera Feed', frame)
    
    # Print debugging information in the terminal
    if embedding2 is None:
        print(f"No face detected. Detection time: {detection_time:.2f}ms, FPS: {fps:.2f}")
    else:
        total_time = detection_time + alignment_time + embedding_time
        print(f"Detection: {detection_time:.2f}ms, Alignment: {alignment_time:.2f}ms, Embedding: {embedding_time:.2f}ms, Total: {total_time:.2f}ms, FPS: {fps:.2f}")
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




