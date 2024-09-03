##Convert the image to tensor and get the embedding
import torch
from torchvision import transforms
from EdgeFace.face_alignment import align
from facenet_pytorch import MTCNN
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

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=False, device='cpu')

def get_embedding(image):
    start_time = time.time()
    
    # Detect and align the face using MTCNN
    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        logging.warning("No face detected")
        return None, (time.time() - start_time) * 1000  # Return None for embedding and time taken in ms
    
    # Crop the face from the image
    x1, y1, x2, y2 = map(int, boxes[0])
    face = image[y1:y2, x1:x2]
    aligned = align.get_aligned_face(face)
    # Convert the face to a tensor
    #face_pil = Image.fromarray(face)
    transformed_input = transform(aligned).unsqueeze(0)
    
    # Get the embedding
    with torch.no_grad():
        embedding = model(transformed_input)
    
    embedding_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    return embedding.squeeze(), embedding_time

arch = "edgeface_xs_gamma_06"  # or edgeface_xs_gamma_06
model = get_model(arch)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

checkpoint_path = f'EdgeFace/checkpoints/{arch}.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location='cuda:0'))
model.eval()

# Paths for the first image
image_path1 = 'EdgeFace/checkpoints/pey1.jpg'

# Get embedding for the first image
image1 = cv2.imread(image_path1)
if image1 is None:
    logging.error(f"Failed to load image: {image_path1}")
    exit()

image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
embedding1, _ = get_embedding(image1_rgb)
if embedding1 is None:
    logging.error("No face detected in the first image")
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
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    embedding2, embedding_time = get_embedding(frame_rgb)
    
    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    
    if embedding2 is None:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        embedding2 = F.normalize(embedding2, p=2, dim=0).numpy()
        
        # Compute cosine similarity between the first image and the current frame
        similarity = torch.nn.functional.cosine_similarity(torch.tensor(embedding1).unsqueeze(0), torch.tensor(embedding2).unsqueeze(0))
        similarity_score = similarity.item()
        
        # Display the similarity score on the frame
        cv2.putText(frame, f"Similarity: {similarity_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Optionally, you can set a threshold for face matching
        threshold = 0.7  # This is an example threshold, adjust as needed
        if similarity_score > threshold:
            cv2.putText(frame, "Match!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Match", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Live Camera Feed', frame)
    
    # Print debugging information in the terminal
    if embedding2 is None:
        print(f"No face detected. Time taken: {embedding_time:.2f}ms, FPS: {fps:.2f}")
    else:
        print(f"Embedding time: {embedding_time:.2f}ms, FPS: {fps:.2f}")
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




