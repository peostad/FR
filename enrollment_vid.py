import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import logging
import os
import torch.nn.functional as F
import time
from facedet import FaceDetector
import onnxruntime as ort

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize face detection model
face_detector = FaceDetector()

# Initialize ONNX Runtime session
onnx_path = "edgeface_xs_gamma_06.onnx"
ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def get_embedding(face_img):
    face_tensor = transform(Image.fromarray(face_img)).unsqueeze(0).numpy()
    
    # Generate embedding using ONNX
    ort_inputs = {ort_session.get_inputs()[0].name: face_tensor}
    embedding = ort_session.run(None, ort_inputs)[0].squeeze()
    
    # Normalize the embedding
    embedding = F.normalize(torch.tensor(embedding), p=2, dim=0).numpy()
    
    return embedding

def process_camera_feed():
    embeddings = []
    camera_index = 0
    max_attempts = 3

    for attempt in range(max_attempts):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logging.error(f"Failed to open camera {camera_index}. Trying next...")
            camera_index += 1
            continue

        logging.info(f"Successfully opened camera {camera_index}")
        
        frame_count = 0
        start_time = time.time()
        duration = 10  # Run for 10 seconds

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to grab frame. Skipping...")
                continue

            frame_count += 1
            
            # Detect faces using facedet
            faces = face_detector.infer(frame, threshold=0.3)
            
            for face in faces:
                x1, y1, x2, y2 = face["points"]
                confidence = float(face["confidence"])
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Face: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                face_img = frame[y1:y2, x1:x2]
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                # Generate embedding
                embedding = get_embedding(face_rgb)
                
                # Verify the shape and data type of the embedding
                logging.info(f"Generated embedding shape: {embedding.shape}")
                logging.info(f"Generated embedding data type: {embedding.dtype}")
                
                embeddings.append(embedding)

            # Display the frame
            cv2.imshow('Camera Feed', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if frame_count > 0:
            logging.info(f"Processed {frame_count} frames")
            return np.array(embeddings)

    logging.error("Failed to open any camera after multiple attempts")
    return np.array(embeddings)

# Usage
face_embeddings = process_camera_feed()

# Verify the shape and data type of the embeddings
logging.info(f"Generated embeddings shape: {face_embeddings.shape}")
logging.info(f"Generated embeddings data type: {face_embeddings.dtype}")

# Save the embeddings
np.savez('face_embeddings_vid.npz', embeddings=face_embeddings)
logging.info(f"Processed {len(face_embeddings)} faces and saved embeddings.")