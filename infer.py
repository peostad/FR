'''
This is the main code for face detection and verification
It uses YoloV7 for face detection and the EdgeFace model for face verification
Both models are in ONNX format and use CUDA for GPU acceleration

TODO:
1. Add the code for the face alignment


Author: Peyman Ostad
Date: 2024-09-06+
'''

import cv2
import numpy as np
import onnxruntime as ort
import torch
import time
from PIL import Image
from facedet import FaceDetector
import logging
import argparse
from embeddings_util import get_embedding, generate_embedding_from_image, load_embedding_from_npz, load_model


# Configure logging
logging.basicConfig(level=logging.INFO)

# Existing code from infer.py
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def main():
    parser = argparse.ArgumentParser(description='Face Detection and Recognition')
    parser.add_argument('--npz', type=str, help='Path to .npz file containing embeddings')
    parser.add_argument('--image', type=str, help='Path to image file for generating embedding')
    parser.add_argument('--embed', type=str, help='Path to face embedding model file',default="edgeface")
    args = parser.parse_args()
    face_detector = FaceDetector()
    face_embedder = load_model(args.embed)

    cap = cv2.VideoCapture(0)  # 0 for default camera

    frame_count = 0
    start_time = time.time()
    fps = 0  # Initialize fps variable
    
    # Load pey1_embedding based on arguments
    if args.npz:
        source_embedding = load_embedding_from_npz(args.npz)
    elif args.image:
        source_embedding = generate_embedding_from_image(args.image, face_embedder)
    else:
        source_embedding = generate_embedding_from_image("EdgeFace/checkpoints/pey1.jpg", face_embedder)
    
    # Normalize source_embedding if it's not already normalized
    source_embedding = source_embedding / np.linalg.norm(source_embedding)
    

    
    # Convert source_embedding to PyTorch tensor
    source_embedding_tensor = torch.from_numpy(source_embedding).to(device)
    
    # Check and log the format of source_embedding
    # logging.info(f"source_embedding shape: {source_embedding.shape}")
    # logging.info(f"source_embedding dtype: {source_embedding.dtype}")
    # logging.info(f"source_embedding norm: {np.linalg.norm(source_embedding)}")
    # logging.info(f"source_embedding first 5 values: {source_embedding[:5]}")
    
    previous_embedding = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()

        detection_start = time.time()
        faces = face_detector.infer(frame, threshold=0.3)
        detection_time = (time.time() - detection_start) * 1000

        total_embedding_time = 0
        for face in faces:
            x1, y1, x2, y2 = face["points"]
            confidence = float(face["confidence"])
            color = (0, 255, 0) 
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"Face: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)
            
            face_img = frame[y1:y2, x1:x2]
            cv2.imshow("Face Detection", face_img)
            embedding, embedding_time = get_embedding(face_img, face_embedder)
            total_embedding_time += embedding_time
            if embedding is not None:
                # Log current frame embedding information
                # logging.info(f"Current frame embedding shape: {embedding.shape}")
                # logging.info(f"Current frame embedding dtype: {embedding.dtype}")
                # logging.info(f"Current frame embedding norm: {np.linalg.norm(embedding)}")
                
                # # Log first 5 values of current frame embedding
                # logging.info(f"Current frame embedding first 5 values: {embedding[:5]}")
                
                # Convert current frame embedding to PyTorch tensor
                embedding_tensor = torch.from_numpy(embedding).to(device)
                
                # Compare embeddings using PyTorch cosine similarity
                similarity = torch.nn.functional.cosine_similarity(source_embedding_tensor.unsqueeze(0), embedding_tensor.unsqueeze(0))
                similarity_score = similarity.item()
                
                # Log similarity score
                logging.info(f"Similarity score: {similarity_score:.4f}")
                
                similarity_scores = []
                window_size = 5

                # In the main loop
                similarity_scores.append(similarity_score)
                if len(similarity_scores) > window_size:
                    similarity_scores.pop(0)
                average_similarity = sum(similarity_scores) / len(similarity_scores)

                if average_similarity > 0.7:  # Use the average similarity
                    cv2.putText(frame, f"Peyman: {average_similarity:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Not Match: {average_similarity:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # Display times
                cv2.putText(frame, f"Embed: {embedding_time:.2f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                if previous_embedding is not None:
                    frame_similarity = torch.nn.functional.cosine_similarity(
                        torch.from_numpy(previous_embedding).unsqueeze(0),
                        torch.from_numpy(embedding).unsqueeze(0)
                    )
                    logging.info(f"Similarity between consecutive frames: {frame_similarity.item():.4f}")

                previous_embedding = embedding

        # Display detection time, total embedding time, and FPS
        cv2.putText(frame, f"Detect: {detection_time:.2f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        #cv2.putText(frame, f"Total Embed: {total_embedding_time:.2f}ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Face Detection and Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()