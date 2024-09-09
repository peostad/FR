import cv2
import numpy as np
import onnxruntime as ort
import torch
import time
from PIL import Image
from facedet import FaceDetector
import logging
import argparse

# Suppress onnxruntime warnings
ort.set_default_logger_verbosity(3)
# Configure logging
logging.basicConfig(level=logging.INFO)

# Existing code from infer.py
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

onnx_path = "edgeface_xs_gamma_06.onnx"
ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def get_embedding(image):
    face_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    face_pil = face_pil.resize((112, 112))
    face_np = np.array(face_pil).transpose((2, 0, 1)).astype(np.float32) / 255.0
    face_np = (face_np - 0.5) / 0.5
    face_np = np.expand_dims(face_np, axis=0)
    
    embedding_start = time.time()
    ort_inputs = {ort_session.get_inputs()[0].name: face_np}
    embedding = ort_session.run(None, ort_inputs)[0]
    embedding_time = (time.time() - embedding_start) * 1000
    
    # Normalize the embedding
    embedding = embedding.squeeze()
    embedding = embedding / np.linalg.norm(embedding)
    
    # Log embedding information
    logging.info(f"Embedding shape: {embedding.shape}")
    logging.info(f"Embedding dtype: {embedding.dtype}")
    logging.info(f"Embedding norm: {np.linalg.norm(embedding)}")
    
    return embedding, embedding_time

def generate_embedding_from_file(image_path):
    image = cv2.imread(image_path)
    face_detector = FaceDetector()
    faces = face_detector.infer(image, threshold=0.3)
    if faces:
        face = faces[0]
        x1, y1, x2, y2 = face["points"]
        face_img = image[y1:y2, x1:x2]
        embedding, _ = get_embedding(face_img)
        return embedding
    return None

def load_embedding_from_npz(npz_path):
    with np.load(npz_path) as data:
        embedding = data['embeddings'][0]  # Assuming we want the first embedding
    return embedding

def main():
    parser = argparse.ArgumentParser(description='Face Detection and Recognition')
    parser.add_argument('--npz', type=str, help='Path to .npz file containing embeddings')
    parser.add_argument('--image', type=str, help='Path to image file for generating embedding')
    args = parser.parse_args()

    model = FaceDetector()

    cap = cv2.VideoCapture(0)  # 0 for default camera

    frame_count = 0
    start_time = time.time()
    fps = 0  # Initialize fps variable
    
    # Load pey1_embedding based on arguments
    if args.npz:
        pey1_embedding = load_embedding_from_npz(args.npz)
    elif args.image:
        pey1_embedding = generate_embedding_from_file(args.image)
    else:
        pey1_embedding = generate_embedding_from_file("EdgeFace/checkpoints/pey1.jpg")
    
    # Normalize pey1_embedding if it's not already normalized
    pey1_embedding = pey1_embedding / np.linalg.norm(pey1_embedding)
    
    # Log pey1_embedding information
    logging.info(f"pey1_embedding shape: {pey1_embedding.shape}")
    logging.info(f"pey1_embedding dtype: {pey1_embedding.dtype}")
    logging.info(f"pey1_embedding norm: {np.linalg.norm(pey1_embedding)}")
    
    # Log first 5 values of pey1_embedding
    logging.info(f"pey1_embedding first 5 values: {pey1_embedding[:5]}")
    
    # Convert pey1_embedding to PyTorch tensor
    pey1_embedding_tensor = torch.from_numpy(pey1_embedding).to(device)
    
    # Check and log the format of pey1_embedding
    logging.info(f"pey1_embedding shape: {pey1_embedding.shape}")
    logging.info(f"pey1_embedding dtype: {pey1_embedding.dtype}")
    logging.info(f"pey1_embedding norm: {np.linalg.norm(pey1_embedding)}")
    
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
        faces = model.infer(frame, threshold=0.3)
        detection_time = (time.time() - detection_start) * 1000

        total_embedding_time = 0
        for face in faces:
            x1, y1, x2, y2 = face["points"]
            confidence = float(face["confidence"])
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"Face: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)
            
            face_img = frame[y1:y2, x1:x2]
            cv2.imshow("Face Detection", face_img)
            embedding, embedding_time = get_embedding(face_img)
            total_embedding_time += embedding_time
            if embedding is not None:
                # Log current frame embedding information
                logging.info(f"Current frame embedding shape: {embedding.shape}")
                logging.info(f"Current frame embedding dtype: {embedding.dtype}")
                logging.info(f"Current frame embedding norm: {np.linalg.norm(embedding)}")
                
                # Log first 5 values of current frame embedding
                logging.info(f"Current frame embedding first 5 values: {embedding[:5]}")
                
                # Convert current frame embedding to PyTorch tensor
                embedding_tensor = torch.from_numpy(embedding).to(device)
                
                # Compare embeddings using PyTorch cosine similarity
                similarity = torch.nn.functional.cosine_similarity(pey1_embedding_tensor.unsqueeze(0), embedding_tensor.unsqueeze(0))
                similarity_score = similarity.item()
                
                # Log similarity score
                logging.info(f"Similarity score: {similarity_score:.4f}")
                
                if similarity_score > 0.5:  # Very low threshold for testing
                    cv2.putText(frame, f"Peyman: {similarity_score:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Not Match: {similarity_score:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
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
        cv2.putText(frame, f"Total Embed: {total_embedding_time:.2f}ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Face Detection and Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()