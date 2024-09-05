import cv2
import numpy as np
import onnxruntime as ort
import torch
import time
from PIL import Image
from facedet import FaceDetector

# Suppress onnxruntime warnings
ort.set_default_logger_severity(3)


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
    
    return embedding.squeeze(), embedding_time

def main():
    model = FaceDetector()

    cap = cv2.VideoCapture(0)  # 0 for default camera

    frame_count = 0
    start_time = time.time()
    fps = 0  # Initialize fps variable

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
                # Display times
                cv2.putText(frame, f"Embed: {embedding_time:.2f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display detection time, total embedding time, and FPS
        cv2.putText(frame, f"Detect: {detection_time:.2f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Total Embed: {total_embedding_time:.2f}ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Face Detection and Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()