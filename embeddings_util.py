import cv2
import numpy as np
import onnxruntime as ort
import time
from PIL import Image
import logging
from facedet import FaceDetector

# Suppress onnxruntime warnings
ort.set_default_logger_verbosity(3)

# Initialize the ONNX session (you might need to adjust the path)
models = ["edgeface", "arcface"]


def load_model(model_name):

    if model_name not in models:
        raise ValueError(f"Model {model_name} not found")
    if model_name == "edgeface":
        onnx_path = "models/edgeface_xs_gamma_06.onnx"
        ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print("Using EdgeFace model")
    elif model_name == "arcface":
        onnx_path = "models/arcface_mobilefacenet.onnx"
        ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print("Using ArcFace model")
    used_provider = ort_session.get_providers()[0]
    print(f'Embedding: Using provider: {used_provider}')
    return ort_session

def get_embedding(image, ort_session):
    if not isinstance(ort_session, ort.InferenceSession):
        raise ValueError("ort_session must be an onnxruntime.InferenceSession object")

    # Ensure image is in numpy array format
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Resize image to 112x112
    if image.shape[:2] != (112, 112):
        image = cv2.resize(image, (112, 112))

    # Check if the image is already in float format
    if image.dtype != np.uint8:
        face_np = image
    else:
        # Convert to RGB if necessary
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        face_np = image.astype(np.float32) / 255.0

    # Ensure the image is in the correct shape (1, 3, 112, 112)
    if face_np.shape != (1, 3, 112, 112):
        face_np = face_np.transpose((2, 0, 1))  # Change to (3, 112, 112)
        face_np = np.expand_dims(face_np, axis=0)  # Add batch dimension

    # Normalize to [-1, 1]
    face_np = (face_np - 0.5) / 0.5
    
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

def generate_embedding_from_image(image_path, face_embedder):
    image = cv2.imread(image_path)
    face_detector = FaceDetector()
    faces = face_detector.infer(image, threshold=0.3)
    if faces:
        face = faces[0]
        x1, y1, x2, y2 = face["points"]
        face_img = image[y1:y2, x1:x2]
        embedding, _ = get_embedding(face_img, face_embedder)
        return embedding
    return None


def load_embedding_from_npz(npz_path):
    with np.load(npz_path) as data:
        embedding = data['embeddings'][0]  # Assuming we want the first embedding
    return embedding