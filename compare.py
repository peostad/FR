import numpy as np
import torch
import torch.nn.functional as F
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load embeddings from .npz files
embedding_file1 = 'embeddings/pey1_embedding.npz'
embedding_file2 = 'embeddings/pey2_embedding.npz'

# Load the embeddings
embedding1 = np.load(embedding_file1)['embedding']
embedding2 = np.load(embedding_file2)['embedding']

# Log the shape and data type of the embeddings
logging.info(f"Embedding 1 shape: {embedding1.shape}, data type: {embedding1.dtype}")
logging.info(f"Embedding 2 shape: {embedding2.shape}, data type: {embedding2.dtype}")

# Compute cosine similarity
similarity = F.cosine_similarity(torch.tensor(embedding1).unsqueeze(0), torch.tensor(embedding2).unsqueeze(0))

print(f"Cosine similarity between the two embeddings: {similarity.item()}")

# Optionally, you can set a threshold for face matching
threshold = 0.7  # This is an example threshold, adjust as needed
if similarity > threshold:
    print("The faces match!")
else:
    print("The faces do not match.")
