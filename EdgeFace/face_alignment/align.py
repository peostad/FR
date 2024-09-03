import sys
import os

from EdgeFace.face_alignment import mtcnn
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime
import numpy as np

mtcnn_model = mtcnn.MTCNN(device='cpu', crop_size=(112, 112))

def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def get_aligned_face(image):
    if isinstance(image, np.ndarray):
        # Convert NumPy array to PIL Image
        img = Image.fromarray(image)
    else:
        # Load image from file path
        img = Image.open(image).convert('RGB')
    
    # Perform face alignment (this is a placeholder, replace with actual alignment code)
    aligned_face = img  # Replace with actual face alignment logic
    
    return aligned_face


