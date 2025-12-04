# @title **2.2 Multiobject dataset instance**

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def denormalize(tensor):
    """Reverses ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def visualize_sample(sample, title=None):
    """
    Takes a sample from the DataLoader and plots it with the 2D Bounding Box.
    Args:
        sample (dict): The dictionary returned by __getitem__
        title (str): Optional title for the plot
    """
    img_tensor = sample['image']
    bbox = sample['bbox'] # [x, y, w, h]
    
    # 1. Denormalize
    img_vis = denormalize(img_tensor)
    
    # 2. Convert to Numpy (H, W, C)
    img_np = img_vis.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    
    # 3. Prepare for OpenCV (Convert to 0-255 uint8)
    # Note: We copy() to ensure memory is contiguous
    img_draw = (img_np * 255).astype(np.uint8).copy()
    
    # 4. Draw Box
    if bbox is not None:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # 5. Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(img_draw)
    if title:
        plt.title(title)
    else:
        plt.title(f"Object {sample['obj_id']} | Image: {sample.get('image_id', 'N/A')}")
    plt.axis('off')
    plt.show()

def draw_3d_box(img, K, R, t, dimensions):
    """
    Draws a 3D bounding box on the image.
    Args:
        img (numpy array): The image (0-255, uint8)
        K, R, t: Camera and Pose matrices
        dimensions: (w, h, d) of the object
    """
    # (Paste the draw_3d_box logic we wrote earlier here if you need it for Phase 3)
    pass