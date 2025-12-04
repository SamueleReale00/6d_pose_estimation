# @title **2.2 Multiobject dataset instance**

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

dataset_root = "/content/datasets/linemod/Linemod_preprocessed/data"

# 1. Define the Transform Pipeline
# ToTensor(): Converts Numpy (H,W,C) 0-255 -> Tensor (C,H,W) 0.0-1.0
# Normalize(): Subtracts Mean and divides by Std (Standard ImageNet values)
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 2. Create Dataset
full_train_dataset = MultiObjectLineModDataset(
    root_dir=dataset_root,
    transform=data_transform
)

# 3. Create Loader
train_loader = DataLoader(full_train_dataset, batch_size=4, shuffle=True, num_workers=2)

# 4. Output
print(f"Successfully loaded {len(full_train_dataset)} images.")

# @title **2.3 Multiobject dataset test**

random_sample_id = torch.randint(low=0, high=len(full_train_dataset), size=(1,))

# Get one sample to check
sample = full_train_dataset[random_sample_id]
print(f"ImageID:\t {sample['image_id']}")
print(f"ObjectID:\t {sample['obj_id']}")
print(f"Image Shape:\t {sample['image'].shape}")
print(f"Bounding Box:\t {sample['bbox']}")
print(f"Image path:\t {sample['image_path']}")