# @title **2.1 Custom dataset code - Multiobjects loader**

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import yaml
import os

class MultiObjectLineModDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Path to the main 'data' folder containing subfolders '01', '02', etc.
            transform (callable, optional): Optional transform.
        """
        self.root_dir = root_dir
        self.transform = transform

        # This list will store metadata for EVERY image found in ALL folders
        # Format: [ {'path': '/.../01/rgb/0000.png', 'bbox': [...], 'obj_id': 1}, ... ]
        self.all_samples = []

        print("Scanning dataset folders...")

        # 1. Loop through possible object folders (1 to 15)
        # LineMOD typically has 15 objects. We iterate to find them.
        for i in range(1, 16):
            folder_name = f"{i:02d}" # Converts 1 -> "01", 2 -> "02"
            folder_path = os.path.join(root_dir, folder_name)

            # Check if folder exists (This automatically skips missing Object 3)
            if not os.path.exists(folder_path):
                continue

            print(f"Loading Object {folder_name}...")

            # 2. Load the GT file for THIS specific folder
            gt_path = os.path.join(folder_path, "gt.yml")
            if not os.path.exists(gt_path):
                print(f"Warning: gt.yml missing in {folder_name}")
                continue

            with open(gt_path, 'r') as f:
                # Load and force integer keys
                folder_gt = yaml.safe_load(f)

            # 3. Process all images in this folder
            # We iterate through the keys in the GT file (which correspond to image IDs)
            for img_id, objects in folder_gt.items():
                # LineMOD images usually have 1 object per image, but gt is a list.
                obj_data = objects[0]

                # Construct full image path
                img_filename = f"{img_id:04d}.png"
                img_full_path = os.path.join(folder_path, "rgb", img_filename)

                # Extract Data
                bbox = np.array(obj_data['obj_bb'], dtype=np.float32)
                obj_id = int(obj_data['obj_id'])

                # Store everything needed to load this sample later
                self.all_samples.append({
                    'path': img_full_path,
                    'bbox': bbox,
                    'obj_id': obj_id,
                    'original_img_id': img_id # Useful for debugging
                })

        #print(f"Total images loaded: {len(self.all_samples)}")

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):

        # 1. Retrieve the metadata we stored in __init__
        sample_info = self.all_samples[idx]

        # 2. Load Image from the stored full path
        image = cv2.imread(sample_info['path'])
        if image is None:
            raise FileNotFoundError(f"Image read error: {sample_info['path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. Apply Transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # 4. Return
        return {
            'image': image,                             # The torch.FloatTensor ->  [3, 480, 640]
            'bbox': sample_info['bbox'],                # NumPy Array           ->  [x_top_left, y_top_left, width, height]
            'obj_id': sample_info['obj_id'],            # The class (e.g., 1 for Ape)
            'image_id': sample_info['original_img_id'], # The image id for the class (e.g., 0)
            'image_path': sample_info['path']           # The file location
        }