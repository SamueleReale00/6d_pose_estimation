import os
import shutil
import yaml
import gdown
import zipfile
import numpy as np
from tqdm import tqdm
from dataset.data_loader import MultiObjectLineModDataset

# --- DEFAULT CONSTANTS ---
DEFAULT_LINEMOD_PATH = "/content/datasets/linemod/Linemod_preprocessed/data"
DEFAULT_YOLO_PATH = "/content/yolo_dataset"
DEFAULT_YAML_PATH = "/content/linemod.yaml"

def convert_linemod_to_yolo(source_path=DEFAULT_LINEMOD_PATH, output_path=DEFAULT_YOLO_PATH, train_split=0.8):
    """
    Converts LineMOD data to YOLO structure.
    Args:
        source_path: Path to the raw 'data' folder (numbered 01, 02...).
        output_path: Where to create the 'train' and 'val' folders.
        train_split: Percentage of data to use for training (0.0 - 1.0).
    """
    print(f"🔄 Converting LineMOD from {source_path} to YOLO format at {output_path}...")
    
    # 1. Setup Folders
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)

    # 2. Load Dataset
    if not os.path.exists(source_path):
        print(f"❌ Error: Source path not found: {source_path}")
        return

    dataset = MultiObjectLineModDataset(root_dir=source_path, transform=None)

    # 3. Conversion Loop
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        
        img_path = sample['image_path']
        bbox = sample['bbox']
        obj_id = sample['obj_id']
        img_id = sample['image_id']
        
        # Get dimensions
        _, h_img, w_img = sample['image'].shape 
        
        # Normalize
        x, y, w, h = bbox
        cx = (x + w / 2) / w_img
        cy = (y + h / 2) / h_img
        nw = w / w_img
        nh = h / h_img
        
        # Map Class (1->0)
        class_id = obj_id - 1
        
        # Split Logic
        split = 'train' if (hash(f"{obj_id}_{img_id}") % 100) < (train_split * 100) else 'val'
        
        # Save
        unique_name = f"{obj_id:02d}_{img_id:04d}"
        
        # Copy Image
        shutil.copy(img_path, os.path.join(output_path, split, 'images', f"{unique_name}.jpg"))
        
        # Save Label
        with open(os.path.join(output_path, split, 'labels', f"{unique_name}.txt"), 'w') as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    print("✅ Conversion Complete.")

def download_yolo_dataset(drive_link, output_path=DEFAULT_YOLO_PATH):
    """
    Downloads a pre-converted YOLO dataset zip from Drive.
    """
    print(f"⬇️ Downloading YOLO dataset to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    
    zip_path = os.path.join(output_path, "yolo_dataset.zip")
    
    # Download
    gdown.download(drive_link, zip_path, quiet=False, fuzzy=True)
    
    # Extract
    if os.path.exists(zip_path):
        print("📦 Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        # os.remove(zip_path) # Optional cleanup
        print("✅ YOLO Dataset Ready.")
    else:
        print("❌ Download failed.")

def create_yolo_yaml(dataset_path=DEFAULT_YOLO_PATH, output_yaml_path=DEFAULT_YAML_PATH):
    """
    Creates the data.yaml file required by YOLO training.
    """
    config = {
        'path': dataset_path,
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'Ape',
            1: 'Benchvise',
            2: 'Bowl',        # Missing folder 03 
            3: 'Cam',
            4: 'Can',         # This is the "Waterer" (Red can)
            5: 'Cat',
            6: 'Cup',         # Missing folder 07 
            7: 'Driller',
            8: 'Duck',
            9: 'Eggbox',
            10: 'Glue',
            11: 'Holepuncher',
            12: 'Iron',
            13: 'Lamp',
            14: 'Phone'
        }
    }
    
    with open(output_yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print(f"✅ YAML config created at: {output_yaml_path}")