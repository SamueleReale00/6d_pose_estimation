# 📦 Dataset Module Documentation

This package handles all data-related operations for the LineMOD 6D Pose Estimation project. It includes tools for downloading the raw data, loading it into PyTorch, and visualizing the results with bounding boxes.

## 🚀 Quick Start (Colab)

If you have cloned the repository into Google Colab, you can import these modules immediately:

```python
# 1. Setup path so Python sees the repo
import sys
import os
sys.path.append('/content/6d_pose_estimation') 

# 2. Import modules
from dataset import download_dataset, MultiObjectLineModDataset, visualize_sample
from torch.utils.data import DataLoader
from torchvision import transforms

# 3. Download Data
download_dataset() 
    # or
download_dataset(url=new_link)

# 4. Create Loader
# Standard ImageNet normalization is required for Pretrained ResNet
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Point to the extracted data folder
data_path = "/content/datasets/linemod/Linemod_preprocessed/data"
dataset = MultiObjectLineModDataset(root_dir=data_path, transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

