# @title **1.1 Dataset download and extraction**

import os
import gdown

%cd /content

dataset_dir = "/content/datasets/linemod"  # absolute path

# Check if dataset is already present
if os.path.exists(dataset_dir):
    print("Dataset ready to use")
else:
    # Create dataset directory if not exists
    !mkdir -p datasets/linemod/
    %cd datasets/linemod/

    # Download dataset zip from Google Drive
    !gdown --fuzzy https://drive.google.com/file/d/1qQ8ZjUI6QauzFsiF8EpaaI2nKFWna_kQ/view?usp=drive_link -O Linemod_preprocessed.zip
    !unzip Linemod_preprocessed.zip

    print("Dataset downloaded and extracted")