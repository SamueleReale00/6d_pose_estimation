# @title **1.1 Dataset download and extraction**

import os
import gdown
import zipfile

# Default URL from your project (can be overwritten)
DEFAULT_DATASET_URL = "https://drive.google.com/file/d/1mHrrxVMQJFZRjqDt184wP2hxjicpc68C/view?usp=drive_link"
DEFAULT_DESTINATION = "/content/datasets/linemod"

def download_dataset(url=DEFAULT_DATASET_URL, output_dir=DEFAULT_DESTINATION):
    """
    Downloads and extracts the LineMOD dataset from a Google Drive URL.
    
    Args:
        url (str): The Google Drive share link.
        output_dir (str): The local folder where data should be saved.
    """
    
    # 1. Define paths
    # We check for the inner folder 'Linemod_preprocessed' to know if we are done
    expected_path = os.path.join(output_dir, "Linemod_preprocessed")
    zip_path = os.path.join(output_dir, "Linemod_preprocessed.zip")
    
    # 2. Check if already exists
    if os.path.exists(expected_path):
        print(f"✅ Dataset already ready at: {expected_path}")
        return

    print(f"⬇️ Dataset not found. Downloading from: {url}")

    # 3. Create Directory
    os.makedirs(output_dir, exist_ok=True)

    # 4. Download
    # fuzzy=True allows gdown to extract the ID from the 'view' URL automatically
    output = gdown.download(url, zip_path, quiet=False, fuzzy=True)

    # 5. Extract
    if output and os.path.exists(zip_path):
        print("📦 Extracting zip file...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            # Optional: Remove zip to save space
            # os.remove(zip_path) 
            print(f"✅ Success! Data located at: {output_dir}")
            
        except zipfile.BadZipFile:
            print("❌ Error: The downloaded file is not a valid zip. Check the URL.")
    else:
        print("❌ Download failed. Check the URL or Google Drive permissions.")

if __name__ == "__main__":
    # You can now run this script from terminal with a custom URL if needed
    # Or just run it as is to use the default.
    download_dataset()