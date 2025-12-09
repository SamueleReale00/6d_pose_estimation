# YOLO  

## YOLO Weights

**Absolute path** =  `6d_pose_estimation/yolo/weights/best.pt`

## 🛠️ YOLO Dataset Manager Module

This module (`yolo/dataset_manager.py`) provides essential tools for preparing and managing datasets for YOLO object detection. It bridges the gap between raw LineMOD data and the format required by Ultralytics YOLO.

---

### 1. `convert_linemod_to_yolo`

**Description:**
Converts a raw LineMOD dataset (organized in object folders `01`, `02`, etc.) into the standard YOLO folder structure (`train/images`, `train/labels`, etc.). It normalizes bounding boxes and maps class IDs to a 0-based index.

**Parameters:**
* `source_path` *(str)*: Path to the raw `data` folder containing numbered subfolders (e.g., `/content/datasets/linemod/Linemod_preprocessed/data`).
* `output_path` *(str)*: Destination folder for the converted dataset (e.g., `/content/yolo_dataset`).
* `train_split` *(float, default=0.8)*: Fraction of data to use for training (0.0 to 1.0). The rest goes to validation.

**Example Usage:**
```python
from yolo import convert_linemod_to_yolo

# Use default paths (if your data is in the standard location)
convert_linemod_to_yolo()

# OR specify custom paths
convert_linemod_to_yolo(
    source_path="/content/my_raw_data",
    output_path="/content/my_yolo_data",
    train_split=0.9
)
```

---

### 2. `download_yolo_dataset`

**Description:**
This function serves as a shortcut to retrieve a dataset that has already been converted to the YOLO format. Instead of running the time-consuming conversion script every session, you can download the ready-to-use ZIP file directly from Google Drive.

**Parameters:**
* **drive_link** *(string)*: The shared Google Drive URL pointing to your `yolo_dataset.zip` file.
* **output_path** *(string)*: The local directory where you want the dataset to be saved and extracted (e.g., the root content folder). Default DEFAULT_YOLO_PATH="/content/yolo_dataset"

**What it does:**
1.  **Preparation:** It checks if the destination folder exists and creates it if it does not.
2.  **Download:** It uses the `gdown` tool to download the ZIP file from the provided Google Drive link. This tool is specifically designed to handle large files from Drive that `wget` or `curl` might struggle with.
3.  **Extraction:** Once the download is complete, it automatically unzips the contents into your specified output path.
4.  **Cleanup:** It removes the downloaded ZIP file to free up disk space, leaving only the extracted folders ready for training.

**Example Usage:**

```python
from yolo import download_yolo_dataset

# Link to your zipped dataset on Drive
link = "google_drive_link"

# Download and extract to the current directory
download_yolo_dataset(link, output_path="/content")
```

---

### 3. `create_yolo_yaml`

**Description:**
Generates the configuration file (typically named `linemod.yaml` or `data.yaml`) required to initiate the YOLO training process. This configuration file acts as a roadmap for the model, defining exactly where the data resides and what objects it needs to detect.

**Parameters:**
* **dataset_path** *(string, default="/content/yolo_dataset")*: The root directory containing your `train` and `val` subfolders.
* **output_yaml_path** *(string, default="/content/linemod.yaml")*: The complete file path (including the filename) where the new YAML configuration file will be saved.

**What it does:**
1.  **Defines Structure:** It constructs a configuration dictionary that specifies:
    * **Root Path:** The absolute path to the dataset directory.
    * **Split Paths:** The relative locations of the training and validation image sets.
    * **Class Mapping:** A predefined dictionary that maps numerical class IDs (0, 1, 2...) to their corresponding human-readable names (e.g., 0="Ape", 1="Benchvise").
2.  **File Creation:** It writes this configuration into a `.yaml` file at the location specified by `output_yaml_path`, making it ready to be passed directly to the YOLO training command.

**Example Usage:**

```python
from yolo import create_yolo_yaml

# 1. Use Default Paths (Recommended)
# Saves to /content/linemod.yaml pointing to /content/yolo_dataset
create_yolo_yaml()

# 2. Use Custom Paths
create_yolo_yaml(
    dataset_path="/content/my_custom_data",
    output_yaml_path="/content/config/my_data.yaml"
)
```

---