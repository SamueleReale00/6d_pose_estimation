# **6D POSE ESTIMATION PROJECT**

## How to clone the git and install the 'requirements' on colab

```python
!git clone https://github.com/SamueleReale00/6d_pose_estimation.git
%cd 6d_pose_estimation
!pip install -r requirements.txt  # <--- Automatically installs everything above
```

## How to import modules

```python
from dataset import download_dataset, MultiObjectLineModDataset
from yolo import convert_linemod_to_yolo, download_yolo_dataset, create_yolo_yaml
```

## How to use the weights

```python
# 1. Clone the repo (if not already done)
if not os.path.exists('6d_pose_estimation'):
    !git clone https://github.com/SamueleReale00/6d_pose_estimation.git

# 2. Define the path to the weight file
weights_path = '/content/6d_pose_estimation/yolo/weights/best.pt'

# 3. Load the model
if os.path.exists(weights_path):
    print(f"✅ Found weights at: {weights_path}")
    model = YOLO(weights_path)
```