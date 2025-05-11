# Project Installation and Usage Guide

## Requirements
- Python 3.x (Recommended: Python 3.8 or higher)

## Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies
The project requires the following Python packages (with specific versions):

```
matplotlib==3.10.3
numpy==2.2.5
scikit_learn==1.6.1
scipy==1.14.1
Shapely==2.1.0
opencv-python==4.10.0.84
```

## Usage

To run the image stitching algorithm:

```bash
python main.py <imgs_path> <range_imgs> <result_path> [-s <image_data_path>] [-h <image_data_path>] [-m <kp_map_path>] [-b <1-3>]
```

### Arguments:
- `imgs_path`: Folder where images are stored
- `range_imgs`: Number of images to process (integer)
- `result_path`: Folder to save the resulting stitched image

### Optional Flags:
- `-s <path>`: Save metadata for images to specified path (cannot combine with -h)
- `-h <path>`: Load metadata for images from specified path (cannot combine with -s)
- `-m <path>`: Save keypoint maps to specified path
- `-b <1|2|3>`: Blending mode (default=1)

### Notes:
- Either `-s` or `-h` must be used, but not both
- `-m` and `-b` are optional
- Blending modes:
  - `1`: Blending with overlap
  - `2`: 50/50 mixing
  - `3`: Gradient blending