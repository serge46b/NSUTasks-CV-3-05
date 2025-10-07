# NSUTasks-CV-3-05: Computer Vision Coin Analysis Project

A comprehensive computer vision project for coin detection, counting, and defect analysis using OpenCV and advanced image processing techniques.

## 🎯 Project Overview

This project consists of two main modules that work together to provide complete coin analysis capabilities:

1. **CDetectionSubmodule**: Coin contour detection and counting
2. **DDetectionSubmodule**: Defect detection in coins using SSIM (Structural Similarity Index)

The main application (`src/main.py`) integrates both modules to provide a unified interface for coin analysis with defect detection using SSIM comparison.

## 🏗️ Project Structure

```
NSUTasks-CV-3-05/
├── CDetectionSubmodule/          # Coin contour detection module
│   ├── coins_contour_detection.py
│   ├── binarisation_helper.py
│   ├── requirements.txt
│   └── README.md
├── DDetectionSubmodule/         # Defect detection module
│   ├── find_defects.py
│   ├── coins/                   # Sample coin images
│   ├── result/                  # Output images
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── README.md
├── src/
│   └── main.py                  # Main application
└── README.md
```

## 🚀 Features

- **Unified Interface**: Combines both detection modules
- **SSIM Implementation**: Custom SSIM calculation with Gaussian windowing
- **Batch Processing**: Processes multiple images simultaneously
- **Interactive Visualization**: Shows results with OpenCV windows
- **Command-line Interface**: Easy-to-use CLI with configurable parameters

## 📋 Requirements

### System Requirements

- Python 3.13+ (recommended)
- OpenCV 4.11.0+
- NumPy 2.3.3+
- SciPy (for SSIM calculations)

### Dependencies

The project uses different dependency sets for each module:

**CDetectionSubmodule:**

- numpy>=2.1.3
- opencv-python>=4.12.0.88
- scikit-image>=0.25.0

**DDetectionSubmodule:**

- numpy>=2.3.3
- opencv-python>=4.11.0.86

## 🛠️ Installation

```bash
# Install dependencies for coin detection
cd CDetectionSubmodule
pip install -r requirements.txt

# Install dependencies for defect detection
cd ../DDetectionSubmodule
pip install -r requirements.txt

# Install additional dependencies for main application
pip install scipy
```

## 🎮 Usage

### Main Application (Recommended)

The main application provides a unified interface for both coin detection and defect analysis:

```bash
python src/main.py -i <image-directory> -r <reference-index> -s <ssim-threshold> -n <include-non-defects>
```

**Parameters:**

- `-i, --image`: Path to directory containing coin images
- `-r, --reference`: Index of reference image (0-based)
- `-s, --ssim`: SSIM threshold for defect detection (0.0-1.0)
- `-n, --non-defects`: Include non-defects in results (True/False)

**Example:**

```bash
python src/main.py -i DDetectionSubmodule/coins -r 0 -s 0.8 -n True
```

### Individual Modules

#### Coin Contour Detection

```bash
cd CDetectionSubmodule
python coins_contour_detection.py
```

#### Defect Detection

```bash
cd DDetectionSubmodule
python find_defects.py --dir coins --standart coin1.png --out result
```

## 🔧 Configuration

### Coin Detection Parameters

- **Canny Thresholds**: `lowThreshCanny=50, highThreshCanny=150`
- **Binary Thresholds**: `lowThresh=127, highThresh=255`
- **Area Filtering**: `minArea=200, maxArea=5000`

### Defect Detection Parameters

- **SSIM Threshold**: Configurable (default: 0.8)
- **Canny Thresholds**: `low_threshold=50, high_threshold=150`
- **Morphological Kernel**: 5x5 for noise reduction

## 📊 Output

### Coin Detection Output

- **Console**: Number of detected coins
- **Image**: `res.png` with highlighted contours
- **Visualization**: Green contours on original image

### Defect Detection Output

- **Console**: Defect detection results and SSIM scores
- **Image**: `result/defect.jpg` with highlighted defects
- **Visualization**: Red contours for defects, green for non-defects

## 🧪 Sample Data

The project includes sample coin images in `DDetectionSubmodule/coins/`:

- `coin1.png` - Reference coin (no defects)
- `coin2.png` - Normal coin
- `coin3.png` - Normal coin
- `coin4.png` - Normal coin
- `defect_coin.png` - Coin with visible defects

## 📚 References

- OpenCV Documentation: https://docs.opencv.org/
- SSIM Paper: Wang, Z., et al. "Image quality assessment: from error visibility to structural similarity"
- Scikit-image: https://scikit-image.org/
