```
  _____   _____  _____  ____  _____  ______ 
 |  __ \ / ____|/ ____|/ __ \|  __ \|  ____|
 | |__) | (___ | |    | |  | | |  | | |__   
 |  ___/ \___ \| |    | |  | | |  | |  __|  
 | |     ____) | |____| |__| | |__| | |____ 
 |_|    |_____/ \_____|\____/|_____/|______| 
```                                                              

# DinoTerrainMax: Semantic Segmentation Analysis

**High-Fidelity Terrain Classification System**

[![Live Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](http://43.204.148.104:8000)

---

## Overview

DinoTerrainMax is an advanced semantic segmentation system designed for precise terrain classification. Built upon state-of-the-art deep learning architectures, this system provides real-time, accurate pixel-level terrain analysis with support for 11 distinct environmental classes.

---

## Architecture

The system employs a sophisticated dual-path architecture that balances global semantic understanding with local geometric precision:

### Core Components

**Encoder**  
DINOv2 (Vision Transformer Base/14) provides robust feature extraction through self-supervised learning, enabling strong representational capabilities across diverse terrain types.

**Context Module**  
Atrous Spatial Pyramid Pooling (ASPP) captures multi-scale spatial information, allowing the model to understand terrain features at various resolutions simultaneously.

**Decoder**  
Deep residual blocks integrated with attention gates enable refined feature upsampling while maintaining spatial accuracy and semantic consistency.

**Edge Pathway**  
A parallel low-level convolutional stream preserves fine geometric details and boundary information throughout the processing pipeline.

---

## Terrain Classification

The system identifies and segments the following terrain classes:

| Class ID | Classification      | Class ID | Classification      | Class ID | Classification |
|----------|---------------------|----------|---------------------|----------|----------------|
| 0        | Background          | 4        | Dry Bushes          | 8        | Rocks          |
| 1        | Trees               | 5        | Ground Clutter      | 9        | Landscape      |
| 2        | Lush Bushes         | 6        | Flowers             | 10       | Sky            |
| 3        | Dry Grass           | 7        | Logs                |          |                |

---

## Project Structure

```
KrackSegmentation/
│
├── train.py               # Training pipeline with TensorBoard logging
├── test.py                # Evaluation script with visual outputs
├── evaluation.py          # Metric computation utilities
├── requirments.txt        # Python dependencies
│
├── data/                  # Dataset directory (user must prepare)
│   ├── train/
│   │   ├── Color_Images/
│   │   └── Segmentation/
│   ├── val/
│   │   ├── Color_Images/
│   │   └── Segmentation/
│   └── test/
│       ├── Color_Images/
│       └── Segmentation/
│
├── stats/                 # Training outputs and model checkpoints
│   ├── best_model.pth     # Best model weights (auto-saved)
│   ├── logs/              # TensorBoard logs (auto-created)
│   └── test_results/      # Test visualizations (auto-created)
│
└── server/                # Web application
    ├── app.py             # Flask backend server
    ├── static/            # Runtime generated files
    └── templates/
        └── index.html     # Web interface frontend
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/Sakshamsetia/KrackSegmentation.git
cd KrackSegmentation
```

### Step 2: Install Dependencies

```bash
pip install -r requirments.txt
```

### Step 3: Prepare Dataset

Organize your dataset in the following structure:

```bash
mkdir -p data/train/Color_Images data/train/Segmentation
mkdir -p data/val/Color_Images data/val/Segmentation
mkdir -p data/test/Color_Images data/test/Segmentation
```

**Place your data files:**
- Training images in `data/train/Color_Images/`
- Training masks in `data/train/Segmentation/`
- Validation images in `data/val/Color_Images/`
- Validation masks in `data/val/Segmentation/`
- Test images in `data/test/Color_Images/`
- Test masks in `data/test/Segmentation/`

**Mask Format:**  
Ground truth masks should use the following pixel value mapping:
```
{0, 100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000}
```
corresponding to class IDs 0-10.

---

## Workflow

### 1. Training the Model

Start training with TensorBoard logging:

```bash
python train.py
```

**Training Configuration:**
- Batch Size: 4
- Learning Rate: 6e-5
- Epochs: 10
- Image Resolution: 476 x 266
- Automatic mixed precision (AMP) enabled

**Training Features:**
- Class-weighted loss function for handling imbalanced data
- Data augmentation (random flips, rotations)
- Validation after each epoch
- Best model checkpoint saved to `stats/best_model.pth`
- TensorBoard logs saved to `stats/logs/`

### 2. Monitoring Training with TensorBoard

While training is running, open a new terminal and launch TensorBoard:

```bash
tensorboard --logdir=stats/logs
```

Then open your browser and navigate to `http://localhost:6006`

**TensorBoard Visualizations:**
- Training and validation loss curves
- Mean IoU (mIoU) over steps
- Side-by-side comparisons: Input | Ground Truth | Prediction

![TensorBoard Visualization](images/tensorboard.png)

### 3. Testing and Evaluation

After training, evaluate the model on the test set:

```bash
python test.py
```

**Visual Outputs:**  
Comparison images saved to `stats/test_results/` with format:
```
test_sample_001.png: [Original Image | Ground Truth | Prediction]
test_sample_002.png: [Original Image | Ground Truth | Prediction]
```
### 4. Running the Web Interface

Launch the Flask application for interactive inference:

```bash
python server/app.py
```

The web server will start at `http://127.0.0.1:8000`

**Server Configuration:**
- Automatically loads `stats/best_model.pth`
- Detects GPU availability (falls back to CPU)
- Serves static files from `server/static/`

![Web Server Interface](images/server.png)

---

## Using the Web Interface

### Basic Usage

1. **Navigate to** `http://127.0.0.1:8000` in your web browser

2. **Upload an Input Image:**
   - Click "Choose File" under "Select Image"
   - Choose a terrain image (JPG, PNG, JPEG supported)
   - Image will be automatically resized to 476 x 266

3. **Optional - Upload Ground Truth:**
   - Click "Choose File" under "Ground Truth Mask (Optional)"
   - Upload corresponding segmentation mask
   - System will compute and display mIoU and Pixel Accuracy

4. **Submit for Processing:**
   - Click "Segment Image"
   - Wait for processing (typically 50-1000ms depending on hardware)

### Understanding Results

**Visual Outputs:**

| Display | Description |
|---------|-------------|
| Input Image | Your original uploaded image |
| Segmentation Result | Color-coded prediction overlay |
| Raw Prediction Mask | Grayscale class indices (downloadable) |
| Ground Truth | Color-coded reference mask (if provided) |

**Performance Metrics:**

- **Inference Time:** Processing duration in milliseconds
- **Device Status:** Shows GPU or CPU utilization
- **Mean IoU:** Intersection over Union score (if ground truth provided)
- **Pixel Accuracy:** Percentage of correctly classified pixels

**Class Distribution:**

The interface displays a detailed breakdown of terrain composition:

```
Trees:           23.4%
Sky:             18.7%
Landscape:       15.2%
Ground Clutter:  12.8%
Lush Bushes:      8.3%
Rocks:            7.6%
...
```

## Visualization Examples

### Training Visualization (TensorBoard)

TensorBoard provides real-time monitoring with:
- **Scalars:** Loss curves, mIoU progression, accuracy trends
- **Images:** Training samples with predictions overlaid
- **Histograms:** Model weight distributions
- **Graphs:** Complete model architecture visualization

### Test Results Visualization

Test output images show three-way comparisons:

```
┌─────────────────┬─────────────────┬─────────────────┐
│  Original Image │  Ground Truth   │   Prediction    │
│   (RGB Input)   │  (Color-coded)  │  (Color-coded)  │
└─────────────────┴─────────────────┴─────────────────┘
```

This format allows immediate visual assessment of:
- Segmentation accuracy
- Boundary precision
- Class confusion patterns
- Model failure cases

### Web Interface Visualization

The web interface provides:
- Real-time inference with live preview
- Downloadable prediction masks
- Metrics computed on-the-fly
- Percentage breakdown by terrain class

---

## Performance Specifications

| Metric | Specification |
|--------|---------------|
| Input Resolution | 476 x 266 pixels |
| Model Parameters | ~86 million |
| GPU Inference Time | 50-100ms per image |
| CPU Inference Time | 500-1000ms per image |
| GPU Memory | ~2GB VRAM |
| Training Time | ~2-4 hours (10 epochs on single GPU) |

---

## Advanced Configuration

Edit `train.py` to customize:

### Custom Class Weights

Adjust class weights for imbalanced datasets:

```python
CLASS_WEIGHTS = torch.tensor([
    1.0,  # Background
    2.0,  # Trees
    10.0, # Lush Bushes (rare class - higher weight)
    ...
])
```

### Data Augmentation

Training includes built-in augmentation:
- Random horizontal/vertical flips
- Random rotation (90°, 180°, 270°)
- Color jittering
- Normalization (ImageNet statistics)

---

## Model Architecture Details

**DinoTerrainMax Components:**

```
Input (3, 266, 476)
    ↓
DINOv2 Encoder (ViT-B/14)
    ↓
ASPP Module (Multi-scale Context)
    ├── Branch 1: 1×1 conv
    ├── Branch 2: 3×3 conv (dilation=6)
    ├── Branch 3: 3×3 conv (dilation=12)
    ├── Branch 4: 3×3 conv (dilation=18)
    └── Branch 5: Global Average Pooling
    ↓
Residual Decoder with Attention Gates
    ↓
Edge Enhancement Pathway (parallel)
    ↓
Feature Fusion
    ↓
Output (11, 266, 476)
```

---


## Acknowledgments

Built upon foundations from:
- DINOv2 (Meta AI Research)
- DeepLabV3+ Architecture
- PyTorch Framework
- Flask Web Framework

---

[View Repository](https://github.com/Sakshamsetia/KrackSegmentation)
