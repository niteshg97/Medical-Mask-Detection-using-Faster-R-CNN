
# 🩺 Medical Mask Detection using Faster R-CNN

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-FaceMaskDetection-lightgrey.svg)](https://github.com/Pseudo-Lab/Tutorial-Book-Utils)

A **deep learning-based face mask detection** system using **Faster R-CNN (ResNet-50-FPN)**.  
This repository demonstrates a **two-stage object detector** applied to a **medical mask dataset**, including:

- ✅ Data preprocessing & annotation parsing  
- ✅ Model fine-tuning via transfer learning  
- ✅ Bounding box visualization & mAP evaluation  
- ✅ Emphasis on **hyperparameter tuning** for improved accuracy  

---

## 📘 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Environment Setup](#️-environment-setup)
- [Model Architecture](#-model-architecture)
- [Training Configuration](#-training-configuration)
- [Evaluation & Inference](#-evaluation--inference)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Results](#-results)
- [References](#-references)

---

## 🚀 Overview

This project uses **Faster R-CNN** (Feature Pyramid Network enabled) to detect and classify three mask-related classes:

| Class ID | Label | Description |
|-----------|--------|-------------|
| 1 | With Mask | Person wearing a mask correctly |
| 2 | Mask Worn Incorrectly | Person wearing a mask improperly |
| 3 | Without Mask | Person without a mask |

The model utilizes **transfer learning** from COCO-pretrained weights to adapt to the **Face Mask Detection** dataset efficiently.

---

## 📂 Dataset

Dataset provided via **Pseudo-Lab’s utilities**:


git clone https://github.com/Pseudo-Lab/Tutorial-Book-Utils
python Tutorial-Book-Utils/PL_data_loader.py --data FaceMaskDetection
unzip -q Face\ Mask\ Detection.zip


Data split:

* 683 training images
* 170 testing images
* XML annotations in **PASCAL VOC format**

---

## ⚙️ Environment Setup

### Install Dependencies

```bash
pip install torch torchvision numpy matplotlib beautifulsoup4 pillow tqdm
```

### GPU/CPU Configuration

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

> 💡 *Colab GPUs may vary — rerun if you face memory limitations.*

---

## 🧠 Model Architecture

Using `torchvision.models.detection.fasterrcnn_resnet50_fpn`:

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=4)
```

* Pretrained backbone: **ResNet-50-FPN**
* Modified classifier head for 3 custom classes + 1 background
* Supports **transfer learning**

---

## 🏋️ Training Configuration

### Key Hyperparameters

| Parameter            | Description         | Default  |
| -------------------- | ------------------- | -------- |
| `num_epochs`         | Training epochs     | `10`     |
| `batch_size`         | Batch size          | `4`      |
| `learning_rate (lr)` | Learning rate       | `0.005`  |
| `momentum`           | Momentum            | `0.9`    |
| `weight_decay`       | L2 regularization   | `0.0005` |
| `optimizer`          | Optimization method | `SGD`    |

Example configuration:

```python
optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.005, momentum=0.9, weight_decay=0.0005
)
```

Training log:

```
epoch : 1, Loss : 3.21, time : 68.4s
epoch : 2, Loss : 2.47, time : 66.9s
...
```

---

## 🔍 Evaluation & Inference

After training, save and reload model weights:

```python
torch.save(model.state_dict(), "model_10.pt")
model.load_state_dict(torch.load("model_10.pt"))
```

Run predictions with a **confidence threshold**:

```python
preds = make_prediction(model, imgs, threshold=0.5)
```

Bounding box colors:

* 🟥 **Red** – With Mask
* 🟩 **Green** – Mask Incorrect
* 🟧 **Orange** – Without Mask

---

## 🎯 Hyperparameter Tuning

**Hyperparameter tuning** plays a *crucial* role in improving Faster R-CNN’s performance.
Fine-tuning the following parameters can greatly impact **mAP** and convergence:

| Hyperparameter         | Range       | Effect                                      |
| ---------------------- | ----------- | ------------------------------------------- |
| `learning_rate`        | 1e-4 → 1e-2 | Balances convergence vs. overshooting       |
| `batch_size`           | 2 → 8       | Affects gradient stability and memory       |
| `IoU threshold`        | 0.5 → 0.7   | Controls overlap precision                  |
| `epochs`               | 10 → 50     | Affects learning depth and overfitting risk |
| `confidence threshold` | 0.3 → 0.7   | Balances precision and recall               |

> 🔬 Proper tuning can yield **significant improvements** in detection accuracy and mAP.

---

## 📈 Results

After training for **10 epochs**:

| Metric | Class          | AP       |
| ------ | -------------- | -------- |
| mAP    | —              | **0.67** |
| AP     | With Mask      | **0.91** |
| AP     | Mask Incorrect | **0.36** |
| AP     | Without Mask   | **0.65** |

**Faster R-CNN** demonstrates strong performance even with minimal tuning — outperforming RetinaNet on the same dataset.

---

## 📚 References

* [Faster R-CNN (Ren et al., 2015)](https://arxiv.org/abs/1506.01497)
* [TorchVision Detection Models](https://pytorch.org/vision/stable/models.html)
* [Pseudo-Lab Tutorial-Book-Utils](https://github.com/Pseudo-Lab/Tutorial-Book-Utils)

---

## 🧾 License

This project is licensed under the **MIT License** — free to use and modify with attribution.

---

## ✨ Author

**Developed by:** *[Nitesh Kumar]*
**Inspired by:** Pseudo-Lab Object Detection Tutorials

```
```
