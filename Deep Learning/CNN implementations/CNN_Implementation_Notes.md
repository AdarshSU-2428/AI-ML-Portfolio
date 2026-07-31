# 📘 Convolutional Neural Networks (CNN) — Complete Implementation & Practice Notes

---

## 📌 Executive Summary

This document serves as a comprehensive technical reference and experimental benchmark for Convolutional Neural Network (CNN) implementations built using PyTorch:
1. **PyTorch MNIST Pipeline (`MNIST_implementation.ipynb`)**: Production computer vision pipeline evaluated on grayscale $28 \times 28$ handwritten digit images (60,000 samples across 10 digit classes), achieving **99.05% test accuracy** in 5 epochs.
2. **PyTorch CIFAR-10 Pipeline (`CIFAR-10_implementation.ipynb`)**: Production deep vision pipeline evaluated on full $32 \times 32 \times 3$ RGB color object images (60,000 samples across 10 object classes), utilizing Data Augmentation (`RandomCrop`, `RandomHorizontalFlip`), Batch Normalization (`nn.BatchNorm2d`), Dropout (`0.5`), and Adam optimization with Weight Decay (`1e-4`).

---

## 🧠 Part 1: Dataset Specifications & Model Architectures

### 1.1 MNIST Model Architecture (`MNIST_implementation.ipynb`)
- **Input Resolution:** `(1, 28, 28)` (Grayscale)
- **Normalizing Parameters:** $\mu = 0.1307, \quad \sigma = 0.3081$
- **Dataset Partitioning:** 54,000 Train / 6,000 Validation / 10,000 Test samples.
- **Network Pipeline:**
  1. `Conv2d(1 -> 32, kernel=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (Output: $32 \times 14 \times 14$)
  2. `Conv2d(32 -> 64, kernel=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (Output: $64 \times 7 \times 7$)
  3. `Flatten()` (Vector size: $64 \times 7 \times 7 = 3136$)
  4. `LazyLinear(128)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(128 -> 10)`

### 1.2 CIFAR-10 Model Architecture (`CIFAR-10_implementation.ipynb`)
- **Input Resolution:** `(3, 32, 32)` (RGB Color)
- **Normalizing Parameters:** $\mu = (0.4914, 0.4822, 0.4465), \quad \sigma = (0.2023, 0.1994, 0.2010)$
- **Dataset Partitioning:** 45,000 Train / 5,000 Validation / 10,000 Test samples.
- **Data Augmentation:**
  - `RandomCrop(32, padding=4)` (Pads image to $40 \times 40$, crops random $32 \times 32$ patch for translation invariance)
  - `RandomHorizontalFlip(p=0.5)` (Mirrors images for orientation invariance)
- **Network Pipeline:**
  1. `Conv2d(3 -> 32, k=3, p=1)` $\rightarrow$ `BatchNorm2d(32)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (Output: $32 \times 16 \times 16$)
  2. `Conv2d(32 -> 64, k=3, p=1)` $\rightarrow$ `BatchNorm2d(64)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (Output: $64 \times 8 \times 8$)
  3. `Conv2d(64 -> 128, k=3, p=1)` $\rightarrow$ `BatchNorm2d(128)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (Output: $128 \times 4 \times 4$)
  4. `Flatten()` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `LazyLinear(256)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(256 -> 10)`

---

## 📊 Part 2: Performance Benchmarks & Results Summary

### 2.1 Training Progression & Evaluation Comparison

| Metric / Benchmark | MNIST Pipeline | CIFAR-10 Pipeline |
| :--- | :---: | :---: |
| **Epoch 1 Train Accuracy / Loss** | 95.60% / 0.1415 | ~55.20% / 1.2540 |
| **Epoch 1 Val Accuracy / Loss** | 98.33% / 0.0609 | ~62.40% / 1.0820 |
| **Epoch 5 Train Accuracy / Loss** | 99.50% / 0.0160 | ~75.80% / 0.6840 |
| **Epoch 5 Val Accuracy / Loss** | 98.83% / 0.0422 | ~77.10% / 0.6510 |
| **Final Epoch Target** | **5 Epochs** | **40 Epochs** |
| **Final Test Accuracy** | **98.78%** | **83.80%** |

---

## 🔬 Part 3: Technical Deep Dive & Key Engineering Takeaways

### 3.1 Role of Batch Normalization (`nn.BatchNorm2d`)
- **Prevents Covariate Shift:** Normalizes mean to 0 and variance to 1 across mini-batches for each channel feature map independently.
- **Enables Higher Learning Rates:** Stabilizes activation magnitudes, allowing the network to use $lr=0.001$ without vanishing or exploding gradients.

### 3.2 Role of Regularization (`Dropout` + `Weight Decay`)
- **`nn.Dropout(0.5)`:** Zeroes out 50% of feature connections during training, forcing individual neurons to co-operate rather than co-adapt.
- **`weight_decay=1e-4`:** Adds $L_2$ weight penalty to the loss function, preventing weights from growing excessively large and keeping model predictions smooth.

### 3.3 Why Low-Resolution ($32 \times 32$) CIFAR-10 Images Work
- Human vision requires high pixel density ($1920 \times 1080$) to recognize fine details because human eyes perceive global scenes.
- CNNs process images via local $3 \times 3$ kernel activations. Edge detectors, color contrast gradients, and texture combinations provide sufficient mathematical signals for $85\%+$ classification accuracy even on pixelated $32 \times 32$ images.

### 3.4 Auto-Dimension Inferencing with `nn.LazyLinear`
- `nn.LazyLinear(out_features)` removes the need for manual spatial math ($C \times H \times W$).
- When the first mini-batch passes through `forward()`, PyTorch dynamically inspects the incoming tensor shape and instantiates the linear layer weight matrix automatically.
