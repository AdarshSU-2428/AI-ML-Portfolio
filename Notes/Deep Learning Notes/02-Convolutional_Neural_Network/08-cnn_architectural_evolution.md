# 08. Evolution of CNN Architectures: LeNet $\to$ AlexNet $\to$ VGG $\to$ ResNet

Understanding the architectural evolution of Convolutional Neural Networks reveals how deep learning advanced from digit recognition in 1998 to large-scale computer vision models today.

---

## 1. Architectural Timeline & Key Milestones

```
LeNet-5 (1998) ────────► AlexNet (2012) ────────► VGG-16 (2014) ────────► ResNet-50 (2015)
(Digit OCR, 60K params)   (GPU Boost, 60M params)  (Small 3x3s, 138M params) (Skip Shortcuts, 25M params)
```

---

## 2. LeNet-5 (Yann LeCun et al., 1998)

**LeNet-5** is the pioneering CNN designed for handwritten digit recognition (MNIST dataset).

### Key Innovations:
* Introduced the classic **Conv $\to$ Pool $\to$ Conv $\to$ Pool $\to$ FC** pipeline.
* Used Average Pooling (Sub-sampling) and Sigmoid/Tanh activations.

### LeNet-5 Architecture & Tensor Data Flow:
* **Input:** `[N, 1, 32, 32]` (Grayscale images padded to $32 \times 32$)

```
Input [N, 1, 32, 32]
  │
  ▼
Conv1 (6 filters, 5x5, S=1) ──► [N, 6, 28, 28]
  │
  ▼
AvgPool1 (2x2, S=2)        ──► [N, 6, 14, 14]
  │
  ▼
Conv2 (16 filters, 5x5, S=1)──► [N, 16, 10, 10]
  │
  ▼
AvgPool2 (2x2, S=2)        ──► [N, 16, 5, 5]
  │
  ▼
Flatten                    ──► [N, 400]    (16 * 5 * 5 = 400)
  │
  ▼
Linear1 (400 ──► 120)       ──► [N, 120]
  │
  ▼
Linear2 (120 ──► 84)        ──► [N, 84]
  │
  ▼
Linear3 (84 ──► 10)         ──► [N, 10]    (10 Class Outputs)
```

* **Total Parameters:** $\sim 60,000$ (60K).

### ⚠️ Limitations of LeNet-5 (Why AlexNet Was Introduced):
1. **Shallow Depth & Small Capacity:** Only 2 convolutional layers ($\sim 60\text{K}$ parameters). Designed strictly for small $32 \times 32$ single-channel grayscale digits.
2. **Saturating Activations (Tanh/Sigmoid):** Caused vanishing gradients if attempts were made to stack more layers.
3. **Average Pooling Blur:** Average pooling smoothed out sharp edge details and fine textures.
4. **No GPU Acceleration:** Could not scale to compute massive 3-channel RGB datasets (like ImageNet).

---

## 3. AlexNet (Alex Krizhevsky, Ilya Sutskever, & Geoffrey Hinton, 2012)

**AlexNet** won the ImageNet 2012 competition by a massive margin, igniting the modern Deep Learning revolution.

### Key Innovations:
1. **GPU Acceleration:** Parallel training across 2 GPUs.
2. **ReLU Activations:** Replaced Sigmoid/Tanh with ReLU to accelerate training speed.
3. **Dropout Regularization:** Used $p=0.5$ dropout in fully connected layers to stop heavy overfitting.
4. **Data Augmentation:** Flips, crops, and color variations.

### AlexNet Architecture & Tensor Data Flow:
* **Input:** `[N, 3, 224, 224]` (RGB ImageNet images)

```
Input [N, 3, 224, 224]
  │
  ▼
Conv1 (96 filters, 11x11, S=4, P=2) ──► [N, 96, 55, 55]
  │
  ▼
MaxPool1 (3x3, S=2)                 ──► [N, 96, 27, 27]
  │
  ▼
Conv2 (256 filters, 5x5, S=1, P=2)  ──► [N, 256, 27, 27]
  │
  ▼
MaxPool2 (3x3, S=2)                 ──► [N, 256, 13, 13]
  │
  ▼
Conv3 (384 filters, 3x3, S=1, P=1)  ──► [N, 384, 13, 13]
  │
  ▼
Conv4 (384 filters, 3x3, S=1, P=1)  ──► [N, 384, 13, 13]
  │
  ▼
Conv5 (256 filters, 3x3, S=1, P=1)  ──► [N, 256, 13, 13]
  │
  ▼
MaxPool3 (3x3, S=2)                 ──► [N, 256, 6, 6]
  │
  ▼
Flatten                             ──► [N, 9216]   (256 * 6 * 6 = 9216)
  │
  ▼
Linear1 + Dropout(0.5) (9216 ──► 4096)──► [N, 4096]
  │
  ▼
Linear2 + Dropout(0.5) (4096 ──► 4096)──► [N, 4096]
  │
  ▼
Linear3 (4096 ──► 1000)             ──► [N, 1000]  (1,000 ImageNet Classes)
```

* **Total Parameters:** $\sim 60 \text{ Million}$ (mostly in the FC layers).

### ⚠️ Limitations of AlexNet (Why VGG Was Introduced):
1. **Large, Aggressive Filter Sizes ($11 \times 11$ and $5 \times 5$):** Taking large stride steps ($S=4$) in early layers missed fine-grained, localized visual textures.
2. **Unstandardized Architecture:** Filter sizes and channel numbers were chosen heuristically without a clean modular rule.
3. **Heavy Parameter Overhead ($\sim 60\text{M}$):** Dense FC layers (`Linear(9216, 4096)`) accounted for over $90\%$ of total weights, causing heavy overfitting risks.

---

## 4. VGG-16 (Visual Geometry Group, Oxford, 2014)

**VGG-16** proved that network depth and simplicity in design lead to superior visual representations.

### Key Innovations:
* **Factorized Small Kernels ($3 \times 3$):** Replaced large $11 \times 11$ and $5 \times 5$ filters with stacks of small $3 \times 3$ filters.
  * *Why?* Two stacked $3 \times 3$ conv layers have an effective receptive field of $5 \times 5$, but use **fewer parameters** ($2 \times 3^2 = 18$ vs. $5^2 = 25$) and introduce more non-linearity!
* **Modular Block Design:** Stacking blocks of 2 or 3 Conv layers followed by Max Pooling.

### VGG-16 Architecture & Tensor Data Flow:
* **Input:** `[N, 3, 224, 224]`

```
Input [N, 3, 224, 224]
  │
  ▼
Block 1: [Conv(3->64, 3x3, P=1) x2] ──► [N, 64, 224, 224]  ──► MaxPool ──► [N, 64, 112, 112]
  │
  ▼
Block 2: [Conv(64->128, 3x3, P=1) x2]──► [N, 128, 112, 112] ──► MaxPool ──► [N, 128, 56, 56]
  │
  ▼
Block 3: [Conv(128->256, 3x3, P=1) x3]──►[N, 256, 56, 56]   ──► MaxPool ──► [N, 256, 28, 28]
  │
  ▼
Block 4: [Conv(256->512, 3x3, P=1) x3]──►[N, 512, 28, 28]   ──► MaxPool ──► [N, 512, 14, 14]
  │
  ▼
Block 5: [Conv(512->512, 3x3, P=1) x3]──►[N, 512, 14, 14]   ──► MaxPool ──► [N, 512, 7, 7]
  │
  ▼
Flatten                             ──► [N, 25088]  (512 * 7 * 7 = 25088)
  │
  ▼
Linear1 (25088 ──► 4096)            ──► [N, 4096]
  │
  ▼
Linear2 (4096 ──► 4096)             ──► [N, 4096]
  │
  ▼
Linear3 (4096 ──► 1000)             ──► [N, 1000]
```

* **Total Parameters:** $\sim 138 \text{ Million}$ (heavy memory footprint due to FC layers).

### ⚠️ Limitations of VGG-16 (Why ResNet Was Introduced):
1. **Extreme Parameter & Memory Footprint ($\sim 138\text{M}$):** VGG-16 requires over $500\text{ MB}$ of VRAM just to store weights! The first two FC layers (`Linear(25088, 4096)`) account for over $100\text{ Million}$ parameters alone.
2. **Computationally Slow:** High memory traffic makes inference and deployment slow on edge devices.
3. **The Degradation Barrier:** Stacking layers beyond 16–19 layers caused training accuracy to degrade rapidly due to optimization difficulties in plain deep networks.

---

## 5. ResNet-50 (Kaiming He et al., 2015)

**ResNet** broke depth barriers by introducing **Skip/Residual Connections**, scaling networks to 50, 101, and 152 layers while reducing parameters compared to VGG!

### Key Innovations:
1. **Residual Connections ($F(x) + x$):** Solved the degradation problem.
2. **Global Average Pooling (GAP):** Replaced heavy 4096-node FC layers with $1 \times 1$ spatial averaging, slashing total parameters from 138M down to **25.5M**!
3. **Bottleneck Blocks:** Uses $1 \times 1 \to 3 \times 3 \to 1 \times 1$ conv blocks to reduce compute.

### ResNet-50 Architecture & Tensor Data Flow:
* **Input:** `[N, 3, 224, 224]`

```
Input [N, 3, 224, 224]
  │
  ▼
Stage 1: Conv1 (64 filters, 7x7, S=2, P=3) ──► [N, 64, 112, 112] ──► MaxPool(3x3, S=2) ──► [N, 64, 56, 56]
  │
  ▼
Stage 2: 3 x Bottleneck Blocks (64 ──► 256 channels)        ──► [N, 256, 56, 56]
  │
  ▼
Stage 3: 4 x Bottleneck Blocks (256 ──► 512 channels, S=2)  ──► [N, 512, 28, 28]
  │
  ▼
Stage 4: 6 x Bottleneck Blocks (512 ──► 1024 channels, S=2) ──► [N, 1024, 14, 14]
  │
  ▼
Stage 5: 3 x Bottleneck Blocks (1024 ──► 2048 channels, S=2)──► [N, 2048, 7, 7]
  │
  ▼
Global Average Pooling (AdaptiveAvgPool2d(1,1))             ──► [N, 2048, 1, 1]
  │
  ▼
Flatten(1)                                                  ──► [N, 2048]
  │
  ▼
Linear Classifier (2048 ──► 1000)                           ──► [N, 1000]
```

* **Total Parameters:** $\sim 25.5 \text{ Million}$ (significantly lighter than VGG-16 despite being 3x deeper!).

### ⚠️ Limitations of ResNet-50 (Why Future Models Were Introduced):
1. **Sequential Channel Redundancy:** Intermediate feature maps are added ($F(x) + x$) rather than concatenated, limiting feature reuse across non-adjacent layers (addressed by **DenseNet**).
2. **Local Receptive Field Limitations:** Standard $3 \times 3$ convolutions process spatial details locally, lacking long-range global contextual relationships across distant image regions (addressed by **Vision Transformers / ViTs**).
3. **High Compute for Edge Devices:** Still too heavy for real-time mobile deployment (addressed by **MobileNet** using depthwise separable convolutions).

---

## Evolution Summary Comparison Table

| Model | Year | Layers | Key Innovations | Major Limitations | Total Params | Top-5 Error |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LeNet-5** | 1998 | 5 | First Conv-Pool pipeline | Shallow, Tanh vanishing gradients, CPU-only | ~60K | N/A (MNIST) |
| **AlexNet** | 2012 | 8 | GPUs, ReLU, Dropout | Large $11 \times 11$ filters miss fine details, 60M params | ~60M | 15.3% |
| **VGG-16** | 2014 | 16 | Small $3 \times 3$ filter stacks | Massive 138M params, 100M+ in FC head, degradation limit | ~138M | 7.3% |
| **ResNet-50**| 2015 | 50 | Residual Skip ($F(x)+x$), GAP | Local receptive fields lack global self-attention context | **~25.5M** | **3.57%** |
