# 02. Image Representation and Pixel Normalization

Before a Convolutional Neural Network can process an image, the visual data must be translated into numerical tensors.

---

## 1. Types of Image Representations

Computers see images as multi-dimensional numerical grids (matrices or tensors) where each element represents a **pixel intensity**.

### A. Binary Images (1 Channel)
* **Pixel Values:** Only two possible values: `0` (Black) or `1` (White).
* **Shape:** $[H, W]$ or $[1, H, W]$.
* **Common Use:** Document scanning, optical character recognition (OCR), binary segmentation masks.

---

### B. Grayscale Images (1 Channel)
* **Pixel Values:** Integer values ranging from `0` (pure black) to `255` (pure white).
* **Channels:** $C = 1$.
* **Shape in PyTorch:** $[1, H, W]$ (where $H =$ Height, $W =$ Width).
* **Example:** A $28 \times 28$ MNIST handwritten digit image has tensor shape $[1, 28, 28]$.

```
Grayscale Pixel Grid (0 = Dark, 255 = Bright):
┌─────┬─────┬─────┐
│  0  │ 128 │ 255 │  ──► 2D Matrix of Shape [H, W]
├─────┼─────┼─────┤
│ 45  │ 200 │ 90  │
└─────┴─────┴─────┘
```

---

### C. RGB Color Images (3 Channels)
* **Pixel Values:** Three overlapping color planes: **Red (R)**, **Green (G)**, and **Blue (B)**. Each pixel has three intensity values ranging from $0$ to $255$.
* **Channels:** $C = 3$.
* **Shape in PyTorch:** $[3, H, W]$.
* **Example:** A $32 \times 32$ CIFAR-10 color image has shape $[3, 32, 32]$ ($3 \times 32 \times 32 = 3,072$ total pixel values).

```
RGB Image Tensor Composition:
   Red Channel   [H, W] ───┐
   Green Channel [H, W] ───┼──► Stacked Tensor [3, H, W]
   Blue Channel  [H, W] ───┘
```

---

### D. Other Image Formats
* **RGBA (4 Channels):** Includes an additional **Alpha (A)** channel representing transparency ($0 = \text{fully transparent}, 255 = \text{opaque}$). Shape: $[4, H, W]$.
* **HSV / HSL (3 Channels):** Represents **Hue** (color angle), **Saturation** (color purity), and **Value / Lightness** (brightness). Frequently used in computer vision color-based filtering.

---

## 2. Tensor Memory Layouts: `[C, H, W]` vs. `[H, W, C]`

Different deep learning libraries organize image dimensions in different channel orders:

* **PyTorch Standard (`[C, H, W]`):** **Channel-First**.  
  * Batch Tensor: $[N, C, H, W]$ ($N =$ Batch Size, $C =$ Channels, $H =$ Height, $W =$ Width).
* **OpenCV / TensorFlow Standard (`[H, W, C]`):** **Channel-Last**.  
  * Batch Tensor: $[N, H, W, C]$.

> **PyTorch Tip:** OpenCV loads images as `[H, W, C]` in BGR format. You must transpose them using `np.transpose(image, (2, 0, 1))` or `torch.permute()` to convert them to `[C, H, W]` before passing them to PyTorch `nn.Conv2d`!

---

## 3. Pixel Normalization Techniques

Raw image pixels are represented as 8-bit unsigned integers (`uint8`) with values in the range **$[0, 255]$**.

### A. Min-Max Scaling ($[0, 1]$ Range)
Scales all pixel values linearly into the range $[0.0, 1.0]$:

$$x_{\text{norm}} = \frac{x}{255.0}$$

* `0` becomes `0.0`
* `255` becomes `1.0`
* `128` becomes $\approx 0.5019$

---

### B. Standard Z-Score Normalization ($\mu=0, \sigma=1$)
Standardizes pixel intensities to have a mean ($\mu$) of $0$ and a standard deviation ($\sigma$) of $1$:

$$x_{\text{standard}} = \frac{x - \mu}{\sigma}$$

#### Popular Example: ImageNet Standardization
For 3-channel RGB images pre-trained on ImageNet:
$$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

---

## 4. Why Pixel Normalization is Helpful for CNNs

* **Optimization Stability & Convergence:** Pixel normalization places inputs on a more convenient numerical scale, typically improving optimization stability and convergence speed.
* **Uniform Channel Scaling:** Prevents channels with larger raw magnitudes or brightness variations from dominating early weight updates during backpropagation.
* **Helps Gradient Behavior:** Standardized inputs help maintain well-behaved gradient updates across early layers. However, pixel normalization **does not by itself prevent exploding or vanishing gradients** (which are primarily governed by network depth, weight initialization, and activation function choices).

```
Unnormalized Input Scale ([0, 255]):            Normalized Input Scale ([0, 1] or Mean 0):

     Large, wide parameter steps                    Balanced, well-behaved steps
          \   |   /                                       \     /
           \  |  /                                         \   /
          ---(o)---                                         (o)  (Faster convergence!)
           /  |  \                                         /   \
```
