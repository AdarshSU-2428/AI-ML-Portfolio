# 07. Deep CNNs, Degradation Problem, and Residual Connections (ResNet)

As deep learning developed, researchers noticed a clear trend: **making neural networks deeper allowed them to learn richer, higher-level visual representations**. However, stacking dozens of layers introduced a major obstacle known as the **Degradation Problem**.

This note explains why plain deep networks fail and how **Residual Connections (ResNet)** solved this problem.

---

## 1. Deeper Networks & Hierarchical Feature Learning

In a CNN, each added layer increases the network's **abstraction capacity**:

```
Layer 1-2 (Edges) ──► Layer 3-5 (Textures/Corners) ──► Layer 6-10 (Object Parts) ──► Layer 20+ (Full Objects)
```

* **Shallow Networks (5-10 layers):** Learn simple textures and basic shapes.
* **Deep Networks (50-152+ layers):** Build complex semantic concepts (e.g., distinguishing a Siberian Husky from an Alaskan Malamute by analyzing subtle eye shapes, fur patterns, and snout proportions).

However, simply stacking more standard convolutional layers does not work automatically.

---

## 2. Vanishing Gradients vs. The Degradation Problem

Historically, training deep networks suffered from two distinct phenomena:

### A. The Vanishing Gradient Problem (Solved by ReLU & BatchNorm)
* **What happens:** Gradients shrink exponentially as they propagate backward through deep layers, causing early layers to stop updating.
* **Solution:** Using **ReLU** activations and **Batch Normalization** largely resolved vanishing gradients, ensuring non-zero gradient flow across deep networks.

---

### B. The Degradation Problem (The Unsolved Mystery before ResNet)
Even when Batch Normalization and ReLU guaranteed that gradients **did NOT vanish**, researchers (He et al., 2015) discovered a baffling paradox:

> **The Degradation Paradox:** Adding more layers to a plain network caused **training accuracy to saturate and then degrade rapidly**!

```
20-Layer Plain Net:    Lower Training Error  (Better performance)
56-Layer Plain Net:    HIGHER Training Error (Worse performance!)
```

```
Training Error
  ^
  |          /  56-Layer Plain Net (Higher Error!)
  |         /
  |        /---- 20-Layer Plain Net (Lower Error)
  +---------------------------------------------> Epochs
```

#### Why is this a paradox?
Mathematically, a 56-layer network *should* be able to perform at least as well as a 20-layer network. Worst-case scenario, the extra 36 layers could simply learn to perform an **Identity Mapping** ($f(x) = x$), leaving the input untouched!

However, standard non-linear layers ($W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$) **struggled immensely to learn an identity transformation** $f(x) = x$. The network degraded because solver algorithms could not optimize plain deep stacks.

---

## 3. Residual Connections (The ResNet Innovation)

In 2015, Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun introduced **Residual Networks (ResNet)**, winning the ImageNet competition by training networks with **152 layers**—8x deeper than VGG!

### The Core Idea: Skip Connections (Shortcuts)
Instead of forcing a stack of layers to directly fit an underlying target mapping $H(x)$, ResNet reformulates the target:

```
PLAIN BLOCK (Tries to learn H(x) directly):
Input x ──► [ Conv ──► ReLU ──► Conv ] ──► Output H(x)   (Hard to learn H(x) = x)


RESIDUAL BLOCK (Learns Residual F(x) = H(x) - x):
Input x ──────┬───────────────────────────────┐ (Skip / Shortcut Connection)
              │                               │
              ▼                               │
       [ Conv ──► ReLU ──► Conv ]             │
              │                               │
              ▼                               │
       Residual F(x)                          │
              │                               │
              ▼                               ▼
          ( F(x)   +   x )  ◄─────────────────┘ (Element-wise Addition)
              │
              ▼
          Output H(x) = F(x) + x
```

---

## 4. Why Residual Connections Work (Intuitive Breakdown)

### Intuition 1: The "Highway" Analogy 🚗
Think of a skip connection as an **express highway lane** alongside a local side road:
* **The Skip Connection ($x$):** The express highway. Information and gradients can travel completely unimpeded straight through the network.
* **The Conv Layers ($F(x)$):** The local side road. It learns *modifications* or *delta refinements* to the signal.

---

### Intuition 2: Why Learning $F(x) = 0$ is Easy
Suppose a deeper layer is completely redundant and needs to perform an identity pass-through ($H(x) = x$):

* **Plain Layer ($H(x) = x$):** The solver must precisely tune weights and biases such that non-linear activations yield $W_2 \cdot \text{ReLU}(W_1 x + b) = x$. This is mathematically very difficult for SGD to discover.
* **Residual Block ($H(x) = F(x) + x$):** The solver simply pushes the weights of $F(x)$ toward **zero** ($F(x) \to 0$).
  $$H(x) = 0 + x = x$$
  Pushing weights to zero is trivial for weight decay and gradient descent! The network defaults to identity mapping unless a layer finds a useful feature to add.

---

### Intuition 3: Unobstructed Backward Gradient Highway
During Backpropagation, the gradient of the loss $L$ with respect to input $x$ is:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial H(x)} \cdot \frac{\partial H(x)}{\partial x} = \frac{\partial L}{\partial H(x)} \cdot \left( \frac{\partial F(x)}{\partial x} + 1 \right) = \frac{\partial L}{\partial H(x)} \cdot \frac{\partial F(x)}{\partial x} + \frac{\partial L}{\partial H(x)}$$

Notice the $+ \frac{\partial L}{\partial H(x)}$ term at the end!

> **Gradient Highway:** Even if the layer gradient $\frac{\partial F(x)}{\partial x}$ approaches zero, the term $+ 1$ ensures that the gradient $\frac{\partial L}{\partial H(x)}$ flows **unimpeded directly back** through the skip connection to early layers without vanishing!

---

## 5. ResNet Building Block Architectures

ResNet uses two main types of residual blocks depending on model depth:

### A. Basic Block (Used in ResNet-18 and ResNet-34)
Consists of two $3 \times 3$ convolutional layers with a shortcut addition.

```python
# PyTorch Basic Block Concept
class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x                     # Save shortcut input
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity                  # Element-wise Addition: F(x) + x
        out = self.relu(out)
        return out
```

---

### B. Bottleneck Block (Used in ResNet-50, ResNet-101, ResNet-152)
To keep computational cost low in very deep networks, ResNet uses a **$1 \times 1 \to 3 \times 3 \to 1 \times 1$ Bottleneck design**:

1. **$1 \times 1$ Conv:** Reduces channel dimensions (e.g., $256 \to 64$).
2. **$3 \times 3$ Conv:** Performs spatial convolution on reduced channels ($64 \to 64$).
3. **$1 \times 1$ Conv:** Restores channel dimensions back ($64 \to 256$).

```
Input [N, 256, H, W] ──► 1x1 Conv (Reduce 256->64) ──► 3x3 Conv (64->64) ──► 1x1 Conv (Expand 64->256) ──► (+) ──► Output
  │                                                                                                         ▲
  └───────────────────────────────────── Shortcut Identity Connection [N, 256, H, W] ───────────────────────┘
```

---

## Summary Key Points

* **Degradation Problem:** Plain deep networks suffer from higher training error due to optimization difficulties fitting identity mappings.
* **Residual Connection Formula:** $H(x) = F(x) + x$.
* **Core Benefit:** Allows gradients to flow backward unimpeded via the $+1$ identity shortcut.
* **Result:** Enabled training networks with **100 to 1,000+ layers** without performance degradation!
