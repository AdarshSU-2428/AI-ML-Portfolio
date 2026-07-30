# 04. Receptive Fields, Convolution Hyperparameters, and ReLU

This note covers key CNN architectural concepts: **Local Receptive Fields**, **Effective Receptive Field Growth**, **Parameter Sharing**, spatial hyperparameters (**Stride, Padding, Dilation**), output dimension formulas, and the role of **ReLU**.

---

## 1. Local Receptive Field & Parameter Sharing

CNNs achieve parameter efficiency through two foundational design principles:

### A. Local Receptive Field
In a standard ANN, a neuron connects to *every single pixel* in the input image. In a CNN, a neuron connects only to a small localized spatial patch of the input image (e.g., $3 \times 3$ pixels). This local input patch is called the neuron's **Local Receptive Field**.

* **Why it works:** Visual patterns in real-world images are locally correlated (pixels near each other form meaningful edges and textures).

---

### B. Effective Receptive Field Growth (Hierarchical Feature Expansion)
While an individual $3 \times 3$ filter in Layer 1 sees only a tiny $3 \times 3$ patch of the original image, **stacking multiple convolutional layers** causes deeper neurons to depend on increasingly larger spatial regions of the original raw input image.

```
Input Image Pixels
      │
      ▼
Layer 1 Conv (3×3 Filter)  ──► Sees 3×3 region of raw image
      │
      ▼
Layer 2 Conv (3×3 Filter)  ──► Sees 5×5 effective region of raw image!
      │
      ▼
Layer 3 Conv (3×3 Filter)  ──► Sees 7×7 effective region of raw image!
      │
      ▼
Deeper Layers              ──► Capture large-scale structures (e.g., whole objects)
```

> **Key Rule:** Stacking two $3 \times 3$ conv layers yields an effective receptive field of $5 \times 5$, while using fewer parameters ($2 \times (3 \times 3) = 18$ weights) than a single $5 \times 5$ conv layer ($25$ weights)!

---

### C. Parameter Sharing (Weight Sharing)
The exact same filter (same set of weights $W$) is used to scan across every location of the input image.

---

### D. Combination: Local Receptive Field + Parameter Sharing

```
┌───────────────────────────┬────────────────────────────────────────────────────────┐
│ Principle                 │ Impact on Network                                      │
├───────────────────────────┼────────────────────────────────────────────────────────┤
│ Local Receptive Field     │ Dramatically reduces connections per neuron.           │
│ Parameter Sharing         │ Reuses the same small set of weights across the image. │
├───────────────────────────┼────────────────────────────────────────────────────────┤
│ COMBINED EFFECT           │ 1. Drastic Reduction in Parameters (prevents overfit)  │
│                           │ 2. Achieves Translation Equivariance                   │
└───────────────────────────┴────────────────────────────────────────────────────────┘
```

> **Translation Equivariance:** Convolutional layers are **translation equivariant**. If a feature (like a sharp edge or eye) moves in the input image, its corresponding activation shifts by the same amount in the feature map ($f(g(x)) = g(f(x))$). Later pooling and aggregation layers provide additional translation robustness/invariance.

---

## 2. Convolution Hyperparameters: Stride, Padding, and Dilation

When defining a convolutional layer (`nn.Conv2d`), three key hyperparameters govern how the filter moves and controls the output spatial size ($H_{out}, W_{out}$):

### A. Stride ($S$)
**Stride** is the step size (number of pixels) the filter shifts as it slides across the image.

* **Stride = 1:** Filter slides 1 pixel at a time (output size remains large).
* **Stride = 2:** Filter skips 2 pixels at a time (halves the spatial output size, performing spatial downsampling).

---

### B. Padding ($P$)
**Padding** is the process of adding border pixels (usually zeros, known as **Zero-Padding**) around the perimeter of the input matrix before convolution.

#### Why is Padding Needed?
1. **Preserves Spatial Dimensions:** Without padding, every convolution operation shrinks spatial dimensions ($H \times W$). Padding prevents early spatial collapse in deep networks.
2. **Processes Border Regions Extensively:** Padding allows filters to process border regions more extensively, reducing the rapid loss of information near image boundaries.

#### Types of Padding:
* **Valid Padding ($P = 0$):** No padding added. Output spatial dimension shrinks: $O = N - K + 1$.
* **Same Padding:** Padding is added such that the output spatial size **matches the input size** ($O = N$) when Stride $S = 1$.
  $$\text{For Kernel } K, \quad P = \frac{K - 1}{2} \quad (\text{e.g., for } K=3, P=1; \text{ for } K=5, P=2)$$

---

### C. Dilation ($D$)
**Dilation** (also known as *Atrous Convolution*) defines the spacing between kernel elements. It inflates the filter's receptive field without adding extra parameters!

* **Dilation = 1 (Standard):** Kernel elements are adjacent.
* **Dilation = 2:** Kernel elements have a 1-pixel gap between them (a $3 \times 3$ kernel covers a $5 \times 5$ spatial area).

```
Standard Convolution (Dilation D = 1):     Dilated Convolution (Dilation D = 2):
       [ * ][ * ][ * ]                            [ * ][ . ][ * ][ . ][ * ]
       [ * ][ * ][ * ]                            [ . ][ . ][ . ][ . ][ . ]
       [ * ][ * ][ * ]                            [ * ][ . ][ * ][ . ][ * ]
 (Covers 3x3 area, 9 weights)              (Covers 5x5 area, 9 weights!)
```

---

## 3. Full Output Spatial Dimension Formulas

Given an input image of height/width $N$, Kernel size $K$, Stride $S$, Padding $P$, and Dilation $D$:

### Standard Output Formula (Dilation $D = 1$):

$$O = \left\lfloor \frac{N + 2P - K}{S} \right\rfloor + 1$$

---

### Complete Output Formula (With Dilation $D \ge 1$):

$$O = \left\lfloor \frac{N + 2P - D(K - 1) - 1}{S} \right\rfloor + 1$$

Where $\lfloor \cdot \rfloor$ denotes the floor function (rounding down to the nearest integer).

---

## 4. Comparison: Stride vs. Padding

| Feature | Stride ($S$) | Padding ($P$) |
| :--- | :--- | :--- |
| **Definition** | Step size of filter movement | Extra border pixels added around input |
| **Primary Purpose** | Downsamples feature maps & speeds up compute | Preserves spatial dimensions & processes border regions |
| **Effect on Output Size** | Larger stride **decreases** output size ($O \propto \frac{1}{S}$) | Larger padding **increases** output size |
| **Border Data Handling** | May skip border details when $S > 1$ | Processes border pixels more extensively |

---

## 5. Role of ReLU in Convolutional Neural Networks

After every Convolutional layer, activations are passed through the **ReLU (Rectified Linear Unit)** activation function:

$$\text{ReLU}(x) = \max(0, x)$$

### Why ReLU is Used in CNNs:
1. **Introduces Non-Linearity:** Convolution is a purely linear operation (weighted multiplication and addition). Stacking linear conv layers without ReLU would collapse the network into a single linear model.
2. **Preserves Tensor Shapes Exactly:** ReLU operates element-wise. It changes all negative activations to `0` without altering tensor dimensions:
   $$[N, C, H, W] \xrightarrow{\text{ReLU}} [N, C, H, W]$$
3. **Mitigates Vanishing Gradients:** ReLU helps **mitigate** vanishing gradients compared with saturating activations such as Sigmoid and Tanh because its derivative is `1.0` for positive inputs ($x > 0$). However, it does not eliminate gradient problems entirely and can suffer from the **Dying ReLU** problem if neurons get permanently deactivated.
