# 04. Activation Functions

An **Activation Function** decides whether an artificial neuron should "fire" (pass its signal forward) or stay silent. It introduces non-linearity into the network, transforming simple weighted sums into powerful pattern recognizers.

---

## 1. Why Activation Functions? (The Non-Linearity Secret)

Without activation functions, a neural network is just a series of linear matrix multiplications:

$$z_1 = W_1 x + b_1 \implies z_2 = W_2 z_1 + b_2 = W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2)$$

Notice that $(W_2 W_1)$ is just a new single matrix $W'$, and $(W_2 b_1 + b_2)$ is just a new bias $b'$. 

> **Crucial Fact:** A neural network with 100 hidden layers using NO activation functions is mathematically equivalent to a **single-layer linear regression model**!

By placing a **non-linear activation function** after each layer, we enable the network to learn complex shapes, curves, and non-linear boundaries.

---

## 2. Common Activation Functions

### A. Linear Activation Function
* **Formula:** $f(x) = x$
* **Range:** $(-\infty, +\infty)$
* **Characteristics:** Passes input directly to output without any transformation.
* **Use Case:** Used exclusively in the **Output Layer** for linear regression problems.

---

### B. Sigmoid Activation Function
* **Formula:** $\sigma(x) = \frac{1}{1 + e^{-x}}$
* **Range:** $(0, 1)$
* **S-shaped curve** that compresses any input value into a probability between 0 and 1.
* **Pros:** Smooth, clear probabilistic interpretation.
* **Cons:** Causes **Vanishing Gradient Problem** in deep networks (saturates for large positive/negative inputs where derivative becomes nearly 0).

---

### C. Tanh (Hyperbolic Tangent) Activation Function
* **Formula:** $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
* **Range:** $(-1, 1)$
* **S-shaped curve** centered around zero (**Zero-Centered**).
* **Pros:** Outperforms Sigmoid in hidden layers because zero-centered outputs prevent directional bias during gradient descent updates.
* **Cons:** Still suffers from the vanishing gradient problem for extreme values.

---

### D. ReLU (Rectified Linear Unit)
* **Formula:** $f(x) = \max(0, x)$
* **Range:** $[0, +\infty)$
* **How it works:** Outputs $0$ if input is negative; outputs $x$ if input is positive.
* **Pros:** Extremely computationally fast; **eliminates vanishing gradients** for positive inputs. Default choice for hidden layers in modern deep learning!
* **Cons:** **"Dying ReLU" Problem**—if a neuron gets stuck outputting negative numbers, its gradient becomes 0 and the neuron dies permanently.

---

### E. Leaky ReLU
* **Formula:** $f(x) = \max(\alpha x, x)$ where $\alpha$ is a small constant (e.g., $0.01$).
* **Range:** $(-\infty, +\infty)$
* **How it works:** Instead of returning $0$ for negative inputs, it returns a small negative slope ($\alpha x$).
* **Pros:** Fixes the "Dying ReLU" problem by ensuring a small non-zero gradient even for negative inputs.

---

## 3. Output Layer Activation Functions Guide

Selecting the correct activation function for the **Output Layer** depends strictly on your task type:

```
                                  [ Problem Type ]
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
[ Binary Classification ]    [ Multi-Class Classification ]       [ Regression ]
   (Spam vs Not Spam)             (Digit Recognition 0-9)       (Predicting House Price)
         |                               |                               |
         v                               v                               v
  Sigmoid (1 Neuron)            Softmax (N Neurons)             Linear (1 Neuron)
  Outputs prob [0, 1]           Outputs prob sum = 1            Outputs continuous (-∞, +∞)
```

| Task Type | Target | Output Neurons | Output Activation | Example Output |
| :--- | :--- | :--- | :--- | :--- |
| **Binary Classification** | 2 mutually exclusive classes (0 or 1) | **1 neuron** | **Sigmoid** | `0.87` (87% prob of Class 1) |
| **Multi-Class Classification** | $K$ mutually exclusive classes | **$K$ neurons** | **Softmax** | `[0.1, 0.7, 0.2]` (Sums to 1.0) |
| **Regression** | Continuous numeric value | **1 neuron** | **Linear / Identity** | `$452,100` |

---

## Summary Cheat Sheet

| Activation | Range | Zero-Centered? | Primary Use |
| :--- | :--- | :--- | :--- |
| **Linear** | $(-\infty, +\infty)$ | Yes | Output layer for Regression |
| **Sigmoid** | $(0, 1)$ | No | Output layer for Binary Classification |
| **Tanh** | $(-1, 1)$ | Yes | Hidden layers (older models) |
| **ReLU** | $[0, +\infty)$ | No | **Default for Hidden Layers** |
| **Leaky ReLU** | $(-\infty, +\infty)$ | No | Hidden layers when neurons are dying |
| **Softmax** | $(0, 1)$ | No | Output layer for Multi-Class Classification |
