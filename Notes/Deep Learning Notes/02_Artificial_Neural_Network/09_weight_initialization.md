# 09. Weight Initialization in Neural Networks

**Weight Initialization** is the process of setting the starting internal weights ($W$) of a neural network before training begins. Proper initialization is crucial for fast convergence, numerical stability, and avoiding **Vanishing** or **Exploding Gradients**.

---

## 1. Why Weight Initialization Matters

When a neural network starts training, its weights are updated iteratively using Gradient Descent:

$$w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w}$$

If initial weights are chosen poorly:
* **Too Large Initial Weights:** Cause activations ($z = \sum w_i x_i + b$) to blow up to massive values. This saturates non-linear activations (like Sigmoid or Tanh), producing tiny derivatives ($\sigma'(z) \approx 0$) and causing **Vanishing Gradients** or numerical overflow (`NaN`).
* **Too Small Initial Weights:** Cause signal outputs to shrink closer and closer to $0$ as they pass through deep layers. Signals vanish before reaching the output layer.
* **Goal of Weight Initialization:** Maintain the **variance of activations** (forward pass) and **variance of gradients** (backward pass) roughly equal across all layers!

$$\text{Var}(a^{[l]}) \approx \text{Var}(a^{[l-1]}) \quad \text{and} \quad \text{Var}\left(\frac{\partial L}{\partial z^{[l]}}\right) \approx \text{Var}\left(\frac{\partial L}{\partial z^{[l+1]}}\right)$$

---

## 2. Naive & Flawed Initialization Strategies

### A. Zero Initialization ($W = 0$)
Setting all initial weights to zero is a critical mistake.
* **The Symmetry Problem:** If all weights are 0, every neuron in a hidden layer computes the exact same weighted sum ($z = 0$), outputs the exact same activation, and receives the exact same gradient during backpropagation.
* **Result:** All neurons update identically, behaving like a single neuron. The network **fails to break symmetry** and cannot learn complex features!

---

### B. Constant Initialization ($W = c$)
Setting all weights to a non-zero constant (e.g., $W = 0.5$) suffers from the exact same **symmetry problem** as zero initialization.

---

### C. Small Random Initialization (e.g., $W \sim \mathcal{N}(0, 0.01^2)$)
* Works fine for shallow networks (1-2 hidden layers).
* **Fails in Deep Networks:** As inputs pass through 10+ layers, multiplying by tiny numbers ($< 1$) repeatedly shrinks activation values to 0. Signal vanishes completely.

---

### D. Large Random Initialization (e.g., $W \sim \mathcal{N}(0, 1^2)$)
* Causes activations $z = \sum w_i x_i + b$ to become very large (e.g., $z = +15$ or $z = -12$).
* Saturates Sigmoid/Tanh activations into flat regions where derivative is near $0$, stopping backpropagation updates (**Vanishing Gradient**).

---

## 3. The Concepts of Fan-In ($n_{in}$) and Fan-Out ($n_{out}$)

To scale weights properly, we must account for the **architecture of each layer**.

```
Input Neurons (fan_in)               Target Neuron                 Output Neurons (fan_out)
       ( x₁ ) ───────────(w₁)───────────┐
       ( x₂ ) ───────────(w₂)───────────┼───────> ( Neuron j ) ───────> ( Next Layer Neurons )
       ( x₃ ) ───────────(w₃)───────────┘
     n_in = 3 inputs                                                 n_out = number of outputs
```

### Definitions:
* **$fan\_in$ ($n_{in}$):** The number of incoming input connections (neurons) feeding into a given layer.
* **$fan\_out$ ($n_{out}$):** The number of outgoing connections (neurons in the next layer receiving signals from this layer).

---

### Mathematical Proof: Variance of $fan\_in$

Consider a single neuron computing the weighted sum of $n_{in}$ independent inputs $x_i$ with weights $w_i$:

$$z = \sum_{i=1}^{n_{in}} w_i x_i$$

Assuming inputs and weights are independent with zero mean ($\mathbb{E}[x] = 0, \mathbb{E}[w] = 0$):

$$\text{Var}(z) = \text{Var}\left(\sum_{i=1}^{n_{in}} w_i x_i\right) = \sum_{i=1}^{n_{in}} \text{Var}(w_i x_i)$$

For independent zero-mean variables: $\text{Var}(w_i x_i) = \text{Var}(w_i) \cdot \text{Var}(x_i)$.

If all inputs have variance $\text{Var}(x)$ and all weights have variance $\text{Var}(w)$:

$$\text{Var}(z) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$$

> **Key Insight:** To keep the variance of the output equal to the variance of the input ($\text{Var}(z) = \text{Var}(x)$), we must set:
> $$\text{Var}(w) = \frac{1}{n_{in}}$$

This fundamental equation forms the basis of modern initialization techniques!

---

## 4. Modern Weight Initialization Methods

### A. Xavier / Glorot Initialization

Introduced by **Xavier Glorot and Yoshua Bengio (2010)**.

* **Best Activation Functions:** **Sigmoid** and **Tanh** (functions that are linear around the origin $0$).
* **Core Philosophy:** Balances forward variance ($\text{Var}(w) = \frac{1}{n_{in}}$) and backward gradient variance ($\text{Var}(w) = \frac{1}{n_{out}}$) by taking their average.

#### Variance Formula:
$$\text{Var}(w) = \frac{2}{n_{in} + n_{out}}$$

#### Distributions:
1. **Xavier Normal Distribution:**
   $$W \sim \mathcal{N}\left(0, \sigma^2 = \frac{2}{n_{in} + n_{out}}\right)$$

2. **Xavier Uniform Distribution:**
   $$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, +\sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

*(Note: If considering $fan\_in$ only, $\text{Var}(w) = \frac{1}{n_{in}}$).*

---

### B. He / Kaiming Initialization

Introduced by **Kaiming He et al. (2015)**.

* **Best Activation Functions:** **ReLU** and **Leaky ReLU**.

#### Why Xavier Fails on ReLU:
* ReLU zeroes out all negative inputs ($\max(0, x)$).
* Because half of the input signals are set to $0$, ReLU **halves the variance** of activations at each layer:
  $$\text{Var}(\text{ReLU}(z)) = \frac{1}{2} \text{Var}(z)$$
* To compensate for losing half the variance, weights must be scaled up by a factor of 2!

#### Variance Formula:
$$\text{Var}(w) = \frac{2}{n_{in}}$$

#### Distributions:
1. **He Normal Distribution:**
   $$W \sim \mathcal{N}\left(0, \sigma^2 = \frac{2}{n_{in}}\right)$$

2. **He Uniform Distribution:**
   $$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}}}, +\sqrt{\frac{6}{n_{in}}}\right)$$

---

### C. LeCun Initialization

Introduced by **Yann LeCun (1998)**.

* **Best Activation Functions:** **SELU** (Scaled Exponential Linear Unit) or standard Tanh.
* **Variance Formula:**
  $$\text{Var}(w) = \frac{1}{n_{in}}$$
* **Distributions:**
  $$W \sim \mathcal{N}\left(0, \sigma^2 = \frac{1}{n_{in}}\right) \quad \text{or} \quad W \sim \mathcal{U}\left(-\sqrt{\frac{3}{n_{in}}}, +\sqrt{\frac{3}{n_{in}}}\right)$$

---

## 5. How to Initialize Bias ($b$)?

Unlike weights, **bias vectors ($b$)** do not suffer from the symmetry problem because weights are already randomly initialized.

* **Standard Practice:** Initialize all bias terms to **`0`** ($b = 0$).
* **Exception (Leaky ReLU / ReLU):** Sometimes initialized to a small positive constant like **`0.01`** to ensure neurons fire initially and avoid dying ReLUs during early steps.

---

## 6. Summary Selection Guide

| Initialization Method | Target Activation Functions | Variance Formula $\text{Var}(w)$ | Uniform Bound $[-\text{limit}, +\text{limit}]$ |
| :--- | :--- | :--- | :--- |
| **Xavier / Glorot** | **Sigmoid, Tanh, Softmax** | $\frac{2}{n_{in} + n_{out}}$ | $\pm \sqrt{\frac{6}{n_{in} + n_{out}}}$ |
| **He / Kaiming** | **ReLU, Leaky ReLU, PReLU** | $\frac{2}{n_{in}}$ | $\pm \sqrt{\frac{6}{n_{in}}}$ |
| **LeCun** | **SELU, Tanh** | $\frac{1}{n_{in}}$ | $\pm \sqrt{\frac{3}{n_{in}}}$ |
