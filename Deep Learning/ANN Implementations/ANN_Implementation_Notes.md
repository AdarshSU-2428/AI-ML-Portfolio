# 📘 Artificial Neural Networks (ANN) — Complete Implementation & Practice Notes

---

## 📌 Executive Summary

This document provides a comprehensive technical reference for three Artificial Neural Network implementations:
1. **NumPy Implementation From Scratch (`ANN_scratch.py`)**: A fundamental 2-layer ANN built using pure Python and NumPy to demystify matrix math, forward propagation, manual backpropagation, chain rule calculus, and gradient descent.
2. **Standard PyTorch Implementation (`Real_dataset_implementation.ipynb`)**: A production deep learning pipeline on tabular data ([adult_income.csv](adult_income.csv)) using scikit-learn preprocessing, stratified 3-way data splitting, PyTorch `DataLoader` batching, default PyTorch layer initialization, and Early Stopping.
3. **PyTorch Implementation with Kaiming Weight Initialization (`Real_dataset_with_weight_initialization.ipynb`)**: An enhanced deep learning pipeline introducing explicit **Kaiming (He) Normal Weight Initialization** (`nn.init.kaiming_normal_`) and zero bias initialization to optimize convergence stability for ReLU layers.

---

## 🧠 Part 1: ANN From Scratch using NumPy (`ANN_scratch.py`)

### 1.1 Overview & Problem Domain
- **File:** [ANN_scratch.py](ANN_scratch.py)
- **Objective:** Train a 2-layer Neural Network from scratch without deep learning frameworks to solve the non-linearly separable **XOR Logic Gate**.
- **Architecture:** 
  - **Input Layer:** 2 nodes ($x_1, x_2$)
  - **Hidden Layer:** 4 neurons with Sigmoid activation
  - **Output Layer:** 1 neuron with Sigmoid activation

---

### 1.2 Mathematical Foundations & Formulas

#### **1. Forward Propagation**
- **Layer 1 (Hidden Layer):**
  $$\mathbf{Z}_1 = \mathbf{X} \mathbf{W}_1 + \mathbf{b}_1, \quad \mathbf{A}_1 = \sigma(\mathbf{Z}_1)$$
- **Layer 2 (Output Layer):**
  $$\mathbf{Z}_2 = \mathbf{A}_1 \mathbf{W}_2 + \mathbf{b}_2, \quad \mathbf{A}_2 = \sigma(\mathbf{Z}_2)$$

#### **2. Activation Function (Sigmoid)**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Derivative of Sigmoid in terms of activation $a = \sigma(z)$:**
  $$\sigma'(z) = a (1 - a)$$

#### **3. Loss Function (Mean Squared Error)**
$$L(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

---

### 1.3 Execution Results (`ANN_scratch.py`)
```text
Epoch    0 | Loss = 0.347522
Epoch 1000 | Loss = 0.088241
Epoch 2000 | Loss = 0.007681
Epoch 3000 | Loss = 0.002824
Epoch 4000 | Loss = 0.001642

Predictions:
[[0.]
 [1.]
 [1.]
 [0.]]
```

---

## 🚀 Part 2: Standard PyTorch ANN Implementation (`Real_dataset_implementation.ipynb`)

### 2.1 Overview & Pipeline Architecture
- **File:** [Real_dataset_implementation.ipynb](Real_dataset_implementation.ipynb)
- **Dataset:** [adult_income.csv](adult_income.csv) (32,537 clean rows after purging 24 duplicates).
- **Goal:** Binary classification predicting whether annual income exceeds `$50K` (`<=50K` vs `>50K`).
- **Preprocessing:** `ColumnTransformer` applying `StandardScaler` (6 numeric features) and `OneHotEncoder` (8 categorical features), expanding input space to **108 features**.
- **Data Splitting:** Stratified 3-way split (**80% Train**, **10% Validation**, **10% Test**).
- **Architecture:** 3-layer network (`108 -> 64 (ReLU) -> 32 (ReLU) -> 1 (Logit)`).

---

### 2.2 Model Performance Results (`Real_dataset_implementation.ipynb`)

```text
Final Test Results:
- Accuracy:  86.54%
- Precision: 74.30%
- Recall:    67.47%
- F1-Score:  0.7072

Confusion Matrix:
[[2287  183]
 [ 255  529]]
```

---

## ⚡ Part 3: PyTorch ANN with Kaiming Weight Initialization (`Real_dataset_with_weight_initialization.ipynb`)

### 3.1 Overview & Key Architectural Enhancements
- **File:** [Real_dataset_with_weight_initialization.ipynb](Real_dataset_with_weight_initialization.ipynb)
- **What Changed:** Introduced an explicit `initialize_weights()` method inside the `AdultANN` class to initialize layer parameters before training.

```python
class AdultANN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

        self.initialize_weights()

    def initialize_weights(self):
        # Kaiming (He) Normal Initialization for ReLU activation layers
        nn.init.kaiming_normal_(
            self.fc1.weight,
            mode="fan_in",
            nonlinearity="relu"
        )
        nn.init.kaiming_normal_(
            self.fc2.weight,
            mode="fan_in",
            nonlinearity="relu"
        )
        # Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
```

---

### 3.2 Impact of Kaiming (He) Normal Initialization
1. **Mathematical Principle:** Kaiming initialization samples weights from a Gaussian distribution with zero mean and variance:
   $$\text{Var}(W) = \frac{2}{\text{fan\_in}}$$
   This accounts for the fact that ReLU zeroes out half of its inputs on average, preserving gradient variance across deep layers.
2. **Accelerated Convergence:**
   - The validation loss rapidly dropped to its global minimum of **0.3028** by **Epoch 3** (compared to Epoch 4 in default initialization).
   - Early stopping triggered cleanly at **Epoch 8** (restoring optimal Epoch 3 parameters).

---

### 3.3 Model Performance Results (`Real_dataset_with_weight_initialization.ipynb`)

```text
Final Test Results:
- Accuracy:  86.36%
- Precision: 73.74%
- Recall:    67.35%
- F1-Score:  0.7040

Confusion Matrix:
[[2282  188]
 [ 256  528]]
```

---

## 📊 Part 4: 3-Way Implementation Comparison & Summary

### 4.1 Side-by-Side Comparison Table

| Dimension | NumPy Scratch (`ANN_scratch.py`) | Standard PyTorch (`Real_dataset_implementation.ipynb`) | PyTorch + Kaiming Init (`Real_dataset_with_weight_initialization.ipynb`) |
| :--- | :--- | :--- | :--- |
| **Primary Focus** | Mathematical Fundamentals & Chain Rule | Scalable Deep Learning Pipeline | Optimized Parameter Initialization & Convergence |
| **Dataset & Task** | XOR Gate (4 Samples) | Adult Income (32,537 Rows, 108 Cols) | Adult Income (32,537 Rows, 108 Cols) |
| **Weight Initialization** | Gaussian Random (`np.random.randn`) | PyTorch Default Uniform | **Kaiming (He) Normal (`nn.init.kaiming_normal_`)** |
| **Bias Initialization** | Zero (`np.zeros`) | PyTorch Default Uniform | **Zero Initialization (`nn.init.zeros_`)** |
| **Activation Functions** | Sigmoid | ReLU (Hidden) + Raw Logit (Output) | ReLU (Hidden) + Raw Logit (Output) |
| **Convergence Point** | ~3,000 Epochs | Lowest Val Loss at Epoch 4 (0.3037) | **Lowest Val Loss at Epoch 3 (0.3028)** |
| **Test Accuracy** | 100% (XOR) | **86.54%** | **86.36%** |
| **Test Precision** | N/A | **74.30%** | **73.74%** |
| **Test Recall** | N/A | **67.47%** | **67.35%** |
| **Test F1-Score** | N/A | **0.7072** | **0.7040** |

---

### 4.2 Summary of Key Insights

1. **Explicit Weight Initialization Matters:**
   - Applying **Kaiming (He) Normal initialization** prevents vanishing/exploding gradients in networks using ReLU activation functions, stabilizing gradient flow and leading to faster early convergence (Epoch 3 vs Epoch 4).
2. **Robust Real-World Pipeline Strategy:**
   - Combining stratified 3-way data splitting, `ColumnTransformer` preprocessing, PyTorch mini-batch `DataLoaders`, and early stopping ensures strong generalization performance (~86.5% accuracy, ~0.707 F1-score) on real-world imbalanced tabular data.
