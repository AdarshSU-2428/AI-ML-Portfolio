# 📘 Artificial Neural Networks (ANN) — Complete Implementation & Practice Notes

---

## 📌 Executive Summary

This document serves as a comprehensive technical reference and experimental benchmark for Artificial Neural Network (ANN) implementations built during the course:
1. **NumPy Implementation From Scratch (`ANN_scratch.py`)**: A fundamental 2-layer ANN built using pure Python and NumPy to demystify matrix math, forward propagation, manual backpropagation, chain rule calculus, and gradient descent on the XOR problem.
2. **Production PyTorch Real-Dataset Pipelines**: Real-world deep learning pipelines evaluated on tabular data ([adult_income.csv](adult_income.csv), 32,537 clean rows, 108 features) across seven distinct hyperparameter and architectural experiments:
   - **Baseline + Early Stopping**
   - **Kaiming (He) Weight Initialization**
   - **Kaiming Initialization + AdamW Optimizer**
   - **Baseline + Dropout (0.2) Regularization**
   - **Baseline + Batch Normalization (`nn.BatchNorm1d`)**
   - **Baseline + BatchNorm + Dropout (0.2)**
   - **Baseline + ReduceLROnPlateau Scheduler**

---

## 🧠 Part 1: ANN From Scratch using NumPy (`ANN_scratch.py`)

### 1.1 Overview & Problem Domain
- **File:** [ANN_scratch.py](ANN_scratch.py)
- **Objective:** Train a 2-layer Neural Network from scratch without deep learning frameworks to solve the non-linearly separable **XOR Logic Gate**.
- **Architecture:** 
  - **Input Layer:** 2 nodes ($x_1, x_2$)
  - **Hidden Layer:** 4 neurons with Sigmoid activation
  - **Output Layer:** 1 neuron with Sigmoid activation

### 1.2 Mathematical Foundations
- **Forward Pass:** $\mathbf{Z}_1 = \mathbf{X} \mathbf{W}_1 + \mathbf{b}_1, \, \mathbf{A}_1 = \sigma(\mathbf{Z}_1), \, \mathbf{Z}_2 = \mathbf{A}_1 \mathbf{W}_2 + \mathbf{b}_2, \, \mathbf{A}_2 = \sigma(\mathbf{Z}_2)$
- **Sigmoid Derivative:** $\sigma'(z) = a(1 - a)$
- **MSE Loss & Derivative:** $L = \frac{1}{N} \sum (\hat{y} - y)^2, \quad \frac{\partial L}{\partial \mathbf{A}_2} = \frac{2}{N}(\mathbf{A}_2 - y)$
- **Output Layer Gradients:** $\delta_2 = \frac{2}{N}(\mathbf{A}_2 - y) \odot \mathbf{A}_2(1 - \mathbf{A}_2), \quad \nabla_{\mathbf{W}_2} L = \mathbf{A}_1^T \delta_2$
- **Hidden Layer Gradients:** $\delta_1 = (\delta_2 \mathbf{W}_2^T) \odot \mathbf{A}_1(1 - \mathbf{A}_1), \quad \nabla_{\mathbf{W}_1} L = \mathbf{X}^T \delta_1$

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

## 🚀 Part 2: Real-World PyTorch Pipeline Setup

### 2.1 Dataset Specifications & Preprocessing
- **Dataset:** [adult_income.csv](adult_income.csv) (32,537 clean rows after removing 24 duplicate records).
- **Target:** Binary classification ($y \in \{0, 1\}$) predicting income (`<=50K` $\rightarrow 0$, `>50K` $\rightarrow 1$).
- **Missing Values:** String `'?'` entries in `workclass`, `occupation`, and `native-country` imputed as `'Unknown'`.
- **Feature Scaling & Encoding:** `ColumnTransformer` applying `StandardScaler` (6 numerical columns) and `OneHotEncoder` (8 categorical columns), expanding the input dimension to **108 features**.
- **Data Partitioning:** Stratified 3-way split maintaining class balance (~75.9% Class 0 vs ~24.1% Class 1):
  - **Train Set (80%):** 26,029 samples
  - **Validation Set (10%):** 3,254 samples
  - **Test Set (10%):** 3,254 samples

### 2.2 Network Architecture (`AdultANN`)
- **Structure:** `108 (Inputs) -> 64 (Hidden 1 + ReLU) -> 32 (Hidden 2 + ReLU) -> 1 (Output Logit)`
- **Loss Function:** `nn.BCEWithLogitsLoss()`
- **Batching & Streaming:** PyTorch `DataLoader` with `batch_size=128`.

---

## 🧪 Part 3: Detailed Experimental Suite & Benchmark Results

### 3.1 Overview of Conducted Experiments

1. **Experiment 1: Baseline + Early Stopping ([Real_dataset_implementation.ipynb](Real_dataset_implementation.ipynb))**
   - Standard PyTorch linear layer initialization, Adam optimizer (`lr=0.001`), and Early Stopping with `patience=5`.
2. **Experiment 2: Kaiming (He) Weight Initialization ([Real_dataset_with_weight_initialization.ipynb](Real_dataset_with_weight_initialization.ipynb))**
   - Added explicit `nn.init.kaiming_normal_` weight initialization (`nonlinearity='relu'`) and zero bias initialization (`nn.init.zeros_`).
3. **Experiment 3: Kaiming Initialization + AdamW Optimizer ([Real_dataset_with_weight_initialization.ipynb](Real_dataset_with_weight_initialization.ipynb))**
   - Retained Kaiming initialization and upgraded the optimizer to **AdamW** (`torch.optim.AdamW`, `lr=0.001`), decoupling weight decay L2 regularization from adaptive gradient updates.
4. **Experiment 4: Baseline + Dropout (0.2) Regularization ([Real_dataset_implementation.ipynb](Real_dataset_implementation.ipynb))**
   - Added `nn.Dropout(p=0.2)` layers after ReLU activations to randomly zero out 20% of neuron outputs during training to combat overfitting.
5. **Experiment 5: Baseline + Batch Normalization ([Real_dataset_implementation.ipynb](Real_dataset_implementation.ipynb))**
   - Added `nn.BatchNorm1d` layers after linear layers (`fc1`, `fc2`) and before ReLU activations to normalize intermediate activation distributions per batch.
6. **Experiment 6: Baseline + BatchNorm + Dropout (0.2) ([Real_dataset_implementation.ipynb](Real_dataset_implementation.ipynb))**
   - Combined `nn.BatchNorm1d` and `nn.Dropout(p=0.2)` in sequence (`fc -> bn -> relu -> dropout`) to balance internal covariate shift stabilization with activation regularization.
7. **Experiment 7: Baseline + ReduceLROnPlateau Scheduler ([Real_dataset_implementation.ipynb](Real_dataset_implementation.ipynb))**
   - Added `ReduceLROnPlateau(factor=0.5, patience=2)`. Learning rate automatically decayed from `0.001` to `0.0005` at Epoch 7 and down to `0.00025` at Epoch 11, reaching lowest validation loss (**0.3031**) at Epoch 8.

---

### 3.2 Master Performance Comparison Matrix

#### **Table 1: Test Set Performance Metrics ($N=3,254$)**

| Experiment | Accuracy | Precision | Recall | F1-Score | Best Val Loss Epoch | LR Decay Events |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline + Early Stopping** | 86.570% | 74.198% | 67.857% | 70.886% | Epoch 10 | None |
| **Kaiming Weight Initialization** | 86.355% | 73.743% | 67.347% | 70.400% | Epoch 3 | None |
| **Kaiming + AdamW Optimizer** | 86.325% | 73.706% | 67.219% | 70.314% | Epoch 2 | None |
| **Baseline + Dropout (0.2)** | **86.755%** | **75.178%** | 67.219% | 70.976% | Epoch 5 | None |
| **Baseline + Batch Normalization** | 86.601% | 75.072% | 66.454% | 70.501% | **Epoch 2** | None |
| **Baseline + BatchNorm + Dropout (0.2)** | 86.601% | 74.438% | 67.602% | 70.856% | Epoch 6 | None |
| **Baseline + ReduceLROnPlateau** | **86.755%** | 74.685% | **68.112%** | **71.247%** | **Epoch 8** | **Epoch 7 (0.0005), Epoch 11 (0.00025)** |

---

#### **Table 2: Confusion Matrix Breakdown ($N=3,254$)**

| Experiment | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline + Early Stopping** | 2,285 | 185 | 252 | 532 |
| **Kaiming Weight Initialization** | 2,282 | 188 | 256 | 528 |
| **Kaiming + AdamW Optimizer** | 2,282 | 188 | 257 | 527 |
| **Baseline + Dropout (0.2)** | **2,296** | 174 | 257 | 527 |
| **Baseline + Batch Normalization** | **2,297** | **173** | 263 | 521 |
| **Baseline + BatchNorm + Dropout (0.2)** | 2,288 | 182 | 254 | 530 |
| **Baseline + ReduceLROnPlateau** | 2,289 | 181 | **250** | **534** |

---

## 📊 Part 4: Technical Analysis & Key Engineering Takeaways

### 4.1 Impact of Learning Rate Scheduler (`ReduceLROnPlateau`)
- **Highest Test F1-Score & Recall:** Adding `ReduceLROnPlateau` achieved the highest F1-score (**71.247%**) and highest recall (**68.112%**), correctly detecting **534 True Positives**.
- **Dynamic Fine-Tuning:** Halving learning rate at Epoch 7 (`0.0005`) and Epoch 11 (`0.00025`) enabled the optimizer to take smaller steps near the loss minimum, driving validation loss down to **0.3031**.

### 4.2 Impact of Regularization (Dropout 0.2)
- **Highest Overall Accuracy & Precision:** Adding **Dropout (p=0.2)** achieved the highest test accuracy (**86.755%**) and highest precision (**75.178%**).
- **Reduction in False Positives:** Dropout reduced False Positives from 185 down to **174**, demonstrating that randomly deactivating neurons forces the network to learn redundant representations rather than over-relying on specific categorical features.

### 4.3 Impact of Batch Normalization (`nn.BatchNorm1d`)
- **Fastest Initial Convergence:** Batch Normalization achieved its lowest validation loss (**0.3061**) in just **2 Epochs**, demonstrating how normalizing layer inputs reduces internal covariate shift and allows faster stable learning.
- **Highest True Negatives:** Achieved **2,297 True Negatives** and reduced False Positives down to **173**.

### 4.4 Synergy of BatchNorm + Dropout (0.2)
- **Balanced Generalization & Recall:** Combining BatchNorm with Dropout (0.2) yielded **86.601% Accuracy**, **70.856% F1-score**, and high recall (**67.602%**), restoring 530 True Positives while keeping training stable.

### 4.5 Impact of Kaiming (He) Normal Initialization
- **Accelerated Initial Convergence:** Applying `nn.init.kaiming_normal_` allowed validation loss to reach its lowest point by **Epoch 3** (compared to Epoch 10 in baseline).
- **Variance Stabilization:** Kaiming initialization scales initial weight variance by $\text{Var}(W) = \frac{2}{\text{fan\_in}}$, ensuring gradients neither vanish nor explode across hidden ReLU activations.

### 4.6 Impact of AdamW Optimizer
- **Decoupled Weight Decay:** Standard Adam applies L2 weight decay by adding it directly to gradient calculations. **AdamW (`torch.optim.AdamW`)** decouples weight decay from adaptive momentum terms ($\hat{m}_t, \hat{v}_t$), yielding smoother parameter regularization and stable training progression.

---

### 4.7 Summary Recommendation
- **Best Classification Model (F1 & Recall):** **Baseline + ReduceLROnPlateau** ([Real_dataset_implementation.ipynb](Real_dataset_implementation.ipynb)) provides the best overall F1-score (**71.247%**) and Recall (**68.112%**).
- **Best Generalization Model (Precision & Accuracy):** **Baseline + Dropout (0.2)** achieves the highest Precision (**75.178%**) and fewest False Positives (**174**).
- **Best Training Speed:** **Baseline + Batch Normalization** reaches peak validation performance in just **2 Epochs**.
