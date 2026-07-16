# Support Vector Machine (SVM) Study Notes

This document contains detailed study notes on Support Vector Machines (SVM) based on the following implementations:
*   **Toy Dataset Implementation (make_classification / PCA visualization)**: [toy_dataset_implementation.ipynb](../Machine%20Learning/Support%20Vector%20Machines/toy_dataset_implementation.ipynb)
*   **Real-world Dataset Implementation (Heart Disease Prediction)**: [Real_dataset_implement.ipynb](../Machine%20Learning/Support%20Vector%20Machines/Real_dataset_implement.ipynb)

---

## 1. Intuition

Imagine you have a table covered with blue and red balls, and you want to place a wooden stick on the table to separate them. 

1.  **Multiple Solutions**: There are many angles and positions where the stick can separate the balls.
2.  **The Optimal Boundary (Maximal Margin)**: If you place the stick too close to the red balls, a slight nudge might cause a new red ball to end up on the wrong side. To make the division as safe and robust as possible, you should place the stick exactly in the middle, maximizing the empty space (the **margin**) on both sides.
3.  **Support Vectors**: The balls that lie closest to the stick on either side are the ones holding the boundary in place. If you move any other ball farther away, the position of the stick remains unchanged. These critical boundary points are the **Support Vectors**.
4.  **Hard Margin vs. Soft Margin**: 
    *   **Hard Margin**: Assumes the data is perfectly separable and allows zero classification errors.
    *   **Soft Margin**: Allows a few points to cross the boundary or fall inside the margin to keep the boundary simple and generalizable (controlled by the regularization parameter $C$).
5.  **The Kernel Trick**: What if the balls are mixed in a circle (red in the center, blue on the outside)? No straight stick can separate them in 2D. However, if you "lift" the table into a 3D bowl shape, you can pass a flat sheet of paper (a 2D hyperplane) straight through to cut the red balls away from the blue ones. This mathematical "lifting" into higher dimensions is called the **Kernel Trick**.

### Timeline of SVM Evolution

The progression of SVM design helps explain why kernel transformations were invented:

```text
Linear Data
     │
     ▼
Linear SVM (Hard Margin) ──► Fails if any noise or overlap is present
     │
     ▼
Soft Margin SVM          ──► Introduces slack variables (C) to handle noise
     │
     ▼
Non-linear Data          ──► Fails if the true decision boundary is curved
     │
     ▼
Kernel Trick             ──► Projects data to higher dimensions implicitly
     │
     ▼
RBF / Polynomial Kernels ──► Solves complex non-linear classification tasks
```

---

## 2. Model Comparison Table

| Feature | Logistic Regression | Decision Trees | K-Nearest Neighbors (KNN) | Support Vector Machine (SVM) |
| :--- | :--- | :--- | :--- | :--- |
| **Model Type** | Parametric | Non-parametric | Non-parametric (Instance-based) | Non-parametric (Kernel-based) |
| **Training Complexity**| $O(N \cdot D)$ (Iterative) | $O(N \cdot D \log N)$ (Recursive) | $O(1)$ (Lazy learner) | $O(N^2 \cdot D)$ to $O(N^3 \cdot D)$ (Quadratic Programming) |
| **Testing/Inference** | $O(D)$ (Dot product) | $O(\text{Tree Depth})$ (Fast) | $O(N \cdot D)$ (Slow, distance check) | $O(N_{sv} \cdot D)$ (Fast, dot product with support vectors) |
| **Feature Scaling** | Recommended for convergence | **Not Required** | **Strictly Required** | **Strictly Required** (Highly sensitive to scale differences) |
| **Outliers** | Sensitive | Robust | Sensitive | Robust (Only boundary points matter, but outliers near margin shift it) |
| **Non-Linearity** | Requires manual engineering | Handles naturally | Handles naturally | Handles naturally (via Kernel Trick) |
| **Correlated Features** | Handled moderately | Handled naturally | Handled naturally | Handled well (finds separating hyperplane) |

*(where $N$ = number of samples, $D$ = number of features, $N_{sv}$ = number of support vectors)*

---

## 3. How SVM Works (Step-by-Step Inference)

Unlike lazy learners like KNN, SVM is an **eager learner** that solves a convex optimization problem during training to construct a separating hyperplane.

```text
Query Point (X_test)
     │
     ▼
[ Step 1: Feature Scaling ]                    ──► Standardize features using training mean/std (z-score)
     │
     ▼
[ Step 2: Kernel Similarity w/ Support Vectors ] ──► Compute similarity only with the learned Support Vectors
     │
     ▼
[ Step 3: Compute Decision Function ]           ──► Calculate f(x) = sum(alpha_i * y_i * K(x_i, x) + b)
     │
     ▼
[ Step 4: Sign Classification ]                 ──► Determine class based on the sign (+1 or -1) of f(x)
     │
     ▼
Final Prediction (y_pred)
```

---

## 4. Key Assumptions of SVM

*   **Feature Scale Uniformity**: SVM assumes all features are on the same scale. Features with larger scales (like Cholesterol ranging from 100-500) will dominate the margin calculation over smaller features (like age or binary features).
*   **Linear Separability in Some Dimension**: SVM assumes that either the data is linearly separable in its raw space, or it can be projected using a kernel into a higher-dimensional space where it becomes linearly separable.
*   **Binary Target**: Standard SVM natively assumes binary classification (targets are $-1$ and $+1$). Multiclass classification is handled via wrappers like One-vs-Rest (OvR) or One-vs-One (OvO).

---

## 5. Mathematical Underpinnings

### A. The Hyperplane Equation
The decision boundary is represented as:

$$w^T x + b = 0$$

Where $w$ is the weight vector (perpendicular to the hyperplane) and $b$ is the bias. 

For any data point $x_i$, the model predicts:
*   $w^T x_i + b \ge +1$ if the true label $y_i = +1$
*   $w^T x_i + b \le -1$ if the true label $y_i = -1$

This can be unified as: $y_i (w^T x_i + b) \ge 1$.

### B. Margin and Optimization Objective
The distance between the hyperplane and the closest support vectors is $\frac{1}{\|w\|}$. The total margin width is:

$$\text{Margin} = \frac{2}{\|w\|}$$

To maximize this margin, we need to minimize $\|w\|$. For mathematical convenience, we minimize $\frac{1}{2}\|w\|^2$.

#### 1. Hard Margin SVM (No errors allowed):
$$\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{subject to } y_i (w^T x_i + b) \ge 1, \quad \forall i$$

#### 2. Soft Margin SVM (Tolerating errors with Slack Variables $\xi_i$):
$$\min_{w, b, \xi} \left( \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \right)$$

$$\text{subject to } y_i (w^T x_i + b) \ge 1 - \xi_i \quad \text{and} \quad \xi_i \ge 0, \quad \forall i$$

*   **$\xi_i$ (Slack)**: Measures how far a data point is on the wrong side of the margin boundary.
*   **$C$ (Hyperparameter)**: Controls the trade-off between margin width and training errors.

### C. Hinge Loss
SVM can be seen as minimizing a loss function called **Hinge Loss** combined with L2 regularization:

$$\text{Hinge Loss} = \max(0, 1 - y_i(w^T x_i + b))$$

If a point is correctly classified and outside the margin, the loss is $0$. Otherwise, the loss increases linearly with the distance from the margin.

### D. The Kernel Trick
When data is not linearly separable in the raw space, we project it into a higher-dimensional space using a mapping function $\phi(x)$. The optimization problem only depends on the dot products of the samples: $\phi(x_i)^T \phi(x_j)$. 

Instead of explicitly calculating the high-dimensional vectors (which is computationally expensive), we use a **Kernel Function** $K(x_i, x_j)$ to compute this dot product directly in the lower-dimensional space:

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

#### Common Kernels & Comparison:

| Kernel | Formula | Boundary | Speed | When to Use |
| :--- | :--- | :--- | :--- | :--- |
| **Linear** | $K(x_i, x_j) = x_i^T x_j$ | Straight Line | Fastest | High-dimensional data, linear separation |
| **Polynomial** | $K(x_i, x_j) = (\gamma x_i^T x_j + r)^d$ | Curved | Medium | Polynomial / structured relationships |
| **RBF (Gaussian)** | $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$ | Highly Flexible | Slower | Most nonlinear datasets (general purpose) |

---

## 6. Hyperparameters of SVM

### A. $C$ (Regularization Parameter)
*   **What it does**: Controls the penalty for misclassified training points.
*   **Small $C$** (e.g. $0.01$): Strong regularization. Wide margin, tolerates more training errors. Focuses on a simple, generalizable boundary (reduces variance, prevents overfitting).
*   **Large $C$** (e.g. $100$): Weak regularization. Narrow margin, tries to classify every training point correctly. Focuses on minimizing training error (increases variance, risks overfitting).

### B. Gamma ($\gamma$)
*   **What it does**: Defines the radius of influence of a single training point (used in `rbf`, `poly`, and `sigmoid` kernels).
*   **Small $\gamma$**: A single point has a wide radius of influence. The decision boundary is smooth and simple.
*   **Large $\gamma$**: A single point has a narrow radius of influence. The model creates complex, tight boundary shapes around individual points (risks overfitting).

### C. Degree ($d$)
*   **What it does**: Set only for the polynomial kernel (`kernel='poly'`). Controls the complexity of the polynomial boundary.

---

## 7. Advantages & Disadvantages

### Advantages
*   **Effective in High Dimensions**: Excellent when the number of features exceeds the number of samples.
*   **Memory Efficient**: Uses only a subset of training points (support vectors) in the decision function.
*   **Versatile**: Different kernel functions can be specified to fit complex, custom decision boundaries.
*   **Convex Optimization**: The objective function has a unique global minimum, meaning it does not get stuck in local minima (unlike Neural Networks).

### Disadvantages
*   **Not Scalable to Large Datasets**: Training complexity scales between $O(N^2)$ and $O(N^3)$, making it very slow for datasets with $>100,000$ samples.
*   **Highly Sensitive to Noise**: If classes overlap heavily, noisy boundary points (outliers) will corrupt the margin.
*   **No Direct Probabilities**: Natively outputs distance from the hyperplane. Probabilities (`predict_proba()`) must be estimated using expensive internal cross-validation (Platt Scaling).
*   **Sensitivity to Hyperparameters**: Choosing the correct kernel, $C$, and $\gamma$ is crucial; a bad choice leads to poor generalization.

---

## 8. Implementation Analysis & Performance Comparison

### A. Toy Dataset (Synthetic Classification)
*   **Task**: Binary classification on a synthetic dataset generated with 6 features.
*   **Observations**:
    *   **Scaling Impact**: Features were scaled using `StandardScaler`.
    *   **Generalization Gap**: Under the default parameters ($C=1.0$), the RBF model showed high overfitting (an 8-9% gap between training and testing accuracy).
    *   **Tuning Solution**: By decreasing $C$ to $0.1$ (increasing regularization), the gap was reduced to under 4%, stabilizing test performance.
*   **Results**:
    *   **RBF Kernel** ($C=0.1$): Training 91.50% | Testing **88.00%** (Best non-linear separation)
    *   **Linear Kernel** ($C=0.1$): Training 89.75% | Testing **84.00%**
    *   **Poly Kernel** ($C=0.1$, $deg=3$): Training 88.75% | Testing **83.00%**

---

### B. Real-world Dataset (Heart Disease Prediction)
*   **Task**: Predict the presence of heart disease using 13 clinical attributes.
*   **Data Cleaning & Preprocessing**:
    *   **Duplicates Removed**: Dropped 723 duplicate rows to prevent data leakage, leaving 302 unique records.
    *   **One-Hot Encoding**: Nominal categories (`cp`, `restecg`, `slope`, `thal`) were one-hot encoded using `pd.get_dummies` with `drop_first=True`. This expanded the dataset to 19 features.
    *   **Scaling**: `StandardScaler` was applied to numeric features to align range scales (e.g. standardizing Cholesterol range 126-564 down to comparable z-scores).
*   **Hyperparameter Tuning**:
    *   **Best Parameters**: `{'C': 0.1, 'kernel': 'linear', 'gamma': 'scale'}`
    *   **Training Accuracy**: **87.55%**
    *   **Testing Accuracy**: **83.61%**
    *   **Overfitting Gap**: **3.95%**
*   **Insight on Linear Kernel Success**: The linear kernel performed best on this dataset because the one-hot encoding expanded the feature space to 19 dimensions, rendering the classes easily separable by a linear hyperplane. Additionally, key clinical markers (age, blood pressure, cholesterol) exhibit a naturally monotonic, linear relationship with heart disease risk.

---

## 9. When to Use / Avoid SVM

### Avoid SVM when:
| Don't use when... | Reason |
| :--- | :--- |
| **Dataset size is very large ($N > 100,000$)** | The $O(N^3)$ computational scaling makes training unacceptably slow. |
| **Data has high class overlap (noise)** | Outliers will distort the margin, leading to poor generalization. |
| **Calibrated probability outputs are needed** | Platt scaling to get stable probabilities is computationally expensive and slow. |
| **Interpretability is a priority** | High-dimensional RBF kernel spaces act as black boxes; feature importances are not easily extractable. |

---

## 10. Quick Revision Summary Table

| Property | Value |
| :--- | :--- |
| **Type** | Supervised |
| **Problem** | Classification / Regression (SVR) |
| **Parametric** | No (Non-parametric) |
| **Lazy Learner** | No (Eager learner) |
| **Scaling Required** | **Strictly Required** (Z-score StandardScaler) |
| **Training Speed** | Slow ($O(N^2 \cdot D)$ to $O(N^3 \cdot D)$) |
| **Prediction Speed** | Fast ($O(N_{sv} \cdot D)$) |
| **Sensitive to Outliers** | Yes (Specifically outliers near the margin) |
| **Sensitive to Scaling** | Yes (Highly sensitive) |
| **Main Hyperparameters**| `C` (Inverse regularization), `kernel`, `gamma` (influence scale) |
| **Loss Function** | Hinge Loss + L2 Regularization |
| **Core Principle** | Maximize the margin width $\frac{2}{\|w\|}$ between classes |
