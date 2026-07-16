# Naive Bayes Study Notes

This document contains detailed study notes on Naive Bayes classifiers based on the following implementations:
*   **Toy Dataset Implementation (Car Evaluation)**: [toy_dataset_implementation.ipynb](../Machine%20Learning/Naive%20Bayes/toy_dataset_implementation.ipynb)
*   **Real-world Dataset Implementation (Disease Prediction)**: [real_dataset_implementation.ipynb](../Machine%20Learning/Naive%20Bayes/real_dataset_implementation.ipynb)

---

## 1. Intuition

Imagine you are a doctor diagnosing a patient. The patient has a fever. You know that:
1. Malaria is a rare disease in your city (low overall chance / **Prior Probability**).
2. However, most people with Malaria suffer from a fever (high chance of symptom given disease / **Likelihood**).
3. Many other common diseases (like cold or flu) also cause fever.

To make an accurate diagnosis, you weigh the prior probability of the disease against how strongly the symptom points to it. This is the core of **Bayes' Theorem**.

It is called **"Naive"** because it makes a highly simplified assumption: it assumes that all symptoms (e.g., fever, cough, and chills) are **completely independent** of each other. Even though a cough and fever often go hand-in-hand in the real world, Naive Bayes ignores this correlation to make the mathematical calculations incredibly fast and simple.

---

## 2. Model Comparison Table

| Feature | Logistic Regression | Decision Trees | K-Nearest Neighbors (KNN) | Naive Bayes |
| :--- | :--- | :--- | :--- | :--- |
| **Model Type** | Parametric | Non-parametric | Non-parametric (Instance-based) | Parametric (Probabilistic) |
| **Training Complexity**| $O(N \cdot D)$ (Iterative) | $O(N \cdot D \log N)$ (Recursive) | $O(1)$ (Lazy learner) | $O(N \cdot D)$ (Fast single pass count) |
| **Testing/Inference** | $O(D)$ (Dot product) | $O(\text{Tree Depth})$ (Fast) | $O(N \cdot D)$ (Slow, distance check) | $O(C \cdot D)$ (Fast probability product) |
| **Feature Scaling** | Recommended for convergence | **Not Required** | **Strictly Required** | **Not Required** |
| **Outliers** | Highly sensitive | Robust | Sensitive | Robust (Calculates frequencies/means) |
| **Non-Linearity** | Requires manual engineering | Handles naturally | Handles naturally | Works well if features are independent |
| **Correlated Features** | Handled moderately | Handled naturally | Handled naturally | **Highly Sensitive** (Double-counts correlated features) |

*(where $N$ = number of samples, $D$ = number of features, $C$ = number of classes)*

---

## 3. How Naive Bayes Works (Step-by-Step Inference)

Naive Bayes is an **eager learner** that constructs a probability table during training. During testing, it performs rapid probability multiplications:

```text
Query Features (X_test)
     │
     ▼
[ Step 1: Prior Probabilities ]   ──► Retrieve prior probability P(y) for each class
     │
     ▼
[ Step 2: Likelihoods ]          ──► Calculate conditional probabilities P(x_i | y) for each feature
     │
     ▼
[ Step 3: Probability Product ]  ──► Multiply priors and likelihoods together for each class
     │
     ▼
[ Step 4: Argmax Selection ]     ──► Select class with the highest posterior probability
     │
     ▼
Final Prediction (y_pred)
```

---

## 4. Key Assumptions of Naive Bayes

*   **Conditional Independence Assumption**: It assumes that the presence (or value) of a particular feature is completely independent of the presence (or value) of any other feature, given the class label.
*   **Feature Equivalence**: It assumes all features contribute equally to the final prediction decision.
*   **Categorical/Continuous Alignment**: It assumes that the continuous features follow a normal distribution (for Gaussian NB) or that discrete features follow a multinomial/bernoulli distribution.

---

## 5. Mathematical Underpinnings

### A. Bayes' Theorem
The foundation of the classifier is Bayes' Theorem, which calculates the posterior probability $P(y \mid X)$:

$$P(y \mid X) = \frac{P(X \mid y) \cdot P(y)}{P(X)}$$

Where:
*   **$P(y \mid X)$** (Posterior Probability): Probability of class $y$ given features $X$.
*   **$P(X \mid y)$** (Likelihood): Probability of features $X$ occurring given class $y$.
*   **$P(y)$** (Prior Probability): Base probability of class $y$ occurring in the dataset.
*   **$P(X)$** (Evidence): Probability of the feature combination $X$ occurring (acts as a normalizing constant and is ignored during prediction).

### B. The Naive Assumption Formula
Given a feature vector $X = [x_1, x_2, \dots, x_D]$, the joint likelihood is simplified by assuming feature independence:

$$P(X \mid y) = P(x_1 \mid y) \cdot P(x_2 \mid y) \cdot \dots \cdot P(x_D \mid y) = \prod_{i=1}^{D} P(x_i \mid y)$$

Thus, the classification objective is to find the class $y$ that maximizes:

$$\hat{y} = \arg\max_{y} \left( P(y) \prod_{i=1}^{D} P(x_i \mid y) \right)$$

### C. Laplace Smoothing (Additive Smoothing)
If a specific feature value (e.g., a symptom) never appeared with a disease in the training set, then the likelihood $P(x_i \mid y) = 0$. Because we multiply probabilities, this single zero will make the entire posterior probability $0$:

$$\text{Probability} = P(y) \cdot P(x_1 \mid y) \cdot 0 \cdot P(x_D \mid y) = 0$$

To prevent this **"Zero-Frequency Problem"**, we add a small smoothing factor $\alpha$ (typically $\alpha = 1$) to the counts:

$$P(x_i \mid y) = \frac{N_{iy} + \alpha}{N_y + \alpha \cdot D}$$

Where:
*   $N_{iy}$ = number of times feature $i$ occurs in class $y$.
*   $N_y$ = total count of all features in class $y$.
*   $D$ = total number of features (vocabulary size).

---

## 6. The 5 Variants of Naive Bayes

| Model Variant | Feature Distribution | Best Used For |
| :--- | :--- | :--- |
| **`BernoulliNB`** | Binary / Boolean (0 or 1) | Presence/absence of symptoms, document classification (word exists / doesn't exist). |
| **`MultinomialNB`** | Discrete counts (integers $\ge 0$) | Text mining (word count frequencies in documents), rating stars (1 to 5). |
| **`CategoricalNB`** | Categorical classes ($> 2$ categories) | Datasets where features are non-binary categories (e.g. Car color: Red/Blue/Green). |
| **`GaussianNB`** | Continuous / Normal (Bell Curve) | Numerical measurements (e.g. blood pressure, temperature, house prices, age). |
| **`ComplementNB`** | Skewed Multinomial | Imbalanced text classification datasets (e.g., spam vs ham when spam is rare). |

---

## 7. Advantages & Disadvantages

### Advantages
*   **Incredibly Fast**: Training requires a single pass over the data to calculate frequencies; predictions are fast dot products.
*   **Performs well with high dimensions**: Excellent for text mining and document classification where features exceed samples.
*   **Robust to Noise**: Unrelated features don't highly distort the probability distribution because they average out.
*   **No scaling needed**: Since Naive Bayes estimates probability distributions rather than relying on distance calculations or gradient optimization, feature scaling is generally unnecessary.

### Disadvantages
*   **The Independence Assumption is unrealistic**: In real data, features are often highly correlated, causing the model to over-rely on duplicate signals.
*   **Zero-frequency issue**: Requires smoothing (Laplace) to handle unseen categories during inference.
*   **Poor probability estimators**: While it outputs the correct *class* prediction (via argmax), the raw probability outputs (like `predict_proba()`) are often extreme (very close to 0 or 1) because of the independent multiplication.

---

## 8. Implementation Analysis & Performance Comparison

### A. Toy Dataset (Car Evaluation)
*   **Task**: Predict car acceptability class (`unacc`, `acc`, `good`, `vgood`) based on price, maintenance costs, capacity, and safety features.
*   **Preprocessing**: Categorical columns were integer-encoded using `LabelEncoder`.
*   **Model Comparison**:

| Model Variant | Test Accuracy | Train Accuracy | Key Insights |
| :--- | :--- | :--- | :--- |
| **`CategoricalNB`** | **81.50%** | **86.61%** | Achieves high accuracy because it assumes a categorical feature distribution, matching the discrete nature of the car attributes. |
| **`GaussianNB`** | **62.43%** | **64.04%** | Fails significantly. It treats the encoded integers (0, 1, 2...) as continuous values following a bell curve, which is mathematically invalid for ordinal categories. |

---

### B. Real-world Dataset (Disease Prediction)
*   **Task**: Predict one of 41 diseases using 132 binary symptom indicators.
*   **Data Cleaning & Preprocessing**:
    *   **Unnamed Column Dropped**: Removed `Unnamed: 133` which was filled with `NaN` due to trailing commas in the CSV.
    *   **Duplicates Removed**: Dropped **4,616 exact duplicate rows** (out of 4,920 total rows) leaving **304 unique rows**. Keeping duplicates would cause **data leakage**, artificially inflating test performance to 100% since identical rows would end up in both splits.
    *   **Index Reset**: Renumbered remaining rows from `0` to `303`.
    *   **Stratified Split**: Split the data into 80% train and 20% test using `stratify=y_encoded` to ensure all 41 diseases are represented in both sets.
*   **Model Setup**: Trained using `BernoulliNB` because the features are binary flags.
*   **Performance Results**:
    *   **Training Accuracy**: 98.77%
    *   **Testing Accuracy**: 96.72%
    *   **Cross-Validation (5-Fold)**: Mean Accuracy: 90.96% (std: 3.3%)
        *   *Note*: The split threw a warning because a disease had only 4 unique samples, meaning it was impossible to put $\ge 1$ sample in all 5 validation folds. Setting `cv=4` resolves this warning.
    *   **Confusion Matrix Analysis**: 
        *   The model misclassified a single **Heart attack** patient (Class 18) as having **GERD** (Class 16). Because they share chest and gastric symptoms, the model predicted GERD.
        *   This left 0 predicted samples for "Heart attack" in the test set, triggering an `UndefinedMetricWarning` for precision (division by zero), which was solved by setting `zero_division=0` in the classification report.

---

## 9. When to Use / Avoid Naive Bayes

### Avoid Naive Bayes when:
| Don't use when... | Reason |
| :--- | :--- |
| **Features are highly correlated** | The conditional independence assumption breaks, leading to double-counting of signals. |
| **Need calibrated probabilities** | The predicted probabilities (`predict_proba()`) can be poorly calibrated (often pushed to extreme 0 or 1 values). |
| **Complex feature interactions matter** | The model cannot capture dependencies between features (e.g., "Feature A is only useful if Feature B is present"). |

---

## 10. Quick Revision Summary Table

| Property | Value |
| :--- | :--- |
| **Type** | Supervised |
| **Problem** | Classification (primarily) / Regression |
| **Parametric** | Yes (Estimates probability parameters) |
| **Lazy Learner** | No (Eager learner) |
| **Scaling Required** | No |
| **Training Speed** | Fast ($O(N \cdot D)$) |
| **Prediction Speed** | Fast ($O(C \cdot D)$) |
| **Sensitive to Outliers** | No |
| **Sensitive to Correlations**| Yes (Highly sensitive) |
| **Smoothing Factor** | $\alpha$ (Laplace Smoothing) |
| **Core Assumption** | Features are conditionally independent given the class |
