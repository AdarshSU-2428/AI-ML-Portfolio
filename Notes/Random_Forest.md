# Random Forest Study Notes

This document contains detailed study notes on Random Forests based on the following implementations:
*   **Toy Dataset Implementation (Play Tennis)**: [toy_dataset_implementation.ipynb](../Machine%20Learning/Random%20Forest/toy_dataset_implementation.ipynb)
*   **Real-world Dataset Implementation (Loan Prediction)**: [Real_dataset_implementation.ipynb](../Machine%20Learning/Random%20Forest/Real_dataset_implementation.ipynb)

---

## 1. Intuition

Imagine you need to decide which stock to invest in. 
1. If you ask a single financial advisor, they might give you an opinion based on their personal bias (high variance, overfitting to specific past experiences).
2. Instead, you ask a committee of **100 diverse advisors**. 
3. To make sure they don't influence each other or share the exact same dataset:
   - You give each advisor a slightly different historical report (bootstrap sampling).
   - At each step of their research, you restrict them to looking at a random subset of market indicators (feature bagging).
4. Each advisor builds their own investment checklist (decision tree) independently.
5. Finally, you take a vote among all 100 advisors. The majority decision is your final investment.

This "wisdom of the crowd" ensemble method is a **Random Forest**. By combining many independent, slightly noisy decision trees, the errors of individual trees cancel out, producing a highly robust and accurate model.

---

## 2. Model Comparison Table

| Feature | Linear / Logistic Regression | Decision Trees | Random Forests |
| :--- | :--- | :--- | :--- |
| **Model Type** | Parametric | Non-parametric | Ensemble (Non-parametric) |
| **Data Scaling** | **Required** | **Not Required** | **Not Required** |
| **Non-Linearity** | Requires manual feature engineering | Handles non-linearity natively | Handles complex non-linearity and interactions natively |
| **Multicollinearity** | **Sensitive** (inflates coefficient variances) | **Robust** (redundant features ignored) | **Highly Robust** (reduces variance, feature bagging spreads splits) |
| **Outliers** | Highly sensitive | Robust | Extremely robust (ensemble averages out individual errors) |
| **Variance / Bias** | Low variance, High bias | High variance, Low bias | Lower variance than Decision Tree, similar or slightly higher bias, usually better overall generalization |
| **Interpretability** | High (coefficients) | High for small trees (visual path) | Low (black-box model consisting of hundreds of trees) |

### Why Random Forests Do Not Require Feature Scaling
Since the base estimators in a Random Forest are Decision Trees, Random Forests inherit their scale-invariant properties. 
* Trees partition data based on ordinal sorting (e.g., `Income <= 50000`).
* Any monotonic transformation (scaling, standardization, normalization) preserves the relative order of data points.
* Thus, the split points remain functionally identical, making scaling completely redundant.

---

## 3. Interpreting an Ensemble Diagram

A Random Forest outputs predictions by aggregating the independent predictions of its constituent trees:

```text
               [ Input Data Row ]
           /           |            \
     [ Tree 1 ]    [ Tree 2 ] ... [ Tree N ]   <-- Trained on bootstrap samples & random features
         |             |              |
      (Class A)     (Class B)      (Class A)   <-- Individual predictions
           \           |            /
            \          |           /
             [ Majority Voting ]               <-- Aggregation layer
                       |
                [ Final Class A ]              <-- Robust ensemble prediction
```

* **Bootstrap Aggregating (Bagging)**: Each tree is trained on a distinct subset of the data sampled with replacement.
* **Aggregating Predictions**: 
  * **Classification**: Majority vote determines the final class.
  * **Regression**: The average predicted numerical value of all trees is taken.

---

## 4. Assumptions of Random Forests

Random Forests are a non-parametric ensemble method and **make no assumptions about the data distribution**:
* **No Distributional Assumptions**: No normality, homoscedasticity, or linearity requirements.
* **No Independence of Features**: Natively handles collinear features. Feature bagging ensures that even correlated variables are distributed across different trees.
* **No Input Scaling Requirements**: Works directly on raw scale differences.

---

## 5. Advantages & Disadvantages

### Advantages
* **Much More Resistant to Overfitting than a Single Decision Tree**: By averaging multiple trees, the ensemble variance is significantly lower. However, it can still overfit if the trees are extremely deep, the data is very noisy, or hyperparameters are poorly tuned.
* **High Predictive Accuracy**: Frequently outperforms single estimators on structured/tabular datasets.
* **Handles High-Dimensional Data**: Excellent at processing thousands of input features due to random feature selection at each node.
* **Out-of-Bag (OOB) Evaluation**: Natively estimates test error during training without needing a separate validation set.
* **Measures Feature Importance**: Automatically scores which features contribute most to the split decisions.

### Disadvantages
* **Black-Box Model**: Hard to interpret. You cannot easily follow the exact path of a prediction across 100+ separate trees.
* **High Memory & Computation**: Training and storing hundreds of trees requires more memory and CPU cycles than simpler models.
* **Slow Real-Time Inference**: Calculating predictions requires passing the input through every single tree in the forest, which can limit latency-sensitive deployments.

---

## 6. When should I use this? (Use Cases)

* **Fraud Detection**: Inspecting bank transactions where features are highly non-linear and interactions are complex.
* **Customer Churn / Customer Lifetime Value (CLV)**: Predicting churn and spending habits based on historical behavior, demographics, and support interactions.
* **Bioinformatics**: Classifying gene expressions where the number of features ($d$) greatly exceeds the number of samples ($N$).

---

## 7. Hyperparameters (sklearn's `RandomForestClassifier`)

Hyperparameters control either the forest's ensemble structure or the shape of individual trees:

### Forest-Specific Hyperparameters
* `n_estimators`: The number of trees in the forest (default is 100). Higher numbers improve stability and reduce variance but increase computation time.
* `max_features`: The size of the random subset of features to consider when looking for the best split:
  * `'sqrt'` (default): $\sqrt{\text{total features}}$.
  * `'log2'`: $\log_2(\text{total features})$.
  * `None`: Considers all features (equivalent to standard bagging).
* `bootstrap`: Whether bootstrap samples are used when building trees (default is `True`).
* `oob_score`: Whether to use Out-of-Bag samples to estimate generalization accuracy (default is `False`).

### Tree-Pruning Hyperparameters (inherited from DecisionTree)
* `max_depth`: Limits the depth of individual trees.
* `max_leaf_nodes`: Limits the maximum number of leaf nodes in individual trees.
* `min_samples_split` / `min_samples_leaf`: Limits tree growth to prevent trees from isolating noise.

---

## 8. Mathematical Underpinnings of Random Forest

### A. Bootstrap Aggregating (Bagging)
Given a training set $D$ of size $N$, bagging generates $T$ new datasets $D_t$, each of size $N$, by sampling from $D$ uniformly and **with replacement**.
* The probability of a specific sample *not* being selected in a bootstrap sample of size $N$ is:
  $$\left(1 - \frac{1}{N}\right)^N \approx e^{-1} \approx 0.368$$
* This means approximately **36.8%** of the training samples are left out of each tree. These are called **Out-of-Bag (OOB)** samples.

### B. Random Feature Selection (Feature Bagging)
When splitting a node in a tree, instead of searching across all $d$ features, the algorithm randomly selects a subset of features $m \ll d$ (typically $m = \sqrt{d}$). The split search is restricted *only* to these $m$ features.
* **Goal**: This decorrelates the trees. If one feature is a dominant predictor, standard bagging would place it at the root of almost every tree. Feature bagging forces some trees to split on alternative features, exposing different patterns.

### C. Out-of-Bag (OOB) Error
Since OOB samples were not used in training a particular tree, they act as an implicit validation set:
$$\text{OOB Error} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(y_i \neq \hat{y}_i^{OOB}\right)$$
Where $\hat{y}_i^{OOB}$ is the majority vote prediction for sample $i$ using only the trees that did *not* contain sample $i$ in their bootstrap training set.

### D. Feature Importance (Mean Decrease in Impurity - MDI)
Feature importance measures how much a feature's split reduces impurity (Gini or Entropy) across all nodes in all trees:
$$\text{Importance}(X_j) = \frac{1}{T} \sum_{t=1}^T \sum_{n \in \text{splits on } X_j} \frac{N_n}{N_{\text{total}}} \Delta I(n)$$
Where $N_n$ is the number of samples reaching node $n$, and $\Delta I(n)$ is the decrease in Gini or Entropy impurity resulting from the split at node $n$.

---

## 9. Overfitting & Generalization Comparison

### Play Tennis Dataset
* **Decision Tree (Unconstrained)**: Perfect 100% train accuracy but highly overfit. A single noisy day can alter the entire tree.
* **Random Forest**: On the 14-row dataset, Random Forest also achieved perfect training accuracy, but this dataset is too small to meaningfully compare generalization between Decision Tree and Random Forest. The toy dataset was used primarily to understand the algorithm rather than evaluate performance.

### Loan Prediction Dataset
* **Decision Tree (Unconstrained)**:
  * Train Accuracy: **100%**
  * Test Accuracy: **68.29%** (severe overfitting)
* **Random Forest Classifier**:
  * Evaluated on the test set, the Random Forest Classifier achieved **81.30%** test accuracy.
  * More importantly, it achieved a high precision of **92%** for negative predictions (`Loan_Status` = No) and a recall of **97%** for positive predictions (`Loan_Status` = Yes), showing much better generalization compared to a single unconstrained tree.

---

## 10. Feature Engineering (Loan Prediction)

To optimize the Random Forest split search, we engineered three domain-specific financial ratios:

1. **`TotalIncome`**:
   $$\text{TotalIncome} = \text{ApplicantIncome} + \text{CoapplicantIncome}$$
   * **Purpose**: Reflects the actual purchasing power of the household.

2. **`LoanToIncomeRatio`**:
   $$\text{LoanToIncomeRatio} = \frac{\text{LoanAmount} \times 1000}{\text{TotalIncome}}$$
   * **Purpose**: Standardizes the size of the loan relative to combined earnings.

3. **`IncomePerDependent`**:
   $$\text{IncomePerDependent} = \frac{\text{TotalIncome}}{\text{Dependents} + 1}$$
   * **Purpose**: Adjusts the available household income based on family size.

---

## 11. Quick Revision Summary Table

| Property | Value |
| :--- | :--- |
| **Type** | Supervised |
| **Problem** | Classification / Regression |
| **Parametric** | No |
| **Lazy Learner** | No (Eager learner) |
| **Scaling Required** | No |
| **Training Speed** | Medium (Slower than single tree, but highly parallelizable) |
| **Prediction Speed** | Medium-Slow ($O(\text{n\_estimators} \times \text{Tree Depth})$) |
| **Sensitive to Outliers** | No (Extremely robust) |
| **Sensitive to Scaling** | No |
| **Main Hyperparameter** | `n_estimators` / `max_features` / `max_depth` |
