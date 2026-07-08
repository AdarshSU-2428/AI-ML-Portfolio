# Decision Tree Study Notes

This document contains detailed study notes on Decision Trees based on the following implementations:
*   **Toy Dataset Implementation (Play Tennis)**: [toy_dataset_implementation.ipynb](../Machine%20Learning/Decision%20Tree/toy_dataset_implementation.ipynb)
*   **Real-world Dataset Implementation (Loan Prediction)**: [Real_dataset_implementation.ipynb](../Machine%20Learning/Decision%20Tree/Real_dataset_implementation.ipynb)

---

## 1. Intuition

Suppose you are deciding whether to play tennis today.
You look outside: 
1. Is it overcast? If yes, you play.
2. Is it sunny? If yes, you check the humidity. If humidity is high, you don't play.
3. Is it raining? If yes, you check the wind. If windy, you don't play.

This series of nested questions forms a flowchart. In machine learning, a **Decision Tree** automatically learns these question boundaries from data, splitting the dataset step-by-step to arrive at a final classification or regression value.

---

## 2. Decision Tree vs. Linear Models Comparison Table

| Feature | Linear / Logistic Regression | Decision Trees |
| :--- | :--- | :--- |
| **Model Type** | Parametric (assumes a functional shape) | Non-parametric (makes no assumptions about shape) |
| **Data Scaling** | **Required** (highly sensitive to feature scales) | **Not Required** (scaling has zero effect on split order) |
| **Non-Linearity** | Requires manual feature transformations | Handles complex non-linear relationships natively |
| **Multicollinearity** | **Sensitive** (inflates coefficient variances) | **Robust** (redundant features are simply ignored) |
| **Outliers** | Highly sensitive (outliers pull fitting lines/planes) | Robust (splits are based on ordering, not magnitudes) |
| **Interpretability** | High (using weights/coefficients) | High for small trees (visual path); Low for very deep trees |

### Why Decision Trees Do Not Require Feature Scaling
Decision Trees compare feature values using relational thresholds (like $\le$ or $>$).
* **Example**: `Age <= 30` or `Income <= 50000`.
* Scaling transformations (like normalization or standardization) change the raw numeric values but **strictly preserve their relative ordering**.
  * **Original feature order**: $20 < 40 < 60$
  * **Scaled feature order**: $-1.2 < 0.3 < 1.5$
* Since the relative sorting/ordering is completely unchanged, the candidate thresholds evaluated by the splitting algorithm yield the exact same split divisions. Thus, scaling is completely redundant for trees.

---

## 3. Interpreting a Decision Tree Diagram

Many developers can train a tree but struggle to interpret its visual diagram. Reading a tree follows a top-down flow:

```text
      [ Root Node ]          <-- Most important feature split (highest impurity reduction)
      /           \
   True          False       <-- Decisions based on comparison (left is True, right is False)
    /               \
[ Decision Node ] [ Leaf Node ] <-- Leaves represent the terminal prediction (no more splits)
```

* **Parent vs. Child Nodes**: Each split yields two child nodes.
* **Colors & Purity**: In visualization plots, nodes are shaded according to class purity. A pure node has a deep single color, while an impure/mixed node is lighter or white.
* **Information in Nodes**:
  * **Split Rule** (e.g. `Credit_History <= 0.5`)
  * **Impurity Metric** (e.g. `gini = 0.423`)
  * **Samples**: The number of training instances in that node (e.g. `samples = 491`).
  * **Value**: The breakdown of samples per target class (e.g. `value = [149, 342]`).
  * **Class**: The majority class predicted by this node (e.g. `class = Yes`).

---

## 4. Assumptions of Decision Trees

Unlike parametric models, **Decision Trees make no assumptions about the underlying distribution of data.**
*   **No Linearity Assumption**: Features do not need a linear relationship with the target.
*   **No Normality Assumption**: Residuals/errors do not need to follow a normal distribution.
*   **No Homoscedasticity Assumption**: Error variance does not need to be constant.
*   **No Multicollinearity Concerns**: Highly correlated features do not bias the split search.

---

## 5. Advantages & Disadvantages

### Advantages
*   **Simple to Understand and Visualize**: Can output an intuitive tree diagram mapping the exact logic of predictions.
*   **Minimal Preprocessing**: No feature scaling (standardization/normalization) or centering required.
*   **Handles Mixed Data Types**: Natively accommodates both numerical and categorical data.
*   **Implicit Feature Selection**: The most important features are naturally placed closer to the root of the tree.

### Disadvantages
*   **Highly Prone to Overfitting**: Without limits, a tree will grow complex enough to memorize noise and outliers (yielding 100% train accuracy but poor test accuracy).
*   **Unstable (High Variance)**: A tiny change in the training data can result in a completely different set of splits.
*   **Greedy Approach**: Splits are chosen to maximize immediate gain at each node (local optimization), which does not guarantee the globally optimal tree.

---

## 6. When should I use this? (Use Cases)

*   **Credit Scoring & Loan Eligibility**: Classifying applicants as high-risk or low-risk based on credit history, income, and debt.
*   **Medical Diagnosis**: Mapping patient symptoms through a logical sequence of tests to predict a disease.
*   **Customer Churn Analysis**: Identifying high-probability churn segments by dividing customers based on contract type, usage, and age.

---

## 7. Hyperparameters (sklearn's `DecisionTreeClassifier`)

Pruning hyperparameters are used to control the tree's size and prevent overfitting:

*   `criterion`: The function to measure split quality. Supported values are `'gini'` (Gini impurity) and `'entropy'` (information gain).
*   `max_depth`: The maximum vertical levels allowed.
    *   *Too deep*: Leads to high variance (overfitting).
    *   *Too shallow*: Leads to high bias (underfitting).
*   `max_leaf_nodes`: Limits the total number of terminal leaves in best-first fashion. Often the most effective way to prune.
*   `min_samples_leaf`: The minimum samples required to be at a terminal leaf node. Prevents the tree from isolating outliers.
*   `min_samples_split`: The minimum samples required in a node to split it further.

---

## 8. Mathematical Underpinnings of Decision Trees

Decision trees recursively partition the feature space. At each step, they choose the split that maximizes the **purity** of the resulting child nodes.

### A. Gini Impurity (default)
Gini Impurity measures the likelihood of a random sample being incorrectly labeled if it were randomly classified according to the distribution of labels in the subset.
For a node $m$ with classes $i \in \{1, 2, \dots, C\}$:
$$G(m) = 1 - \sum_{i=1}^{C} (p_i)^2$$
Where $p_i$ is the proportion of samples belonging to class $i$ in node $m$.

*   **Pure node**: $G(m) = 0$
*   **Max Impurity (binary)**: $G(m) = 0.5$

### B. Entropy
Entropy measures the degree of disorder or uncertainty in the node:
$$H(m) = - \sum_{i=1}^{C} p_i \log_2(p_i)$$

### C. Information Gain
When splitting a parent node $P$ into left child $L$ and right child $R$, the quality of the split is defined by the reduction in impurity:
$$IG = I(P) - \left( \frac{N_L}{N_P} I(L) + \frac{N_R}{N_P} I(R) \right)$$
Where:
*   $I(\cdot)$ is the impurity measure (Gini or Entropy).
*   $N_P, N_L, N_R$ are the number of samples in the parent, left child, and right child nodes, respectively.

### D. Splitting Continuous Features
For a continuous feature $X_j$:
1.  Sort the unique values of $X_j$ present in the node: $x^{(1)} < x^{(2)} < \dots < x^{(k)}$.
2.  Calculate midpoints between adjacent sorted values: $t_i = \frac{x^{(i)} + x^{(i+1)}}{2}$.
3.  Evaluate the Information Gain ($IG$) for each threshold split: $X_j \le t_i$.
4.  Choose the feature $X_j$ and threshold $t_i$ that maximizes the Information Gain.

---

## 9. Overfitting Example (Play Tennis vs. Loan Prediction)

Decision Trees are highly sensitive to overfitting because they naturally keep splitting until every leaf contains only pure instances.

### Play Tennis Dataset
When training an unconstrained tree on all **14 rows** of the Play Tennis dataset:
* **Training Accuracy = 100%**: The model created rules to perfectly separate every single day.
* **Overfitting**: The model memorizes noise (e.g., matching windiness to a specific day rather than a general rule).

### Loan Prediction Dataset
When training an unconstrained tree on the Loan Prediction dataset:
* **Training Accuracy = 100%**
* **Test Accuracy = 68.29%** (unconstrained baseline tree)
* **The Solution (Regularization)**: By applying pruning parameters like `max_depth = 4` and `max_leaf_nodes = 6`, we regularize the tree structure:
  * **Training Accuracy: ~84%**
  * **Test Accuracy: 81.30%** (improved generalization on new data by over **13%**).

---

## 10. Feature Engineering (Loan Prediction)

Adding custom features based on domain knowledge can significantly improve decision boundaries. For the Loan Prediction dataset, we engineered the following variables:

1. **`TotalIncome`**:
   $$\text{TotalIncome} = \text{ApplicantIncome} + \text{CoapplicantIncome}$$
   * **Reasoning**: Banks assess loan eligibility based on the combined household income rather than just the primary applicant's income.

2. **`LoanToIncomeRatio`**:
   $$\text{LoanToIncomeRatio} = \frac{\text{LoanAmount} \times 1000}{\text{TotalIncome}}$$
   * **Reasoning**: Measures the affordability of the requested loan. Scaling the loan amount by 1000 aligns both numerator and denominator to actual dollars.

3. **`IncomePerDependent`**:
   $$\text{IncomePerDependent} = \frac{\text{TotalIncome}}{\text{Dependents} + 1}$$
   * **Reasoning**: Represents the household's remaining financial capacity. A high total income with 4 dependents leaves less disposable income than the same income with 0 dependents.

---

## 11. Quick Revision Summary Table

| Property | Value |
| :--- | :--- |
| **Type** | Supervised |
| **Problem** | Classification / Regression |
| **Parametric** | No |
| **Lazy Learner** | No (Eager learner) |
| **Scaling Required** | No |
| **Training Speed** | Medium-Slow ($O(N \cdot D \log N)$) |
| **Prediction Speed** | Very Fast ($O(\text{Tree Depth})$) |
| **Sensitive to Outliers** | No (Robust) |
| **Sensitive to Scaling** | No |
| **Main Hyperparameter** | `max_depth` / `min_samples_leaf` / `max_leaf_nodes` |
