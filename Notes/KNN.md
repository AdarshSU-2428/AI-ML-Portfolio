# K-Nearest Neighbors (KNN) Study Notes

This document contains detailed study notes on K-Nearest Neighbors (KNN) based on the following implementations:
*   **Toy Dataset Implementation (Iris)**: [toy_dataset_implementation.ipynb](../Machine%20Learning/K%20Nearest%20Neighbour/toy_dataset_implementation.ipynb)
*   **Real-world Dataset Implementation (Titanic)**: [real_dataset_implementation.ipynb](../Machine%20Learning/K%20Nearest%20Neighbour/real_dataset_implementation.ipynb)

---

## 1. Intuition

Imagine you move to a new neighborhood and want to know which political party is most popular there. You don't have time to interview everyone, so you knock on the doors of the **5 closest neighbors** to your house and ask them. If 4 of them support Party A and 1 supports Party B, you assume your household is also likely in a Party A dominated zone.

This is the exact intuition of **K-Nearest Neighbors (KNN)**. It is a distance-based, non-parametric algorithm that classifies a new data point based on the majority class of its $K$ closest neighbors in the feature space.

---

## 2. Model Comparison Table

| Feature | Logistic Regression | Decision Trees | K-Nearest Neighbors (KNN) |
| :--- | :--- | :--- | :--- |
| **Model Type** | Parametric | Non-parametric | Non-parametric (Instance-based) |
| **Training Complexity**| $O(N \cdot D)$ (Iterative optimization) | $O(N \cdot D \log N)$ (Recursive splits)| $O(1)$ (No training phase, lazy learner) |
| **Testing/Inference** | $O(D)$ (Simple dot product) | $O(\text{Tree Depth})$ (Fast traversals) | $O(N \cdot D)$ (Slow, must compute distance to all points) |
| **Feature Scaling** | Recommended for faster convergence | **Not Required** (Splits based on threshold ordering) | **Strictly Required** (Highly sensitive to scale differences) |
| **Outliers** | Highly sensitive | Robust (Uses threshold splits) | Sensitive (Outliers alter local neighborhoods) |
| **Non-Linearity** | Requires manual feature engineering | Handles non-linearity natively | Handles complex non-linear boundaries natively |

---

## 3. How KNN Works (Step-by-Step Inference)

Unlike most machine learning models, KNN is a **lazy learner**, meaning it does not learn a general discriminative function during training. Instead, it stores the entire training dataset and postpones all computation until inference:

```text
Query Point (X_test)
     │
     ▼
[ Step 1: Calculate Distances ] ──► Compute distance (e.g. Euclidean) to all X_train points
     │
     ▼
[ Step 2: Sort and Filter ]     ──► Sort distances ascendingly and select top K neighbors
     │
     ▼
[ Step 3: Gather Labels ]       ──► Retrieve target labels of the chosen K neighbors
     │
     ▼
[ Step 4: Voting / Aggregation ]──► Take majority vote (classification) or mean (regression)
     │
     ▼
Final Prediction (y_pred)
```

---

## 4. Assumptions of KNN

While KNN is non-parametric (making no assumptions about the shape of the decision boundary), it operates under key structural assumptions:
*   **Distance-Similarity Correspondence**: It assumes that points that are close to each other in the feature space share similar target labels (local homogeneity).
*   **Irrelevant Feature Minimalist**: It assumes all features are relevant. Noisy or irrelevant features can corrupt the distance calculations (Curse of Dimensionality).
*   **Scale Uniformity**: It assumes all features have been scaled uniformly. Otherwise, features with larger numerical ranges will dominate distance metrics.

---

## 5. Advantages & Disadvantages

### Advantages
*   **Extremely Simple**: Easy to understand, explain, and implement.
*   **Zero Training Time**: Training consists of just saving the data, making online updates trivial.
*   **Adapts Natively**: Can handle multi-class classification and non-linear boundaries without modifications.
*   **No Parametric Assumptions**: Works well when the decision boundary is highly irregular.

### Disadvantages
*   **High Inference Cost**: Testing is very slow ($O(N \cdot D)$) and computationally expensive for large datasets.
*   **Memory Intensive**: Requires holding the entire training set in RAM during prediction.
*   **Curse of Dimensionality**: As the number of features increases, the high-dimensional volume grows exponentially, making all points seem equidistant.
*   **Highly Sensitive to Scaling and Outliers**: Outliers or poorly scaled features drastically bias predictions.

## 6. When to Use / Avoid KNN

### Use KNN when:
*   **Dataset is small to medium**: KNN works best when the number of samples is manageable, as there is no intensive training phase.
*   **Decision boundary is complex**: Being non-parametric, KNN can naturally map irregular and non-linear boundaries.
*   **Fast training is desired**: The training phase is $O(1)$, simply storing the dataset.
*   **Data is properly scaled**: Distance calculations are meaningful when all features are standardized.

### Avoid KNN when:
*   **Dataset is huge**: As data grows, memory usage and prediction latency ($O(N)$ per query) become prohibitive.
*   **Many irrelevant features**: Noisy features degrade the distance calculation accuracy.
*   **High-dimensional data**: The "Curse of Dimensionality" makes all points appear equidistant, causing KNN to lose its grouping power.
*   **Fast prediction is required**: Classification at inference time requires searching the entire dataset, which is relatively slow.

---

## 7. Hyperparameters (sklearn's `KNeighborsClassifier`)

*   `n_neighbors` ($K$): The number of neighbors to query.
    *   *Low $K$ (e.g. $K=1$)*: High variance/low bias. Prone to overfitting (captures noise and outliers).
    *   *High $K$ (e.g. $K=30$)*: Low variance/high bias. Leads to underfitting (boundary becomes too smooth, swallowing minority classes).
    *   *Note*: For binary classification, an odd value of $K$ is commonly chosen to reduce the chance of ties. However, odd $K$ is not mandatory, especially for multi-class problems where ties are less common.
*   `weights`:
    *   `'uniform'`: All neighbors in the local neighborhood get an equal vote.
    *   `'distance'`: Closer neighbors have a stronger influence on the prediction (weight is inversely proportional to distance).
    *   *Note*: When `weights='distance'`, closer neighbors influence the prediction more than farther neighbors, which can improve performance when nearby samples are more reliable.
*   `metric`: The distance metric used for the neighborhood search:
    *   `'euclidean'`: Standard straight-line distance.
    *   `'manhattan'`: Grid-based distance. Robust for high-dimensional or categorical data.
    *   `'minkowski'`: General distance formulation with parameter $p$.
*   `algorithm`: The method used to compute nearest neighbors (`'brute'`, `'kd_tree'`, `'ball_tree'`). Use `'auto'` in scikit-learn to let the package choose the most efficient algorithm based on data size and dimensions.

---

## 8. Mathematical Underpinnings

### A. Distance Metrics
The core of KNN is measuring the distance between two points $p$ and $q$ in a $D$-dimensional space:

*   **Euclidean Distance ($L_2$ norm)**:
    $$d(p, q) = \sqrt{\sum_{i=1}^{D} (p_i - q_i)^2}$$
*   **Manhattan Distance ($L_1$ norm)**:
    $$d(p, q) = \sum_{i=1}^{D} |p_i - q_i|$$
*   **Minkowski Distance**:
    $$d(p, q) = \left( \sum_{i=1}^{D} |p_i - q_i|^r \right)^{1/r}$$
    *   If $r = 1$: Manhattan Distance.
    *   If $r = 2$: Euclidean Distance.

### B. Optimal $K$ Selection (Bias-Variance Tradeoff)

```text
Decision Boundary Complexity
   ▲
   │    ░░░ (K = 1) High Variance (Overfitting - jagged boundaries)
   │   ░░░░░
   │  ░░░░░░░ (K = 5) Balanced Decision Boundary
   │ ░░░░░░░░░
   │░░░░░░░░░░ (K = 30) High Bias (Underfitting - overly smooth)
   └──────────────────────────────────► Number of Neighbors (K)
```

---

## 9. Why KNN Requires Feature Scaling

Because KNN calculates distances, features with larger numeric scales will completely dominate features with smaller scales.

### Numeric Demonstration
Consider classifying a passenger's survival using two features: `Age` (years) and `Fare` (dollars).
*   **Passenger A**: `Age = 20`, `Fare = $10`
*   **Passenger B**: `Age = 22`, `Fare = $10`
*   **Passenger C**: `Age = 20`, `Fare = $300`

Let's calculate the Euclidean distance between Passenger A and the others without scaling:
$$d(A, B) = \sqrt{(20-22)^2 + (10-10)^2} = \sqrt{4 + 0} = 2$$
$$d(A, C) = \sqrt{(20-20)^2 + (10-300)^2} = \sqrt{0 + 84100} = 290$$

Without scaling, Passenger B is considered extremely close to A, while C is considered incredibly distant. However, a \$290 difference in ticket fare might just reflect class/cabin difference, while a 2-year difference in age is minimal. Scaling transforms both features (e.g. using `StandardScaler` to have $\mu=0$ and $\sigma=1$), putting them on a level playing field so that both variables contribute equally to similarity measures.

---

## 10. Implementation Analysis & Performance Comparison

### A. Toy Dataset (Iris Flowers)
*   **Task**: Classify flower species into 3 categories (Setosa, Versicolor, Virginica) using sepal and petal features.
*   **Split**: 67% Training, 33% Testing (50 samples).
*   **Base Performance (K=5, scaled)**:
    *   **Training Accuracy**: 96.00%
    *   **Testing Accuracy**: 98.00%
    *   **Baseline 5-Fold Cross-Validation Accuracy**: 97.33% (std: 2.49%)
*   **Confusion Matrix**:
    $$\begin{bmatrix} 19 & 0 & 0 \\ 0 & 15 & 0 \\ 0 & 1 & 15 \end{bmatrix}$$
*   **Insight**: The Iris dataset has highly compact, well-separated clusters, allowing a simple baseline KNN to achieve near-perfect classification.

### B. Real-world Dataset (Titanic Survival)
*   **Task**: Predict whether a passenger survived based on demographics, ticket details, and cabin class.
*   **Preprocessing**:
    *   Irrelevant columns (`PassengerId`, `Name`, `Ticket`, `Cabin`) dropped.
    *   Missing `Age` imputed with the median (28.0). Missing `Embarked` rows dropped.
    *   Categorical columns (`Sex` and `Embarked`) encoded using `LabelEncoder`.
    *   Features scaled using `StandardScaler` to prevent fare dominance.
*   **Model Performance**:

| Model Setup | Test Accuracy | CV Accuracy | Hyperparameters | Key Insights |
| :--- | :--- | :--- | :--- | :--- |
| **Base KNN** | **79.78%** | — | $K=5$, Euclidean | Evaluated on 20% test split. Confusion matrix shows TN=90, TP=52, FP=19, FN=17. |
| **K-Fold Cross-Validation** | — | **80.09%** | $K=5$, Euclidean | 5-fold cross-validation on the scaled features set, with a standard deviation of **1.95%**. |

---

## 11. Feature Engineering Details (Titanic)

To optimize the KNN classifier's performance, the following feature engineering techniques were applied:

1.  **`FamilySize`**:
    $$\text{FamilySize} = \text{SibSp} + \text{Parch} + 1$$
    *   *Reasoning*: Combines sibling/spouse and parent/child counts to capture the total group size. Larger families on the Titanic often faced delays or stayed together, affecting survival.
2.  **`IsAlone`**:
    $$\text{IsAlone} = \begin{cases} 1 & \text{if FamilySize} = 1 \\ 0 & \text{otherwise} \end{cases}$$
    *   *Reasoning*: A binary indicator capturing whether a passenger traveled alone, representing a different risk profile.

---

## 12. Quick Revision Summary Table

| Property | Value |
| :--- | :--- |
| **Type** | Supervised |
| **Problem** | Classification / Regression |
| **Parametric** | No |
| **Lazy Learner** | Yes |
| **Scaling Required** | Yes |
| **Training Speed** | Very Fast ($O(1)$) |
| **Prediction Speed** | Slow ($O(N \cdot D)$) |
| **Sensitive to Outliers** | Yes |
| **Sensitive to Scaling** | Yes |
| **Main Hyperparameter** | `n_neighbors` |
