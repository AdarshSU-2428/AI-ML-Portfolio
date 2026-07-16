# Dimensionality Reduction Study Notes

This document contains detailed study notes on Dimensionality Reduction based on the following implementations:
*   **Toy Dataset Implementation (Blobs/Moons)**: [toy_dataset_implementation.ipynb](../Machine%20Learning/Dimensionality%20Reduction/toy_dataset_implementation.ipynb)
*   **Real-world Dataset Implementation (Mall Customers)**: [real_dataset_implementation.ipynb](../Machine%20Learning/Dimensionality%20Reduction/real_dataset_implementation.ipynb)

---

## 1. Intuition

Imagine you are holding a **3D globe** of the Earth. If you shine a flashlight on it, it casts a flat **2D shadow** on the wall. 
*   Depending on the angle of the light, the shadow will show the shapes of the continents. 
*   You have reduced the dimensionality of the globe from **3D** to a **2D shadow**, but you can still recognize the shapes of the continents.
*   However, some information is lost—for example, countries on opposite sides of the globe might overlap on the shadow.

This is **Dimensionality Reduction**: the process of reducing the number of input variables (features) in a dataset while keeping as much of the original structure and information as possible.

---

## 2. Model Comparison Table

| Feature | PCA (Principal Component Analysis) | t-SNE (t-Distributed Stochastic Neighbor Embedding) |
| :--- | :--- | :--- |
| **Type** | Linear | Non-linear |
| **Primary Goal** | Maximize variance (preserves global structure) | Preserve local similarities (preserves neighborhoods) |
| **Output Coordinates** | Deterministic (always gives the same output) | Stochastic (uses random initialization, output varies slightly per run) |
| **Distance Meaning** | Actual distance between points represents variance. | Cluster grouping is meaningful, but distances between clusters are arbitrary. |
| **Crowding Problem** | N/A (linear projection). | Solved using a Student-t distribution in the lower-dimensional space. |
| **Tuning Parameters** | None (only select `n_components`). | `perplexity` (defines the balance between local and global data views). |
| **Scaling Required** | **Strictly Required** (sensitive to variance scales). | **Strictly Required** (sensitive to coordinate distances). |
| **Speed** | Very fast and computationally cheap. | Slower, computationally expensive, especially on large datasets. |

---

## 3. How the Algorithms Work (Step-by-Step)

### A. PCA (Principal Component Analysis)
PCA finds the orthogonal directions of maximum variance (Principal Components) and projects the data onto them.

```text
       Standardize Features (Mean=0, Variance=1)
                        │
                        ▼
          Compute the Covariance Matrix
                        │
                        ▼
    Calculate Eigenvalues & Eigenvectors of the Matrix
                        │
                        ▼
   Sort Eigenvectors by Eigenvalues (Highest Variance First)
                        │
                        ▼
Select Top K Eigenvectors (PCs) & Project Data into Lower-D Space
```

### B. t-SNE (t-Distributed Stochastic Neighbor Embedding)
t-SNE maps high-dimensional points to low-dimensional points such that similar neighbors remain close.

```text
    Compute similarity probabilities in High-D (Gaussian)
                        │
                        ▼
   Initialize Low-D points randomly and compute similarities (Student-t)
                        │
                        ▼
  Minimize divergence (KL Divergence) between the two probability distributions
                        │
                        ▼
   Update Low-D coordinates using Gradient Descent until Convergence
```

---

## 4. Key Terms & Concepts Explained

### Curse of Dimensionality
As the number of features (dimensions) grows, the volume of the space increases exponentially, making the data points extremely sparse. In high-dimensional space, all points look almost equally far apart, which confuses clustering and distance-based algorithms. Dimensionality reduction helps overcome this.

### Global vs. Local Structure
*   **Global Structure:** Preserves the overall layout and macro-distances of the entire dataset. (E.g., Cluster A is twice as far from Cluster B as it is from Cluster C). **PCA excels at this.**
*   **Local Structure:** Preserves the micro-distances between close neighbors, ignoring the absolute positions of distant clusters. (E.g., make sure points inside Cluster A stay close to each other). **t-SNE excels at this.**

### Explained Variance Ratio
In PCA, this represents the proportion of the dataset's total variance that lies along each principal component. Looking at the cumulative explained variance helps you decide how many components (dimensions) to keep.

### Perplexity
A key hyperparameter in t-SNE that determines how to balance the focus between local and global aspects of your data. It can be thought of as a target number of nearest neighbors for each point. Typical values range from 5 to 50.

### The Crowding Problem
When we map high-dimensional data into a lower-dimensional space (like 2D), the volume of the space decreases drastically. Neighbors in high dimensions end up crowded together on top of each other. t-SNE solves this by using a **Student-t distribution** in 2D (which has much heavier tails than a Gaussian distribution), allowing clusters to spread out and separate visually.

---

## 5. Key Assumptions & Constraints

### PCA Assumptions & Constraints:
*   **Linearity**: Assumes relationships between features are linear. It will fail to unfold complex curved structures (like moons or spirals).
*   **Normal Distribution**: Performs best when features are symmetric and normally distributed.
*   **Outlier Sensitivity**: Outliers can heavily skew the variance and distort the principal components.

### t-SNE Assumptions & Constraints:
*   **No Global Interpretation**: Distances between separate clusters on a t-SNE plot do not represent real physical distances.
*   **Stochastic Nature**: Since it uses random initialization, running t-SNE twice will result in slightly different visual shapes, unless a `random_state` is fixed.
*   **No Transferability**: You cannot "fit" t-SNE on a training set and "transform" new test data later; it must reduce all points at the same time.

---

## 6. Summary of Notebook Implementations

### A. Toy Dataset: Blobs & Moons ([toy_dataset_implementation.ipynb](../Machine%20Learning/Dimensionality%20Reduction/toy_dataset_implementation.ipynb))
- **5D Blobs**: We generated a 5-dimensional dataset with 4 spherical clusters and projected them to 2D and 3D. 
  - *PCA* successfully captured the global variance, making the 4 distinct clouds clearly visible.
  - *t-SNE* pulled points into extremely dense, separated clusters based on neighbor proximity.
- **2D Moons**: We generated interlocking crescent shapes.
  - *PCA* failed to separate the crescents because it is a linear projection (it only rotated the moons).
  - *t-SNE* successfully separated the two interlocking curves into two distinct, isolated clusters.

### B. Real-world Dataset: Mall Customers ([real_dataset_implementation.ipynb](../Machine%20Learning/Dimensionality%20Reduction/real_dataset_implementation.ipynb))
- **EDA & Preprocessing**: We plotted pairwise relationships (pairplots), checked feature skewness (histograms), and correlation heatmaps. We dropped `CustomerID` and mapped categorical `Gender` to numerical values (`Male` -> 0, `Female` -> 1).
- **PCA vs. t-SNE Projection**: We standardized the numeric features and ran both reductions side-by-side.
  - *PCA* created a continuous gradient, where segments overlapped because of linear constraints.
  - *t-SNE* clearly resolved **5 customer clusters** representing distinct market personas: *Spendthrifts* (high income/high spend), *Careful* (high income/low spend), *Sensible* (average income/spend), *Careless* (low income/high spend), and *Thrifty* (low income/low spend).

