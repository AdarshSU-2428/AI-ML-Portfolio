# Clustering Study Notes

This document contains detailed study notes on Clustering based on the following implementations:
*   **Toy Dataset Implementation (Blobs/Moons)**: [toy_dataset_implementation.ipynb](../Machine%20Learning/Clustering/toy_dataset_implementation.ipynb)
*   **Real-world Dataset Implementation (Mall Customers)**: [Real_dataset_implementation.ipynb](../Machine%20Learning/Clustering/Real_dataset_implementation.ipynb)

---

## 1. Intuition

Imagine you own a large grocery store and want to organize thousands of items on shelves so that customers can find them easily. You don't have a pre-existing catalog of categories. Instead, you look at their features:
*   Items that are cold, need refrigeration, and come in cartons (milk, yogurt, cheese) are placed together.
*   Fresh, leafy, colorful raw goods (spinach, apples, carrots) are grouped in another section.
*   Dry, boxed items (cereal, oats, pasta) go on aisle 4.

You have just performed **Clustering**. 

Clustering is an **unsupervised learning** technique that automatically groups similar data points together. Unlike classification (where you know the categories beforehand), clustering discovers hidden patterns and natural groupings in unlabeled data.

---

## 2. Model Comparison Table

| Feature | K-Means | DBSCAN |
| :--- | :--- | :--- |
| **Clustering Type** | Partitioning (Centroid-based) | Density-based |
| **Number of Clusters ($K$)**| Must be specified in advance | Determined automatically by the algorithm |
| **Outlier Handling** | **Highly Sensitive**: Forces outliers into the nearest cluster (can distort cluster centers). | **Highly Robust**: Isolates outliers and flags them as noise (`-1`). |
| **Cluster Shape** | Spherical, convex, and isotropic (assumes equal variance). | Arbitrary shapes (lines, circles, spirals, moons). |
| **Feature Scaling** | **Strictly Required** (sensitive to coordinate scales). | **Strictly Required** (distance calculations define local neighborhoods). |
| **Time Complexity** | $O(T \cdot K \cdot N \cdot D)$ (Very fast and scales well). | $O(N^2)$ (Slow for large datasets). |
| **Density Variance** | Handles varying density clusters relatively well. | Fails to identify clusters of varying density. |

---

## 3. How the Algorithms Work (Step-by-Step)

### A. K-Means
K-Means groups data by minimizing the distance between data points and the center of their assigned cluster (centroid).

```text
Initialize Centroids (Choose K)
     │
     ▼
┌──────────────────────────────┐
│ Step 1: Assignment Phase     │ ──► Assign each point to the closest centroid
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│ Step 2: Update Phase         │ ──► Compute new centroids (mean of assigned points)
└──────────────────────────────┘
     │
     ▼
  Converged? (Centroids stop moving)
     ├──► NO  ──► Loop back to Step 1
     └──► YES ──► Final Clusters Assigned
```

### B. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
DBSCAN groups points that are close to each other in dense regions, leaving points in low-density regions as noise. It categorizes points into three types:
1.  **Core Point**: Has at least `min_samples` within a radius of `eps`.
2.  **Border Point**: Close to a core point (within `eps` radius) but has fewer than `min_samples` in its own neighborhood.
3.  **Noise Point**: Any point that is neither a core point nor a border point.

```text
     For each unvisited point:
               │
               ▼
[ Count neighbors within radius 'eps' ]
               │
               ▼
   Is neighbor count >= 'min_samples'?
     ├──► YES ──► Label as CORE POINT and expand cluster recursively
     └──► NO  ──► Label as NOISE/BORDER POINT
```

---

## 4. Key Assumptions & Constraints

### K-Means Assumptions:
*   **Spherical Clusters**: Assumes the groups are spherical. It struggles with elongated, crescent, or non-linear shapes (as demonstrated in the moons dataset).
*   **Equal Variance/Size**: Assumes clusters have similar variance and sizes.
*   **Complete Membership**: Assumes every point belongs to a cluster (no noise concept).

### DBSCAN Assumptions:
*   **Uniform Density**: Assumes clusters have similar densities. If clusters have varying densities, a single `eps` value cannot separate them effectively.
*   **Parameters Sensitivity**: Assumes `eps` and `min_samples` can be tuned correctly. A small change in `eps` can merge separate clusters or turn everything into noise.

---

## 5. Hyperparameter Tuning

### K-Means: Finding the Optimal $K$
1.  **Elbow Method**: Plot the Within-Cluster Sum of Squares (WCSS / Inertia) for a range of $K$. Look for the "elbow" point where the drop in WCSS slows down significantly.
2.  **Silhouette Score**: Measures how close a point is to its own cluster compared to other clusters. It ranges from $-1$ (bad) to $+1$ (good). Choose $K$ with the highest silhouette score.

### DBSCAN: Tuning `eps` and `min_samples`
1.  **Setting `min_samples`**: A standard heuristic is $2 \times \text{Dimensions}$. For 3D data, set `min_samples = 6`.
2.  **Finding `eps` using K-Distance Graph**:
    *   Compute the distance to the $k$-th nearest neighbor (where $k = \text{min\_samples} - 1$) for each point.
    *   Sort the distances in ascending order and plot them.
    *   The point of maximum curvature (the "elbow") on the plot represents the optimal `eps` value.

---

## 6. Advantages & Disadvantages

### K-Means
*   **Advantages**:
    *   Very fast and computationally efficient ($O(N)$ complexity).
    *   Easy to understand and implement.
    *   Guarantees convergence.
*   **Disadvantages**:
    *   You must specify $K$ manually.
    *   Extremely sensitive to outliers and initialization (can get stuck in local minima).
    *   Fails on non-linear or complex shapes.

### DBSCAN
*   **Advantages**:
    *   Does not require specifying the number of clusters in advance.
    *   Robust to outliers (isolates noise).
    *   Can find clusters of arbitrary shapes (e.g., spirals, concentric rings).
*   **Disadvantages**:
    *   Cannot cluster datasets with varying densities.
    *   Struggles with high-dimensional data (Curse of Dimensionality affects distance).
    *   Sensitive to parameter selection.

---

## 7. Real-world Customer Segmentation Insights

When clustering the **Mall Customers** dataset (using `Age`, `Annual Income`, and `Spending Score`):
1.  **Feature Standardization is Mandatory**: Since `Age` ranges from 18 to 70 and `Annual Income` goes up to 137, standardizing the features using `StandardScaler` ensures that the distance metrics are computed fairly without income dominating age.
2.  **VIP Segment**: Identified as high-income, high-spending customers. The store should target them with luxury brand launches and exclusive invites.
3.  **Frugal Segment**: High-income but low-spending. They need bulk deals, utility-focused products, and value-oriented ads.
4.  **Age Division**: High spending scores are strongly grouped under the age of 40, meaning younger consumers are the store's primary target for trending/impulsive purchases.
