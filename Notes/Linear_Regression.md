# Linear Regression Study Notes

This document contains detailed study notes on Linear Regression based on the following implementations:
*   **From-Scratch Implementation**: [Scratch_linReg.py](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Linear Regression/Scratch_linReg.py)
*   **Toy Dataset Implementation (Diabetes)**: [toy_dataset_implmentation.ipynb](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Linear Regression/toy_dataset_implmentation.ipynb)
*   **Real-world Dataset Implementation (California Housing)**: [Real_dataset_implementation.ipynb](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Linear Regression/Real_dataset_implementation.ipynb)

---

## 1. Mathematical Underpinnings of Linear Regression

Linear Regression models the relationship between a dependent target variable $y$ and one or more independent feature variables $X$.

### The Model
For a single sample with $p$ features, the prediction $\hat{y}$ is represented as:
$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_p x_p + b = \mathbf{x}^T \mathbf{w} + b$$

For a dataset of $N$ samples, this is expressed in vectorized form:
$$\mathbf{\hat{y}} = \mathbf{X}\mathbf{w} + b\mathbf{1}$$
Where:
*   $\mathbf{X}$ is an $N \times p$ matrix of features.
*   $\mathbf{w}$ is a $p \times 1$ vector of weights.
*   $b$ is the bias (intercept) scalar.
*   $\mathbf{\hat{y}}$ is the $N \times 1$ vector of predictions.

### Objective Function: Mean Squared Error (MSE)
The goal is to find the weights $\mathbf{w}$ and bias $b$ that minimize the Mean Squared Error (MSE) loss function:
$$L(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 = \frac{1}{N} \|\mathbf{X}\mathbf{w} + b - \mathbf{y}\|^2_2$$

### Optimization via Gradient Descent
To minimize the loss function iteratively, we calculate the partial derivatives of $L$ with respect to $\mathbf{w}$ and $b$:

1.  **Gradient with respect to Weights ($\mathbf{w}$)**:
    $$\frac{\partial L}{\partial \mathbf{w}} = \frac{2}{N} \mathbf{X}^T (\mathbf{\hat{y}} - \mathbf{y})$$
2.  **Gradient with respect to Bias ($b$)**:
    $$\frac{\partial L}{\partial b} = \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)$$

The parameter update steps with learning rate $\eta$ are:
$$\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{\partial L}{\partial \mathbf{w}}$$
$$b \leftarrow b - \eta \frac{\partial L}{\partial b}$$

---

## 2. From-Scratch Implementation (`Scratch_linReg.py`)

The scratch script [Scratch_linReg.py](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Linear Regression/Scratch_linReg.py) demonstrates gradient descent optimization from first principles using `numpy`.

### Code Structure
*   **[LinearRegression](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Linear Regression/Scratch_linReg.py#L3)**: Class representing the linear regression model.
    *   **[__init__](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Linear Regression/Scratch_linReg.py#L4)**: Sets `learning_rate` (default `0.01`) and `n_iter` (default `1000`).
    *   **[fit](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Linear Regression/Scratch_linReg.py#L10)**: Initializes weights $\mathbf{w}$ to a zero vector of shape `(n_features,)` and bias $b$ to `0`. It runs gradient descent for `n_iter` iterations.
    *   **[predict](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Linear Regression/Scratch_linReg.py#L25)**: Computes predictions using the linear equation.
*   **[mse](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Linear Regression/Scratch_linReg.py#L29)**: Standard metric function to evaluate mean squared error.

### Vectorized Update Logic in `fit`
```python
# Compute prediction
y_pred = np.dot(X, self.weights) + self.bias

# Compute gradients
dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
db = (2 / n_samples) * np.sum(y_pred - y)

# Update parameters
self.weights = self.weights - self.learning_rate * dw
self.bias = self.bias - self.learning_rate * db
```

### Synthetic Experiment Details
*   **Data Generation**: Generates $100$ samples of 1D data where $y = 4 + 3x + \text{Gaussian Noise}$.
*   **Train-Test Split**: Sequential split (first 80% for training, remaining 20% for testing).
*   **Results**:
    *   **Learned Weight**: $\approx 3.01$ (matches true coefficient $3$)
    *   **Learned Bias**: $\approx 3.92$ (matches true intercept $4$)
    *   **Test MSE**: $\approx 0.95$

---

## 3. Toy Dataset Implementation (Diabetes)

The notebook [toy_dataset_implmentation.ipynb](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Linear Regression/toy_dataset_implmentation.ipynb) applies Scikit-Learn's `LinearRegression` model to the standard diabetes dataset.

### Exploratory Data Analysis (EDA)
1.  **Dimensions & Info**: The dataset contains $442$ instances and $10$ physiological baseline features (age, sex, bmi, average blood pressure, and six blood serum measurements).
2.  **Missing Values**: Evaluated using `df.isnull().sum()`, which returned $0$ null values across all columns.
3.  **Correlation Analysis**:
    *   Correlations of all features with the target variable (`target`) were computed and visualised using a sorted Seaborn barplot.
    *   **Strongest Predictor**: Body Mass Index (`bmi`) exhibits the highest correlation ($\approx 0.59$) with disease progression.
    *   A Seaborn `regplot` was generated to plot `bmi` vs. `target` showing a clear positive linear trend.

### Model and Metrics
*   **Train-Test Split**: 80% train, 20% test (random_state=42).
*   **Model**: Scikit-Learn `LinearRegression` (OLS solver).
*   **Test Metrics**:
    *   **Mean Squared Error (MSE)**: $2900.19$
    *   **Mean Absolute Error (MAE)**: $42.79$
    *   **R² Score**: $0.4526$ (explains $\approx 45.3\%$ of the variance in the target variable)

---

## 4. Real-world Dataset Implementation (California Housing)

The notebook [Real_dataset_implementation.ipynb](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Linear Regression/Real_dataset_implementation.ipynb) demonstrates a rigorous, production-grade ML pipeline on the California Housing dataset.

### Advanced Exploratory Data Analysis (EDA)
A comprehensive $3 \times 2$ grid of plots was generated to explore the dataset's characteristics:
1.  **Target Distribution (`MedHouseVal`)**: Shows positive skewness and a significant cap/boundary effect at exactly $5.0$ (representing the upper limit of the house value collection).
2.  **Bivariate Linear Relationship**: Median Income (`MedInc`) is plotted against `MedHouseVal` via `regplot`, revealing a strong positive linear relationship.
3.  **Multicollinearity Heatmap**: A correlation matrix heatmap identifies extreme collinearity between average rooms (`AveRooms`) and average bedrooms (`AveBedrms`) ($r = 0.85$).
4.  **Outlier Boxplots**: Identified heavy-tailed outliers in features `AveRooms`, `AveBedrms`, `AveOccup`, and `Population`.
5.  **Geographical Scatter Map**: Plotted `Latitude` vs. `Longitude` where color represented `MedHouseVal` and size represented `Population`, clearly mapping out the high-value coastal regions of California.

### Data Cleaning & Feature Engineering
To address the issues discovered in the EDA:
*   **Outlier Filtering**: Handled extreme outliers by filtering:
    *   `AveRooms < 15`
    *   `AveBedrms < 5`
    *   `AveOccup < 6`
    *   `Population < 10000`
*   **Capped Value Removal**: Dropped records with `MedHouseVal >= 5.0` to prevent the model from learning from artificial ceiling limits.
*   **Feature Transformation**: Resolves multicollinearity by creating a new ratio feature:
    $$\text{BedroomsPerRoom} = \frac{\text{AveBedrms}}{\text{AveRooms}}$$
    The highly redundant feature `AveBedrms` was then dropped.
*   **Spatial Feature Reduction**: Dropped `Latitude` and `Longitude` because standard linear regression models cannot capture complex geographic boundary clusters linearly.

### Feature Scaling & Data Leakage Prevention
To prevent data leakage, a `StandardScaler` was used. The scaler was fitted strictly on the training set, and applied as a transformation to both the training and test sets:
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Regularization Comparison on Cleaned Features
Three linear models were evaluated on the processed dataset:
1.  **Ordinary Least Squares (OLS)**: `LinearRegression()`
2.  **Ridge Regression (L2 Penalty)**: Adds $\alpha \sum w_j^2$ to shrink weights ($\alpha=1.0$).
3.  **Lasso Regression (L1 Penalty)**: Adds $\alpha \sum |w_j|$ to enforce sparsity ($\alpha=0.01$).

#### Baseline Results Comparison:
| Model | Test MSE | Test RMSE | Test MAE | Test R² Score |
| :--- | :--- | :--- | :--- | :--- |
| **OLS Linear Regression** | $0.3976$ | $0.6306$ | $0.4782$ | $0.5842$ |
| **Ridge Regression ($\alpha=1.0$)** | $0.3976$ | $0.6306$ | $0.4782$ | $0.5842$ |
| **Lasso Regression ($\alpha=0.01$)** | $0.3999$ | $0.6324$ | $0.4804$ | $0.5817$ |

*Takeaway*: The baseline OLS, Ridge, and Lasso models yield almost identical performance. This indicates that the baseline feature space is low-dimensional ($7$ features) and the baseline model is not overfitting.

---

## 5. Polynomial Features & High-Dimensional Overfitting

To model non-linear interactions, polynomial feature expansion (Degree 3) was applied:
```python
poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
```

### The Curse of Dimensionality & Feature Explosion
As the polynomial degree increases, the number of generated features increases exponentially:
*   **Linear (Degree 1)**: $8$ features
*   **Quadratic (Degree 2)**: $44$ features
*   **Cubic (Degree 3)**: $164$ features
*   **Quartic (Degree 4)**: $494$ features

This rapid expansion increases computational complexity and introduces a severe risk of overfitting (model starts fitting complex, noisy "wiggly" curves rather than generalizing).

### High-Dimensional Regularized Results (Degree 3)
| Model | Polynomial Test R² Score | Description & Evaluation |
| :--- | :--- | :--- |
| **Standard OLS (Poly)** | $0.6295$ | Suffers from unconstrained feature coefficients. |
| **Ridge Regression (Poly, $\alpha=10.0$)** | $0.6298$ | Shrinks all weights to reduce variance slightly. |
| **Lasso Regression (Poly, $\alpha=0.001$)** | **$0.6315$** | **Best Performance**. The L1 penalty zeroes out noisy/redundant interaction features, performing automated feature selection. |

---

## 6. Key Takeaways & Best Practices

1.  **Vectorized Optimization**: When implementing regression from scratch, vectorizing the gradient calculations using matrix dot products ($X^T \cdot (\hat{y} - y)$) is computationally superior to looping over samples.
2.  **Addressing Multicollinearity**: Multicollinearity (like $r=0.85$ between rooms and bedrooms) destabilizes linear model coefficients, making features appear statistically insignificant. Ratio engineering (e.g. `BedroomsPerRoom`) resolves this while keeping the key information.
3.  **Data Leakage Prevention**: Never fit feature scalers on the entire dataset. Always split the data first, fit on the training portion, and transform the test portion.
4.  **Regularization Selection**:
    *   Use **Ridge (L2)** when most features are relevant and coefficients need to be kept small.
    *   Use **Lasso (L1)** when the feature space is high-dimensional (e.g. after Polynomial Expansion) and sparse feature selection is needed to eliminate noise.
5.  **Capped Target Values**: Artificial maximum limits in datasets (like $5.0$ in California Housing) cause linear models to underestimate values at the high end. Filtering them out improves model learning and generalization on standard continuous distributions.
