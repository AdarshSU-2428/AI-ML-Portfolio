# Logistic Regression Study Notes

This document contains detailed study notes on Logistic Regression based on the following implementations:
*   **From-Scratch Implementation**: [Scratch_logReg.py](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Logistic Regression/Scratch_logReg.py)
*   **Toy Dataset Implementation (Iris)**: [toy_dataset_implementation.ipynb](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Logistic Regression/toy_dataset_implementation.ipynb)
*   **Real-world Dataset Implementation (Titanic)**: [Real_dataset_implementation.ipynb](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Logistic Regression/Real_dataset_implementation.ipynb)

---

## 1. Mathematical Underpinnings of Logistic Regression

Logistic Regression is a supervised classification model used to estimate the probability that an instance belongs to a particular class (binary classification).

### The Logistic (Sigmoid) Function
To map any real-valued linear combination $z = \mathbf{x}^T \mathbf{w} + b$ into a probability range $[0, 1]$, we use the Sigmoid activation function:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

*   **Asymptotic Limits**: As $z \to \infty$, $\sigma(z) \to 1$; as $z \to -\infty$, $\sigma(z) \to 0$.
*   **Symmetry**: $\sigma(0) = 0.5$, which represents the typical decision threshold for binary classification.
*   **Derivative**: The derivative of the sigmoid function exhibits a clean, self-referential mathematical property:
    $$\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$$

### Objective Function: Binary Cross-Entropy Loss (Log Loss)
Instead of Mean Squared Error, which leads to a non-convex optimization surface in classification, Logistic Regression minimizes the negative log-likelihood of the dataset. For $N$ samples:
$$L(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
Where $\hat{y}_i = \sigma(\mathbf{x}_i^T \mathbf{w} + b)$ is the predicted probability.

### Deriving the Gradients
Applying the chain rule, the partial derivatives w.r.t parameters are:
1.  **Gradient w.r.t Weights ($\mathbf{w}$)**:
    $$\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{N} \mathbf{X}^T (\mathbf{\hat{y}} - \mathbf{y})$$
2.  **Gradient w.r.t Bias ($b$)**:
    $$\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)$$

> [!IMPORTANT]
> **Key Gradient Scaler Difference**: 
> Note that the gradient formulas for MSE loss (Linear Regression) contain a scalar coefficient of $\frac{2}{N}$, while Binary Cross-Entropy loss (Logistic Regression) contains a scalar coefficient of $\frac{1}{N}$.

---

## 2. From-Scratch Implementation (`Scratch_logReg.py`)

The python script [Scratch_logReg.py](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Logistic Regression/Scratch_logReg.py) implements logistic regression by subclassing the from-scratch [LinearRegression](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Linear Regression/Scratch_linReg.py#L3) class.

### Inheritance and Class Design
*   **[LogisticRegression](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Logistic Regression/Scratch_logReg.py#L17)**: Inherits hyperparameters (`learning_rate`, `n_iter`) from the baseline linear model, but overrides and implements classification-specific logic.
*   **[_sigmoid](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Logistic Regression/Scratch_logReg.py#L18)**: Computes the element-wise logistic sigmoid mapping.
*   **[fit](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Logistic Regression/Scratch_logReg.py#L21)**: Initializes weights $\mathbf{w}$ to a zero vector and bias $b$ to $0$. It runs gradient descent for `n_iter` iterations using the cross-entropy derivative:
    ```python
    linear_model = np.dot(X, self.weights) + self.bias
    y_pred = self._sigmoid(linear_model)
    
    dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
    db = (1/n_samples) * np.sum(y_pred - y)
    ```
*   **[predict_proba](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Logistic Regression/Scratch_logReg.py#L38)**: Computes and returns the probability scores.
*   **[predict](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Logistic Regression/Scratch_logReg.py#L42)**: Hard-classifies predictions to $1$ or $0$ based on a $0.5$ probability threshold.

### Experiment on Breast Cancer Dataset
*   **Dataset**: Breast Cancer dataset ($569$ samples, $30$ features).
*   **Pipeline**: 80/20 train-test split, scaling via `StandardScaler` to prevent feature magnitude dominance, and training at `learning_rate=0.01` for `n_iter=1000`.
*   **Results**:
    *   **Test Accuracy**: $98.25\%$
    *   **Confusion Matrix**: 
        $$\begin{bmatrix} 42 & 1 \\ 1 & 70 \end{bmatrix}$$
        Only 2 instances misclassified out of 114 test samples (1 False Positive, 1 False Negative).

---

## 3. Toy Dataset Implementation (Iris)

The notebook [toy_dataset_implementation.ipynb](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Logistic Regression/toy_dataset_implementation.ipynb) leverages `scikit-learn` to solve a multi-class classification problem.

### Multi-Class Mechanics
*   The Iris dataset contains three target classes: *setosa*, *versicolor*, and *virginica*.
*   Scikit-Learn's `LogisticRegression` automatically abstracts multi-class targets using a multinomial loss formulation (Softmax function) or One-vs-Rest (OvR) scheme under the hood.
*   **Pipeline**:
    1.  Splits data 80/20 (random_state=42).
    2.  Applies `StandardScaler` to standardize feature variances.
    3.  Fits `LogisticRegression()` and predicts on scaled test data.
*   **Evaluation**:
    *   **Test Accuracy**: $100\%$ (perfect classification on all 30 test samples).
    *   **Confusion Matrix**:
        $$\begin{bmatrix} 10 & 0 & 0 \\ 0 & 9 & 0 \\ 0 & 0 & 11 \end{bmatrix}$$

---

## 4. Real-world Dataset Implementation (Titanic)

The notebook [Real_dataset_implementation.ipynb](file:///c:/Users/ADARSH S SAHOO/Desktop/Engineering folder/AI-ML_2027/Machine Learning/Logistic Regression/Real_dataset_implementation.ipynb) demonstrates a full-lifecycle classification pipeline on the raw Titanic dataset.

### Advanced Data Cleaning & Imputation
1.  **Drop High-Missing Column**: Dropped `Cabin` because it was missing $>70\%$ of its records, making reliable imputation impossible.
2.  **Row Deletion**: Dropped the 2 rows with missing `Embarked` values since it represents a minor fraction of the data.
3.  **Mean Imputation**: Filled missing `Age` values with the column mean.
4.  **Drop Uninformative Columns**:
    *   `Name` & `Ticket`: Dropped due to high cardinality (unique strings) that do not aid in finding generalizable rules.
    *   `PassengerId`: Dropped because serial numbers act as arbitrary identifiers and can cause overfitting.

### Preprocessing & Feature Engineering
*   **Encoding**: Converted categorical variables `Sex` and `Embarked` to integers using `LabelEncoder`.
*   **Downcasting**: Cast the entire DataFrame to `int` to standardize representation.
*   **Feature Engineering**:
    *   `FamilySize = SibSp + Parch + 1`: Captures the group size travel context.
    *   `IsAlone = (FamilySize == 1)`: Binary indicator flagging solo travelers.
    *   *Rationale*: Evacuation priorities on the Titanic ("women and children first") favored families over solo travelers. Creating these features gives the model direct paths to linear splits.

### Feature Scaling
Because L2 regularization is active by default in Scikit-Learn's logistic solver, scaling features via `StandardScaler` is **mandatory** to ensure the penalty is applied uniformly across features. It was fitted only on the training split to prevent data leakage.

### Model Performance (Titanic Test Set)
*   **Accuracy**: $80.34\%$
*   **Precision**: $73.61\%$
*   **Recall**: $76.81\%$
*   **F1-Score**: $75.18\%$
*   **Confusion Matrix**:
    *   **True Negatives (Not Survived)**: $90$
    *   **False Positives**: $19$
    *   **False Negatives**: $16$
    *   **True Positives (Survived)**: $53$

### Interpretability & Weight Heatmap
Plotting the model coefficients ($\mathbf{w}$) on a `coolwarm` scale yields physical insights:
*   **Sex Coefficient** ($-1.268$): Since Male is encoded as $1$ and Female as $0$, this strongly negative coefficient indicates that being male drastically reduces survival probability.
*   **Pclass Coefficient** ($-0.897$): Being in a higher numbered class ($2$nd or $3$rd class) has a highly negative coefficient, indicating a significantly reduced probability of survival.
*   **Age Coefficient** ($-0.567$): Shows that older passengers had a lower probability of survival.
*   **Fare Coefficient** ($+0.046$): Positive coefficient shows higher ticket prices slightly increased survival probability.

---

## 5. Key Best Practices & Takeaways

1.  **Gradient Descent Implementations**:
    *   Understand the gradient scaling: $\frac{1}{N}$ scaling for Binary Cross Entropy vs $\frac{2}{N}$ for Mean Squared Error.
2.  **Regularization & Scaling**:
    *   Always scale features (e.g., using `StandardScaler`) when using regularized (L1/L2) Logistic Regression. Without scaling, features with larger magnitudes are penalized less severely, distorting coefficient importance.
3.  **Demographic & Social Feature Engineering**:
    *   In real-world contexts, aggregate demographic features (e.g. `FamilySize`, `IsAlone`) are highly predictive because they capture historical and situational realities (such as evacuation protocols).
4.  **Handling Missing Data**:
    *   Delete features missing over $>50\%$ of data if imputation is too speculative.
    *   Impute continuous missing variables with mean or median, and drop records for categorical variables with negligible missingness.
5.  **Model Coefficients as Feature Importance**:
    *   Unlike black-box models, Logistic Regression provides highly interpretable coefficients. Visualizing coefficients using a 1D heatmap helps verify model fairness and align learned weights with physical domain knowledge.
