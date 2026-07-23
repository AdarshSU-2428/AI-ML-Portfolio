# 06. Loss Functions in ANNs (Introduction)

Once **Forward Propagation** computes a prediction ($\hat{y}$), the neural network must evaluate how accurate that guess was. This evaluation is handled by the **Loss Function**.

---

## 1. Prediction vs. Actual

To calculate error, we compare two key variables:

* **Actual Target Value ($y$):** The true correct answer from your labeled dataset (also called *Ground Truth*).
* **Predicted Value ($\hat{y}$):** The output produced by the neural network during forward propagation.

### Examples:
| Task | Actual Target ($y$) | Model Prediction ($\hat{y}$) | Status |
| :--- | :--- | :--- | :--- |
| **House Price** | `$500,000` | `$410,000` | Underestimated by $90k |
| **Spam Filter** | `1` (Spam Email) | `0.92` (92% prob Spam) | Very Accurate |
| **Medical Scan** | `1` (Tumor Present) | `0.15` (15% prob Tumor) | Dangerous Error! |

---

## 2. Error Calculation & Why Raw Differences Aren't Enough

Why can't we simply define Error as $(y - \hat{y})$?

Suppose we test two data samples:
* Sample 1: Actual $y = 10$, Prediction $\hat{y} = 15 \implies \text{Error} = 10 - 15 = -5$
* Sample 2: Actual $y = 10$, Prediction $\hat{y} = 5 \implies \text{Error} = 10 - 5 = +5$

If we average these errors: $(-5) + (+5) = 0$. The raw errors cancel each other out, making a terribly inaccurate model look "perfect"!

### Mathematical Solutions:
To prevent negative and positive errors from canceling out, loss functions square the error or take absolute values:

1. **Mean Squared Error (MSE) for Regression:**
   $$L(y, \hat{y}) = \frac{1}{2} (y - \hat{y})^2$$

2. **Binary Cross-Entropy Loss for Classification:**
   $$L(y, \hat{y}) = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$

---

## 3. Why Loss is Needed in Artificial Neural Networks

The Loss Function plays three critical roles in deep learning:

```
  +-----------------------+      +-----------------------+      +-----------------------+
  |  Forward Propagation  | ---> |     Loss Function     | ---> |    Backpropagation    |
  |  Calculates Guess ŷ   |      | Evaluates Error Score |      | Adjusts Weights W & b |
  +-----------------------+      +-----------------------+      +-----------------------+
```

1. **The Objective Metric:** It converts model performance into a single numerical score (Loss). Lower Loss = Better Model.
2. **Bridge to Backpropagation:** The Loss value provides the starting point for calculating gradients ($\frac{\partial L}{\partial w}$) backwards through the network.
3. **Training Benchmark:** By tracking Loss across epochs, you monitor whether the model is learning, overfitting, or underfitting.
