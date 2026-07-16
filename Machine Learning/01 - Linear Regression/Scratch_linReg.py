import numpy as np
import matplotlib.pyplot as plt
class LinearRegression:
    def __init__(self, learning_rate = 0.01, n_iter = 1000) :
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y) :
        n_samples, n_features = X.shape
        y = y.ravel() 
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter) :
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X) : 
        return np.dot(X, self.weights) + self.bias


def mse(y_true, y_pred) :
    n = len(y_true)
    return (1/n) * np.sum((y_pred - y_true)**2)


np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.squeeze() + np.random.randn(100)

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = LinearRegression(learning_rate=0.01, n_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

error = mse(y_test, predictions)

print("--- Model Results ---")
print(f"MSE Error on Test Data: {error}")
print(f"Learned Weight: {model.weights}")
print(f"Learned Bias: {model.bias}")

# 6. Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="blue", alpha=0.6, label="Data Points")

# Generate points for the regression line
X_line = np.linspace(0, 2, 100).reshape(-1, 1)
y_line = model.predict(X_line)

plt.plot(X_line, y_line, color="red", linewidth=2.5, label="Fitted Line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression from Scratch")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

