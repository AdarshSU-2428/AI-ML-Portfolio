import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sibling_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Linear Regression"))

if sibling_dir not in sys.path:
    sys.path.append(sibling_dir)

from Scratch_linReg import LinearRegression

class LogisticRegression(LinearRegression) :
    def _sigmoid(self, z) :
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y) :
        n_samples, n_features = X.shape
        y = y.ravel()
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter) :
            linear_model = np.dot(X, self.weights) + self.bias

            y_pred = self._sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X) :
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X) :
        return np.where(self.predict_proba(X) >= 0.5, 1, 0)

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_model = LogisticRegression(learning_rate=0.01, n_iter=1000)
log_model.fit(X_train, y_train)

pred = log_model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print(f"Accuracy: {accuracy}")

print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=data.target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))
