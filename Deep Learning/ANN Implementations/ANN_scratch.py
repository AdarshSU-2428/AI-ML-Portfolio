import numpy as np

class ArtificialNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        np.random.seed(42)

        self.learning_rate = learning_rate 

        # Weights initialized with Gaussian distribution; biases initialized to zero
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def sigmoid_derivative(self, output):
        # Derivative of sigmoid expressed in terms of activation: σ'(z) = a * (1 - a)
        return output * (1 - output)

    def mse_loss_derivative(self, y_true, y_pred):
        # dL/dA2 = (2 / n) * (y_pred - y_true)
        n = y_true.shape[0]
        return (2 / n) * (y_pred - y_true)

    def forward(self, X):
        # Input layer -> Hidden layer: Z1 = X @ W1 + b1, A1 = σ(Z1)
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.sigmoid(self.Z1)

        # Hidden layer -> Output layer: Z2 = A1 @ W2 + b2, A2 = σ(Z2)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    def backward(self, X, y):
        # --- Output Layer (Layer 2) Gradients ---
        dA2 = self.mse_loss_derivative(y, self.A2)
        dZ2 = dA2 * self.sigmoid_derivative(self.A2)        # dZ2 = (dL/dA2) * σ'(Z2)

        dW2 = self.A1.T @ dZ2                                # dW2 = A1^T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)            # Sum gradients across samples

        # --- Hidden Layer (Layer 1) Gradients ---
        dA1 = dZ2 @ self.W2.T                               # Propagate output error to hidden layer
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)        # dZ1 = dA1 * σ'(Z1)

        dW1 = X.T @ dZ1                                     # dW1 = X^T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)            # Sum gradients across samples

        # --- Parameter Update (Gradient Descent) ---
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def training_loop(self, X, y, epochs=5000):
        for epoch in range(epochs):
            # Full training cycle: Forward pass -> Loss computation -> Backpropagation
            pred = self.forward(X)
            loss = self.mse_loss(y, pred)
            self.backward(X, y)

            if epoch % 500 == 0:
                print(f"Epoch {epoch:4d} | Loss = {loss:.6f}")

    def predict(self, X):
        predictions = self.forward(X)
        # Binary classification using threshold of 0.5
        return np.round(predictions)


# --- Dataset: XOR Truth Table ---
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])


# --- Instantiate and Train Neural Network ---
nn = ArtificialNeuralNetwork(
    input_size=2,
    hidden_size=4,
    output_size=1,
    learning_rate=1.0
)

nn.training_loop(X, y, epochs=5000)

print("Predictions:")
print(nn.predict(X))

print("\nRaw Output:")
print(nn.forward(X))