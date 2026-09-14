# 03. How Encoder and Decoder Work: Deep Dive

This note provides a comprehensive deep dive into the internal mathematical mechanics of the **Encoder** and **Decoder** networks, explaining progressive layer scaling, activation functions, tied weights, the mathematical connection to **PCA**, and a complete worked numerical walkthrough.

---

## 1. The Encoder Network ($f_{\theta}$)

The **Encoder** acts as a non-linear compression function that maps an input vector $x \in \mathbb{R}^{d_x}$ to a low-dimensional latent vector $z \in \mathbb{R}^{d_z}$ ($d_z \ll d_x$).

```
Input x (784-D) ──► [ Linear(784->256) + ReLU ] ──► [ Linear(256->64) + ReLU ] ──► [ Linear(64->16) ] ──► Latent Code z (16-D)
```

### Layer-by-Layer Mathematical Equations:

For a 3-layer deep encoder:

1. **First Hidden Layer:**
   $$h_e^{(1)} = \text{ReLU}\left( W_e^{(1)} x + b_e^{(1)} \right)$$
   * Shape: $[256, 1] = [256, 784] \times [784, 1] + [256, 1]$

2. **Second Hidden Layer:**
   $$h_e^{(2)} = \text{ReLU}\left( W_e^{(2)} h_e^{(1)} + b_e^{(2)} \right)$$
   * Shape: $[64, 1] = [64, 256] \times [256, 1] + [64, 1]$

3. **Bottleneck Output (Latent Vector $z$):**
   $$z = W_e^{(3)} h_e^{(2)} + b_e^{(3)}$$
   * Shape: $[16, 1] = [16, 64] \times [64, 1] + [16, 1]$

> **Activation Choice on $z$:**  
> The final bottleneck layer producing $z$ is often kept **linear** (no activation) or passed through $\tanh$ to bound latent coordinates between $[-1, +1]$.

---

## 2. The Decoder Network ($g_{\phi}$)

The **Decoder** acts as an expansion function that takes the latent vector $z \in \mathbb{R}^{d_z}$ and reconstructs an output vector $\hat{x} \in \mathbb{R}^{d_x}$.

```
Latent Code z (16-D) ──► [ Linear(16->64) + ReLU ] ──► [ Linear(64->256) + ReLU ] ──► [ Linear(256->784) + Sigmoid ] ──► Output x̂ (784-D)
```

### Layer-by-Layer Mathematical Equations:

1. **First Expansion Layer:**
   $$h_d^{(1)} = \text{ReLU}\left( W_d^{(1)} z + b_d^{(1)} \right)$$
   * Shape: $[64, 1] = [64, 16] \times [16, 1] + [64, 1]$

2. **Second Expansion Layer:**
   $$h_d^{(2)} = \text{ReLU}\left( W_d^{(2)} h_d^{(1)} + b_d^{(2)} \right)$$
   * Shape: $[256, 1] = [256, 64] \times [64, 1] + [256, 1]$

3. **Final Reconstruction Layer ($\hat{x}$):**
   $$\hat{x} = \text{Sigmoid}\left( W_d^{(3)} h_d^{(2)} + b_d^{(3)} \right)$$
   * Shape: $[784, 1] = [784, 256] \times [256, 1] + [784, 1]$

> **Activation Choice on $\hat{x}$:**
> * If input pixels are normalized to $[0, 1]$, use a **Sigmoid** activation on the final decoder layer.
> * If input data is standardized (mean 0, variance 1) or continuous values, use a **Linear** output layer.

---

## 3. Tied Weights vs. Untied Weights

In an Autoencoder, you have two choices for defining decoder weights:

```
┌───────────────────────────────────┬───────────────────────────────────┐
│ Untied Weights (Standard Default) │ Tied Weights (Weight Sharing)     │
├───────────────────────────────────┼───────────────────────────────────┤
│ The encoder weights W_e and       │ The decoder weights are set to    │
│ decoder weights W_d are completely│ the exact transpose of the        │
│ independent learnable matrices:   │ encoder weights:                  │
│                                   │                                   │
│    W_d ≠ W_eᵀ                     │    W_d = W_eᵀ                     │
│                                   │                                   │
│ • Higher model capacity.          │ • Cuts model parameters by 50%.   │
│ • Standard in modern deep nets.   │ • Strong regularization effect.   │
└───────────────────────────────────┴───────────────────────────────────┘
```

---

## 4. Linear Autoencoders vs. Non-Linear (Connection to PCA)

An important foundational concept in machine learning is the mathematical relationship between Autoencoders and **Principal Component Analysis (PCA)**:

```
========================================================================================================================
                                     LINEAR AUTOENCODER vs. PCA
========================================================================================================================

  1. LINEAR AUTOENCODER (All activations are Linear: f(z) = z):
     • If you build an undercomplete autoencoder with purely linear layers and Mean Squared Error (MSE) loss,
       the network learns the EXACT SAME optimal subspace as Principal Component Analysis (PCA)!
     • The latent dimensions span the principal components of the data.

  2. NON-LINEAR AUTOENCODER (Activations like ReLU, Tanh, Sigmoid):
     • PCA can only discover FLAT linear hyperplanes.
     • Non-linear autoencoders can learn CURVED, complex non-linear manifolds (e.g. the Swiss Roll)!

========================================================================================================================
```

```
           PCA (Linear Flat Plane)                   NON-LINEAR AUTOENCODER (Curved Manifold)
           
                 /                                             .--~~~--.
                /                                            .-'         `-.
               /                                            (  Curved Space )
              /                                              `-.         .-'
             /                                                 `--...--'
   (Cannot bend around curves!)                           (Flexibly bends to fit data!)
```

---

## 5. Step-by-Step Worked Numerical Example

Let's trace a simplified numerical forward pass through a 1-hidden-layer Autoencoder:
* Input: $x = \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix}$ (2D input)
* Bottleneck: $z$ (1D latent space)
* Encoder Weights: $W_e = \begin{bmatrix} 0.6 & 0.8 \end{bmatrix}, \quad b_e = -0.2$
* Decoder Weights: $W_d = \begin{bmatrix} 0.6 \\ 0.8 \end{bmatrix}, \quad b_d = \begin{bmatrix} 0.1 \\ 0.1 \end{bmatrix}$

---

### Step 1: Encoder Forward Pass (Compress $x \to z$)
$$u_e = W_e x + b_e = (0.6 \times 1.0) + (0.8 \times 0.5) - 0.2 = 0.6 + 0.4 - 0.2 = 0.8$$
$$z = \text{ReLU}(0.8) = \mathbf{0.8}$$

* The 2D input $[1.0, 0.5]^T$ is compressed into the 1D scalar $z = \mathbf{0.8}$!

---

### Step 2: Decoder Forward Pass (Reconstruct $z \to \hat{x}$)
$$u_d = W_d z + b_d = \begin{bmatrix} 0.6 \\ 0.8 \end{bmatrix} (0.8) + \begin{bmatrix} 0.1 \\ 0.1 \end{bmatrix} = \begin{bmatrix} 0.48 + 0.1 \\ 0.64 + 0.1 \end{bmatrix} = \begin{bmatrix} 0.58 \\ 0.74 \end{bmatrix}$$
$$\hat{x} = \text{Sigmoid}(u_d) = \begin{bmatrix} \sigma(0.58) \\ \sigma(0.74) \end{bmatrix} \approx \begin{bmatrix} \mathbf{0.641} \\ \mathbf{0.677} \end{bmatrix}$$

---

### Step 3: Reconstruction Error Calculation
$$\text{Reconstruction Error (MSE)} = \frac{1}{2} \left[ (1.0 - 0.641)^2 + (0.5 - 0.677)^2 \right] = \frac{1}{2} \left[ (0.359)^2 + (-0.177)^2 \right] \approx \mathbf{0.0801}$$

---

## 6. PyTorch Implementation Example

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        
        # 1. Encoder Network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)  # Bottleneck Output
        )
        
        # 2. Decoder Network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Reconstructs pixel values in [0, 1]
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# Test with a batch of images
model = Autoencoder(input_dim=784, latent_dim=32)
sample_input = torch.randn(64, 784)
reconstruction, latent_code = model(sample_input)

print("Reconstruction Shape:", reconstruction.shape)  # [64, 784]
print("Latent Code Shape:", latent_code.shape)        # [64, 32]
```

---

## Summary Key Points

* **Encoder:** Compresses high-dimensional $x \in \mathbb{R}^{d_x}$ into latent vector $z \in \mathbb{R}^{d_z}$.
* **Decoder:** Reconstructs latent vector $z$ back into $\hat{x} \in \mathbb{R}^{d_x}$.
* **Tied Weights:** $W_d = W_e^T$ cuts parameters in half and regularizes training.
* **PCA Equivalence:** A linear autoencoder with MSE loss learns the exact principal subspace of PCA, whereas non-linear autoencoders learn curved manifolds.
