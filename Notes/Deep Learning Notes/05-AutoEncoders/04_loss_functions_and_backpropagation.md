# 04. Loss Functions and Backpropagation in Autoencoders

To train an Autoencoder, we need an objective function that mathematically quantifies how closely the reconstructed output $\hat{x}$ matches the original input $x$. This loss is called the **Reconstruction Loss**.

This note explains the two primary reconstruction loss functions and provides a detailed step-by-step walkthrough of **Backpropagation** through the Decoder, Bottleneck, and Encoder.

---

## 1. Reconstruction Loss Functions

Depending on whether your input data consists of continuous numbers or normalized image pixels, you choose between two standard loss functions:

```
┌───────────────────────────────────┬───────────────────────────────────┐
│ 1. Mean Squared Error (MSE / L2)  │ 2. Binary Cross-Entropy (BCE)     │
├───────────────────────────────────┼───────────────────────────────────┤
│ Best for continuous, unconstrained│ Best for normalized image pixels  │
│ or standardized real-valued data. │ scaled strictly between [0, 1].   │
└───────────────────────────────────┴───────────────────────────────────┘
```

---

### A. Mean Squared Error (MSE Loss)

Measures the average squared Euclidean distance between original input features $x$ and reconstructed features $\hat{x}$:

$$L_{\text{MSE}}(x, \hat{x}) = \frac{1}{D} \sum_{i=1}^{D} \left( x_i - \hat{x}_i \right)^2$$

Where $D$ is the total number of input features / pixels (e.g., $D = 784$).

* **When to Use:**
  * Audio spectrograms, tabular sensor features, physical measurements, or any unbounded continuous data.
  * Paired with a **Linear output layer** on the decoder.

---

### B. Binary Cross-Entropy Loss (BCE Loss)

When image pixel intensities are normalized to the range $[0.0, 1.0]$, each pixel can be treated as an independent Bernoulli probability (probability of the pixel being "white" or "active"):

$$L_{\text{BCE}}(x, \hat{x}) = - \frac{1}{D} \sum_{i=1}^{D} \left[ x_i \log\left( \hat{x}_i \right) + (1 - x_i) \log\left( 1 - \hat{x}_i \right) \right]$$

* **When to Use:**
  * Grayscale or RGB images normalized to $[0, 1]$ (e.g., MNIST, Fashion-MNIST).
  * Paired with a **Sigmoid output layer** on the decoder.
* **Why BCE over MSE for Images?**
  * BCE penalizes confidently incorrect pixel predictions much more aggressively than MSE, resulting in sharper edges and faster convergence for images!

---

## 2. Complete Backpropagation Flow in Autoencoders

During training, error gradients flow backward through the entire architecture across three consecutive stages:

```
========================================================================================================================
                                     AUTOENCODER BACKPROPAGATION DATA FLOW
========================================================================================================================

                         RECONSTRUCTION LOSS L(x, x̂) = MSE(x, x̂) or BCE(x, x̂)
                                                  │
                                                  ▼ (∂L/∂x̂)
                                   ┌──────────────────────────────┐
                                   │     RECONSTRUCTED OUTPUT x̂   │
                                   └──────────────────────────────┘
                                                  │
                                                  ▼ (Stage 1: Backprop through Decoder)
                                   ┌──────────────────────────────┐
                                   │       DECODER WEIGHTS W_d    │ ──► Update: W_d ← W_d - η(∂L/∂W_d)
                                   └──────────────────────────────┘
                                                  │
                                                  ▼ (Stage 2: Gradient at Bottleneck: ∂L/∂z)
                                   ┌──────────────────────────────┐
                                   │      LATENT BOTTLENECK z     │
                                   └──────────────────────────────┘
                                                  │
                                                  ▼ (Stage 3: Backprop through Encoder)
                                   ┌──────────────────────────────┐
                                   │       ENCODER WEIGHTS W_e    │ ──► Update: W_e ← W_e - η(∂L/∂W_e)
                                   └──────────────────────────────┘
                                                  │
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │       ORIGINAL INPUT x       │
                                   └──────────────────────────────┘

========================================================================================================================
```

---

## 3. The 3 Backpropagation Stages Explained

---

### Stage 1: Gradient Computation at Decoder
1. The error gradient $\frac{\partial L}{\partial \hat{x}}$ is computed directly at the output layer:
   * For MSE Loss: $\frac{\partial L}{\partial \hat{x}_i} = -\frac{2}{D} (x_i - \hat{x}_i)$
2. Gradients propagate backward through the decoder's activation function and linear layers.
3. Accumulates weight updates for all decoder layers:
   $$\frac{\partial L}{\partial W_d} = \left( \frac{\partial L}{\partial u_d} \right) z^T$$
   $$\Delta W_d = - \eta \frac{\partial L}{\partial W_d}$$

---

### Stage 2: Passing Gradients Across the Bottleneck ($z$)
The total error gradient arriving at the latent bottleneck vector $z$ is:

$$\frac{\partial L}{\partial z} = W_d^T \left( \frac{\partial L}{\partial u_d} \right)$$

* **Intuition:** This vector tells the network: *"How should the compressed latent coordinates in $z$ shift to make the decoder's reconstruction more accurate?"*

---

### Stage 3: Gradient Computation at Encoder
1. The bottleneck gradient $\frac{\partial L}{\partial z}$ enters the top layer of the encoder.
2. Gradients backpropagate through the encoder's activation functions (e.g., ReLU) and hidden layers:
   $$\frac{\partial L}{\partial u_e} = \frac{\partial L}{\partial z} \odot \text{ReLU}'(u_e)$$
3. Accumulates weight updates for all encoder layers:
   $$\frac{\partial L}{\partial W_e} = \left( \frac{\partial L}{\partial u_e} \right) x^T$$
   $$\Delta W_e = - \eta \frac{\partial L}{\partial W_e}$$

---

## 4. Complete PyTorch Training Loop Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Instantiate Model, Loss Function, and Optimizer
model = Autoencoder(input_dim=784, latent_dim=32)
criterion = nn.MSELoss()  # Or nn.BCELoss() if using Sigmoid output
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Single Training Epoch Loop
for batch_idx, (images, _) in enumerate(train_loader):
    # Flatten 28x28 images to 784-dim vectors (No labels needed!)
    images = images.view(images.size(0), -1)
    
    # 1. Forward Pass
    reconstructed, latent_z = model(images)
    
    # 2. Compute Reconstruction Loss: Compare output x̂ to original input x!
    loss = criterion(reconstructed, images)
    
    # 3. Backward Pass (Compute Gradients)
    optimizer.zero_grad()
    loss.backward()
    
    # 4. Update Encoder and Decoder Weights Jointly
    optimizer.step()

print("Epoch complete! Loss:", loss.item())
```

---

## Summary Key Points

* **Reconstruction Loss:** Compares $\hat{x}$ directly to $x$ without requiring human-annotated labels ($y = x$).
* **MSE Loss:** Preferred for unbounded or continuous measurements.
* **BCE Loss:** Preferred for normalized $[0, 1]$ image pixels.
* **End-to-End Gradient Flow:** Error signals flow from **Output $\hat{x}$** $\to$ **Decoder ($W_d$)** $\to$ **Bottleneck ($z$)** $\to$ **Encoder ($W_e$)** in a single backward pass.
