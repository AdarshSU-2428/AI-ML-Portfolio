# 07. Stacked (Deep) Recurrent Neural Networks

Just as adding more convolutional layers to a CNN allows it to learn higher-level visual features (from edges to textures to full objects), **stacking multiple RNN layers on top of each other** allows recurrent networks to learn **hierarchical sequential representations**.

This note explains **Stacked (Deep) RNNs**, how information flows across layers, and how to implement them in PyTorch.

---

## 1. Why Stack RNN Layers? (Hierarchical Abstraction)

A single-layer RNN can struggle to model highly complex language structures or long-range temporal relationships. Stacking layers creates a **deep hierarchy of temporal features**:

```
                  HIERARCHICAL TEMPORAL FEATURE LEARNING
                  
  Layer 3 (High-Level):    Learns overall paragraph intent, topic & sentiment.
           ▲
           │ (Hidden state sequence h_t^(2) passed upward as input)
  Layer 2 (Mid-Level):     Learns grammar, sentence syntax & phrase structure.
           ▲
           │ (Hidden state sequence h_t^(1) passed upward as input)
  Layer 1 (Low-Level):     Learns local word combinations, stems & characters.
           ▲
     Input Sequence x_t
```

---

## 2. How Stacked RNNs Work (Layer-to-Layer Data Flow)

In a **Stacked RNN with $L$ layers**:

1. **Layer 1:** Receives the original external input sequence $x_t$ and computes hidden state sequence $h_t^{(1)}$.
2. **Layer 2:** Takes the hidden state sequence $h_t^{(1)}$ from Layer 1 **as its input sequence** ($x_t^{(2)} = h_t^{(1)}$) and computes new hidden state sequence $h_t^{(2)}$.
3. **Layer $L$ (Top Layer):** Computes final hidden state $h_t^{(L)}$, which is passed to the output layer for prediction $\hat{y}_t$.

---

## 3. Complete Data Flow Diagram of a 2-Layer Stacked RNN

```
========================================================================================================================
                                    2-LAYER STACKED RNN DATA FLOW (UNROLLED)
========================================================================================================================

                                         Prediction ŷ₁           Prediction ŷ₂           Prediction ŷ_T
                                               ▲                       ▲                       ▲
                                               │ (W_hy)                │ (W_hy)                │ (W_hy)
                                            ┌─────┐                 ┌─────┐                 ┌─────┐
  LAYER 2 (Top Layer):             h₀^(2)─► │h₁^(2)│ ──(W_hh^(2))─► │h₂^(2)│ ──(W_hh^(2))─► │h_T^(2)│
                                            └─────┘                 └─────┘                 └─────┘
                                               ▲                       ▲                       ▲
                                               │ (Inter-layer)         │ (Inter-layer)         │ (Inter-layer)
                                               │ (Dropout)             │ (Dropout)             │ (Dropout)
                                            ┌─────┐                 ┌─────┐                 ┌─────┐
  LAYER 1 (Bottom Layer):          h₀^(1)─► │h₁^(1)│ ──(W_hh^(1))─► │h₂^(1)│ ──(W_hh^(1))─► │h_T^(1)│
                                            └─────┘                 └─────┘                 └─────┘
                                               ▲                       ▲                       ▲
                                               │ (W_xh^(1))            │ (W_xh^(1))            │ (W_xh^(1))
                                            Input x₁                Input x₂                Input x_T
                                            ("The")                 ("movie")               ("was")

========================================================================================================================
```

---

## 4. Mathematical Equations & Symbol Breakdown

For a Stacked RNN with layer index $l \in \{1, 2, \dots, L\}$ at time step $t$:

### 1. First Layer Equation ($l = 1$):
Receives external input $x_t$:

$$h_t^{(1)} = \tanh\left( W_{xh}^{(1)} x_t + W_{hh}^{(1)} h_{t-1}^{(1)} + b_h^{(1)} \right)$$

### 2. Intermediate Layer Equation ($l > 1$):
Receives hidden state $h_t^{(l-1)}$ from the layer directly below:

$$h_t^{(l)} = \tanh\left( W_{xh}^{(l)} h_t^{(l-1)} + W_{hh}^{(l)} h_{t-1}^{(l)} + b_h^{(l)} \right)$$

### 3. Output Equation (Top Layer $L$):
$$\hat{y}_t = \text{softmax}\left( W_{hy} h_t^{(L)} + b_y \right)$$

---

### Symbol Breakdown Table:

| Symbol | Meaning | Vector / Matrix Dimension |
| :--- | :--- | :--- |
| **$l$** | Layer index ($l = 1, 2, \dots, L$) | Scalar integer |
| **$x_t$** | External input vector at time step $t$ | $[d_x, 1]$ |
| **$h_t^{(l)}$** | Hidden state vector of layer $l$ at time step $t$ | $[d_h^{(l)}, 1]$ |
| **$h_{t-1}^{(l)}$** | Hidden state vector of layer $l$ at previous time step $t-1$ | $[d_h^{(l)}, 1]$ |
| **$W_{xh}^{(l)}$** | Input-to-Hidden weight matrix for layer $l$ | $[d_h^{(l)}, d_h^{(l-1)}]$ |
| **$W_{hh}^{(l)}$** | Recurrent Hidden-to-Hidden weight matrix for layer $l$ | $[d_h^{(l)}, d_h^{(l)}]$ |
| **$b_h^{(l)}$** | Bias vector for layer $l$ | $[d_h^{(l)}, 1]$ |
| **$W_{hy}$** | Output weight matrix mapping top layer $h_t^{(L)} \to \hat{y}_t$ | $[d_y, d_h^{(L)}]$ |

---

## 5. Regularization: Inter-Layer Dropout

As stacked RNNs grow deeper, they become prone to overfitting. **Dropout** is applied to the **vertical connections between layers** (NOT to the horizontal recurrent connections over time).

```
  Layer 2 (Top Layer):      h_t^(2)
                              ▲
                              │  ◄── APPLY DROPOUT HERE (Vertical inter-layer connection)
  Layer 1 (Bottom Layer):   h_t^(1)
```

* **Why Vertical Dropout?** Applying dropout to horizontal recurrent connections over time disrupts the network's ability to retain memory across time steps. Applying dropout vertically between layers regularizes feature representations without destroying long-term memory!

---

## 6. PyTorch Implementation Example

In PyTorch, creating a Stacked LSTM or GRU is straightforward using the `num_layers` and `dropout` arguments:

```python
import torch
import torch.nn as nn

# Model Hyperparameters
batch_size = 32
seq_len = 20
input_dim = 100
hidden_dim = 256
num_layers = 3     # <-- Stacks 3 LSTM layers on top of each other!
num_classes = 5

# Create a 3-Layer Stacked LSTM with 20% Inter-Layer Dropout
stacked_lstm = nn.LSTM(
    input_size=input_dim,
    hidden_size=hidden_dim,
    num_layers=num_layers,  # 3 Stacked Layers
    dropout=0.2,            # Applies 20% Dropout between layers (l=1->2 and l=2->3)
    batch_first=True
)

# Input Tensor: [Batch=32, SeqLen=20, InputDim=100]
inputs = torch.randn(batch_size, seq_len, input_dim)

# Forward pass
output, (h_n, c_n) = stacked_lstm(inputs)

# Output Shapes:
# output: [32, 20, 256]  (Contains top-layer hidden states for all 20 steps)
# h_n:    [3, 32, 256]   (Contains final hidden state for ALL 3 layers!)
print("Top-Layer Output Shape:", output.shape)
print("All-Layers Hidden States (h_n) Shape:", h_n.shape)

# Classifier head connected to top layer output at final step
fc = nn.Linear(hidden_dim, num_classes)
logits = fc(output[:, -1, :])  # Shape: [32, 5]
```

---

## Summary Key Points

- [x] **Stacked (Deep) RNNs** place multiple recurrent layers on top of each other ($L > 1$).
- [x] **Hierarchical Feature Learning:** Lower layers capture local syntax/patterns; higher layers capture semantic intent.
- [x] **Data Flow:** Layer $l$ receives the hidden state sequence $h_t^{(l-1)}$ of layer $l-1$ as its input sequence.
- [x] **Inter-Layer Dropout:** Dropout is applied vertically between layers (`dropout=0.2`) to prevent overfitting without breaking horizontal time memory.
