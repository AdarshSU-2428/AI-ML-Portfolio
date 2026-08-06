# 06. Bidirectional Recurrent Neural Networks (BiRNN / BiLSTM / BiGRU)

Standard Recurrent Neural Networks process sequences strictly in one direction—from left to right (**past to future**). However, in many real-world tasks, understanding the meaning of a word or event requires knowing **future context** (words coming later in the sequence) just as much as past context.

To solve this, **Bidirectional RNNs (BiRNNs)** process sequential data in both directions simultaneously.

---

## 1. Why Bidirectional? (The "Teddy" Problem)

### The Limitation of Unidirectional RNNs
Imagine reading the following two sentences word-by-word:

1. *"He said **Teddy** bears are cute."*
2. *"He said **Teddy** Roosevelt was a great president."*

When a standard unidirectional RNN reaches word #3 (**"Teddy"**):
* It has only seen: *"He said Teddy..."*
* It cannot know whether "Teddy" refers to a **toy bear** or a **US President** until it reads the next word (*"bears"* or *"Roosevelt"*).

```
Unidirectional RNN (Left-to-Right Only):
"He" ──► "said" ──► "Teddy" ──► ??? (Has NO idea what comes next!)
```

### The Bidirectional Solution
A **Bidirectional RNN** reads the sentence in **both directions at the same time**:
* **Forward Pass (Left-to-Right):** Sees *"He said..."*
* **Backward Pass (Right-to-Left):** Sees *"...bears are cute"*

By combining information from both directions, the network instantly knows at word #3 (**"Teddy"**) that it is followed by *"bears"*, enabling precise understanding!

---

## 2. Architecture & How It Works

A Bidirectional RNN duplicates the recurrent layer into two independent hidden state tracks:

```
┌───────────────────────────────┬────────────────────────────────────────────────────────┐
│ Layer Track                   │ Processing Direction & Function                        │
├───────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. Forward Layer (h⃗_t)        │ Processes sequence from left to right: t = 1 ──► T      │
│                               │ Captures PAST context.                                 │
│ 2. Backward Layer (h⃖_t)       │ Processes sequence from right to left: t = T ──► 1     │
│                               │ Captures FUTURE context.                               │
└───────────────────────────────┴────────────────────────────────────────────────────────┘
```

At each time step $t$, the outputs from both hidden layers are **concatenated** into a single combined context vector:

$$h_t = \left[ \overrightarrow{h}_t \,;\, \overleftarrow{h}_t \right]$$

---

## 3. Complete Data Flow Diagram of a Bidirectional RNN

```
========================================================================================================================
                                    BIDIRECTIONAL RNN DATA FLOW (UNROLLED)
========================================================================================================================

                                         Prediction ŷ₁           Prediction ŷ₂           Prediction ŷ_T
                                               ▲                       ▲                       ▲
                                               │                       │                       │
                                            (Concat)                (Concat)                (Concat)
                                           [h⃗₁, h⃖₁]               [h⃗₂, h⃖₂]               [h⃗_T, h⃖_T]
                                               ▲                       ▲                       ▲
                                               │                       │                       │
  Backward Layer (Right-to-Left):   h⃖₁ ◄───────────── (W⃖_{hh}) ────── h⃖₂ ◄───────────── (W⃖_{hh}) ────── h⃖_T (Start)
                                       │                       │                       │
                                       ▲                       ▲                       ▲
  Forward Layer  (Left-to-Right):   h⃗₁ (Start) ────── (W⃗_{hh}) ──────► h⃗₂ ────── (W⃗_{hh}) ──────► h⃗_T
                                       ▲                       ▲                       ▲
                                       │ (W⃗_{xh})              │ (W⃗_{xh})              │ (W⃗_{xh})
                                    Input x₁                Input x₂                Input x_T
                                    ("Teddy")               ("bears")               ("cute")

========================================================================================================================
```

---

## 4. Mathematical Equations & Symbol Breakdown

At time step $t$, the two hidden states are computed independently using separate weight matrices:

### 1. Forward Hidden State Equation:
$$\overrightarrow{h}_t = \tanh\left( \overrightarrow{W}_{xh} x_t + \overrightarrow{W}_{hh} \overrightarrow{h}_{t-1} + \vec{b}_h \right)$$

### 2. Backward Hidden State Equation:
$$\overleftarrow{h}_t = \tanh\left( \overleftarrow{W}_{xh} x_t + \overleftarrow{W}_{hh} \overleftarrow{h}_{t+1} + \vec{b}_h \right)$$

### 3. Combined Output Equation:
$$\hat{y}_t = \text{softmax}\left( W_{hy} \left[ \overrightarrow{h}_t \,;\, \overleftarrow{h}_t \right] + b_y \right)$$

---

### Symbol Breakdown Table:

| Symbol | Meaning | Range / Dimension |
| :--- | :--- | :--- |
| **$x_t$** | Input vector at time step $t$ | $[d_x, 1]$ |
| **$\overrightarrow{h}_t$** | Forward hidden state vector (left-to-right context) | $[d_h, 1]$ |
| **$\overleftarrow{h}_t$** | Backward hidden state vector (right-to-left context) | $[d_h, 1]$ |
| **$[\overrightarrow{h}_t \,;\, \overleftarrow{h}_t]$** | Concatenation of forward and backward hidden states | $[2 \cdot d_h, 1]$ |
| **$\overrightarrow{W}_{xh}, \overrightarrow{W}_{hh}$** | Weight matrices for the Forward RNN layer | $[d_h, d_x]$ and $[d_h, d_h]$ |
| **$\overleftarrow{W}_{xh}, \overleftarrow{W}_{hh}$** | Weight matrices for the Backward RNN layer | $[d_h, d_x]$ and $[d_h, d_h]$ |
| **$W_{hy}$** | Weight matrix mapping concatenated hidden state to output | $[d_y, 2 \cdot d_h]$ |
| **$\hat{y}_t$** | Final output prediction vector at time step $t$ | $[d_y, 1]$ |

> **Note on Output Dimensions:** Because the forward state (size $d_h$) and backward state (size $d_h$) are concatenated, the total hidden feature dimension doubles to **$2 \cdot d_h$**!

---

## 5. When to Use vs. When NOT to Use Bidirectional RNNs

```
┌───────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────┐
│ Use Bidirectional RNNs When...                            │ Do NOT Use Bidirectional RNNs When...                     │
├───────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
│ • Full sequence is available at once (Offline Processing) │ • Real-time stream prediction (Causal Online Systems)     │
│ • Text Classification & Sentiment Analysis                │ • Stock Market Price Forecasting (Future data unknown!)   │
│ • Named Entity Recognition (NER) & POS Tagging            │ • Real-time Speech Auto-completion / Live Chatbots       │
│ • Machine Translation Encoding (Reading source sentence)  │ • Audio Signal Streaming Generation                       │
└───────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────┘
```

* **Why?** Real-time streaming applications cannot look into the future because future time steps $x_{t+1}, x_{t+2}$ have not happened yet!

---

## 6. PyTorch Implementation Example

In PyTorch, enabling bidirectionality for `nn.RNN`, `nn.LSTM`, or `nn.GRU` requires setting just one parameter: `bidirectional=True`.

```python
import torch
import torch.nn as nn

# Define hyperparams
batch_size = 32
seq_len = 10
input_dim = 300
hidden_dim = 128
num_classes = 2

# Create a Bidirectional LSTM Layer
bilstm = nn.LSTM(
    input_size=input_dim,
    hidden_size=hidden_dim,
    batch_first=True,
    bidirectional=True  # <-- Enables Forward + Backward layers!
)

# Input Tensor: [Batch=32, SeqLen=10, InputDim=300]
inputs = torch.randn(batch_size, seq_len, input_dim)

# Forward pass through BiLSTM
output, (h_n, c_n) = bilstm(inputs)

# Output shape: [32, 10, 256]  (256 = 2 * hidden_dim because of forward + backward concat!)
print("BiLSTM Output Shape:", output.shape)

# Final Linear Layer to map [2 * hidden_dim] -> num_classes
fc = nn.Linear(hidden_dim * 2, num_classes)
logits = fc(output[:, -1, :])  # Shape: [32, 2]
```

---

## Summary Key Points

- [x] **Bidirectional RNNs** process sequences in both directions: Forward ($t=1 \to T$) and Backward ($t=T \to 1$).
- [x] Solves the missing future context problem (e.g. distinguishing *"Teddy bears"* vs. *"Teddy Roosevelt"*).
- [x] Outputs are concatenated: $h_t = [\overrightarrow{h}_t \,;\, \overleftarrow{h}_t]$, doubling output hidden dimensions to $2 \cdot d_h$.
- [x] Used in offline tasks (NER, Text Classification, Translation), but **cannot be used in real-time causal forecasting** (stock prices).
