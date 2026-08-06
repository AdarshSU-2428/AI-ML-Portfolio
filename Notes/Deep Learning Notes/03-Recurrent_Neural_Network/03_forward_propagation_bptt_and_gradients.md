# 03. Forward Propagation, BPTT, and Vanishing Gradients in RNNs

This note covers the complete mathematical mechanics of Recurrent Neural Networks: **Forward Propagation** equations, **Complete Forward & Backward Data Flows**, **Backpropagation Through Time (BPTT)** with a detailed **Symbol Breakdown**, and the **Vanishing & Exploding Gradient Problem**.

---

## 1. Forward Propagation in an RNN

Forward propagation in an RNN computes hidden states and output predictions step-by-step from time $t = 1$ to $t = T$.

### Forward Propagation Data Flow Diagram

```
========================================================================================
                          FORWARD PROPAGATION DATA FLOW (t = 1 to T)
========================================================================================

  Time Step t=1                 Time Step t=2                      Time Step t=T
  
    Ground Truth y₁               Ground Truth y₂                    Ground Truth y_T
          │                             │                                  │
          ▼ (Compute L₁)                ▼ (Compute L₂)                     ▼ (Compute L_T)
       Loss L₁                       Loss L₂                            Loss L_T
          ▲                             ▲                                  ▲
          │                             │                                  │
      Prediction ŷ₁                 Prediction ŷ₂                      Prediction ŷ_T
          ▲                             ▲                                  ▲
          │ (W_{hy})                    │ (W_{hy})                         │ (W_{hy})
       ┌─────┐                       ┌─────┐                            ┌─────┐
h₀ ──► │ h₁  │ ─────(W_{hh})───────► │ h₂  │ ────────...──────────────► │ h_T │
(0)    └─────┘                       └─────┘                            └─────┘
          ▲                             ▲                                  ▲
          │ (W_{xh})                    │ (W_{xh})                         │ (W_{xh})
       Input x₁                      Input x₂                           Input x_T
  ("The")                       ("cat")                            ("sat")

                                                                           │
                                                                           ▼
                                                                  Sum Losses L = ∑ L_t
```

---

### The Two Core Equations of an RNN

At every time step $t$, the RNN executes two sequential equations:

#### Equation 1: Hidden State Update Equation

$$h_t = \tanh\left( W_{hh} h_{t-1} + W_{xh} x_t + b_h \right)$$

#### Equation 2: Output Prediction Equation

$$\hat{y}_t = \text{softmax}\left( W_{hy} h_t + b_y \right) \quad \text{or} \quad \hat{y}_t = W_{hy} h_t + b_y$$

---

### Detailed Variable & Matrix Dimension Breakdown

```
┌───────────┬───────────────────────────────────────────┬───────────────────┐
│ Variable  │ Description                               │ Matrix Dimension  │
├───────────┼───────────────────────────────────────────┼───────────────────┤
│ x_t       │ Input vector at time step t               │ [d_x, 1]          │
│ h_t       │ Current hidden state vector at time t     │ [d_h, 1]          │
│ h_{t-1}   │ Previous hidden state vector at time t-1  │ [d_h, 1]          │
│ W_{xh}    │ Input-to-Hidden weight matrix             │ [d_h, d_x]        │
│ W_{hh}    │ Hidden-to-Hidden (recurrent) weight matrix│ [d_h, d_h]        │
│ W_{hy}    │ Hidden-to-Output weight matrix            │ [d_y, d_h]        │
│ b_h       │ Hidden state bias vector                  │ [d_h, 1]          │
│ b_y       │ Output bias vector                        │ [d_y, 1]          │
│ ŷ_t       │ Output prediction vector at time t        │ [d_y, 1]          │
│ tanh      │ Hyperbolic tangent activation function    │ Element-wise      │
└───────────┴───────────────────────────────────────────┴───────────────────┘
```

Where:
* $d_x =$ Dimension of the input feature vector (e.g., word embedding size = 300).
* $d_h =$ Dimension of the hidden state memory vector (e.g., 128 hidden units).
* $d_y =$ Dimension of the output prediction vector (e.g., number of classes = 10).

---

### Why the $\tanh$ Activation Function?
The Hyperbolic Tangent ($\tanh$) activation function squashes all incoming values strictly into the range **$[-1, +1]$**.

* **Why it matters:** Because $h_t$ is passed repeatedly through a loop over hundreds of steps, using an unbounded activation function like linear or standard ReLU could cause activations to explode exponentially to infinity. $\tanh$ keeps memory values strictly bounded!

---

### Worked Numerical Example

Suppose:
* Input dimension $d_x = 2$, Hidden dimension $d_h = 2$, Output dimension $d_y = 1$.
* Initial memory $h_0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$.
* Current input $x_1 = \begin{bmatrix} 1 \\ 0.5 \end{bmatrix}$.
* Weights:
  $$W_{xh} = \begin{bmatrix} 0.5 & 1.0 \\ -0.5 & 0.2 \end{bmatrix}, \quad W_{hh} = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \end{bmatrix}, \quad b_h = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

#### Step 1: Compute Input Term ($W_{xh} x_1$)
$$W_{xh} x_1 = \begin{bmatrix} 0.5 & 1.0 \\ -0.5 & 0.2 \end{bmatrix} \begin{bmatrix} 1 \\ 0.5 \end{bmatrix} = \begin{bmatrix} (0.5 \times 1) + (1.0 \times 0.5) \\ (-0.5 \times 1) + (0.2 \times 0.5) \end{bmatrix} = \begin{bmatrix} 1.0 \\ -0.4 \end{bmatrix}$$

#### Step 2: Compute Recurrent Memory Term ($W_{hh} h_0$)
$$W_{hh} h_0 = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \end{bmatrix} \begin{bmatrix} 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

#### Step 3: Compute Hidden State $h_1$
$$z_1 = \begin{bmatrix} 1.0 \\ -0.4 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 1.0 \\ -0.4 \end{bmatrix}$$

$$h_1 = \tanh\left( \begin{bmatrix} 1.0 \\ -0.4 \end{bmatrix} \right) \approx \begin{bmatrix} 0.7616 \\ -0.3799 \end{bmatrix}$$

This new vector $h_1 = \begin{bmatrix} 0.7616 \\ -0.3799 \end{bmatrix}$ now serves as the memory carried into step $t = 2$!

---

## 2. Backpropagation Through Time (BPTT)

To train an RNN, we must compute gradients of the loss function with respect to our weight matrices ($W_{xh}, W_{hh}, W_{hy}$) and update them using Gradient Descent.

Because the network is unrolled across time steps, standard backpropagation is extended into **Backpropagation Through Time (BPTT)**.

---

### Backward Propagation Data Flow Diagram

```
========================================================================================
                      BACKPROPAGATION THROUGH TIME (BPTT) DATA FLOW
========================================================================================

                         TOTAL LOSS L = L₁ + L₂ + ... + L_T
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 ▼                       ▼                       ▼
            Loss L₁                 Loss L₂                 Loss L_T
                 │                       │                       │
                 ▼ (∂L₁/∂ŷ₁)              ▼ (∂L₂/∂ŷ₂)              ▼ (∂L_T/∂ŷ_T)
             Error ŷ₁                Error ŷ₂                Error ŷ_T
                 │                       │                       │
                 ├───────────► ∂L/∂W_{hy} ┼───────────► ∂L/∂W_{hy} ┤ ──► Accumulate ∂L/∂W_{hy}
                 │                       │                       │
                 ▼                       ▼                       ▼
         ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
         │ Memory h₁ Grad│ ◄─────│ Memory h₂ Grad│ ◄─────│Memory h_T Grad│  (Gradient flows backward
         └───────┬───────┘ (W_{hh})──────┬───────┘ (W_{hh})──────┬───────┘   through time!)
                 │                       │                       │
                 ├──────► Accumulate     ├──────► Accumulate     ├──────► Accumulate
                 │        ∂L/∂W_{hh} &   │        ∂L/∂W_{hh} &   │        ∂L/∂W_{hh} &
                 │        ∂L/∂W_{xh}     │        ∂L/∂W_{xh}     │        ∂L/∂W_{xh}
                 ▼                       ▼                       ▼
             Input x₁                Input x₂                Input x_T
```

---

### Comprehensive BPTT Symbol & Notation Breakdown Table

Below is a complete reference table explaining **every mathematical symbol** used in BPTT equations so you know exactly what each symbol represents:

```
┌─────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Symbol                  │ Detailed Meaning & Function in BPTT                                    │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ L                       │ Total loss accumulated over the entire sequence of length T.           │
│ L_t                     │ Per-step loss at time t (e.g. Cross-Entropy loss between ŷ_t and y_t). │
│ T                       │ Total sequence length (total number of time steps, e.g. T = 100).      │
│ t                       │ Current time step index (t = 1, 2, ..., T).                            │
│ k                       │ An earlier historical time step index (k = 1, 2, ..., t).              │
│ j                       │ Running product index stepping backward through time from t to k+1.    │
│ y_t                     │ Ground truth target label vector at time step t.                       │
│ ŷ_t                     │ Predicted output probability vector at time step t.                    │
│ h_t                     │ Current hidden state memory vector at time step t.                     │
│ h_{t-1}                 │ Previous hidden state memory vector at time step t-1.                  │
│ h_k                     │ Hidden state memory vector at an earlier historical time step k.       │
│ W_{hy}                  │ Hidden-to-Output weight matrix (maps h_t -> ŷ_t).                      │
│ W_{hh}                  │ Recurrent Hidden-to-Hidden weight matrix (maps h_{t-1} -> h_t).        │
│ W_{xh}                  │ Input-to-Hidden weight matrix (maps x_t -> h_t).                       │
│ ∂L / ∂W_{hy}            │ Total gradient of total loss L with respect to weight matrix W_{hy}.   │
│ ∂L / ∂W_{hh}            │ Total gradient of total loss L with respect to recurrent weight W_{hh}.│
│ ∂L / ∂W_{xh}            │ Total gradient of total loss L with respect to input weight W_{xh}.    │
│ ∂L_t / ∂W_{hh}          │ Gradient of loss at time step t with respect to weight matrix W_{hh}.  │
│ ∂L_t / ∂ŷ_t             │ Error gradient at the prediction output at step t.                     │
│ ∂ŷ_t / ∂h_t             │ Derivative of output prediction with respect to hidden state h_t.      │
│ ∂h_t / ∂h_k             │ Temporal gradient chain measuring how past memory h_k influences h_t.  │
│ ∏_{j=k+1}^t             │ Product symbol: multiplies terms together for all steps j from k+1 to t│
│ ∂h_j / ∂h_{j-1}         │ Jacobian matrix of partial derivatives between two consecutive states. │
└─────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

### BPTT Step 1: Total Loss Accumulation

The network calculates a loss $L_t$ at each time step $t$. The **total loss $L$** across the full sequence of length $T$ is the sum of losses at all time steps:

$$L = \sum_{t=1}^{T} L_t$$

---

### BPTT Step 2: Gradients for Output Weights ($W_{hy}$)

The weight matrix $W_{hy}$ connects hidden state $h_t$ directly to output prediction $\hat{y}_t$ at time $t$. Its gradient depends only on the current time step's state:

$$\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hy}}$$

---

### BPTT Step 3: Gradients for Recurrent Weights ($W_{hh}$) — The Time Chain Rule

Calculating the gradient for the recurrent weight matrix $W_{hh}$ is more complex because $h_t$ depends on $h_{t-1}$, which depends on $h_{t-2}$, all the way back to $h_0$!

Using the multivariable Chain Rule, the gradient of loss at time step $t$ ($L_t$) with respect to $W_{hh}$ is:

$$\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^{t} \frac{\partial L_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial h_k} \cdot \frac{\partial h_k}{\partial W_{hh}}$$

#### Breakdown of the Chain Rule Terms:
1. **$\frac{\partial L_t}{\partial \hat{y}_t}$**: Output prediction error at time step $t$.
2. **$\frac{\partial \hat{y}_t}{\partial h_t}$**: How much changing the hidden state $h_t$ changes the prediction $\hat{y}_t$.
3. **$\frac{\partial h_t}{\partial h_k}$**: **The Temporal Memory Chain!** Measures how much a past memory state at time $k$ ($h_k$) influences the current memory state at time $t$ ($h_t$).
4. **$\frac{\partial h_k}{\partial W_{hh}}$**: How much changing weight matrix $W_{hh}$ affects hidden state $h_k$.

#### Explaining the Temporal Chain Product ($\frac{\partial h_t}{\partial h_k}$):

$$\frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} = \frac{\partial h_t}{\partial h_{t-1}} \cdot \frac{\partial h_{t-1}}{\partial h_{t-2}} \dots \frac{\partial h_{k+1}}{\partial h_k}$$

To compute how historical memory $h_k$ affects current memory $h_t$, we must multiply the step-by-step gradients through every single intermediate time step $j$ from $k+1$ up to $t$!

---

## 3. The Vanishing and Exploding Gradient Problem

The chain product $\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}$ is the mathematical root cause of why standard RNNs fail on long sequences.

### Mathematical Explanation

The Jacobian matrix $\frac{\partial h_j}{\partial h_{j-1}}$ contains the recurrent weight matrix $W_{hh}^T$ multiplied by the derivative of the $\tanh$ activation function:

$$\frac{\partial h_j}{\partial h_{j-1}} = W_{hh}^T \cdot \text{diag}\left( 1 - \tanh^2(z_j) \right)$$

When backpropagating over $N$ time steps (e.g., $N = 100$), this term causes $W_{hh}^T$ to be **multiplied by itself 100 times**:

$$\frac{\partial h_t}{\partial h_1} \propto (W_{hh}^T)^{100}$$

```
                                  (W_{hh}^T)¹⁰⁰
                                       │
         ┌─────────────────────────────┴─────────────────────────────┐
         ▼                                                           ▼
Case 1: Eigenvalues of W_{hh} < 1                           Case 2: Eigenvalues of W_{hh} > 1
(0.9)¹⁰⁰ ≈ 0.0000265                                        (1.5)¹⁰⁰ ≈ 4.06 × 10¹⁷
  │                                                           │
  ▼                                                           ▼
VANISHING GRADIENT PROBLEM                                  EXPLODING GRADIENT PROBLEM
Early layers receive ~0 gradient!                           Gradients become massive!
Network FORGETS long-term dependencies.                     Causes NaN numerical overflow.
```

---

### Practical Impact of Vanishing Gradients

Consider processing a long paragraph:
> *"The **cat**, which ate the fish, the mouse, and slept on the rug all afternoon, **was** hungry."*

To predict the verb *"was"* (singular) at step 100, the network must connect back to the subject *"cat"* (singular) at step 1.

* With **Vanishing Gradients**, the gradient signal from step 100 vanishes before reaching step 1.
* The RNN suffers from **short-term memory loss**—it remembers recent words (*"rug"*, *"afternoon"*), but completely forgets words from 20+ steps ago!

---

## 4. Solutions to Gradient Problems in RNNs

### 1. Solution for Exploding Gradients: Gradient Clipping
If the norm of the gradient vector exceeds a maximum threshold $C$ (e.g., $C = 5.0$), rescales the gradient vector down:

$$\text{If } \|\mathbf{g}\| > C \implies \mathbf{g}_{\text{clipped}} = C \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|}$$

In PyTorch: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)`

---

### 2. Solution for Vanishing Gradients: Orthogonal Weight Initialization
Initialize the recurrent weight matrix $W_{hh}$ as an **Orthogonal Matrix** (where $W^T W = I$, so eigenvalues equal 1.0). This prevents matrix multiplication from shrinking or blowing up gradients during early steps.

---

### 3. The Ultimate Structural Solution: Gated Architectures (LSTMs & GRUs)
To permanently solve vanishing gradients over long sequences, researchers invented **Gated Architectures**:
* **LSTM (Long Short-Term Memory):** Introduces a **Cell State** conveyor belt and 3 memory gates (Forget, Input, Output).
* **GRU (Gated Recurrent Unit):** A streamlined 2-gate variant (Reset and Update gates).

---

## Summary Key Points

- [x] **Forward Prop Equations:** $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$ and $\hat{y}_t = \text{softmax}(W_{hy} h_t + b_y)$.
- [x] **Forward Data Flow:** Input $x_t \to h_t \to \hat{y}_t \to L_t \to L$.
- [x] **Backward Data Flow (BPTT):** Error gradients flow backward through time ($h_T \gets h_{T-1} \gets \dots \gets h_1$) accumulating gradients for shared weights $W_{hh}, W_{xh}, W_{hy}$.
- [x] **BPTT Chain Rule:** Uses $\frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}}$ to propagate gradients across historical steps.
- [x] **Vanishing Gradients:** Repeated matrix multiplication $(W_{hh}^T)^N \to 0$ causes short-term memory loss beyond 10-20 steps.
- [x] **Solutions:** Gradient Clipping (exploding), Orthogonal Initialization, and **LSTMs / GRUs** (vanishing).
