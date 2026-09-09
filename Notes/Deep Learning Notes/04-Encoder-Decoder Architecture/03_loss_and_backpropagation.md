# 03. Loss Calculation and Backpropagation Through Time (BPTT)

To train an Encoder-Decoder network, we must measure the discrepancy between the model's predicted sequence and the true target sequence, then propagate error gradients backward across both the Decoder and Encoder networks using **Backpropagation Through Time (BPTT)**.

---

## 1. Per-Step Loss Calculation

At each decoder time step $t$, the predicted probability distribution $\hat{y}_t$ over the target vocabulary ($V$) is compared against the true one-hot target word $y_t$ using **Categorical Cross-Entropy Loss**:

$$L_t = - \sum_{k=1}^{V} y_{t, k} \log\left( \hat{y}_{t, k} \right) = - \log\left( \hat{y}_{t, \text{true\_class}} \right)$$

* **Intuition:** If the model assigned probability $0.85$ to the correct word *"Comment"*, the loss is:
  $$L_1 = -\log(0.85) \approx 0.1625$$
* If the model assigned a low probability $0.05$ to the correct word, the loss spikes to:
  $$L_1 = -\log(0.05) \approx 2.9957$$

---

## 2. Total Sequence Loss Calculation

The total sequence loss $L$ is computed by taking the average loss across all decoder time steps $t = 1, \dots, T_y$:

$$L = \frac{1}{T_y} \sum_{t=1}^{T_y} L_t$$

### Handling `<PAD>` Tokens (Loss Masking)
In real-world mini-batch training, sentences have varying lengths and are padded with `<PAD>` tokens to create a uniform rectangular tensor:

```
Sentence 1: [ "Comment", "vas",  "tu",   "<EOS>", "<PAD>", "<PAD>" ]
Sentence 2: [ "Je",      "vais", "bien", "merci", "<EOS>", "<PAD>" ]
```

> **Loss Masking Rule:**  
> The loss values computed at `<PAD>` positions are multiplied by **0** (masked out) so that the artificial padding tokens do not contribute to gradients or parameter updates!

---

## 3. Backpropagation Through Time (BPTT Flow)

Gradients flow backward through the entire Seq2Seq architecture in three sequential stages:

```
========================================================================================================================
                              BACKPROPAGATION THROUGH TIME (BPTT) DATA FLOW
========================================================================================================================

                         TOTAL LOSS L = (L₁ + L₂ + L₃ + L₄) / 4
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
   Loss L₁                          Loss L₂                          Loss L_T
        │ (∂L₁/∂ŷ₁)                      │ (∂L₂/∂ŷ₂)                      │ (∂L_T/∂ŷ_T)
        ▼                                ▼                                ▼
   Decoder s₁                       Decoder s₂                       Decoder s_T
        │ ◄──────────────────────────────┤ ◄──────────────────────────────┤ (Stage 1: Backprop through Decoder)
        │
        ▼
   Decoder Initial State s₀
        │
        ▼  (Stage 2: Gradient passes through Context Vector Bridge: ∂L/∂c = ∂L/∂s₀)
   Context Vector c
        │
        ▼
   Encoder Final State h_{T_x}
        │ ◄──────────────────────────────┤ ◄──────────────────────────────┤ (Stage 3: Backprop through Encoder)
   Encoder h₁                       Encoder h₂                       Encoder h_{T_x}
        │                                │                                │
        ▼                                ▼                                ▼
   Word x₁                          Word x₂                          Word x_{T_x}
  ("How")                          ("are")                          ("you")

========================================================================================================================
```

---

## 4. The 3 Backpropagation Stages Explained

### Stage 1: Backpropagation Inside the Decoder
1. Error gradients $\frac{\partial L_t}{\partial \hat{y}_t}$ enter each decoder time step from the loss function.
2. Gradients propagate backward through time across decoder hidden states:
   $$s_{T_y} \longleftarrow s_{T_y-1} \longleftarrow \dots \longleftarrow s_1 \longleftarrow s_0$$
3. This accumulates weight updates for:
   * Output Linear Projection Layer weights: $\frac{\partial L}{\partial W_s}$
   * Decoder recurrent weights: $\frac{\partial L}{\partial W_{\text{dec}}}$

---

### Stage 2: Crossing the Context Vector Bridge
Because the decoder's starting state was initialized directly from the context vector ($s_0 = c$), the accumulated gradient at $s_0$ flows directly into the Context Vector $c$:

$$\frac{\partial L}{\partial c} = \frac{\partial L}{\partial s_0}$$

This gradient vector represents: *"How should the compressed summary of the input sentence change to reduce the decoder's translation errors?"*

---

### Stage 3: Backpropagation Inside the Encoder
1. Because $c = h_{T_x}$, the gradient $\frac{\partial L}{\partial c}$ enters the final hidden state of the encoder ($h_{T_x}$).
2. Gradients propagate backward through time across all encoder time steps:
   $$h_{T_x} \longleftarrow h_{T_x-1} \longleftarrow \dots \longleftarrow h_2 \longleftarrow h_1$$
3. This accumulates weight updates for:
   * Encoder recurrent weights: $\frac{\partial L}{\partial W_{\text{enc}}}$
   * Input Word Embedding weights: $\frac{\partial L}{\partial W_{\text{embed}}}$

---

## Summary Key Points

- [x] **Loss Function:** Categorical Cross-Entropy $L_t = -\log(\hat{y}_{t, \text{true}})$ computed at every decoder step.
- [x] **Loss Masking:** Multiplies `<PAD>` token losses by 0 to prevent padding from affecting gradients.
- [x] **End-to-End BPTT:** Gradients travel through **Decoder** $\to$ **Context Bridge** ($\frac{\partial L}{\partial c}$) $\to$ **Encoder** $\to$ **Input Embeddings**.
- [x] **Joint Optimization:** Both the Encoder and Decoder are trained together simultaneously in an end-to-end manner.
