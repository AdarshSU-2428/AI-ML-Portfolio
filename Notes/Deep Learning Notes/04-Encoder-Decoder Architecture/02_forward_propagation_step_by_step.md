# 02. Forward Propagation: Step-by-Step Walkthrough with an Example

This note walks through the complete mathematical mechanics of **Forward Propagation** in an Encoder-Decoder network using a concrete Machine Translation example.

---

## 1. Example Setup

To understand how data flows through the network, we will follow a complete English-to-French translation:

* **Source Sentence (English):** *"How are you"* ($T_x = 3$)
* **Target Sentence (French):** *"Comment vas tu"* ($T_y = 3$)
* **Target Sequence with Control Tokens:** `["<SOS>", "Comment", "vas", "tu", "<EOS>"]`

---

## 2. Phase 1: The Encoder Forward Pass

### How Raw Words are Fed into the Encoder (One-Hot Encoding & Embeddings)

Neural networks cannot process raw text strings directly. Before a word enters the encoder cell, it goes through a numerical conversion:

1. **Vocabulary Indexing:** Each word in the source vocabulary $V_{\text{src}}$ is mapped to a unique integer index.
2. **One-Hot Encoding Vector ($x_t^{\text{one-hot}}$):** The word is represented as a sparse vector of size $|V_{\text{src}}| \times 1$ where the index of the word is `1` and all other positions are `0`.
3. **Dense Word Embedding ($x_t$):** The one-hot vector is multiplied by a learnable Embedding Matrix $E_{\text{src}}$ to extract a rich, dense embedding vector of dimension $d_x$:
   $$x_t = E_{\text{src}} \cdot x_t^{\text{one-hot}}$$

```
  Raw Word: "How"
     │
     ▼ (Vocabulary Lookup)
  One-Hot Vector x₁^(one-hot): [ 1, 0, 0, 0, ... 0 ]ᵀ  (Size: |V_src| × 1)
     │
     ▼ (Multiply by Embedding Matrix E_src)
  Dense Input Vector x₁:       [ 0.42, -0.85, 0.19, ... ]ᵀ  (Size: d_x × 1)
     │
     ▼
  Enters Encoder RNN Cell!
```

---

### Step-by-Step Encoder State Updates

The Encoder network reads the embedded vectors $x_t$ one word at a time from left to right. At each time step $t$, it updates its internal hidden state $h_t$:

$$h_t = \text{RNN}_{\text{enc}}\left( x_t, h_{t-1} \right) = \tanh\left( W_{xh} x_t + W_{hh} h_{t-1} + b_h \right)$$

```
========================================================================================================================
                                         ENCODER FORWARD PROCESSING
========================================================================================================================

  Step 1 (t = 1):
    • Input Word:    "How" ──► One-Hot [1, 0, 0]ᵀ ──► Embedded Vector x₁
    • Initial State: h₀ = 0 (Zero Vector)
    • Hidden State:  h₁ = RNN_enc(x₁, h₀)  ──► Stores meaning of "How"

  Step 2 (t = 2):
    • Input Word:    "are" ──► One-Hot [0, 1, 0]ᵀ ──► Embedded Vector x₂
    • Previous State:h₁
    • Hidden State:  h₂ = RNN_enc(x₂, h₁)  ──► Stores meaning of "How are"

  Step 3 (t = 3):
    • Input Word:    "you" ──► One-Hot [0, 0, 1]ᵀ ──► Embedded Vector x₃
    • Previous State:h₂
    • Hidden State:  h₃ = RNN_enc(x₃, h₂)  ──► Stores meaning of "How are you"

========================================================================================================================
```

---

### Extracting the Context Vector ($c$)

The intermediate hidden states ($h_1, h_2$) are discarded. The final hidden state of the encoder ($h_3$) represents the accumulated semantic summary of the entire input sentence and becomes the **Context Vector**:

$$c = h_{T_x} = h_3$$

---

## 3. Phase 2: Bridging to the Decoder

The Context Vector $c$ is passed to the Decoder to initialize its starting hidden state $s_0$:

$$s_0 = c$$

> **Note for LSTM Models:** If the model uses LSTMs instead of basic RNNs, both the final hidden state $h_{T_x}$ and the final cell state $c_{T_x}$ are passed across the bridge:
> $$s_0 = h_{T_x}, \quad C_0^{(\text{dec})} = c_{T_x}$$

---

## 4. Phase 3: The Decoder Forward Pass (Step-by-Step Unrolling)

The Decoder generates the target sequence **one token at a time in an auto-regressive loop**.

At each decoder time step $t$, the decoder:
1. Receives the previous token input $y_{t-1}$ and previous hidden state $s_{t-1}$.
2. Computes the new hidden state:
   $$s_t = \text{RNN}_{\text{dec}}\left( y_{t-1}, s_{t-1} \right)$$
3. Projects the hidden state $s_t$ through a Linear layer ($W_s$) and Softmax to produce vocabulary class probabilities:
   $$P(y_t \mid y_{<t}, c) = \text{softmax}\left( W_s s_t + b_s \right)$$
4. Selects the word with the highest probability: $\hat{y}_t = \arg\max P(y_t)$.

```
========================================================================================================================
                                    DECODER STEP-BY-STEP GENERATION FLOW
========================================================================================================================

  Time Step t = 1:
    • Input Token:    y₀ = "<SOS>" (Start Token)
    • Hidden State:   s₀ = c (Context Vector from Encoder)
    • Update State:   s₁ = RNN_dec("<SOS>", s₀)
    • Output Logits:  P(y₁) = softmax(W_s · s₁ + b_s)
    • Predicted Word: ŷ₁ = "Comment"

  Time Step t = 2:
    • Input Token:    y₁ = "Comment"
    • Hidden State:   s₁
    • Update State:   s₂ = RNN_dec("Comment", s₁)
    • Output Logits:  P(y₂) = softmax(W_s · s₂ + b_s)
    • Predicted Word: ŷ₂ = "vas"

  Time Step t = 3:
    • Input Token:    y₂ = "vas"
    • Hidden State:   s₂
    • Update State:   s₃ = RNN_dec("vas", s₂)
    • Output Logits:  P(y₃) = softmax(W_s · s₃ + b_s)
    • Predicted Word: ŷ₃ = "tu"

  Time Step t = 4:
    • Input Token:    y₃ = "tu"
    • Hidden State:   s₃
    • Update State:   s₄ = RNN_dec("tu", s₃)
    • Output Logits:  P(y₄) = softmax(W_s · s₄ + b_s)
    • Predicted Word: ŷ₄ = "<EOS>"  ──► STOP GENERATION! (Translation Finished)

========================================================================================================================
```

---

## 5. The Role of Teacher Forcing in Forward Propagation

During decoder training, a major challenge arises: **What input should we feed to decoder step $t$?**

Should we feed the model's **own predicted word from step $t-1$**, or should we feed the **true ground-truth target word**? This technique is known as **Teacher Forcing** (Williams & Zipser, 1989).

---

### The Classroom Analogy 👨‍🏫
Imagine a young child learning to read a full sentence aloud: *"The cat sat on the mat"*.

* **Without a Teacher (Free Running):** If the child misreads word #2 as *"car"* (*"The car..."*), they will continue inventing a completely nonsensical sentence (*"The car drove on the road"*). The entire remainder of the sentence is ruined, and the child learns nothing about words #3, #4, and #5.
* **With a Teacher (Teacher Forcing):** When the child says *"car"*, the teacher immediately steps in: *"No, the word is 'cat'. Now read the next word."* The child gets back on track and successfully learns to read *"sat"*, *"on"*, *"the"*, *"mat"*.

---

### How Teacher Forcing Works in Seq2Seq Training

```
========================================================================================================================
                          WITHOUT TEACHER FORCING vs. WITH TEACHER FORCING
========================================================================================================================

1. WITHOUT TEACHER FORCING (Slow & Unstable Training):
   
   Step 1: "<SOS>" ──► Predicts "Pourquoi" (WRONG!)
                                │
                                ▼ (Feeds wrong word into Step 2!)
   Step 2: "Pourquoi" ──► Predicts "est" (Completely derailed from "Comment vas tu"!)
                                │
                                ▼ (Errors cascade exponentially!)
   Step 3: "est" ──────► Predicts nonsense!

------------------------------------------------------------------------------------------------------------------------

2. WITH TEACHER FORCING (Fast & Stable Training):
   
   Step 1: "<SOS>" ──► Predicts "Pourquoi" (Calculate Loss L₁)
                                
   Step 2: TRUE WORD "Comment" ──► Predicts "vas" (Calculate Loss L₂)  ◄── Teacher forces correct input!
                                
   Step 3: TRUE WORD "vas"     ──► Predicts "tu"  (Calculate Loss L₃)  ◄── Keeps decoder strictly on track!

========================================================================================================================
```

---

### Why Teacher Forcing is Critical for Seq2Seq:

1. **Prevents Error Compounding:** In early training epochs, a random network will almost always output wrong words at step 1. Without teacher forcing, subsequent steps would receive garbage inputs, making it impossible for the decoder to learn long sequences.
2. **Accelerates Convergence:** The model learns correct grammar and token transitions much faster because each step is trained under optimal conditions.
3. **Enables Parallel Forward Pass during Training:** Because all ground-truth input tokens ($y_0^*, y_1^*, \dots, y_{T-1}^*$) are known in advance, the entire target sequence can be processed in a single vectorized pass!

---

### The Dilemma: Exposure Bias & Scheduled Sampling

While Teacher Forcing makes training fast, it introduces a problem called **Exposure Bias**:

```
┌───────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────┐
│ Training Phase (With Teacher)                             │ Inference Phase (Without Teacher in Real Life)            │
├───────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
│ The model ALWAYS receives 100% perfect ground-truth       │ The model MUST feed its OWN potentially flawed            │
│ inputs from the teacher at every step.                    │ predictions back into itself.                             │
│                                                           │                                                           │
│ • Result: The model becomes dependent on a "crutch" and   │ • Problem: If the model makes a single mistake at step 1, │
│   never learns how to recover from its own mistakes!      │   it has never practiced recovering from flawed inputs!   │
└───────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────┘
```

#### The Solution: Scheduled Sampling (Bengio et al., 2015)
To fix exposure bias, models use **Scheduled Sampling**:
* **Early Epochs:** Use Teacher Forcing $100\%$ of the time to learn basic language structure.
* **Later Epochs:** Gradually decay the teacher forcing probability (e.g. $100\% \to 70\% \to 30\%$), randomly feeding the model's own predictions so it learns to self-correct during inference!

---

## Summary Key Points

- [x] **Encoder Phase:** Unrolls over $T_x$ steps to compute final context vector $c = h_{T_x}$.
- [x] **Bridge Phase:** Initializes decoder state $s_0 = c$.
- [x] **Decoder Phase:** Auto-regressively generates tokens step-by-step until `<EOS>` is emitted.
- [x] **Role of Teacher Forcing:** Feeds ground-truth tokens $y_{t-1}^*$ during training to prevent error cascading and accelerate convergence.
- [x] **Exposure Bias:** The training-inference gap solved by **Scheduled Sampling** (gradually weaning the model off the teacher).
