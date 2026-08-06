# 02. RNN Intuition, Hidden State, and Unrolling Through Time

To solve the limitations of standard ANNs and CNNs, **Recurrent Neural Networks (RNNs)** introduce an internal memory mechanism. This note provides an intuitive, step-by-step breakdown of how RNNs think, maintain memory through the **Hidden State**, and process sequences by **Unrolling Through Time**.

---

## 1. Recurrent Neural Network (RNN) Intuition

### The Human Reading Analogy 📖
Imagine how you read a book:
1. You read word #1 (*"The"*). You remember it.
2. You read word #2 (*"movie"*). You combine it with your memory of word #1 (*"The movie"*).
3. You read word #3 (*"was"*). You combine it with your memory of (*"The movie"*).
4. You read word #4 (*"terrible"*). Your brain combines *"terrible"* with your accumulated memory (*"The movie was"*) to form the final sentiment: **Negative Review**!

> **Key Insight:** You do **not** clear your mind after reading each word! You read word-by-word while continuously updating a **running mental summary** of what you've read so far.

An RNN operates in the exact same way. It processes an input sequence **one element at a time**, passing a summary of past information to the next step.

---

### The Recurrent Feedback Loop
Unlike standard feedforward networks where data flows in a straight line from input to output, an RNN contains a **recurrent feedback loop**:

```
 COMPACT RECURRENT LOOP VIEW:
 
            ┌──────────────┐
            │              │ (Recurrent Loop: Passes memory back to itself)
            ▼              │
   x_t ──► [ RNN Cell ] ───┴──► ŷ_t
           (Memory: h_t)
```

The output of the RNN cell at the current time step ($t$) is fed **back into the cell** as an input for the next time step ($t+1$).

---

## 2. The Hidden State ($h_t$) — The Network's Memory Vector

The secret behind an RNN's memory is a special vector called the **Hidden State** ($h_t$).

### What is the Hidden State ($h_t$)?
The Hidden State $h_t$ is a vector of real numbers that acts as the **working memory** of the neural network at time step $t$. It stores a compressed representation of all past inputs ($x_1, x_2, \dots, x_t$) seen by the network up to that moment.

---

### How $h_t$ is Updated at Every Time Step

At any time step $t$, the RNN receives **two separate inputs**:
1. **$x_t$**: The current external input (e.g., word vector for word #$t$).
2. **$h_{t-1}$**: The previous hidden state (the memory carried over from step $t-1$).

It combines these two inputs to compute the **new hidden state** ($h_t$):

$$h_t = \tanh\left( W_{hh} h_{t-1} + W_{xh} x_t + b_h \right)$$

```
                        Previous Memory (h_{t-1})
                                   │
                                   ▼  (Weighted by W_{hh})
   Current Input (x_t) ──► [ Sum & Activation ] ──► New Memory (h_t)
   (Weighted by W_{xh})             │
                                   ▼
                         Activation Function (tanh)
```

---

### Understanding the Three Weight Matrices

An RNN uses three distinct weight matrices to govern its memory and predictions:

```
┌──────────────┬───────────────────────────────┬────────────────────────────────────────────────────────┐
│ Weight Matrix│ Name                          │ Intuitive Role                                         │
├──────────────┼───────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. W_{xh}    │ Input-to-Hidden Weight        │ Controls how strongly the NEW input (x_t) affects      │
│              │                               │ current memory.                                        │
│ 2. W_{hh}    │ Hidden-to-Hidden Weight       │ Controls how much PAST memory (h_{t-1}) is retained or  │
│              │                               │ transformed into new memory.                           │
│ 3. W_{hy}    │ Hidden-to-Output Weight       │ Controls how the current memory (h_t) is converted     │
│              │                               │ into a final prediction (ŷ_t).                        │
└──────────────┴───────────────────────────────┴────────────────────────────────────────────────────────┘
```

#### Bias Vectors:
* $b_h$: Bias added when computing the hidden state $h_t$.
* $b_y$: Bias added when computing the output prediction $\hat{y}_t$.

---

## 3. Unrolling (Unfolding) Through Time

To understand how an RNN processes a full sequence, we **unroll** (or unfold) the recurrent loop across time steps $t = 1, 2, \dots, T$.

### Visualizing an Unrolled RNN

Suppose we feed a 3-word sentence: *"AI is amazing"* into an RNN. Here is what the network looks like when unrolled over 3 time steps:

```
UNROLLED RNN OVER TIME (t = 1 to t = 3):

     Time t = 1                    Time t = 2                    Time t = 3
     
       ŷ₁                            ŷ₂                            ŷ₃
       ▲                             ▲                             ▲
       │ (W_{hy})                    │ (W_{hy})                    │ (W_{hy})
    ┌─────┐                       ┌─────┐                       ┌─────┐
    │ h₁  │ ─────(W_{hh})───────► │ h₂  │ ─────(W_{hh})───────► │ h₃  │
    └─────┘                       └─────┘                       └─────┘
       ▲                             ▲                             ▲
       │ (W_{xh})                    │ (W_{xh})                    │ (W_{xh})
       x₁                            x₂                            x₃
  ("AI")                        ("is")                        ("amazing")
```

### What Happens Step-by-Step?

1. **At Step $t = 1$ (Input $x_1 =$ *"AI"*):**
   * The network starts with an initial zero memory ($h_0 = \mathbf{0}$).
   * Computes hidden state $h_1$ based on $x_1$ and $h_0$.
   * $h_1$ captures context of *"AI"*.

2. **At Step $t = 2$ (Input $x_2 =$ *"is"*):**
   * The network receives $x_2$ (*"is"*) AND the memory vector $h_1$ (*"AI"*).
   * Computes hidden state $h_2$ combining both.
   * $h_2$ captures context of *"AI is"*.

3. **At Step $t = 3$ (Input $x_3 =$ *"amazing"*):**
   * The network receives $x_3$ (*"amazing"*) AND the memory vector $h_2$ (*"AI is"*).
   * Computes hidden state $h_3$ combining all three words.
   * Produces final output $\hat{y}_3$ predicting sentiment = **Positive**!

---

## 4. The Golden Rule of RNNs: Parameter Sharing Across Time

> **CRITICAL CONCEPT:** Look closely at the unrolled diagram above. Notice that the weight matrices $W_{xh}, W_{hh}, W_{hy}$ and biases $b_h, b_y$ are **the exact same labels at every single time step**!

Unlike a standard neural network that creates new weights for every layer, an RNN reuses the **exact same set of weights ($W_{xh}, W_{hh}, W_{hy}$) at every single time step**.

```
Time Step t=1:  Uses W_{xh}, W_{hh}, W_{hy}
Time Step t=2:  Uses W_{xh}, W_{hh}, W_{hy}  <-- SAME WEIGHTS!
Time Step t=3:  Uses W_{xh}, W_{hh}, W_{hy}  <-- SAME WEIGHTS!
...
Time Step t=T:  Uses W_{xh}, W_{hh}, W_{hy}  <-- SAME WEIGHTS!
```

### Why Parameter Sharing Across Time is Revolutionary:
1. **Fixed Model Size:** Total parameters remain small and constant, whether processing a sequence of 3 words or 3,000 words!
2. **Position Invariance:** A grammatical rule learned at the start of a sentence applies equally at the end of a sentence because the exact same weight matrices process every position.

---

## Summary Key Points

- [x] **RNNs** process sequences element-by-element while maintaining a running memory vector called the **Hidden State ($h_t$)**.
- [x] **$h_t$** is updated using the current input $x_t$ and previous hidden state $h_{t-1}$.
- [x] Three core weight matrices: **$W_{xh}$** (Input effect), **$W_{hh}$** (Memory retention), and **$W_{hy}$** (Output prediction).
- [x] **Unrolling** means expanding the recurrent loop across time steps $t=1, 2, \dots, T$.
- [x] **Parameter Sharing:** The exact same weight matrices ($W_{xh}, W_{hh}, W_{hy}$) are reused at every time step.
