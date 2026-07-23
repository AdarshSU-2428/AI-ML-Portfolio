# 05. AdaGrad (Adaptive Gradient Algorithm)

Up until now, algorithms like SGD and SGD with Momentum use a **single, global learning rate** for every weight in the network. **AdaGrad** introduced a major innovation: giving **each individual weight its own customized learning rate**.

---

## 1. What is AdaGrad?

**AdaGrad** stands for **Adaptive Gradient Algorithm**. It automatically adapts the learning rate of each parameter based on its past history of gradients.

* Weights that receive **frequent/large updates** will have their learning rates automatically **reduced**.
* Weights that receive **rare/small updates** will maintain a **larger learning rate** so they can catch up!

---

## 2. Historical Gradient Accumulation (How it Works)

AdaGrad keeps track of the sum of squared historical gradients for each weight parameter:

1. **Accumulate Squared Gradients ($G_t$):**
   $$G_t = G_{t-1} + g_t^2$$
   * $g_t$: Current gradient for a specific weight.
   * $G_t$: Cumulative sum of squared gradients from step 1 up to step $t$.

2. **Adapt the Learning Rate:**
   When updating the weight, AdaGrad divides the initial learning rate $\eta$ by the square root of $G_t$:
   $$w_{t+1} = w_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot g_t$$
   * $\epsilon$ (Epsilon): A tiny number (e.g., $10^{-8}$) added to prevent division by zero.

---

## 3. Handling Sparse Features

AdaGrad is uniquely beneficial for dataset features that are **sparse** (infrequent):

### Text Processing & NLP Example 💬
In Natural Language Processing, common words like *"the"*, *"is"*, and *"and"* appear in almost every document (frequent features), producing frequent gradient updates. Rare technical words like *"photosynthesis"* appear very rarely (sparse features).

* **Frequent Features:** $G_t$ becomes large fast $\rightarrow$ Learning rate shrinks $\rightarrow$ Takes small, cautious steps.
* **Sparse Features:** $G_t$ stays small $\rightarrow$ Learning rate stays large $\rightarrow$ Takes bigger steps when those rare features appear!

---

## 4. The Critical Limitation of AdaGrad ⚠️

While adaptive learning rates are brilliant, AdaGrad has one fatal flaw: **The Diminishing Learning Rate Problem**.

Because $G_t$ keeps adding squared gradients ($g_t^2 \ge 0$) at every single step:
1. $G_t$ only **increases** monotonically over time (it never shrinks).
2. As $G_t$ grows larger and larger, $\frac{\eta}{\sqrt{G_t + \epsilon}}$ shrinks closer and closer to **zero**.
3. Eventually, the learning rate becomes so small that the network **completely stops learning** (weights stop updating), even if the model hasn't reached the minimum error yet!

```
Training Time --->
G_t (Sum of squared gradients): 5 -> 25 -> 120 -> 5000 -> 999999 (Keeps growing!)
Effective Learning Rate:        0.1 -> 0.02 -> 0.001 -> 0.0000001 (Shrinks to ZERO!) 🛑
```

---

## Summary Cheat Sheet

| Feature | AdaGrad |
| :--- | :--- |
| **Learning Rate Type** | Per-parameter Adaptive |
| **Gradient Tracking** | Sum of ALL past squared gradients ($G_t = G_{t-1} + g_t^2$) |
| **Best Used For** | Sparse data (e.g., Text classification, NLP) |
| **Major Drawback** | Learning rate continually decays to 0, prematurely freezing training. |
