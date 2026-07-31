# 06. RMSProp (Root Mean Squared Propagation)

**RMSProp** (Root Mean Squared Propagation) was proposed by Geoffrey Hinton to solve the fatal flaw of **AdaGrad**—specifically, the problem of the learning rate continually shrinking to zero.

---

## 1. Why AdaGrad Fails (Quick Recap)

In AdaGrad, the historical gradient accumulator $G_t = G_{t-1} + g_t^2$ sums up **all past gradients** from step 1. Because squared terms are always positive, $G_t$ grows infinitely, driving the effective learning rate down to zero and freezing network training prematurely.

---

## 2. The RMSProp Solution

RMSProp asks a clever question: *Why should gradients from 5,000 steps ago carry the same weight as the gradient from the last step?*

Instead of accumulating *all* past gradients equally, RMSProp uses an **Exponential Moving Average (EMA)** of squared gradients. It gradually **forgets ancient history** and focuses primarily on **recent gradients**!

---

## 3. Exponential Moving Average (EMA) Concept

An Exponential Moving Average gives exponentially higher weight to recent data points while exponentially decaying old ones.

### The Weather Analogy 🌤️
If you want to predict today's temperature, last week's weather matters a lot more than the weather from 6 months ago!

---

## 4. How RMSProp Works (Math & Update Rule)

1. **Calculate Moving Average of Squared Gradients ($v_t$):**
   $$v_t = \beta v_{t-1} + (1 - \beta) g_t^2$$
   * $g_t$: Current gradient for the weight.
   * $v_t$: Exponentially weighted average of squared gradients.
   * $\beta$ (Decay rate / Smoothing factor): Typically set to **`0.9`**.
     * $90\%$ weight is given to previous historical average $v_{t-1}$.
     * $10\%$ weight is given to the new gradient squared $g_t^2$.

2. **Update Weights:**
   $$w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \cdot g_t$$
   * $\eta$: Initial learning rate (e.g., `0.001`).
   * $\epsilon$: Small constant ($10^{-8}$) to avoid division by zero.

---

## 5. Why RMSProp Works So Well

* **Prevents Learning Rate Decay:** Because $v_t$ is a moving average (not an endless sum), $v_t$ does NOT grow indefinitely. The learning rate remains active throughout training!
* **Handles Oscillations:** If a gradient is oscillating wildly (large $g_t^2$), $v_t$ increases, which **reduces the step size**. If a gradient is smooth and small, $v_t$ decreases, **scaling up the step size**.
* **Excellent for Non-Stationary Environments:** Highly effective for Recurrent Neural Networks (RNNs) and complex deep networks.

---

## AdaGrad vs RMSProp Comparison

| Feature | AdaGrad | RMSProp |
| :--- | :--- | :--- |
| **Gradient Memory** | Sums ALL past squared gradients. | Exponential Moving Average of squared gradients. |
| **History Weighting** | Equal weight to old and new gradients. | High weight to recent gradients; old ones decay. |
| **Learning Rate Trend** | Shrinks to 0 (freezes training). | Remains flexible and adaptive throughout. |
| **Performance** | Great for sparse data, poor for deep models. | Excellent default for deep models & RNNs. |
