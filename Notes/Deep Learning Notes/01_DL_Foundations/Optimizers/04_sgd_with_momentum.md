# 04. SGD with Momentum

Standard Stochastic Gradient Descent (SGD) works fine, but it frequently suffers from a major issue: **oscillations** (zig-zagging back and forth across steep slopes) which slow down training. **SGD with Momentum** fixes this problem!

---

## 1. Why Momentum?

### The Heavy Ball Rolling Downhill Analogy ⚽
Imagine a heavy bowling ball rolling down a steep, winding hill:
* When the ball rolls downhill, gravity pulls it forward.
* As it gains speed, its **momentum** carries it straight along the main downhill direction, making it less affected by small bumps or cross-wise slopes.
* Standard SGD acts like a lightweight ping-pong ball—every small bump knocks it sideways in a crazy zig-zag.
* **Momentum** gives the weight updates physics-like inertia!

```
Standard SGD (Zig-Zagging):             SGD with Momentum (Smooth & Fast):

      \  /  \  /                             \
       \/    \/                               \________
       /\    /\                                        \_______
      /  \  /  \                                               v
```

---

## 2. The Velocity Concept

To implement momentum, we introduce a new variable called **Velocity ($v$)**, which represents a running history of past gradients.

Instead of updating weights based *only* on the current gradient, we update weights based on **Velocity**:

1. **Calculate Velocity:**
   $$v_t = \gamma v_{t-1} + \eta \nabla L(w)$$
   * $v_t$: Current velocity vector.
   * $v_{t-1}$: Previous step's velocity vector.
   * $\gamma$ (Gamma / Momentum coefficient): Usually set to **`0.9`**. It determines how much memory of past steps to retain.
   * $\eta$: Learning rate.
   * $\nabla L(w)$: Current gradient.

2. **Update Weights:**
   $$w_{\text{new}} = w_{\text{old}} - v_t$$

> **What does $\gamma = 0.9$ mean?**  
> It means $90\%$ of your current step size comes from your momentum (past velocity), and only $10\%$ comes from the immediate gradient slope!

---

## 3. Oscillation Reduction

In many loss landscapes (such as long, narrow valleys or ravines), the slope is extremely steep on the sides, but very gentle along the bottom toward the minimum.

* **Without Momentum:** SGD jumps back and forth between the steep side walls, wasting time moving sideways instead of forward.
* **With Momentum:** 
  * The vertical jumps point in opposite directions (+ and -), so they **cancel each other out** over time.
  * The horizontal steps point in the consistent forward direction, so they **accumulate and build speed**!

```
Vertical Direction:   (+5) + (-4) + (+5) + (-5)  --> Cancels out near 0
Horizontal Direction: (+2) + (+2) + (+3) + (+2)  --> Accumulates to +9 (Fast forward momentum!)
```

---

## 4. Faster Convergence

Because Momentum cancels out cross-wise noise and accelerates along persistent downhill slopes:

* It reaches the minimum in significantly **fewer iterations/epochs**.
* It gains enough speed to easily **roll over small local bumps** or flat plateaus in the loss landscape.
* It accelerates training dramatically in complex neural network architectures.

---

## Summary Key Points

* **Problem Solved:** Reduces wild zig-zagging in steep ravines.
* **Core Idea:** Keeps a moving average of past gradients (**Velocity**).
* **Key Hyperparameter:** $\gamma$ (Momentum factor, typically `0.9`).
* **Result:** Faster convergence and smoother trajectories toward the global minimum.
