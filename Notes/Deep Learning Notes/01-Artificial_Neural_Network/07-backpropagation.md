# 07. Backpropagation

**Backpropagation** (short for *Backward Propagation of Errors*) is the central algorithm that enables multi-layer neural networks to learn. Introduced by David Rumelhart, Geoffrey Hinton, and Ronald Williams in 1986, it calculates how much each weight contributed to the network's prediction error.

---

## 1. What is Backpropagation?

### The Workplace Analogy 🏢
Imagine a corporate company project that fails to meet client expectations:
* **Forward Pass:** The CEO gives orders $\rightarrow$ Managers delegate $\rightarrow$ Workers execute $\rightarrow$ Product delivered to client (Prediction).
* **Client Feedback:** Client is unhappy with the final product (Loss / Error).
* **Backpropagation:** The CEO traces the mistake backward: *Did the final assemblers mess up? Did managers give bad instructions? Did designers make poor specs?*
* **Weight Update:** Everyone adjusts their workflow based on their share of blame so the next product succeeds!

---

## 2. The Chain Rule of Calculus

How does an error at the final output layer calculate the blame for a weight buried deep inside hidden layer 1? 

Through the **Chain Rule of Calculus**!

### Mathematical Intuition
If variable $Y$ depends on $U$, and $U$ depends on $X$, then the rate of change of $Y$ with respect to $X$ is the product of their individual rates of change:

$$\frac{\partial Y}{\partial X} = \frac{\partial Y}{\partial U} \times \frac{\partial U}{\partial X}$$

### Chain Rule in a Neural Neuron

```
  Weight (w) ───> Weighted Sum (z) ───> Activation (a) ───> Loss (L)
```

To find how much changing weight $w$ affects total Loss $L$ ($\frac{\partial L}{\partial w}$), we multiply the partial derivatives backward along the chain:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \times \frac{\partial a}{\partial z} \times \frac{\partial z}{\partial w}$$

1. $\frac{\partial L}{\partial a}$: How Loss changes with output activation.
2. $\frac{\partial a}{\partial z}$: Derivative of the activation function at $z$.
3. $\frac{\partial z}{\partial w}$: Derivative of weighted sum with respect to weight ($= x$, the input value).

---

## 3. Error Propagation & Gradient Computation

During Backpropagation, error signals flow backwards layer by layer:

```
[ Input Layer ] <=== [ Hidden Layer 1 ] <=== [ Hidden Layer 2 ] <=== [ Output Layer ] <=== Loss (L)
  (Compute ∂L/∂w¹)      (Compute ∂L/∂w²)      (Compute ∂L/∂w³)     (Compute ∂L/∂ŷ)
```

1. **Start at Output:** Compute error gradient at output $\delta^{[\text{output}]} = \frac{\partial L}{\partial \hat{y}}$.
2. **Propagate Backward:** Pass $\delta$ backward through weights matrix $W^T$ to hidden layers.
3. **Compute Gradients:** Compute gradient $\frac{\partial L}{\partial W^{[l]}}$ and $\frac{\partial L}{\partial b^{[l]}}$ for every layer $l$.

---

## 4. Weight Update Process & The Complete Learning Loop

Once all gradients $\frac{\partial L}{\partial w}$ are computed, the optimizer (e.g., SGD or Adam) updates every weight and bias:

$$w_{\text{new}} = w_{\text{old}} - \left(\text{Learning Rate} \times \frac{\partial L}{\partial w_{\text{old}}}\right)$$

$$b_{\text{new}} = b_{\text{old}} - \left(\text{Learning Rate} \times \frac{\partial L}{\partial b_{\text{old}}}\right)$$

### The Complete Learning Cycle

```
                       ┌─────────────────────────┐
                       │ 1. Forward Propagation  │
                       │    Compute Prediction ŷ │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ 2. Compute Loss         │
                       │    Loss = L(y, ŷ)       │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ 3. Backpropagation      │
                       │    Compute ∂L/∂w via    │
                       │    Chain Rule           │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ 4. Weight Update        │
                       │    w = w - (LR * ∂L/∂w) │
                       └────────────┬────────────┘
                                    │
                                    └──────> (Repeat for next epoch!)
```

This 4-step loop repeats thousands of times until the network's loss reaches a low, stable minimum!
