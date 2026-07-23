# 05. Forward Propagation

**Forward Propagation** is the process by which raw input features travel forward through the layers of a neural network—undergoing weighted sums and non-linear activation transformations—to produce a final prediction.

---

## 1. Flow of Information

In a standard feedforward neural network, information travels strictly in **one direction**:

```
[ Input Layer X ] ───> [ Hidden Layer 1 ] ───> [ Hidden Layer 2 ] ───> [ Output Layer ] ───> Prediction (ŷ)
```

There are no loops or backward jumps during forward propagation. Data enters at the Input Layer and exits at the Output Layer as a final guess ($\hat{y}$).

---

## 2. Layer-wise Computation (The Core Two Steps)

At every neuron in every hidden and output layer, forward propagation executes two sequential mathematical steps:

```
                          Incoming Activations (a^{[l-1]})
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ 1. Weighted Sum Computation   │
                       │    z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}│
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ 2. Non-Linear Activation      │
                       │    a^{[l]} = g^{[l]}(z^{[l]}) │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                          Outgoing Activation (a^{[l]})
```

### Step 1: Weighted Sum ($z^{[l]}$)
The weighted sum $z^{[l]}$ for layer $l$ is computed by multiplying the weight matrix $W^{[l]}$ by the activations from the previous layer $a^{[l-1]}$ and adding the bias vector $b^{[l]}$:

$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$

*(Note: For the very first hidden layer, $a^{[0]} = X$, the input data vector).*

### Step 2: Activation ($a^{[l]}$)
The weighted sum $z^{[l]}$ is passed through the layer's non-linear activation function $g^{[l]}$:

$$a^{[l]} = g^{[l]}(z^{[l]})$$

The resulting activation vector $a^{[l]}$ then serves as the input to the next layer!

---

## 3. Step-by-Step Worked Numerical Example

Let me show you a mini network with **2 Inputs ($x_1=2, x_2=3$)**, **1 Hidden Layer with 2 Neurons**, and **1 Output Neuron**.

```
Input Layer           Hidden Layer (ReLU)             Output Layer (Sigmoid)

 x₁ = 2 ───────────────> ( h₁ ) ──────────────────────\
        X                                              >─────> Output (ŷ)
 x₂ = 3 ───────────────> ( h₂ ) ──────────────────────/
```

### Step A: Compute Hidden Layer Neurons ($h_1, h_2$)
Suppose weights for Hidden Neuron 1 are $w_{11}=0.5, w_{21}=1.0$, bias $b_1 = 0$:
$$z_1^{[1]} = (0.5 \times 2) + (1.0 \times 3) + 0 = 1.0 + 3.0 = 4.0$$
$$a_1^{[1]} = \text{ReLU}(4.0) = 4.0$$

Suppose weights for Hidden Neuron 2 are $w_{12}=-1.0, w_{22}=0.5$, bias $b_2 = 1$:
$$z_2^{[1]} = (-1.0 \times 2) + (0.5 \times 3) + 1 = -2.0 + 1.5 + 1.0 = 0.5$$
$$a_2^{[1]} = \text{ReLU}(0.5) = 0.5$$

---

### Step B: Compute Output Layer Neuron ($\hat{y}$)
Now use $a_1^{[1]}=4.0$ and $a_2^{[1]}=0.5$ as inputs. Suppose output weights are $w_1^{[2]}=0.2, w_2^{[2]}=-0.4$, bias $b^{[2]} = 0$:
$$z^{[2]} = (0.2 \times 4.0) + (-0.4 \times 0.5) + 0 = 0.8 - 0.2 = 0.6$$
$$\hat{y} = \text{Sigmoid}(0.6) = \frac{1}{1 + e^{-0.6}} \approx 0.6456$$

**Final Prediction:** The network outputs $\hat{y} = 0.6456$ (a 64.56% probability).

---

## 4. Vectorized Notation across Multiple Samples

When processing a full batch of $M$ samples simultaneously:

$$Z^{[1]} = W^{[1]} X + B^{[1]}$$
$$A^{[1]} = g^{[1]}(Z^{[1]})$$
$$Z^{[2]} = W^{[2]} A^{[1]} + B^{[2]}$$
$$\hat{Y} = A^{[2]} = g^{[2]}(Z^{[2]})$$

Vectorization replaces slow `for` loops with lightning-fast GPU matrix operations!

---

## Summary Key Takeaways

* **Forward Propagation** flows from Input $\rightarrow$ Hidden Layers $\rightarrow$ Output Layer.
* Every neuron performs: **(1) Weighted Sum ($z$)** then **(2) Activation ($a$)**.
* Activations $a^{[l]}$ of layer $l$ become inputs to layer $l+1$.
* The output layer produces the final prediction score or probability $\hat{y}$.
