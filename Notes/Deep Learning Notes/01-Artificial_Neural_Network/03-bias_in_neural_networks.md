# 03. Bias in Neural Networks

In every artificial neuron, the total input is calculated as $z = w_1 x_1 + w_2 x_2 + \dots + b$. While **weights ($w$)** scale the inputs, the **bias ($b$)** is an extra trainable constant. Why is bias so essential?

---

## 1. What is Bias and Why is it Required?

### The Straight Line Analogy 📐
Recall the equation of a straight line from high school algebra:
$$y = mx + c$$
* $m$: Slope of the line (controlled by **Weights $w$** in neural networks).
* $c$: $y$-intercept (controlled by **Bias $b$** in neural networks).

If you set $c = 0$, the line equation becomes $y = mx$. This line **can rotate** to change its slope, but it is **permanently pinned to pass through the origin $(0,0)$**.

```
    WITHOUT BIAS (y = mx)                      WITH BIAS (y = mx + c)
    Line MUST pass through (0,0)              Line can SHIFT anywhere!
    
           |  /                                       |   /
           | /                                        |  /
    -------+------- (0,0)                      -------+------- (0,0)
          /|                                         /|  
         / |                                        / |
```

### Why Bias is Essential in ANNs
Bias gives neurons the freedom to **shift the activation threshold up or down**, allowing the decision boundary to move anywhere in the coordinate space independent of the input values!

---

## 2. Effect of Removing Bias ($b = 0$)

What happens if you build a neural network without any bias terms ($z = w \cdot x$)?

1. **Forced Through Origin:** The decision boundary is strictly forced to pass through the origin $(0,0)$.
2. **Cannot Output Non-Zero for Zero Inputs:** If all input features are zero ($x_1 = 0, x_2 = 0$), the weighted sum $z = w_1(0) + w_2(0) = 0$. 
   * Without bias, no matter what weight values you set, the output is permanently locked at zero!
3. **Severe Underfitting:** The network lacks the flexibility to model datasets whose natural decision boundaries do not intersect $(0,0)$.

---

## 3. Decision Boundary & Flexibility

Bias provides **flexibility** to the decision boundary.

Consider a binary classification problem where data points of Class A cluster around $(x_1=5, x_2=5)$ and Class B clusters around $(x_1=8, x_2=8)$:

$$\text{Decision Boundary Equation: } w_1 x_1 + w_2 x_2 + b = 0$$

* Changing **Weights ($w$)** rotates the decision line (changing its angle).
* Changing **Bias ($b$)** slides the decision line left/right or up/down (shifting its location).

```
      x₂ ^
         |      Class B  (● ●)
         |       ●   ● 
         |  =================  <-- Decision Boundary shifted by Bias b
         |     ◯   ◯
         |    Class A  (◯ ◯)
         +-------------------------> x₁
```

Without the bias term $b$, the decision line could only tilt around $(0,0)$ and could **never** slide up between Class A and Class B!


