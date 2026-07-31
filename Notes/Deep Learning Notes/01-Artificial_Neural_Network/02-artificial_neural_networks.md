# 02. Artificial Neural Networks (ANN)

An **Artificial Neural Network (ANN)**, often called a **Multi-Layer Perceptron (MLP)**, is a computational model composed of interconnected layers of artificial neurons. ANNs overcome the linear limitations of single perceptrons by learning complex, highly non-linear patterns.

---

## 1. Single Layer vs. Multi-Layer Perceptron (MLP)

* **Single-Layer Neural Network:** Contains only an Input Layer connected directly to an Output Layer (no hidden layers). It can only draw straight decision lines (linear classification) and cannot solve problems like XOR.
* **Multi-Layer Perceptron (MLP):** Contains **one or more Hidden Layers** between the Input and Output layers. By combining non-linear activation functions in hidden layers, MLPs can learn curves, clusters, and arbitrary decision boundaries!

---

## 2. The Three Core Layers of an ANN

An ANN architecture is organized into three distinct types of layers:

```
[ Input Layer ]         [ Hidden Layer(s) ]        [ Output Layer ]
 (Receives Data)       (Extracts Features)       (Makes Prediction)

     ( x₁ ) -------------> ( h₁ ) ---------------> ( y₁ )
     ( x₂ ) -------------> ( h₂ ) ---------------> ( y₂ )
     ( x₃ ) -------------> ( h₃ )
```

### A. Input Layer
* **Role:** The entry point for raw input data.
* **Computation:** Performs **no mathematical transformations or activations**. It simply passes raw feature values to the first hidden layer.
* **Rule:** The number of neurons in the input layer equals the number of features in your dataset (e.g., if predicting house price based on 4 features: Size, Bedrooms, Age, Location $\rightarrow$ 4 Input Neurons).

---

### B. Hidden Layer(s)
* **Role:** The "brain" of the neural network located between input and output layers.
* **Computation:** Each neuron computes a weighted sum of inputs and applies a **non-linear activation function** (like ReLU or Tanh).
* **Why "Hidden"?** Because its intermediate outputs are not directly observed in the external dataset—they are internal feature representations discovered automatically by the network.

---

### C. Output Layer
* **Role:** Produces the final prediction or decision.
* **Computation:** Calculates the final outputs using an activation function tailored to the specific problem type (e.g., Sigmoid for binary output, Softmax for multi-class probabilities, Linear for regression values).
* **Rule:** 
  * Binary Classification $\rightarrow$ 1 neuron.
  * Multi-Class (10 classes) $\rightarrow$ 10 neurons.
  * Regression (1 continuous number) $\rightarrow$ 1 neuron.

---

## 3. Number of Neurons & Layer Counting Rule

> **Standard Convention:** When stating the "number of layers" in a neural network, **do not count the input layer**! A network with 1 Input Layer, 2 Hidden Layers, and 1 Output Layer is called a **3-Layer Neural Network**.

---

## 4. Network Depth vs. Width

The design/shape of a neural network is defined by two key dimensions:

```
          DEEP NETWORK (High Depth)                 WIDE NETWORK (High Width)
          
             Layer 1  Layer 2  Layer 3                   Layer 1
               (o)      (o)      (o)                       (o)
               (o)      (o)      (o)                       (o)
                                                           (o)
                                                           (o)
                                                           (o)
```

* **Network Depth:** Refers to the **number of hidden layers** in the architecture.
  * *Deeper Networks* excel at **hierarchical abstraction** (e.g., Layer 1 detects edges $\rightarrow$ Layer 2 detects shapes $\rightarrow$ Layer 3 detects faces).
* **Network Width:** Refers to the **number of neurons per hidden layer**.
  * *Wider Networks* excel at memorizing a larger volume of parallel features at the same level of abstraction.

---

## 5. Universal Approximation Theorem (Introduction)

The **Universal Approximation Theorem** is a fundamental mathematical theorem in Deep Learning proved by George Cybenko (1989) and Kurt Hornik (1991).

### What the Theorem States:
> A feedforward neural network with a **single hidden layer** containing a finite number of neurons, using non-linear activation functions, can approximate **any continuous mathematical function** to arbitrary accuracy!

### Why is this significant?
It proves that Artificial Neural Networks are **universal function approximators**. Given enough neurons and non-linear activations, an ANN can represent any complex relationship between inputs and outputs—whether predicting stock prices, translating languages, or recognizing objects in images!
