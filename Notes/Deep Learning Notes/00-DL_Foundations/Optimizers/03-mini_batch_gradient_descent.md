# 03. Mini-Batch Gradient Descent

**Mini-Batch Gradient Descent** is the gold standard and default optimization approach used across almost all modern Deep Learning frameworks (PyTorch, TensorFlow, etc.). It combines the best features of both **Batch Gradient Descent** and **Stochastic Gradient Descent (SGD)**.

---

## 1. What is Mini-Batch Gradient Descent?

Instead of taking the entire dataset at once (Batch GD) or taking 1 sample at a time (SGD), Mini-Batch GD splits the dataset into **small subsets called mini-batches** (typically containing 32, 64, 128, or 256 samples).

### The Study Group Analogy 👥
* **Batch GD:** Reading the whole library before answering any question (Too slow).
* **SGD:** Asking 1 random person for every single question (Too noisy/unreliable).
* **Mini-Batch GD:** Working in a small team of 32-64 classmates to solve problems together in batches (Fast, efficient, and reliable!).

---

## 2. How it Works (Working Mechanism)

1. **Shuffle Dataset:** Randomly shuffle the dataset at the beginning of each epoch.
2. **Divide into Batches:** Divide the dataset of $N$ samples into mini-batches of size $B$ (e.g., $B = 64$).
3. **Loop Over Batches:** For each mini-batch:
   a. Pass the batch of $B$ samples through the network.
   b. Calculate the average loss across the $B$ samples.
   c. Compute the average gradient for the batch.
   d. Update the weights:
      $$w_{\text{new}} = w_{\text{old}} - \left(\text{Learning Rate} \times \frac{1}{B} \sum_{i=1}^{B} \nabla L_i(w)\right)$$
4. **Repeat:** Continue until all mini-batches in the epoch are processed.

```
Full Dataset (e.g., 10,000 samples)
  |
  +---> [ Mini-Batch 1: 64 samples ] ---> Compute Gradient ---> Update Weights
  +---> [ Mini-Batch 2: 64 samples ] ---> Compute Gradient ---> Update Weights
  +---> [ Mini-Batch 3: 64 samples ] ---> Compute Gradient ---> Update Weights
  ...
```

---

## 3. Batch Size Selection

The **batch size** ($B$) is a critical hyperparameter:

* **Common Sizes:** $16, 32, 64, 128, 256, 512$ (Powers of 2 due to GPU memory architecture).
* **Impact of Batch Size:**
  * **Smaller Batch Size (e.g., 32):** Adds a small amount of helpful noise, improves generalization, uses less memory, but requires more steps per epoch.
  * **Larger Batch Size (e.g., 512):** Smoother gradient, faster epoch completion, but consumes more GPU memory and can lead to slight degradation in generalization performance.

---

## 4. GPU Efficiency & Hardware Parallelism

Why is Mini-Batch GD so popular in modern AI?

Modern graphics processors (GPUs) and Tensor Processing Units (TPUs) contain thousands of tiny computing cores designed specifically to execute **matrix algebra in parallel**.

* **Processing 1 sample (SGD):** GPU cores sit mostly idle (massive waste of hardware).
* **Processing 64 or 128 samples (Mini-Batch):** GPU loads data as a multi-dimensional matrix (tensor) and calculates predictions for all 64 images **simultaneously in a single clock cycle**.

```
Single Sample (SGD):     [ Core 1 active ] (Cores 2-1000 idle...) ❌
Mini-Batch (Size 64):    [ Cores 1 to 64 all working in parallel ] ✅
```

---

## 5. Key Advantages

1. **Best of Both Worlds:** Combines the computational stability of Batch GD with the speed and update frequency of SGD.
2. **Optimal GPU Utilization:** Maximizes hardware vectorization and parallel computing power.
3. **Smooth yet Flexible Convergence:** The loss curve is significantly smoother than pure SGD while still retaining enough variation to hop out of local minima.
4. **Memory Scalable:** Easily fits into GPU VRAM by choosing an appropriate batch size (e.g., 32 or 64).

---

## Spectrum Comparison of Optimizers

```
Pure SGD                      Mini-Batch GD                      Batch GD
(Batch Size = 1)          (Batch Size = 32-256)           (Batch Size = N)
       |                            |                            |
       v                            v                            v
Maximum Noise,               Balanced Speed,               Maximum Memory,
No GPU Efficiency            High GPU Efficiency,          Very Slow Updates,
                             Industry Standard             Smooth Loss
```
