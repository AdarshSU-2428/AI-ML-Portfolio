# 02. Stochastic Gradient Descent (SGD)

**Stochastic Gradient Descent (SGD)** is the exact opposite of Batch Gradient Descent. The word *"stochastic"* means random or probabilistic.

---

## 1. What is Stochastic Gradient Descent?

Instead of waiting to look at the whole dataset, SGD picks **one single training sample at a time** (at random), computes the prediction and loss for that single sample, and immediately updates the network weights.

### The Quiz Analogy 📝
Imagine a student taking a quiz:
* **Batch GD:** Solves all 100 questions, waits for the teacher to grade the whole paper, and then reflects on errors.
* **SGD:** Solves question #1, checks the answer key, immediately adjusts their strategy, solves question #2, checks the key, adjusts again, and repeats for all 100 questions!

---

## 2. How it Works (Working Mechanism)

1. **Shuffle Dataset:** Randomly shuffle the training data before each epoch.
2. **Loop Over Each Sample:** For each sample $(x_i, y_i)$ in the dataset:
   a. Compute the network prediction $\hat{y}_i$.
   b. Calculate the loss for just that one sample $L_i$.
   c. Compute the gradient of $L_i$ with respect to weights.
   d. **Immediately update weights:**
      $$w_{\text{new}} = w_{\text{old}} - (\text{Learning Rate} \times \nabla L_i(w))$$
3. **Repeat:** Do this for all $N$ individual samples in one epoch ($N$ updates per epoch).

```
Loop for i = 1 to N:
  [ Single Sample (x_i, y_i) ] 
               |
               v
  [ Compute Prediction & Loss ]
               |
               v
  [ Update Weights Immediately! ] (Happens N times per epoch)
```

---

## 3. Advantages

* **Frequent & Fast Updates:** Weights are updated $N$ times per epoch. The model starts learning immediately!
* **Low Memory Footprint:** Only 1 sample needs to be loaded into memory at any instant, so it works even on systems with very low RAM.
* **Escapes Local Minima & Saddle Points:** Because predictions for individual samples vary wildly, the loss curve fluctuates noisily. This "noise" helps push the optimizer out of bad local minima or flat saddle points.
* **Supports Online Learning:** Ideal for streaming data where new data arrives continuously, allowing the model to adapt in real time.

---

## 4. Limitations

* **Noisy / Zig-Zag Loss Curve:** The loss curve bounces around wildly because single samples can contain outliers or noise.
* **Never Settle Perfectly:** Due to continuous noise, pure SGD tends to bounce around near the minimum rather than settling down smoothly into the exact center.
* **Inconsistent Direction:** Individual samples might point in conflicting directions, causing unnecessary zig-zagging.
* **Poor Hardware (GPU) Efficiency:** GPUs are optimized for matrix operations on large blocks of data. Processing 1 sample at a time fails to utilize GPU parallel hardware efficiently.

```
Loss Curve Comparison:

Batch GD (Smooth):         \         /
                            \       /
                             \_____/

SGD (Noisy/Zig-zag):       \/\  /\  /\ /
                              \/\ \/ /
                                \/\/
```

---

## Summary Comparison

| Metric | Stochastic Gradient Descent (SGD) |
| :--- | :--- |
| **Data per Weight Update** | 1 single sample |
| **Updates per Epoch** | $N$ updates (where $N$ = total dataset size) |
| **Loss Curve** | Highly noisy / zig-zagging |
| **Memory Footprint** | Extremely Low |
| **GPU Efficiency** | Very Low |
