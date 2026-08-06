# 📘 Recurrent Neural Networks (RNN, LSTM & GRU) — Complete Implementation & Benchmark Notes

---

## 📌 Executive Summary

This document serves as a comprehensive technical reference and experimental benchmark for Recurrent Neural Network (RNN) architectures—**Simple RNN (`nn.RNN`)**, **LSTM (`nn.LSTM`)**, and **GRU (`nn.GRU`)**—implemented using PyTorch:
1. **Synthetic Sentiment Pipeline (`sentiments.csv`)**: Evaluated on 3,000 balanced text sentences across 3 sentiment classes (`positive`, `negative`, `neutral`), comparing baseline Simple RNN vs. regularized LSTM & GRU models with **Global Max Pooling**.
2. **Full IMDb Movie Reviews Pipeline (`compressed_data.csv.gz`)**: Production NLP pipeline evaluated on 50,000 real-world movie reviews (40,000 training, 10,000 unseen test samples) across binary classification (`positive`, `negative`), analyzing sequence truncation (`MAX_LEN = 150`), vocabulary capping (top 10,000 words), batch size generalization, and early stopping convergence.

---

## 🧠 Part 1: Dataset Specifications & Model Architectures

### 1.1 Synthetic Sentiment Dataset (`sentiments.csv`)
- **Dataset Size:** 3,000 sentences (1,000 Positive / 1,000 Negative / 1,000 Neutral).
- **Partitioning:** 2,400 Train (80%) / 600 Test (20%) via stratified sampling.
- **Preprocessing & Cleaning:** Lowercasing, regex punctuation stripping (`re.sub(r"[^\w\s]", "", text)`), word-level tokenization.
- **Vocabulary Setup:** `VOCAB_SIZE` $\approx 1,200+$ unique words. Special tokens: `<PAD>=0`, `<UNK>=1`.
- **Sequence Length:** `MAX_LEN` = Dynamic max sequence length across dataset (~15–20 words).

### 1.2 Full IMDb Movie Reviews Dataset (`compressed_data.csv.gz`)
- **Dataset Size:** 50,000 real-world reviews (25,000 Positive / 25,000 Negative).
- **Partitioning:** 40,000 Train (80%) / 10,000 Test (20%) via stratified sampling.
- **Preprocessing & Cleaning:** Lowercasing, HTML `<br />` tag stripping (`re.sub(r"<br\s*/?>", " ", text)`), regex punctuation removal.
- **Vocabulary Setup:** Capped to top **10,000 most frequent words** (`Counter.most_common(10000)`). Special tokens: `<PAD>=0`, `<UNK>=1`.
- **Sequence Length:** Capped at **`MAX_LEN = 150`** words for optimal CPU matrix computation.

### 1.3 Model Architecture Implementations
All three recurrent architectures share a unified PyTorch layer layout:
1. **Embedding Layer:** `nn.Embedding(vocab_size, embed_dim=32, padding_idx=0)` — Maps integer token IDs to 32D continuous vectors.
2. **Dropout Layer:** `nn.Dropout(0.2)` — Regularizes embeddings and hidden states.
3. **Recurrent Layer:**
   - **Simple RNN:** `nn.RNN(embed_dim=32, hidden_dim=32, batch_first=True)`
   - **LSTM:** `nn.LSTM(embed_dim=32, hidden_dim=32, batch_first=True)`
   - **GRU:** `nn.GRU(embed_dim=32, hidden_dim=32, batch_first=True)`
4. **Global Max Pooling Layer:** `torch.max(recurrent_out, dim=1)[0]` — Extracts peak feature activations across timesteps.
5. **Linear Classifier Layer:** `nn.Linear(hidden_dim=32, output_dim)` $\rightarrow$ Output Logits (`output_dim=3` for Synthetic, `output_dim=2` for IMDb).

---

## 📊 Part 2: Performance Benchmarks & Results Comparison

### 2.1 Synthetic Dataset Benchmark Comparison (`sentiments.csv` — 3,000 Samples)

| Metric / Architecture | Simple RNN Model | LSTM Model | GRU Model |
| :--- | :---: | :---: | :---: |
| **Recurrent Layer** | `nn.RNN` | `nn.LSTM` | `nn.GRU` |
| **Gating Mechanism** | None (1 State Vector) | 3 Gates ($i_t, f_t, o_t$) + Cell State ($c_t$) | 2 Gates ($r_t, z_t$) |
| **Epochs Trained** | 20 | 10 | 20 |
| **Training Accuracy** | 99.96% | 99.64% | 100.00% |
| **Test Accuracy (600 Samples)** | **100.00%** | **100.00%** | **100.00%** |
| **Real-Time Inference (20 Test Sentences)** | 19 / 20 (95.0%) | **20 / 20 (100.0%)** 🏆 | 19 / 20 (95.0%) |
| **Key Advantage** | Simple & lightweight | Perfect long-term memory | Fast execution & high precision |

---

### 2.2 Full IMDb Movie Reviews Benchmark Comparison (`compressed_data.csv.gz` — 50,000 Samples)

| Metric / Benchmark | Simple RNN Model | LSTM Model | GRU Model |
| :--- | :---: | :---: | :---: |
| **Dataset Size** | 50,000 Reviews | 50,000 Reviews | 50,000 Reviews |
| **Test Set Size** | 10,000 Unseen Reviews | 10,000 Unseen Reviews | 10,000 Unseen Reviews |
| **Batch Size (`BATCH_SIZE`)** | 64 | 64 | 64 |
| **Optimal Stopping Epoch** | 10 Epochs | 5 Epochs | **5 Epochs** 🏆 |
| **Train Accuracy at Peak** | 89.99% | 88.21% | **89.30%** |
| **Test Accuracy at Peak** | **86.47%** | **86.13%** | **86.81%** 🏆 |
| **Train-Test Generalization Gap** | 3.52% | **2.08%** (Tighter) | **2.49%** (Ideal) |
| **Real-Time Inference Score** | 19 / 20 (95.0%) | **20 / 20 (100.0%)** | **20 / 20 (100.0%)** 🎯 |

---

## 🔬 Part 3: Technical Deep Dive & Key Engineering Takeaways

### 3.1 Data-Centric AI & Spurious Correlation Elimination
- **The Toy Dataset Bias Problem:** In small datasets (100 rows), words like *"extremely"* or *"service"* appear exclusively in positive or negative templates. Neural networks learn these spurious shortcuts (e.g., `extremely` $\rightarrow$ `NEGATIVE`), leading to high-confidence misclassifications.
- **Cross-Context Vocabulary Balancing:** Ensuring words and intensifiers appear across all sentiment categories forces the model to learn true semantic relationships rather than keyword shortcuts.

### 3.2 Role of Global Max Pooling (`torch.max(out, dim=1)`)
- Standard RNNs extract only the last timestep (`out[:, -1, :]`), which suffers from memory decay over long sequences.
- **Global Max Pooling** scans all timesteps and retains the maximum activation value for each hidden feature channel. If a strong sentiment word (e.g., *"masterpiece"* or *"terrible"*) occurs anywhere in a 150-word sentence, its signal is preserved into the classification layer.

### 3.3 Early Stopping & Bias-Variance Tradeoff
- **Over-Memorization Onset:** In 50,000 IMDb reviews, training past 5 epochs for LSTM/GRU pushes Train Accuracy from 88% to 94%, but Test Accuracy plateaus at ~86.5%, doubling the Train-Test gap from 2.49% to 7.71%.
- **Optimal Early Stopping Point:** Epoch 5 strikes the perfect balance of maximum generalization and minimal overfitting.

### 3.4 Batch Size Generalization Gap (`64` vs `256`)
- **Stochastic Noise Benefit:** Small-to-medium batch sizes (`BATCH_SIZE = 64`) introduce healthy stochastic noise during gradient descent, helping Adam escape sharp local minima.
- **Large Batch Over-smoothing:** `BATCH_SIZE = 256` averages out gradient noise too aggressively, trapping the optimizer in sharp minima and reducing test accuracy.

### 3.5 Recurrent Layer Comparison: Simple RNN vs. LSTM vs. GRU
- **Simple RNN (`nn.RNN`):** Single hidden state $h_t$. Fastest per epoch, but susceptible to vanishing/exploding gradients.
- **GRU (`nn.GRU`):** Two gates (Reset $r_t$ and Update $z_t$). Solves vanishing gradients, trains 25% faster than LSTM, and achieved the highest test accuracy (**86.81%**).
- **LSTM (`nn.LSTM`):** Three gates (Input $i_t$, Forget $f_t$, Output $o_t$) and dedicated Cell State $c_t$. Provides superior long-range context preservation, reaching 100% real-time prediction accuracy.
