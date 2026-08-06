# 08. Sequence Architectures: One-to-Many, Many-to-One, and Many-to-Many

Not all sequential problems are structured the same way. Depending on whether the input, output, or both are sequences, recurrent models fall into **five distinct architectural categories**.

This note provides a complete guide to all sequence mapping architectures, complete with ASCII diagrams, real-world examples, and PyTorch output shape tracking.

---

## 1. Overview of Sequence Mapping Types

```
  1. One-to-One           2. One-to-Many          3. Many-to-One
  ┌───┐ ──► ┌───┐         ┌───┐ ──► ┌───┐ ──► ┌───┐ ──► ┌───┐     ┌───┐ ──► ┌───┐ ──► ┌───┐
  │ x │     │ y │         │ x │     │y₁ │     │y₂ │     │y₃ │     │x₁ │     │x₂ │     │x₃ │
  └───┘     └───┘         └───┘     └───┘     └───┘     └───┘     └───┘     └───┘     └───┘
                                                                    │         │         │
                                                                    └─────────┴─────────┼──► ┌───┐
                                                                                        │    │ y │
                                                                                        ▼    └───┘

  4. Many-to-Many (Synchronous: T_in = T_out)     5. Many-to-Many (Seq2Seq: T_in ≠ T_out)
  ┌───┐ ──► ┌───┐ ──► ┌───┐                       ┌───┐ ──► ┌───┐           ┌───┐ ──► ┌───┐ ──► ┌───┐
  │x₁ │     │x₂ │     │x₃ │                       │x₁ │     │x₂ │           │y₁ │     │y₂ │     │y₃ │
  └───┘     └───┘     └───┘                       └───┘     └───┘           └───┘     └───┘     └───┘
    │         │         │                           │         │               ▲         ▲         ▲
    ▼         ▼         ▼                           └─────────┴──► Encoder ───┴─────────┼─────────┘
  ┌───┐     ┌───┐     ┌───┐                                      Vector (c)
  │y₁ │     │y₂ │     │y₃ │
  └───┘     └───┘     └───┘
```

---

## 2. Type 1: One-to-One (Standard Non-Sequential Baseline)

* **Input:** Single static input vector ($T_{in} = 1$).
* **Output:** Single static output vector ($T_{out} = 1$).
* **Description:** Standard feedforward network (ANN or CNN) without temporal memory. Included as a baseline.

```
Input x ──► [ Feedforward Layer / CNN ] ──► Output y
```

### Real-World Examples:
* **Image Classification:** Input a $224 \times 224$ photo $\longrightarrow$ Output class label *"Cat"*.
* **House Price Prediction:** Input tabular features (bedrooms, area) $\longrightarrow$ Output price estimate.

---

## 3. Type 2: One-to-Many (Sequence Generation)

* **Input:** Single static input vector ($T_{in} = 1$).
* **Output:** Sequence of output vectors over time ($T_{out} > 1$).
* **Description:** Takes a single fixed input and unrolls an RNN decoder to generate a continuous sequence over time.

```
                        ŷ₁                      ŷ₂                      ŷ_T
                        ▲                       ▲                       ▲
                        │                       │                       │
                     ┌─────┐                 ┌─────┐                 ┌─────┐
h₀ ────────────────► │ h₁  │ ──────────────► │ h₂  │ ──────────────► │ h_T │
                     └─────┘                 └─────┘                 └─────┘
                        ▲
                        │
                     Input x
```

### Real-World Examples:
1. **Image Captioning:** Input a single photo $\longrightarrow$ Output description sentence: *"A dog playing in the grass"*.
2. **Music Generation:** Input a single genre prompt (e.g., *"Jazz"*) $\longrightarrow$ Output sequence of musical notes.
3. **Text Generation from Topic:** Input topic keyword (*"Climate"*) $\longrightarrow$ Output full paragraph essay.

---

## 4. Type 3: Many-to-One (Sequence Classification)

* **Input:** Sequence of input vectors over time ($T_{in} > 1$).
* **Output:** Single final output prediction ($T_{out} = 1$).
* **Description:** Processes a sequence element-by-element, accumulating context into the hidden state. Only the **final hidden state ($h_T$)** is passed to the classifier.

```
                                                                        Final Output ŷ
                                                                              ▲
                                                                              │
                     ┌─────┐                 ┌─────┐                 ┌─────┐  │
h₀ ────────────────► │ h₁  │ ──────────────► │ h₂  │ ──────────────► │ h_T │ ─┘
                     └─────┘                 └─────┘                 └─────┘
                        ▲                       ▲                       ▲
                        │                       │                       │
                     Input x₁                Input x₂                Input x_T
                     ("This")                ("movie")               ("great")
```

### Real-World Examples:
1. **Sentiment Analysis:** Input sentence *"This movie was great"* $\longrightarrow$ Output sentiment label: **Positive**.
2. **Spam Detection:** Input email text $\longrightarrow$ Output label: **Spam** / **Not Spam**.
3. **Video Action Recognition:** Input sequence of 30 video frames $\longrightarrow$ Output action label: *"Jumping"*.

### PyTorch Pattern:
```python
output, (h_n, c_n) = lstm(sequence_inputs)
# Extract only the final time step hidden state for classification
final_output = fc(output[:, -1, :])  # Shape: [Batch, NumClasses]
```

---

## 5. Type 4: Many-to-Many (Synchronous / Equal Length)

* **Input:** Sequence of input vectors ($T_{in} > 1$).
* **Output:** Sequence of output vectors of **equal length** ($T_{out} = T_{in}$).
* **Description:** The network outputs a prediction at **every single time step** in sync with the inputs.

```
                       ŷ₁                      ŷ₂                      ŷ_T
                        ▲                       ▲                       ▲
                        │                       │                       │
                     ┌─────┐                 ┌─────┐                 ┌─────┐
h₀ ────────────────► │ h₁  │ ──────────────► │ h₂  │ ──────────────► │ h_T │
                     └─────┘                 └─────┘                 └─────┘
                        ▲                       ▲                       ▲
                        │                       │                       │
                     Input x₁                Input x₂                Input x_T
                     ("Apple")               ("is")                  ("tasty")
```

### Real-World Examples:
1. **Part-of-Speech (POS) Tagging:**  
   * Input: *"Apple is tasty"* (3 words)  
   * Output: *Noun, Verb, Adjective* (3 tags)
2. **Named Entity Recognition (NER):**  
   * Input: *"George visited London"*  
   * Output: *Person, Other, Location*
3. **Video Frame-by-Frame Tracking:** Labeling objects in every frame of a live video stream.

---

## 6. Type 5: Many-to-Many (Asynchronous / Seq2Seq / Encoder-Decoder)

* **Input:** Sequence of input vectors of length $T_{in}$.
* **Output:** Sequence of output vectors of **different length** ($T_{out} \neq T_{in}$).
* **Description:** Consists of two sub-networks:
  1. **Encoder RNN:** Reads the input sequence and compresses it into a single context vector ($c$).
  2. **Decoder RNN:** Takes the context vector ($c$) and generates an output sequence of arbitrary length.

```
========================================================================================
                          ENCODER - DECODER (Seq2Seq) ARCHITECTURE
========================================================================================

    ENCODER (Reads Input Sentence)              DECODER (Generates Output Sentence)
    
        ┌─────┐         ┌─────┐                      ┌─────┐         ┌─────┐
h₀ ───► │ h₁  │ ──────► │ h₂  │ ──► Context Vector ─►│ h₁  │ ──────► │ h₂  │ ──► ...
        └─────┘         └─────┘         (c)          └─────┘         └─────┘
           ▲               ▲                            │               │
           │               │                            ▼               ▼
        x₁ ("I")       x₂ ("love")                   y₁ ("J'aime")   y₂ ("l'IA")

========================================================================================
```

### Real-World Examples:
1. **Machine Translation:**  
   * Input English (5 words): *"How are you today?"*  
   * Output French (4 words): *"Comment allez-vous?"*
2. **Text Summarization:** Input 500-word article $\longrightarrow$ Output 30-word summary.
3. **Conversational Chatbots:** Input user query $\longrightarrow$ Output generated response.

---

## Summary Comparison Matrix

| Architecture Type | Input Length ($T_{in}$) | Output Length ($T_{out}$) | Relationship | Primary Use Cases |
| :--- | :--- | :--- | :--- | :--- |
| **One-to-One** | 1 | 1 | $T_{in} = T_{out} = 1$ | Image Classification, Tabular ML |
| **One-to-Many** | 1 | $> 1$ | $T_{in} < T_{out}$ | Image Captioning, Music Generation |
| **Many-to-One** | $> 1$ | 1 | $T_{in} > T_{out}$ | Sentiment Analysis, Spam Detection |
| **Many-to-Many (Sync)** | $> 1$ | $> 1$ | **$T_{in} = T_{out}$** | POS Tagging, NER, Video Frame Tagging |
| **Many-to-Many (Seq2Seq)**| $> 1$ | $> 1$ | **$T_{in} \neq T_{out}$** | Machine Translation, Text Summarization |
