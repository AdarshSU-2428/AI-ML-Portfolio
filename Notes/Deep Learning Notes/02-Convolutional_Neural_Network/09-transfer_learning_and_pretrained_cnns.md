# 09. Transfer Learning and Pre-trained CNNs

Training a deep CNN from scratch requires hundreds of thousands of labeled images, massive GPU resources, and days of computation. **Transfer Learning** bypasses these demands by repurposing a model trained on a massive benchmark dataset (like ImageNet) for a new target task.

---

## 1. What is Transfer Learning?

### The Driver Analogy 🚗
If you already know how to drive a car, learning to drive a truck doesn't require starting from scratch. You reuse basic skills (steering, braking, traffic laws) and only learn truck-specific controls.

In Deep Learning, **Transfer Learning** takes a CNN pre-trained on a large dataset (e.g., ImageNet's 1.4 million images across 1,000 classes) and uses its learned visual feature extractors for a new domain (e.g., classifying medical X-rays or rare plant diseases with only 500 images).

```
PRE-TRAINED SOURCE MODEL (ImageNet)                    TARGET TASK (Medical X-Rays)
┌────────────────────────────────────────┐             ┌────────────────────────────────────────┐
│ Conv Layers (Learned Edges & Shapes)   │  ────────►  │ Conv Layers (REUSED / FROZEN)          │
├────────────────────────────────────────┤             ├────────────────────────────────────────┤
│ FC Head (1,000 ImageNet Classes)       │  ── Replace │ NEW FC Head (2 Classes: Normal/Pneum)  │
└────────────────────────────────────────┘             └────────────────────────────────────────┘
```

---

## 2. Why Pre-trained Features are Universal

As established in early notes, early and middle convolutional layers learn **generic visual primitives**:

* **Layer 1-2:** Detect vertical/horizontal edges, color blobs, and simple gradients.
* **Layer 3-4:** Detect textures, corners, grid patterns, and simple shapes.

These early visual primitives are **identical** whether you are classifying dogs, cars, or lung X-rays! Only the final deep layers and classification head are task-specific.

---

## 3. Two Core Strategies: Feature Extraction vs. Fine-Tuning

Depending on your target dataset size and similarity to the original dataset, you choose between two strategies:

### Strategy 1: Feature Extraction (Freezing Backbone)
* **How it Works:** Freeze all weights in the convolutional feature extractor (`param.requires_grad = False`). Replace the final Linear classification head with a new randomly initialized classifier, and train **only the new classifier head**.
* **Best Used For:** **Small target datasets** that are similar to the pre-training domain.
* **Training Speed:** Extremely fast (requires computing gradients for only a few thousand parameters).

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# 1. Load pre-trained ResNet-18
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# 2. Freeze all convolutional backbone parameters
for param in model.parameters():
    param.requires_grad = False

# 3. Replace final FC layer (ImageNet 1000 outputs -> New 2 outputs)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)  # Only model.fc has requires_grad = True!
```

---

### Strategy 2: Fine-Tuning (Unfreezing Backbone)
* **How it Works:** Unfreeze some or all convolutional layers (`param.requires_grad = True`) and train the entire network end-to-end using a **very small learning rate** (e.g., $\text{LR} = 10^{-5}$).
* **Why Small LR?** Using a large learning rate would overwrite and destroy the valuable pre-trained weights (*Catastrophic Forgetting*).
* **Best Used For:** **Larger target datasets** or domains that differ from the original dataset.

```python
# Unfreeze deep layers (e.g., layer4 and FC) for Fine-Tuning
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Use a small learning rate for fine-tuning pre-trained weights
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},  # Small LR for backbone
    {'params': model.fc.parameters(), 'lr': 1e-3}       # Standard LR for new head
])
```

---

## 4. The Decision Matrix: Choosing the Right Strategy

Use this $2 \times 2$ decision matrix based on your target dataset characteristics:

```
                                TARGET DATASET SIMILARITY TO IMAGENET
                               High Similarity          Low Similarity
                         ┌────────────────────────┬────────────────────────┐
                         │   Feature Extraction   │   Feature Extraction   │
         Small Dataset   │   (Freeze Backbone,    │  (From Earlier Layers  │
                         │   Train FC Head Only)  │   or Retrain Head)     │
TARGET                   ├────────────────────────┼────────────────────────┤
DATASET                  │     Fine-Tuning        │   Train From Scratch   │
SIZE                     │   (Unfreeze Deep       │   or Fine-Tune Entire  │
                         │    Layers with Small   |   Network End-to-End   |
         Large Dataset   |           LR)          │                        │
                         └────────────────────────┴────────────────────────┘
```

### Breakdown of the 4 Scenarios:

1. **Small Dataset + High Similarity (e.g., 200 Cat vs. Dog photos):**
   * *Action:* **Feature Extraction**. Freeze the backbone; train only the new output head. Prevents overfitting on the small dataset.
2. **Large Dataset + High Similarity (e.g., 50,000 car model photos):**
   * *Action:* **Fine-Tuning**. Unfreeze the network and fine-tune with a small LR to adapt features precisely.
3. **Small Dataset + Low Similarity (e.g., 300 Medical Ultrasound scans):**
   * *Action:* **Feature Extraction from Early Layers**. High-level ImageNet features (dog ears) aren't useful, but early features (edges) are. Extract features from early layers or train a small head.
4. **Large Dataset + Low Similarity (e.g., 100,000 Satellite Infrared images):**
   * *Action:* **Train from Scratch or Fine-Tune Full Network**. Since data is abundant, you can train all layers end-to-end.

---

## Summary Cheat Sheet

| Parameter | Feature Extraction | Fine-Tuning |
| :--- | :--- | :--- |
| **Backbone Status** | **Frozen** (`requires_grad = False`) | **Unfrozen** (`requires_grad = True`) |
| **Trained Parameters** | Only the output classifier head ($\sim 0.1\%$) | Deep layers or entire network ($100\%$) |
| **Learning Rate** | Standard ($\text{LR} = 10^{-3}$) | Very Small ($\text{LR} = 10^{-5}$) |
| **Training Time** | Extremely fast (minutes) | Moderate to slow (hours) |
| **Risk of Overfitting** | Very Low | Moderate (if target dataset is small) |
