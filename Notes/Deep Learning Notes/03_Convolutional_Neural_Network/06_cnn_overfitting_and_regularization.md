# 06. Overfitting and Regularization Techniques in CNNs

As Convolutional Neural Networks grow deeper and incorporate millions of parameters, they become prone to **Overfitting**—memorizing training images rather than learning general visual features. This note explains key regularization techniques used to build robust, generalizable CNNs.

---

## 1. What is Overfitting in CNNs?

**Overfitting** occurs when a CNN achieves extremely high accuracy on the training dataset (e.g., $99\%$), but performs poorly on unseen test data (e.g., $65\%$).

### Why CNNs Overfit:
* **Memorizing Background Noise:** Instead of learning that a "dog" has ears and a snout, the CNN memorizes that dogs appear on green grass backgrounds.
* **Excess Capacity:** Millions of filter weights can easily memorize specific pixel arrangements of a small training set.

```
       UNDERFITTING                        JUST RIGHT                         OVERFITTING
 (Model too simple: High Error)  (Balanced: Generalizes Well)    (Model memorizes training data)
 
         \       /                         \       /                         \/\   /\  /\ /
          \     /                           \     /                            \/\ \/ /
           \___/                             \___/                                \/\/
   Training Error High               Training & Test Error Low         Train Error ~0, Test Error High
```

---

## 2. Data Augmentation

**Data Augmentation** is a technique that artificially expands the size and diversity of a training dataset by creating modified versions of existing images.

### Key Principle:
It teaches the network that an object remains the same category regardless of orientation, scale, or lighting (**Invariance**).

### Common Augmentation Transformations:
* **Horizontal Flip:** Flipping a dog horizontally creates a valid new image of a dog (`RandomHorizontalFlip(p=0.5)`).
* **Random Rotation:** Slightly rotating images ($\pm 15^\circ$) handles camera tilts (`RandomRotation(degrees=15)`).
* **Random Cropping & Resizing:** Zooming into parts of an image forces the model to recognize objects from partial views (`RandomResizedCrop`).
* **Color Jitter:** Slightly tweaking brightness, contrast, saturation, and hue handles different lighting conditions (`ColorJitter`).

```python
import torchvision.transforms as T

# Standard PyTorch Data Augmentation Pipeline
train_transforms = T.Compose([
    T.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## 3. Batch Normalization in CNNs (`nn.BatchNorm2d`)

Introduced by Sergey Ioffe and Christian Szegedy (2015), **Batch Normalization** normalizes the layer activations across the batch dimension for each channel independently.

### How `BatchNorm2d` Works in CNNs:
For an input activation tensor of shape $[N, C, H, W]$:

1. **Calculate Channel Means & Variances:** For each channel $c$, compute mean $\mu_c$ and variance $\sigma_c^2$ across all $N \times H \times W$ values in the batch.
2. **Normalize:**
   $$\hat{x}_c = \frac{x_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}$$
3. **Scale and Shift (Learnable Parameters):**
   $$y_c = \gamma_c \hat{x}_c + \beta_c$$
   * $\gamma_c$ (scale) and $\beta_c$ (shift) allow the network to undo the normalization if necessary.

```
Activation Tensor [N, C, H, W]
  │
  ├──► Channel 1: Compute μ₁, σ₁² across all N×H×W pixels ──► Normalize with γ₁, β₁
  ├──► Channel 2: Compute μ₂, σ₂² across all N×H×W pixels ──► Normalize with γ₂, β₂
  └──► Channel C: Compute μ_c, σ_c² across all N×H×W pixels ──► Normalize with γ_c, β_c
```

### Benefits of Batch Normalization:
* **Faster Convergence:** Allows using significantly higher learning rates.
* **Stabilizes Training:** Reduces *Internal Covariate Shift* (changes in layer input distributions during training).
* **Mild Regularization:** Adds minor batch-level noise during training, reducing reliance on heavy Dropout.

> **Standard CNN Placement:** `Conv2D` $\longrightarrow$ `BatchNorm2d` $\longrightarrow$ `ReLU` $\longrightarrow$ `MaxPool2d`.

---

## 4. Dropout & Spatial Dropout (`nn.Dropout2d`)

**Dropout** randomly deactivates a fraction $p$ (e.g., $p = 0.5$) of neurons during each training step, forcing the network to learn redundant, co-adapted feature paths.

### Standard Dropout (`nn.Dropout`) vs. Spatial Dropout (`nn.Dropout2d`):

```
Standard Dropout (nn.Dropout):                  Spatial Dropout (nn.Dropout2d):
Drops individual random pixels in 2D grids      Drops ENTIRE feature channels ([H x W] grids)

┌───┬───┬───┐    ┌───┬───┬───┐                 ┌───┬───┬───┐    ┌───┬───┬───┐
│ 0 │ 4 │ 0 │    │ 2 │ 0 │ 1 │                 │ 0 │ 0 │ 0 │    │ 2 │ 4 │ 1 │
├───┼───┼───┤    ├───┼───┼───┤                 ├───┼───┼───┤    ├───┼───┼───┤
│ 1 │ 0 │ 3 │    │ 0 │ 5 │ 0 │                 │ 0 │ 0 │ 0 │    │ 0 │ 6 │ 2 │
└───┴───┴───┘    └───┴───┴───┘                 └───┴───┴───┘    └───┴───┴───┘
 (Ineffective: Neighbors fill in values)       (Effective: Forces channel independence)
```

* **Standard `nn.Dropout`:** Dropping individual pixels in a feature map is ineffective because adjacent spatial pixels are highly correlated.
* **Spatial `nn.Dropout2d`:** Drops **entire 2D feature map channels** at random, forcing the CNN not to rely on any single feature map channel.

---

## 5. Weight Decay ($L_2$ Regularization)

**Weight Decay** penalizes large weight magnitudes by adding a squared weight penalty to the loss function:

$$L_{\text{total}} = L_{\text{data}} + \frac{\lambda}{2} \sum w^2$$

* **Why it works:** Large weight values lead to sharp, volatile decision boundaries sensitive to input noise. Weight decay forces weights to stay small and smooth.
* **In PyTorch:** Configured directly inside the optimizer:
  ```python
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
  ```

---

## 6. Learning-Rate Scheduling

Static learning rates often get stuck in sub-optimal plateaus. A **Learning Rate Scheduler** dynamically decreases the learning rate during training.

### Popular Schedulers:
1. **StepLR:** Decays the learning rate by a factor $\gamma$ every $k$ epochs (e.g., multiply LR by $0.1$ every 10 epochs).
2. **ReduceLROnPlateau:** Reduces LR when validation loss stops improving for a specified number of epochs (*patience*).
3. **CosineAnnealingLR:** Smoothly decays the learning rate following a cosine curve down to a minimum value.

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
```

---

## 7. Early Stopping

**Early Stopping** monitors validation loss after every epoch and halts training when validation loss fails to decrease for a specified number of consecutive epochs (*patience*).

```
Loss
  ^
  |  Training Loss (Keeps dropping...)
  |  \
  |   \       Validation Loss (Starts rising -> Overfitting!)
  |    \     /
  |     +---X  <-- STOP TRAINING HERE!
  +---------------------------------------------> Epochs
```

* Prevents wasting compute cycles when the model begins overfitting.
* Saves the best model checkpoint from the epoch with the lowest validation loss.

---

## Summary Cheat Sheet

| Technique | Primary Mechanism | How It Is Applied in PyTorch |
| :--- | :--- | :--- |
| **Data Augmentation** | Transforms images to expand dataset variety | `torchvision.transforms.Compose([...])` |
| **Batch Normalization** | Normalizes channel activations per batch | `nn.BatchNorm2d(num_features=channels)` |
| **Spatial Dropout** | Drops entire feature map channels randomly | `nn.Dropout2d(p=0.2)` |
| **Weight Decay** | Penalizes large weight values ($L_2$ norm) | `optimizer = Adam(..., weight_decay=1e-4)` |
| **LR Scheduler** | Decreases step size as training settles | `scheduler.step(val_loss)` |
| **Early Stopping** | Stops training when val loss stops improving | Custom loop tracking `val_loss` & patience |
