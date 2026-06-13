# Distorted Visual Sequence Pattern Recognition using Deep Learning

A deep learning OCR pipeline that recognizes text sequences from heavily distorted, noisy grayscale images using a CRNN (CNN + BiLSTM) architecture with CTC decoding.

---

## 🧠 Problem Statement

Real-world text recognition systems often fail when images are unclear, noisy, or intentionally distorted. This project builds a model that can accurately predict ordered character sequences from visually corrupted images containing distortions such as:

- Background noise
- Blur and visual artifacts
- Overlapping symbols
- Shape deformation
- Occlusion and random patches
- Irregular spacing and alignment

---

## 🏗️ Architecture

```
Input Image (200×100 RGB)
        ↓
Grayscale Conversion + CLAHE Enhancement
        ↓
CNN Backbone (VGG-style feature extractor)
        ↓
Reshape → (T, B, C) sequence of feature columns
        ↓
Bidirectional LSTM × 2 layers (256 hidden units)
        ↓
Linear → CTC Decode
        ↓
Predicted Text ("AXU323")
```

### Model Components

| Component | Details |
|---|---|
| **CNN Backbone** | 5 ConvBlocks with BatchNorm + ReLU, MaxPooling to collapse height |
| **Sequence Model** | 2-layer Bidirectional LSTM (256 hidden units per direction) |
| **Decoder** | CTC (Connectionist Temporal Classification) greedy decode |
| **Loss** | `nn.CTCLoss` with `zero_infinity=True` |
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Scheduler** | ReduceLROnPlateau (factor=0.5, patience=3) |

---

## 📁 Dataset

| Split | Size |
|---|---|
| Training images | 20,000 |
| Test images | 5,000 |
| Image size | 200 × 100 (W × H), RGB |
| Label length | 6–9 characters |
| Character set | A–Z + 0–9 (36 classes + 1 CTC blank = 37 total) |

**Dataset structure:**
```
cig_ps/
├── train_images/
│   ├── train-0.png
│   ├── train-1.png
│   └── ...
├── test_images/
│   ├── test-0.png
│   └── ...
└── train-labels.csv   # columns: image, text
```

---

## ⚙️ Preprocessing

- Convert RGB → Grayscale
- Resize to 100×200
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
- Normalize to [-1, 1]

### Data Augmentation (training only)

| Transform | Probability |
|---|---|
| ShiftScaleRotate | 35% |
| GaussNoise | 35% |
| MotionBlur | 15% |
| RandomBrightnessContrast | 30% |
| CoarseDropout (occlusion) | 25% |

---

## 📊 Evaluation Metric

**Character Error Rate (CER)** — based on Levenshtein distance:

```
CER = (Insertions + Deletions + Substitutions) / Length of True Text
```

Lower CER = better performance. A perfect prediction gives CER = 0.

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install torch torchvision albumentations opencv-python pandas scikit-learn tqdm
```

### 2. Train the model
Open `notebook.ipynb` in Kaggle or Jupyter and run all cells top to bottom.

> ⚠️ **Enable GPU** before training: Settings → Accelerator → GPU T4 → Save

### 3. Generate predictions
The final cell saves predictions to:
```
submission.csv
```

**Format:**
```
image,text
test-0.png,BU522X
test-1.png,XQ8NE2
```

---

## 📂 Repository Structure

```
├── notebook.ipynb          # Full training + inference pipeline
├── submission.csv          # Test set predictions
├── README.md               # This file
└── best_crnn_ctc_model.pth # Saved best model checkpoint
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| PyTorch | Model building and training |
| OpenCV | Image loading and CLAHE |
| Albumentations | Data augmentation |
| Pandas | Data loading and CSV handling |
| scikit-learn | Train/val split |
| tqdm | Training progress bar |
| Kaggle (GPU T4) | Training environment |

---

## 📈 Results

| Metric | Value |
|---|---|
| Evaluation Metric | Character Error Rate (CER) |
| Decoder | CTC Greedy Decode |
| Best model saved based on | Lowest validation CER |

---

## 💡 Possible Improvements

- **Beam search decoding** instead of greedy (lower CER)
- **ResNet backbone** for stronger feature extraction
- **Transformer encoder** instead of BiLSTM
- **Test-Time Augmentation (TTA)** for more robust predictions
- **Larger image resolution** to preserve fine character detail

---
