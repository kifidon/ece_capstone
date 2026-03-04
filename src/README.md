# Model Training

## Model Training V1.0

### Problem

Initial training with a fixed 70/15/15 split showed severe overfitting: 99.9% train accuracy but ~49% validation accuracy. With only ~7 videos per class, a fixed split left just 1 video per class in validation — too little to get a stable signal.

### Changes

1. **K-Fold Cross-Validation (GroupKFold, k=5):** Instead of a single fixed split, we rotate which videos are used for validation across 5 folds. Every video gets evaluated exactly once. `GroupKFold` ensures all frames from the same video stay in the same fold (no data leakage). This gives a much more reliable accuracy estimate with limited data.

2. **Data Augmentation (Gaussian noise, std=0.01):** We add small random noise to the y,x keypoint coordinates during training (confidence scores are left unchanged). Each training set is augmented with 2 additional noisy copies (3x total), which forces the model to generalize rather than memorize exact keypoint positions.

3. **Increased Dropout (0.2 → 0.4):** Higher dropout forces the network to spread learned features across more neurons, reducing co-adaptation and overfitting.

## Model Training V1.01

### Problem

V1.0 K-Fold results (mean 92.6%) masked a deeper issue. Leave-One-Video-Out (LOVO) cross-validation revealed that 29/30 videos scored 100%, but **`Lying04.mp4` scored 0%** — the model classified every frame as Reaching or Sitting.

Visualizing mean keypoints across all Lying videos showed the cause: Lying04 was recorded with the person facing the opposite direction. The model learned "Lying = head on the left, feet on the right" and couldn't recognize the same pose mirrored.

### Changes

1. **Leave-One-Video-Out (LOVO) Cross-Validation:** Replaced K-Fold with LOVO — each of the 30 videos is held out exactly once. This gives a per-video accuracy score and pinpoints exactly which videos the model struggles with.

2. **Horizontal Keypoint Mirroring:** Added a mirror augmentation that negates x-coordinates and swaps left/right keypoint pairs (e.g. left_shoulder ↔ right_shoulder). Every training fold now sees both the original and mirrored version of each pose, making the model direction-invariant.

3. **Confidence Scores Added:** MoveNet outputs a confidence score (0–1) per keypoint. Previously we discarded it (`[:2]`). Now all 3 values (y, x, confidence) are kept, giving the classifier 51 input features instead of 34. This lets the model learn to weight uncertain keypoints lower.

4. **Configurable Fold Count:** Cross-validation now accepts an `n_folds` parameter. `None` (default) = LOVO, any integer = GroupKFold with that many folds. This lets you trade off between diagnostic granularity (LOVO) and training speed (fewer folds).

5. **Per-Frame Predictions Saved:** Every fold now collects true labels and predicted probabilities for each validation frame. These are saved to `checkpoints/cv_predictions.npz` for generating confusion matrices, ROC curves, and precision-recall analysis.

6. **Final Model Trained on All Data:** After cross-validation (evaluation only), a final model is trained on all 30 videos with full augmentation and saved as `checkpoints/best_model.keras`. This is the production model.

### Results

| Version | Validation | Mean Accuracy | Notes |
|---------|-----------|---------------|-------|
| Baseline | Fixed 70/15/15 | ~49% | Severe overfitting |
| V1.0 | K-Fold (k=5) | 92.6% | Noise augmentation + dropout |
| V1.01 pre-mirror | LOVO (n=30) | 96.7% | Identified Lying04 as the only failure (0%) |
| **V1.01** | **LOVO (n=30)** | **100%** | + mirroring + confidence scores |

## Architecture

**MLP (Multi-Layer Perceptron)** with 3 hidden layers:

- Input: 51 features (17 keypoints × 3: y, x, confidence)
- Hidden: 128 → 64 → 32 neurons, each with BatchNormalization + Dropout (0.4)
- Output: 4 classes (Lying, Reaching, Sitting, Standing) with softmax
- Optimizer: Adam (lr=0.001)
- Loss: Categorical crossentropy

## Data Augmentation Pipeline

Each training fold's data is expanded 6x:

1. Original keypoints
2. Mirrored keypoints (x negated, left/right pairs swapped)
3. 2× noisy copies of original (Gaussian noise, std=0.01 on y,x only)
4. 2× noisy copies of mirrored

## Training Pipeline

1. **Cross-validation** (LOVO or GroupKFold) — evaluates model generalization, saves per-frame predictions for analysis
2. **Final model** — trained on all data with full augmentation, saved as `best_model.keras`

## Usage

```bash
# LOVO (default, one fold per video)
python scripts/train_model.py

# 5-fold GroupKFold
python scripts/train_model.py --folds 5

# Specific CSV
python scripts/train_model.py --csv data/csvs/poses_data_20260303.csv
```

## Outputs

| File | Description |
|------|-------------|
| `checkpoints/best_model.keras` | Final model trained on all data |
| `checkpoints/label_encoder.json` | Class name ↔ index mapping |
| `checkpoints/cv_results.json` | Per-fold accuracy and loss |
| `checkpoints/cv_predictions.npz` | Per-frame true labels + predicted probabilities |
| `checkpoints/training.log` | Full training log |
