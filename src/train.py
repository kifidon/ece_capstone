"""
Train the pose classifier: cross-validation by video, data augmentation, checkpointing.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from .model import build_mlp
from .preprocessor import get_features_and_labels, load_pose_csv

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str) -> None:
    """Configure root logger to write to both stdout and a log file."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    root.handlers.clear()
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logger.info("Logging to %s", log_path)


MIRROR_SWAP_PAIRS = [
    (1, 2),    # left_eye <-> right_eye
    (3, 4),    # left_ear <-> right_ear
    (5, 6),    # left_shoulder <-> right_shoulder
    (7, 8),    # left_elbow <-> right_elbow
    (9, 10),   # left_wrist <-> right_wrist
    (11, 12),  # left_hip <-> right_hip
    (13, 14),  # left_knee <-> right_knee
    (15, 16),  # left_ankle <-> right_ankle
]


def augment_keypoints(X: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to y,x coordinates; leave confidence columns unchanged."""
    X_aug = X.copy()
    for i in range(X.shape[1]):
        if i % 3 != 2:
            X_aug[:, i] += np.random.normal(0, noise_std, size=X.shape[0])
    return X_aug


def mirror_keypoints(X: np.ndarray) -> np.ndarray:
    """Horizontally flip keypoints: negate x values and swap left/right pairs."""
    X_mir = X.copy()
    for i in range(X.shape[1] // 3):
        X_mir[:, i * 3 + 1] *= -1

    for a, b in MIRROR_SWAP_PAIRS:
        a_cols = [a * 3, a * 3 + 1, a * 3 + 2]
        b_cols = [b * 3, b * 3 + 1, b * 3 + 2]
        X_mir[:, a_cols], X_mir[:, b_cols] = X_mir[:, b_cols].copy(), X_mir[:, a_cols].copy()

    return X_mir


def _build_augmented_data(X, y, augment_factor, noise_std):
    """Build augmented dataset: original + mirrored + noisy copies of both."""
    X_mirrored = mirror_keypoints(X)
    batches = [X, X_mirrored]
    labels = [y, y]
    for _ in range(augment_factor):
        batches.append(augment_keypoints(X, noise_std=noise_std))
        batches.append(augment_keypoints(X_mirrored, noise_std=noise_std))
        labels.extend([y, y])
    X_aug = np.concatenate(batches, axis=0)
    y_aug = np.concatenate(labels, axis=0)
    shuffle_idx = np.random.permutation(len(X_aug))
    return X_aug[shuffle_idx], y_aug[shuffle_idx]


def train(
    csv_path: str,
    checkpoint_dir: str = "checkpoints",
    n_folds: int | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    hidden_layers: tuple[int, int, int] = (128, 64, 32),
    dropout: float = 0.4,
    noise_std: float = 0.01,
    augment_factor: int = 2,
    random_state: int = 42,
) -> None:
    """
    Cross-validation by video with data augmentation.

    When n_folds is None or equals the number of unique videos, uses
    Leave-One-Video-Out (LOVO). Otherwise uses GroupKFold with the
    specified number of folds.

    Collects per-frame predictions across all folds for confusion matrix
    and ROC curve generation.

    Args:
        csv_path: Path to combined pose CSV.
        checkpoint_dir: Directory for model checkpoints.
        n_folds: Number of CV folds. None = LOVO (one fold per video).
        epochs: Max epochs per fold.
        batch_size: Batch size.
        hidden_layers: MLP hidden layer units.
        dropout: Dropout rate.
        noise_std: Std dev of Gaussian noise added to y,x during augmentation.
        augment_factor: How many augmented copies to add (on top of original).
        random_state: Random seed.
    """
    setup_logging(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    np.random.seed(random_state)

    df = load_pose_csv(csv_path)
    if df.empty:
        raise ValueError("CSV is empty.")

    X_df, y = get_features_and_labels(df)
    X_all = X_df.values.astype(np.float32)
    video_ids = df["video_id"].values
    frame_ids = df["frame_id"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)

    labels_path = Path(checkpoint_dir) / "label_encoder.json"
    with open(labels_path, "w") as f:
        json.dump({"classes": le.classes_.tolist()}, f)
    logger.info("Saved label encoder to %s", labels_path)

    unique_videos = sorted(set(video_ids))
    n_videos = len(unique_videos)

    if n_folds is None or n_folds >= n_videos:
        splitter = LeaveOneGroupOut()
        n_folds = n_videos
        cv_name = "LOVO"
        logger.info("Using Leave-One-Video-Out (%d folds)", n_folds)
    else:
        splitter = GroupKFold(n_splits=n_folds)
        cv_name = f"{n_folds}-Fold"
        logger.info("Using GroupKFold with %d folds", n_folds)

    fold_results = []
    all_true_labels = np.full(len(X_all), -1, dtype=int)
    all_pred_probs = np.zeros((len(X_all), num_classes), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X_all, y_enc, groups=video_ids), 1):
        val_videos = sorted(set(video_ids[val_idx]))
        val_label = val_videos[0] if len(val_videos) == 1 else f"{len(val_videos)} videos"
        logger.info("=== Fold %d/%d — held-out: %s ===", fold, n_folds, val_label)

        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_enc[train_idx], y_enc[val_idx]

        n_train_videos = len(set(video_ids[train_idx]))
        logger.info("Train: %d samples (%d videos), Val: %d samples (%d videos)",
                     len(X_train), n_train_videos, len(X_val), len(val_videos))

        X_train_aug, y_train_aug = _build_augmented_data(X_train, y_train, augment_factor, noise_std)

        y_train_cat = keras.utils.to_categorical(y_train_aug, num_classes)
        y_val_cat = keras.utils.to_categorical(y_val, num_classes)

        model = build_mlp(
            input_dim=X_train.shape[1],
            hidden_layers=hidden_layers,
            num_classes=num_classes,
            dropout=dropout,
        )
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=10,
                restore_best_weights=True,
            ),
        ]

        model.fit(
            X_train_aug, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=1,
        )

        val_preds = model.predict(X_val, verbose=0)
        all_true_labels[val_idx] = y_val
        all_pred_probs[val_idx] = val_preds

        val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
        fold_results.append({
            "fold": fold,
            "val_videos": val_videos,
            "val_accuracy": val_acc,
            "val_loss": val_loss,
        })
        logger.info("Fold %d [%s] — val_accuracy: %.4f, val_loss: %.4f", fold, val_label, val_acc, val_loss)

    accs = [r["val_accuracy"] for r in fold_results]
    logger.info("=== %s Results ===", cv_name)
    for r in fold_results:
        vids = r["val_videos"]
        label = vids[0] if len(vids) == 1 else f"Fold {r['fold']} ({len(vids)} videos)"
        logger.info("  %s: val_accuracy=%.4f", label, r["val_accuracy"])
    logger.info("  Mean: %.4f, Std: %.4f", np.mean(accs), np.std(accs))

    results_path = Path(checkpoint_dir) / "cv_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "cv_method": cv_name,
            "n_folds": n_folds,
            "folds": fold_results,
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
        }, f, indent=2)
    logger.info("Saved CV results to %s", results_path)

    predictions_path = Path(checkpoint_dir) / "cv_predictions.npz"
    np.savez(
        predictions_path,
        frame_ids=frame_ids,
        true_labels=all_true_labels,
        pred_probs=all_pred_probs,
        class_names=le.classes_,
    )
    logger.info("Saved per-frame predictions to %s", predictions_path)

    logger.info("=== Training final model on all data ===")
    X_final, y_final = _build_augmented_data(X_all, y_enc, augment_factor, noise_std)
    y_final_cat = keras.utils.to_categorical(y_final, num_classes)

    final_model = build_mlp(
        input_dim=X_all.shape[1],
        hidden_layers=hidden_layers,
        num_classes=num_classes,
        dropout=dropout,
    )
    final_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    final_model.fit(X_final, y_final_cat, epochs=epochs, batch_size=batch_size, verbose=1)

    model_path = Path(checkpoint_dir) / "best_model.keras"
    final_model.save(model_path)
    logger.info("Saved final model to %s", model_path)
