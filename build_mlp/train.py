"""
Train the pose classifier: load data, fit MLP, checkpoint, validate.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from .model import build_mlp
from .preprocessor import get_features_and_labels, split_train_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _load_and_split_data(
    csv_path: str,
    train_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load pose CSV, split by video, extract features/labels."""
    train_df, test_df = split_train_test(
        csv_path,
        train_ratio=train_ratio,
        random_state=random_state,
    )
    if train_df.empty or test_df.empty:
        raise ValueError("Train or test set is empty. Add more videos.")

    X_train_df, y_train = get_features_and_labels(train_df)
    X_test_df, y_test = get_features_and_labels(test_df)
    X_train = X_train_df.values.astype(np.float32)
    X_test = X_test_df.values.astype(np.float32)
    return X_train, X_test, y_train, y_test


def _encode_labels(
    y_train,
    y_test,
) -> tuple[LabelEncoder, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode string labels to integers and one-hot for softmax."""
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    y_train_cat = keras.utils.to_categorical(y_train_enc)
    y_test_cat = keras.utils.to_categorical(y_test_enc)
    return le, y_train_enc, y_test_enc, y_train_cat, y_test_cat


def _create_validation_split(
    X_train: np.ndarray,
    y_train_cat: np.ndarray,
    val_ratio: float,
    X_test: np.ndarray,
    y_test_cat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split off validation from train."""
    n_val = int(len(X_train) * val_ratio)
    if n_val > 0:
        X_val = X_train[-n_val:]
        y_val = y_train_cat[-n_val:]
        X_train = X_train[:-n_val]
        y_train_cat = y_train_cat[:-n_val]
    else:
        X_val = X_test
        y_val = y_test_cat
    return X_train, y_train_cat, X_val, y_val


def _build_and_compile_model(
    input_dim: int,
    num_classes: int,
    hidden_layers: tuple[int, int, int],
    dropout: float,
) -> keras.Model:
    """Build MLP and compile with Adam, categorical_crossentropy."""
    model = build_mlp(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        num_classes=num_classes,
        dropout=dropout,
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _create_training_callbacks(checkpoint_dir: Path) -> list:
    """ModelCheckpoint and EarlyStopping callbacks."""
    model_path = checkpoint_dir / "best_model.keras"
    return [
        keras.callbacks.ModelCheckpoint(
            str(model_path), monitor="val_accuracy", save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True,
        ),
    ]


def _fit_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train_cat: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    callbacks: list,
) -> keras.callbacks.History:
    """Run model.fit with validation."""
    logger.info("Training on %d samples, validating on %d", len(X_train), len(X_val))
    return model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )


def _save_label_encoder(le: LabelEncoder, checkpoint_dir: Path) -> None:
    """Save class mapping to JSON for inference."""
    labels_path = checkpoint_dir / "label_encoder.json"
    with open(labels_path, "w") as f:
        json.dump({"classes": le.classes_.tolist()}, f)
    logger.info("Saved label encoder to %s", labels_path)


def _evaluate_on_test(
    model: keras.Model,
    X_test: np.ndarray,
    y_test_cat: np.ndarray,
) -> tuple[float, float]:
    """Evaluate on test set."""
    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    logger.info("Test accuracy: %.4f, Test loss: %.4f", test_acc, test_loss)
    return test_loss, test_acc


def _plot_evaluation_curves(
    model: keras.Model,
    X_test: np.ndarray,
    y_test_cat: np.ndarray,
    class_names: list[str],
    checkpoint_dir: Path,
) -> None:
    """Save confusion matrix, ROC, and PR curves."""
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(ax=ax, cmap="Blues")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(checkpoint_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    logger.info("Saved confusion matrix to %s", checkpoint_dir / "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_cat[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance (0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Test Set, One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(checkpoint_dir / "roc_curve.png", dpi=150)
    plt.close()
    logger.info("Saved ROC curve to %s", checkpoint_dir / "roc_curve.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test_cat[:, i], y_pred_proba[:, i])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label=f"{name} (AP = {pr_auc:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (Test Set, One-vs-Rest)")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(checkpoint_dir / "pr_curve.png", dpi=150)
    plt.close()
    logger.info("Saved PR curve to %s", checkpoint_dir / "pr_curve.png")


def train(
    csv_path: str,
    checkpoint_dir: str = "checkpoints",
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    epochs: int = 50,
    batch_size: int = 32,
    hidden_layers: tuple[int, int, int] = (128, 64, 32),
    dropout: float = 0.2,
    random_state: int = 42,
) -> keras.Model:
    """
    Load pose CSV, split by video, fit MLP with checkpoint and validation.
    """
    checkpoint_dir = Path(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = _load_and_split_data(csv_path, train_ratio, random_state)
    le, _, _, y_train_cat, y_test_cat = _encode_labels(y_train, y_test)
    X_train, y_train_cat, X_val, y_val = _create_validation_split(
        X_train, y_train_cat, val_ratio, X_test, y_test_cat
    )

    input_dim = X_train.shape[1]
    num_classes = len(le.classes_)
    model = _build_and_compile_model(input_dim, num_classes, hidden_layers, dropout)
    callbacks = _create_training_callbacks(checkpoint_dir)

    _fit_model(model, X_train, y_train_cat, X_val, y_val, epochs, batch_size, callbacks)
    _save_label_encoder(le, checkpoint_dir)
    _evaluate_on_test(model, X_test, y_test_cat)
    _plot_evaluation_curves(model, X_test, y_test_cat, le.classes_.tolist(), checkpoint_dir)

    return model
