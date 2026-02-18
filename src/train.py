"""
Train the pose classifier: load data, fit MLP, checkpoint, validate.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
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
    Load pose CSV, split train/test by video, fit MLP with checkpoint and validation.

    Args:
        csv_path: Path to combined pose CSV.
        checkpoint_dir: Directory for model checkpoints.
        train_ratio: Fraction of videos for training (rest for test).
        val_ratio: Fraction of train for validation (from train set).
        epochs: Max epochs.
        batch_size: Batch size.
        hidden_layers: MLP hidden layer units.
        dropout: Dropout rate.
        random_state: Random seed.

    Returns:
        Trained model (best by val_accuracy).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_df, test_df = split_train_test(
        csv_path,
        train_ratio=train_ratio,
        random_state=random_state,
    )
    if train_df.empty or test_df.empty:
        raise ValueError("Train or test set is empty. Add more videos.")

    X_train_df, y_train = get_features_and_labels(train_df)
    X_test_df, y_test = get_features_and_labels(test_df)

    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    X_train = X_train_df.values.astype(np.float32)
    X_test = X_test_df.values.astype(np.float32)
    y_train_cat = keras.utils.to_categorical(y_train_enc)
    y_test_cat = keras.utils.to_categorical(y_test_enc)

    n_val = int(len(X_train) * val_ratio)
    if n_val > 0:
        X_val = X_train[-n_val:]
        y_val = y_train_cat[-n_val:]
        X_train = X_train[:-n_val]
        y_train_cat = y_train_cat[:-n_val]
    else:
        X_val = X_test
        y_val = y_test_cat

    input_dim = X_train.shape[1]
    num_classes = len(le.classes_)
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

    model_path = Path(checkpoint_dir) / "best_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
        ),
    ]

    logger.info("Training on %d samples, validating on %d", len(X_train), len(X_val))
    model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    labels_path = Path(checkpoint_dir) / "label_encoder.json"
    with open(labels_path, "w") as f:
        json.dump({"classes": le.classes_.tolist()}, f)
    logger.info("Saved label encoder to %s", labels_path)

    test_loss, test_acc = model.evaluate(X_test, y_test_cat)
    logger.info("Test accuracy: %.4f, Test loss: %.4f", test_acc, test_loss)

    return model
