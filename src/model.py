"""
Pose classifier: MLP with 3 hidden layers.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_mlp(
    input_dim: int,
    hidden_layers: tuple[int, int, int] = (128, 64, 32),
    num_classes: int = 2,
    dropout: float = 0.2,
) -> keras.Model:
    """
    Build a 3-layer MLP for pose classification.

    Args:
        input_dim: Number of input features (34 for 17 keypoints × 2 coords).
        hidden_layers: Units per hidden layer (default 128, 64, 32).
        num_classes: Number of pose classes.
        dropout: Dropout rate between layers.

    Returns:
        Compiled Keras Model.
    """
    inputs = keras.Input(shape=(input_dim,))
    x = inputs

    for units in hidden_layers:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="pose_classifier")
    return model
