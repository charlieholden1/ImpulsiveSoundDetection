"""
model.py – EfficientNetB0 binary classifier for gunshot detection.

Architecture:
  - EfficientNetB0 base (ImageNet weights, frozen initially)
  - GlobalAveragePooling2D
  - Dense(128, relu) with Dropout(0.4)
  - Dense(1, sigmoid)

Two-stage training:
  Stage A: Train head only (15 epochs, frozen base)
  Stage B: Fine-tune top 30 layers (20 epochs, lower LR)
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import logging

logger = logging.getLogger(__name__)


def build_metrics() -> list[tf.keras.metrics.Metric]:
    """
    Build metrics that are consistent with a logits-output binary model.

    Returns
    -------
    list[tf.keras.metrics.Metric]
        Metrics configured for raw-logit predictions.
    """
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0),
        tf.keras.metrics.Precision(name="precision", thresholds=0.0),
        tf.keras.metrics.Recall(name="recall", thresholds=0.0),
        tf.keras.metrics.AUC(name="auc", from_logits=True),
        tf.keras.metrics.AUC(name="auprc", curve="PR", from_logits=True),
    ]


def build_cnn_model(
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = 0.4,
) -> tf.keras.Model:
    """
    Build EfficientNetB0-based binary classifier.

    Parameters
    ----------
    input_shape : tuple
        Input shape (height, width, channels).
    dropout_rate : float
        Dropout rate in dense layers.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model.
    """
    logger.info("Building EfficientNetB0 model with input shape %s", input_shape)

    # Load EfficientNetB0 with ImageNet weights
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    # Freeze base model initially (Stage A)
    base_model.trainable = False

    # Build classification head
    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate * 0.75)(x)  # Slightly lower dropout in last dense
    # Output logits (no activation) - BinaryCrossentropy with from_logits=True expects this
    outputs = layers.Dense(1, activation=None)(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    logger.info("Model built. Total parameters: %d", model.count_params())

    return model, base_model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
) -> None:
    """
    Compile model for binary classification with imbalance-aware settings.

    Parameters
    ----------
    model : tf.keras.Model
        The model to compile.
    learning_rate : float
        Learning rate for Adam optimizer.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # CRITICAL FIX: from_logits=True because output layer has NO activation (outputs raw logits)
    # This provides numerical stability for imbalanced binary classification
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    metrics = build_metrics()

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    logger.info("Model compiled with lr=%.0e (from_logits=True for numerical stability)", learning_rate)


def unfreeze_top_layers(base_model: tf.keras.Model, num_layers: int = 30) -> None:
    """
    Unfreeze the top N layers of the base model for fine-tuning.

    Parameters
    ----------
    base_model : tf.keras.Model
        The base model (EfficientNetB0).
    num_layers : int
        Number of layers to unfreeze from the top.
    """
    base_model.trainable = True

    # Freeze all but the top num_layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    logger.info("Unfroze top %d layers of base model", num_layers)
