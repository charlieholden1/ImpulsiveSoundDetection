"""
train – Training pipeline for the gunshot detection CNN.

Modules:
  - dataset.py: tf.data pipeline for spectrogram images
  - model.py: EfficientNetB0 architecture with two-stage training
  - train.py: Top-level training orchestration
  - evaluate.py: Confusion matrix, ROC, PR curves, cross-validation
"""
