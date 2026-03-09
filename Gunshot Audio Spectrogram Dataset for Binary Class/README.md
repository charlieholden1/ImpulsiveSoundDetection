# Gunshot Audio Spectrogram Dataset for Binary Classification Using FFT, LogMel, and MFCC Features

## Overview

This dataset consists of 15,962 labeled spectrogram images generated from audio recordings related to firearm discharge and various non-gunshot sounds. The primary objective of this dataset is to support research in gunshot detection using machine learning and deep learning techniques, particularly convolutional neural networks (CNNs).

Gun-related violence remains a major public safety challenge worldwide, and automated acoustic detection systems have emerged as a promising tool to support real-time surveillance and forensic investigation. This dataset provides a pre-processed and standardized corpus for binary classification tasks: **Gunshot** vs. **Non-Gunshot**.

## Dataset Composition

- **Total files**: 15,962 spectrogram images
  - **Gunshot (positive class)**: 5,614
  - **Non-Gunshot (negative class)**: 10,348

Each spectrogram represents a 5-second audio clip, extracted and processed from multiple public audio repositories.

## Source Datasets

The audio samples were collected from the following public datasets:
- UrbanSound8k (Salamon et al., 2014)
- ESC-50 (Piczak, 2015)
- Gunshot/Gunfire Audio Dataset (Kabealo et al., 2023)
- Gunshot Audio Dataset (Tuncer et al., 2021)
- Gunshot Audio Forensics Dataset (Lilien, 2018)

## Preprocessing and Feature Extraction

All audio files were:
- Converted to WAV format
- Standardized to 5 seconds (via zero-padding or trimming)
- Pre-emphasized using a digital high-pass filter with α = 0.97
- Normalized in amplitude

Each audio sample was then transformed into **one of three spectrogram types**:
- **FFT (Fast Fourier Transform)**
- **MFCC (Mel-Frequency Cepstral Coefficients)**
- **Log-Mel Spectrograms**

Feature extraction was performed using the [Librosa](https://librosa.org) Python library.

## File Naming Convention

Files are named using the following pattern:

```
<FEATURE>_<CLASS>_XXXXXX.png
```

- `<FEATURE>` ∈ {`FFT`, `MFCC`, `LOGMEL`}
- `<CLASS>` ∈ {`GUN`, `NOGUN`}
- `XXXXXX` is a zero-padded sequential index

**Example**:
- `FFT_GUN_000123.png`
- `MFCC_NOGUN_005614.png`

## Intended Use

This dataset is intended for:
- Training and evaluating gunshot detection models
- Research in acoustic event detection
- Forensic audio analysis
- Real-time alert systems for urban surveillance

## License

This dataset is provided for academic and research purposes only. Please cite the original sources when using any portion of the data in publications or derived works.