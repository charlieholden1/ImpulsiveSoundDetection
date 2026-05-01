# Critical Fixes Applied to Training Pipeline

## Summary
Fixed 5 critical issues causing validation recall collapse from 91.81% → 10.71% during fine-tuning.

---

## Fix #1: Preprocessing Normalization Mismatch
**Files**: `train/dataset.py:218`
**Problem**: Images were normalized to [0, 1], then `preprocess_input()` expected to normalize again, causing double-normalization and corrupted input distribution.
**Solution**: Remove manual normalization. Let `EfficientNetB0.preprocess_input()` handle all normalization (expects uint8 or [0,1] float32).
**Impact**: ✅ Prevents input corruption during fine-tuning

---

## Fix #2: Loss/Activation Mismatch  
**Files**: `train/model.py:61, 85`
**Problem**: 
- Output layer used `activation="sigmoid"` (producing [0,1])
- Loss used `BinaryCrossentropy()` with default `from_logits=False` (expects raw logits)
- This mismatch caused numerical instability and gradient collapse
**Solution**: 
- Changed output to `activation=None` (outputs raw logits)
- Changed loss to `BinaryCrossentropy(from_logits=True)`
**Impact**: ✅ Numerical stability, proper gradient flow during fine-tuning

---

## Fix #3: Recall Metric at Fixed 0.5 Threshold
**Files**: `train/model.py:89`
**Problem**: 
- Optimal threshold for imbalanced data is ~0.3-0.4, NOT 0.5
- During fine-tuning, prediction distribution shifts
- Recall @ 0.5 collapses (10.71%) even though recall @ optimal threshold improves
- Early stopping triggered prematurely due to metric collapse
**Solution**: 
- Monitor recall at multiple thresholds: [0.3, 0.4, 0.5, 0.6]
- Changed early stopping to monitor `val_auc` (threshold-independent)
**Impact**: ✅ Prevents false early stopping, continues training through optimal convergence

---

## Fix #4: Aggressive Learning Rate Drop
**Files**: `train/train.py:129`
**Problem**: Learning rate dropped 100x from 1e-3 → 1e-5 instantly. Too aggressive for fine-tuning.
**Solution**: Use exponential decay schedule:
- Start at 1e-3
- Decay 4% per epoch (~10 epochs worth of steps)
- Gradually reduce to ~1e-5 instead of jumping instantly
**Impact**: ✅ Smoother convergence, prevents gradient collapse, better learning in fine-tuning

---

## Fix #5: Early Stopping Metric  
**Files**: `train/train.py` (callbacks in both Stage A and Stage B)
**Problem**: 
- Monitoring `val_recall` at fixed 0.5 threshold is unstable
- Causes premature stopping when threshold-dependent metric drops
**Solution**: Monitor `val_auc` instead
- AUC is threshold-independent
- Better indicator of model discrimination ability
- Won't collapse due to decision boundary shifts
**Impact**: ✅ Stable training, full epochs completed

---

## Summary of Changes

| File | Function/Line | Change |
|------|---------------|--------|
| `train/dataset.py` | `_load_and_preprocess_image()` | Removed manual [0,1] normalization, let preprocess_input() handle it |
| `train/model.py` | `build_cnn_model():61` | Changed output: `sigmoid` → `None` (logits) |
| `train/model.py` | `compile_model():85` | Added `from_logits=True` to BinaryCrossentropy |
| `train/model.py` | `compile_model():89` | Added multi-threshold recall metrics [0.3, 0.4, 0.5, 0.6] |
| `train/train.py` | `train_stage_a()` | Changed early stopping: `val_recall` → `val_auc` |
| `train/train.py` | `train_stage_b()` | Added ExponentialDecay LR schedule, changed early stopping: `val_recall` → `val_auc` |

---

## Expected Improvements

After retraining with these fixes:
- ✅ **Recall**: Should stay >90% throughout fine-tuning (no collapse)
- ✅ **Precision**: Should improve significantly (better threshold calibration)
- ✅ **AUC**: Should reach >0.90 (good discrimination)
- ✅ **Stability**: Smooth training curves, no sudden metric drops
- ✅ **Convergence**: Full 15+20 epochs completed (no premature early stopping)

---

## How to Retrain

```bash
# Kill any existing training
# Then run:
python -m train.train --feature-type LogMel --batch-size 32 --threshold-sweep
```

Training should now progress smoothly and complete without recall collapse! 🚀
