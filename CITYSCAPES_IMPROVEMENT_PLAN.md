# Cityscapes Performance Improvement Plan
## Achieving 90%+ Accuracy on Cityscapes Dataset

**Date**: November 21, 2025  
**Current Performance**: δ<1.25 = 38.25% (AbsRel = 0.3736)  
**Target Performance**: δ<1.25 = 90%+  
**Gap**: ~52% improvement needed

---

## Table of Contents
1. [Problem Analysis](#problem-analysis)
2. [Why Current Performance is Low](#why-current-performance-is-low)
3. [Solution Options](#solution-options)
4. [Recommended Approach](#recommended-approach)
5. [Training Requirements](#training-requirements)
6. [Alternative Quick Wins](#alternative-quick-wins)
7. [Implementation Roadmap](#implementation-roadmap)

---

## Problem Analysis

### Current Results (KITTI vs Cityscapes)

| Metric | KITTI (full_sh) | Cityscapes (full_sh) | Gap |
|--------|----------------|---------------------|-----|
| **AbsRel** | 0.0584 | 0.3736 | 6.4× worse |
| **δ<1.25** | 96.09% | 38.25% | -57.84% |
| **RMSE** | 3.38m | ~20-30m (estimated) | 6-9× worse |

### Domain Shift Issues

The model was trained exclusively on KITTI data, which has different characteristics than Cityscapes:

| Characteristic | KITTI | Cityscapes |
|----------------|-------|------------|
| **Location** | Karlsruhe, Germany (suburban/rural) | 50+ European cities (urban) |
| **Scene Type** | Highways, suburban roads | Dense urban streets |
| **Camera Height** | ~1.65m | Different mounting |
| **Camera FOV** | Narrower | Wider angle |
| **Baseline** | ~0.54m (stereo) | 0.22m (stereo) |
| **Focal Length** | ~721 pixels | 2262 pixels |
| **Image Size** | 1242×375 | 2048×1024 |
| **Depth Range** | Optimized for 0-80m | More varied urban scenes |

---

## Why Current Performance is Low

### 1. **Domain Gap** (Primary Issue)
- Model learned KITTI-specific features (road textures, scene layouts, depth distributions)
- Cityscapes has different:
  - Building densities and heights
  - Street layouts (narrower, more complex)
  - Object distributions (more pedestrians, cyclists, parked cars)
  - Lighting conditions (urban shadows, reflections)

### 2. **Camera Calibration Mismatch**
- Different intrinsic parameters (focal length, distortion)
- Different stereo baselines
- Different depth scales

### 3. **Scale Ambiguity**
- Even with median scaling, the depth distributions are fundamentally different
- Urban scenes (Cityscapes) have more multi-modal depth distributions
- KITTI scenes are more "highway-like" with simpler depth layouts

### 4. **Resolution Mismatch**
- Model trained on 640×192
- Cityscapes native is 2048×1024
- Downsampling loses fine urban details

---

## Solution Options

### Option 1: ⭐ **Fine-tune on Cityscapes** (RECOMMENDED)
**Pros:**
- Will achieve 90%+ accuracy if done properly
- Leverages existing KITTI knowledge (transfer learning)
- Relatively fast (few epochs needed)
- Standard practice in research

**Cons:**
- Requires training setup (GPU, PyTorch training loop)
- Need to prepare Cityscapes training data
- Takes 1-3 days on modern GPU

**Expected Results:**
- δ<1.25: 85-95% (comparable to KITTI)
- AbsRel: 0.08-0.15 (3-5× improvement)

**Estimated Time:**
- Setup: 2-4 hours
- Training: 8-24 hours (depends on GPU)
- Total: 1-2 days

---

### Option 2: **Train from Scratch on Cityscapes**
**Pros:**
- Could achieve best possible Cityscapes performance
- No KITTI bias

**Cons:**
- Much longer training time (1-2 weeks)
- Requires more computational resources
- Will lose KITTI performance (unless multi-dataset training)

**Expected Results:**
- δ<1.25: 90-95% (best case)
- AbsRel: 0.06-0.10

**Estimated Time:**
- Training: 1-2 weeks on modern GPU

---

### Option 3: **Multi-Dataset Training** (Best for Generalization)
**Pros:**
- Works well on BOTH datasets
- More robust, generalizes better
- State-of-the-art approach

**Cons:**
- Most complex to implement
- Requires careful dataset balancing
- Longest training time

**Expected Results:**
- KITTI: δ<1.25: 93-96%
- Cityscapes: δ<1.25: 88-93%

**Estimated Time:**
- Training: 1-3 weeks

---

### Option 4: 🚀 **Domain Adaptation** (No Training)
**Pros:**
- No training required
- Can be implemented quickly
- Uses existing weights

**Cons:**
- Limited improvements (maybe 45-55% accuracy)
- Complex implementation
- Not a complete solution

**Techniques:**
- Test-time adaptation
- Style transfer preprocessing
- Camera parameter normalization

**Expected Results:**
- δ<1.25: 45-55% (modest improvement)
- AbsRel: 0.25-0.30

**Estimated Time:**
- Implementation: 1-2 days

---

## Recommended Approach

### 🎯 **Best Solution: Fine-tune on Cityscapes**

This is the standard approach in research and will give you the best results in the shortest time.

#### Why Fine-tuning Works:
1. Model already learned general depth features from KITTI
2. Only needs to adapt to Cityscapes-specific patterns
3. Much faster than training from scratch
4. Proven to work in literature (Godard et al., Zhou et al.)

#### What You Need:

**Hardware:**
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3070/3080/4090, or cloud GPU)
- Alternative: Google Colab Pro (~$10/month), AWS/Azure GPU instance
- Your MacBook MPS won't be fast enough for training (inference only)

**Data:**
- Cityscapes stereo image pairs (you already have these!)
- ~23,000 training images from train split
- Use disparity maps as supervision (you have these too!)

**Code:**
- Training script (needs to be created/adapted)
- Loss functions (photometric + smoothness)
- Learning rate scheduler
- Data augmentation pipeline

---

## Training Requirements

### Detailed Fine-tuning Specifications

```python
# Hyperparameters for Fine-tuning on Cityscapes
BATCH_SIZE = 12  # Depends on GPU memory
LEARNING_RATE = 1e-5  # Lower than initial training (1e-4)
NUM_EPOCHS = 20  # Fine-tuning needs fewer epochs
WEIGHT_DECAY = 1e-4
OPTIMIZER = "Adam"

# Data Augmentation
- Random horizontal flip
- Color jittering
- Random brightness/contrast

# Loss Functions
- Photometric loss (L1 + SSIM)
- Smoothness loss (edge-aware)
- Optional: Self-occlusion masking

# Training Strategy
1. Freeze encoder for first 2 epochs (only train decoder)
2. Unfreeze encoder, train full model
3. Use cosine learning rate schedule
4. Early stopping on validation loss
```

### Expected Training Time (per model)

| GPU | Training Time | Cost (if cloud) |
|-----|---------------|-----------------|
| RTX 4090 | 8-12 hours | $10-15 |
| RTX 3080 | 12-18 hours | $15-20 |
| A100 (cloud) | 4-6 hours | $20-30 |
| V100 (cloud) | 8-10 hours | $25-35 |
| Google Colab Pro | 12-20 hours | $10 (monthly) |

### Storage Requirements
- Cityscapes training data: ~30GB
- Model checkpoints: ~5GB
- Total: ~35GB

---

## Alternative Quick Wins (Without Training)

If you can't train right now, here are some techniques to improve performance:

### 1. **Better Preprocessing** (Quick, +3-5% accuracy)
```python
# Camera intrinsics adjustment
# Normalize depth scale based on Cityscapes camera parameters
# Apply histogram equalization for lighting normalization
```

### 2. **Test-Time Augmentation** (Medium effort, +2-4% accuracy)
```python
# Average predictions from:
# - Original image
# - Horizontally flipped image
# - Multiple scales
```

### 3. **Post-processing Refinement** (Medium effort, +3-6% accuracy)
```python
# Use CRF (Conditional Random Field) for edge refinement
# Bilateral filtering for smoothness
# Outlier removal
```

### 4. **Ensemble Models** (Easy, +4-7% accuracy)
```python
# Average predictions from multiple model variants
# Weighted ensemble based on KITTI performance
```

**Combined Quick Wins:** Could get you to 50-55% accuracy without training!

---

## Implementation Roadmap

### Phase 1: Quick Improvements (1-2 days, no training)
**Goal:** Boost accuracy from 38% to ~50%

- [ ] Implement test-time augmentation
- [ ] Add ensemble of all 6 models
- [ ] Improve preprocessing (camera normalization)
- [ ] Post-processing refinement
- [ ] Re-evaluate on Cityscapes

**Expected Result:** δ<1.25 ≈ 48-52%

---

### Phase 2: Setup Training Infrastructure (1-2 days)
**Goal:** Prepare for fine-tuning

- [ ] Set up GPU access (local or cloud)
- [ ] Prepare Cityscapes training data
- [ ] Create training script
  - Data loaders
  - Loss functions
  - Training loop
  - Validation
- [ ] Verify training pipeline on small subset

---

### Phase 3: Fine-tune Models (1-3 days)
**Goal:** Achieve 90%+ accuracy

- [ ] Fine-tune best model (full_sh_640_192) first
- [ ] Validate on Cityscapes val set
- [ ] If successful, fine-tune other models
- [ ] Compare results

**Expected Result:** δ<1.25 ≈ 88-93%

---

### Phase 4: Hyperparameter Optimization (Optional, 2-3 days)
**Goal:** Squeeze out maximum performance

- [ ] Grid search learning rates
- [ ] Try different augmentations
- [ ] Adjust loss weights
- [ ] Longer training if needed

**Expected Result:** δ<1.25 ≈ 92-95%

---

## Code Implementation Examples

### Quick Win: Test-Time Augmentation

```python
def predict_with_tta(model, image):
    """Test-time augmentation for better predictions"""
    predictions = []
    
    # Original
    pred = model.predict(image)
    predictions.append(pred)
    
    # Horizontal flip
    image_flipped = torch.flip(image, dims=[3])
    pred_flipped = model.predict(image_flipped)
    pred_flipped = torch.flip(pred_flipped, dims=[3])
    predictions.append(pred_flipped)
    
    # Average predictions
    final_pred = torch.mean(torch.stack(predictions), dim=0)
    return final_pred
```

### Quick Win: Model Ensemble

```python
def ensemble_predict(models, image):
    """Ensemble multiple models for robust prediction"""
    predictions = []
    weights = [1.0, 0.9, 0.8, 0.8, 0.7, 0.7]  # Based on KITTI performance
    
    for model, weight in zip(models, weights):
        pred = model.predict(image)
        predictions.append(pred * weight)
    
    final_pred = torch.sum(torch.stack(predictions), dim=0) / sum(weights)
    return final_pred
```

### Training Setup Template

```python
# Fine-tuning script structure
def finetune_on_cityscapes():
    # 1. Load pre-trained KITTI model
    encoder, decoder = load_pretrained_model('weights/RTMonoDepth/full/sh_640_192')
    
    # 2. Prepare Cityscapes data
    train_loader = get_cityscapes_dataloader('train', batch_size=12)
    val_loader = get_cityscapes_dataloader('val', batch_size=8)
    
    # 3. Setup optimizer (lower learning rate for fine-tuning)
    optimizer = torch.optim.Adam([
        {'params': encoder.parameters(), 'lr': 1e-5},
        {'params': decoder.parameters(), 'lr': 1e-4}
    ])
    
    # 4. Training loop
    for epoch in range(20):
        train_loss = train_epoch(encoder, decoder, train_loader, optimizer)
        val_metrics = validate(encoder, decoder, val_loader)
        
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, δ<1.25={val_metrics['a1']:.4f}")
        
        # Save checkpoint
        if val_metrics['a1'] > best_accuracy:
            save_checkpoint(encoder, decoder, f'checkpoint_epoch{epoch}.pth')
```

---

## Research Papers for Reference

### Fine-tuning for Domain Adaptation:
1. **Godard et al., 2019**: "Digging Into Self-Supervised Monocular Depth Estimation"
   - Used fine-tuning successfully on multiple datasets
   
2. **Ranftl et al., 2020**: "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer"
   - Multi-dataset training approach

3. **Zhao et al., 2020**: "Towards Better Generalization: Joint Depth-Pose Learning without PoseNet"
   - Domain adaptation techniques

---

## Budget Options

### Free/Low-Cost Training Options:

1. **Google Colab Pro** ($10/month)
   - A100 GPU access
   - ~6-8 hours training time
   - Easy to use, Jupyter notebook interface

2. **Kaggle Kernels** (Free!)
   - 30 hours/week GPU time
   - T4 or P100 GPUs
   - Good for experimentation

3. **AWS EC2 Spot Instances** (~$0.40/hour)
   - g4dn.xlarge (T4 GPU)
   - Only pay for what you use
   - Can be interrupted

4. **Lambda Labs** (~$0.50/hour)
   - RTX 3090 access
   - Reliable
   - Good for short training runs

---

## Recommendation Summary

### For 90%+ Cityscapes Accuracy:

**Best Path:** Fine-tune on Cityscapes
- **Time:** 2-3 days total
- **Cost:** $10-30 (cloud GPU)
- **Complexity:** Medium
- **Success Rate:** Very high (90-95% likely to achieve 90%+ accuracy)

### Immediate Next Steps:

1. **Try Quick Wins First** (today)
   - Implement ensemble + TTA
   - See if you can get to 50%
   
2. **If you want 90%+, set up training** (this week)
   - Get GPU access (Colab Pro is easiest)
   - I can help write the training script
   - Fine-tune the best model
   
3. **Document everything** (ongoing)
   - Update EVALUATION_DOCUMENTATION.md
   - Create TRAINING_LOG.md
   - Keep track of hyperparameters

---

## Decision Matrix

| Goal | Method | Time | Cost | Difficulty | Accuracy |
|------|--------|------|------|-----------|----------|
| Quick improvement | Ensemble + TTA | 1 day | $0 | Easy | 48-52% |
| Good performance | Fine-tune | 2-3 days | $10-30 | Medium | 88-93% |
| Best performance | Train from scratch | 1-2 weeks | $50-100 | Hard | 92-95% |
| Universal model | Multi-dataset | 2-3 weeks | $100-200 | Hard | 90-94% |

---

## Conclusion

**Yes, training/fine-tuning is necessary to achieve 90%+ accuracy on Cityscapes.**

The domain gap between KITTI and Cityscapes is too large for simple techniques alone. However, fine-tuning is relatively quick and cheap (~$10-30 for cloud GPU), and I can help you implement it.

**My Recommendation:**
1. Start with quick wins today (ensemble + TTA) → get to ~50%
2. Set up fine-tuning this week → achieve 90%+
3. Document everything for your paper

Would you like me to:
- [ ] Implement the quick win ensemble/TTA code now?
- [ ] Help you set up a fine-tuning training script?
- [ ] Create a Colab notebook for training?
- [ ] All of the above?

---

**Document Version**: 1.0  
**Last Updated**: November 21, 2025  
**Status**: Action Plan - Awaiting Decision

© 2025 RT-MonoDepth Improvement Project
