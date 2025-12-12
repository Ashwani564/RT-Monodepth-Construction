# 🎯 Quick Start: Colab H100 Training

**5-minute checklist to get training started**

---

## ✅ Pre-Flight Checklist

### 1️⃣ **Validate Dataset** (5 minutes)

```bash
cd /Users/ashwani/Desktop/RT-Monodepth-Construction/yolo12-training/colab_setup
python validate_dataset.py
```

**Expected output:**
```
✅ ALL CHECKS PASSED!
   Dataset is ready for Colab H100 training.
```

If you see errors, fix them before proceeding.

---

### 2️⃣ **Zip Dataset** (10-15 minutes)

```bash
cd /Users/ashwani/Desktop/RT-Monodepth-Construction/yolo12-training
zip -r merged_construction_safety.zip merged_construction_safety/
```

**Verify zip:**
```bash
ls -lh merged_construction_safety.zip
# Should be ~25-30 GB
```

---

### 3️⃣ **Upload to Google Drive** (1-2 hours)

1. Go to [Google Drive](https://drive.google.com)
2. Create folder structure:
   ```
   MyDrive/
   └── YOLOv12_Training/
       └── merged_construction_safety.zip  ← Upload here
   ```
3. Upload `merged_construction_safety.zip`
4. **Wait for upload to complete** (DO NOT close browser!)

**Tip:** Use Google Drive desktop app for faster upload:
- [Download Google Drive](https://www.google.com/drive/download/)
- Copy zip file to `Google Drive` folder
- It will auto-sync in background

---

### 4️⃣ **Subscribe to Colab Pro+** (1 minute)

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **Upgrade** → **Subscribe to Pro+** ($50/month)
3. Confirm payment

**Why Pro+?**
- H100 GPU access (80GB VRAM, 15x faster than free T4)
- Training time: 4-8 hours (vs 48+ hours on free tier)
- Total cost: ~$10-20 for this project (cancel after 1 month)

---

### 5️⃣ **Upload Notebook to Colab** (1 minute)

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File** → **Upload notebook**
3. Select: `train_yolo12n_h100.ipynb`
4. Notebook opens in new tab

---

### 6️⃣ **Allocate H100 GPU** (1 minute)

1. In Colab, click **Runtime** → **Change runtime type**
2. Settings:
   - **Runtime type:** Python 3
   - **Hardware accelerator:** GPU
   - **GPU type:** **H100** ← SELECT THIS
   - **Runtime shape:** High-RAM
3. Click **Save**

**Important:** H100 is not always available. If you don't see it:
- Try again in 30 minutes (demand varies by time of day)
- Use A100 as fallback (slower but still good)
- **Never use T4/V100** - training will take 10-20x longer

---

### 7️⃣ **Run Training** (4-8 hours)

1. In Colab notebook, click **Runtime** → **Run all**
2. Cells will execute in order:
   - ✅ GPU verification
   - ✅ Install Ultralytics
   - ✅ Mount Google Drive
   - ✅ Extract dataset (10-15 min)
   - ✅ Train model (4-8 hours)
   - ✅ Evaluate results
   - ✅ Save to Drive

3. **DO NOT CLOSE THE TAB!** If you do, training will stop.
   - Keep tab open in background
   - Your computer can sleep (training runs on Google's servers)
   - Check progress every 1-2 hours

---

### 8️⃣ **Monitor Progress** (during training)

**Live metrics:** Check the training cell output for:
```
Epoch    GPU_mem    box_loss    cls_loss    dfl_loss    Instances    Size
1/100    12.5G      1.234       0.567       0.890       123          640
```

**What to look for:**
- `box_loss`, `cls_loss`, `dfl_loss` should **decrease** over time
- `GPU_mem` should stay under 80GB (you're fine if ~40-60GB)
- `Instances` shows objects per batch

**Progress estimate:**
- Each epoch: ~2-4 minutes on H100
- 100 epochs: ~4-8 hours total
- You'll see periodic saves every 10 epochs

---

### 9️⃣ **Download Results** (after training)

**Option 1: Direct download from Colab (fastest)**
```python
# Run the last cell in notebook
# Downloads best.pt to your computer
```

**Option 2: From Google Drive**
1. Go to Google Drive
2. Navigate to: `MyDrive/YOLOv12_Training/results/construction_safety_yolo12n/weights/`
3. Download files:
   - `best.pt` (best model, ~6-10 MB)
   - `last.pt` (final epoch)

**Files to review:**
- `best.pt` - Use this for inference!
- `results.png` - Training curves
- `confusion_matrix.png` - Per-class performance
- `F1_curve.png`, `PR_curve.png` - Model quality metrics

---

## 🎉 Success Criteria

Training is successful if:

✅ **mAP50 > 0.50** (50%+ accuracy at IoU=0.5)
✅ **mAP50-95 > 0.30** (30%+ across all IoU thresholds)
✅ **Loss curves decrease smoothly** (no spikes)
✅ **No major class imbalances** in confusion matrix

---

## 🚨 Troubleshooting Quick Fixes

### ❌ "H100 not available"
**Fix:** Wait 30 min and try again, or use A100 (change `batch=32`)

### ❌ "Dataset not found"
**Fix:** Check Google Drive path: `MyDrive/YOLOv12_Training/merged_construction_safety.zip`

### ❌ "CUDA out of memory"
**Fix:** Reduce batch size in training config cell:
```python
'batch': 32,  # Was 64
```

### ❌ "Training stopped halfway"
**Fix:** Resume from checkpoint:
```python
model = YOLO('/content/runs/detect/construction_safety_yolo12n/weights/last.pt')
results = model.train(resume=True)
```

### ❌ "Extraction taking forever"
**Fix:** This is normal! 25-30GB dataset takes 10-15 min to extract.

### ❌ "Poor mAP (< 0.3)"
**Fix:** 
1. Check class distribution (imbalanced?)
2. Increase epochs to 150-200
3. Adjust augmentation parameters
4. Verify label quality

---

## 💰 Cost Breakdown

| Item | Cost | Notes |
|------|------|-------|
| Colab Pro+ (1 month) | $50 | Cancel anytime |
| H100 GPU usage | $2-5/hr | Included in Pro+ |
| Google Drive storage | Free | If < 15GB, or $2/mo for 100GB |
| **Total** | **$50-52** | **For one training run** |

**Tip:** Cancel Pro+ after training to avoid recurring charge.

---

## ⏱️ Time Estimate

| Task | Duration | Can Parallelize? |
|------|----------|------------------|
| Validate dataset | 5 min | ❌ |
| Zip dataset | 10-15 min | ❌ |
| Upload to Drive | 1-2 hours | ✅ (use Drive desktop app) |
| Subscribe Colab Pro+ | 1 min | ❌ |
| Setup notebook | 5 min | ❌ |
| Extract dataset | 10-15 min | ❌ |
| **Train model** | **4-8 hours** | ❌ (keep tab open) |
| Download results | 5 min | ❌ |
| **Total (end-to-end)** | **6-12 hours** | |

**Most time is automated** - you only need to be present for ~30 min total!

---

## 📞 Support

If you encounter issues:

1. **Check cell outputs** - Error messages are detailed
2. **Re-run failed cell** - Sometimes works on retry
3. **Restart runtime** - Runtime → Restart runtime
4. **Check H100 availability** - Try different time of day

**Common fixes:**
- 90% of errors: wrong GPU type or missing dataset
- 9% of errors: batch size too large (reduce)
- 1% of errors: actual bugs (file an issue)

---

## ✨ Next Steps After Training

1. **Copy `best.pt` to project:**
   ```bash
   cp ~/Downloads/best.pt /Users/ashwani/Desktop/RT-Monodepth-Construction/
   ```

2. **Update inference script:**
   ```python
   # In realtime_depth_video.py
   yolo_model = YOLO('best.pt')  # Use your trained model
   ```

3. **Test on construction video:**
   ```bash
   python realtime_depth_video.py --video construction_site.mp4
   ```

4. **Evaluate on real data:**
   - Run on 10-20 construction videos
   - Check if person and equipment are detected
   - Adjust confidence threshold if needed

---

**Ready to start?** Run the validation script first! ✅

```bash
python validate_dataset.py
```
