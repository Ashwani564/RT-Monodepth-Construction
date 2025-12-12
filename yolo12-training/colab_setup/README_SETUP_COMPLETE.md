# ✅ Colab H100 Training Setup - COMPLETE

**Status:** All files ready for Google Colab H100 training  
**Date:** Generated automatically  
**Dataset:** 36,623 images (18 classes: 17 construction equipment + 1 person)

---

## 📁 Files Created

All files are in: `/Users/ashwani/Desktop/RT-Monodepth-Construction/yolo12-training/colab_setup/`

| File | Purpose | Size |
|------|---------|------|
| `train_yolo12n_h100.ipynb` | Main Colab training notebook | Complete |
| `COLAB_H100_SETUP_GUIDE.md` | Detailed setup guide (620 lines) | Complete |
| `QUICK_START.md` | 5-minute quick start checklist | Complete |
| `validate_dataset.py` | Dataset validation script | Complete |
| `README_SETUP_COMPLETE.md` | This file | Complete |

---

## ✅ Dataset Validation Results

```
Dataset: merged_construction_safety/
├── Training images:    31,964
├── Validation images:   4,659
├── Total images:       36,623
├── Training labels:    30,032
├── Validation labels:   4,598
├── Classes:            18 (verified)
└── Estimated size:     18.3 GB (unzipped)
```

**Class Distribution (sample of 100 labels):**
- Class 17 (person): 297 annotations ✓
- Class 1 (Excavator): 46 annotations
- Class 0 (Dump truck): 19 annotations
- Class 5 (Gazelle): 11 annotations
- Class 6 (Forklift): 8 annotations
- ... (13 more equipment classes)

**Warnings:**
- ~6% of images have no labels (normal for background/negative samples)
- Training can proceed without issues

---

## 🚀 What to Do Next (Step by Step)

### **STEP 1: Zip the Dataset** (Required before upload)

```bash
cd /Users/ashwani/Desktop/RT-Monodepth-Construction/yolo12-training
zip -r merged_construction_safety.zip merged_construction_safety/
```

**Expected output:**
```
adding: merged_construction_safety/ (stored 0%)
adding: merged_construction_safety/train/ (stored 0%)
adding: merged_construction_safety/train/images/ (stored 0%)
...
```

**This will take 10-15 minutes.**

Verify the zip file:
```bash
ls -lh merged_construction_safety.zip
# Should show ~18-20 GB
```

---

### **STEP 2: Upload to Google Drive** (1-2 hours)

**Option A: Via Web Browser (Slower)**
1. Go to [Google Drive](https://drive.google.com)
2. Click "New" → "Folder" → Name it: `YOLOv12_Training`
3. Open the folder
4. Click "Upload" → Select `merged_construction_safety.zip`
5. **Wait for upload to complete** (DO NOT close browser!)

**Option B: Via Google Drive Desktop App (Faster, Recommended)**
1. Install [Google Drive for Desktop](https://www.google.com/drive/download/)
2. Sign in with your Google account
3. Create folder: `Google Drive/YOLOv12_Training/`
4. Copy `merged_construction_safety.zip` into that folder
5. It will auto-sync in the background (faster than web upload)

**Upload time:** 1-2 hours (depends on your internet speed)

---

### **STEP 3: Subscribe to Colab Pro+** ($50/month)

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **Upgrade** → **Subscribe to Colab Pro+**
3. Confirm payment ($50/month, cancel anytime)

**Why Pro+?**
- H100 GPU: 80GB VRAM, 15x faster than free T4
- Training time: 4-8 hours (vs 48+ hours on free tier)
- High-RAM runtime (important for caching)

**Cost estimate for this project:**
- Subscribe: $50 (first month)
- H100 usage: Included in Pro+
- **Total: ~$50** (cancel after training if you want)

---

### **STEP 4: Upload Notebook to Colab**

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File** → **Upload notebook**
3. Select: `/Users/ashwani/Desktop/RT-Monodepth-Construction/yolo12-training/colab_setup/train_yolo12n_h100.ipynb`
4. Notebook opens in new tab

---

### **STEP 5: Allocate H100 GPU**

1. In Colab, click **Runtime** → **Change runtime type**
2. Configure:
   - **Runtime type:** Python 3
   - **Hardware accelerator:** GPU
   - **GPU type:** **H100** ← SELECT THIS
   - **Runtime shape:** High-RAM
3. Click **Save**
4. Click **Connect** (top right)

**Important:** H100 is not always available. If you don't see it:
- Try again in 30 minutes (demand varies by time of day)
- Try early morning (US time) or late evening
- Use A100 as fallback (slower but still good)

---

### **STEP 6: Run Training** (4-8 hours)

1. In Colab notebook, click **Runtime** → **Run all**
2. Watch the cells execute:
   - Cell 1: GPU verification (should show H100)
   - Cell 2: Install Ultralytics
   - Cell 3: Mount Google Drive (will ask for permission)
   - Cell 4: Extract dataset (10-15 min)
   - Cell 5-7: Setup
   - **Cell 8: Training** (4-8 hours) ← Main training loop
   - Cell 9-11: Evaluation and visualization
   - Cell 12: Save to Google Drive

3. **DO NOT CLOSE THE TAB!** Keep it open in the background.
   - Your computer can sleep (training runs on Google's servers)
   - Check progress every 1-2 hours

---

### **STEP 7: Monitor Training Progress**

**What to watch for in Cell 8 output:**

```
Epoch    GPU_mem    box_loss    cls_loss    dfl_loss    Instances    Size
1/100    12.5G      1.234       0.567       0.890       123          640
2/100    45.2G      1.123       0.543       0.845       118          640
3/100    45.1G      1.045       0.521       0.812       125          640
...
```

**Good signs:**
- ✅ `box_loss`, `cls_loss`, `dfl_loss` **decrease** over time
- ✅ `GPU_mem` stays under 80GB (typically 40-60GB is normal)
- ✅ Each epoch takes 2-4 minutes
- ✅ Losses stabilize after 30-50 epochs

**Bad signs (need to fix):**
- ❌ `GPU_mem` says "CUDA out of memory" → Reduce batch size to 32
- ❌ Losses increase or spike → Learning rate too high
- ❌ Training very slow (>10 min/epoch) → Not using H100 GPU

---

### **STEP 8: Download Results** (After training completes)

**Automatic download (easiest):**
- Cell 13 will automatically download `best.pt` to your computer

**Manual download from Google Drive:**
1. Go to Google Drive
2. Navigate to: `YOLOv12_Training/results/construction_safety_yolo12n/weights/`
3. Download:
   - `best.pt` (best model, ~6-10 MB) ← Use this!
   - `last.pt` (final epoch, for resuming)

**Review training results:**
- `results.png` - Loss curves over epochs
- `confusion_matrix.png` - Per-class performance
- `F1_curve.png` - F1 score at different confidence thresholds
- `PR_curve.png` - Precision-Recall curve

---

## 📊 Expected Results

**Benchmark targets:**

| Metric | Target | What it means |
|--------|--------|---------------|
| mAP50 | > 0.50 | 50%+ accuracy at IoU=0.5 |
| mAP50-95 | > 0.30 | 30%+ accuracy across all IoU |
| Precision | > 0.60 | 60%+ of detections are correct |
| Recall | > 0.50 | 50%+ of objects are detected |

**Training should achieve:**
- ✅ **mAP50: 0.55-0.70** (Good performance)
- ✅ **mAP50-95: 0.35-0.50** (Excellent across all IoU thresholds)
- ✅ Loss curves: Smooth decrease, plateau after 70-80 epochs
- ✅ Confusion matrix: Strong diagonal (correct predictions)

---

## 🎯 Using the Trained Model

### **1. Copy model to project**

```bash
cp ~/Downloads/best.pt /Users/ashwani/Desktop/RT-Monodepth-Construction/
```

### **2. Test with YOLO CLI**

```bash
cd /Users/ashwani/Desktop/RT-Monodepth-Construction

# Test on single image
yolo detect predict model=best.pt source=test_image.jpg conf=0.25

# Test on video
yolo detect predict model=best.pt source=construction_video.mp4 conf=0.25
```

### **3. Integrate with RT-MonoDepth pipeline**

Update `realtime_depth_video.py`:

```python
# Replace this line:
yolo_model = YOLO('yolo11n.pt')  # Old pretrained model

# With this:
yolo_model = YOLO('best.pt')  # Your trained model
```

### **4. Adjust confidence threshold if needed**

```python
# If too many false positives, increase confidence:
results = yolo_model.predict(frame, conf=0.35, iou=0.45)  # Was 0.25

# If missing detections, decrease confidence:
results = yolo_model.predict(frame, conf=0.15, iou=0.45)  # Was 0.25
```

---

## 🚨 Troubleshooting

### **Issue 1: "H100 not available"**

**Solution:**
- Wait 30 min and try again
- Try different time of day (early morning US time is best)
- Use A100 instead (adjust batch size to 32)
- Check Colab Pro+ subscription is active

---

### **Issue 2: "Dataset not found" in Colab**

**Solution:**
1. Check Google Drive path is exactly: `MyDrive/YOLOv12_Training/merged_construction_safety.zip`
2. Make sure upload is 100% complete (don't interrupt)
3. In Colab Cell 3, verify the path matches
4. Re-run Cell 3 to remount Drive

---

### **Issue 3: "CUDA out of memory"**

**Solution:**
Edit Cell 7 training config:
```python
'batch': 32,  # Reduce from 64
'cache': False,  # Disable caching if still failing
```

Then re-run Cell 8 (training).

---

### **Issue 4: Training stopped halfway**

**Solution:**
Resume from checkpoint:
```python
# In a new cell:
from ultralytics import YOLO
model = YOLO('/content/runs/detect/construction_safety_yolo12n/weights/last.pt')
results = model.train(resume=True)
```

---

### **Issue 5: Poor mAP (< 0.30)**

**Possible causes:**
1. **Imbalanced classes** → Check confusion matrix
2. **Not enough epochs** → Increase to 150-200
3. **Bad labels** → Re-validate dataset
4. **Wrong hyperparameters** → Try different learning rate

**Quick fix:**
```python
# Increase epochs and adjust learning rate
'epochs': 150,
'lr0': 0.0005,  # Lower learning rate
```

---

### **Issue 6: Extraction taking too long (>30 min)**

**Normal:** 18GB dataset takes 10-15 min to extract.

**If >30 min:**
- This is still normal for large datasets
- Do NOT interrupt! Let it finish
- Colab's SSD is fast, but 36K images take time

---

## 💰 Cost Summary

| Item | Cost | Notes |
|------|------|-------|
| Colab Pro+ subscription | $50/month | Required for H100 |
| H100 GPU compute | Included | Part of Pro+ |
| Google Drive storage (100GB plan) | $2/month | Optional (free if <15GB) |
| **Total for this project** | **$50-52** | **Cancel Pro+ after training** |

**Tips to save money:**
1. Cancel Colab Pro+ after training (you can re-subscribe anytime)
2. Delete dataset from Drive after training (saves storage costs)
3. Train multiple models in one month if planning more experiments

---

## ⏱️ Total Time Estimate

| Phase | Duration | Active Work Required |
|-------|----------|---------------------|
| Zip dataset | 10-15 min | ❌ Automated |
| Upload to Drive | 1-2 hours | ❌ Automated (keep browser open) |
| Setup Colab | 5 min | ✅ Manual |
| Extract dataset in Colab | 10-15 min | ❌ Automated |
| **Training** | **4-8 hours** | ❌ **Automated (keep tab open)** |
| Evaluation & download | 10 min | ✅ Manual |
| **Total** | **6-11 hours** | **~20 min of manual work** |

**You only need to be present for ~20 minutes total!** Rest is automated.

---

## 📚 Reference Documentation

1. **Quick Start Guide:** `QUICK_START.md` (5-min checklist)
2. **Detailed Setup Guide:** `COLAB_H100_SETUP_GUIDE.md` (comprehensive 620-line guide)
3. **Training Notebook:** `train_yolo12n_h100.ipynb` (Colab notebook)
4. **Validation Script:** `validate_dataset.py` (dataset checker)

---

## ✅ Pre-Flight Checklist

Before starting Colab training, verify:

- [x] Dataset validated (ran `python validate_dataset.py`)
- [ ] Dataset zipped (`merged_construction_safety.zip` created)
- [ ] Uploaded to Google Drive (`YOLOv12_Training/merged_construction_safety.zip`)
- [ ] Subscribed to Colab Pro+ ($50/month)
- [ ] Notebook uploaded to Colab (`train_yolo12n_h100.ipynb`)
- [ ] H100 GPU allocated in Colab
- [ ] Ready to run training (4-8 hours)

---

## 🎉 Success Metrics

Training is successful if you achieve:

✅ **mAP50 > 0.50** (50%+ detection accuracy)  
✅ **mAP50-95 > 0.30** (30%+ across all IoU thresholds)  
✅ **Loss curves decrease smoothly** (no major spikes)  
✅ **Person class detects well** (check confusion matrix)  
✅ **Model file size ~6-10 MB** (YOLOv12n is small)  

---

## 🔄 What Happens Next

After training completes:

1. **Model saved to Google Drive:**
   - `YOLOv12_Training/results/construction_safety_yolo12n/weights/best.pt`

2. **Download model to local machine:**
   - Copy to: `/Users/ashwani/Desktop/RT-Monodepth-Construction/best.pt`

3. **Integrate with RT-MonoDepth:**
   - Update `realtime_depth_video.py` to use `best.pt`
   - Test on construction site videos
   - Verify person + equipment detection works

4. **Evaluate on real data:**
   - Run on 10-20 construction videos
   - Measure FPS (should be 30-60 FPS on good GPU)
   - Check if all 18 classes are detected correctly

5. **Optional fine-tuning:**
   - If mAP < 0.50, train for more epochs (150-200)
   - If specific classes perform poorly, add more training data
   - Adjust confidence threshold based on use case

---

## 📞 Need Help?

If you encounter issues:

1. **Check the cell outputs** - Error messages are detailed
2. **Re-run the failed cell** - Sometimes works on retry
3. **Restart runtime** - `Runtime → Restart runtime` in Colab
4. **Check H100 availability** - Try different time of day
5. **Review troubleshooting section** - Most common issues covered above

**Most common errors (90% of issues):**
- Wrong GPU type selected (not H100)
- Dataset not uploaded to correct Drive path
- Batch size too large for available VRAM

---

## 🎓 Learning Resources

- **Ultralytics YOLO Docs:** https://docs.ultralytics.com/
- **Colab Pro+ Info:** https://colab.research.google.com/signup
- **YOLO Training Guide:** https://docs.ultralytics.com/modes/train/
- **Model Export Guide:** https://docs.ultralytics.com/modes/export/

---

**Ready to start? Follow the steps above!** 🚀

**First step:** Zip the dataset (see STEP 1 above)

```bash
cd /Users/ashwani/Desktop/RT-Monodepth-Construction/yolo12-training
zip -r merged_construction_safety.zip merged_construction_safety/
```

---

**Estimated total cost:** $50-52  
**Estimated total time:** 6-11 hours (mostly automated)  
**Expected result:** `best.pt` model with mAP50 > 0.50  

**Good luck!** 🎯
