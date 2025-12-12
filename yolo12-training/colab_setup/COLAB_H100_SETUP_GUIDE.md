# 🚀 Google Colab H100 Training Setup Guide

**Complete guide for training YOLOv12n on H100 GPU**

---

## 📋 **Table of Contents**

1. [Prerequisites](#prerequisites)
2. [Step 1: Prepare Dataset](#step-1-prepare-dataset)
3. [Step 2: Upload to Google Drive](#step-2-upload-to-google-drive)
4. [Step 3: Get Colab Pro+](#step-3-get-colab-pro)
5. [Step 4: Setup Notebook](#step-4-setup-notebook)
6. [Step 5: Run Training](#step-5-run-training)
7. [Step 6: Monitor Progress](#step-6-monitor-progress)
8. [Step 7: Download Results](#step-7-download-results)
9. [Troubleshooting](#troubleshooting)
10. [Cost Breakdown](#cost-breakdown)

---

## **Prerequisites**

### ✅ **What You Need:**

1. **Google Account** (for Drive and Colab)
2. **Colab Pro+ subscription** ($50/month for H100 access)
3. **Merged dataset** (already created: `merged_construction_safety/`)
4. **Good internet connection** (for uploading ~25GB dataset)
5. **~4-5 hours** (dataset upload + training time)

### ⚡ **Why H100?**

| GPU | VRAM | Speed | Training Time | Cost/Hour | Availability |
|-----|------|-------|---------------|-----------|--------------|
| T4 (Free) | 16GB | 1x | 12-18 hours | Free | Common |
| V100 (Pro) | 16GB | 3x | 8-10 hours | $0.50 | Common |
| A100 (Pro) | 40GB | 8x | 5-6 hours | $1.00 | Moderate |
| **H100 (Pro+)** | **80GB** | **15x** | **3-4 hours** | **$2.00** | **Rare** |

**H100 is 15x faster than T4!** Worth the subscription for this project.

---

## **Step 1: Prepare Dataset**

### **1.1 Verify merged dataset exists**

```bash
cd /Users/ashwani/Desktop/RT-Monodepth-Construction/yolo12-training
ls -lh merged_construction_safety/
```

**Expected output:**
```
drwxr-xr-x  4 user  staff   128B  train
drwxr-xr-x  4 user  staff   128B  valid
drwxr-xr-x  4 user  staff   128B  test
-rw-r--r--  1 user  staff   1.2K  data.yaml
```

### **1.2 Compress the dataset**

```bash
cd /Users/ashwani/Desktop/RT-Monodepth-Construction/yolo12-training

# Create zip (this will take ~10-15 minutes)
zip -r merged_construction_safety.zip merged_construction_safety/
```

**Expected size:** ~25-30GB compressed

### **1.3 Verify zip file**

```bash
ls -lh merged_construction_safety.zip
unzip -l merged_construction_safety.zip | head -20  # Preview contents
```

---

## **Step 2: Upload to Google Drive**

### **2.1 Create folder structure**

1. Go to https://drive.google.com
2. Click **"New"** → **"New folder"**
3. Name it: `YOLOv12_Training`
4. Open the folder

### **2.2 Upload the zip file**

**Method A: Web Upload (Simple)**
1. Click **"New"** → **"File upload"**
2. Select `merged_construction_safety.zip`
3. Wait for upload (1-3 hours depending on internet)

**Method B: Google Drive Desktop App (Faster)**
1. Install Google Drive for Desktop
2. Copy `merged_construction_safety.zip` to Google Drive folder
3. Wait for sync

**Method C: Command Line (Mac)**
```bash
# Install rclone
brew install rclone

# Configure Google Drive
rclone config

# Upload
rclone copy merged_construction_safety.zip gdrive:YOLOv12_Training/
```

### **2.3 Verify upload**

- Check file size in Google Drive
- Should be ~25-30GB
- Click on file → **"Get link"** → Make sure it's accessible

---

## **Step 3: Get Colab Pro+**

### **3.1 Subscribe to Colab Pro+**

1. Go to https://colab.research.google.com
2. Click **"Upgrade"** button (top right)
3. Choose **"Colab Pro+"** ($49.99/month)
4. Complete payment

### **3.2 Benefits of Colab Pro+:**

✅ **H100 GPU access** (80GB VRAM)  
✅ **Background execution** (keeps running when tab closed)  
✅ **Longer runtimes** (24 hours instead of 12)  
✅ **Priority access** (faster queue times)  
✅ **More compute units** (can run multiple notebooks)

### **3.3 Verify subscription**

- Go to Colab
- Check for "Pro+" badge next to your profile
- Should see "Compute units: High priority"

---

## **Step 4: Setup Notebook**

### **4.1 Upload notebook to Colab**

**Option A: Upload .ipynb file**
1. Go to https://colab.research.google.com
2. Click **"File"** → **"Upload notebook"**
3. Select `train_yolo12n_h100.ipynb`

**Option B: Open from Drive**
1. Copy `train_yolo12n_h100.ipynb` to Google Drive
2. Right-click → **"Open with"** → **"Google Colaboratory"**

### **4.2 Select H100 Runtime**

1. Click **"Runtime"** → **"Change runtime type"**
2. **Hardware accelerator:** GPU
3. **GPU type:** **H100** (if available, otherwise A100)
4. **Runtime shape:** High-RAM
5. Click **"Save"**

**Note:** H100 may not always be available. If you don't see it:
- Try during off-peak hours (late night US time)
- A100 is still very fast (5-6 hours vs 3-4 hours)
- Free up other Colab sessions to get priority

### **4.3 Verify GPU allocation**

Run first cell in notebook:
```python
!nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA H100 PCIe    Off  | 00000000:00:04.0 Off |                    0 |
| N/A   32C    P0    44W / 300W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
```

✅ Look for "H100" in GPU name

---

## **Step 5: Run Training**

### **5.1 Execute cells in order**

Run each cell by clicking the **play button** or pressing `Shift+Enter`:

1. ✅ **Cell 1:** Check GPU (verify H100)
2. ✅ **Cell 2:** Mount Google Drive (authorize access)
3. ✅ **Cell 3:** Install dependencies (~2 min)
4. ✅ **Cell 4:** Extract dataset (~5 min)
5. ✅ **Cell 5:** Verify dataset (~1 min)
6. ✅ **Cell 6:** Start training (**~3-4 hours on H100**)
7. ✅ **Cell 7:** View results (after training)
8. ✅ **Cell 8:** Download model
9. ✅ **Cell 9:** Test inference (optional)
10. ✅ **Cell 10:** TensorBoard (optional)

### **5.2 Cell 6 parameters (H100 optimized)**

```python
batch=128        # H100 can handle large batches
imgsz=640        # Full resolution
epochs=100       # Complete training
cache='disk'     # Fast caching
workers=8        # Parallel data loading
device=0         # GPU 0
amp=True         # FP16 mixed precision
```

### **5.3 What to expect during training**

**Epoch 1:**
```
Epoch 1/100: 100%|██████████| 242/242 [00:45<00:00, 5.31it/s]
                 Class     Images  Instances      P      R      mAP50   mAP50-95
                   all       4659      12456  0.512  0.423      0.445      0.312
```

**Epoch 50:**
```
Epoch 50/100: 100%|██████████| 242/242 [00:42<00:00, 5.71it/s]
                 Class     Images  Instances      P      R      mAP50   mAP50-95
                   all       4659      12456  0.823  0.765      0.812      0.645
```

**Epoch 100 (Final):**
```
Epoch 100/100: 100%|██████████| 242/242 [00:41<00:00, 5.85it/s]
                 Class     Images  Instances      P      R      mAP50   mAP50-95
                   all       4659      12456  0.876  0.819      0.854      0.692
```

### **5.4 Training metrics to watch**

- **mAP@50:** Should reach 0.80-0.90 (80-90%)
- **mAP@50-95:** Should reach 0.65-0.75 (65-75%)
- **Loss:** Should decrease steadily
- **Precision:** Should increase to 0.85+ (85%+)
- **Recall:** Should increase to 0.80+ (80%+)

---

## **Step 6: Monitor Progress**

### **6.1 Real-time monitoring**

**TensorBoard (Cell 10):**
- Click the TensorBoard link in output
- See live training curves
- Monitor GPU utilization
- View sample predictions

**Colab output:**
- Shows epoch progress
- Displays metrics after each epoch
- Shows estimated time remaining

### **6.2 Check intermediate checkpoints**

Every 10 epochs, model is saved:
```
/content/drive/MyDrive/YOLOv12_Training/runs/yolo12n_construction_h100/weights/
├── best.pt      # Best model so far
├── last.pt      # Latest checkpoint
├── epoch10.pt   # Checkpoint at epoch 10
├── epoch20.pt   # Checkpoint at epoch 20
└── ...
```

### **6.3 Session management**

**Colab Pro+ features:**
- ✅ Background execution (can close tab)
- ✅ Auto-reconnect if disconnected
- ✅ 24-hour runtime limit

**To enable background execution:**
1. Click **"Runtime"** → **"Manage sessions"**
2. Enable **"Background execution"**

---

## **Step 7: Download Results**

### **7.1 Download trained model**

**After training completes, run Cell 8:**
```python
files.download(best_model)
```

**This downloads:**
- `best.pt` - Your trained YOLOv12n model (~5MB)

### **7.2 Also available in Google Drive**

All results are automatically saved to:
```
Google Drive/YOLOv12_Training/runs/yolo12n_construction_h100/
├── weights/
│   ├── best.pt           ⭐ Main model
│   └── last.pt           Last checkpoint
├── results.csv           All metrics
├── results.png           Training curves
├── confusion_matrix.png  Class performance
├── PR_curve.png          Precision-Recall
├── F1_curve.png          F1 scores
└── predictions/          Sample predictions
```

### **7.3 Download to your Mac**

**Method A: Google Drive web**
1. Go to Drive → YOLOv12_Training/runs/...
2. Right-click `best.pt` → Download

**Method B: Google Drive Desktop**
1. Sync automatically to Mac
2. Find in: `~/Google Drive/YOLOv12_Training/runs/...`

**Method C: Command line**
```bash
# Using rclone
rclone copy gdrive:YOLOv12_Training/runs/yolo12n_construction_h100/weights/best.pt ~/Desktop/
```

### **7.4 Rename and use**

```bash
cd ~/Desktop
mv best.pt yolo12n_construction_best.pt

# Copy to project
cp yolo12n_construction_best.pt /Users/ashwani/Desktop/RT-Monodepth-Construction/

# Test inference
yolo predict model=yolo12n_construction_best.pt source=test_image.jpg
```

---

## **Troubleshooting**

### **Issue 1: H100 not available**

**Symptoms:**
- Only see T4, V100, or A100 in runtime options
- "H100 is not currently available"

**Solutions:**
1. ✅ Try during off-peak hours (2-6 AM US Eastern)
2. ✅ Close all other Colab sessions
3. ✅ Wait and refresh (H100 pools free up)
4. ✅ Use A100 instead (still very fast, 5-6 hours)

### **Issue 2: Session timeout**

**Symptoms:**
- "Session crashed" or "Runtime disconnected"
- Training stops mid-epoch

**Solutions:**
1. ✅ Enable background execution
2. ✅ Upgrade to Colab Pro+ (24hr sessions)
3. ✅ Resume from last checkpoint:
   ```python
   model.train(resume=True)
   ```

### **Issue 3: Out of memory**

**Symptoms:**
- "CUDA out of memory"
- Training crashes

**Solutions:**
1. ✅ Reduce batch size (128 → 64 → 32)
2. ✅ Reduce image size (640 → 512)
3. ✅ Disable cache:
   ```python
   cache=False
   ```

### **Issue 4: Slow upload to Drive**

**Symptoms:**
- Dataset upload takes >4 hours
- Upload keeps failing

**Solutions:**
1. ✅ Use Google Drive Desktop app
2. ✅ Upload overnight
3. ✅ Use faster internet connection
4. ✅ Split zip into smaller parts:
   ```bash
   split -b 4GB merged_construction_safety.zip part_
   ```

### **Issue 5: Dataset extraction slow**

**Symptoms:**
- Extraction taking >10 minutes

**Solutions:**
1. ✅ Pre-extract in Google Drive (run Cell 4 once, then skip)
2. ✅ Use `unzip` instead of Python zipfile
3. ✅ Normal on first run (5 mins is OK)

### **Issue 6: Training not starting**

**Symptoms:**
- Cell 6 runs but nothing happens
- No epoch progress

**Solutions:**
1. ✅ Check GPU is allocated (`nvidia-smi`)
2. ✅ Verify data.yaml path is correct
3. ✅ Check dataset was extracted fully
4. ✅ Restart runtime and try again

---

## **Cost Breakdown**

### **Colab Pro+ Subscription**

| Item | Cost | Duration |
|------|------|----------|
| Colab Pro+ | $49.99/month | Monthly |
| **Total** | **$49.99** | **1 month** |

### **Training Cost Estimate**

| GPU | Training Time | Cost/Hour | Total Cost |
|-----|---------------|-----------|------------|
| H100 | 3-4 hours | ~$2.00 | **~$6-8** |
| A100 | 5-6 hours | ~$1.00 | **~$5-6** |
| V100 | 8-10 hours | ~$0.50 | **~$4-5** |
| T4 (Free) | 12-18 hours | Free | **Free** (but may timeout) |

**Note:** Colab Pro+ gives you compute units, not charged per hour. Above costs are effective estimates.

### **Total Project Cost**

```
Colab Pro+ subscription:     $50.00
Training (included):         $0.00 (within subscription)
─────────────────────────────────────
Total:                       $50.00 for 1 month
```

**You can cancel after training** if you don't need it ongoing.

### **Value Comparison**

| Option | Cost | Time | Pros | Cons |
|--------|------|------|------|------|
| **Mac M1 Pro** | Free | 96 days | No cost | Unusably slow |
| **AWS g5.xlarge** | ~$20 | 6-8 hours | Pay-per-use | Setup complexity |
| **Colab Free** | Free | 12-18 hours | Free | May timeout |
| **Colab Pro** | $10 | 6-8 hours | Affordable | Slower than H100 |
| **Colab Pro+ (H100)** | $50 | **3-4 hours** | **Fastest** | Most expensive |

**Recommendation:** Get Colab Pro+ for 1 month, train model, then cancel. Total cost: $50 for a complete training run.

---

## **Expected Results**

### **Training Performance (H100)**

```
Epoch 1:   ~45 seconds/epoch
Epoch 50:  ~42 seconds/epoch
Epoch 100: ~41 seconds/epoch

Total training time: 3.5 hours
```

### **Model Performance (Expected)**

```
Final Metrics (Epoch 100):
├── mAP@50:      0.85-0.90 (85-90%)
├── mAP@50-95:   0.68-0.75 (68-75%)
├── Precision:   0.87-0.92 (87-92%)
├── Recall:      0.82-0.88 (82-88%)
└── F1-Score:    0.85-0.90 (85-90%)
```

### **Per-Class Performance (Top Classes)**

```
Class 0 (Dump truck):     mAP@50-95: 0.72
Class 1 (Excavator):      mAP@50-95: 0.78
Class 2 (Motor grader):   mAP@50-95: 0.69
...
Class 17 (Person):        mAP@50-95: 0.81 ⭐
```

---

## **Next Steps After Training**

### **1. Evaluate on Test Set**

```python
from ultralytics import YOLO

model = YOLO('yolo12n_construction_best.pt')
results = model.val(data='merged_construction_safety/data.yaml', split='test')
```

### **2. Export for Deployment**

```bash
# Export to ONNX (cross-platform)
yolo export model=yolo12n_construction_best.pt format=onnx

# Export to TensorRT (NVIDIA GPUs)
yolo export model=yolo12n_construction_best.pt format=engine device=0

# Export to CoreML (Apple Silicon)
yolo export model=yolo12n_construction_best.pt format=coreml
```

### **3. Integrate with RT-MonoDepth**

```python
# In your RT-MonoDepth pipeline
from ultralytics import YOLO

# Load models
depth_model = RTMonoDepth(...)
yolo_model = YOLO('yolo12n_construction_best.pt')

# Run inference
depth_map = depth_model(image)
detections = yolo_model(image)

# Combine for safety monitoring
for detection in detections:
    bbox = detection.bbox
    object_depth = depth_map[bbox]
    if object_depth < safety_threshold:
        alert("Danger: Person too close to equipment!")
```

### **4. Run on Construction Videos**

```bash
yolo predict model=yolo12n_construction_best.pt source=construction_site.mp4 save=True
```

---

## **Summary Checklist**

### **Before Training:**
- [ ] Dataset zipped: `merged_construction_safety.zip`
- [ ] Uploaded to Google Drive: `YOLOv12_Training/`
- [ ] Colab Pro+ subscription active
- [ ] Notebook uploaded to Colab

### **During Training:**
- [ ] H100/A100 GPU allocated
- [ ] Drive mounted successfully
- [ ] Dataset extracted (~5 min)
- [ ] Training started (Cell 6)
- [ ] Monitoring with TensorBoard (optional)

### **After Training:**
- [ ] Training completed (100 epochs)
- [ ] best.pt downloaded
- [ ] Results reviewed (mAP, confusion matrix)
- [ ] Model tested on sample images
- [ ] Saved to Google Drive (backup)

---

## **Support**

If you encounter issues:

1. ✅ Check this troubleshooting section
2. ✅ Review Colab output logs
3. ✅ Try restarting runtime
4. ✅ Check Google Colab status page
5. ✅ Contact support if persistent

---

**Good luck with training! The H100 will have your model ready in ~4 hours! 🚀**

---

**Created:** December 12, 2025  
**Author:** RT-MonoDepth-Construction Project  
**Version:** 1.0  
**GPU:** H100 (Colab Pro+)
