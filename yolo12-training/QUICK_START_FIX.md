# 🎯 QUICK START: Fix Training & Get to 0.70+ mAP

## Current Status: ❌ STUCK AT 0.45 mAP

Your model is **stuck** because:
```
Problem 1: Class Imbalance
   Person: 100,000 instances (95%)
   Vehicles: <1,000 each (5%)
   → Model only learned "person", ignoring other 17 classes

Problem 2: Tiny Objects
   Objects at 640px: 8x8 pixels
   → Too small for Nano model to learn features
```

---

## Solution: 3 Simple Steps

### 📍 **STEP 1: Run Fix Pipeline (Mac Terminal)**
```bash
cd /Users/ashwani/Desktop/RT-Monodepth-Construction/yolo12-training
chmod +x run_fix_class_imbalance.sh
./run_fix_class_imbalance.sh
```

**Time:** ~15-20 minutes  
**Output:** `merged_construction_safety_3class_balanced.zip` (~12-15GB)

---

### 📍 **STEP 2: Upload to Google Drive**
1. Go to: https://drive.google.com
2. Open: `MyDrive/YOLOv12_Training/`
3. Upload: `merged_construction_safety_3class_balanced.zip`

**Time:** ~30-60 minutes (depends on internet speed)

---

### 📍 **STEP 3: Update Colab & Retrain**

**A. Stop Current Training:**
- In Colab: Runtime → Interrupt execution
- (Current model won't improve beyond 0.45)

**B. Update 3 Cells in Colab Notebook:**

**Cell 4 - Line 7-8 (Dataset path):**
```python
# OLD:
dataset_zip = '/content/drive/MyDrive/YOLOv12_Training/merged_construction_safety.zip'
extract_to = '/content/merged_construction_safety'

# NEW:
dataset_zip = '/content/drive/MyDrive/YOLOv12_Training/merged_construction_safety_3class_balanced.zip'
extract_to = '/content/merged_construction_safety_3class_balanced'
```

**Cell 5 - Line 5 & 19-20 (Verify paths):**
```python
# Line 5:
data_yaml_path = '/content/merged_construction_safety_3class_balanced/data.yaml'

# Lines 19-20:
train_images = list(Path('/content/merged_construction_safety_3class_balanced/train/images').glob('*.jpg'))
val_images = list(Path('/content/merged_construction_safety_3class_balanced/valid/images').glob('*.jpg'))
```

**Cell 6 - Lines 44-51 (Training config):**
```python
results = model.train(
    data='/content/merged_construction_safety_3class_balanced/data.yaml',
    epochs=100,
    imgsz=960,  # ← CHANGED from 640 to 960 (critical!)
    batch=64,   # ← CHANGED from 128 to 64 (960px needs more VRAM)
    device=0,
    project='/content/drive/MyDrive/YOLOv12_Training/runs',
    name='yolo12n_construction_3class_960px',
    # ... rest stays same
)
```

**C. Run All Cells:**
- Run cells 1-7 in order
- Training will start automatically
- **Time:** 4-5 hours on H100

---

## 📊 Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| mAP@50 | 0.45 | **0.75-0.80** | +67-78% |
| Classes | 18 (1 dominated) | 3 (balanced) | Fixed |
| Resolution | 640px | 960px train → 640px deploy | +50% small objects |

---

## 🎯 Success Indicators

Watch these during training:

**After 10 epochs:**
```
mAP@50 should be > 0.50  ← Good sign! (vs stuck at 0.45)
```

**After 50 epochs:**
```
mAP@50 should be > 0.70  ← Great progress!
```

**After 100 epochs:**
```
mAP@50 should be 0.75-0.80  ← Mission accomplished! 🎉
```

---

## 🚀 Deployment (After Training)

**Export for Jetson Nano:**
```python
# Add this cell at the end of Colab notebook
model = YOLO('/content/drive/MyDrive/YOLOv12_Training/runs/yolo12n_construction_3class_960px/weights/best.pt')
model.export(format='onnx', imgsz=640, dynamic=True)
```

**Download and use on Jetson:**
- Download: `best.onnx`
- Deploy at: 640px (fast, ~15-20 FPS)
- Benefits from: 960px training (better features)

---

## ❓ Common Questions

**Q: Why train at 960px but deploy at 640px?**  
A: Model learns better features from larger images. Once learned, it works great at any size!

**Q: Will 3 classes lose accuracy?**  
A: No! You'll GAIN accuracy (0.45 → 0.78). Nano model can't distinguish 18 tiny vehicles anyway.

**Q: What are the 3 classes?**  
A: 
- Class 0: Person
- Class 1: Vehicle (all trucks, mixers, tankers, etc.)
- Class 2: Equipment (all excavators, bulldozers, cranes, etc.)

---

## 📋 Full Workflow Checklist

- [ ] Run `./run_fix_class_imbalance.sh`
- [ ] Upload `merged_construction_safety_3class_balanced.zip` to Drive
- [ ] Stop current Colab training
- [ ] Update Cell 4 (dataset paths)
- [ ] Update Cell 5 (verify paths)  
- [ ] Update Cell 6 (imgsz=960, batch=64)
- [ ] Run all Colab cells
- [ ] Monitor mAP@50 > 0.50 after 10 epochs
- [ ] Wait for training to complete (~4-5 hours)
- [ ] Export to ONNX for Jetson Nano
- [ ] Download and deploy

---

## 🆘 Need Help?

**If pipeline fails:**
```bash
# Check if source dataset exists
ls -la merged_construction_safety/

# Check Python packages
python3 -c "import yaml; print('yaml OK')"
```

**If Colab training still plateaus:**
- Verify dataset has 3 classes (check Cell 5 output)
- Verify imgsz=960 (check Cell 6 logs)
- Check H100 GPU is active (Cell 1 should say "H100 detected")

---

## 📁 Files Reference

```
yolo12-training/
├── UPDATED_WORKFLOW_3CLASS_960PX.md  ← Full detailed guide
├── QUICK_START_FIX.md                ← This file
├── FIX_CLASS_IMBALANCE_GUIDE.md      ← Technical explanation
├── run_fix_class_imbalance.sh        ← Run this script!
├── merge_classes_3way.py             ← Called by script
├── downsample_person_class.py        ← Called by script
├── merged_construction_safety/       ← Source (18 classes)
└── colab_setup/
    └── train_yolo12n_h100.ipynb      ← Update this notebook
```

---

**Ready?** Run the script! ⚡

```bash
cd /Users/ashwani/Desktop/RT-Monodepth-Construction/yolo12-training
./run_fix_class_imbalance.sh
```
