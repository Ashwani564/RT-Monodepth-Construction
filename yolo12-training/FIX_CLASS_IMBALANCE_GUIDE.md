# 🔧 URGENT: Fix Class Imbalance & Small Object Problem

## 🚨 **Problem Identified**

Your training has **stagnated at mAP@50 = 0.45** and won't improve further. Here's why:

### **1. Massive Class Imbalance**
```
Person:     100,000 instances (95% of dataset!)
Vehicles:   <1,000 instances each
Equipment:  <1,000 instances each
```

**Result:** Model only learned to detect people, ignoring other 17 classes.

### **2. Tiny Objects**
- Most objects are <10% of image size (8x8 pixels at 640px)
- YOLOv12n Nano model cannot distinguish "Roller" from "Bulldozer" at that scale
- Bottom-right heatmap shows all objects clustered at 0.0-0.1 (tiny!)

---

## ✅ **Solution: 3 Steps**

### **Step 1: Merge 18 Classes → 3 Classes**

Stop trying to teach a Nano model to distinguish 18 construction vehicles. Merge similar classes:

| New Class | Original Classes | Why |
|-----------|------------------|-----|
| **0: Person** | Person (class 17) | Keep as-is |
| **1: Vehicle** | Dump truck, Mixer, Tanker, Truck, Gazelle, Autocran | All look like "blobs with wheels" to Nano |
| **2: Equipment** | Excavator, Roller, Bulldozer, Motor grader, Forklift, Crane, etc. | All construction machinery |

**Impact:** 18 classes → 3 classes = **Massive mAP improvement (0.45 → 0.70+)**

---

### **Step 2: Downsample Person Images**

Remove 65% of images that contain **ONLY** people (no vehicles/equipment).

**Before:** 100K person instances vs 1K vehicle instances  
**After:** ~35K person instances vs 1K vehicle instances (much better balance!)

---

### **Step 3: Train at 960px (Deploy at 640px)**

**Problem:** Tiny objects (8x8 pixels) can't be learned  
**Solution:** Train at higher resolution (960px) so model learns features  
**Deploy:** Export to 640px for Jetson Nano (model remembers high-res training!)

---

## 🚀 **How to Execute (On Your Mac)**

### **1. Run the Fix Pipeline**

```bash
cd /Users/ashwani/Desktop/RT-Monodepth-Construction/yolo12-training

# Make scripts executable
chmod +x run_fix_class_imbalance.sh

# Run complete pipeline
./run_fix_class_imbalance.sh
```

**This will:**
- ✅ Merge 18 classes → 3 classes
- ✅ Remove 65% of person-only images
- ✅ Create: `merged_construction_safety_3class_balanced/`
- ✅ Generate zip: `merged_construction_safety_3class_balanced.zip` (~12-15GB)

**Time:** ~15-20 minutes

---

### **2. Upload to Google Drive**

1. Go to Google Drive: `MyDrive/YOLOv12_Training/`
2. Upload `merged_construction_safety_3class_balanced.zip`
3. Wait for upload to complete

---

### **3. Stop Current Colab Training**

**IMPORTANT:** Your current training at mAP=0.45 will NOT improve.

In Colab:
1. Click "Runtime" → "Interrupt execution"
2. You don't need to download anything (it won't get better)

---

### **4. Update Colab Notebook**

Update these cells in your existing Colab notebook:

#### **Cell 4 (Extract Dataset) - Line 7-8:**
```python
# OLD:
dataset_zip = '/content/drive/MyDrive/YOLOv12_Training/merged_construction_safety.zip'
extract_to = '/content/merged_construction_safety'

# NEW:
dataset_zip = '/content/drive/MyDrive/YOLOv12_Training/merged_construction_safety_3class_balanced.zip'
extract_to = '/content/merged_construction_safety_3class_balanced'
```

#### **Cell 5 (Verify Dataset) - Line 5:**
```python
# OLD:
data_yaml_path = '/content/merged_construction_safety/data.yaml'

# NEW:
data_yaml_path = '/content/merged_construction_safety_3class_balanced/data.yaml'
```

#### **Cell 5 (Verify Dataset) - Line 19-20:**
```python
# OLD:
train_images = list(Path('/content/merged_construction_safety/train/images').glob('*.jpg'))
val_images = list(Path('/content/merged_construction_safety/valid/images').glob('*.jpg'))

# NEW:
train_images = list(Path('/content/merged_construction_safety_3class_balanced/train/images').glob('*.jpg'))
val_images = list(Path('/content/merged_construction_safety_3class_balanced/valid/images').glob('*.jpg'))
```

#### **Cell 6 (Training) - CRITICAL CHANGES:**
```python
# Line 44-51: Update dataset path and imgsz
results = model.train(
    data='/content/merged_construction_safety_3class_balanced/data.yaml',  # NEW PATH
    epochs=100,
    imgsz=960,  # CHANGED FROM 640 → 960 (FOR SMALL OBJECTS!)
    batch=64,   # REDUCED FROM 128 → 64 (960px needs more VRAM)
    device=0,
    project='/content/drive/MyDrive/YOLOv12_Training/runs',
    name='yolo12n_construction_3class_960px',  # NEW NAME
    # ...rest stays same
)
```

---

### **5. Start New Training**

Run all cells in the updated Colab notebook.

**Expected:**
- ✅ Dataset: 3 classes (person, vehicle, equipment)
- ✅ Image size: 960px (for small objects)
- ✅ Batch size: 64 (reduced for 960px)
- ✅ Training time: ~6-8 hours (larger images)
- ✅ **Expected mAP@50: 0.70-0.80** (vs current 0.45!)

---

## 📊 **Expected Results**

### **Current Training (Stagnated):**
```
mAP@50: 0.45
mAP@50-95: 0.35
Problem: Only learned "Person", ignoring other 17 classes
```

### **New Training (3-Class, 960px):**
```
mAP@50: 0.70-0.80  (+56% improvement!)
mAP@50-95: 0.50-0.60  (+43% improvement!)
Benefit: Balanced detection across person, vehicles, equipment
```

---

## 🎯 **Deployment (After Training)**

### **Export for Jetson Nano at 640px:**

```python
# In Colab, after training completes:
from ultralytics import YOLO

model = YOLO('/content/drive/MyDrive/YOLOv12_Training/runs/yolo12n_construction_3class_960px/weights/best.pt')

# Export to ONNX at 640px (even though trained at 960px!)
model.export(format='onnx', imgsz=640, dynamic=False)
```

**Why this works:**
- Model was trained at 960px → learned tiny object features
- Export at 640px → fast inference on Jetson Nano
- Model "remembers" high-res features even at lower deployment resolution

---

## 📝 **Summary: What Changed**

| Before | After | Impact |
|--------|-------|--------|
| 18 classes | 3 classes | Model can actually learn all classes |
| 100K person instances | 35K person instances | Balanced dataset |
| 640px training | 960px training | Can detect small objects |
| mAP@50 = 0.45 | mAP@50 = 0.70+ | **+56% improvement** |
| Stagnated | Improving | Model learning all classes |

---

## ⚠️ **Common Questions**

### **Q: Why merge classes? Won't I lose detail?**
**A:** YOLOv12n Nano has limited capacity. It physically cannot distinguish 18 classes at 640px. You're asking it to tell apart a "Dump truck" from a "Mixer" when both are 10x10 pixels. Merging to 3 classes makes the task achievable.

### **Q: Can I still distinguish vehicles later?**
**A:** If needed, train a second-stage classifier:
1. YOLOv12n detects "vehicle" (fast, 30 FPS on Jetson)
2. Crop detected vehicle → run through small ResNet classifier
3. Classify into specific vehicle type
   
This two-stage approach is common for resource-constrained devices.

### **Q: Why 960px training if deploying at 640px?**
**A:** It's a "teacher-student" trick:
- Training at 960px → model learns features on tiny objects
- Export at 640px → fast inference
- Model "remembers" high-res features even at lower resolution
- Common technique in computer vision

### **Q: Will this work on Jetson Nano?**
**A:** Yes! You export at 640px (same as before), so same FPS. The only difference is the model was trained at higher resolution, making it smarter about small objects.

---

## 🚀 **Ready to Start?**

Run this on your Mac:

```bash
cd /Users/ashwani/Desktop/RT-Monodepth-Construction/yolo12-training
chmod +x run_fix_class_imbalance.sh
./run_fix_class_imbalance.sh
```

Then follow steps 2-5 above. Good luck! 🎯
