# Expected Benchmark Execution Time

## System Specifications
- **GPU:** Intel(R) Iris(R) Xe Graphics
- **Shared GPU Memory:** 7.9 GB
- **Driver Version:** 32.0.101.6737
- **Platform:** Windows (PCI bus 0, device 2, function 0)

---

## Benchmark Stages Execution Time

### **Stage 1: Per-Frame Accuracy Evaluation**

#### Dataset: NYUv2 (Eigen Split - 654 images)
- **Per-image inference time:** ~2-5 seconds (Intel Iris Xe, CPU mode)
- **Total processing time:** **30-60 minutes**

#### Dataset: NYUv2 (Full - 1449 images)
- **Per-image inference time:** ~2-5 seconds
- **Total processing time:** **2-4 hours**

#### Model Variants to Test (6 models):
| Model Path | Resolution | Estimated Time (Eigen) | Estimated Time (Full) |
|------------|------------|------------------------|----------------------|
| `weights/RTMonoDepth/s/m_640_192` | 640x192 | 30-45 min | 2-3 hours |
| `weights/RTMonoDepth/s/ms_640_192` | 640x192 | 30-45 min | 2-3 hours |
| `weights/RTMonoDepth/full/s_640_192` | 640x192 | 40-60 min | 3-4 hours |
| `weights/RTMonoDepth/full/sh_640_192` | 640x192 | 40-60 min | 3-4 hours |
| `weights/RTMonoDepth/full/m_640_192` | 640x192 | 45-70 min | 3.5-5 hours |
| `weights/RTMonoDepth/full/ms_640_192` | 640x192 | 45-70 min | 3.5-5 hours |

**Stage 1 Total (all models, Eigen split):** **4-6 hours**  
**Stage 1 Total (all models, Full dataset):** **18-24 hours**

---

### **Stage 2: Temporal Consistency Evaluation**

#### Single Video Sequence (~500 frames)
- **Per-frame inference:** ~2-5 seconds
- **Processing time per sequence:** **1-2 hours**
- **Metrics computation:** ~5-10 minutes
- **Total per sequence:** **1.5-2.5 hours**

#### Recommended Test Sequences (3-5 sequences):
- **3 sequences:** **4.5-7.5 hours**
- **5 sequences:** **7.5-12.5 hours**

#### Per Model Variant:
| Model Type | Single Sequence | 3 Sequences | 5 Sequences |
|------------|----------------|-------------|-------------|
| Small models (`/s/`) | 1-1.5 hours | 3-4.5 hours | 5-7.5 hours |
| Full models (`/full/`) | 1.5-2.5 hours | 4.5-7.5 hours | 7.5-12.5 hours |

**Stage 2 Total (1 model, 3 sequences):** **4.5-7.5 hours**  
**Stage 2 Total (all 6 models, 3 sequences each):** **27-45 hours**

---

### **Stage 3: Performance (FPS) Benchmark**

#### Single Model Benchmarking:
- **Warmup phase (30 iterations):** ~1-2 minutes
- **Benchmark phase (100 iterations):** ~3-5 minutes
- **Memory profiling:** ~1 minute
- **Total per model:** **5-10 minutes**

#### Multiple Resolution Testing (per model):
| Resolution | Iterations | Estimated Time |
|------------|-----------|----------------|
| 320x96 | 100 | 2-3 minutes |
| 640x192 | 100 | 3-5 minutes |
| 1024x320 | 100 | 5-8 minutes |

**Stage 3 Total (1 model, 3 resolutions):** **10-16 minutes**  
**Stage 3 Total (all 6 models, 3 resolutions each):** **60-90 minutes**

---

## Complete Benchmark Timeline

### **Quick Test (Recommended First Run):**
- Stage 1: Single model, 10 sample images → **~30 seconds**
- Stage 3: Single model, single resolution → **~5 minutes**
- **Total:** **~6 minutes**

### **Standard Evaluation (1 model):**
- Stage 1: Eigen split (654 images) → **45-60 minutes**
- Stage 2: 3 video sequences → **4.5-7.5 hours**
- Stage 3: 3 resolutions → **10-16 minutes**
- **Total:** **~6-8 hours**

### **Comprehensive Evaluation (All 6 models):**
- Stage 1: All models, Eigen split → **4-6 hours**
- Stage 2: All models, 3 sequences → **27-45 hours**
- Stage 3: All models, 3 resolutions → **60-90 minutes**
- **Total:** **32-52 hours** (can be parallelized)

### **Full Dataset Evaluation (All 6 models):**
- Stage 1: All models, Full 1449 images → **18-24 hours**
- Stage 2: All models, 3 sequences → **27-45 hours**
- Stage 3: All models, 3 resolutions → **60-90 minutes**
- **Total:** **46-70 hours** (2-3 days continuous)

---

## Optimization Strategies

### **1. Parallel Execution:**
If you have access to multiple machines or cloud instances:
- Run different models in parallel
- **Time reduction:** 6x faster (divide by number of models)
- Example: 32 hours → ~5-6 hours with 6 parallel instances

### **2. Subset Testing:**
For quick validation:
- Use Eigen split (654 images) instead of full dataset
- Test 1-2 video sequences instead of 5
- **Time reduction:** ~70% faster

### **3. Model Selection:**
Start with fastest models:
1. `weights/RTMonoDepth/s/m_640_192` (fastest)
2. `weights/RTMonoDepth/full/s_640_192` (best accuracy/speed balance)
3. Full evaluation only for final paper/publication

### **4. Overnight/Weekend Runs:**
**Recommended Schedule:**
- **Night 1:** Stage 1 - Single model, Eigen split (6-8 hours)
- **Night 2:** Stage 2 - Single model, 3 sequences (6-8 hours)
- **Weekend:** Full evaluation all models (48+ hours)

---

## Estimated FPS on Intel Iris Xe Graphics

Based on integrated GPU performance:

| Model Variant | Expected FPS (640x192) | Real-time (>30 FPS) |
|---------------|------------------------|---------------------|
| `s/m_640_192` (small) | 8-12 FPS | ❌ No |
| `s/ms_640_192` (small) | 8-12 FPS | ❌ No |
| `full/s_640_192` | 5-8 FPS | ❌ No |
| `full/sh_640_192` | 5-8 FPS | ❌ No |
| `full/m_640_192` | 4-6 FPS | ❌ No |
| `full/ms_640_192` | 4-6 FPS | ❌ No |

**Note:** Intel Iris Xe is not suitable for real-time depth estimation. For >30 FPS:
- **Minimum:** NVIDIA GTX 1660 or better
- **Recommended:** NVIDIA RTX 3060 or Apple M1 Pro
- **Optimal:** NVIDIA RTX 4070 or better

---

## Dataset Download Time

### **NYUv2 Labeled Dataset:**
- **File size:** ~2.8 GB (compressed .mat file)
- **Download time (50 Mbps):** ~7-10 minutes
- **Download time (10 Mbps):** ~40-60 minutes

### **NYUv2 Raw Video Sequences:**
- **File size:** ~400 GB (full dataset)
- **Download time (50 Mbps):** ~18-24 hours
- **Recommended:** Download only needed sequences (~5-10 GB)

---

## Recommended Execution Plan

### **Day 1: Setup & Quick Test (1 hour)**
```powershell
# Download dataset
# Run quick test with 10 images
python evaluate_nyu.py --num_samples 10 --model_path weights/RTMonoDepth/s/m_640_192
```

### **Day 2: Stage 1 - Accuracy (overnight, 6 hours)**
```powershell
# Run full Eigen split evaluation
python evaluate_nyu.py --split eigen --model_path weights/RTMonoDepth/full/s_640_192
```

### **Day 3: Stage 3 - FPS (1 hour)**
```powershell
# Run performance benchmarks
python benchmark_fps.py --test_all
```

### **Weekend: Stage 2 - Temporal (8+ hours)**
```powershell
# Run temporal consistency on video sequences
python evaluate_temporal.py --video_path datasets/nyu_raw --model_path weights/RTMonoDepth/full/s_640_192
```

### **Total Calendar Time: 4-5 days** (mostly automated overnight runs)

---

## Important Notes

⚠️ **Limitations with Intel Iris Xe:**
- Inference will run on **CPU mode** (no CUDA support)
- Performance will be **significantly slower** than dedicated GPUs
- Expected FPS will **NOT achieve real-time** performance
- Benchmarks are for **accuracy evaluation**, not deployment

✅ **What Works Well:**
- Stage 1 (Accuracy) - Fully compatible
- Stage 2 (Temporal) - Compatible but slow
- Stage 3 (FPS) - Will run but show low FPS

🚀 **For Production Deployment:**
- Use NVIDIA GPU or Apple M1/M2 hardware
- Expected real-time performance: 30-60+ FPS on proper hardware

---

## Summary Table

| Stage | Task | Quick Test | Standard | Comprehensive |
|-------|------|------------|----------|---------------|
| **1** | Per-frame Accuracy | 30 sec | 45-60 min | 4-6 hours |
| **2** | Temporal Consistency | N/A | 4.5-7.5 hours | 27-45 hours |
| **3** | FPS Benchmark | 5 min | 10-16 min | 60-90 min |
| | **TOTAL** | **~6 min** | **~6-8 hours** | **32-52 hours** |

**Recommended:** Start with **Standard** evaluation for a single best model, then expand to comprehensive if needed.
