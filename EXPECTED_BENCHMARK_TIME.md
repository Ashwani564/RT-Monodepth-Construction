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

## Performance Comparison: Intel Iris Xe vs M1 Pro

### **Expected FPS at 640x192 Resolution**

| Model Variant | Intel Iris Xe (Windows) | MacBook M1 Pro (MPS) | Speedup |
|---------------|------------------------|----------------------|---------|
| `s/m_640_192` (small) | 8-12 FPS | 55-65 FPS | **~6x faster** |
| `s/ms_640_192` (small) | 8-12 FPS | 50-60 FPS | **~5.5x faster** |
| `full/s_640_192` | 5-8 FPS | 42-50 FPS | **~7x faster** |
| `full/sh_640_192` | 5-8 FPS | 40-48 FPS | **~6.5x faster** |
| `full/m_640_192` | 4-6 FPS | 35-42 FPS | **~8x faster** |
| `full/ms_640_192` | 4-6 FPS | 32-40 FPS | **~7x faster** |

### **Real-Time Capability Comparison**

| Platform | Real-time (>30 FPS) | Best Use Case |
|----------|---------------------|---------------|
| **Intel Iris Xe** | ❌ No (4-12 FPS) | Offline evaluation, benchmarking |
| **MacBook M1 Pro** | ✅ Yes (32-65 FPS) | Real-time deployment, production |

### **Benchmark Execution Time Comparison**

#### Stage 1: Per-Frame Accuracy (Eigen Split - 654 images)

| Model Variant | Intel Iris Xe | MacBook M1 Pro | Time Savings |
|---------------|---------------|----------------|--------------|
| `s/m_640_192` (small) | 30-45 min | 5-8 min | **~5.5x faster** |
| `full/s_640_192` | 40-60 min | 7-10 min | **~6x faster** |
| `full/m_640_192` | 45-70 min | 8-12 min | **~6.5x faster** |
| **All 6 models** | **4-6 hours** | **40-65 minutes** | **~5-6x faster** |

#### Stage 2: Temporal Consistency (3 video sequences)

| Model Type | Intel Iris Xe | MacBook M1 Pro | Time Savings |
|------------|---------------|----------------|--------------|
| Small models (`/s/`) | 3-4.5 hours | 30-50 min | **~5.5x faster** |
| Full models (`/full/`) | 4.5-7.5 hours | 45-75 min | **~6x faster** |
| **All 6 models** | **27-45 hours** | **4.5-7.5 hours** | **~6x faster** |

#### Complete Benchmark Timeline Comparison

| Evaluation Type | Intel Iris Xe | MacBook M1 Pro | Difference |
|-----------------|---------------|----------------|------------|
| **Quick Test** | 6 min | ~1 min | 5 min faster |
| **Standard (1 model)** | 6-8 hours | 1-1.5 hours | 5-6.5 hours faster |
| **Comprehensive (6 models)** | 32-52 hours | 5-8.5 hours | 24-44 hours faster |
| **Full Dataset (6 models)** | 46-70 hours | 7.5-12 hours | 38-58 hours faster |

### **Platform Capabilities Summary**

#### Intel Iris Xe Graphics (Your Current System)
- **Architecture:** Integrated GPU, shared memory
- **Acceleration:** CPU mode (no CUDA, limited GPU acceleration)
- **FPS Range:** 4-12 FPS (not real-time)
- **Best For:** 
  - ✅ Accuracy evaluation and benchmarking
  - ✅ Model comparison studies
  - ✅ Initial testing and development
  - ❌ Real-time video processing
  - ❌ Production deployment
- **Benchmark Time:** 32-52 hours (comprehensive)

#### MacBook M1 Pro (Optimized Platform)
- **Architecture:** Apple Silicon, unified memory
- **Acceleration:** MPS (Metal Performance Shaders) or MLX
- **FPS Range:** 32-65 FPS (real-time capable)
- **Best For:**
  - ✅ Real-time video processing
  - ✅ Production deployment
  - ✅ Fast benchmark execution
  - ✅ Interactive development
  - ✅ Construction site monitoring (as designed)
- **Benchmark Time:** 5-8.5 hours (comprehensive)

### **Memory Usage Comparison**

| Platform | Model Memory | Peak Usage | Available |
|----------|-------------|------------|-----------|
| **Intel Iris Xe** | 512-640 MB | ~2-3 GB | 7.9 GB shared |
| **MacBook M1 Pro** | 512-640 MB | ~1.5-2 GB | 16 GB unified |

**Note:** M1 Pro's unified memory architecture provides faster memory access and better efficiency.

### **Recommendations Based on Platform**

#### If Using Intel Iris Xe (Current System):
1. **Focus on accuracy evaluation** - Your system is perfect for Stage 1 benchmarks
2. **Run overnight jobs** - Stage 1 can run while you sleep (6-8 hours)
3. **Start with small models** - Test `s/m_640_192` first (fastest on your hardware)
4. **Use for development** - Write and test evaluation scripts
5. **Skip real-time demos** - FPS will be too low for smooth video playback

#### If You Have Access to M1 Pro:
1. **Run all benchmarks** - Complete evaluation in one workday (5-8 hours)
2. **Real-time testing** - Can actually test construction site monitoring use case
3. **Interactive development** - Fast iteration on model improvements
4. **Video processing** - Smooth 30-60 FPS for all model variants
5. **Production deployment** - Ready for actual deployment scenarios

### **Cost-Benefit Analysis for Upgrading/Cloud Access**

If you need faster benchmarking:

| Option | Cost | Time Savings | When to Use |
|--------|------|--------------|-------------|
| **Current Intel Iris Xe** | $0 | Baseline | Accuracy-only benchmarks |
| **Cloud M1 Mac (AWS EC2 mac2.metal)** | ~$1.10/hour | 5-6x faster | One-time comprehensive test (~$9 for 8 hours) |
| **Cloud NVIDIA GPU (g5.xlarge)** | ~$1.00/hour | 8-10x faster | Best price/performance (~$5 for full benchmark) |
| **Local M1 Pro/Max** | One-time hardware | 5-6x faster | Frequent testing, development |

**Recommendation:** For your research, the Intel Iris Xe is sufficient for Stage 1 (accuracy) evaluation. If you need Stage 2 (temporal) or Stage 3 (FPS) for publication, consider one weekend on a cloud GPU instance (~$10-20 total cost).

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
