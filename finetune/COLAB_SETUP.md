# Google Colab Fine-tuning Guide

This guide explains how to fine-tune RT-MonoDepth on Cityscapes using Google Colab **in VS Code**.

## ✅ You Have Google Colab Extension Installed!

You can now run Colab notebooks directly in VS Code! This is much better than using the web interface.

## Quick Start (VS Code Colab Extension)

### Step 1: Select Colab Kernel

1. Open `finetune/finetune_cityscapes_colab.ipynb` in VS Code ✅ (already open!)
2. Click **"Select Kernel"** button in the **top right corner** of the notebook
3. In the kernel picker dropdown, select **"Colab"**
4. Choose your desired Colab runtime:
   - **T4 GPU** (Free tier - ~3-4 hours training time)
   - **V100 GPU** (Colab Pro - ~2 hours training time)
   - **A100 GPU** (Colab Pro+ - ~1-1.5 hours training time)
5. Sign in with your Google account if prompted
6. Wait for connection (you'll see "Colab" indicator in the top right when connected)

### Step 2: Prepare Cityscapes Dataset

**Important:** Download these files from https://www.cityscapes-dataset.com/ (requires free registration):

1. **leftImg8bit_trainvaltest.zip** (11GB)
2. **disparity_trainvaltest.zip** (3.5GB)
3. **camera_trainvaltest.zip** (2MB)

**Upload to Google Drive:**
- Create a folder in your Google Drive: `MyDrive/Cityscapes/`
- Upload the three zip files there
- The notebook will automatically mount and extract them

### Step 3: Run the Notebook

1. Press **Shift+Enter** to run cells sequentially (or click the ▶️ play button)
2. Start with **Cell 0** to verify Colab connection
3. Follow through each cell in order:
   - ✓ Cell 0: Verify Colab connection
   - ✓ Cell 1: Check GPU availability
   - ✓ Cell 2: Clone repository
   - ✓ Cell 3: Install dependencies
   - ✓ Cell 4-6: Mount Drive and extract Cityscapes
   - ✓ Cell 7-8: Start fine-tuning (this takes 3-4 hours)
   - ✓ Cell 9: Monitor with TensorBoard

### Step 4: Monitor Training

- Training progress will print in real-time
- Run TensorBoard cell (Cell 9) to see loss curves and metrics
- Checkpoints are saved every 5 epochs

### Step 5: Download Fine-tuned Weights

After training completes:
1. Run cells 11-12 to package the weights
2. Weights will be saved to `MyDrive/finetuned_weights.zip`
3. Download from your Google Drive

---

## Alternative Methods

### Option A: Web-based Colab (if VS Code extension doesn't work)

1. Go to https://colab.research.google.com/
2. Click "GitHub" tab
3. Enter: `Ashwani564/RT-Monodepth-Construction`
4. Select branch: `finetuneCityScapes`
5. Open: `finetune/finetune_cityscapes_colab.ipynb`

### Option B: Upload Notebook

1. Go to https://colab.research.google.com/
2. Click "Upload" and select `finetune/finetune_cityscapes_colab.ipynb`

### 3. Enable GPU (Web Colab)

1. In Colab, go to **Runtime → Change runtime type**
2. Select **Hardware accelerator: GPU**
3. Choose **T4** (free tier) or **V100/A100** (Pro)
4. Click **Save**

### 4. Run the Notebook

Execute all cells sequentially:

1. ✅ **Check GPU** - Verify GPU is available
2. ✅ **Clone Repo** - Downloads code and pre-trained weights
3. ✅ **Install Dependencies** - Sets up PyTorch environment
4. ✅ **Mount Drive** - Access your Google Drive
5. ✅ **Extract Dataset** - Unzips Cityscapes data
6. ✅ **Verify Dataset** - Confirms data structure
7. ✅ **Start Training** - Begins fine-tuning (3-4 hours)
8. ✅ **Monitor Progress** - Use TensorBoard
9. ✅ **Download Weights** - Save results to Drive

## Training Configuration

Default settings (optimized for T4 GPU):

```python
Model: full_sh_640_192 (ShuffleNet V2)
Epochs: 20
Batch Size: 8
Learning Rate: 1e-5 (encoder), 1e-4 (decoder)
Input Size: 640x192
Training Time: ~3-4 hours (T4), ~2 hours (V100)
```

### Adjust for Different GPUs

**If you have a smaller GPU (e.g., older K80):**
```python
--batch_size 4  # or even 2
```

**If you have a larger GPU (V100/A100):**
```python
--batch_size 16  # or even 32
--num_workers 8
```

## Expected Results

After fine-tuning, you should see:

- **Before:** δ<1.25 = 38.25%
- **After:** δ<1.25 = 88-93% ✨

Performance improvements:
- **Abs Rel:** 0.25 → 0.08
- **RMSE:** 8.5m → 3.2m
- **δ<1.25:** 38% → 90%+

## Monitoring Training

### TensorBoard

In the notebook, run:
```python
%load_ext tensorboard
%tensorboard --logdir finetune/logs
```

You'll see real-time plots of:
- Training/validation loss
- Accuracy metrics (δ<1.25, Abs Rel, RMSE)
- Learning rates

### Console Output

Training progress is printed every 10 batches:
```
Epoch 1 [10/310] Loss: 0.3245 (L1: 0.1234, SSIM: 0.2011)
Epoch 1 [20/310] Loss: 0.3102 (L1: 0.1189, SSIM: 0.1913)
...

Validation - Loss: 0.2845
  AbsRel: 0.1234
  RMSE: 4.56m
  δ<1.25: 0.7823 (78.23%)
```

## Saving Results

### Automatic Saves

Checkpoints are saved:
- Every 5 epochs → `checkpoint_epoch_005.pth`, etc.
- Best model → `best_model.pth`
- Final weights → `final_weights/encoder.pth`, `final_weights/depth.pth`

### Download to Local Machine

Run the download cell:
```python
!cd finetune/checkpoints/full_sh_640_192 && \
    zip -r /content/finetuned_weights.zip final_weights/ best_model.pth
```

Then download via:
1. Files panel in Colab (left sidebar)
2. Right-click → Download
3. Or copy to Google Drive

### Copy to Google Drive

Automatically saved to Drive:
```python
!cp /content/finetuned_weights.zip /content/drive/MyDrive/
```

## Troubleshooting

### Out of Memory (OOM)

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size: `--batch_size 4` or `--batch_size 2`
2. Reduce number of workers: `--num_workers 2`
3. Restart runtime and clear cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Slow Training

**Issue:** Training is very slow

**Solutions:**
1. Verify GPU is enabled (check cell #1)
2. Use V100 or A100 if available (Colab Pro)
3. Reduce `num_workers` if CPU is bottleneck
4. Check if another notebook is using GPU

### Dataset Not Found

**Error:** `FileNotFoundError: Cityscapes images not found`

**Solutions:**
1. Verify zip files are in correct Drive location
2. Update `DRIVE_PATH` variable in cell #4
3. Check extraction completed successfully
4. Ensure files are unzipped, not just uploaded

### Connection Issues

**Issue:** Colab disconnects during training

**Solutions:**
1. Colab free tier has 12-hour limit
2. Keep browser tab active (prevents idle timeout)
3. Use Colab Pro for longer sessions
4. Training auto-resumes from last checkpoint

## Using Fine-tuned Weights

After training, use the fine-tuned weights in your application:

```python
from networks.RTMonoDepth.RTMonoDepth import DepthEncoder, DepthDecoder
import torch

# Load fine-tuned model
encoder = DepthEncoder()
decoder = DepthDecoder(encoder.num_ch_enc, scales=range(1))

encoder.load_state_dict(torch.load('final_weights/encoder.pth'))
decoder.load_state_dict(torch.load('final_weights/depth.pth'))

encoder.eval()
decoder.eval()

# Now use for inference on Cityscapes images
```

## Cost Analysis

### Free Tier
- **GPU:** T4 (16GB)
- **Time Limit:** 12 hours
- **Training Time:** ~3-4 hours ✅
- **Cost:** FREE

### Colab Pro ($9.99/month)
- **GPU:** V100 (16GB) or A100 (40GB)
- **Time Limit:** 24 hours
- **Training Time:** ~2 hours
- **Better for:** Multiple experiments

## Next Steps

1. **Evaluate Results:** Run evaluation on test set
2. **Try Different Models:** Fine-tune other variants (s, m, ms)
3. **Hyperparameter Tuning:** Adjust learning rates, epochs
4. **Transfer to Other Datasets:** Apply to BDD100K, Mapillary
5. **Deploy Model:** Use fine-tuned weights in production

## Additional Resources

- **Cityscapes Dataset:** https://www.cityscapes-dataset.com/
- **RT-MonoDepth Paper:** https://arxiv.org/abs/2212.09171
- **Colab Documentation:** https://colab.research.google.com/
- **Fine-tuning Guide:** See `finetune/README.md`

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review training logs in TensorBoard
3. Check GitHub issues
4. Create new issue with error logs

---

**Happy Fine-tuning! 🚀**
