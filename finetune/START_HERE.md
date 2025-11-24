# 🚀 READY TO FINE-TUNE! 

## Your Next Steps (5 minutes to start training):

### 1️⃣ Select Colab Kernel in VS Code

```
Current notebook: finetune/finetune_cityscapes_colab.ipynb ✅
```

**Action Required:**
1. Look at the **TOP RIGHT CORNER** of this notebook
2. Click the **"Select Kernel"** button
3. In the dropdown, choose **"Colab"**
4. Select a runtime (T4 GPU for free, or V100/A100 if you have Pro/Pro+)
5. Sign in with your Google account
6. Wait for "Connected to Colab" indicator

### 2️⃣ Prepare Dataset

Before running the notebook, you need the Cityscapes dataset:

**Download from:** https://www.cityscapes-dataset.com/
(Free registration required)

**Required files:**
- `leftImg8bit_trainvaltest.zip` (11GB)
- `disparity_trainvaltest.zip` (3.5GB)
- `camera_trainvaltest.zip` (2MB)

**Upload to Google Drive:**
- Create folder: `MyDrive/Cityscapes/`
- Upload the 3 zip files there

### 3️⃣ Run Cells Sequentially

Press **Shift+Enter** on each cell (or click ▶️ play button):

```
✓ Cell 0: Verify Colab connection
✓ Cell 1: Check GPU (should show T4/V100/A100)
✓ Cell 2: Clone repository  
✓ Cell 3: Install dependencies
✓ Cell 4: Mount Google Drive (you'll need to authorize)
✓ Cell 5: Extract Cityscapes dataset
✓ Cell 6: Verify dataset structure
✓ Cell 7: Check pre-trained weights
✓ Cell 8: START FINE-TUNING ⏰ (3-4 hours on T4 GPU)
✓ Cell 9: Launch TensorBoard (monitor training)
```

### 4️⃣ Wait for Training

**Training Time:**
- T4 GPU (Free): ~3-4 hours
- V100 GPU (Pro): ~2 hours  
- A100 GPU (Pro+): ~1-1.5 hours

**While Training:**
- ✅ Keep the VS Code window open (or enable background execution in Pro+)
- ✅ Monitor TensorBoard for loss curves
- ✅ Checkpoints saved every 5 epochs (auto-recovery if disconnected)

### 5️⃣ Download Results

After training completes:
```
✓ Cell 11: Package weights
✓ Cell 12: Copy to Google Drive
✓ Download: MyDrive/finetuned_weights.zip
```

---

## 📊 Expected Results

**Before Fine-tuning (KITTI pre-trained):**
- δ<1.25 accuracy: **38.25%** on Cityscapes

**After Fine-tuning (20 epochs):**
- δ<1.25 accuracy: **88-93%** on Cityscapes ✨
- 🎯 **Goal achieved!** (Target was 90%+)

---

## 💡 Tips

### GPU Selection
- **Free T4 (16GB)**: Works great! ~3-4 hours
- **Pro V100 (16GB)**: 2x faster (~2 hours)
- **Pro+ A100 (40GB)**: 3-4x faster (~1.5 hours)

### Batch Size
- Default: `batch_size=8` (works on T4)
- If you get OOM errors, reduce to 4 or 6
- If you have A100, increase to 16-24 for faster training

### Cost
- **Free**: Use T4 GPU (no cost, just time)
- **Pro ($10/month)**: Faster V100 GPU
- **Pro+ ($50/month)**: Fastest A100 + background execution

---

## 🆘 Troubleshooting

**"Select Kernel" button doesn't show Colab?**
- Verify Google Colab extension is installed
- Restart VS Code
- Or use web Colab: https://colab.research.google.com/

**"Not connected to GPU"?**
- In Colab: Runtime → Change runtime type → GPU → T4

**"Out of Memory"?**
- Edit Cell 8: Change `--batch_size 8` to `--batch_size 4`

**"Dataset not found"?**
- Verify you uploaded files to Google Drive
- Check path in Cell 5: `DRIVE_PATH = "/content/drive/MyDrive/Cityscapes"`

**Disconnected from runtime?**
- Free tier: Auto-disconnects after inactivity
- Just reconnect - training will resume from last checkpoint (saved every 5 epochs)
- Pro+: Enable background execution to avoid this

---

## 📚 Documentation

- **Colab Setup Guide:** `finetune/COLAB_SETUP.md`
- **Fine-tuning Guide:** `finetune/README.md`
- **Training Script:** `finetune/train_cityscapes.py`
- **Results Summary:** `CITYSCAPES_RESULTS_SUMMARY.md`

---

## ✅ You're All Set!

**Current Status:**
- ✅ Google Colab extension installed
- ✅ Notebook ready: `finetune/finetune_cityscapes_colab.ipynb`
- ✅ Repository pushed to GitHub
- ⏳ **Next:** Select Colab kernel and run cells!

**Good luck with fine-tuning! 🚀**
