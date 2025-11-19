#!/bin/bash
# prepare_datasets_for_sharing.sh
# Automated script to prepare RT-MonoDepth datasets for sharing with colleagues
# Usage: bash prepare_datasets_for_sharing.sh

set -e  # Exit on error

echo "📦 RT-MonoDepth Dataset Sharing Preparation Tool"
echo "================================================="
echo ""

# Check if datasets exist
if [ ! -d "datasets" ]; then
    echo "❌ Error: datasets/ folder not found!"
    echo "   Run this script from the project root directory."
    exit 1
fi

# Display current dataset sizes
echo "📊 Current Dataset Sizes:"
echo "-------------------------"
du -sh datasets/nyu_depth_v2 2>/dev/null || echo "  ⚠️  NYU Depth V2: Not found"
du -sh datasets/kitti 2>/dev/null || echo "  ⚠️  KITTI: Not found"
du -sh datasets/cityscapes 2>/dev/null || echo "  ⚠️  Cityscapes: Not found"
du -sh datasets/make3d 2>/dev/null || echo "  ⚠️  Make3D: Not found"
echo "-------------------------"
du -sh datasets/ 2>/dev/null
echo ""

# Ask user what they want to do
echo "🎯 What would you like to do?"
echo ""
echo "1) Create compressed archives (for cloud upload)"
echo "2) Generate checksums only"
echo "3) Create README for sharing"
echo "4) Full package (archives + checksums + README)"
echo "5) Quick verify datasets"
echo "6) Exit"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "🗜️  Creating compressed archives..."
        echo "   This will take 10-30 minutes depending on your CPU."
        echo ""
        
        # Check if pigz is available (parallel gzip)
        if command -v pigz &> /dev/null; then
            echo "✅ Using pigz (parallel compression) - FAST"
            COMPRESS_CMD="pigz"
        else
            echo "⚠️  pigz not found, using standard gzip (slower)"
            echo "   Install pigz for faster compression: brew install pigz"
            COMPRESS_CMD="gzip"
        fi
        
        # Create compressed archives
        mkdir -p shared_datasets
        
        if [ -d "datasets/nyu_depth_v2" ]; then
            echo "  📦 Compressing NYU Depth V2 + Make3D..."
            tar -I $COMPRESS_CMD -cf shared_datasets/datasets_nyu_make3d.tar.gz \
                datasets/nyu_depth_v2/ datasets/make3d/ 2>/dev/null || \
                tar -czf shared_datasets/datasets_nyu_make3d.tar.gz \
                datasets/nyu_depth_v2/ datasets/make3d/
            echo "  ✅ Done: $(du -sh shared_datasets/datasets_nyu_make3d.tar.gz | cut -f1)"
        fi
        
        if [ -d "datasets/kitti" ]; then
            echo "  📦 Compressing KITTI (this is large, ~15-20 min)..."
            tar -I $COMPRESS_CMD -cf shared_datasets/datasets_kitti.tar.gz \
                datasets/kitti/ 2>/dev/null || \
                tar -czf shared_datasets/datasets_kitti.tar.gz datasets/kitti/
            echo "  ✅ Done: $(du -sh shared_datasets/datasets_kitti.tar.gz | cut -f1)"
        fi
        
        if [ -d "datasets/cityscapes" ]; then
            echo "  📦 Compressing Cityscapes (this is large, ~15-20 min)..."
            tar -I $COMPRESS_CMD -cf shared_datasets/datasets_cityscapes.tar.gz \
                datasets/cityscapes/ 2>/dev/null || \
                tar -czf shared_datasets/datasets_cityscapes.tar.gz datasets/cityscapes/
            echo "  ✅ Done: $(du -sh shared_datasets/datasets_cityscapes.tar.gz | cut -f1)"
        fi
        
        echo ""
        echo "✅ Archives created in: shared_datasets/"
        ls -lh shared_datasets/*.tar.gz
        echo ""
        echo "📊 Total compressed size:"
        du -sh shared_datasets/
        ;;
        
    2)
        echo ""
        echo "🔐 Generating checksums..."
        
        if [ -d "shared_datasets" ] && [ "$(ls -A shared_datasets/*.tar.gz 2>/dev/null)" ]; then
            cd shared_datasets
            shasum -a 256 *.tar.gz > CHECKSUMS.txt
            echo "✅ Checksums saved to: shared_datasets/CHECKSUMS.txt"
            echo ""
            cat CHECKSUMS.txt
            cd ..
        else
            echo "❌ No compressed archives found!"
            echo "   Run option 1 first to create archives."
        fi
        ;;
        
    3)
        echo ""
        echo "📝 Creating README for sharing..."
        
        cat > shared_datasets/README.md << 'EOF'
# RT-MonoDepth Construction Site Benchmark Datasets
**Version:** 1.0
**Date:** November 2025
**Total Size:** 31 GB (uncompressed), ~28 GB (compressed)

## 📦 Contents

This package contains four standard depth estimation benchmark datasets organized for RT-MonoDepth evaluation:

### 1. NYU Depth V2 (2.8 GB)
- **Purpose:** Indoor depth baseline
- **Scenes:** 654 test images (Eigen split)
- **Ground Truth:** Dense depth maps from Kinect
- **Original Source:** http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/

### 2. KITTI (13 GB)
- **Purpose:** Outdoor/driving scenes (construction site proxy)
- **Scenes:** 697 test images (Eigen split)
- **Ground Truth:** Velodyne LiDAR depth
- **Original Source:** http://www.cvlibs.net/datasets/kitti/

### 3. Cityscapes (14 GB)
- **Purpose:** Urban pedestrian scenarios
- **Scenes:** Validation set with outdoor scenes
- **Ground Truth:** Disparity maps (convertible to depth)
- **Original Source:** https://www.cityscapes-dataset.com/

### 4. Make3D (700 MB)
- **Purpose:** Diverse outdoor scenes
- **Scenes:** 134 test images
- **Ground Truth:** Laser scanner depth
- **Original Source:** http://make3d.cs.cornell.edu/

## 📁 Dataset Structure

After extraction, your datasets should be organized as:

```
datasets/
├── nyu_depth_v2/
│   └── nyu_depth_v2_labeled.mat
├── kitti/
│   ├── data_depth_annotated/
│   └── raw_data_downloader/
├── cityscapes/
│   ├── leftImg8bit_trainvaltest/
│   ├── camera_trainvaltest/
│   └── disparity_trainvaltest/
└── make3d/
    ├── *.jpg (images)
    └── *.dat (depth files)
```

## 🚀 Usage

1. **Extract Archives:**
   ```bash
   tar -xzf datasets_nyu_make3d.tar.gz
   tar -xzf datasets_kitti.tar.gz
   tar -xzf datasets_cityscapes.tar.gz
   ```

2. **Verify Integrity:**
   ```bash
   shasum -a 256 -c CHECKSUMS.txt
   ```

3. **Run Evaluation:**
   ```bash
   # See REVISED_BENCHMARK_PLAN.md for full evaluation pipeline
   python evaluate_depth_multi_dataset.py --datasets nyu kitti --output results/
   ```

## 📚 Citations

If you use these datasets, please cite the original papers:

**NYU Depth V2:**
```bibtex
@inproceedings{silberman2012indoor,
  title={Indoor segmentation and support inference from RGBD images},
  author={Silberman, Nathan and Hoiem, Derek and Kohli, Pushmeet and Fergus, Rob},
  booktitle={ECCV},
  year={2012}
}
```

**KITTI:**
```bibtex
@inproceedings{geiger2012we,
  title={Are we ready for autonomous driving? the kitti vision benchmark suite},
  author={Geiger, Andreas and Lenz, Philip and Urtasun, Raquel},
  booktitle={CVPR},
  year={2012}
}
```

**Cityscapes:**
```bibtex
@inproceedings{cordts2016cityscapes,
  title={The cityscapes dataset for semantic urban scene understanding},
  author={Cordts, Marius and others},
  booktitle={CVPR},
  year={2016}
}
```

**Make3D:**
```bibtex
@article{saxena2009make3d,
  title={Make3d: Learning 3d scene structure from a single still image},
  author={Saxena, Ashutosh and Sun, Min and Ng, Andrew Y},
  journal={PAMI},
  year={2009}
}
```

## 📄 License

These datasets retain their original licenses:
- **NYU Depth V2:** Academic use allowed
- **KITTI:** Non-commercial use only
- **Cityscapes:** Academic use with registration
- **Make3D:** Academic use allowed

**For commercial use, contact the original dataset authors.**

## 🔗 Related Research

This dataset collection is used in:
> **"Real-Time Metric Depth Estimation via Synergistic Detection-Depth Fusion for Construction Safety"**
> [Your Name], [Institution], 2025

For the complete benchmark evaluation pipeline, see:
- REVISED_BENCHMARK_PLAN.md
- DATASET_DOWNLOAD_GUIDE.md
- DATASET_SHARING_GUIDE.md

## 📧 Contact

**Maintainer:** [Your Name]
**Email:** [Your Email]
**Institution:** [Your Institution]
**Project:** RT-MonoDepth-Construction
**GitHub:** https://github.com/[your-repo]

## ⚠️ Important Notes

1. **Cityscapes Access:** Requires registration at https://www.cityscapes-dataset.com/
2. **Large Files:** Total ~31 GB uncompressed
3. **Extraction Time:** ~3-5 minutes depending on your system
4. **Recommended Storage:** SSD for faster data loading during evaluation

## 🆘 Troubleshooting

**Q: Extraction fails?**
- Ensure you have enough disk space (40+ GB free)
- Check archive integrity with checksums

**Q: Missing files after extraction?**
- Verify CHECKSUMS.txt matches
- Re-download corrupted archives

**Q: How to download original datasets?**
- See DATASET_DOWNLOAD_GUIDE.md for detailed instructions

---

**Last Updated:** November 2025
**Version:** 1.0
EOF

        echo "✅ README created: shared_datasets/README.md"
        echo ""
        echo "Preview:"
        head -20 shared_datasets/README.md
        echo "..."
        ;;
        
    4)
        echo ""
        echo "📦 Creating full sharing package..."
        echo ""
        
        # Run all steps
        $0 <<< "1"  # Create archives
        sleep 2
        $0 <<< "2"  # Generate checksums
        sleep 1
        $0 <<< "3"  # Create README
        
        echo ""
        echo "✅ Full package ready in: shared_datasets/"
        echo ""
        echo "📊 Package contents:"
        ls -lh shared_datasets/
        echo ""
        echo "📤 Next steps:"
        echo "  1. Upload to Zenodo: https://zenodo.org/deposit/new"
        echo "  2. Or share via Google Drive"
        echo "  3. Or copy to external drive"
        echo ""
        echo "See DATASET_SHARING_GUIDE.md for detailed instructions."
        ;;
        
    5)
        echo ""
        echo "🔍 Verifying datasets..."
        echo ""
        
        # Check NYU
        if [ -f "datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat" ]; then
            SIZE=$(du -sh datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat | cut -f1)
            echo "✅ NYU Depth V2: Found ($SIZE)"
        else
            echo "❌ NYU Depth V2: NOT FOUND"
        fi
        
        # Check KITTI
        if [ -d "datasets/kitti/data_depth_annotated" ]; then
            SIZE=$(du -sh datasets/kitti | cut -f1)
            echo "✅ KITTI: Found ($SIZE)"
        else
            echo "❌ KITTI: NOT FOUND or incomplete"
        fi
        
        # Check Cityscapes
        if [ -d "datasets/cityscapes/leftImg8bit_trainvaltest" ]; then
            SIZE=$(du -sh datasets/cityscapes | cut -f1)
            echo "✅ Cityscapes: Found ($SIZE)"
        else
            echo "❌ Cityscapes: NOT FOUND or incomplete"
        fi
        
        # Check Make3D
        if [ -d "datasets/make3d" ] && [ "$(ls -A datasets/make3d/*.jpg 2>/dev/null)" ]; then
            SIZE=$(du -sh datasets/make3d | cut -f1)
            COUNT=$(ls datasets/make3d/*.jpg | wc -l | tr -d ' ')
            echo "✅ Make3D: Found ($SIZE, $COUNT images)"
        else
            echo "❌ Make3D: NOT FOUND or empty"
        fi
        
        echo ""
        echo "📊 Total storage used:"
        du -sh datasets/
        echo ""
        echo "✅ Verification complete!"
        ;;
        
    6)
        echo "👋 Exiting..."
        exit 0
        ;;
        
    *)
        echo "❌ Invalid choice. Please run again and select 1-6."
        exit 1
        ;;
esac

echo ""
echo "================================================="
echo "✅ Task complete!"
echo ""
echo "💡 Tip: See DATASET_SHARING_GUIDE.md for sharing options"
echo "   - Zenodo (best for publication)"
echo "   - Google Drive (easy for colleagues)"
echo "   - External SSD (fastest for local team)"
echo ""
