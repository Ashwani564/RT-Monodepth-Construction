#!/bin/bash
# Complete pipeline to fix class imbalance and prepare for 960px training

set -e  # Exit on error

echo "=============================================================================="
echo "FIX CLASS IMBALANCE & SMALL OBJECT PROBLEM - COMPLETE PIPELINE"
echo "=============================================================================="
echo ""
echo "This pipeline will:"
echo "  1. Merge 18 classes → 3 classes (person, vehicle, equipment)"
echo "  2. Downsample person-only images (65% removal)"
echo "  3. Create new dataset: merged_construction_safety_3class_balanced"
echo "  4. Generate updated data.yaml for 960px training"
echo ""
echo "Expected improvements:"
echo "  - Class balance: 95% person → 40-50% person"
echo "  - Small objects: Train at 960px (deploy at 640px)"
echo "  - mAP improvement: 0.45 → 0.70+"
echo "=============================================================================="
echo ""

# Step 1: Merge 18 classes to 3 classes
echo "📋 Step 1: Merging 18 classes → 3 classes..."
python3 merge_classes_3way.py

if [ $? -ne 0 ]; then
    echo "❌ Merge failed!"
    exit 1
fi

echo ""
echo "✅ Merge complete!"
echo ""

# Step 2: Downsample person-only images
echo "📋 Step 2: Downsampling person-only images (65%)..."
python3 downsample_person_class.py merged_construction_safety_3class 0.65

if [ $? -ne 0 ]; then
    echo "❌ Downsampling failed!"
    exit 1
fi

echo ""
echo "✅ Downsampling complete!"
echo ""

# Step 3: Rename to final dataset name
echo "📋 Step 3: Creating final balanced dataset..."
if [ -d "merged_construction_safety_3class_balanced" ]; then
    echo "  Removing existing balanced dataset..."
    rm -rf merged_construction_safety_3class_balanced
fi

mv merged_construction_safety_3class merged_construction_safety_3class_balanced
echo "✅ Renamed to: merged_construction_safety_3class_balanced"
echo ""

# Step 4: Validate dataset
echo "📋 Step 4: Validating dataset..."
DATASET_PATH="merged_construction_safety_3class_balanced"

if [ -f "$DATASET_PATH/data.yaml" ]; then
    echo "✅ data.yaml found"
else
    echo "❌ data.yaml not found!"
    exit 1
fi

TRAIN_IMAGES=$(find "$DATASET_PATH/train/images" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
VALID_IMAGES=$(find "$DATASET_PATH/valid/images" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)

echo "  Train images: $TRAIN_IMAGES"
echo "  Valid images: $VALID_IMAGES"
echo "  Total: $((TRAIN_IMAGES + VALID_IMAGES))"
echo ""

# Step 5: Create zip for Colab
echo "📋 Step 5: Creating zip file for Colab..."
ZIP_NAME="merged_construction_safety_3class_balanced.zip"

if [ -f "$ZIP_NAME" ]; then
    echo "  Removing existing zip..."
    rm "$ZIP_NAME"
fi

echo "  Compressing dataset (this may take 10-15 minutes)..."
zip -r -q "$ZIP_NAME" "$DATASET_PATH"

ZIP_SIZE=$(du -h "$ZIP_NAME" | cut -f1)
echo "✅ Created: $ZIP_NAME ($ZIP_SIZE)"
echo ""

# Final summary
echo "=============================================================================="
echo "✅ PIPELINE COMPLETE!"
echo "=============================================================================="
echo ""
echo "📊 Dataset Summary:"
echo "  Name: merged_construction_safety_3class_balanced"
echo "  Classes: 3 (person, vehicle, equipment)"
echo "  Train images: $TRAIN_IMAGES"
echo "  Valid images: $VALID_IMAGES"
echo "  Zip file: $ZIP_NAME ($ZIP_SIZE)"
echo ""
echo "📤 Next Steps:"
echo "  1. Upload $ZIP_NAME to Google Drive:"
echo "     MyDrive/YOLOv12_Training/$ZIP_NAME"
echo ""
echo "  2. Stop current Colab training (it won't improve)"
echo ""
echo "  3. Update Colab notebook Cell 4 (Extract Dataset):"
echo "     dataset_zip = '/content/drive/MyDrive/YOLOv12_Training/$ZIP_NAME'"
echo "     extract_to = '/content/merged_construction_safety_3class_balanced'"
echo ""
echo "  4. Update Colab notebook Cell 6 (Training):"
echo "     data='/content/merged_construction_safety_3class_balanced/data.yaml'"
echo "     imgsz=960  # TRAIN AT 960PX FOR SMALL OBJECTS!"
echo "     batch=64   # Reduce batch size (960px needs more VRAM)"
echo ""
echo "  5. Start new training:"
echo "     - Expected mAP@50: 0.70-0.80 (vs current 0.45)"
echo "     - Training time: ~6-8 hours (larger images)"
echo ""
echo "  6. Export for Jetson Nano (after training):"
echo "     yolo export model=best.pt format=onnx imgsz=640"
echo "     (Deploy at 640px even though trained at 960px)"
echo ""
echo "=============================================================================="
echo ""
echo "💡 Key Changes:"
echo "  ✅ 18 classes → 3 classes (solves class imbalance)"
echo "  ✅ 65% person images removed (balances dataset)"
echo "  ✅ Train at 960px (solves small object problem)"
echo "  ✅ Deploy at 640px (maintains Jetson Nano speed)"
echo ""
echo "Expected results:"
echo "  - mAP@50: 0.70-0.80 (vs current 0.45)"
echo "  - Better generalization on vehicles/equipment"
echo "  - Same FPS on Jetson Nano (640px deployment)"
echo "=============================================================================="
