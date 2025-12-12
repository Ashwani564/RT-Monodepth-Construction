#!/bin/bash
# Complete YOLOv12n Training Pipeline
# Converts, merges, and trains on construction safety dataset

set -e  # Exit on error

echo "======================================================================"
echo "YOLOv12n Training Pipeline - Construction Safety Dataset"
echo "======================================================================"
echo ""
echo "This script will:"
echo "  1. Convert person dataset from CSV to YOLO format (class 17)"
echo "  2. Merge person + construction equipment datasets"
echo "  3. Train YOLOv12n model (100 epochs)"
echo ""
echo "Estimated time: 6-30 hours (depending on hardware)"
echo "======================================================================"
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

# Navigate to yolo12-training directory
cd "$(dirname "$0")"

echo ""
echo "======================================================================"
echo "Step 1/3: Converting Person Dataset (CSV → YOLO format)"
echo "======================================================================"
echo ""

python convert_person_csv_to_yolo.py

if [ $? -ne 0 ]; then
    echo "❌ Error: Person dataset conversion failed!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Step 2/3: Merging Datasets"
echo "======================================================================"
echo ""

python merge_datasets.py

if [ $? -ne 0 ]; then
    echo "❌ Error: Dataset merging failed!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Step 3/3: Training YOLOv12n Model"
echo "======================================================================"
echo ""
echo "This will take several hours. Training can be monitored with:"
echo "  tensorboard --logdir=runs/yolo12n_construction_safety"
echo ""

sleep 3

python train_yolo12n.py

if [ $? -ne 0 ]; then
    echo "❌ Error: Training failed!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✓ Complete Pipeline Finished Successfully!"
echo "======================================================================"
echo ""
echo "Trained model location:"
echo "  runs/yolo12n_construction_safety/weights/best.pt"
echo ""
echo "Next steps:"
echo "  1. Evaluate: yolo val model=runs/.../weights/best.pt data=data.yaml"
echo "  2. Export: yolo export model=runs/.../weights/best.pt format=onnx"
echo "  3. Integrate with RT-MonoDepth pipeline"
echo ""
echo "======================================================================"
