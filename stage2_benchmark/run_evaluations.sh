#!/bin/bash

# Run all YOLO evaluations for Stage 2
# Author: Ashwani
# Date: November 24, 2025

set -e  # Exit on error

echo "========================================================================"
echo "Stage 2: YOLO Object Detection Evaluation"
echo "========================================================================"
echo ""
echo "Model: custom_yolo11n.pt"
echo "Datasets: PPE Detection, COCO val2017"
echo ""
echo "========================================================================"
echo ""

# Check if model exists
if [ ! -f "custom_yolo11n.pt" ]; then
    echo "Error: Model file 'custom_yolo11n.pt' not found!"
    echo "Please ensure the model is in the project root directory."
    exit 1
fi

# Check if datasets exist
if [ ! -d "datasets/yolo/ppe-detection" ]; then
    echo "Warning: PPE detection dataset not found at datasets/yolo/ppe-detection"
    echo "PPE evaluation will be skipped."
fi

if [ ! -d "datasets/yolo/val2017" ]; then
    echo "Warning: COCO val2017 dataset not found at datasets/yolo/val2017"
    echo "COCO evaluation will be skipped."
fi

echo ""
echo "Starting evaluations..."
echo ""

# Run the combined evaluation script
python stage2_benchmark/run_all_evaluations.py "$@"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ All evaluations completed successfully!"
    echo "========================================================================"
    echo ""
    echo "Results saved in: stage2_benchmark/results/"
    echo ""
    echo "View results:"
    echo "  - PPE: stage2_benchmark/results/ppe/"
    echo "  - COCO: stage2_benchmark/results/coco/"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo "✗ Some evaluations failed. Check the output above for details."
    echo "========================================================================"
    echo ""
    exit 1
fi
