"""
Stage 2 Benchmark: YOLO Object Detection Evaluation

This module provides comprehensive evaluation scripts for YOLO object detection
models on multiple datasets.

Datasets:
    - PPE Detection: Personal Protective Equipment (10 classes)
    - COCO val2017: Standard COCO validation set (80 classes)

Usage:
    # Run all evaluations
    python -m stage2_benchmark.run_all_evaluations
    
    # Run individual evaluations
    python -m stage2_benchmark.evaluate_yolo_ppe
    python -m stage2_benchmark.evaluate_yolo_coco

Author: Ashwani
Date: November 24, 2025
"""

__version__ = "1.0.0"
__author__ = "Ashwani"

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Default paths
MODEL_PATH = PROJECT_ROOT / "custom_yolo11n.pt"
DATASETS_ROOT = PROJECT_ROOT / "datasets" / "yolo"
RESULTS_ROOT = PROJECT_ROOT / "stage2_benchmark" / "results"

__all__ = [
    "PROJECT_ROOT",
    "MODEL_PATH",
    "DATASETS_ROOT",
    "RESULTS_ROOT",
]
