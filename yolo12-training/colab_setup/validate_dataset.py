#!/usr/bin/env python3
"""
Validate merged dataset before uploading to Google Drive for H100 training.

This script checks:
1. Dataset directory structure
2. YAML configuration
3. Image counts and formats
4. Label file validity
5. Class distribution

Run this BEFORE zipping and uploading to catch any issues early.
"""

import os
import yaml
from pathlib import Path
from collections import Counter
import sys

def validate_dataset(dataset_path):
    """Validate the merged construction safety dataset."""
    
    print("=" * 80)
    print("DATASET VALIDATION FOR COLAB H100 TRAINING")
    print("=" * 80)
    
    dataset_path = Path(dataset_path)
    errors = []
    warnings = []
    
    # 1. Check directory structure
    print("\n📁 Checking directory structure...")
    
    required_dirs = [
        dataset_path / "train" / "images",
        dataset_path / "valid" / "images",
        dataset_path / "train" / "labels",
        dataset_path / "valid" / "labels",
    ]
    
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"  ✓ {dir_path.relative_to(dataset_path)}")
        else:
            errors.append(f"Missing directory: {dir_path}")
            print(f"  ✗ {dir_path.relative_to(dataset_path)} - MISSING!")
    
    # 2. Check YAML file
    print("\n📄 Checking data.yaml...")
    
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        errors.append("data.yaml not found!")
        print("  ✗ data.yaml - MISSING!")
    else:
        print(f"  ✓ data.yaml exists")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['train', 'val', 'nc', 'names']
        for field in required_fields:
            if field in config:
                print(f"  ✓ Field '{field}': {config[field] if field != 'names' else f'{len(config[field])} classes'}")
            else:
                errors.append(f"Missing field in data.yaml: {field}")
                print(f"  ✗ Field '{field}' - MISSING!")
        
        # Validate class count
        expected_classes = 18  # 0-16 equipment + 17 person
        actual_classes = config.get('nc', 0)
        if actual_classes == expected_classes:
            print(f"  ✓ Class count: {actual_classes} (correct)")
        else:
            errors.append(f"Class count mismatch: expected {expected_classes}, got {actual_classes}")
            print(f"  ✗ Class count: {actual_classes} (expected {expected_classes})")
        
        # Validate class names
        if 'names' in config:
            if len(config['names']) == actual_classes:
                print(f"  ✓ Class names: {len(config['names'])} (matches nc)")
                # Handle both list and dict formats for names
                if isinstance(config['names'], list):
                    print(f"    Classes: {', '.join(config['names'][:3])}...{config['names'][-1]}")
                elif isinstance(config['names'], dict):
                    names_list = [config['names'][i] for i in sorted(config['names'].keys())]
                    print(f"    Classes: {', '.join(names_list[:3])}...{names_list[-1]}")
            else:
                errors.append(f"Class names count ({len(config['names'])}) doesn't match nc ({actual_classes})")
    
    # 3. Count images
    print("\n🖼️  Counting images...")
    
    train_images_dir = dataset_path / "train" / "images"
    val_images_dir = dataset_path / "valid" / "images"
    
    train_images = []
    val_images = []
    
    if train_images_dir.exists():
        train_images = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png"))
        print(f"  ✓ Training images: {len(train_images):,}")
    else:
        errors.append("Training images directory not found")
    
    if val_images_dir.exists():
        val_images = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png"))
        print(f"  ✓ Validation images: {len(val_images):,}")
    else:
        errors.append("Validation images directory not found")
    
    total_images = len(train_images) + len(val_images)
    print(f"  ✓ Total images: {total_images:,}")
    
    if total_images == 0:
        errors.append("No images found in dataset!")
    elif total_images < 100:
        warnings.append(f"Very few images ({total_images}). Consider adding more data.")
    
    # 4. Count and validate labels
    print("\n🏷️  Checking labels...")
    
    train_labels_dir = dataset_path / "train" / "labels"
    val_labels_dir = dataset_path / "valid" / "labels"
    
    train_labels = []
    val_labels = []
    
    if train_labels_dir.exists():
        train_labels = list(train_labels_dir.glob("*.txt"))
        print(f"  ✓ Training labels: {len(train_labels):,}")
    
    if val_labels_dir.exists():
        val_labels = list(val_labels_dir.glob("*.txt"))
        print(f"  ✓ Validation labels: {len(val_labels):,}")
    
    total_labels = len(train_labels) + len(val_labels)
    print(f"  ✓ Total labels: {total_labels:,}")
    
    # Check for missing labels
    missing_train = len(train_images) - len(train_labels)
    missing_val = len(val_images) - len(val_labels)
    
    if missing_train > 0:
        warnings.append(f"{missing_train} training images missing labels")
        print(f"  ⚠️  {missing_train} training images missing labels")
    
    if missing_val > 0:
        warnings.append(f"{missing_val} validation images missing labels")
        print(f"  ⚠️  {missing_val} validation images missing labels")
    
    # 5. Sample label validation and class distribution
    print("\n📊 Analyzing class distribution...")
    
    all_classes = []
    invalid_labels = 0
    
    # Sample 100 labels for quick validation
    sample_labels = (train_labels[:50] + val_labels[:50]) if len(train_labels) > 0 or len(val_labels) > 0 else []
    
    for label_file in sample_labels:
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # class x y w h
                        class_id = int(parts[0])
                        all_classes.append(class_id)
                        
                        # Validate class ID range
                        if class_id < 0 or class_id >= expected_classes:
                            warnings.append(f"Invalid class ID {class_id} in {label_file.name}")
                    else:
                        invalid_labels += 1
        except Exception as e:
            invalid_labels += 1
    
    if all_classes:
        class_counts = Counter(all_classes)
        print(f"  ✓ Sampled {len(sample_labels)} label files")
        print(f"  ✓ Found {len(all_classes)} annotations")
        print(f"\n  Top 5 classes by frequency:")
        
        # Get class names (handle both list and dict)
        if isinstance(config['names'], list):
            class_names = config['names']
        elif isinstance(config['names'], dict):
            class_names = [config['names'][i] for i in range(len(config['names']))]
        else:
            class_names = []
        
        for class_id, count in class_counts.most_common(5):
            class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown({class_id})"
            print(f"    Class {class_id} ({class_name}): {count}")
        
        # Check if person class (17) is present
        if 17 in class_counts:
            print(f"  ✓ Person class (17) detected: {class_counts[17]} annotations")
        else:
            warnings.append("Person class (17) not found in sample. Verify merge was successful.")
    
    if invalid_labels > 0:
        warnings.append(f"{invalid_labels} invalid label entries found")
    
    # 6. Estimate dataset size
    print("\n💾 Estimating dataset size...")
    
    total_size = 0
    for img in train_images[:100] + val_images[:100]:  # Sample 100 images
        total_size += img.stat().st_size
    
    if len(train_images) + len(val_images) > 0:
        avg_size = total_size / min(200, len(train_images) + len(val_images))
        estimated_total = (avg_size * total_images) / (1024**3)  # Convert to GB
        print(f"  ✓ Average image size: {avg_size / 1024:.1f} KB")
        print(f"  ✓ Estimated total size: {estimated_total:.2f} GB")
        
        if estimated_total > 50:
            warnings.append(f"Large dataset ({estimated_total:.1f} GB) - upload may take a long time")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if errors:
        print("\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"  • {error}")
        print("\n⛔ FIX THESE ERRORS BEFORE PROCEEDING!")
        return False
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  • {warning}")
        print("\n✅ Dataset is valid but has minor issues (can proceed)")
    else:
        print("\n✅ ALL CHECKS PASSED!")
        print("   Dataset is ready for Colab H100 training.")
    
    # Next steps
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("\n1. Zip the dataset:")
    print(f"   cd {dataset_path.parent}")
    print(f"   zip -r merged_construction_safety.zip merged_construction_safety/")
    print("\n2. Upload to Google Drive:")
    print("   • Create folder: MyDrive/YOLOv12_Training/")
    print("   • Upload: merged_construction_safety.zip (~{:.1f} GB)".format(estimated_total if 'estimated_total' in locals() else 25))
    print("\n3. Open Colab notebook:")
    print("   • Upload: train_yolo12n_h100.ipynb")
    print("   • Select: Runtime > Change runtime type > H100 GPU")
    print("\n4. Run all cells in order")
    print("\n" + "=" * 80)
    
    return True


if __name__ == "__main__":
    # Default path (relative to this script)
    default_path = Path(__file__).parent.parent / "merged_construction_safety"
    
    # Allow custom path from command line
    if len(sys.argv) > 1:
        dataset_path = Path(sys.argv[1])
    else:
        dataset_path = default_path
    
    if not dataset_path.exists():
        print(f"❌ Error: Dataset path not found: {dataset_path}")
        print(f"\nUsage: python {sys.argv[0]} [dataset_path]")
        print(f"Example: python {sys.argv[0]} ../merged_construction_safety")
        sys.exit(1)
    
    success = validate_dataset(dataset_path)
    sys.exit(0 if success else 1)
