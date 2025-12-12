#!/usr/bin/env python3
"""
Merge 18 classes into 3 classes to fix class imbalance and help Nano model.

Original 18 classes → New 3 classes:
- Class 0: Person (was class 17)
- Class 1: Vehicle (Dump truck, Mixer, Tanker, Truck, Gazelle, Autocran)
- Class 2: Equipment (Excavator, Roller, Bulldozer, Motor grader, Crane manipulator, Forklift, etc.)

This addresses the massive "Person" dominance (100K instances vs <1K for others).
"""

import os
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

# Class mapping: original_class_id → new_class_id
CLASS_MAPPING = {
    # Person → Class 0
    17: 0,
    
    # Vehicles → Class 1
    0: 1,   # Dump truck
    2: 1,   # Mixer
    3: 1,   # Tanker
    4: 1,   # Truck
    5: 1,   # Gazelle
    13: 1,  # Autocran
    
    # Equipment → Class 2
    1: 2,   # Excavator
    6: 2,   # Forklift Standart
    7: 2,   # Roller Hamm
    8: 2,   # Roller Pobeda
    9: 2,   # Bulldozer
    10: 2,  # Motor grader
    11: 2,  # Crane manipulator
    12: 2,  # Truck excavator
    14: 2,  # Bucket loader
    15: 2,  # Cleaning equipment
    16: 2,  # Asphalt distributor
}

NEW_CLASS_NAMES = {
    0: 'person',
    1: 'vehicle',
    2: 'equipment'
}


def remap_label_file(input_path, output_path):
    """Remap class IDs in a single YOLO label file."""
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            old_class = int(parts[0])
            if old_class in CLASS_MAPPING:
                new_class = CLASS_MAPPING[old_class]
                new_line = f"{new_class} {' '.join(parts[1:])}\n"
                new_lines.append(new_line)
    
    with open(output_path, 'w') as f:
        f.writelines(new_lines)


def merge_classes(source_dir, output_dir):
    """Merge 18-class dataset into 3-class dataset."""
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    print("="*80)
    print("MERGING 18 CLASSES → 3 CLASSES")
    print("="*80)
    print(f"\nSource: {source_path}")
    print(f"Output: {output_path}")
    
    # Class mapping info
    print("\n📊 Class Mapping:")
    print("  Class 0: Person (was class 17)")
    print("  Class 1: Vehicle (Dump truck, Mixer, Tanker, Truck, Gazelle, Autocran)")
    print("  Class 2: Equipment (Excavator, Roller, Bulldozer, Motor grader, etc.)")
    
    # Create output structure
    for split in ['train', 'valid', 'test']:
        for subdir in ['images', 'labels']:
            (output_path / split / subdir).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        source_images = source_path / split / 'images'
        source_labels = source_path / split / 'labels'
        output_images = output_path / split / 'images'
        output_labels = output_path / split / 'labels'
        
        if not source_images.exists():
            continue
        
        print(f"\n🔄 Processing {split} split...")
        
        # Copy images (no change needed)
        image_files = list(source_images.glob('*.jpg')) + list(source_images.glob('*.png'))
        for img_file in tqdm(image_files, desc=f"  Copying {split} images"):
            shutil.copy2(img_file, output_images / img_file.name)
        
        # Remap labels
        label_files = list(source_labels.glob('*.txt'))
        for label_file in tqdm(label_files, desc=f"  Remapping {split} labels"):
            output_label = output_labels / label_file.name
            remap_label_file(label_file, output_label)
        
        print(f"  ✓ {len(image_files):,} images copied")
        print(f"  ✓ {len(label_files):,} labels remapped")
    
    # Create new data.yaml
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 3,
        'names': NEW_CLASS_NAMES
    }
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Created: {yaml_path}")
    
    # Print statistics
    print("\n" + "="*80)
    print("MERGE COMPLETE")
    print("="*80)
    print(f"\n📁 Output directory: {output_path}")
    print(f"📄 Configuration: {yaml_path}")
    print("\n📊 New dataset structure:")
    print("  - 3 classes (person, vehicle, equipment)")
    print("  - Significantly reduced class imbalance")
    print("  - Same number of images")
    print("\n🎯 Next steps:")
    print("  1. Zip this dataset: merged_construction_safety_3class.zip")
    print("  2. Upload to Google Drive")
    print("  3. Train with imgsz=960 (for small objects)")
    print("  4. Expected mAP improvement: 0.45 → 0.70+")
    print("="*80)


if __name__ == '__main__':
    import sys
    
    # Default paths
    default_source = Path(__file__).parent / 'merged_construction_safety'
    default_output = Path(__file__).parent / 'merged_construction_safety_3class'
    
    # Allow command-line override
    source_dir = sys.argv[1] if len(sys.argv) > 1 else default_source
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    merge_classes(source_dir, output_dir)
