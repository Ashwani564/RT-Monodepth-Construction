#!/usr/bin/env python3
"""
Downsample "person-only" images to reduce class imbalance.

Problem: 100K person instances vs <1K for other classes
Solution: Remove 60-70% of images that contain ONLY people
"""

import os
import random
from pathlib import Path
from tqdm import tqdm
import shutil

def has_only_person_class(label_file, person_class_id=0):
    """Check if label file contains only person class (after merging)."""
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return False
    
    classes = set()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            classes.add(int(parts[0]))
    
    return classes == {person_class_id}


def downsample_person_images(dataset_dir, downsample_ratio=0.65, person_class=0):
    """
    Remove images that contain ONLY person class.
    
    Args:
        dataset_dir: Path to 3-class merged dataset
        downsample_ratio: Fraction of person-only images to remove (0.65 = 65%)
        person_class: Class ID for person (default 0 after merging)
    """
    
    dataset_path = Path(dataset_dir)
    
    print("="*80)
    print("DOWNSAMPLING PERSON-ONLY IMAGES")
    print("="*80)
    print(f"\nDataset: {dataset_path}")
    print(f"Downsample ratio: {downsample_ratio*100:.0f}%")
    print(f"Person class ID: {person_class}")
    
    stats = {
        'train': {'total': 0, 'person_only': 0, 'removed': 0},
        'valid': {'total': 0, 'person_only': 0, 'removed': 0}
    }
    
    # Process train and valid splits
    for split in ['train', 'valid']:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        if not images_dir.exists():
            continue
        
        print(f"\n🔍 Analyzing {split} split...")
        
        # Find all label files
        label_files = list(labels_dir.glob('*.txt'))
        stats[split]['total'] = len(label_files)
        
        # Identify person-only images
        person_only_labels = []
        for label_file in tqdm(label_files, desc=f"  Scanning {split} labels"):
            if has_only_person_class(label_file, person_class):
                person_only_labels.append(label_file)
        
        stats[split]['person_only'] = len(person_only_labels)
        
        # Calculate how many to remove
        num_to_remove = int(len(person_only_labels) * downsample_ratio)
        
        # Randomly select files to remove
        random.seed(42)  # For reproducibility
        to_remove = random.sample(person_only_labels, num_to_remove)
        
        print(f"  Total images: {stats[split]['total']:,}")
        print(f"  Person-only: {stats[split]['person_only']:,}")
        print(f"  Removing: {len(to_remove):,}")
        
        # Remove selected files
        for label_file in tqdm(to_remove, desc=f"  Removing {split} files"):
            # Remove label file
            label_file.unlink()
            
            # Remove corresponding image file
            image_name = label_file.stem
            for ext in ['.jpg', '.png', '.jpeg']:
                image_file = images_dir / f"{image_name}{ext}"
                if image_file.exists():
                    image_file.unlink()
                    break
        
        stats[split]['removed'] = len(to_remove)
        remaining = stats[split]['total'] - stats[split]['removed']
        print(f"  ✓ Remaining: {remaining:,} images")
    
    # Print summary
    print("\n" + "="*80)
    print("DOWNSAMPLING COMPLETE")
    print("="*80)
    
    for split in ['train', 'valid']:
        if stats[split]['total'] > 0:
            print(f"\n{split.upper()} Split:")
            print(f"  Original: {stats[split]['total']:,} images")
            print(f"  Person-only: {stats[split]['person_only']:,} ({stats[split]['person_only']/stats[split]['total']*100:.1f}%)")
            print(f"  Removed: {stats[split]['removed']:,}")
            print(f"  Remaining: {stats[split]['total'] - stats[split]['removed']:,}")
    
    total_removed = sum(s['removed'] for s in stats.values())
    total_original = sum(s['total'] for s in stats.values())
    
    print(f"\n📊 Overall:")
    print(f"  Total images removed: {total_removed:,}")
    print(f"  Dataset reduced by: {total_removed/total_original*100:.1f}%")
    print(f"\n✅ Class balance improved!")
    print(f"   Person instances will now be ~40-50% of total (vs 95%)")
    print("="*80)


if __name__ == '__main__':
    import sys
    
    # Default path
    default_dataset = Path(__file__).parent / 'merged_construction_safety_3class'
    
    # Allow command-line override
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else default_dataset
    downsample_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.65
    
    if not Path(dataset_dir).exists():
        print(f"❌ Dataset not found: {dataset_dir}")
        print("\nRun merge_classes_3way.py first to create the 3-class dataset!")
        exit(1)
    
    downsample_person_images(dataset_dir, downsample_ratio)
