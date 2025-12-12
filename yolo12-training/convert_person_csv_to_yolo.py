#!/usr/bin/env python3
"""
Convert Person Dataset from CSV format to YOLO format
Assigns person class as ID 17 (to merge with construction equipment 0-16)
"""

import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

def csv_to_yolo(csv_path, images_dir, output_labels_dir, class_id=17):
    """
    Convert CSV annotations to YOLO format.
    
    Args:
        csv_path: Path to CSV annotation file
        images_dir: Directory containing images
        output_labels_dir: Directory to save YOLO format labels
        class_id: Class ID for person (default 17)
    """
    # Read CSV
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Create output directory
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Group by filename
    grouped = df.groupby('filename')
    
    converted = 0
    skipped = 0
    
    print(f"Converting {len(grouped)} images...")
    
    for filename, group in tqdm(grouped, desc="Converting"):
        # Get image dimensions (assuming all boxes use same image dimensions)
        if len(group) == 0:
            continue
            
        # Get first row to get image dimensions
        img_width = group.iloc[0]['width']
        img_height = group.iloc[0]['height']
        
        # Create label file
        label_filename = Path(filename).stem + '.txt'
        label_path = os.path.join(output_labels_dir, label_filename)
        
        # Convert each bounding box
        yolo_lines = []
        for _, row in group.iterrows():
            # CSV format: xmin, ymin, xmax, ymax
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']
            
            # Convert to YOLO format: x_center, y_center, width, height (normalized)
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # YOLO format: class x_center y_center width height
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)
        
        # Write to file
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        converted += 1
    
    print(f"\n✓ Converted {converted} images to YOLO format")
    print(f"✓ Labels saved to: {output_labels_dir}")
    return converted


def main():
    """Convert person dataset from CSV to YOLO format."""
    
    base_dir = Path(__file__).parent / 'person_dataset'
    
    # Train set
    train_csv = base_dir / 'train' / 'train' / '_annotations.csv'
    train_images = base_dir / 'train' / 'train'
    train_labels = base_dir / 'train' / 'labels'
    
    # Valid set
    valid_csv = base_dir / 'valid' / 'valid' / '_annotations.csv'
    valid_images = base_dir / 'valid' / 'valid'
    valid_labels = base_dir / 'valid' / 'labels'
    
    # Test set (if exists)
    test_csv = base_dir / 'test' / 'test' / '_annotations.csv'
    test_images = base_dir / 'test' / 'test'
    test_labels = base_dir / 'test' / 'labels'
    
    print("="*70)
    print("Converting Person Dataset to YOLO Format")
    print("="*70)
    print(f"Person Class ID: 17")
    print("="*70)
    
    # Convert train
    if train_csv.exists():
        print("\n[1/3] Converting TRAIN set...")
        csv_to_yolo(train_csv, train_images, train_labels, class_id=17)
    else:
        print(f"Warning: Train CSV not found at {train_csv}")
    
    # Convert valid
    if valid_csv.exists():
        print("\n[2/3] Converting VALID set...")
        csv_to_yolo(valid_csv, valid_images, valid_labels, class_id=17)
    else:
        print(f"Warning: Valid CSV not found at {valid_csv}")
    
    # Convert test
    if test_csv.exists():
        print("\n[3/3] Converting TEST set...")
        csv_to_yolo(test_csv, test_images, test_labels, class_id=17)
    else:
        print("Info: Test set not found (optional)")
    
    print("\n" + "="*70)
    print("✓ Conversion Complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run: python merge_datasets.py")
    print("2. Run: python train_yolo12n.py")


if __name__ == '__main__':
    main()
