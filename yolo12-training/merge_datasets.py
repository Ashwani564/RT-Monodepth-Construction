#!/usr/bin/env python3
"""
Merge Person Dataset (class 17) and Construction Equipment Dataset (classes 0-16)
Creates unified dataset for YOLOv12n training
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def copy_files(src_dir, dst_dir, file_ext, desc="Copying"):
    """Copy files from source to destination."""
    os.makedirs(dst_dir, exist_ok=True)
    
    src_path = Path(src_dir)
    files = list(src_path.glob(f"*{file_ext}"))
    
    copied = 0
    for file in tqdm(files, desc=desc):
        dst_file = Path(dst_dir) / file.name
        shutil.copy2(file, dst_file)
        copied += 1
    
    return copied


def merge_datasets():
    """Merge person and construction equipment datasets."""
    
    base_dir = Path(__file__).parent
    
    # Source directories
    person_dataset = base_dir / 'person_dataset'
    construction_dataset = base_dir / 'construction_equipment-dataset'
    
    # Merged dataset directory
    merged_dataset = base_dir / 'merged_construction_safety'
    
    print("="*70)
    print("Merging Datasets for YOLOv12n Training")
    print("="*70)
    print(f"Person Dataset: {person_dataset}")
    print(f"Construction Equipment: {construction_dataset}")
    print(f"Merged Output: {merged_dataset}")
    print("="*70)
    
    # Create merged directory structure
    for split in ['train', 'valid', 'test']:
        (merged_dataset / split / 'images').mkdir(parents=True, exist_ok=True)
        (merged_dataset / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Merge TRAIN set
    print("\n[1/3] Merging TRAIN set...")
    
    # Person train
    person_train_images = person_dataset / 'train' / 'train'
    person_train_labels = person_dataset / 'train' / 'labels'
    
    # Construction train
    construction_train_images = construction_dataset / 'train' / 'images'
    construction_train_labels = construction_dataset / 'train' / 'labels'
    
    # Merged train
    merged_train_images = merged_dataset / 'train' / 'images'
    merged_train_labels = merged_dataset / 'train' / 'labels'
    
    # Copy person train
    if person_train_images.exists() and person_train_labels.exists():
        person_img_count = copy_files(person_train_images, merged_train_images, '.jpg', 
                                      desc="  Person train images")
        person_lbl_count = copy_files(person_train_labels, merged_train_labels, '.txt',
                                      desc="  Person train labels")
        print(f"  ✓ Person: {person_img_count} images, {person_lbl_count} labels")
    else:
        print(f"  ⚠ Person train not found")
        person_img_count = 0
    
    # Copy construction train
    if construction_train_images.exists() and construction_train_labels.exists():
        const_img_count = copy_files(construction_train_images, merged_train_images, '.jpg',
                                     desc="  Construction train images")
        const_lbl_count = copy_files(construction_train_labels, merged_train_labels, '.txt',
                                     desc="  Construction train labels")
        print(f"  ✓ Construction: {const_img_count} images, {const_lbl_count} labels")
    else:
        print(f"  ⚠ Construction train not found")
        const_img_count = 0
    
    print(f"  ✓ Total TRAIN: {person_img_count + const_img_count} images")
    
    # Merge VALID set
    print("\n[2/3] Merging VALID set...")
    
    # Person valid
    person_valid_images = person_dataset / 'valid' / 'valid'
    person_valid_labels = person_dataset / 'valid' / 'labels'
    
    # Construction valid
    construction_valid_images = construction_dataset / 'valid' / 'images'
    construction_valid_labels = construction_dataset / 'valid' / 'labels'
    
    # Merged valid
    merged_valid_images = merged_dataset / 'valid' / 'images'
    merged_valid_labels = merged_dataset / 'valid' / 'labels'
    
    # Copy person valid
    if person_valid_images.exists() and person_valid_labels.exists():
        person_val_img = copy_files(person_valid_images, merged_valid_images, '.jpg',
                                    desc="  Person valid images")
        person_val_lbl = copy_files(person_valid_labels, merged_valid_labels, '.txt',
                                    desc="  Person valid labels")
        print(f"  ✓ Person: {person_val_img} images, {person_val_lbl} labels")
    else:
        print(f"  ⚠ Person valid not found")
        person_val_img = 0
    
    # Copy construction valid
    if construction_valid_images.exists() and construction_valid_labels.exists():
        const_val_img = copy_files(construction_valid_images, merged_valid_images, '.jpg',
                                   desc="  Construction valid images")
        const_val_lbl = copy_files(construction_valid_labels, merged_valid_labels, '.txt',
                                   desc="  Construction valid labels")
        print(f"  ✓ Construction: {const_val_img} images, {const_val_lbl} labels")
    else:
        print(f"  ⚠ Construction valid not found")
        const_val_img = 0
    
    print(f"  ✓ Total VALID: {person_val_img + const_val_img} images")
    
    # Merge TEST set (optional)
    print("\n[3/3] Merging TEST set (if available)...")
    
    # Person test
    person_test_images = person_dataset / 'test' / 'test'
    person_test_labels = person_dataset / 'test' / 'labels'
    
    # Construction test
    construction_test_images = construction_dataset / 'test' / 'images'
    construction_test_labels = construction_dataset / 'test' / 'labels'
    
    # Merged test
    merged_test_images = merged_dataset / 'test' / 'images'
    merged_test_labels = merged_dataset / 'test' / 'labels'
    
    test_count = 0
    
    # Copy person test
    if person_test_images.exists() and person_test_labels.exists():
        person_test_img = copy_files(person_test_images, merged_test_images, '.jpg',
                                     desc="  Person test images")
        person_test_lbl = copy_files(person_test_labels, merged_test_labels, '.txt',
                                     desc="  Person test labels")
        print(f"  ✓ Person: {person_test_img} images, {person_test_lbl} labels")
        test_count += person_test_img
    
    # Copy construction test
    if construction_test_images.exists() and construction_test_labels.exists():
        const_test_img = copy_files(construction_test_images, merged_test_images, '.jpg',
                                    desc="  Construction test images")
        const_test_lbl = copy_files(construction_test_labels, merged_test_labels, '.txt',
                                    desc="  Construction test labels")
        print(f"  ✓ Construction: {const_test_img} images, {const_test_lbl} labels")
        test_count += const_test_img
    
    if test_count > 0:
        print(f"  ✓ Total TEST: {test_count} images")
    else:
        print(f"  ℹ No test set (optional)")
    
    print("\n" + "="*70)
    print("✓ Dataset Merge Complete!")
    print("="*70)
    print(f"\nMerged dataset location: {merged_dataset}")
    print(f"Total TRAIN images: {person_img_count + const_img_count}")
    print(f"Total VALID images: {person_val_img + const_val_img}")
    print(f"Total TEST images: {test_count}")
    print("\nNext step:")
    print("Run: python train_yolo12n.py")


if __name__ == '__main__':
    merge_datasets()
