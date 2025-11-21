#!/usr/bin/env python3
"""
Dataset Loaders for Multi-Dataset Depth Evaluation
Supports: NYU Depth V2, KITTI, Cityscapes
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import scipy.io
from pathlib import Path


class NYUDepthV2Dataset(Dataset):
    """
    NYU Depth V2 dataset loader (Eigen split - 654 test images)
    """
    def __init__(self, data_path, split='test', transform=None):
        """
        Args:
            data_path: Path to datasets/nyu_depth_v2/
            split: 'test' for Eigen test split
            transform: Optional transform to apply to images
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.split = split
        
        # Load the .mat file
        mat_path = self.data_path / 'nyu_depth_v2_labeled.mat'
        if not mat_path.exists():
            raise FileNotFoundError(f"NYU dataset not found at {mat_path}")
        
        print(f"Loading NYU Depth V2 from {mat_path}...")
        
        # Try loading with scipy first, if it fails use h5py
        try:
            mat_data = scipy.io.loadmat(str(mat_path))
            images_raw = mat_data['images']  # Shape: [480, 640, 3, 1449]
            depths_raw = mat_data['depths']  # Shape: [480, 640, 1449]
            using_h5py = False
        except (NotImplementedError, ValueError):
            # MATLAB v7.3 files require h5py - load all data into memory to avoid pickling issues
            import h5py
            with h5py.File(str(mat_path), 'r') as f:
                # h5py: images shape is (1449, 3, 640, 480), depths shape is (1449, 640, 480)
                images_raw = np.array(f['images'])  # Load all into memory
                depths_raw = np.array(f['depths'])  # Load all into memory
            using_h5py = True
        
        # Use Eigen test split (indices)
        # Standard split: use every 10th image starting from index 1
        if split == 'test':
            indices = list(range(1, min(1449, images_raw.shape[-1] if not using_h5py else images_raw.shape[0]), 2))
            # Actually, standard Eigen split uses 654 images
            indices = indices[:654]
        else:
            indices = list(range(0, images_raw.shape[-1] if not using_h5py else images_raw.shape[0]))
        
        # Pre-load and convert all images and depths for the split
        self.images = []
        self.depths = []
        
        for mat_idx in indices:
            if using_h5py:
                # h5py format: (1449, 3, 640, 480) for images, (1449, 640, 480) for depths
                image = images_raw[mat_idx]  # [3, 640, 480]
                depth = depths_raw[mat_idx]  # [640, 480]
                # Transpose image to [480, 640, 3]
                image = image.transpose(2, 1, 0)  # [3, 640, 480] -> [480, 640, 3]
                depth = depth.T  # [640, 480] -> [480, 640]
            else:
                # scipy format: (480, 640, 3, 1449) for images, (480, 640, 1449) for depths
                image = images_raw[:, :, :, mat_idx]  # [480, 640, 3]
                depth = depths_raw[:, :, mat_idx]     # [480, 640]
            
            self.images.append(image)
            self.depths.append(depth)
        
        print(f"Loaded {len(self.images)} images from NYU Depth V2")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Data is already pre-loaded and formatted
        image = self.images[idx]  # [480, 640, 3]
        depth = self.depths[idx]  # [480, 640]
        
        # Convert to PIL Image
        image = Image.fromarray(image.astype(np.uint8))
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        # Convert depth to tensor
        depth = torch.from_numpy(depth).float()
        
        return {
            'image': image,
            'depth': depth,
            'filename': f'nyu_{idx:04d}',
        }


class KITTIDataset(Dataset):
    """
    KITTI dataset loader (Eigen split - 697 test images)
    """
    def __init__(self, data_path, split='eigen', transform=None):
        """
        Args:
            data_path: Path to datasets/kitti/
            split: 'eigen' for Eigen test split
            transform: Optional transform to apply to images
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.split = split
        
        # Load file list
        self.samples = self._load_eigen_split()
        
        print(f"Loaded {len(self.samples)} images from KITTI {split} split")
    
    def _load_eigen_split(self):
        """
        Load Eigen split file list
        Matches RGB images from raw_data_downloader with depth from data_depth_annotated
        """
        samples = []
        
        # Look for annotated depth data
        depth_dir = self.data_path / 'data_depth_annotated'
        raw_dir = self.data_path / 'raw_data_downloader'
        
        if not depth_dir.exists():
            print(f"Warning: KITTI depth data not found at {depth_dir}")
            return samples
        
        if not raw_dir.exists():
            print(f"Warning: KITTI raw RGB data not found at {raw_dir}")
            return samples
        
        # Search train and val folders
        for split in ['train', 'val']:
            split_dir = depth_dir / split
            if not split_dir.exists():
                continue
            
            # Iterate through sequence folders
            for seq_folder in sorted(split_dir.iterdir()):
                if not seq_folder.is_dir():
                    continue
                
                # Extract date and drive info from folder name
                # Format: 2011_09_26_drive_0014_sync
                seq_name = seq_folder.name
                parts = seq_name.split('_')
                if len(parts) < 5:
                    continue
                
                date = f"{parts[0]}_{parts[1]}_{parts[2]}"  # 2011_09_26
                
                # Look for groundtruth depth folders (image_02 or image_03)
                gt_base = seq_folder / 'proj_depth' / 'groundtruth'
                if not gt_base.exists():
                    continue
                
                # Check both image_02 and image_03
                for cam in ['image_02', 'image_03']:
                    gt_folder = gt_base / cam
                    if not gt_folder.exists():
                        continue
                    
                    # Find corresponding RGB image folder
                    img_folder = raw_dir / date / seq_name / cam / 'data'
                    if not img_folder.exists():
                        continue
                    
                    # Match depth and RGB files
                    for gt_file in sorted(gt_folder.glob('*.png')):
                        img_file = img_folder / gt_file.name
                        if img_file.exists():
                            samples.append({
                                'image': img_file,
                                'depth': gt_file,
                                'filename': f"{date}_{seq_name}_{cam}_{gt_file.stem}"
                            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image']).convert('RGB')
        
        # Load depth (KITTI depth is stored as uint16 PNG)
        depth = Image.open(sample['depth'])
        depth = np.array(depth, dtype=np.float32) / 256.0  # Convert to meters
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        # Convert depth to tensor
        depth = torch.from_numpy(depth).float()
        
        return {
            'image': image,
            'depth': depth,
            'filename': sample['filename'],
        }


class CityscapesDataset(Dataset):
    """
    Cityscapes dataset loader (validation split)
    """
    def __init__(self, data_path, split='val', transform=None):
        """
        Args:
            data_path: Path to datasets/cityscapes/
            split: 'val' or 'test'
            transform: Optional transform to apply to images
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.split = split
        
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} images from Cityscapes {split} split")
    
    def _load_samples(self):
        """Load Cityscapes samples"""
        samples = []
        
        # Image folder
        img_base = self.data_path / 'leftImg8bit_trainvaltest' / 'leftImg8bit' / self.split
        if not img_base.exists():
            print(f"Warning: Cityscapes images not found at {img_base}")
            return samples
        
        # Disparity folder (if available)
        disp_base = self.data_path / 'disparity_trainvaltest' / 'disparity' / self.split
        has_disparity = disp_base.exists()
        
        # Iterate through city folders
        for city_folder in sorted(img_base.iterdir()):
            if not city_folder.is_dir():
                continue
            
            # Find all images
            for img_file in sorted(city_folder.glob('*_leftImg8bit.png')):
                sample = {
                    'image': img_file,
                    'filename': f"{city_folder.name}_{img_file.stem}",
                }
                
                # Look for corresponding disparity
                if has_disparity:
                    disp_file = disp_base / city_folder.name / img_file.name.replace('leftImg8bit', 'disparity')
                    if disp_file.exists():
                        sample['disparity'] = disp_file
                
                samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image']).convert('RGB')
        
        # Load disparity if available
        depth = None
        if 'disparity' in sample:
            disp = Image.open(sample['disparity'])
            disp = np.array(disp, dtype=np.float32)
            
            # Convert Cityscapes disparity to depth
            # Cityscapes disparity format: disparity = (float)disparity / 256.0
            # Depth (meters) = (baseline * focal) / disparity
            # Cityscapes camera: baseline = 0.209313 m, focal = 2262.52 pixels
            disp = disp / 256.0  # Decode disparity
            
            # Avoid division by zero and clip depth to reasonable range
            baseline_focal = 0.209313 * 2262.52  # ~473.5 m·pixels
            depth = np.zeros_like(disp)
            # Only compute depth for disparity > min_disp (corresponding to max_depth of 80m)
            min_disp = baseline_focal / 80.0  # ~5.92 pixels
            valid_mask = disp > min_disp
            depth[valid_mask] = baseline_focal / disp[valid_mask]
            # Clip depth to [0, 80] meters (similar to KITTI range)
            depth = np.clip(depth, 0, 80)
            
            depth = torch.from_numpy(depth).float()
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'depth': depth,
            'filename': sample['filename'],
        }


def get_dataset(dataset_name, data_root, split='test', transform=None):
    """
    Factory function to get dataset by name
    
    Args:
        dataset_name: 'nyu', 'kitti', or 'cityscapes'
        data_root: Root path to datasets/ folder
        split: Dataset split to use
        transform: Optional transform to apply
    
    Returns:
        Dataset object
    """
    data_root = Path(data_root)
    
    if dataset_name.lower() == 'nyu':
        return NYUDepthV2Dataset(
            data_path=data_root / 'nyu_depth_v2',
            split=split,
            transform=transform
        )
    elif dataset_name.lower() == 'kitti':
        return KITTIDataset(
            data_path=data_root / 'kitti',
            split='eigen',
            transform=transform
        )
    elif dataset_name.lower() == 'cityscapes':
        return CityscapesDataset(
            data_path=data_root / 'cityscapes',
            split=split if split in ['val', 'test'] else 'val',
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'nyu', 'kitti', 'cityscapes'")


if __name__ == "__main__":
    # Test dataset loaders
    import sys
    
    data_root = Path(__file__).parent.parent / 'datasets'
    
    print("Testing dataset loaders...")
    print(f"Data root: {data_root}\n")
    
    # Test NYU
    try:
        print("1. Testing NYU Depth V2...")
        nyu_dataset = get_dataset('nyu', data_root)
        print(f"   Dataset size: {len(nyu_dataset)}")
        sample = nyu_dataset[0]
        print(f"   Image shape: {sample['image'].shape if hasattr(sample['image'], 'shape') else 'PIL Image'}")
        print(f"   Depth shape: {sample['depth'].shape}")
        print(f"   Filename: {sample['filename']}")
        print("   ✅ NYU loader OK\n")
    except Exception as e:
        print(f"   ❌ NYU loader failed: {e}\n")
    
    # Test KITTI
    try:
        print("2. Testing KITTI...")
        kitti_dataset = get_dataset('kitti', data_root)
        print(f"   Dataset size: {len(kitti_dataset)}")
        if len(kitti_dataset) > 0:
            sample = kitti_dataset[0]
            print(f"   Image shape: {sample['image'].size if hasattr(sample['image'], 'size') else 'Unknown'}")
            print(f"   Depth shape: {sample['depth'].shape}")
            print(f"   Filename: {sample['filename']}")
        print("   ✅ KITTI loader OK\n")
    except Exception as e:
        print(f"   ❌ KITTI loader failed: {e}\n")
    
    # Test Cityscapes
    try:
        print("3. Testing Cityscapes...")
        cityscapes_dataset = get_dataset('cityscapes', data_root)
        print(f"   Dataset size: {len(cityscapes_dataset)}")
        if len(cityscapes_dataset) > 0:
            sample = cityscapes_dataset[0]
            print(f"   Image shape: {sample['image'].size if hasattr(sample['image'], 'size') else 'Unknown'}")
            print(f"   Depth available: {sample['depth'] is not None}")
            print(f"   Filename: {sample['filename']}")
        print("   ✅ Cityscapes loader OK\n")
    except Exception as e:
        print(f"   ❌ Cityscapes loader failed: {e}\n")
    
    print("Dataset loader testing complete!")
