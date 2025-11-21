#!/usr/bin/env python3
"""
Visualize Depth Predictions: Pre-trained vs Fine-tuned
Compare KITTI pre-trained model with Cityscapes fine-tuned model
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import sys

sys.path.append(str(Path(__file__).parent.parent))

from networks.RTMonoDepth.RTMonoDepth import DepthDecoder, DepthEncoder
from networks.RTMonoDepth.RTMonoDepth_s import DepthDecoder as DepthDecoderS, DepthEncoder as DepthEncoderS
from layers import disp_to_depth


def load_model(model_path, model_type='full', device='cuda'):
    """Load model from checkpoint"""
    encoder_path = Path(model_path) / 'encoder.pth'
    depth_path = Path(model_path) / 'depth.pth'
    
    if not encoder_path.exists() or not depth_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    # Load appropriate architecture
    if model_type == 's':
        encoder = DepthEncoderS()
        decoder = DepthDecoderS(encoder.num_ch_enc, scales=range(1))
    else:
        encoder = DepthEncoder()
        decoder = DepthDecoder(encoder.num_ch_enc, scales=range(1))
    
    # Load weights
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'), strict=False)
    decoder.load_state_dict(torch.load(depth_path, map_location='cpu'), strict=False)
    
    encoder.to(device).eval()
    decoder.to(device).eval()
    
    return encoder, decoder


@torch.no_grad()
def predict_depth(encoder, decoder, image, device='cuda'):
    """Predict depth for a single image"""
    # Convert to tensor
    if not isinstance(image, torch.Tensor):
        to_tensor = transforms.ToTensor()
        image = to_tensor(image).unsqueeze(0)
    
    image = image.to(device)
    
    # Forward pass
    features = encoder(image)
    outputs = decoder(features)
    
    # Get disparity and convert to depth
    disp = outputs[("disp", 0)]
    _, depth = disp_to_depth(disp, 0.1, 100.0)
    
    return depth.squeeze().cpu().numpy()


def visualize_comparison(image_path, pretrained_path, finetuned_path, output_path, model_type='full', device='cuda'):
    """Compare pre-trained and fine-tuned models"""
    
    print(f"Loading image: {image_path}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize for model
    transform = transforms.Compose([
        transforms.Resize((192, 640)),
    ])
    image_resized = transform(image)
    
    print("Loading pre-trained model...")
    pretrained_encoder, pretrained_decoder = load_model(pretrained_path, model_type, device)
    
    print("Loading fine-tuned model...")
    finetuned_encoder, finetuned_decoder = load_model(finetuned_path, model_type, device)
    
    print("Predicting depth with pre-trained model...")
    depth_pretrained = predict_depth(pretrained_encoder, pretrained_decoder, image_resized, device)
    
    print("Predicting depth with fine-tuned model...")
    depth_finetuned = predict_depth(finetuned_encoder, finetuned_decoder, image_resized, device)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Input Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Pre-trained depth
    im1 = axes[0, 1].imshow(depth_pretrained, cmap='magma', vmin=0, vmax=80)
    axes[0, 1].set_title('Pre-trained (KITTI weights)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label='Depth (m)')
    
    # Fine-tuned depth
    im2 = axes[1, 0].imshow(depth_finetuned, cmap='magma', vmin=0, vmax=80)
    axes[1, 0].set_title('Fine-tuned (Cityscapes)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04, label='Depth (m)')
    
    # Difference map
    depth_diff = np.abs(depth_finetuned - depth_pretrained)
    im3 = axes[1, 1].imshow(depth_diff, cmap='hot', vmin=0, vmax=20)
    axes[1, 1].set_title('Absolute Difference', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04, label='|Δ Depth| (m)')
    
    # Overall title
    fig.suptitle(f'Depth Estimation Comparison\n{Path(image_path).name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    
    # Also display if in interactive mode
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize Pre-trained vs Fine-tuned Models')
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--pretrained_path', type=str, 
                        default='weights/RTMonoDepth/full/sh_640_192',
                        help='Path to pre-trained (KITTI) weights')
    parser.add_argument('--finetuned_path', type=str,
                        default='finetune/checkpoints/full_sh_640_192/final_weights',
                        help='Path to fine-tuned (Cityscapes) weights')
    parser.add_argument('--output', type=str, default='finetune/visualizations/comparison.png',
                        help='Output path for visualization')
    parser.add_argument('--model_type', type=str, default='full', choices=['full', 's'],
                        help='Model architecture type')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"\n{'='*60}")
    print(f"Depth Prediction Comparison")
    print(f"{'='*60}")
    print(f"Image: {args.image}")
    print(f"Pre-trained: {args.pretrained_path}")
    print(f"Fine-tuned: {args.finetuned_path}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    visualize_comparison(
        args.image,
        args.pretrained_path,
        args.finetuned_path,
        args.output,
        args.model_type,
        device
    )
    
    print(f"\n{'='*60}")
    print(f"Comparison complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
