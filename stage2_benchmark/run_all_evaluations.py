#!/usr/bin/env python3
"""
YOLO Object Detection - Combined Evaluation Runner
===================================================

Runs evaluation on both PPE and COCO val2017 datasets sequentially.

Author: Ashwani
Date: November 24, 2025
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from stage2_benchmark.evaluate_yolo_ppe import PPEEvaluator
from stage2_benchmark.evaluate_yolo_coco import COCOEvaluator


def run_all_evaluations(model_path, datasets_root, output_root, skip_coco=False, skip_ppe=False):
    """
    Run evaluations on all datasets.
    
    Args:
        model_path: Path to YOLO model
        datasets_root: Root directory containing datasets
        output_root: Root directory for outputs
        skip_coco: Skip COCO evaluation
        skip_ppe: Skip PPE evaluation
    """
    results = {}
    
    # PPE Evaluation
    if not skip_ppe:
        print("\n" + "="*70)
        print("EVALUATION 1/2: PPE DETECTION DATASET")
        print("="*70)
        
        try:
            ppe_evaluator = PPEEvaluator(
                model_path=model_path,
                data_path=datasets_root / 'ppe-detection',
                output_dir=output_root / 'ppe'
            )
            results['ppe'] = ppe_evaluator.evaluate(
                conf_thres=0.25,
                iou_thres=0.45,
                img_size=640
            )
            print("\n✓ PPE evaluation completed successfully!")
        except Exception as e:
            print(f"\n✗ PPE evaluation failed: {e}")
            results['ppe'] = {'error': str(e)}
    
    # COCO Evaluation
    if not skip_coco:
        print("\n" + "="*70)
        print("EVALUATION 2/2: COCO val2017 DATASET")
        print("="*70)
        
        try:
            coco_evaluator = COCOEvaluator(
                model_path=model_path,
                data_path=datasets_root / 'val2017',
                annotations_path=datasets_root / 'annotations',
                output_dir=output_root / 'coco'
            )
            results['coco'] = coco_evaluator.evaluate(
                conf_thres=0.001,
                iou_thres=0.6,
                img_size=640
            )
            print("\n✓ COCO evaluation completed successfully!")
        except Exception as e:
            print(f"\n✗ COCO evaluation failed: {e}")
            results['coco'] = {'error': str(e)}
    
    # Print final summary
    print_final_summary(results)
    
    return results


def print_final_summary(results):
    """Print final summary of all evaluations."""
    print("\n" + "="*70)
    print("FINAL SUMMARY - ALL EVALUATIONS")
    print("="*70)
    
    for dataset_name, metrics in results.items():
        if 'error' in metrics:
            print(f"\n{dataset_name.upper()}: FAILED")
            print(f"  Error: {metrics['error']}")
        else:
            print(f"\n{dataset_name.upper()}:")
            if 'overall' in metrics:
                overall = metrics['overall']
                print(f"  mAP@50:      {overall['mAP50']:.4f} ({overall['mAP50']*100:.2f}%)")
                print(f"  mAP@50-95:   {overall['mAP50-95']:.4f} ({overall['mAP50-95']*100:.2f}%)")
                print(f"  Precision:   {overall['precision']:.4f} ({overall['precision']*100:.2f}%)")
                print(f"  Recall:      {overall['recall']:.4f} ({overall['recall']*100:.2f}%)")
                
                # Show speed metrics if available
                if 'speed' in metrics:
                    speed = metrics['speed']
                    fps = 1000.0 / speed['total_ms'] if speed['total_ms'] > 0 else 0
                    print(f"  Inference:   {speed['inference_ms']:.2f} ms ({fps:.1f} FPS)")
                
                # Show number of images if available
                if 'num_images' in metrics:
                    print(f"  Images:      {metrics['num_images']}")
                
                # Check if all metrics are zero (indicates a problem)
                if all(v == 0.0 for v in [overall['mAP50'], overall['mAP50-95'], overall['precision'], overall['recall']]):
                    print(f"\n  ⚠️  WARNING: All metrics are zero! This may indicate:")
                    print(f"      - Model was not trained on these classes")
                    print(f"      - Dataset configuration issue (missing labels)")
                    print(f"      - Incorrect data.yaml mapping")
                    print(f"      - Labels not found or in wrong format")
                    if dataset_name.lower() == 'coco':
                        print(f"      - COCO requires YOLO format labels (see warnings above)")
            else:
                print(f"  ⚠️  No metrics available")
    
    print("\n" + "="*70)
    print("All evaluations completed!")
    print("="*70 + "\n")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run YOLO object detection evaluations on multiple datasets'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='custom_yolo11n.pt',
        help='Path to YOLO model (default: custom_yolo11n.pt)'
    )
    parser.add_argument(
        '--skip-coco',
        action='store_true',
        help='Skip COCO evaluation'
    )
    parser.add_argument(
        '--skip-ppe',
        action='store_true',
        help='Skip PPE evaluation'
    )
    
    args = parser.parse_args()
    
    # Paths
    model_path = PROJECT_ROOT / args.model
    datasets_root = PROJECT_ROOT / 'datasets' / 'yolo'
    output_root = PROJECT_ROOT / 'stage2_benchmark' / 'results'
    
    # Check model exists
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    # Run evaluations
    results = run_all_evaluations(
        model_path=model_path,
        datasets_root=datasets_root,
        output_root=output_root,
        skip_coco=args.skip_coco,
        skip_ppe=args.skip_ppe
    )
    
    # Exit with error code if any evaluation failed
    if any('error' in r for r in results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
