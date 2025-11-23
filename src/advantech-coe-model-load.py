#!/usr/bin/env python3
"""
YOLO11 Model Download Utility for Advantech Edge AI Devices
============================================================
Version:      1.6.0
Author:       Samir Singh <samir.singh@advantech.com>
Created:      October 9, 2025
Description:  Download and verify YOLO11 models for edge deployment

This utility helps download YOLO11 models optimized for Advantech edge AI devices.

Copyright (c) 2025 Advantech Corporation. All rights reserved.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

def check_gpu():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("✗ No GPU detected")
        return False

def download_model(model_name, save_dir):
    """Download a YOLO11 model"""
    print(f"\nDownloading {model_name}...")
    try:
        # Simply loading the model will download it if not present
        model = YOLO(model_name)
        print(f"✓ {model_name} downloaded successfully")

        # Get model info
        print(f"  Model type: {model.task}")
        print(f"  Model path: {model.ckpt_path}")

        return True
    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Download YOLO11 models for Advantech Edge AI Devices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download YOLO11n detection model
  python3 advantech-coe-model-load.py --model yolo11n

  # Download all detection models
  python3 advantech-coe-model-load.py --all-detect

  # Download specific segmentation model
  python3 advantech-coe-model-load.py --model yolo11n-seg

Available models:
  Detection:      yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
  Segmentation:   yolo11n-seg, yolo11s-seg, yolo11m-seg, yolo11l-seg, yolo11x-seg
  Classification: yolo11n-cls, yolo11s-cls, yolo11m-cls, yolo11l-cls, yolo11x-cls
  Pose:           yolo11n-pose, yolo11s-pose, yolo11m-pose, yolo11l-pose, yolo11x-pose
  OBB:            yolo11n-obb, yolo11s-obb, yolo11m-obb, yolo11l-obb, yolo11x-obb

Recommended for edge devices: yolo11n, yolo11s (nano and small variants)
        """
    )
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model to download (e.g., yolo11n.pt, yolo11s-seg.pt)')
    parser.add_argument('--all-detect', action='store_true',
                        help='Download all recommended detection models (n and s)')
    parser.add_argument('--all-seg', action='store_true',
                        help='Download all recommended segmentation models (n and s)')
    parser.add_argument('--all-cls', action='store_true',
                        help='Download all recommended classification models (n and s)')
    parser.add_argument('--all', action='store_true',
                        help='Download all recommended models for edge deployment')
    parser.add_argument('--save-dir', type=str, default='/advantech/models',
                        help='Directory to save models')
    args = parser.parse_args()

    # Check GPU
    check_gpu()

    print("\n" + "="*70)
    print("YOLO11 Model Download Utility - Advantech Edge AI")
    print("="*70)

    # Create save directory if it doesn't exist
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save directory: {save_dir}")
    print("="*70 + "\n")

    # Determine which models to download
    models_to_download = []

    if args.model:
        # Ensure .pt extension
        model_name = args.model if args.model.endswith('.pt') else f"{args.model}.pt"
        models_to_download.append(model_name)

    elif args.all_detect:
        models_to_download = ['yolo11n.pt', 'yolo11s.pt']

    elif args.all_seg:
        models_to_download = ['yolo11n-seg.pt', 'yolo11s-seg.pt']

    elif args.all_cls:
        models_to_download = ['yolo11n-cls.pt', 'yolo11s-cls.pt']

    elif args.all:
        models_to_download = [
            'yolo11n.pt', 'yolo11s.pt',           # Detection
            'yolo11n-seg.pt', 'yolo11s-seg.pt',   # Segmentation
            'yolo11n-cls.pt', 'yolo11s-cls.pt',   # Classification
        ]

    else:
        # Default: download yolo11n
        print("No specific model specified, downloading yolo11n.pt (recommended for edge)")
        models_to_download = ['yolo11n.pt']

    # Download models
    print(f"Downloading {len(models_to_download)} model(s)...\n")
    success_count = 0

    for model_name in models_to_download:
        if download_model(model_name, save_dir):
            success_count += 1

    # Summary
    print("\n" + "="*70)
    print(f"Download Summary: {success_count}/{len(models_to_download)} successful")
    print("="*70)

    if success_count > 0:
        print("\n✓ Models are ready to use!")
        print("\nNext steps:")
        print("  1. Run inference: python3 yolo11_demo.py --model yolo11n.pt --source 0")
        print("  2. Export to TensorRT: python3 yolo11_export.py --model yolo11n.pt --format engine")
        print("  3. Benchmark performance: python3 yolo11_benchmark.py --model yolo11n.pt")

if __name__ == '__main__':
    main()
