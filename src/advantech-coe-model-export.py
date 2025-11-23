#!/usr/bin/env python3
"""
YOLO11 Model Export Utility for Advantech Edge AI Devices
==========================================================
Version:      1.6.0
Author:       Samir Singh <samir.singh@advantech.com>
Created:      October 9, 2025
Description:  Export YOLO11 models to optimized formats (TensorRT, ONNX, etc.)

This utility exports YOLO11 models to various formats for optimized inference
on Advantech edge AI devices with NVIDIA Jetson hardware.

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
        print(f"✓ CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("✗ No GPU detected - TensorRT export requires GPU")
        return False

def export_model(model_path, format_type, half, int8, device):
    """Export model to specified format"""
    print(f"\nExporting {model_path} to {format_type.upper()} format...")

    try:
        # Load model
        model = YOLO(model_path)

        # Export with specified parameters
        export_args = {
            'format': format_type,
            'half': half,
            'int8': int8,
            'device': device,
        }

        # Add imgsz for consistent export
        export_args['imgsz'] = 640

        # Export
        exported_model = model.export(**export_args)

        print(f"✓ Model exported successfully")
        print(f"  Exported to: {exported_model}")

        return exported_model

    except Exception as e:
        print(f"✗ Export failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Export YOLO11 models for Advantech Edge AI Devices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Export Formats:
  onnx      - ONNX format (portable, good compatibility)
  engine    - TensorRT engine (best performance on Jetson, RECOMMENDED)
  torchscript - TorchScript format
  openvino  - OpenVINO format

Optimization Options:
  --half    - FP16 quantization (faster, minimal accuracy loss, RECOMMENDED)
  --int8    - INT8 quantization (fastest, some accuracy loss)

Examples:
  # Export to ONNX with FP16
  python3 advantech-coe-model-export.py --model yolo11n.pt --format onnx --half

  # Export to TensorRT engine with FP16 (RECOMMENDED for Jetson)
  python3 advantech-coe-model-export.py --model yolo11n.pt --format engine --half

  # Export to TensorRT engine with INT8
  python3 advantech-coe-model-export.py --model yolo11n.pt --format engine --int8

  # Export segmentation model
  python3 advantech-coe-model-export.py --model yolo11n-seg.pt --format engine --half

Recommendations for Jetson:
  - Use 'engine' format for best performance (TensorRT)
  - Enable --half (FP16) for good speed/accuracy balance
  - Use INT8 only if you need maximum speed and can tolerate accuracy loss
        """
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to YOLO11 model (e.g., yolo11n.pt)')
    parser.add_argument('--format', type=str, default='engine',
                        choices=['onnx', 'engine', 'torchscript', 'openvino'],
                        help='Export format (default: engine/TensorRT)')
    parser.add_argument('--half', action='store_true',
                        help='Enable FP16 quantization (recommended)')
    parser.add_argument('--int8', action='store_true',
                        help='Enable INT8 quantization')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use for export (0 for GPU, cpu for CPU)')
    args = parser.parse_args()

    # Check GPU availability
    gpu_available = check_gpu()

    if args.format == 'engine' and not gpu_available:
        print("\n✗ ERROR: TensorRT export requires GPU")
        print("  Please run this script on a device with CUDA support")
        return

    print("\n" + "="*70)
    print("YOLO11 Model Export Utility - Advantech Edge AI")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Format: {args.format}")
    print(f"FP16: {args.half}")
    print(f"INT8: {args.int8}")
    print(f"Device: {args.device}")
    print("="*70 + "\n")

    # Validate model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ Error: Model file not found: {args.model}")
        print("\nTip: Run 'python3 yolo11_download.py' to download models first")
        return

    # Export model
    exported_path = export_model(
        args.model,
        args.format,
        args.half,
        args.int8,
        args.device
    )

    if exported_path:
        print("\n" + "="*70)
        print("Export Summary")
        print("="*70)
        print(f"✓ Export completed successfully")
        print(f"✓ Exported model: {exported_path}")
        print("\nNext steps:")
        print(f"  Run inference: python3 yolo11_demo.py --model {exported_path} --source 0")
        print(f"  Benchmark: python3 yolo11_benchmark.py --model {exported_path}")
        print("="*70)
    else:
        print("\n✗ Export failed")

if __name__ == '__main__':
    main()
