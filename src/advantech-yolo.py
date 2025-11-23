#!/usr/bin/env python3
"""
YOLO11 Vision Application for Advantech Edge AI Devices
========================================================
Version:      1.6.0
Author:       Samir Singh <samir.singh@advantech.com>
Created:      October 9, 2025
Updated:      October 28, 2025
Description:  Complete YOLO11 application supporting detection, segmentation, and classification

This script demonstrates how to use YOLO11 for multiple vision tasks with hardware
acceleration on Advantech edge AI devices.

Copyright (c) 2025 Advantech Corporation. All rights reserved.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import torch

def check_gpu():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ PyTorch Version: {torch.__version__}")
        return True
    else:
        print("✗ No GPU detected, using CPU")
        return False

def validate_task_model(task, model_path):
    """Validate that model matches the task"""
    model_name = Path(model_path).stem.lower()

    if task == 'detect':
        # Detection models shouldn't have -seg or -cls suffix
        if '-seg' in model_name or '-cls' in model_name or '-pose' in model_name:
            print(f"⚠ Warning: Model '{model_path}' may not be suitable for detection task")
            print("  Detection models: yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.")
            return False
    elif task == 'segment':
        # Segmentation models should have -seg suffix
        if '-seg' not in model_name:
            print(f"⚠ Warning: Model '{model_path}' may not be suitable for segmentation task")
            print("  Segmentation models: yolo11n-seg.pt, yolo11s-seg.pt, etc.")
            return False
    elif task == 'classify':
        # Classification models should have -cls suffix
        if '-cls' not in model_name:
            print(f"⚠ Warning: Model '{model_path}' may not be suitable for classification task")
            print("  Classification models: yolo11n-cls.pt, yolo11s-cls.pt, etc.")
            return False

    return True

def main():
    parser = argparse.ArgumentParser(
        description='YOLO11 Vision Application for Advantech Edge AI Devices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Object Detection:
    python3 advantech-yolo.py --input data/test.mp4 --task detect --model yolo11n.pt --show
    python3 advantech-yolo.py --input 0 --task detect --model yolo11n.pt --show --save

  Instance Segmentation:
    python3 advantech-yolo.py --input 0 --task segment --model yolo11n-seg.pt --conf 0.3 --show
    python3 advantech-yolo.py --input data/test.mp4 --task segment --model yolo11s-seg.pt --show --save

  Classification:
    python3 advantech-yolo.py --input data/image.jpg --task classify --model yolo11n-cls.pt --save
    python3 advantech-yolo.py --input 0 --task classify --model yolo11s-cls.pt --show
        """
    )

    # Input/Output arguments
    parser.add_argument('--input', type=str, default='0',
                        help='Input source: 0 for webcam, path to video file, or image')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        help='YOLO11 model path (e.g., yolo11n.pt, yolo11s-seg.pt, yolo11n-cls.pt)')
    parser.add_argument('--task', type=str, default='detect',
                        choices=['detect', 'segment', 'classify'],
                        help='Task type: detect, segment, or classify')

    # Inference parameters
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to run on: 0 for GPU, cpu for CPU')

    # Display and save options
    parser.add_argument('--show', action='store_true',
                        help='Display results in window')
    parser.add_argument('--save', action='store_true',
                        help='Save results to output directory')
    parser.add_argument('--save-dir', type=str, default='/advantech/results',
                        help='Directory to save results (default: /advantech/results)')

    args = parser.parse_args()

    # Check GPU availability
    gpu_available = check_gpu()

    # Set device
    if args.device == '0' and not gpu_available:
        print("⚠ Warning: GPU not available, falling back to CPU")
        args.device = 'cpu'

    # Validate task and model compatibility
    validate_task_model(args.task, args.model)

    # Print configuration
    print("\n" + "="*70)
    print("YOLO11 Vision Application - Advantech Edge AI")
    print("="*70)
    print(f"Task:       {args.task.upper()}")
    print(f"Model:      {args.model}")
    print(f"Input:      {args.input}")
    print(f"Device:     {args.device}")
    print(f"Confidence: {args.conf}")
    if args.task != 'classify':
        print(f"IoU:        {args.iou}")
    print(f"Show:       {args.show}")
    print(f"Save:       {args.save}")
    print("="*70 + "\n")

    # Load model
    print(f"Loading YOLO11 model: {args.model}")
    try:
        model = YOLO(args.model)
        print("✓ Model loaded successfully")

        # Verify model task matches requested task
        model_task = model.task
        if model_task != args.task:
            print(f"⚠ Warning: Model task is '{model_task}' but you requested '{args.task}'")
            print("  This may lead to errors. Please use the correct model for the task.")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nTip: Download models using one of these methods:")
        print("  1. python3 src/advantech-coe-model-load.py")
        print("  2. yolo task=detect mode=predict model=yolo11n.pt (auto-downloads)")
        return

    # Convert input to int if it's a digit (for webcam)
    source = int(args.input) if args.input.isdigit() else args.input

    # Prepare inference parameters based on task
    predict_params = {
        'source': source,
        'conf': args.conf,
        'device': args.device,
        'show': args.show,
        'save': args.save,
        'project': args.save_dir,
        'stream': True  # Use streaming for video/webcam
    }

    # Add IoU only for detection and segmentation tasks
    if args.task in ['detect', 'segment']:
        predict_params['iou'] = args.iou

    # Run inference
    print(f"\nRunning {args.task} inference on: {source}")
    print("Press 'q' to quit\n")

    try:
        results = model.predict(**predict_params)

        # Process results based on task
        frame_count = 0
        for result in results:
            frame_count += 1

            if args.task == 'detect':
                # Object Detection
                if result.boxes is not None and len(result.boxes) > 0:
                    num_detections = len(result.boxes)
                    print(f"Frame {frame_count}: {num_detections} objects detected")

                    # Print detected classes
                    classes = result.boxes.cls.cpu().numpy()
                    names = result.names
                    unique_classes = set([names[int(c)] for c in classes])
                    print(f"  Classes: {', '.join(sorted(unique_classes))}")

            elif args.task == 'segment':
                # Instance Segmentation
                if result.masks is not None and len(result.masks) > 0:
                    num_segments = len(result.masks)
                    print(f"Frame {frame_count}: {num_segments} instances segmented")

                    # Print segmented classes
                    if result.boxes is not None:
                        classes = result.boxes.cls.cpu().numpy()
                        names = result.names
                        unique_classes = set([names[int(c)] for c in classes])
                        print(f"  Classes: {', '.join(sorted(unique_classes))}")

            elif args.task == 'classify':
                # Classification
                if result.probs is not None:
                    top1_idx = result.probs.top1
                    top1_conf = result.probs.top1conf.item()
                    class_name = result.names[top1_idx]
                    print(f"Image {frame_count}: {class_name} ({top1_conf:.2%} confidence)")

                    # Show top 5 predictions
                    if hasattr(result.probs, 'top5'):
                        print("  Top 5 predictions:")
                        for idx in result.probs.top5:
                            conf = result.probs.data[idx].item()
                            name = result.names[idx]
                            print(f"    {name}: {conf:.2%}")

            # Break on 'q' key press if showing results
            if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopping inference...")
                break

    except KeyboardInterrupt:
        print("\n\nInference stopped by user")
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n✓ Inference completed ({frame_count} frames processed)")
    if args.save:
        print(f"✓ Results saved to: {args.save_dir}")

if __name__ == '__main__':
    main()
