# Advantech-YOLO-Vision-Applications

A professional toolkit for deploying optimized YOLOv8 vision applications on Advantech edge AI devices with hardware acceleration.

## Overview

This repository provides a streamlined solution for running YOLOv8 computer vision applications on Advantech edge AI hardware. The toolkit automatically detects your device capabilities and sets up an optimized environment for computer vision tasks with full hardware acceleration support.

Designed specifically for Advantech edge AI devices based on NVIDIA Jetson platforms, this toolkit enables rapid deployment of object detection, instance segmentation, and classification applications with minimal configuration required.

## Features

- **Complete Docker Environment**: Pre-configured container with all necessary hardware acceleration settings
- **Optimized Model Management**: Tools for downloading and converting YOLOv8 models to accelerated formats
- **Hardware Acceleration Support**: Full integration with NVIDIA CUDA, TensorRT, and GStreamer
- **X11 Display Support**: Seamless visualization of model outputs directly from the container
- **Multiple Vision Applications**: Ready-to-use applications for object detection, segmentation, and classification

## Applications Included

### Object Detection
- Real-time object detection using YOLOv8
- Support for 80+ COCO dataset classes
- Configurable confidence thresholds and post-processing

### Instance Segmentation
- Pixel-level object segmentation for precise boundary detection
- Multi-class segmentation capabilities
- Visualization tools for segmentation masks

### Object Classification
- High-accuracy image classification
- Support for custom classification tasks
- Class confidence visualization

## Quick Start

1. Clone this repository:
```bash
git clone https://github.com/Advantech-EdgeSync-Containers/Advantech-YOLO-Vision-Applications.git
cd Advantech-YOLO-Vision-Applications
```

2. Start the container environment:
```bash
chmod +x build.sh
./build.sh
```

3. The Docker container will launch with all necessary hardware acceleration. You can access the applications as described in the Usage sections below.

## Utility Usage

### Model Loading Utility

The `advantech-coe-model-load.py` utility helps download optimized YOLOv8 models for your Advantech device:

```bash
python3 src/advantech-coe-model-load.py [--task TASK] [--size SIZE] [--dir DIR]
```

Parameters:
- `--task`: Choose from 'detection', 'segmentation', or 'classification' (default: detection)
- `--size`: Model size, use 'n' for nano or 's' for small (default: based on device)
- `--dir`: Directory to save models (default: current directory)

Examples:
```bash
# Download a YOLOv8n detection model
python3 src/advantech-coe-model-load.py --task detection --size n

# Download a YOLOv8s segmentation model
python3 src/advantech-coe-model-load.py --task segmentation --size s --dir ./models
```

### Model Export Utility

The `advantech-coe-model-export.py` utility converts YOLOv8 models to optimized formats for edge deployment:

```bash
python3 src/advantech-coe-model-export.py [--task TASK] [--size SIZE] [--format FORMAT] [--device DEVICE]
```

Parameters:
- `--task`: Choose from 'detection', 'segmentation', or 'classification'
- `--size`: Model size ('n' or 's' recommended)
- `--format`: Export format (onnx, engine, torchscript)
- `--device`: Device for optimization (cpu or 0 for GPU)
- `--half`: Enable half precision (FP16) for faster inference

Examples:
```bash
# Export YOLOv8n to ONNX format
python3 src/advantech-coe-model-export.py --task detection --size n --format onnx

# Export YOLOv8s segmentation model to TensorRT engine with half precision
python3 src/advantech-coe-model-export.py --task segmentation --size s --format engine --device 0 --half
```

## Application Usage

### YOLOv8 Vision Application

The main `advantech-yolo.py` application offers a complete solution for running YOLOv8 models:

```bash
python3 src/advantech-yolo.py [--input SOURCE] [--model MODEL] [--task TASK] [--conf CONF] [--show]
```

Parameters:
- `--input`: Path to video file, image, or camera device ID (0 for primary camera)
- `--model`: Path to YOLOv8 model file or model name (e.g., 'yolov8n.pt')
- `--task`: Task type ('detect', 'segment', 'classify')
- `--conf`: Confidence threshold (default: 0.25)
- `--show`: Display results in real-time window
- `--save`: Save results to output directory
- `--device`: Device to run inference (cpu or 0 for GPU)

Examples:
```bash
# Run object detection on a test video
python3 src/advantech-yolo.py --input data/test.mp4 --task detect --model yolov8n.pt --show

# Run instance segmentation on camera feed
python3 src/advantech-yolo.py --input 0 --task segment --model yolov8n-seg.pt --conf 0.3 --show

# Run classification on an image
python3 src/advantech-yolo.py --input data/image.jpg --task classify --model yolov8n-cls.pt --save
```

### Step-by-Step Usage Guide

1. **Set Up Environment**:
   - Start the Docker container using `./build.sh`
   - This initializes all hardware acceleration settings and dependencies

2. **Download Models**:
   - Use `advantech-coe-model-load.py` to download appropriate YOLOv8 models
   - Choose model size based on your device capability ('n' or 's' recommended)

3. **Convert Models (Optional)**:
   - Use `advantech-coe-model-export.py` to convert to optimized formats
   - TensorRT format (engine) provides the best performance on Advantech devices

4. **Run Applications**:
   - Use `advantech-yolo.py` with appropriate parameters for your use case
   - Configure input source, model, and task type as needed
   - Enable visualization with `--show` parameter

5. **Analyze Results**:
   - View real-time results on screen or examine saved outputs
   - Output includes annotations, bounding boxes, segments, or classifications based on task

## Directory Structure

```
.
├── data/               # Sample test data (includes test.mp4)
├── src/                # Source code for applications
│   ├── advantech-coe-model-export.py  # Model export utility
│   ├── advantech-coe-model-load.py    # Model download utility
│   ├── advantech-yolo.py              # Main YOLOv8 application
│   └── build.sh                       # Build script
├── docker-compose.yml  # Docker Compose configuration
├── LICENSE             # License information
└── README.md           # This file
```

## Performance Recommendations

For optimal performance on Advantech edge AI devices:

| Task | Recommended Model | Optimal Format |
|------|------------------|----------------|
| Object Detection | YOLOv8n/s | TensorRT Engine |
| Instance Segmentation | YOLOv8n-seg/s-seg | TensorRT Engine |
| Classification | YOLOv8n-cls/s-cls | TensorRT Engine |

The toolkit automatically selects appropriate configurations based on your device, with preference for smaller and more efficient models (n and s variants) to ensure real-time performance.

## Limitations

The current version of the toolkit has the following limitations:

1. **Model Size Constraints**: Only supports 'n' and 's' model variants to maintain real-time performance on edge devices. Larger models may exceed memory or computational capabilities of some devices.

2. **Pre-trained Models Only**: Currently limited to pre-trained YOLOv8 models. Custom model training requires external workflows.

3. **Resolution Limits**: Performance degrades with very high-resolution inputs (>1080p). For best results, use input at 640×640 or 1280×720 resolution.

4. **Network Dependency**: Initial model downloading requires internet connectivity.

5. **X11 Display Requirements**: Visualization features require X11 forwarding to be properly configured.

6. **Fixed Detection Classes**: Models use pre-trained COCO classes and cannot be dynamically changed without retraining.

7. **Single Stream Processing**: Currently supports processing one input stream at a time.

## Future Work

The following enhancements are planned for future releases:

- **Vision Language Models (VLM)**: Integration with multimodal models for combining visual and text understanding
- **Video Analytics**: Adding object counting, dwell time analysis, trajectory tracking, and heat maps
- **Multi-stream Processing**: Support for processing multiple camera streams simultaneously
- **REST API Interface**: HTTP API for remote inference triggering and results retrieval
- **Anomaly Detection**: Algorithms for identifying unusual patterns and behaviors
- **Edge-to-Cloud Integration**: Seamless synchronization with cloud services for data backup and advanced analytics
- **Advanced Camera Integration**: Native support for  multi-camera arrays
- **Federated Learning**: Distributed model improvement across multiple deployed devices
- **Automated Model Optimization**: Dynamic model adaptation based on deployment conditions
- **Scene Understanding**: Moving from object detection to comprehensive scene interpretation
- **Multi-sensor Fusion**: Integration with non-visual sensors for enhanced perception
- **Autonomous Decision Making**: Enabling edge devices to make inferences that trigger actions

## Use Cases

This toolkit is ideal for:

- **Industrial Quality Inspection**: Detect defects and inspect parts with instance segmentation
- **Smart Retail**: Product recognition, customer behavior analysis
- **Smart Cities**: Traffic monitoring, crowd analysis, object tracking
- **Security & Surveillance**: Perimeter monitoring, intrusion detection
- **Agriculture**: Crop monitoring, livestock tracking
- **Healthcare**: Medical image analysis, equipment tracking
- **Robotics**: Environmental perception, object manipulation guidance

## License

Copyright (c) 2025 Advantech Corporation. All rights reserved.

This software is provided by Advantech Corporation "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.

## Contact

Samir Singh - samir.singh@advantech.com

Advantech Center of Excellence