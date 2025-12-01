#!/usr/bin/env python3
# ==========================================================================
# YOLO Export Utility for Advantech Devices
# ==========================================================================
# Version:      2.6.0
# Author:       Samir Singh <samir.singh@advantech.com> and Apoorv Saxena<apoorv.saxena@advantech.com>
# Created:      January 15, 2025
# Last Updated: May 19, 2025
#
# Description:
#   This utility enables batch conversion of YOLO models to optimized formats
#   for deployment on Advantech edge AI devices with hardware acceleration.
#   Supports object detection, segmentation, classification.
#
# Terms and Conditions:
#   1. This software is provided by Advantech Corporation "as is" and any
#      express or implied warranties, including, but not limited to, the implied
#      warranties of merchantability and fitness for a particular purpose are
#      disclaimed.
#   2. In no event shall Advantech Corporation be liable for any direct, indirect,
#      incidental, special, exemplary, or consequential damages arising in any way
#      out of the use of this software.
#   3. Redistribution and use in source and binary forms, with or without
#      modification, are permitted provided that the above copyright notice and
#      this permission notice appear in all copies.
#
# Copyright (c) 2025 Advantech Corporation. All rights reserved.
# ==========================================================================
import sys
import os
import argparse
import time
import platform
import shutil
import subprocess
from collections import OrderedDict
if '/usr/lib/python3/dist-packages' in sys.path:
    sys.path.remove('/usr/lib/python3/dist-packages')
    sys.path.append('/usr/lib/python3/dist-packages')
ONNXRUNTIME_INFO_PATH = "/usr/local/lib/python3.8/dist-packages/onnxruntime_gpu-1.17.0.dist-info"
ONNXRUNTIME_MODULE_PATH = "/usr/local/lib/python3.8/dist-packages/onnxruntime"
if os.path.exists(ONNXRUNTIME_INFO_PATH) and os.path.exists(ONNXRUNTIME_MODULE_PATH):
    print("Detected ONNX Runtime GPU (package onnxruntime-gpu, module onnxruntime)")
    if os.path.dirname(ONNXRUNTIME_MODULE_PATH) not in sys.path:
        sys.path.insert(0, os.path.dirname(ONNXRUNTIME_MODULE_PATH))
else:
    print("Note: Standard ONNX Runtime paths not found")
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BG_BLUE = '\033[44m'
def print_header(text, width=80):
    print(f"\n{Colors.BG_BLUE}{Colors.BOLD}{text.center(width)}{Colors.ENDC}")
def print_subheader(text):
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{text}{Colors.ENDC}")
def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")
def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")
def print_info(text):
    print(f"{Colors.CYAN}ℹ {text}{Colors.ENDC}")
def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")
def print_progress(text):
    print(f"{Colors.BLUE}→ {text}{Colors.ENDC}")
def run_command(cmd, shell=False):
    try:
        if shell:
            output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
        else:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
        return output.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.output.strip()}"
    except Exception as e:
        return f"Error: {str(e)}"
def detect_device():
    device_info = {
        "model_type": "Unknown Device",
        "memory_gb": 0,
        "cuda_cores": 0,
        "architecture": "Unknown",
        "compute_capability": "Unknown",
        "jetpack_version": "Unknown",
    }
    try:
        if os.path.exists('/sys/class/dmi/id/board_vendor'):
            with open('/sys/class/dmi/id/board_vendor', 'r') as f:
                board_vendor = f.read().strip()
                if "Advantech" in board_vendor:
                    device_info["model_type"] = "Advantech Device"
                    if os.path.exists('/sys/class/dmi/id/product_name'):
                        with open('/sys/class/dmi/id/product_name', 'r') as f:
                            product_name = f.read().strip()
                            device_info["model_type"] = f"Advantech {product_name}"
    except:
        pass
    try:
        if os.path.exists('/etc/nv_tegra_release'):
            with open('/etc/nv_tegra_release', 'r') as f:
                jetpack_info = f.read().strip()
                device_info["jetpack_info"] = jetpack_info
                if "R35" in jetpack_info:
                    device_info["jetpack_version"] = "5.1.x"
                elif "R34" in jetpack_info:
                    device_info["jetpack_version"] = "5.0.x"
                elif "R32" in jetpack_info:
                    device_info["jetpack_version"] = "4.x"
                if "t186ref" in jetpack_info:
                    if "Advantech" in device_info["model_type"]:
                        device_info["model_type"] = "Advantech Xavier-based Device"
                    else:
                        device_info["model_type"] = "Jetson Xavier"
                    device_info["architecture"] = "Volta"
                    device_info["compute_capability"] = "7.2"
                elif "t194ref" in jetpack_info:
                    if "Advantech" in device_info["model_type"]:
                        device_info["model_type"] = "Advantech Xavier NX-based Device"
                    else:
                        device_info["model_type"] = "Jetson Xavier NX"
                    device_info["architecture"] = "Volta"
                    device_info["compute_capability"] = "7.2"
                elif "t234ref" in jetpack_info:
                    if "Advantech" in device_info["model_type"]:
                        device_info["model_type"] = "Advantech Orin-based Device"
                    else:
                        device_info["model_type"] = "Jetson Orin"
                    device_info["architecture"] = "Ampere"
                    device_info["compute_capability"] = "8.7"
    except:
        pass
    if device_info["model_type"] == "Unknown Device":
        try:
            import torch
            if torch.cuda.is_available():
                device_info["model_type"] = f"CUDA-enabled Device"
                device_info["cuda_device"] = torch.cuda.get_device_name(0)
                if "Xavier" in device_info["cuda_device"]:
                    device_info["model_type"] = "Jetson Xavier"
                    device_info["architecture"] = "Volta"
                    device_info["compute_capability"] = "7.2"
                elif "Orin" in device_info["cuda_device"]:
                    device_info["model_type"] = "Jetson Orin"
                    device_info["architecture"] = "Ampere"
                    device_info["compute_capability"] = "8.7"
        except:
            pass
    return device_info
def check_dependencies():
    dependencies = {
        "ultralytics": {"installed": False, "version": None},
        "torch": {"installed": False, "version": None, "cuda": False},
        "onnx": {"installed": False, "version": None},
        "onnxruntime": {"installed": False, "version": None},
        "tensorrt": {"installed": False, "version": None},
        "openvino": {"installed": False, "version": None},
    }
    try:
        import ultralytics
        dependencies["ultralytics"]["installed"] = True
        dependencies["ultralytics"]["version"] = ultralytics.__version__
    except:
        pass
    try:
        import torch
        dependencies["torch"]["installed"] = True
        dependencies["torch"]["version"] = torch.__version__
        dependencies["torch"]["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            dependencies["torch"]["cuda_version"] = torch.version.cuda
    except:
        pass
    try:
        import onnx
        dependencies["onnx"]["installed"] = True
        dependencies["onnx"]["version"] = onnx.__version__
    except:
        pass
    ONNXRUNTIME_INFO_PATH = "/usr/local/lib/python3.8/dist-packages/onnxruntime_gpu-1.17.0.dist-info"
    ONNXRUNTIME_MODULE_PATH = "/usr/local/lib/python3.8/dist-packages/onnxruntime"
    if os.path.exists(ONNXRUNTIME_INFO_PATH) and os.path.exists(ONNXRUNTIME_MODULE_PATH):
        dependencies["onnxruntime"]["installed"] = True
        dependencies["onnxruntime"]["version"] = "1.17.0 (GPU)"
        dependencies["onnxruntime"]["cuda"] = True
        dependencies["onnxruntime"]["is_gpu_version"] = True
        dependencies["onnxruntime"]["path"] = ONNXRUNTIME_MODULE_PATH
    else:
        try:
            import onnxruntime
            dependencies["onnxruntime"]["installed"] = True
            try:
                dependencies["onnxruntime"]["version"] = onnxruntime.__version__
            except AttributeError:
                dependencies["onnxruntime"]["version"] = "Unknown"
            if hasattr(onnxruntime, 'get_available_providers'):
                providers = onnxruntime.get_available_providers()
                cuda_provider = any("CUDA" in p for p in providers)
                tensorrt_provider = any("TensorRT" in p for p in providers)
                dependencies["onnxruntime"]["cuda"] = cuda_provider or tensorrt_provider
        except:
            pass
    try:
        import tensorrt
        dependencies["tensorrt"]["installed"] = True
        dependencies["tensorrt"]["version"] = tensorrt.__version__
    except:
        pass
    try:
        import openvino
        dependencies["openvino"]["installed"] = True
        dependencies["openvino"]["version"] = openvino.__version__
    except:
        pass
    return dependencies
def check_system_libraries():
    system_libraries = {
        "cuda": {"installed": False, "version": None, "path": None},
        "cudnn": {"installed": False, "version": None, "path": None},
        "tensorrt_libs": {"installed": False, "version": None, "path": None},
        "onnx_runtime_libs": {"installed": False, "path": None},
    }
    cuda_paths = ["/usr/local/cuda", "/usr/cuda", "/opt/cuda"]
    for path in cuda_paths:
        if os.path.exists(path):
            system_libraries["cuda"]["installed"] = True
            system_libraries["cuda"]["path"] = path
            try:
                nvcc_output = run_command(f"{path}/bin/nvcc --version")
                if "release" in nvcc_output:
                    version = nvcc_output.split("release")[1].split(",")[0].strip()
                    system_libraries["cuda"]["version"] = version
            except:
                pass
            break
    cudnn_paths = [
        "/usr/lib/aarch64-linux-gnu/libcudnn.so",
        "/usr/local/cuda/lib64/libcudnn.so",
        "/usr/lib/libcudnn.so",
    ]
    for path in cudnn_paths:
        if os.path.exists(path):
            system_libraries["cudnn"]["installed"] = True
            system_libraries["cudnn"]["path"] = path
            break
    tensorrt_paths = [
        "/usr/lib/aarch64-linux-gnu/libnvinfer.so",
        "/usr/local/cuda/lib64/libnvinfer.so",
        "/usr/lib/libnvinfer.so",
    ]
    for path in tensorrt_paths:
        if os.path.exists(path):
            system_libraries["tensorrt_libs"]["installed"] = True
            system_libraries["tensorrt_libs"]["path"] = path
            break
    ONNXRUNTIME_INFO_PATH = "/usr/local/lib/python3.8/dist-packages/onnxruntime_gpu-1.17.0.dist-info"
    ONNXRUNTIME_MODULE_PATH = "/usr/local/lib/python3.8/dist-packages/onnxruntime"
    if os.path.exists(ONNXRUNTIME_INFO_PATH) and os.path.exists(ONNXRUNTIME_MODULE_PATH):
        system_libraries["onnx_runtime_libs"]["installed"] = True
        system_libraries["onnx_runtime_libs"]["path"] = ONNXRUNTIME_MODULE_PATH
        system_libraries["onnx_runtime_libs"]["info_path"] = ONNXRUNTIME_INFO_PATH
        system_libraries["onnx_runtime_libs"]["gpu"] = True
        so_files = []
        for root, dirs, files in os.walk(ONNXRUNTIME_MODULE_PATH):
            for file in files:
                if file.endswith('.so'):
                    so_files.append(os.path.join(root, file))
        if so_files:
            system_libraries["onnx_runtime_libs"]["so_files"] = so_files
    else:
        onnxruntime_paths = [
            "/usr/lib/libonnxruntime.so",
            "/usr/local/lib/libonnxruntime.so",
            "/usr/lib/aarch64-linux-gnu/libonnxruntime.so",
        ]
        for path in onnxruntime_paths:
            if os.path.exists(path):
                system_libraries["onnx_runtime_libs"]["installed"] = True
                system_libraries["onnx_runtime_libs"]["path"] = path
                break
        if not system_libraries["onnx_runtime_libs"]["installed"] and os.path.exists(ONNXRUNTIME_MODULE_PATH):
            system_libraries["onnx_runtime_libs"]["installed"] = True
            system_libraries["onnx_runtime_libs"]["path"] = ONNXRUNTIME_MODULE_PATH
    return system_libraries
def get_export_options(device_info, dependencies, system_libraries):
    export_options = []
    onnx_installed = dependencies["onnx"]["installed"]
    onnxruntime_installed = dependencies["onnxruntime"]["installed"]
    onnxruntime_cuda = dependencies["onnxruntime"].get("cuda", False)
    ONNXRUNTIME_INFO_PATH = "/usr/local/lib/python3.8/dist-packages/onnxruntime_gpu-1.17.0.dist-info"
    if os.path.exists(ONNXRUNTIME_INFO_PATH):
        onnxruntime_installed = True
        onnxruntime_cuda = True
    is_jetson = device_info["architecture"] in ["Volta", "Ampere"]
    use_half = is_jetson
    export_options.append({
        "id": 1,
        "format": "onnx",
        "name": "ONNX format (CPU mode)",
        "device": "cpu",
        "optimize": True,
        "half": False,
        "description": "Exports to ONNX format using CPU (compatible with all systems)",
        "requires": ["onnx"],
        "compatible": onnx_installed,
        "recommended": False,
    })
    if dependencies["torch"]["cuda"] or onnxruntime_cuda:
        export_options.append({
            "id": 2,
            "format": "onnx",
            "name": "ONNX format (GPU mode, FP16)" if use_half else "ONNX format (GPU mode)",
            "device": "0",
            "optimize": False,
            "half": use_half,
            "description": "Exports to ONNX format using GPU acceleration with FP16 for Jetson",
            "requires": ["onnx"],
            "compatible": onnx_installed,
            "recommended": onnxruntime_cuda and not dependencies["tensorrt"]["installed"],
        })
    if dependencies["tensorrt"]["installed"] and dependencies["torch"]["cuda"]:
        export_options.append({
            "id": 3,
            "format": "engine",
            "name": "TensorRT Engine (FP16)" if use_half else "TensorRT Engine",
            "device": "0",
            "optimize": False,
            "half": use_half,
            "description": "Exports to TensorRT engine for maximum inference speed on Jetson",
            "requires": ["tensorrt"],
            "compatible": True,
            "recommended": True,
            "workspace": 4,
        })
    if dependencies["openvino"]["installed"]:
        export_options.append({
            "id": 4,
            "format": "openvino",
            "name": "OpenVINO IR",
            "device": "cpu",
            "optimize": True,
            "half": False,
            "description": "Exports to OpenVINO IR format for Intel hardware",
            "requires": ["openvino"],
            "compatible": True,
            "recommended": False,
        })
    export_options.append({
        "id": 5,
        "format": "torchscript",
        "name": "TorchScript",
        "device": "cpu" if not dependencies["torch"]["cuda"] else "0",
        "optimize": False,
        "half": False,
        "description": "Exports to TorchScript format for PyTorch deployment",
        "requires": ["torch"],
        "compatible": dependencies["torch"]["installed"],
        "recommended": False,
    })
    return export_options
def display_device_info(device_info):
    print_subheader("Detected Device")
    print(f"Model: {Colors.BOLD}{device_info['model_type']}{Colors.ENDC}")
    if device_info.get("architecture") != "Unknown":
        print(f"GPU Architecture: {device_info['architecture']}")
        print(f"Compute Capability: {device_info['compute_capability']}")
    if device_info.get("jetpack_version") != "Unknown":
        print(f"JetPack Version: {device_info['jetpack_version']}")
    if device_info.get("jetpack_info"):
        print(f"NVIDIA System Info: {device_info['jetpack_info']}")
def display_dependencies(dependencies, system_libraries):
    print_subheader("Software Dependencies")
    headers = ["Library", "Status", "Version", "Details"]
    rows = []
    for pkg, info in dependencies.items():
        if info["installed"]:
            status = f"{Colors.GREEN}Installed{Colors.ENDC}"
            version = info.get("version", "Unknown")
            details = ""
            if pkg == "torch":
                if info.get("cuda", False):
                    details = f"{Colors.GREEN}CUDA {info.get('cuda_version', '')}{Colors.ENDC}"
                else:
                    details = f"{Colors.YELLOW}CPU only{Colors.ENDC}"
            elif pkg == "onnxruntime":
                if info.get("cuda", False) or info.get("is_gpu_version", False):
                    details = f"{Colors.GREEN}GPU support{Colors.ENDC}"
                else:
                    details = f"{Colors.YELLOW}CPU only{Colors.ENDC}"
        else:
            status = f"{Colors.RED}Not installed{Colors.ENDC}"
            version = "N/A"
            details = ""
        rows.append([pkg, status, version, details])
    print(format_table(headers, rows))
    print_subheader("System Libraries")
    headers = ["Library", "Status", "Path/Details"]
    rows = []
    for lib, info in system_libraries.items():
        if info["installed"]:
            if lib == "onnx_runtime_libs" and info.get("gpu", False):
                status = f"{Colors.GREEN}Found (GPU){Colors.ENDC}"
            else:
                status = f"{Colors.GREEN}Found{Colors.ENDC}"
            path = info.get("path", "Unknown")
            if info.get("version"):
                path += f" (v{info['version']})"
        else:
            status = f"{Colors.RED}Not found{Colors.ENDC}"
            path = "N/A"
        rows.append([lib, status, path])
    print(format_table(headers, rows))
def display_export_options(export_options):
    print_subheader("Export Options")
    for option in export_options:
        if not option["compatible"]:
            continue
        if option["recommended"]:
            print(f"{Colors.BOLD}{Colors.GREEN}[{option['id']}] {option['name']} (RECOMMENDED){Colors.ENDC}")
        else:
            print(f"{Colors.BOLD}[{option['id']}] {option['name']}{Colors.ENDC}")
        print(f"    Format: {option['format'].upper()}, Device: {'GPU' if option['device'] == '0' else 'CPU'}")
        print(f"    Half precision (FP16): {'Yes' if option['half'] else 'No'}, Optimize: {'Yes' if option['optimize'] else 'No'}")
        print(f"    {option['description']}")
        if option["requires"]:
            reqs = ", ".join(option["requires"])
            print(f"    Requires: {reqs}")
        print()
def format_table(headers, rows, widths=None):
    if not widths:
        widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]
    table = f"{Colors.BOLD}"
    for i, header in enumerate(headers):
        table += f" {header.ljust(widths[i])} "
        if i < len(headers) - 1:
            table += "|"
    table += f"{Colors.ENDC}\n"
    table += "-" * (sum(widths) + len(headers) * 3 - 1) + "\n"
    for row in rows:
        for i, cell in enumerate(row):
            table += f" {str(cell).ljust(widths[i])} "
            if i < len(row) - 1:
                table += "|"
        table += "\n"
    return table
def get_model_path(task, size):
    if task == 'detection':
        return f"yolov8{size}.pt"
    elif task == 'segmentation':
        return f"yolov8{size}-seg.pt"
    elif task == 'classification':
        return f"yolov8{size}-cls.pt"
    else:
        raise ValueError(f"Unknown task: {task}")
def get_imgsz_for_task(task):
    if task == 'classification':
        return 224
    else:
        return 640
def load_model(task='detection', model_size='n', custom_model_path=None):
    from ultralytics import YOLO
    try:
        if custom_model_path:
            model_path = custom_model_path
            print_progress(f"Loading custom model: {model_path}")
        else:
            model_path = get_model_path(task, model_size)
            print_progress(f"Loading {task} model: {model_path}")
        if os.path.exists(model_path):
            print_info(f"Using existing model file: {model_path}")
        else:
            print_info(f"Model file not found locally. Will attempt to download: {model_path}")
        start_time = time.time()
        model = YOLO(model_path)
        model_task = model.task
        model_size_info = {
            'parameters': sum(p.numel() for p in model.model.parameters()),
            'gradients': sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        }
        print_success(f"Model loaded successfully in {time.time() - start_time:.2f}s")
        print_info(f"Task: {model_task}")
        print_info(f"Parameters: {model_size_info['parameters']:,}")
        return model
    except Exception as e:
        print_error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None
def export_model(model, export_format, task, export_args):
    print_subheader(f"Exporting model to {export_format.upper()}")
    imgsz = get_imgsz_for_task(task)
    export_args["imgsz"] = imgsz
    print_info(f"Task: {task}")
    print_info(f"Image size: {imgsz}x{imgsz}")
    print_info("Export settings:")
    for key, value in export_args.items():
        print(f"  - {key}: {value}")
    try:
        export_path = model.export(format=export_format, **export_args)
        print_success(f"Model exported successfully to: {export_path}")
        file_size_mb = os.path.getsize(export_path) / (1024 * 1024)
        print_info(f"Export file size: {file_size_mb:.2f} MB")
        return export_path
    except AssertionError as e:
        if "--optimize not compatible with cuda devices" in str(e):
            print_warning("CUDA optimization issue detected, retrying with CPU...")
            export_args["device"] = "cpu"
            try:
                export_path = model.export(format=export_format, **export_args)
                print_success(f"Model exported with CPU fallback to: {export_path}")
                file_size_mb = os.path.getsize(export_path) / (1024 * 1024)
                print_info(f"Export file size: {file_size_mb:.2f} MB")
                return export_path
            except Exception as retry_e:
                print_error(f"Retry failed: {retry_e}")
                return None
        else:
            print_error(f"Export failed: {e}")
            return None
    except Exception as e:
        print_error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None
def list_required_files(export_format, export_path, task):
    print_subheader(f"Required Files for {export_format.upper()} Inference")
    imgsz = get_imgsz_for_task(task)
    if export_format == "onnx":
        print_info("For ONNX inference, you need:")
        print(f"1. The exported model: {os.path.basename(export_path)}")
        print("2. ONNX Runtime libraries (libonnxruntime.so)")
        print("3. Python: onnx, onnxruntime-gpu, numpy, opencv-python")
        print_subheader("Example Inference Code")
        print("```python")
        print("import onnxruntime as ort")
        print("import numpy as np")
        print("import cv2")
        print(f"session = ort.InferenceSession('{os.path.basename(export_path)}',")
        print("    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])")
        print("img = cv2.imread('image.jpg')")
        print(f"img = cv2.resize(img, ({imgsz}, {imgsz}))")
        print("img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)")
        print("img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0")
        print("img = np.expand_dims(img, axis=0)")
        if task == 'classification':
            print("# Classification output: class probabilities")
            print("outputs = session.run(None, {'images': img})")
            print("probs = outputs[0][0]")
            print("class_id = np.argmax(probs)")
            print("confidence = probs[class_id]")
        else:
            print("outputs = session.run(None, {'images': img})")
            print("# Process detection/segmentation outputs")
        print("```")
    elif export_format == "engine":
        print_info("For TensorRT inference, you need:")
        print(f"1. The exported engine: {os.path.basename(export_path)}")
        print("2. TensorRT libraries (libnvinfer.so, libnvonnxparser.so)")
        print("3. CUDA libraries (libcudnn.so, libcublas.so)")
        print("4. Python: tensorrt, pycuda, numpy, opencv-python")
        print_subheader("Example Inference Code")
        print("```python")
        print("from ultralytics import YOLO")
        print(f"model = YOLO('{os.path.basename(export_path)}')")
        print("results = model.predict('image.jpg', conf=0.25, device=0)")
        if task == 'detection':
            print("for r in results:")
            print("    boxes = r.boxes")
            print("    for box in boxes:")
            print("        x1, y1, x2, y2 = box.xyxy[0]")
            print("        conf = box.conf[0]")
            print("        cls = box.cls[0]")
        elif task == 'segmentation':
            print("for r in results:")
            print("    masks = r.masks")
            print("    boxes = r.boxes")
        elif task == 'classification':
            print("for r in results:")
            print("    probs = r.probs")
            print("    top1_idx = probs.top1")
            print("    top1_conf = probs.top1conf")
        print("```")
    elif export_format == "torchscript":
        print_info("For TorchScript inference, you need:")
        print(f"1. The exported model: {os.path.basename(export_path)}")
        print("2. PyTorch libraries (libtorch.so)")
        print("3. Python: torch, torchvision, numpy, opencv-python")
        print_subheader("Example Inference Code")
        print("```python")
        print("import torch")
        print("import cv2")
        print("import numpy as np")
        print(f"model = torch.jit.load('{os.path.basename(export_path)}')")
        print("model.eval()")
        print("if torch.cuda.is_available():")
        print("    model = model.to('cuda')")
        print("img = cv2.imread('image.jpg')")
        print(f"img = cv2.resize(img, ({imgsz}, {imgsz}))")
        print("img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)")
        print("img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0")
        print("img = torch.from_numpy(img).unsqueeze(0)")
        print("if torch.cuda.is_available():")
        print("    img = img.to('cuda')")
        print("with torch.no_grad():")
        print("    outputs = model(img)")
        print("```")
    else:
        print_info(f"Exported model: {export_path}")
        print("Please refer to the YOLOv8 documentation for usage details.")
def display_task_options():
    print_subheader("Available YOLOv8 Tasks")
    tasks = [
        {"id": 1, "name": "Object Detection", "task": "detection", "imgsz": 640, "description": "Detect and localize objects with bounding boxes"},
        {"id": 2, "name": "Instance Segmentation", "task": "segmentation", "imgsz": 640, "description": "Detect objects and create pixel-level masks"},
        {"id": 3, "name": "Classification", "task": "classification", "imgsz": 224, "description": "Classify images into categories (224x224 input)"},
    ]
    for task in tasks:
        print(f"{Colors.BOLD}[{task['id']}] {task['name']}{Colors.ENDC}")
        print(f"    Task: {task['task']}, Input size: {task['imgsz']}x{task['imgsz']}")
        print(f"    {task['description']}")
        print()
    return tasks
def display_model_size_options(task):
    print_subheader(f"Available Model Sizes for {task.capitalize()}")
    sizes = [
        {"id": 1, "size": "n", "name": "Nano", "description": "Fastest inference, best for real-time on Jetson"},
        {"id": 2, "size": "s", "name": "Small", "description": "Good balance for edge devices"},
        {"id": 3, "size": "m", "name": "Medium", "description": "Higher accuracy, moderate speed"},
        {"id": 4, "size": "l", "name": "Large", "description": "High accuracy, slower inference"},
        {"id": 5, "size": "x", "name": "XLarge", "description": "Maximum accuracy, slowest inference"}
    ]
    for size in sizes:
        model_file = get_model_path(task, size['size'])
        print(f"{Colors.BOLD}[{size['id']}] {size['name']} (YOLOv8{size['size']}){Colors.ENDC}")
        print(f"    Model file: {model_file}")
        print(f"    {size['description']}")
        print()
    return sizes
def get_multiple_task_selections():
    selected_tasks = []
    selected_models = []
    available_tasks = display_task_options()
    try:
        while True:
            task_choice = input(f"\n{Colors.BOLD}Select a task (1-{len(available_tasks)}) or 'q' to finish: {Colors.ENDC}")
            if task_choice.lower() == 'q':
                if not selected_tasks:
                    print_warning("You must select at least one task.")
                    continue
                else:
                    break
            try:
                task_id = int(task_choice)
                if 1 <= task_id <= len(available_tasks):
                    selected_task = available_tasks[task_id-1]
                    task_name = selected_task['task']
                    print_success(f"Selected task: {selected_task['name']}")
                    available_sizes = display_model_size_options(task_name)
                    size_choice = input(f"\n{Colors.BOLD}Select model size (1-{len(available_sizes)}): {Colors.ENDC}")
                    try:
                        size_id = int(size_choice)
                        if 1 <= size_id <= len(available_sizes):
                            selected_size = available_sizes[size_id-1]
                            print_success(f"Selected size: {selected_size['name']}")
                            selected_tasks.append(task_name)
                            selected_models.append({
                                'task': task_name,
                                'size': selected_size['size'],
                                'name': f"{selected_task['name']} {selected_size['name']}",
                                'model_file': get_model_path(task_name, selected_size['size']),
                                'imgsz': get_imgsz_for_task(task_name)
                            })
                            print_info(f"Added {selected_task['name']} ({selected_size['name']}) - Input: {get_imgsz_for_task(task_name)}x{get_imgsz_for_task(task_name)}")
                        else:
                            print_error(f"Invalid size selection.")
                    except ValueError:
                        print_error("Please enter a valid number.")
                else:
                    print_error(f"Invalid task selection.")
            except ValueError:
                print_error("Please enter a valid number.")
    except KeyboardInterrupt:
        print("\nTask selection cancelled.")
        return [], []
    print_subheader("Selected Tasks and Models for Export")
    for i, model in enumerate(selected_models):
        print(f"{i+1}. {model['name']} ({model['model_file']}) - Input: {model['imgsz']}x{model['imgsz']}")
    return selected_tasks, selected_models
def batch_export_models(selected_models, export_option):
    successful_exports = []
    failed_exports = []
    print_header("Starting Batch Export", shutil.get_terminal_size().columns)
    for i, model_info in enumerate(selected_models):
        print_subheader(f"Export {i+1}/{len(selected_models)}: {model_info['name']}")
        task = model_info['task']
        imgsz = get_imgsz_for_task(task)
        export_args = {
            "half": export_option["half"],
            "simplify": True,
            "device": export_option["device"],
        }
        if export_option["format"] == "onnx":
            export_args["opset"] = 12
            export_args["dynamic"] = False
        if export_option["format"] == "engine":
            export_args["workspace"] = 4
            export_args["dynamic"] = False
        if export_option["optimize"] and export_option["device"] == "cpu":
            export_args["optimize"] = True
        print_progress(f"Loading model: {model_info['model_file']}")
        model = load_model(
            task=model_info['task'],
            model_size=model_info['size'],
            custom_model_path=None
        )
        if model:
            print_progress(f"Exporting to {export_option['format'].upper()} (Input: {imgsz}x{imgsz})...")
            export_path = export_model(model, export_option["format"], task, export_args)
            if export_path:
                successful_exports.append({
                    'model_info': model_info,
                    'export_path': export_path
                })
                print_success(f"Successfully exported {model_info['name']} to {export_path}")
            else:
                failed_exports.append(model_info)
                print_error(f"Failed to export {model_info['name']}")
        else:
            failed_exports.append(model_info)
            print_error(f"Failed to load {model_info['name']}")
        print("-" * 80)
    print_subheader("Export Summary")
    print(f"Total models: {len(selected_models)}")
    print(f"Successfully exported: {len(successful_exports)}")
    print(f"Failed: {len(failed_exports)}")
    if successful_exports:
        print_subheader("Successful Exports")
        for i, export in enumerate(successful_exports):
            task = export['model_info']['task']
            imgsz = get_imgsz_for_task(task)
            print(f"{i+1}. {export['model_info']['name']} → {export['export_path']} (Input: {imgsz}x{imgsz})")
    if failed_exports:
        print_subheader("Failed Exports")
        for i, model_info in enumerate(failed_exports):
            print(f"{i+1}. {model_info['name']}")
    return successful_exports, failed_exports
def main():
    parser = argparse.ArgumentParser(description='YOLO Export Utility for Advantech Devices')
    parser.add_argument('--batch-mode', action='store_true', help='Enable batch mode for exporting multiple models')
    parser.add_argument('--task', type=str, default='detection', choices=['detection', 'segmentation', 'classification'], help='Task type')
    parser.add_argument('--size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='Model size')
    parser.add_argument('--model', type=str, default=None, help='Custom model path')
    parser.add_argument('--format', type=str, default=None, choices=['onnx', 'engine', 'torchscript', 'openvino'], help='Export format')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/0)')
    parser.add_argument('--half', action='store_true', help='Use FP16 half precision')
    parser.add_argument('--no-half', dest='half', action='store_false', help='Use FP32 full precision')
    parser.add_argument('--imgsz', type=int, default=None, help='Image size (auto-detected by task if not set)')
    parser.add_argument('--optimize', action='store_true', help='Optimize model (CPU only)')
    parser.add_argument('--simplify', action='store_true', default=True, help='Simplify ONNX model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version')
    parser.add_argument('--workspace', type=float, default=4, help='TensorRT workspace size (GB)')
    parser.add_argument('--list', action='store_true', help='List export options and exit')
    parser.add_argument('--show-libs', action='store_true', help='Show required libraries')
    args = parser.parse_args()
    os.system('clear')
    print_header("YOLO Export Utility for Advantech Devices", shutil.get_terminal_size().columns)
    print_progress("Detecting hardware...")
    device_info = detect_device()
    print_progress("Checking software dependencies...")
    dependencies = check_dependencies()
    system_libraries = check_system_libraries()
    export_options = get_export_options(device_info, dependencies, system_libraries)
    display_device_info(device_info)
    display_dependencies(dependencies, system_libraries)
    if args.list or args.show_libs:
        display_export_options(export_options)
        if args.show_libs:
            print_subheader("Required System Libraries by Format")
            print_info("ONNX Format:")
            print("  - libonnxruntime.so")
            print("  - Python: onnx, onnxruntime-gpu, numpy, opencv-python")
            print_info("TensorRT Engine:")
            print("  - libnvinfer.so, libnvonnxparser.so, libcudnn.so, libcublas.so")
            print("  - Python: tensorrt, pycuda, numpy")
            print_info("TorchScript:")
            print("  - libtorch.so")
            print("  - Python: torch, torchvision, numpy")
        return
    batch_mode = args.batch_mode or (args.format is None and args.model is None)
    if batch_mode:
        selected_tasks, selected_models = get_multiple_task_selections()
        if not selected_models:
            print_error("No models selected for export. Exiting.")
            return
        display_export_options(export_options)
        selected_option = None
        try:
            choice = input(f"\n{Colors.BOLD}Enter export format option (1-{len(export_options)}): {Colors.ENDC}")
            choice = int(choice)
            for option in export_options:
                if option["id"] == choice:
                    selected_option = option
                    break
            if not selected_option:
                print_error(f"Invalid option: {choice}")
                return
        except KeyboardInterrupt:
            print("\nExiting...")
            return
        except ValueError:
            print_error("Please enter a valid number")
            return
        print_subheader("Selected Export Configuration")
        print(f"Format: {Colors.BOLD}{selected_option['name']}{Colors.ENDC}")
        print(f"Export format: {selected_option['format'].upper()}")
        print(f"Device: {'GPU' if selected_option['device'] == '0' else 'CPU'}")
        print(f"Half precision (FP16): {'Yes' if selected_option['half'] else 'No'}")
        print()
        try:
            confirm = input(f"{Colors.BOLD}Proceed with batch export of {len(selected_models)} models? (y/n): {Colors.ENDC}").lower()
            if confirm != 'y':
                print("Exiting...")
                return
        except KeyboardInterrupt:
            print("\nExiting...")
            return
        successful_exports, failed_exports = batch_export_models(selected_models, selected_option)
        if successful_exports:
            first_export = successful_exports[0]
            list_required_files(selected_option["format"], first_export['export_path'], first_export['model_info']['task'])
    else:
        selected_option = None
        if args.format:
            matching_options = [opt for opt in export_options if opt["format"] == args.format]
            if matching_options:
                selected_option = matching_options[0]
                if args.device is not None:
                    selected_option["device"] = args.device
                if args.half is not None:
                    selected_option["half"] = args.half
                if args.optimize:
                    selected_option["optimize"] = True
        else:
            display_export_options(export_options)
            try:
                choice = input(f"\n{Colors.BOLD}Enter option number (1-{len(export_options)}): {Colors.ENDC}")
                choice = int(choice)
                for option in export_options:
                    if option["id"] == choice:
                        selected_option = option
                        break
                if not selected_option:
                    print_error(f"Invalid option: {choice}")
                    return
            except KeyboardInterrupt:
                print("\nExiting...")
                return
            except ValueError:
                print_error("Please enter a valid number")
                return
        if not selected_option:
            compatible_options = [opt for opt in export_options if opt["compatible"]]
            if compatible_options:
                selected_option = compatible_options[0]
            else:
                print_error("No compatible export options found")
                return
        task = args.task
        imgsz = args.imgsz if args.imgsz else get_imgsz_for_task(task)
        print_subheader("Selected Export Configuration")
        print(f"Name: {Colors.BOLD}{selected_option['name']}{Colors.ENDC}")
        print(f"Task: {task}")
        print(f"Model: YOLOv8{args.size}")
        print(f"Export format: {selected_option['format'].upper()}")
        print(f"Half precision (FP16): {'Yes' if selected_option['half'] else 'No'}")
        print(f"Image size: {imgsz}x{imgsz}")
        print()
        try:
            confirm = input(f"{Colors.BOLD}Proceed with export? (y/n): {Colors.ENDC}").lower()
            if confirm != 'y':
                print("Exiting...")
                return
        except KeyboardInterrupt:
            print("\nExiting...")
            return
        print_progress("Loading model...")
        model = load_model(
            task=args.task,
            model_size=args.size,
            custom_model_path=args.model
        )
        if not model:
            print_error("Failed to load model. Exiting.")
            return
        export_args = {
            "half": selected_option["half"],
            "simplify": args.simplify,
            "device": selected_option["device"],
        }
        if selected_option["format"] == "onnx":
            export_args["opset"] = args.opset
            export_args["dynamic"] = False
        if selected_option["format"] == "engine":
            export_args["workspace"] = args.workspace
            export_args["dynamic"] = False
        if selected_option["optimize"] and selected_option["device"] == "cpu":
            export_args["optimize"] = True
        if args.imgsz:
            export_args["imgsz"] = args.imgsz
        export_path = export_model(model, selected_option["format"], task, export_args)
        if export_path:
            list_required_files(selected_option["format"], export_path, task)
    print_header("Export Complete", shutil.get_terminal_size().columns)
if __name__ == "__main__":
    main()
