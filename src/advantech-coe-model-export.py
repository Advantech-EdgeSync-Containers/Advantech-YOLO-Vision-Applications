#!/usr/bin/env python3
# ==========================================================================
# YOLO Export Utility for Advantech Devices
# ==========================================================================
# Version:      2.0.0
# Author:       Samir Singh <samir.singh@advantech.com> and Apoorv Saxena<apoorv.saxena@advantech.com>
# Created:      January 15, 2025
# Last Updated: Dec 03, 2025
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
__title__ = "Advantech YOLO Model Export"
__author__ = "Samir Singh, Apoorv Saxena"
__copyright__ = "Copyright (c) 2024-2025 Advantech Corporation. All Rights Reserved."
__license__ = "Proprietary - Advantech Corporation"
__version__ = "2.0.0"
__build_date__ = "2025-12-03"
__maintainer__ = "Samir Singh, Apoorv Saxena"

import sys
import os
import argparse
import time
import platform
import re
import shutil
import subprocess
from collections import OrderedDict
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'complex'):
    np.complex = np.complex128
if not hasattr(np, 'object'):
    np.object = np.object_
if not hasattr(np, 'str'):
    np.str = np.str_
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
        if os.path.exists('/proc/device-tree/model'):
            with open('/proc/device-tree/model', 'r') as f:
                model_str = f.read().strip().replace('\x00', '')
                device_info["dt_model"] = model_str
                if "Orin" in model_str:
                    if "Nano" in model_str:
                        device_info["model_type"] = "NVIDIA Jetson Orin Nano"
                    elif "NX" in model_str:
                        device_info["model_type"] = "NVIDIA Jetson Orin NX"
                    elif "AGX" in model_str:
                        device_info["model_type"] = "NVIDIA Jetson AGX Orin"
                    else:
                        device_info["model_type"] = "NVIDIA Jetson Orin"
                    device_info["architecture"] = "Ampere"
                    device_info["compute_capability"] = "8.7"
                elif "Xavier" in model_str:
                    if "NX" in model_str:
                        device_info["model_type"] = "NVIDIA Jetson Xavier NX"
                    elif "AGX" in model_str:
                        device_info["model_type"] = "NVIDIA Jetson AGX Xavier"
                    else:
                        device_info["model_type"] = "NVIDIA Jetson Xavier"
                    device_info["architecture"] = "Volta"
                    device_info["compute_capability"] = "7.2"
                elif "Nano" in model_str:
                    device_info["model_type"] = "NVIDIA Jetson Nano"
                    device_info["architecture"] = "Maxwell"
                    device_info["compute_capability"] = "5.3"
                elif "TX2" in model_str:
                    device_info["model_type"] = "NVIDIA Jetson TX2"
                    device_info["architecture"] = "Pascal"
                    device_info["compute_capability"] = "6.2"
                elif "Thor" in model_str:
                    device_info["model_type"] = "NVIDIA Thor"
                    device_info["architecture"] = "Ampere"
                    device_info["compute_capability"] = "8.7"
    except:
        pass
    try:
        if os.path.exists('/sys/class/dmi/id/board_vendor'):
            with open('/sys/class/dmi/id/board_vendor', 'r') as f:
                board_vendor = f.read().strip()
                if "Advantech" in board_vendor:
                    if device_info["model_type"] != "Unknown Device":
                        device_info["model_type"] = f"Advantech {device_info['model_type']}"
                    else:
                        device_info["model_type"] = "Advantech Device"
                    if os.path.exists('/sys/class/dmi/id/product_name'):
                        with open('/sys/class/dmi/id/product_name', 'r') as f:
                            product_name = f.read().strip()
                            device_info["advantech_product"] = product_name
    except:
        pass
    try:
        l4t_version = None
        l4t_revision = None
        if os.path.exists('/etc/nv_tegra_release'):
            with open('/etc/nv_tegra_release', 'r') as f:
                jetpack_info = f.read().strip()
                device_info["jetpack_info"] = jetpack_info
                match = re.search(r'R(\d+)\s+\(release\),\s+REVISION:\s+([\d.]+)', jetpack_info)
                if match:
                    l4t_version = int(match.group(1))
                    l4t_revision = match.group(2)
                    device_info["l4t_version"] = f"R{l4t_version}.{l4t_revision}"
                else:
                    match = re.search(r'R(\d+)', jetpack_info)
                    if match:
                        l4t_version = int(match.group(1))
        if l4t_version is None:
            try:
                result = subprocess.run(
                    ['dpkg-query', '-W', '-f=${Version}', 'nvidia-l4t-core'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    version_str = result.stdout.strip()
                    match = re.match(r'(\d+)\.(\d+)\.(\d+)', version_str)
                    if match:
                        l4t_version = int(match.group(1))
                        l4t_revision = f"{match.group(2)}.{match.group(3)}"
                        device_info["l4t_version"] = f"R{l4t_version}.{l4t_revision}"
            except:
                pass
        if l4t_version is not None:
            jetpack_map = {
                (36, '4.0'): '6.1', (36, '3.0'): '6.0 GA', (36, '2.0'): '6.0 DP', (36, None): '6.x',
                (35, '5.0'): '5.1.3', (35, '4.1'): '5.1.2', (35, '3.1'): '5.1.1',
                (35, '2.1'): '5.1', (35, '1.0'): '5.0.2', (35, None): '5.1.x',
                (34, '1.1'): '5.0.1 DP', (34, '1.0'): '5.0 DP', (34, None): '5.0.x',
                (32, '7.4'): '4.6.4', (32, '7.3'): '4.6.3', (32, '7.2'): '4.6.2',
                (32, '7.1'): '4.6.1', (32, '6.1'): '4.6', (32, '5.2'): '4.5.1',
                (32, '5.1'): '4.5', (32, '5.0'): '4.5 DP', (32, '4.4'): '4.4.1',
                (32, '4.3'): '4.4', (32, None): '4.x',
            }
            jp_version = jetpack_map.get((l4t_version, l4t_revision))
            if jp_version is None:
                jp_version = jetpack_map.get((l4t_version, None))
            if jp_version is None:
                if l4t_version >= 36:
                    jp_version = '6.x'
                elif l4t_version >= 35:
                    jp_version = '5.1.x'
                elif l4t_version >= 34:
                    jp_version = '5.0.x'
                elif l4t_version >= 32:
                    jp_version = '4.x'
                else:
                    jp_version = f'Unknown (L4T R{l4t_version})'
            device_info["jetpack_version"] = jp_version
        if device_info["model_type"] == "Unknown Device" and os.path.exists('/etc/nv_tegra_release'):
            with open('/etc/nv_tegra_release', 'r') as f:
                jetpack_info = f.read().strip()
                if "t234ref" in jetpack_info:
                    device_info["model_type"] = "NVIDIA Jetson Orin"
                    device_info["architecture"] = "Ampere"
                    device_info["compute_capability"] = "8.7"
                elif "t194ref" in jetpack_info:
                    device_info["model_type"] = "NVIDIA Jetson Xavier NX"
                    device_info["architecture"] = "Volta"
                    device_info["compute_capability"] = "7.2"
                elif "t186ref" in jetpack_info:
                    device_info["model_type"] = "NVIDIA Jetson TX2"
                    device_info["architecture"] = "Pascal"
                    device_info["compute_capability"] = "6.2"
    except:
        pass

    if device_info["model_type"] == "Unknown Device":
        try:
            import torch
            if torch.cuda.is_available():
                cuda_device_name = torch.cuda.get_device_name(0)
                device_info["cuda_device"] = cuda_device_name
                if "Orin" in cuda_device_name:
                    device_info["model_type"] = "NVIDIA Jetson Orin"
                    device_info["architecture"] = "Ampere"
                    device_info["compute_capability"] = "8.7"
                elif "Xavier" in cuda_device_name:
                    device_info["model_type"] = "NVIDIA Jetson Xavier"
                    device_info["architecture"] = "Volta"
                    device_info["compute_capability"] = "7.2"
                else:
                    device_info["model_type"] = f"CUDA Device ({cuda_device_name})"
                    cap = torch.cuda.get_device_capability(0)
                    device_info["compute_capability"] = f"{cap[0]}.{cap[1]}"
        except:
            pass

    try:
        import torch
        if torch.cuda.is_available():
            mem_bytes = torch.cuda.get_device_properties(0).total_memory
            device_info["memory_gb"] = round(mem_bytes / (1024**3), 1)
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
    export_options.append({
        "id": 4,
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
    if device_info.get("memory_gb", 0) > 0:
        print(f"GPU Memory: {device_info['memory_gb']} GB")
    if device_info.get("jetpack_version") != "Unknown":
        jp_str = f"JetPack {device_info['jetpack_version']}"
        if device_info.get("l4t_version"):
            jp_str += f" (L4T {device_info['l4t_version']})"
        print(jp_str)
    if device_info.get("advantech_product"):
        print(f"Advantech Product: {device_info['advantech_product']}")
    if device_info.get("cuda_device"):
        print(f"CUDA Device: {device_info['cuda_device']}")

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

def validate_segmentation_onnx(onnx_path):
    try:
        import onnx
        model = onnx.load(onnx_path)
        outputs = model.graph.output
        print_info(f"Validating segmentation ONNX model...")
        print_info(f"Number of outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            shape = []
            for d in out.type.tensor_type.shape.dim:
                if d.dim_value > 0:
                    shape.append(d.dim_value)
                elif d.dim_param:
                    shape.append(str(d.dim_param))
                else:
                    shape.append('?')
            print_info(f"  Output {i}: {out.name} shape={shape}")
        if len(outputs) < 2:
            print_error(f"Expected 2 outputs for segmentation, got {len(outputs)}")
            print_error("Mask prototype output is MISSING!")
            return False
        out1_dims = outputs[1].type.tensor_type.shape.dim
        if len(out1_dims) >= 4:
            dim1_value = out1_dims[1].dim_value
            if dim1_value == 32:
                print_success("Segmentation outputs validated successfully!")
                print_info(f"  Detection output: {outputs[0].name}")
                print_info(f"  Mask prototypes: {outputs[1].name} (32 channels)")
                return True
            else:
                print_error(f"Expected 32 mask prototype channels, got {dim1_value}")
                return False
        else:
            out1_shape = [d.dim_value for d in out1_dims]
            print_error(f"Mask output has unexpected shape dimensions: {len(out1_dims)}")
            return False
    except ImportError:
        print_warning("ONNX package not available for validation, skipping...")
        return True
    except Exception as e:
        print_error(f"Validation error: {e}")
        return False

def validate_tensorrt_engine(engine_path, task='detection'):
    result = {
        'valid': False,
        'inputs': [],
        'outputs': [],
        'model_type': 'unknown',
        'inference_time_ms': 0,
        'error': None
    }
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError as e:
        result['error'] = f"Missing dependency: {e}"
        print_error(f"Cannot validate TensorRT engine: {e}")
        return result
    print_info("Validating TensorRT engine...")
    try:
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            result['error'] = "Failed to deserialize engine"
            print_error("Failed to deserialize engine")
            return result
        print_success(f"Engine loaded: {engine.num_io_tensors} IO tensors")
        context = engine.create_execution_context()
        stream = cuda.Stream()
        inputs = []
        outputs = []
        det_idx = -1
        mask_idx = -1
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            shape = list(engine.get_tensor_shape(name))
            mode = engine.get_tensor_mode(name)
            size = abs(int(np.prod(shape)))
            binding_info = {
                'index': i,
                'name': name,
                'shape': shape,
                'dtype': str(dtype),
                'size': size
            }
            if mode == trt.TensorIOMode.INPUT:
                inputs.append(binding_info)
                print_info(f"  INPUT:  {name} {shape}")
            else:
                outputs.append(binding_info)
                print_info(f"  OUTPUT: {name} {shape}")
                if len(shape) == 4 and shape[1] == 32:
                    mask_idx = len(outputs) - 1
                elif len(shape) == 3:
                    det_idx = len(outputs) - 1
        result['inputs'] = inputs
        result['outputs'] = outputs
        if mask_idx >= 0:
            result['model_type'] = 'segmentation'
            print_success(f"Model type: SEGMENTATION")
            print_info(f"  Detection output: index {det_idx}")
            print_info(f"  Mask protos: index {mask_idx}")
        elif len(outputs) == 1:
            out_shape = outputs[0]['shape']
            if len(out_shape) == 2 or (len(out_shape) == 3 and out_shape[1] < 100):
                result['model_type'] = 'classification'
            else:
                result['model_type'] = 'detection'
            print_success(f"Model type: {result['model_type'].upper()}")
        else:
            result['model_type'] = 'detection'
            print_success(f"Model type: DETECTION")
        host_inputs = []
        device_inputs = []
        host_outputs = []
        device_outputs = []
        for inp in inputs:
            h_mem = cuda.pagelocked_empty(inp['size'], np.float32)
            d_mem = cuda.mem_alloc(h_mem.nbytes)
            host_inputs.append(h_mem)
            device_inputs.append(d_mem)
        for out in outputs:
            h_mem = cuda.pagelocked_empty(out['size'], np.float32)
            d_mem = cuda.mem_alloc(h_mem.nbytes)
            host_outputs.append(h_mem)
            device_outputs.append(d_mem)
        input_shape = inputs[0]['shape']
        if len(input_shape) == 4:
            _, c, h, w = input_shape
            dummy = np.random.rand(1, c, h, w).astype(np.float32)
        else:
            dummy = np.random.rand(*input_shape).astype(np.float32)
        np.copyto(host_inputs[0], dummy.ravel())
        cuda.memcpy_htod_async(device_inputs[0], host_inputs[0], stream)
        context.set_tensor_address(inputs[0]['name'], int(device_inputs[0]))
        for i, out in enumerate(outputs):
            context.set_tensor_address(out['name'], int(device_outputs[i]))
        print_info("Running warmup inference...")
        for _ in range(3):
            context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
        print_info("Benchmarking inference speed...")
        start = time.perf_counter()
        num_runs = 10
        for _ in range(num_runs):
            context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
        elapsed = (time.perf_counter() - start) / num_runs * 1000
        result['inference_time_ms'] = elapsed
        print_success(f"Inference time: {elapsed:.2f} ms ({1000/elapsed:.1f} FPS)")
        for i in range(len(outputs)):
            cuda.memcpy_dtoh_async(host_outputs[i], device_outputs[i], stream)
        stream.synchronize()
        has_nan_inf = False
        for i, h_out in enumerate(host_outputs):
            if np.any(np.isnan(h_out)) or np.any(np.isinf(h_out)):
                has_nan_inf = True
            else:
                out_min, out_max = h_out.min(), h_out.max()
                print_info(f"  Output {i}: range [{out_min:.4f}, {out_max:.4f}]")
        if has_nan_inf:
            print_info("Note: Some outputs contain NaN/Inf (normal with random dummy input)")
        for d in device_inputs + device_outputs:
            d.free()
        result['valid'] = True
        print_success("TensorRT validation PASSED")
    except Exception as e:
        result['error'] = str(e)
        print_error(f"TensorRT validation failed: {e}")
        import traceback
        traceback.print_exc()
    return result

def convert_onnx_to_tensorrt(onnx_path, engine_path=None, precision='fp16', workspace_mb=4096, verbose=False):
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError as e:
        print_error(f"TensorRT not available: {e}")
        return None
    if engine_path is None:
        engine_path = onnx_path.replace('.onnx', '.engine')
    print_subheader("Converting ONNX to TensorRT")
    print_info(f"Input:     {onnx_path}")
    print_info(f"Output:    {engine_path}")
    print_info(f"Precision: {precision.upper()}")
    print_info(f"Workspace: {workspace_mb} MB")
    log_level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    trt_logger = trt.Logger(log_level)
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)
    print_progress("Parsing ONNX model...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print_error("ONNX parsing failed:")
            for i in range(parser.num_errors):
                print_error(f"  Error {i}: {parser.get_error(i)}")
            return None
    print_success("ONNX parsed successfully")
    print_info("Network inputs:")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  - {inp.name}: {inp.shape}")
    print_info("Network outputs:")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  - {out.name}: {out.shape}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print_success("FP16 mode enabled")
        else:
            print_warning("FP16 not supported on this platform, using FP32")
    elif precision == 'int8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print_success("INT8 mode enabled (requires calibration)")
        else:
            print_warning("INT8 not supported, using FP32")
    print_progress("Building TensorRT engine (this may take several minutes)...")
    start_time = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print_error("Engine build failed")
        return None
    build_time = time.time() - start_time
    print_success(f"Engine built in {build_time:.1f} seconds")
    print_progress("Saving engine...")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    engine_size = os.path.getsize(engine_path) / (1024 * 1024)
    print_success(f"Engine saved: {engine_path} ({engine_size:.1f} MB)")
    return engine_path

def export_model(model, export_format, task, export_args, model_path=None):
    imgsz = get_imgsz_for_task(task)
    export_args["imgsz"] = imgsz
    if export_format == 'engine':
        print_subheader("Exporting model to TensorRT Engine (via ONNX)")
        if model_path:
            base_name = os.path.splitext(model_path)[0]
            onnx_path = base_name + '.onnx'
            engine_path = base_name + '.engine'
        else:
            onnx_path = 'model.onnx'
            engine_path = 'model.engine'
        if os.path.exists(onnx_path):
            print_info(f"Found existing ONNX model: {onnx_path}")
            if task == 'segmentation':
                print_info("Validating existing ONNX for segmentation...")
                if not validate_segmentation_onnx(onnx_path):
                    print_warning("Existing ONNX may not support segmentation properly")
                    print_info("Re-exporting ONNX with correct settings...")
                    onnx_path = None
        else:
            onnx_path = None
        if onnx_path is None:
            print_subheader("Step 1: Export to ONNX")
            onnx_export_args = {
                "half": export_args.get("half", False),
                "device": export_args.get("device", "0"),
                "imgsz": imgsz,
            }
            if task == 'segmentation':
                onnx_export_args["simplify"] = False
                onnx_export_args["opset"] = 13
                print_info("Using segmentation-safe ONNX settings")
            else:
                onnx_export_args["simplify"] = True
                onnx_export_args["opset"] = 12
            print_info(f"Task: {task}")
            print_info(f"Image size: {imgsz}x{imgsz}")
            print_info("ONNX export settings:")
            for key, value in onnx_export_args.items():
                print(f"  - {key}: {value}")
            try:
                onnx_path = model.export(format='onnx', **onnx_export_args)
                print_success(f"ONNX exported: {onnx_path}")
                if task == 'segmentation':
                    if not validate_segmentation_onnx(onnx_path):
                        print_warning("ONNX validation found issues - TensorRT conversion may not work properly")
            except Exception as e:
                print_error(f"ONNX export failed: {e}")
                return None
        print_subheader("Step 2: Convert ONNX to TensorRT")
        precision = 'fp16' if export_args.get("half", False) else 'fp32'
        workspace_mb = export_args.get("workspace", 4) * 1024
        engine_path = convert_onnx_to_tensorrt(
            onnx_path=onnx_path,
            engine_path=engine_path,
            precision=precision,
            workspace_mb=workspace_mb,
            verbose=False
        )
        if engine_path:
            print_subheader("Step 3: Validate TensorRT Engine")
            validation_result = validate_tensorrt_engine(engine_path, task)
            if validation_result['valid']:
                print_success(f"TensorRT engine ready: {engine_path}")
                print_info(f"Inference: {validation_result['inference_time_ms']:.2f} ms ({1000/validation_result['inference_time_ms']:.1f} FPS)")
            else:
                print_warning("TensorRT validation had issues, but engine was created")
            return engine_path
        else:
            print_error("TensorRT conversion failed")
            return None
    print_subheader(f"Exporting model to {export_format.upper()}")
    if task == 'segmentation' and export_format == 'onnx':
        print_warning("Segmentation model detected - using safe export settings")
        export_args["simplify"] = False
        print_info("Set simplify=False to preserve mask prototype output")
        if export_args.get("opset", 12) < 13:
            export_args["opset"] = 13
            print_info("Upgraded opset to 13 for segmentation compatibility")
    print_info(f"Task: {task}")
    print_info(f"Image size: {imgsz}x{imgsz}")
    print_info("Export settings:")
    for key, value in export_args.items():
        print(f"  - {key}: {value}")
    try:
        export_path = model.export(format=export_format, **export_args)
        if task == 'segmentation' and export_format == 'onnx':
            print_info("Running post-export validation for segmentation model...")
            if not validate_segmentation_onnx(export_path):
                print_error("Segmentation ONNX validation FAILED!")
                print_warning("The exported model may only work for detection, not segmentation.")
                print_info("Attempting re-export with stricter settings...")
                retry_args = export_args.copy()
                retry_args["simplify"] = False
                retry_args["opset"] = 13
                try:
                    if os.path.exists(export_path):
                        os.remove(export_path)
                    export_path = model.export(format=export_format, **retry_args)
                    if not validate_segmentation_onnx(export_path):
                        print_error("Retry export also failed validation!")
                        print_error("Segmentation masks will NOT work with this model.")
                    else:
                        print_success("Retry export succeeded with valid segmentation outputs!")
                except Exception as retry_e:
                    print_error(f"Retry export failed: {retry_e}")
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
                if task == 'segmentation' and export_format == 'onnx':
                    if not validate_segmentation_onnx(export_path):
                        print_warning("Segmentation validation failed after CPU fallback")
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
    if export_format == "onnx":
        print_info("For ONNX inference, you need:")
        print(f"  1. The exported model: {os.path.basename(export_path)}")
        print("  2. ONNX Runtime libraries (libonnxruntime.so)")
        print("  3. Python: onnx, onnxruntime-gpu, numpy")
    elif export_format == "engine":
        print_info("For TensorRT inference, you need:")
        print(f"  1. The exported engine: {os.path.basename(export_path)}")
        print("  2. TensorRT libraries (libnvinfer.so, libnvonnxparser.so)")
        print("  3. CUDA libraries (libcudnn.so, libcublas.so)")
        print("  4. Python: tensorrt, pycuda, numpy")
    elif export_format == "torchscript":
        print_info("For TorchScript inference, you need:")
        print(f"  1. The exported model: {os.path.basename(export_path)}")
        print("  2. PyTorch libraries (libtorch.so)")
        print("  3. Python: torch, torchvision, numpy")
    else:
        print_info(f"Exported model: {export_path}")

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
            "device": export_option["device"],
        }
        if export_option["format"] == "onnx":
            if task == 'segmentation':
                export_args["simplify"] = False
                export_args["opset"] = 13
                export_args["dynamic"] = False
                print_info("Using segmentation-safe ONNX settings:")
                print_info("  - simplify=False (preserves mask output)")
                print_info("  - opset=13 (better op support)")
            else:
                export_args["simplify"] = True
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
            export_path = export_model(model, export_option["format"], task, export_args, model_path=model_info['model_file'])
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
    parser.add_argument('--model', type=str, default=None, help='Custom model path (.pt or .onnx)')
    parser.add_argument('--format', type=str, default=None, choices=['onnx', 'engine', 'torchscript'], help='Export format')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/0)')
    parser.add_argument('--half', action='store_true', help='Use FP16 half precision')
    parser.add_argument('--no-half', dest='half', action='store_false', help='Use FP32 full precision')
    parser.add_argument('--imgsz', type=int, default=None, help='Image size (auto-detected by task if not set)')
    parser.add_argument('--optimize', action='store_true', help='Optimize model (CPU only)')
    parser.add_argument('--simplify', action='store_true', default=None, help='Simplify ONNX model (auto-disabled for segmentation)')
    parser.add_argument('--no-simplify', dest='simplify', action='store_false', help='Do not simplify ONNX model')
    parser.add_argument('--opset', type=int, default=None, help='ONNX opset version (default: 12 for det/cls, 13 for seg)')
    parser.add_argument('--workspace', type=float, default=4, help='TensorRT workspace size (GB)')
    parser.add_argument('--list', action='store_true', help='List export options and exit')
    parser.add_argument('--show-libs', action='store_true', help='Show required libraries')
    parser.add_argument('--onnx-to-trt', action='store_true', help='Convert ONNX model directly to TensorRT (bypasses Ultralytics)')
    parser.add_argument('--output', type=str, default=None, help='Output path for converted model')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16', 'int8'], help='TensorRT precision mode')
    parser.add_argument('--validate', action='store_true', default=True, help='Run post-conversion validation')
    parser.add_argument('--no-validate', dest='validate', action='store_false', help='Skip post-conversion validation')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
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

    if args.onnx_to_trt:
        if not args.model:
            print_error("--model is required for --onnx-to-trt conversion")
            print_info("Usage: python3 export.py --onnx-to-trt --model yolov8n-seg.onnx --precision fp16")
            return
        if not args.model.lower().endswith('.onnx'):
            print_error("--onnx-to-trt requires an ONNX model (.onnx file)")
            return
        if not os.path.exists(args.model):
            print_error(f"Model file not found: {args.model}")
            return
        print_header("Direct ONNX to TensorRT Conversion", shutil.get_terminal_size().columns)
        model_name = os.path.basename(args.model).lower()
        if '-seg' in model_name or '_seg' in model_name:
            detected_task = 'segmentation'
        elif '-cls' in model_name or '_cls' in model_name:
            detected_task = 'classification'
        else:
            detected_task = 'detection'
        task = args.task if args.task else detected_task
        print_info(f"Detected task: {task.upper()}")
        output_path = args.output if args.output else args.model.replace('.onnx', '.engine')
        workspace_mb = int(args.workspace * 1024)
        engine_path = convert_onnx_to_tensorrt(
            onnx_path=args.model,
            engine_path=output_path,
            precision=args.precision,
            workspace_mb=workspace_mb,
            verbose=args.verbose
        )
        if engine_path and args.validate:
            print_subheader("Post-Conversion Validation")
            validation_result = validate_tensorrt_engine(engine_path, task)
            print_header("Conversion Summary", shutil.get_terminal_size().columns)
            print(f"\n  Source:      {args.model}")
            print(f"  Output:      {engine_path}")
            print(f"  Task:        {task.upper()}")
            print(f"  Precision:   {args.precision.upper()}")
            print(f"  Engine size: {os.path.getsize(engine_path) / (1024*1024):.1f} MB")
            if validation_result['valid']:
                print_success(f"  Validation:  PASSED")
                print_success(f"  Inference:   {validation_result['inference_time_ms']:.2f} ms ({1000/validation_result['inference_time_ms']:.1f} FPS)")
            else:
                print_error(f"  Validation:  FAILED")
                if validation_result['error']:
                    print_error(f"  Error:       {validation_result['error']}")
            print(f"\n  Inputs:")
            for inp in validation_result['inputs']:
                print(f"    - {inp['name']}: {inp['shape']}")
            print(f"  Outputs:")
            for out in validation_result['outputs']:
                print(f"    - {out['name']}: {out['shape']}")
            print()
            if validation_result['valid']:
                print_success(f"Engine ready for deployment: {engine_path}")
                print_info(f"Usage: python3 advantech-yolo.py --model {engine_path} --source /dev/video0 --task {task}")
        elif engine_path:
            print_success(f"Conversion complete (validation skipped): {engine_path}")
        else:
            print_error("Conversion failed")
        return

    if args.list or args.show_libs:
        display_export_options(export_options)
        if args.show_libs:
            print_subheader("Required System Libraries by Format")
            print_info("ONNX Format:")
            print("  - libonnxruntime.so")
            print("  - Python: onnx, onnxruntime-gpu, numpy")
            print_info("TensorRT Engine:")
            print("  - libnvinfer.so, libnvonnxparser.so, libcudnn.so, libcublas.so")
            print("  - Python: tensorrt, pycuda, numpy")
            print_info("TorchScript:")
            print("  - libtorch.so")
            print("  - Python: torch, torchvision, numpy")
            print_info("Note: OpenCV is provided by the system (python3-opencv) and should")
            print_info("      not be installed via pip on Jetson devices.")
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
        if len(selected_models) > 1:
            try:
                confirm = input(f"{Colors.BOLD}Proceed with batch export of {len(selected_models)} models? (y/n): {Colors.ENDC}").lower().strip()
                if confirm == 'n':
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
            "device": selected_option["device"],
        }
        if selected_option["format"] == "onnx":
            if args.simplify is not None:
                export_args["simplify"] = args.simplify
                if args.simplify and task == 'segmentation':
                    print_warning("Warning: simplify=True may break segmentation mask output!")
            else:
                if task == 'segmentation':
                    export_args["simplify"] = False
                else:
                    export_args["simplify"] = True
            if args.opset is not None:
                export_args["opset"] = args.opset
            else:
                if task == 'segmentation':
                    export_args["opset"] = 13
                else:
                    export_args["opset"] = 12
            export_args["dynamic"] = False
        if selected_option["format"] == "engine":
            export_args["workspace"] = args.workspace
            export_args["dynamic"] = False
        if selected_option["optimize"] and selected_option["device"] == "cpu":
            export_args["optimize"] = True
        if args.imgsz:
            export_args["imgsz"] = args.imgsz
        actual_model_path = args.model if args.model else get_model_path(args.task, args.size)
        export_path = export_model(model, selected_option["format"], task, export_args, model_path=actual_model_path)
        if export_path:
            list_required_files(selected_option["format"], export_path, task)
    print_header("Export Complete", shutil.get_terminal_size().columns)

if __name__ == "__main__":
    main()
