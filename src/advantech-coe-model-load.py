#!/usr/bin/env python3

# ==========================================================================
# YOLO Model Downloader for Advantech Devices
# ==========================================================================
# Version:      2.6.0
# Author:       Samir Singh <samir.singh@advantech.com> and Apoorv Saxena<apoorv.saxena@advantech.com>
# Created:      February 8, 2025
# Last Updated: May 15, 2025
# 
# Description:
#   This utility detects Advantech device capabilities and provides
#   optimized YOLO model recommendations for detection, segmentation,
#   classification tasks. It automatically downloads
#   models and provides usage instructions based on device specifications.
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
import subprocess
import time
import platform
import glob
import shutil
from collections import OrderedDict

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

def detect_advantech_device():
    device_info = {
        "model_type": "Unknown Advantech Device",
        "product_name": "Unknown",
        "memory_gb": 0,
        "cuda_cores": 0,
        "architecture": "Unknown",
        "compute_capability": "Unknown",
        "board_details": {}
    }
    try:
        if os.path.exists('/sys/class/dmi/id/board_vendor'):
            with open('/sys/class/dmi/id/board_vendor', 'r') as f:
                board_vendor = f.read().strip()
                device_info["board_details"]["vendor"] = board_vendor
        if os.path.exists('/sys/class/dmi/id/board_name'):
            with open('/sys/class/dmi/id/board_name', 'r') as f:
                board_name = f.read().strip()
                device_info["board_details"]["name"] = board_name
                device_info["product_name"] = board_name
        if os.path.exists('/sys/class/dmi/id/product_name'):
            with open('/sys/class/dmi/id/product_name', 'r') as f:
                product_name = f.read().strip()
                device_info["board_details"]["product_name"] = product_name
                if not device_info["product_name"] or device_info["product_name"] == "Unknown":
                    device_info["product_name"] = product_name
        if os.path.exists('/sys/class/dmi/id/product_version'):
            with open('/sys/class/dmi/id/product_version', 'r') as f:
                product_version = f.read().strip()
                device_info["board_details"]["product_version"] = product_version
        is_advantech = False
        if device_info["board_details"].get("vendor", "").lower().find("advantech") >= 0:
            is_advantech = True
        elif device_info["board_details"].get("name", "").lower().find("advantech") >= 0:
            is_advantech = True
        elif device_info["board_details"].get("product_name", "").lower().find("advantech") >= 0:
            is_advantech = True
        if is_advantech:
            device_info["model_type"] = "Advantech Industrial Computer"
            model_identifiers = [
                device_info["board_details"].get("name", ""),
                device_info["board_details"].get("product_name", ""),
                device_info["product_name"]
            ]
            for identifier in model_identifiers:
                if identifier and identifier != "Unknown":
                    device_info["model_type"] = f"Advantech {identifier}"
                    break
    except Exception as e:
        print_warning(f"Error reading board vendor information: {e}")
    if not is_advantech:
        jetpack_info = "Unknown"
        if os.path.exists('/etc/nv_tegra_release'):
            try:
                with open('/etc/nv_tegra_release', 'r') as f:
                    jetpack_info = f.read().strip()
                    device_info["jetpack_info"] = jetpack_info
                    if "t186ref" in jetpack_info:
                        device_info["model_type"] = "Advantech Xavier-based AIE"
                        device_info["memory_gb"] = 16
                        device_info["cuda_cores"] = 512
                        device_info["architecture"] = "Volta"
                        device_info["compute_capability"] = "7.2"
                    elif "t194ref" in jetpack_info:
                        device_info["model_type"] = "Advantech Xavier NX-based AIE"
                        device_info["memory_gb"] = 8
                        device_info["cuda_cores"] = 384
                        device_info["architecture"] = "Volta"
                        device_info["compute_capability"] = "7.2"
                    elif "t234ref" in jetpack_info:
                        device_info["architecture"] = "Ampere"
                        device_info["compute_capability"] = "8.7"
                        total_memory_kb = 0
                        try:
                            with open('/proc/meminfo', 'r') as f:
                                for line in f:
                                    if 'MemTotal' in line:
                                        total_memory_kb = int(line.split()[1])
                                        break
                            total_memory_gb = total_memory_kb / 1024 / 1024
                            device_info["memory_gb"] = round(total_memory_gb)
                            if total_memory_gb > 25:
                                device_info["model_type"] = "Advantech Orin AGX-based AIE"
                                device_info["cuda_cores"] = 2048
                            elif total_memory_gb > 14:
                                device_info["model_type"] = "Advantech Orin NX 16GB-based AIE"
                                device_info["cuda_cores"] = 1024
                            elif total_memory_gb > 7:
                                device_info["model_type"] = "Advantech Orin NX 8GB-based AIE"
                                device_info["cuda_cores"] = 1024
                            else:
                                device_info["model_type"] = "Advantech Orin Nano-based AIE"
                                device_info["cuda_cores"] = 512
                        except:
                            device_info["model_type"] = "Advantech Orin-based AIE"
                            device_info["cuda_cores"] = 1024
            except:
                pass
    try:
        if os.path.exists('/etc/os-release'):
            with open('/etc/os-release', 'r') as f:
                os_release = f.read()
                for line in os_release.split('\n'):
                    if line.startswith('PRETTY_NAME='):
                        os_name = line.split('=')[1].strip('"\'')
                        device_info["board_details"]["os"] = os_name
                        break
    except:
        pass
    try:
        cpu_info = run_command("cat /proc/cpuinfo")
        cpu_model = ""
        core_count = 0
        for line in cpu_info.split('\n'):
            if "model name" in line:
                cpu_model = line.split(': ')[1]
                core_count += 1
        if cpu_model:
            device_info["board_details"]["cpu_model"] = cpu_model
            device_info["board_details"]["cpu_cores"] = core_count
    except:
        pass
    try:
        cuda_output = run_command("nvcc --version")
        if "release" in cuda_output:
            cuda_version = cuda_output.split("release")[1].split(",")[0].strip()
            device_info["cuda_version"] = cuda_version
    except:
        device_info["cuda_version"] = "Not found"
    try:
        mem_info = run_command("free -m")
        mem_lines = mem_info.split('\n')
        if len(mem_lines) > 1:
            mem_values = mem_lines[1].split()
            if len(mem_values) >= 3:
                device_info["memory"] = {
                    "total": int(mem_values[1]),
                    "free": int(mem_values[3])
                }
    except:
        pass
    return device_info

def detect_available_libraries():
    libraries = {
        "ultralytics": {"installed": False, "version": None},
        "numpy": {"installed": False, "version": None, "has_bitgenerator": False},
        "torch": {"installed": False, "version": None, "cuda": False},
        "torchvision": {"installed": False, "version": None},
        "onnx": {"installed": False, "version": None},
        "onnxruntime": {"installed": False, "version": None},
        "tensorrt": {"installed": False, "version": None},
        "cv2": {"installed": False, "version": None},
        "PIL": {"installed": False, "version": None},
    }
    for lib_name in libraries:
        try:
            if lib_name == "numpy":
                import numpy as np
                libraries[lib_name]["installed"] = True
                libraries[lib_name]["version"] = np.__version__
                libraries[lib_name]["has_bitgenerator"] = hasattr(np.random, 'BitGenerator')
            elif lib_name == "torch":
                import torch
                libraries[lib_name]["installed"] = True
                libraries[lib_name]["version"] = torch.__version__
                libraries[lib_name]["cuda"] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    libraries[lib_name]["device_name"] = torch.cuda.get_device_name(0)
            elif lib_name == "torchvision":
                import torchvision
                libraries[lib_name]["installed"] = True
                libraries[lib_name]["version"] = torchvision.__version__
            elif lib_name == "onnx":
                import onnx
                libraries[lib_name]["installed"] = True
                libraries[lib_name]["version"] = onnx.__version__
            elif lib_name == "onnxruntime":
                import onnxruntime
                libraries[lib_name]["installed"] = True
                libraries[lib_name]["version"] = onnxruntime.__version__
            elif lib_name == "tensorrt":
                import tensorrt
                libraries[lib_name]["installed"] = True
                libraries[lib_name]["version"] = tensorrt.__version__
            elif lib_name == "cv2":
                import cv2
                libraries[lib_name]["installed"] = True
                libraries[lib_name]["version"] = cv2.__version__
            elif lib_name == "PIL":
                from PIL import Image
                import PIL
                libraries[lib_name]["installed"] = True
                libraries[lib_name]["version"] = PIL.__version__
            elif lib_name == "ultralytics":
                import ultralytics
                libraries[lib_name]["installed"] = True
                libraries[lib_name]["version"] = ultralytics.__version__
        except:
            pass
    return libraries

def get_recommended_models(device_info):
    recommendations = {
        "detection": {"recommended": "", "max_size": "", "options": []},
        "segmentation": {"recommended": "", "max_size": "", "options": []},
        "classification": {"recommended": "", "max_size": "", "options": []},
    }
    if "Orin" in device_info["model_type"]:
        if "AGX" in device_info["model_type"]:
            for task in recommendations:
                recommendations[task]["recommended"] = "m"
                recommendations[task]["max_size"] = "l"
                recommendations[task]["options"] = ["n", "s", "m", "l"]
        elif "NX" in device_info["model_type"]:
            for task in recommendations:
                recommendations[task]["recommended"] = "s"
                recommendations[task]["max_size"] = "m"
                recommendations[task]["options"] = ["n", "s", "m"]
        else:
            for task in recommendations:
                recommendations[task]["recommended"] = "s"
                recommendations[task]["max_size"] = "s"
                recommendations[task]["options"] = ["n", "s"]
    elif "Xavier" in device_info["model_type"]:
        if "AGX" in device_info["model_type"]:
            for task in recommendations:
                recommendations[task]["recommended"] = "s"
                recommendations[task]["max_size"] = "m"
                recommendations[task]["options"] = ["n", "s", "m"]
        else:
            for task in recommendations:
                recommendations[task]["recommended"] = "n"
                recommendations[task]["max_size"] = "s"
                recommendations[task]["options"] = ["n", "s"]
    else:
        for task in recommendations:
            recommendations[task]["recommended"] = "n"
            recommendations[task]["max_size"] = "n"
            recommendations[task]["options"] = ["n"]
    model_options = []
    option_id = 1
    for task, info in recommendations.items():
        if info["recommended"]:
            size = info["recommended"]
            if task == "detection":
                model_name = f"yolov8{size}.pt"
            elif task == "segmentation":
                model_name = f"yolov8{size}-seg.pt"
            elif task == "classification":
                model_name = f"yolov8{size}-cls.pt"
            else:
                continue
            option = {
                "id": option_id,
                "name": f"YOLOv8{size} {task.capitalize()}",
                "model_name": model_name,
                "task": task,
                "size": size,
                "recommended": True,
                "description": f"Recommended {task} model for {device_info['model_type']}"
            }
            model_options.append(option)
            option_id += 1
    for task, info in recommendations.items():
        for size in info["options"]:
            if size == info["recommended"]:
                continue
            if task == "detection":
                model_name = f"yolov8{size}.pt"
            elif task == "segmentation":
                model_name = f"yolov8{size}-seg.pt"
            elif task == "classification":
                model_name = f"yolov8{size}-cls.pt"
            else:
                continue
            option = {
                "id": option_id,
                "name": f"YOLOv8{size} {task.capitalize()}",
                "model_name": model_name,
                "task": task,
                "size": size,
                "recommended": False,
                "description": f"Alternative {task} model for {device_info['model_type']}"
            }
            model_options.append(option)
            option_id += 1
    return model_options

def display_device_info(device_info):
    print_subheader("Detected Advantech Device")
    print(f"Model: {Colors.BOLD}{device_info['model_type']}{Colors.ENDC}")
    if device_info.get("product_name") and device_info['product_name'] != "Unknown":
        print(f"Product: {device_info['product_name']}")
    if device_info.get("board_details"):
        board_details = device_info["board_details"]
        if board_details.get("vendor"):
            print(f"Vendor: {board_details['vendor']}")
        if board_details.get("name") and board_details['name'] != device_info.get('product_name', ''):
            print(f"Board: {board_details['name']}")
        if board_details.get("product_version"):
            print(f"Version: {board_details['product_version']}")
        if board_details.get("os"):
            print(f"OS: {board_details['os']}")
        if board_details.get("cpu_model"):
            print(f"CPU: {board_details['cpu_model']} ({board_details.get('cpu_cores', 'Unknown')} cores)")
    if device_info.get("architecture") and device_info["architecture"] != "Unknown":
        print(f"GPU Architecture: {device_info['architecture']}")
        print(f"CUDA Cores: {device_info['cuda_cores']}")
        print(f"Compute Capability: {device_info['compute_capability']}")
    if device_info.get("memory") and device_info["memory"].get("total"):
        print(f"Memory: {device_info['memory']['total']} MB total, {device_info['memory']['free']} MB free")
    elif device_info.get("memory_gb"):
        print(f"Memory: {device_info['memory_gb']} GB")
    if device_info.get("cuda_version"):
        print(f"CUDA Version: {device_info['cuda_version']}")
    if device_info.get("jetpack_info"):
        print(f"NVIDIA System Info: {device_info['jetpack_info']}")
    if "Orin" in device_info["model_type"]:
        print_info("This Advantech device is based on NVIDIA Orin - optimal for YOLOv8m/s models")
    elif "Xavier" in device_info["model_type"]:
        print_info("This Advantech device is based on NVIDIA Xavier - optimal for YOLOv8s/n models")
    else:
        if device_info.get("cuda_cores", 0) > 0:
            print_info("This Advantech device has CUDA capabilities - YOLOv8n/s models recommended")
        else:
            print_info("Limited or no CUDA capabilities detected - YOLOv8n model recommended")

def display_library_info(libraries):
    print_subheader("Detected Libraries")
    headers = ["Library", "Status", "Version", "Notes"]
    rows = []
    for lib_name, info in libraries.items():
        if info["installed"]:
            status = f"{Colors.GREEN}Installed{Colors.ENDC}"
            version = info["version"]
            notes = ""
            if lib_name == "numpy":
                notes = f"{Colors.GREEN}Compatible{Colors.ENDC}" if info["has_bitgenerator"] else f"{Colors.RED}Missing BitGenerator{Colors.ENDC}"
            elif lib_name == "torch":
                notes = f"{Colors.GREEN}CUDA Available{Colors.ENDC}" if info.get("cuda", False) else f"{Colors.YELLOW}CPU Only{Colors.ENDC}"
        else:
            status = f"{Colors.RED}Not Installed{Colors.ENDC}"
            version = "N/A"
            notes = ""
        rows.append([lib_name, status, version, notes])
    print(format_table(headers, rows))

def display_model_options(model_options):
    print_subheader("YOLOv8 Models for Your Device")
    for option in model_options:
        if option["recommended"]:
            print(f"{Colors.BOLD}{Colors.GREEN}[{option['id']}] {option['name']} (RECOMMENDED){Colors.ENDC}")
        else:
            print(f"{Colors.BOLD}[{option['id']}] {option['name']}{Colors.ENDC}")
        print(f"    Model: {option['model_name']}")
        print(f"    Task: {option['task']}")
        print(f"    Size: {option['size']}")
        print(f"    {option['description']}")
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

def download_model(model_name):
    print_progress(f"Downloading {model_name}...")
    try:
        import ultralytics
        from ultralytics import YOLO
        if os.path.exists(model_name):
            print_info(f"Model {model_name} already exists locally")
            try:
                model = YOLO(model_name)
                print_success(f"Successfully loaded existing model: {model_name}")
                print_info(f"Model location: {os.path.abspath(model_name)}")
                return model
            except Exception as e:
                print_warning(f"Found existing model file but it seems invalid: {e}")
                print_info("Will attempt to download a fresh copy")
        start_time = time.time()
        model = YOLO(model_name)
        end_time = time.time()
        print_success(f"Model {model_name} downloaded successfully in {end_time - start_time:.2f} seconds")
        print_info(f"Model saved to: {os.path.abspath(model_name)}")
        return model
    except Exception as e:
        print_error(f"Failed to download model: {e}")
        if "No such file or directory" in str(e):
            print_info("Hint: Check if the model name is correct:")
            print("  - Detection models: yolov8n.pt, yolov8s.pt, etc.")
            print("  - Segmentation models: yolov8n-seg.pt, yolov8s-seg.pt, etc.")
            print("  - Classification models: yolov8n-cls.pt, yolov8s-cls.pt, etc.")
        return None

def list_dependencies(model_name, download_dir=None):
    print_subheader(f"Dependencies for {model_name}")
    dependencies = {
        "ultralytics": {"required": True, "for": "Base package for YOLOv8"},
        "torch": {"required": True, "for": "Model inference"},
        "torchvision": {"required": True, "for": "Image processing"},
        "numpy": {"required": True, "for": "Numerical operations"},
        "opencv-python": {"required": True, "for": "Image processing"},
        "onnx": {"required": False, "for": "ONNX export/import"},
        "onnxruntime": {"required": False, "for": "ONNX inference"},
        "tensorrt": {"required": False, "for": "TensorRT inference"},
    }
    if "seg" in model_name:
        dependencies["matplotlib"] = {"required": True, "for": "Segmentation visualization"}
    headers = ["Package", "Status", "Required For"]
    rows = []
    for pkg, info in dependencies.items():
        try:
            pkg_installed = pkg in sys.modules or pkg.replace("-", "_") in sys.modules
            if not pkg_installed:
                import importlib
                importlib.import_module(pkg.replace("-", "_"))
                pkg_installed = True
        except:
            pkg_installed = False
        status = f"{Colors.GREEN}Installed{Colors.ENDC}" if pkg_installed else f"{Colors.RED}Not Installed{Colors.ENDC}"
        required = "Required" if info["required"] else "Optional"
        rows.append([pkg, status, f"{required} - {info['for']}"])
    print(format_table(headers, rows))
    missing_required = [pkg for pkg, row in zip(dependencies.keys(), rows) if "Not Installed" in row[1] and dependencies[pkg]["required"]]
    if missing_required:
        print_warning("Missing required dependencies. Install with:")
        print(f"pip install {' '.join(missing_required)}")
    print_subheader("Next Steps")
    if "onnx" in model_name:
        print_info("To use this ONNX model:")
        print("1. Install required dependencies")
        print("2. Load with: `onnx_model = onnx.load('model.onnx')`")
        print("3. Inference with ONNXRuntime")
    elif "engine" in model_name:
        print_info("To use this TensorRT engine:")
        print("1. Ensure TensorRT is properly installed")
        print("2. Load the engine with TensorRT runtime")
    else:
        print_info("To use this PyTorch model:")
        print("1. Import the YOLO class: `from ultralytics import YOLO`")
        print(f"2. Load the model: `model = YOLO('{model_name}')`")
        print("3. Run inference: `results = model('image.jpg')`")
        print("")
        print_info("To convert to optimized formats:")
        print("1. Export to ONNX: `model.export(format='onnx')`")
        print("2. Export to TensorRT: `model.export(format='engine')`")

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Model Downloader for Advantech Devices')
    parser.add_argument('--list', action='store_true', help='List recommended models and exit')
    parser.add_argument('--model', type=str, help='Specify model to download (e.g., yolov8n.pt)')
    parser.add_argument('--task', type=str, choices=['detection', 'segmentation', 'classification'], 
                      help='Task type to download model for')
    parser.add_argument('--size', type=str, choices=['n', 's', 'm', 'l', 'x'], 
                      help='Model size to download')
    parser.add_argument('--dir', type=str, default='.', help='Directory to save the model')
    args = parser.parse_args()
    os.system('clear')
    print_header("YOLOv8 Model Downloader for Advantech Devices", shutil.get_terminal_size().columns)
    print_progress("Detecting Advantech device...")
    device_info = detect_advantech_device()
    print_progress("Checking installed libraries...")
    libraries = detect_available_libraries()
    model_options = get_recommended_models(device_info)
    display_device_info(device_info)
    display_library_info(libraries)
    if args.dir and args.dir != '.':
        if not os.path.exists(args.dir):
            os.makedirs(args.dir)
        os.chdir(args.dir)
        print_info(f"Changed working directory to {args.dir}")
    if args.model:
        model = download_model(args.model)
        if model:
            list_dependencies(args.model)
        return
    if args.task and args.size:
        if args.task == 'detection':
            model_name = f"yolov8{args.size}.pt"
        elif args.task == 'segmentation':
            model_name = f"yolov8{args.size}-seg.pt"
        elif args.task == 'classification':
            model_name = f"yolov8{args.size}-cls.pt"
        model = download_model(model_name)
        if model:
            list_dependencies(model_name)
        return
    display_model_options(model_options)
    if args.list:
        return
    try:
        choice = input(f"\n{Colors.BOLD}Enter option number (1-{len(model_options)}): {Colors.ENDC}")
        choice = int(choice)
        selected_option = None
        for option in model_options:
            if option["id"] == choice:
                selected_option = option
                break
        if selected_option:
            model = download_model(selected_option["model_name"])
            if model:
                list_dependencies(selected_option["model_name"])
        else:
            print_error(f"Invalid option: {choice}")
    except KeyboardInterrupt:
        print("\nExiting...")
    except ValueError:
        print_error("Please enter a valid number")
    print_header("Download Complete", shutil.get_terminal_size().columns)

if __name__ == "__main__":
    main()
