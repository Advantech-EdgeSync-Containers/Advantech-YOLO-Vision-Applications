#!/usr/bin/env python3
# ==========================================================================
# YOLO Model Downloader for Advantech Devices
# ==========================================================================
# Version:      2.0.0
# Author:       Samir Singh <samir.singh@advantech.com> and Apoorv Saxena<apoorv.saxena@advantech.com>
# Created:      February 8, 2025
# Last Updated: Dec 03, 2025
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

__title__ = "Advantech YOLO Model Loader"
__author__ = "Advantech Co. Ltd"
__copyright__ = "Copyright (c) 2024-2025 Advantech Corporation. All Rights Reserved."
__license__ = "Proprietary - Advantech Corporation"
__version__ = "2.0.0"
__build_date__ = "2025-12-03"
__maintainer__ = "Samir Singh"

import os
import sys
import re
import time
import argparse
import subprocess
from pathlib import Path


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


def print_banner():
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║     █████╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗ ██████╗██╗  ██╗  ║
║    ██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔════╝██║  ██║  ║
║    ███████║██║  ██║╚██╗ ██╔╝███████║██╔██╗ ██║   ██║   █████╗  ██║     ███████║  ║
║    ██╔══██║██║  ██║ ╚████╔╝ ██╔══██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██╔══██║  ║
║    ██║  ██║██████╔╝  ╚██╔╝  ██║  ██║██║ ╚████║   ██║   ███████╗╚██████╗██║  ██║  ║
║    ╚═╝  ╚═╝╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝  ║
║                          YOLO Model Loader v{__version__}                              ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Author: {__author__:<18}  Build: {__build_date__:<14}                               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  {__copyright__}             ║
╚══════════════════════════════════════════════════════════════════════════════════╝"""
    print(banner)


def print_info(text):
    print(f"{Colors.CYAN}ℹ {text}{Colors.ENDC}")


def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def detect_device():
    """Detect Jetson/Advantech device with accurate hardware information."""
    device_info = {
        "model": "Unknown Device",
        "product": "Not Specified",
        "vendor": "Unknown",
        "version": "Not Specified",
        "os": "Unknown",
        "architecture": "Unknown",
        "compute_capability": "Unknown",
        "cuda_cores": 0,
        "memory_gb": 0,
        "jetpack_version": "Unknown",
        "l4t_version": "Unknown",
        "nvidia_info": "",
        "is_jetson": False,
        "is_advantech": False,
        "base_platform": "Unknown",
    }
    
    # Get OS info
    try:
        if os.path.exists('/etc/os-release'):
            with open('/etc/os-release', 'r') as f:
                for line in f:
                    if line.startswith('PRETTY_NAME='):
                        device_info["os"] = line.split('=')[1].strip().strip('"')
                        break
    except:
        pass
    
    # Read device tree model
    dt_model = None
    try:
        if os.path.exists('/proc/device-tree/model'):
            with open('/proc/device-tree/model', 'r') as f:
                dt_model = f.read().strip().replace('\x00', '')
                device_info["product"] = dt_model
    except:
        pass
    
    # Parse L4T/JetPack version and get tegra board type
    l4t_major = None
    l4t_revision = None
    tegra_board = None
    
    try:
        if os.path.exists('/etc/nv_tegra_release'):
            with open('/etc/nv_tegra_release', 'r') as f:
                release_info = f.read().strip()
                device_info["nvidia_info"] = release_info
                
                match = re.search(r'R(\d+)\s*\(release\),\s*REVISION:\s*([\d.]+)', release_info)
                if match:
                    l4t_major = int(match.group(1))
                    l4t_revision = match.group(2)
                    device_info["l4t_version"] = f"R{l4t_major}.{l4t_revision}"
                
                if "t234ref" in release_info:
                    tegra_board = "t234"
                elif "t194ref" in release_info:
                    tegra_board = "t194"
                elif "t186ref" in release_info:
                    tegra_board = "t186"
                elif "t210ref" in release_info:
                    tegra_board = "t210"
    except:
        pass
    
    # Try dpkg for L4T version
    if l4t_major is None:
        try:
            result = subprocess.run(
                ['dpkg-query', '-W', '-f=${Version}', 'nvidia-l4t-core'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                version_str = result.stdout.strip()
                match = re.match(r'(\d+)\.(\d+)\.(\d+)', version_str)
                if match:
                    l4t_major = int(match.group(1))
                    l4t_revision = f"{match.group(2)}.{match.group(3)}"
                    device_info["l4t_version"] = f"R{l4t_major}.{l4t_revision}"
        except:
            pass
    
    # Map L4T to JetPack version
    if l4t_major:
        jetpack_map = {
            (36, '4.0'): '6.1', (36, '3.0'): '6.0 GA', (36, '2.0'): '6.0 DP',
            (35, '6.0'): '5.1.4', (35, '5.0'): '5.1.3', (35, '4.1'): '5.1.2',
            (35, '3.1'): '5.1.1', (35, '2.1'): '5.1', (35, '1.0'): '5.0.2',
            (34, '1.1'): '5.0.1 DP', (34, '1.0'): '5.0 DP',
            (32, '7.4'): '4.6.4', (32, '7.3'): '4.6.3', (32, '7.2'): '4.6.2',
            (32, '7.1'): '4.6.1', (32, '6.1'): '4.6', (32, '5.2'): '4.5.1',
        }
        jp = jetpack_map.get((l4t_major, l4t_revision))
        if jp:
            device_info["jetpack_version"] = jp
        elif l4t_major >= 36:
            device_info["jetpack_version"] = "6.x"
        elif l4t_major >= 35:
            device_info["jetpack_version"] = "5.1.x"
        elif l4t_major >= 34:
            device_info["jetpack_version"] = "5.0.x"
        elif l4t_major >= 32:
            device_info["jetpack_version"] = "4.x"
    
    # Determine device model from device tree OR tegra board
    if dt_model and "Orin" in dt_model:
        device_info["is_jetson"] = True
        device_info["architecture"] = "Ampere"
        device_info["compute_capability"] = "8.7"
        device_info["base_platform"] = "Orin"
        if "AGX" in dt_model:
            device_info["model"] = "NVIDIA Jetson AGX Orin"
            device_info["cuda_cores"] = 2048
        elif "NX" in dt_model:
            device_info["model"] = "NVIDIA Jetson Orin NX"
            device_info["cuda_cores"] = 1024
        elif "Nano" in dt_model:
            device_info["model"] = "NVIDIA Jetson Orin Nano"
            device_info["cuda_cores"] = 1024
        else:
            device_info["model"] = "NVIDIA Jetson Orin"
            device_info["cuda_cores"] = 1024
    
    elif dt_model and "Xavier" in dt_model:
        device_info["is_jetson"] = True
        device_info["architecture"] = "Volta"
        device_info["compute_capability"] = "7.2"
        device_info["base_platform"] = "Xavier"
        if "AGX" in dt_model:
            device_info["model"] = "NVIDIA Jetson AGX Xavier"
            device_info["cuda_cores"] = 512
        elif "NX" in dt_model:
            device_info["model"] = "NVIDIA Jetson Xavier NX"
            device_info["cuda_cores"] = 384
        else:
            device_info["model"] = "NVIDIA Jetson Xavier"
            device_info["cuda_cores"] = 512
    
    elif dt_model and "TX2" in dt_model:
        device_info["is_jetson"] = True
        device_info["model"] = "NVIDIA Jetson TX2"
        device_info["architecture"] = "Pascal"
        device_info["compute_capability"] = "6.2"
        device_info["cuda_cores"] = 256
        device_info["base_platform"] = "TX2"
    
    elif dt_model and "Nano" in dt_model and "Orin" not in dt_model:
        device_info["is_jetson"] = True
        device_info["model"] = "NVIDIA Jetson Nano"
        device_info["architecture"] = "Maxwell"
        device_info["compute_capability"] = "5.3"
        device_info["cuda_cores"] = 128
        device_info["base_platform"] = "Nano"
    
    elif dt_model and "Thor" in dt_model:
        device_info["is_jetson"] = True
        device_info["model"] = "NVIDIA Thor"
        device_info["architecture"] = "Ampere"
        device_info["compute_capability"] = "8.7"
        device_info["cuda_cores"] = 2048
        device_info["base_platform"] = "Thor"
    
    elif tegra_board:
        device_info["is_jetson"] = True
        if tegra_board == "t234":
            device_info["model"] = "NVIDIA Jetson Orin"
            device_info["architecture"] = "Ampere"
            device_info["compute_capability"] = "8.7"
            device_info["cuda_cores"] = 1024
            device_info["base_platform"] = "Orin"
        elif tegra_board == "t194":
            device_info["model"] = "NVIDIA Jetson Xavier"
            device_info["architecture"] = "Volta"
            device_info["compute_capability"] = "7.2"
            device_info["cuda_cores"] = 384
            device_info["base_platform"] = "Xavier"
        elif tegra_board == "t186":
            device_info["model"] = "NVIDIA Jetson TX2"
            device_info["architecture"] = "Pascal"
            device_info["compute_capability"] = "6.2"
            device_info["cuda_cores"] = 256
            device_info["base_platform"] = "TX2"
        elif tegra_board == "t210":
            device_info["model"] = "NVIDIA Jetson Nano"
            device_info["architecture"] = "Maxwell"
            device_info["compute_capability"] = "5.3"
            device_info["cuda_cores"] = 128
            device_info["base_platform"] = "Nano"
    
    # Check for Advantech branding
    try:
        if os.path.exists('/sys/class/dmi/id/board_vendor'):
            with open('/sys/class/dmi/id/board_vendor', 'r') as f:
                vendor = f.read().strip()
                device_info["vendor"] = vendor
                if "Advantech" in vendor:
                    device_info["is_advantech"] = True
                    base = device_info["base_platform"]
                    device_info["model"] = f"Advantech {base}-based AIE"
        
        if os.path.exists('/sys/class/dmi/id/product_version'):
            with open('/sys/class/dmi/id/product_version', 'r') as f:
                device_info["version"] = f.read().strip() or "Not Specified"
    except:
        pass
    
    # Get GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            device_info["memory_gb"] = round(props.total_memory / (1024**3))
            cap = torch.cuda.get_device_capability(0)
            if device_info["compute_capability"] == "Unknown":
                device_info["compute_capability"] = f"{cap[0]}.{cap[1]}"
    except:
        pass
    
    if device_info["memory_gb"] == 0:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                mem_mb = int(result.stdout.strip().split('\n')[0])
                device_info["memory_gb"] = round(mem_mb / 1024)
        except:
            pass
    
    return device_info


def detect_libraries():
    """Detect installed libraries and their versions."""
    libraries = {}
    
    try:
        import ultralytics
        libraries["ultralytics"] = {"installed": True, "version": ultralytics.__version__, "notes": ""}
    except:
        libraries["ultralytics"] = {"installed": False, "version": "N/A", "notes": ""}
    
    try:
        import numpy as np
        libraries["numpy"] = {"installed": True, "version": np.__version__, "notes": f"{Colors.GREEN}Compatible{Colors.ENDC}"}
    except:
        libraries["numpy"] = {"installed": False, "version": "N/A", "notes": ""}
    
    try:
        import torch
        cuda_status = f"{Colors.GREEN}CUDA Available{Colors.ENDC}" if torch.cuda.is_available() else f"{Colors.YELLOW}CPU Only{Colors.ENDC}"
        libraries["torch"] = {"installed": True, "version": torch.__version__, "notes": cuda_status}
    except:
        libraries["torch"] = {"installed": False, "version": "N/A", "notes": ""}
    
    try:
        import torchvision
        libraries["torchvision"] = {"installed": True, "version": torchvision.__version__, "notes": ""}
    except:
        libraries["torchvision"] = {"installed": False, "version": "N/A", "notes": ""}
    
    try:
        import onnx
        libraries["onnx"] = {"installed": True, "version": onnx.__version__, "notes": ""}
    except:
        libraries["onnx"] = {"installed": False, "version": "N/A", "notes": ""}
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        notes = f"{Colors.GREEN}GPU Available{Colors.ENDC}" if 'CUDAExecutionProvider' in providers else f"{Colors.YELLOW}CPU Only{Colors.ENDC}"
        libraries["onnxruntime"] = {"installed": True, "version": ort.__version__, "notes": notes}
    except:
        libraries["onnxruntime"] = {"installed": False, "version": "N/A", "notes": ""}
    
    try:
        import tensorrt as trt
        libraries["tensorrt"] = {"installed": True, "version": trt.__version__, "notes": ""}
    except:
        libraries["tensorrt"] = {"installed": False, "version": "N/A", "notes": ""}
    
    try:
        import cv2
        libraries["cv2"] = {"installed": True, "version": cv2.__version__, "notes": ""}
    except:
        libraries["cv2"] = {"installed": False, "version": "N/A", "notes": ""}
    
    try:
        import PIL
        libraries["PIL"] = {"installed": True, "version": PIL.__version__, "notes": ""}
    except:
        libraries["PIL"] = {"installed": False, "version": "N/A", "notes": ""}
    
    return libraries


def display_device_info(device_info):
    """Display detected device information."""
    header = "Detected Advantech Device" if device_info["is_advantech"] else "Detected Device"
    print(f"{Colors.BOLD}{header}{Colors.ENDC}")
    
    print(f"Model: {device_info['model']}")
    print(f"Product: {device_info['product']}")
    print(f"Vendor: {device_info['vendor']}")
    print(f"Version: {device_info['version']}")
    print(f"OS: {device_info['os']}")
    print(f"GPU Architecture: {device_info['architecture']}")
    print(f"CUDA Cores: {device_info['cuda_cores']}")
    print(f"Compute Capability: {device_info['compute_capability']}")
    print(f"Memory: {device_info['memory_gb']} GB")
    
    if device_info.get("nvidia_info"):
        print(f"NVIDIA System Info: {device_info['nvidia_info']}")
    
    base = device_info.get("base_platform", "Unknown")
    prefix = "Advantech " if device_info['is_advantech'] else ""
    if base == "Orin":
        print_info(f"This {prefix}device is based on NVIDIA Orin - optimal for YOLOv8m/s/n models")
    elif base == "Xavier":
        print_info(f"This {prefix}device is based on NVIDIA Xavier - optimal for YOLOv8s/n models")
    elif base == "TX2":
        print_info(f"This {prefix}device is based on NVIDIA TX2 - optimal for YOLOv8n models")
    elif base == "Nano":
        print_info(f"This {prefix}device is based on NVIDIA Nano - optimal for YOLOv8n models")


def display_libraries(libraries):
    """Display detected libraries in a table."""
    print(f"{Colors.BOLD}Detected Libraries{Colors.ENDC}")
    
    # Calculate column widths
    col_widths = [11, 18, 24, 24]
    
    # Header
    headers = ["Library", "Status", "Version", "Notes"]
    header_line = f"{Colors.BOLD}"
    for i, h in enumerate(headers):
        header_line += f" {h.ljust(col_widths[i])} |"
    header_line = header_line.rstrip('|') + f"{Colors.ENDC}"
    print(header_line)
    
    # Separator
    print("-" * 87)
    
    # Rows
    for lib, info in libraries.items():
        status = f"{Colors.GREEN}Installed{Colors.ENDC}" if info["installed"] else f"{Colors.RED}Not Installed{Colors.ENDC}"
        
        # Calculate padding accounting for ANSI codes
        lib_pad = col_widths[0] - len(lib)
        status_clean = re.sub(r'\033\[[0-9;]*m', '', status)
        status_pad = col_widths[1] - len(status_clean)
        version_pad = col_widths[2] - len(info["version"])
        notes_clean = re.sub(r'\033\[[0-9;]*m', '', info["notes"])
        notes_pad = col_widths[3] - len(notes_clean)
        
        print(f" {lib}{' ' * lib_pad} | {status}{' ' * status_pad} | {info['version']}{' ' * version_pad} | {info['notes']}{' ' * notes_pad} ")


def get_model_options(device_info):
    """Get model options based on device capabilities."""
    memory = device_info.get("memory_gb", 4)
    model_name = device_info.get("model", "Unknown Device")
    
    # Determine available sizes and recommendations based on memory
    if memory >= 16:
        sizes = ['n', 's', 'm', 'l']
        recommended = ['n', 's']
    elif memory >= 8:
        sizes = ['n', 's', 'm']
        recommended = ['n', 's']
    elif memory >= 4:
        sizes = ['n', 's']
        recommended = ['n']
    else:
        sizes = ['n']
        recommended = ['n']
    
    options = []
    option_id = 1
    tasks = ['detection', 'segmentation', 'classification']
    
    # Group by size first, then by task (n-det, n-seg, n-cls, s-det, s-seg, s-cls, ...)
    for size in sizes:
        for task in tasks:
            if task == 'detection':
                model_file = f"yolov8{size}.pt"
            elif task == 'segmentation':
                model_file = f"yolov8{size}-seg.pt"
            else:
                model_file = f"yolov8{size}-cls.pt"
            
            is_recommended = size in recommended
            desc = f"Recommended {task} model for {model_name}" if is_recommended else f"Alternative {task} model for {model_name}"
            
            options.append({
                "id": option_id,
                "model": model_file,
                "task": task,
                "size": size,
                "recommended": is_recommended,
                "description": desc
            })
            option_id += 1
    
    return options


def display_model_options(options):
    """Display available model options."""
    print(f"{Colors.BOLD}YOLOv8 Models for Your Device{Colors.ENDC}")
    
    for opt in options:
        if opt["recommended"]:
            print(f"{Colors.GREEN}[{opt['id']}] YOLOv8{opt['size']} {opt['task'].capitalize()} (RECOMMENDED){Colors.ENDC}")
        else:
            print(f"[{opt['id']}] YOLOv8{opt['size']} {opt['task'].capitalize()}")
        
        print(f"    Model: {opt['model']}")
        print(f"    Task: {opt['task']}")
        print(f"    Size: {opt['size']}")
        print(f"    {opt['description']}")


def load_model(model_path):
    """Load a YOLO model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print_error("ultralytics not installed. Run: pip install ultralytics")
        return None
    
    print(f"\n{Colors.CYAN}Loading model: {model_path}...{Colors.ENDC}")
    
    if not os.path.exists(model_path):
        print_info(f"Model not found locally. Downloading...")
    
    start_time = time.time()
    try:
        model = YOLO(model_path)
        load_time = time.time() - start_time
        print_success(f"Model loaded successfully in {load_time:.2f}s")
        
        task = getattr(model, 'task', 'detection')
        names = getattr(model, 'names', {})
        
        print(f"  Task: {task.upper()}")
        print(f"  Classes: {len(names)}")
        
        try:
            params = sum(p.numel() for p in model.model.parameters())
            print(f"  Parameters: {params / 1e6:.2f}M")
        except:
            pass
        
        return model
        
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        return None


def interactive_mode(device_info, libraries):
    """Run interactive model selection."""
    options = get_model_options(device_info)
    display_model_options(options)
    
    max_id = len(options)
    
    try:
        print(f"\nEnter option number (1-{max_id}): ", end="")
        choice = input().strip()
        
        if not choice.isdigit():
            print_error("Invalid input. Please enter a number.")
            return None
        
        choice = int(choice)
        if choice < 1 or choice > max_id:
            print_error(f"Invalid option. Please enter a number between 1 and {max_id}.")
            return None
        
        selected = options[choice - 1]
        return load_model(selected["model"])
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled.")
        return None
    except EOFError:
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Advantech YOLOv8 Model Loader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                 # Interactive mode
  %(prog)s --model yolov8n.pt              # Load specific model
  %(prog)s --task detection --size s       # Load detection model size s
  %(prog)s --info-only                     # Show device info only
        """
    )
    
    parser.add_argument('--model', '-m', type=str, help='Model path to load')
    parser.add_argument('--task', '-t', type=str, 
                       choices=['detection', 'segmentation', 'classification'],
                       help='Task type')
    parser.add_argument('--size', '-s', type=str, 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size')
    parser.add_argument('--dir', '-d', type=str, default='.', help='Model directory')
    parser.add_argument('--info-only', action='store_true', help='Show device info only')
    parser.add_argument('--no-interactive', action='store_true', help='Disable interactive mode')
    
    args = parser.parse_args()
    
    os.system('clear' if os.name != 'nt' else 'cls')
    print_banner()
    
    device_info = detect_device()
    libraries = detect_libraries()
    
    display_device_info(device_info)
    display_libraries(libraries)
    
    if args.info_only:
        return 0
    
    if not libraries.get("ultralytics", {}).get("installed", False):
        print_error("ultralytics is required. Install with: pip install ultralytics")
        return 1
    
    if args.model:
        model_path = args.model
        if args.dir != '.':
            model_path = os.path.join(args.dir, model_path)
        load_model(model_path)
    elif args.task and args.size:
        if args.task == 'detection':
            model_path = f"yolov8{args.size}.pt"
        elif args.task == 'segmentation':
            model_path = f"yolov8{args.size}-seg.pt"
        else:
            model_path = f"yolov8{args.size}-cls.pt"
        
        if args.dir != '.':
            model_path = os.path.join(args.dir, model_path)
        load_model(model_path)
    elif not args.no_interactive:
        interactive_mode(device_info, libraries)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
