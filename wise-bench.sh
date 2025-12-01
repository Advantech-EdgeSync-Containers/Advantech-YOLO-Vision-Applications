#!/bin/bash
# Advantech YOLO Vision Applications - Hardware Diagnostic & Benchmarking Tool
# Version: 2.0.0
# Author: Samir Singh <samir.singh@advantech.com>
# Copyright (c) 2024-2025 Advantech Corporation. All rights reserved.
set -uo pipefail

LOG_FILE="/advantech/diagnostics/wise-bench.log"
mkdir -p "$(dirname "$LOG_FILE")"
{ echo "==========================================================="; echo ">>> Diagnostic Run: $(date '+%Y-%m-%d %H:%M:%S')"; echo "==========================================================="; } >> "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

GREEN='\033[0;32m' RED='\033[0;31m' YELLOW='\033[0;33m'
BLUE='\033[0;34m' CYAN='\033[0;36m' PURPLE='\033[0;35m'
BOLD='\033[1m' NC='\033[0m'

clear
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════════════════════════════╗"
echo "║     █████╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗ ██████╗██╗  ██╗          ║"
echo "║    ██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔════╝██║  ██║          ║"
echo "║    ███████║██║  ██║╚██╗ ██╔╝███████║██╔██╗ ██║   ██║   █████╗  ██║     ███████║          ║"
echo "║    ██╔══██║██║  ██║ ╚████╔╝ ██╔══██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██╔══██║          ║"
echo "║    ██║  ██║██████╔╝  ╚██╔╝  ██║  ██║██║ ╚████║   ██║   ███████╗╚██████╗██║  ██║          ║"
echo "║    ╚═╝  ╚═╝╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝          ║"
echo "║                      YOLO Vision Hardware Diagnostics Tool                               ║"
echo "║                              Center of Excellence                                        ║"
echo "╚══════════════════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "${YELLOW}${BOLD}▶ Starting hardware acceleration diagnostics...${NC}"
sleep 3

print_header() { echo -e "\n${CYAN}+--- $1 $(printf '%*s' $((50 - ${#1})) | tr ' ' '-')+${NC}"; }
print_row() { printf "| %-28s | %-25s |\n" "$1" "$2"; }
print_line() { echo "+------------------------------+---------------------------+"; }
print_ok() { echo -e "${GREEN}✓${NC} $1"; }
print_warn() { echo -e "${YELLOW}⚠${NC} $1"; }
spinner() {
    local pid=$1 spin='|/-\'
    while kill -0 $pid 2>/dev/null; do
        for c in $(echo $spin | fold -w1); do printf "\b$c"; sleep 0.1; done
    done
    printf "\b"
}

print_header "SYSTEM INFORMATION"
print_line
KERNEL=$(uname -r)
ARCH=$(uname -m)
HOSTNAME=$(hostname)
OS=$(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d'"' -f2 || echo "Unknown")
MEM_TOTAL=$(free -h | awk '/^Mem:/{print $2}')
MEM_USED=$(free -h | awk '/^Mem:/{print $3}')
CPU_MODEL=$(lscpu 2>/dev/null | grep "Model name" | cut -d':' -f2 | sed 's/^[ \t]*//' | head -1 || echo "Unknown")
CPU_CORES=$(nproc --all)
print_row "Hostname" "$HOSTNAME"
print_row "OS" "$OS"
print_row "Kernel" "$KERNEL"
print_row "Architecture" "$ARCH"
print_row "CPU" "$CPU_MODEL"
print_row "CPU Cores" "$CPU_CORES"
print_row "Memory" "$MEM_USED / $MEM_TOTAL"
print_row "Date" "$(date '+%Y-%m-%d %H:%M:%S')"
print_line

print_header "NVIDIA DEVICES"
print_line
printf "| %-20s | %-15s | %-12s |\n" "Device" "Type" "Status"
echo "+----------------------+-----------------+--------------+"
DEVICE_COUNT=0
for dev in /dev/nvhost-* /dev/nvidia*; do
    [[ -e "$dev" ]] || continue
    name=$(basename "$dev")
    type=$(echo "$name" | sed 's/nvhost-//' | sed 's/nvidia/gpu/')
    printf "| %-20s | %-15s | %-12s |\n" "$name" "$type" "Available"
    ((DEVICE_COUNT++))
done
[[ $DEVICE_COUNT -eq 0 ]] && printf "| %-52s |\n" "No NVIDIA devices found"
echo "+----------------------+-----------------+--------------+"
print_ok "Found $DEVICE_COUNT NVIDIA devices"

print_header "CUDA INFORMATION"
echo -e "${YELLOW}"
echo "       ██████╗██╗   ██╗██████╗  █████╗ "
echo "      ██╔════╝██║   ██║██╔══██╗██╔══██╗"
echo "      ██║     ██║   ██║██║  ██║███████║"
echo "      ██║     ██║   ██║██║  ██║██╔══██║"
echo "      ╚██████╗╚██████╔╝██████╔╝██║  ██║"
echo "       ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝"
echo -e "${NC}"
print_line
CUDA_OK=0
if [[ -f "/usr/local/cuda/bin/nvcc" ]]; then
    CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
    CUDA_PATH="/usr/local/cuda"
    print_row "CUDA Version" "$CUDA_VERSION"
    print_row "CUDA Path" "$CUDA_PATH"
    print_row "Status" "✓ Available"
    CUDA_OK=1
else
    NVCC_PATH=$(find /usr -name nvcc 2>/dev/null | head -1)
    if [[ -n "$NVCC_PATH" ]]; then
        CUDA_VERSION=$("$NVCC_PATH" --version | grep "release" | awk '{print $5}' | tr -d ',')
        print_row "CUDA Version" "$CUDA_VERSION"
        print_row "NVCC Path" "$NVCC_PATH"
        print_row "Status" "✓ Available"
        CUDA_OK=1
    else
        print_row "Status" "⚠ Not detected"
    fi
fi
print_line

print_header "OPENCV CUDA TEST"
print_line
OPENCV_INFO=$(python3 -c "
try:
    import cv2
    print(cv2.__version__)
    has_cuda = hasattr(cv2, 'cuda')
    devices = cv2.cuda.getCudaEnabledDeviceCount() if has_cuda else 0
    print(has_cuda)
    print(devices)
except Exception as e:
    print('Error')
    print('False')
    print('0')
" 2>/dev/null)
OPENCV_VER=$(echo "$OPENCV_INFO" | sed -n '1p')
OPENCV_CUDA=$(echo "$OPENCV_INFO" | sed -n '2p')
OPENCV_DEV=$(echo "$OPENCV_INFO" | sed -n '3p')
OPENCV_OK=0
print_row "OpenCV Version" "$OPENCV_VER"
print_row "CUDA Module" "$([[ "$OPENCV_CUDA" == "True" ]] && echo "Available" || echo "Not available")"
print_row "CUDA Devices" "$OPENCV_DEV"
if [[ "$OPENCV_CUDA" == "True" && "$OPENCV_DEV" -gt 0 ]]; then
    print_row "Status" "✓ GPU Accelerated"
    OPENCV_OK=1
else
    print_row "Status" "⚠ CPU Only"
fi
print_line

print_header "PYTORCH CUDA TEST"
print_line
PYTORCH_INFO=$(python3 -c "
try:
    import torch
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
except Exception as e:
    print('Error')
    print('False')
    print('0')
    print('N/A')
" 2>/dev/null)
PYTORCH_VER=$(echo "$PYTORCH_INFO" | sed -n '1p')
PYTORCH_CUDA=$(echo "$PYTORCH_INFO" | sed -n '2p')
PYTORCH_DEV=$(echo "$PYTORCH_INFO" | sed -n '3p')
PYTORCH_NAME=$(echo "$PYTORCH_INFO" | sed -n '4p')
PYTORCH_OK=0
print_row "PyTorch Version" "$PYTORCH_VER"
print_row "CUDA Available" "$([[ "$PYTORCH_CUDA" == "True" ]] && echo "Yes" || echo "No")"
print_row "CUDA Devices" "$PYTORCH_DEV"
print_row "Device Name" "$PYTORCH_NAME"
if [[ "$PYTORCH_CUDA" == "True" ]]; then
    print_row "Status" "✓ GPU Accelerated"
    PYTORCH_OK=1
else
    print_row "Status" "⚠ CPU Only"
fi
print_line

print_header "ONNX RUNTIME TEST"
print_line
ONNX_INFO=$(python3 -c "
try:
    import onnxruntime as ort
    print(ort.__version__)
    providers = ort.get_available_providers()
    print(','.join(providers))
    has_cuda = 'CUDAExecutionProvider' in providers
    has_trt = 'TensorrtExecutionProvider' in providers
    print(has_cuda)
    print(has_trt)
except Exception as e:
    print('Error')
    print('CPUExecutionProvider')
    print('False')
    print('False')
" 2>/dev/null)
ONNX_VER=$(echo "$ONNX_INFO" | sed -n '1p')
ONNX_PROV=$(echo "$ONNX_INFO" | sed -n '2p')
ONNX_CUDA=$(echo "$ONNX_INFO" | sed -n '3p')
ONNX_TRT=$(echo "$ONNX_INFO" | sed -n '4p')
ONNX_OK=0
print_row "ONNX Runtime Version" "$ONNX_VER"
print_row "CUDA Provider" "$([[ "$ONNX_CUDA" == "True" ]] && echo "✓ Available" || echo "⚠ Not available")"
print_row "TensorRT Provider" "$([[ "$ONNX_TRT" == "True" ]] && echo "✓ Available" || echo "⚠ Not available")"
if [[ "$ONNX_CUDA" == "True" ]]; then
    print_row "Status" "✓ GPU Accelerated"
    ONNX_OK=1
else
    print_row "Status" "⚠ CPU Only"
fi
print_line

print_header "TENSORRT TEST"
print_line
TRT_INFO=$(python3 -c "
try:
    import tensorrt as trt
    print(trt.__version__)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    print(builder.platform_has_fast_fp16)
    print(builder.platform_has_fast_int8)
    print(builder.num_DLA_cores)
except Exception as e:
    print('Error')
    print('False')
    print('False')
    print('0')
" 2>/dev/null)
TRT_VER=$(echo "$TRT_INFO" | sed -n '1p')
TRT_FP16=$(echo "$TRT_INFO" | sed -n '2p')
TRT_INT8=$(echo "$TRT_INFO" | sed -n '3p')
TRT_DLA=$(echo "$TRT_INFO" | sed -n '4p')
TRT_OK=0
print_row "TensorRT Version" "$TRT_VER"
print_row "FP16 Support" "$([[ "$TRT_FP16" == "True" ]] && echo "✓ Yes" || echo "No")"
print_row "INT8 Support" "$([[ "$TRT_INT8" == "True" ]] && echo "✓ Yes" || echo "No")"
print_row "DLA Cores" "$TRT_DLA"
if [[ "$TRT_VER" != "Error" ]]; then
    print_row "Status" "✓ Available"
    TRT_OK=1
else
    print_row "Status" "⚠ Not available"
fi
print_line

print_header "YOLO MODEL TEST"
print_line
echo -ne "▶ Testing YOLO inference... "
YOLO_INFO=$(python3 -c "
try:
    from ultralytics import YOLO
    import torch
    model = YOLO('yolov8n.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    import time
    import numpy as np
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for _ in range(3): model(img, device=device, verbose=False)
    start = time.time()
    for _ in range(10): model(img, device=device, verbose=False)
    elapsed = (time.time() - start) / 10
    fps = 1.0 / elapsed
    print(f'{fps:.1f}')
    print(device)
    print(model.task)
except Exception as e:
    print('0')
    print('error')
    print(str(e)[:50])
" 2>/dev/null)
YOLO_FPS=$(echo "$YOLO_INFO" | sed -n '1p')
YOLO_DEV=$(echo "$YOLO_INFO" | sed -n '2p')
YOLO_TASK=$(echo "$YOLO_INFO" | sed -n '3p')
YOLO_OK=0
echo "done"
print_row "Inference Device" "$YOLO_DEV"
print_row "Task Type" "$YOLO_TASK"
print_row "FPS (YOLOv8n)" "$YOLO_FPS"
if [[ "$YOLO_DEV" == "cuda" ]]; then
    print_row "Status" "✓ GPU Accelerated"
    YOLO_OK=1
else
    print_row "Status" "⚠ CPU Only"
fi
print_line

print_header "GSTREAMER NVIDIA PLUGINS"
print_line
GST_PLUGINS=$(gst-inspect-1.0 2>/dev/null | grep -i "nv" | wc -l)
GST_NVVIDCONV=$(gst-inspect-1.0 nvvidconv 2>/dev/null && echo "1" || echo "0")
GST_NVV4L2H264=$(gst-inspect-1.0 nvv4l2h264enc 2>/dev/null && echo "1" || echo "0")
GST_NVV4L2DEC=$(gst-inspect-1.0 nvv4l2decoder 2>/dev/null && echo "1" || echo "0")
GST_OK=0
print_row "NVIDIA Plugins" "$GST_PLUGINS found"
print_row "nvvidconv" "$([[ "$GST_NVVIDCONV" == "1" ]] && echo "✓ Available" || echo "⚠ Missing")"
print_row "nvv4l2h264enc" "$([[ "$GST_NVV4L2H264" == "1" ]] && echo "✓ Available" || echo "⚠ Missing")"
print_row "nvv4l2decoder" "$([[ "$GST_NVV4L2DEC" == "1" ]] && echo "✓ Available" || echo "⚠ Missing")"
[[ "$GST_NVVIDCONV" == "1" ]] && GST_OK=1
print_line

print_header "VIDEO ENCODING TEST"
print_line
VENC_OK=0
if [[ -e "/dev/nvhost-msenc" || -e "/dev/nvhost-nvenc" ]]; then
    print_row "Hardware Encoder" "✓ Detected"
    if gst-launch-1.0 videotestsrc num-buffers=30 ! "video/x-raw,width=640,height=480" ! nvvidconv ! "video/x-raw(memory:NVMM)" ! nvv4l2h264enc ! fakesink -q 2>/dev/null; then
        print_row "H.264 Encoding" "✓ Working"
        VENC_OK=1
    else
        print_row "H.264 Encoding" "⚠ Failed"
    fi
    if gst-launch-1.0 videotestsrc num-buffers=30 ! "video/x-raw,width=640,height=480" ! nvvidconv ! "video/x-raw(memory:NVMM)" ! nvv4l2h265enc ! fakesink -q 2>/dev/null; then
        print_row "H.265 Encoding" "✓ Working"
    else
        print_row "H.265 Encoding" "⚠ Failed"
    fi
else
    print_row "Hardware Encoder" "⚠ Not detected"
fi
print_line

print_header "VIDEO DECODING TEST"
print_line
VDEC_OK=0
if [[ -e "/dev/nvhost-nvdec" ]]; then
    print_row "Hardware Decoder" "✓ Detected"
    VDEC_OK=1
else
    print_row "Hardware Decoder" "⚠ Not detected"
fi
print_line

print_header "CAMERA TEST"
print_line
CAM_OK=0
if [[ -e "/dev/video0" ]]; then
    CAM_INFO=$(v4l2-ctl --device=/dev/video0 --all 2>/dev/null | grep -E "Card type|Width|Height" | head -3)
    CAM_NAME=$(echo "$CAM_INFO" | grep "Card" | cut -d':' -f2 | sed 's/^[ \t]*//')
    print_row "Camera Device" "/dev/video0"
    print_row "Camera Name" "${CAM_NAME:-Unknown}"
    print_row "Status" "✓ Available"
    CAM_OK=1
else
    print_row "Camera Device" "⚠ Not detected"
fi
print_line

print_header "DIAGNOSTICS SUMMARY"
echo -e "${PURPLE}"
echo "      ██╗   ██╗ ██████╗ ██╗      ██████╗ "
echo "      ╚██╗ ██╔╝██╔═══██╗██║     ██╔═══██╗"
echo "       ╚████╔╝ ██║   ██║██║     ██║   ██║"
echo "        ╚██╔╝  ██║   ██║██║     ██║   ██║"
echo "         ██║   ╚██████╔╝███████╗╚██████╔╝"
echo "         ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝ "
echo -e "${NC}"
print_line
printf "| %-28s | %-25s |\n" "Component" "Status"
print_line
print_row "CUDA Toolkit" "$([[ $CUDA_OK -eq 1 ]] && echo "✓ Available" || echo "⚠ Missing")"
print_row "OpenCV CUDA" "$([[ $OPENCV_OK -eq 1 ]] && echo "✓ Accelerated" || echo "⚠ CPU Only")"
print_row "PyTorch CUDA" "$([[ $PYTORCH_OK -eq 1 ]] && echo "✓ Accelerated" || echo "⚠ CPU Only")"
print_row "ONNX Runtime GPU" "$([[ $ONNX_OK -eq 1 ]] && echo "✓ Accelerated" || echo "⚠ CPU Only")"
print_row "TensorRT" "$([[ $TRT_OK -eq 1 ]] && echo "✓ Available" || echo "⚠ Missing")"
print_row "YOLO Inference" "$([[ $YOLO_OK -eq 1 ]] && echo "✓ GPU ($YOLO_FPS FPS)" || echo "⚠ CPU Only")"
print_row "GStreamer NVIDIA" "$([[ $GST_OK -eq 1 ]] && echo "✓ Available" || echo "⚠ Missing")"
print_row "Video Encoding" "$([[ $VENC_OK -eq 1 ]] && echo "✓ Working" || echo "⚠ Not available")"
print_row "Video Decoding" "$([[ $VDEC_OK -eq 1 ]] && echo "✓ Working" || echo "⚠ Not available")"
print_row "Camera" "$([[ $CAM_OK -eq 1 ]] && echo "✓ Detected" || echo "⚠ Not found")"
print_line
TOTAL=$((CUDA_OK + OPENCV_OK + PYTORCH_OK + ONNX_OK + TRT_OK + YOLO_OK + GST_OK + VENC_OK + VDEC_OK + CAM_OK))
MAX=10
PCT=$((TOTAL * 100 / MAX))
BAR_SIZE=20
FILLED=$((BAR_SIZE * TOTAL / MAX))
BAR=""
for ((i=0; i<FILLED; i++)); do BAR="${BAR}█"; done
for ((i=FILLED; i<BAR_SIZE; i++)); do BAR="${BAR}░"; done
print_row "Overall Score" "$PCT% ($TOTAL/$MAX)"
print_row "Progress" "$BAR"
print_line
if [[ $TOTAL -ge 8 ]]; then
    echo -e "\n${GREEN}${BOLD}✓ System is ready for YOLO Vision applications${NC}"
elif [[ $TOTAL -ge 5 ]]; then
    echo -e "\n${YELLOW}${BOLD}⚠ System partially ready - some features may be limited${NC}"
else
    echo -e "\n${RED}${BOLD}✗ System not properly configured - run init.sh first${NC}"
fi
echo -e "\n${CYAN}Log saved to: $LOG_FILE${NC}"
echo -e "${CYAN}Diagnostics completed at: $(date '+%Y-%m-%d %H:%M:%S')${NC}\n"
