#!/usr/bin/env bash
# Advantech YOLO Vision - Environment Init | Version 2.0.0 | Copyright (c) 2024-2025 Advantech Corporation
readonly SCRIPT_VERSION="2.0.0"
readonly ONNX_WHEEL_URL="https://nvidia.box.com/shared/static/iizg3ggrtdkqawkmebbfixo7sce6j365.whl"
readonly ONNX_WHEEL_NAME="onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl"
readonly ONNX_VERSION="1.16.0"
readonly ONNXSIM_VERSION="0.4.36"
readonly FLASK_VERSION="2.3.3"
readonly RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m' BLUE='\033[0;34m'
readonly CYAN='\033[0;36m' BOLD='\033[1m' NC='\033[0m'
SKIP_VERIFY=false
FORCE_REINSTALL=false
ERROR_COUNT=0
TOTAL_STEPS=9
CURRENT_STEP=0
show_progress() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    local percent=$((CURRENT_STEP * 100 / TOTAL_STEPS))
    local filled=$((percent / 4))
    local empty=$((20 - filled))
    local bar=""
    local space=""
    for ((i=0; i<filled; i++)); do bar="${bar}█"; done
    for ((i=0; i<empty; i++)); do space="${space}░"; done
    printf "\r${CYAN}[%3d%%]${NC} ${GREEN}%s%s${NC} %-45s" "$percent" "$bar" "$space" "$1"
    if [[ $percent -ge 100 ]]; then echo ""; fi
}
print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════════════════════════════╗"
    echo "║     █████╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗ ██████╗██╗  ██╗          ║"
    echo "║    ██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔════╝██║  ██║          ║"
    echo "║    ███████║██║  ██║╚██╗ ██╔╝███████║██╔██╗ ██║   ██║   █████╗  ██║     ███████║          ║"
    echo "║    ██╔══██║██║  ██║ ╚████╔╝ ██╔══██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██╔══██║          ║"
    echo "║    ██║  ██║██████╔╝  ╚██╔╝  ██║  ██║██║ ╚████║   ██║   ███████╗╚██████╗██║  ██║          ║"
    echo "║    ╚═╝  ╚═╝╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝          ║"
    echo "║                         YOLO Vision Applications - Environment Setup                     ║"
    echo "║                                    Version ${SCRIPT_VERSION}                                         ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-verify) SKIP_VERIFY=true; shift;;
            --force) FORCE_REINSTALL=true; shift;;
            --help|-h) echo "Usage: $0 [--skip-verify] [--force] [--help]"; exit 0;;
            *) shift;;
        esac
    done
}
run_install() {
    show_progress "Checking system requirements..."
    sleep 0.3
    show_progress "Removing conflicting packages..."
    pip3 uninstall -y onnxruntime onnxruntime-gpu onnxruntime-gpu-tensorrt ort-nightly ort-nightly-gpu onnx onnxsim onnx-simplifier &>/dev/null || true
    pip3 cache purge &>/dev/null || true
    show_progress "Downloading ONNX Runtime GPU..."
    local wheel_path="/tmp/${ONNX_WHEEL_NAME}"
    local need_install=true
    if [[ "$FORCE_REINSTALL" != "true" ]]; then
        if python3 -c "import onnxruntime as ort; exit(0 if 'CUDAExecutionProvider' in ort.get_available_providers() else 1)" &>/dev/null; then
            need_install=false
        fi
    fi
    if [[ "$need_install" == "true" ]]; then
        if [[ ! -f "$wheel_path" ]]; then
            wget -q "$ONNX_WHEEL_URL" -O "$wheel_path" 2>/dev/null || curl -sL -o "$wheel_path" "$ONNX_WHEEL_URL" 2>/dev/null
            if [[ ! -f "$wheel_path" ]] || [[ ! -s "$wheel_path" ]]; then
                echo -e "\n${RED}Download failed${NC}"
                ERROR_COUNT=1
                return
            fi
        fi
        show_progress "Installing ONNX Runtime GPU..."
        if ! pip3 install "$wheel_path" &>/dev/null; then
            echo -e "\n${RED}Install failed${NC}"
            ERROR_COUNT=1
            return
        fi
        rm -f "$wheel_path"
    else
        show_progress "ONNX Runtime GPU already installed..."
    fi
    show_progress "Installing onnx..."
    pip3 uninstall -y onnx &>/dev/null || true
    pip3 install "onnx==${ONNX_VERSION}" --quiet --force-reinstall &>/dev/null || true
    show_progress "Installing onnxsim..."
    pip3 install "onnxsim==${ONNXSIM_VERSION}" --quiet &>/dev/null || pip3 install onnxsim --quiet &>/dev/null || true
    show_progress "Installing Flask..."
    pip3 install "Flask==${FLASK_VERSION}" --quiet &>/dev/null || pip3 install flask --quiet &>/dev/null || true
    show_progress "Installing dependencies..."
    pip3 install --quiet numpy pillow pyyaml tqdm requests &>/dev/null || true
    if [[ "$SKIP_VERIFY" == "false" ]]; then
        show_progress "Verifying installation..."
        if ! python3 -c "import onnxruntime as ort; exit(0 if 'CUDAExecutionProvider' in ort.get_available_providers() else 1)" &>/dev/null; then
            ERROR_COUNT=1
        fi
    else
        show_progress "Skipping verification..."
    fi
}
print_result() {
    echo ""
    if [[ $ERROR_COUNT -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}${BOLD}║                    ✓ Environment initialization completed successfully!                  ║${NC}"
        echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${BOLD}Installed:${NC}"
        python3 -c "import onnxruntime as ort; print(f'  ONNX Runtime GPU: {ort.__version__}')" 2>/dev/null || echo "  ONNX Runtime GPU: Not installed"
        python3 -c "import onnx; print(f'  ONNX: {onnx.__version__}')" 2>/dev/null || echo "  ONNX: Not installed"
        python3 -c "import onnxsim; print(f'  ONNX Simplifier: {onnxsim.__version__}')" 2>/dev/null || echo "  ONNX Simplifier: Not installed"
        python3 -c "import flask; print(f'  Flask: {flask.__version__}')" 2>/dev/null || echo "  Flask: Not installed"
        echo ""
        echo -e "${BOLD}Next Steps:${NC}"
        echo "  1. python3 src/advantech-coe-model-load.py --task detection --size n"
        echo "  2. python3 src/advantech-yolo.py --model yolov8n.pt --source /dev/video0"
    else
        echo -e "${RED}${BOLD}╔══════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}${BOLD}║                    ✗ Environment initialization failed!                                  ║${NC}"
        echo -e "${RED}${BOLD}╚══════════════════════════════════════════════════════════════════════════════════════════╝${NC}"
        echo -e "${YELLOW}Run with --force to reinstall: ./init.sh --force${NC}"
    fi
}
main() {
    parse_args "$@"
    print_banner
    echo -e "${CYAN}Initializing environment...${NC}\n"
    run_install
    print_result
    exit $ERROR_COUNT
}
main "$@"
