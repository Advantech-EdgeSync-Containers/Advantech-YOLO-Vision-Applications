#!/usr/bin/env bash
# ==========================================================================
# Advantech YOLO Vision Applications - Environment Initialization Script
# ==========================================================================
# Version:      2.0.0
# Author:       Samir Singh <samir.singh@advantech.com>
# Created:      January 15, 2025
# Last Updated: November 28, 2025
#
# Description:
#   This script initializes the container environment for Advantech YOLO
#   Vision Applications. It handles:
#   - Uninstalling conflicting ONNX packages
#   - Installing NVIDIA-optimized ONNX Runtime GPU for JetPack 5.1.2
#   - Installing required Python packages (onnx, onnx-simplifier, flask)
#   - Verifying GPU acceleration and all components
#
# Usage:
#   ./init.sh [options]
#
# Options:
#   --skip-verify    Skip verification steps
#   --force          Force reinstall even if already installed
#   --quiet          Suppress verbose output
#   --help           Show this help message
#
# Terms and Conditions:
#   1. This software is provided by Advantech Corporation "as is" and any
#      express or implied warranties, including, but not limited to, the implied
#      warranties of merchantability and fitness for a particular purpose are
#      disclaimed.
#   2. In no event shall Advantech Corporation be liable for any direct, indirect,
#      incidental, special, exemplary, or consequential damages arising in any way
#      out of the use of this software.
#
# Copyright (c) 2024-2025 Advantech Corporation. All rights reserved.
# ==========================================================================

set -euo pipefail

# ==========================================================================
# Constants
# ==========================================================================
readonly SCRIPT_VERSION="2.0.0"
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ONNX Runtime GPU wheel for JetPack 5.1.2 (CUDA 11.4, Python 3.8)
readonly ONNX_WHEEL_URL="https://nvidia.box.com/shared/static/iizg3ggrtdkqawkmebbfixo7sce6j365.whl"
readonly ONNX_WHEEL_NAME="onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl"
readonly ONNX_EXPECTED_VERSION="1.16.0"

# Python packages to install
readonly ONNX_VERSION="1.16.0"
readonly ONNXSIM_VERSION="0.4.36"
readonly FLASK_VERSION="3.0.3"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly PURPLE='\033[0;35m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# ==========================================================================
# Global Variables
# ==========================================================================
SKIP_VERIFY=false
FORCE_REINSTALL=false
QUIET=false
ERROR_COUNT=0
WARNING_COUNT=0

# ==========================================================================
# Logging Functions
# ==========================================================================
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}
log() {
    echo -e "$(timestamp) ${CYAN}[INFO]${NC} $*"
}
log_success() {
    echo -e "$(timestamp) ${GREEN}[SUCCESS]${NC} ✓ $*"
}
log_warning() {
    echo -e "$(timestamp) ${YELLOW}[WARNING]${NC} ⚠ $*"
    ((WARNING_COUNT++)) || true
}
log_error() {
    echo -e "$(timestamp) ${RED}[ERROR]${NC} ✗ $*" >&2
    ((ERROR_COUNT++)) || true
}
log_step() {
    echo -e "\n${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}${BOLD}  $*${NC}"
    echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}
log_substep() {
    echo -e "${PURPLE}→ $*${NC}"
}

# ==========================================================================
# Banner
# ==========================================================================
print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║     █████╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗ ██████╗██╗  ██╗  ║"
    echo "║    ██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔════╝██║  ██║  ║"
    echo "║    ███████║██║  ██║╚██╗ ██╔╝███████║██╔██╗ ██║   ██║   █████╗  ██║     ███████║  ║"
    echo "║    ██╔══██║██║  ██║ ╚████╔╝ ██╔══██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██╔══██║  ║"
    echo "║    ██║  ██║██████╔╝  ╚██╔╝  ██║  ██║██║ ╚████║   ██║   ███████╗╚██████╗██║  ██║  ║"
    echo "║    ╚═╝  ╚═╝╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝  ║"
    echo "║                                                                                  ║"
    echo "║                    YOLO Vision Applications - Environment Setup                  ║"
    echo "║                              Version ${SCRIPT_VERSION}                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}
# ==========================================================================
# Help
# ==========================================================================
show_help() {
    cat << EOF
${BOLD}Advantech YOLO Vision Applications - Environment Initialization${NC}

${BOLD}Usage:${NC}
    $SCRIPT_NAME [options]

${BOLD}Options:${NC}
    --skip-verify    Skip verification steps after installation
    --force          Force reinstall even if packages are already installed
    --quiet          Suppress verbose output
    --help           Show this help message

${BOLD}Description:${NC}
    This script initializes the container environment by:
    1. Checking system requirements (JetPack, CUDA, Python)
    2. Uninstalling conflicting ONNX packages
    3. Installing NVIDIA-optimized ONNX Runtime GPU
    4. Installing required Python packages
    5. Verifying GPU acceleration and all components

${BOLD}Requirements:${NC}
    - JetPack 5.1.1 or 5.1.2
    - CUDA 11.4
    - Python 3.8
    - pip3

${BOLD}Examples:${NC}
    # Standard installation
    ./init.sh

    # Force reinstall all packages
    ./init.sh --force

    # Quick install without verification
    ./init.sh --skip-verify

EOF
}

# ==========================================================================
# Argument Parsing
# ==========================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-verify)
                SKIP_VERIFY=true
                shift
                ;;
            --force)
                FORCE_REINSTALL=true
                shift
                ;;
            --quiet)
                QUIET=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# ==========================================================================
# System Checks
# ==========================================================================
check_system_requirements() {
    log_step "Step 1: Checking System Requirements"

    # Check if running as root or with sudo
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root. This is acceptable inside a container."
    fi

    # Check JetPack version
    log_substep "Checking JetPack version..."
    if [[ -f /etc/nv_tegra_release ]]; then
        local tegra_release
        tegra_release=$(cat /etc/nv_tegra_release)
        log "Tegra release: $tegra_release"
        
        if [[ "$tegra_release" == *"R35"* ]]; then
            log_success "JetPack 5.x detected (L4T R35)"
        elif [[ "$tegra_release" == *"R36"* ]]; then
            log_warning "JetPack 6.x detected (L4T R36) - some features may need adjustment"
        else
            log_warning "Unknown JetPack version. Proceeding anyway..."
        fi
    else
        log_warning "Not running on Jetson device or /etc/nv_tegra_release not found"
    fi

    # Check CUDA
    log_substep "Checking CUDA installation..."
    if command -v nvcc &> /dev/null; then
        local cuda_version
        cuda_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
        log_success "CUDA $cuda_version found"
    else
        log_warning "nvcc not found in PATH. CUDA may still be available via libraries."
    fi

    # Check Python version
    log_substep "Checking Python version..."
    local python_version
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    log "Python version: $python_version"
    
    if [[ "$python_version" == 3.8* ]]; then
        log_success "Python 3.8 detected - compatible with JetPack 5.1.2 wheels"
    else
        log_warning "Python $python_version detected. Expected 3.8 for JetPack 5.1.2 compatibility"
    fi

    # Check pip
    log_substep "Checking pip installation..."
    if command -v pip3 &> /dev/null; then
        local pip_version
        pip_version=$(pip3 --version | cut -d' ' -f2)
        log_success "pip3 version $pip_version found"
    else
        log_error "pip3 not found. Please install python3-pip"
        exit 1
    fi

    # Check available disk space
    log_substep "Checking available disk space..."
    local free_space
    free_space=$(df -BG /tmp | awk 'NR==2 {print $4}' | tr -d 'G')
    if [[ "$free_space" -lt 2 ]]; then
        log_warning "Low disk space: ${free_space}GB free. At least 2GB recommended."
    else
        log_success "Disk space OK: ${free_space}GB free"
    fi
}

# ==========================================================================
# Check Current Installation Status
# ==========================================================================
check_current_status() {
    log_step "Step 2: Checking Current Installation Status"
    log_substep "Checking ONNX Runtime status..."
    if python3 -c "import onnxruntime" 2>/dev/null; then
        local ort_version
        ort_version=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo "unknown")
        local providers
        providers=$(python3 -c "import onnxruntime as ort; print(','.join(ort.get_available_providers()))" 2>/dev/null || echo "")    
        if [[ "$providers" == *"CUDA"* ]] || [[ "$providers" == *"TensorRT"* ]]; then
            if [[ "$FORCE_REINSTALL" == "false" ]]; then
                log_success "ONNX Runtime GPU $ort_version already installed with providers: $providers"
                log "Use --force to reinstall"
            else
                log "ONNX Runtime GPU $ort_version found, will reinstall (--force specified)"
            fi
        else
            log_warning "ONNX Runtime $ort_version installed but no GPU support detected"
            log "Providers: $providers"
        fi
    else
        log "ONNX Runtime not installed"
    fi

    log_substep "Checking ONNX status..."
    if python3 -c "import onnx" 2>/dev/null; then
        local onnx_version
        onnx_version=$(python3 -c "import onnx; print(onnx.__version__)" 2>/dev/null || echo "unknown")
        log "ONNX $onnx_version installed"
    else
        log "ONNX not installed"
    fi

    log_substep "Checking ONNX Simplifier status..."
    if python3 -c "import onnxsim" 2>/dev/null; then
        local onnxsim_version
        onnxsim_version=$(python3 -c "import onnxsim; print(onnxsim.__version__)" 2>/dev/null || echo "unknown")
        log "ONNX Simplifier $onnxsim_version installed"
    else
        log "ONNX Simplifier not installed"
    fi

    log_substep "Checking Flask status..."
    if python3 -c "import flask" 2>/dev/null; then
        local flask_version
        flask_version=$(python3 -c "import flask; print(flask.__version__)" 2>/dev/null || echo "unknown")
        log "Flask $flask_version installed"
    else
        log "Flask not installed"
    fi
}

# ==========================================================================
# Uninstall Conflicting Packages
# ==========================================================================
uninstall_conflicting_packages() {
    log_step "Step 3: Removing Conflicting ONNX Packages"

    log_substep "Uninstalling existing ONNX-related packages..."
    
    # List of packages to uninstall
    local packages=(
        "onnxruntime"
        "onnxruntime-gpu"
        "onnxruntime-gpu-tensorrt"
        "ort-nightly"
        "ort-nightly-gpu"
    )
    for pkg in "${packages[@]}"; do
        log "Removing $pkg (if installed)..."
        pip3 uninstall -y "$pkg" 2>/dev/null || true
    done
    log_substep "Removing ONNX and ONNX Simplifier for clean reinstall..."
    pip3 uninstall -y onnx onnxsim onnx-simplifier 2>/dev/null || true
    log_substep "Cleaning pip cache..."
    pip3 cache purge 2>/dev/null || true
    log_success "Conflicting packages removed"
}

# ==========================================================================
# Install ONNX Runtime GPU
# ==========================================================================
install_onnxruntime_gpu() {
    log_step "Step 4: Installing ONNX Runtime GPU for JetPack 5.1.2"
    if [[ "$FORCE_REINSTALL" == "false" ]]; then
        if python3 -c "import onnxruntime as ort; exit(0 if 'CUDAExecutionProvider' in ort.get_available_providers() else 1)" 2>/dev/null; then
            log_success "ONNX Runtime GPU already installed and working. Skipping."
            return 0
        fi
    fi
    log_substep "Downloading ONNX Runtime GPU wheel..."
    local wheel_path="/tmp/${ONNX_WHEEL_NAME}"
    
    if [[ -f "$wheel_path" ]]; then
        log "Using cached wheel: $wheel_path"
    else
        if ! wget -q --show-progress "$ONNX_WHEEL_URL" -O "$wheel_path"; then
            log_error "Failed to download ONNX Runtime wheel from: $ONNX_WHEEL_URL"
            log "Attempting alternative download method..."
            
            if ! curl -L -o "$wheel_path" "$ONNX_WHEEL_URL"; then
                log_error "All download methods failed"
                exit 1
            fi
        fi
    fi
    if [[ ! -f "$wheel_path" ]] || [[ ! -s "$wheel_path" ]]; then
        log_error "Downloaded wheel file is missing or empty"
        exit 1
    fi
    local file_size
    file_size=$(ls -lh "$wheel_path" | awk '{print $5}')
    log "Downloaded wheel size: $file_size"
    log_substep "Installing ONNX Runtime GPU wheel..."
    if ! pip3 install "$wheel_path"; then
        log_error "Failed to install ONNX Runtime GPU wheel"
        exit 1
    fi
    rm -f "$wheel_path"
    log_success "ONNX Runtime GPU installed successfully"
}

# ==========================================================================
# Install Required Python Packages
# ==========================================================================
install_python_packages() {
    log_substep "Installing onnx==${ONNX_VERSION}..."
    pip3 install "onnx==${ONNX_VERSION}" --quiet || {
        log_warning "Failed to install specific onnx version, trying latest"
        pip3 install onnx --quiet
    }
    log_substep "Installing onnxsim==${ONNXSIM_VERSION}..."
    pip3 install "onnxsim==${ONNXSIM_VERSION}" --quiet || {
        log_warning "Failed to install specific onnxsim version, trying latest"
        pip3 install onnxsim --quiet
    }
    log_substep "Installing Flask==${FLASK_VERSION}..."
    pip3 install "Flask==${FLASK_VERSION}" --quiet || {
        log_warning "Failed to install specific Flask version, trying latest"
        pip3 install flask --quiet
    }
    log_substep "Installing additional dependencies..."
    pip3 install --quiet \
        numpy \
        sentry_sdk \
        pillow \
        pyyaml \
        tqdm \
        requests \
        2>/dev/null || log_warning "Some optional packages may have failed to install"
    log_success "Python packages installed"
}

# ==========================================================================
# Verify Installation
# ==========================================================================
verify_installation() {
    if [[ "$SKIP_VERIFY" == "true" ]]; then
        log "Skipping verification (--skip-verify specified)"
        return 0
    fi
    log_step "Step 6: Verifying Installation"
    local all_passed=true
    log_substep "Verifying ONNX Runtime GPU..."
    if python3 << 'EOF'
import sys
try:
    import onnxruntime as ort
    version = ort.__version__
    providers = ort.get_available_providers()
    print(f"ONNX Runtime Version: {version}")
    print(f"Available Providers: {providers}")
    
    if "CUDAExecutionProvider" in providers:
        print("✓ CUDA Provider: ENABLED")
    else:
        print("✗ CUDA Provider: NOT FOUND")
        sys.exit(1)
        
    if "TensorrtExecutionProvider" in providers:
        print("✓ TensorRT Provider: ENABLED")
    else:
        print("⚠ TensorRT Provider: NOT FOUND (optional)")
    
    sys.exit(0)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
EOF
    then
        log_success "ONNX Runtime GPU verification passed"
    else
        log_error "ONNX Runtime GPU verification failed"
        all_passed=false
    fi
    log_substep "Verifying ONNX..."
    if python3 -c "import onnx; print(f'ONNX Version: {onnx.__version__}')" 2>/dev/null; then
        log_success "ONNX verification passed"
    else
        log_error "ONNX verification failed"
        all_passed=false
    fi
    log_substep "Verifying ONNX Simplifier..."
    if python3 -c "import onnxsim; print(f'ONNX Simplifier Version: {onnxsim.__version__}')" 2>/dev/null; then
        log_success "ONNX Simplifier verification passed"
    else
        log_error "ONNX Simplifier verification failed"
        all_passed=false
    fi
    log_substep "Verifying Flask..."
    if python3 -c "import flask; print(f'Flask Version: {flask.__version__}')" 2>/dev/null; then
        log_success "Flask verification passed"
    else
        log_error "Flask verification failed"
        all_passed=false
    fi
    log_substep "Verifying PyTorch CUDA..."
    if python3 << 'EOF'
import sys
try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        sys.exit(0)
    else:
        print("⚠ CUDA not available in PyTorch")
        sys.exit(0)  # Not a fatal error
except ImportError:
    print("⚠ PyTorch not installed")
    sys.exit(0)  # Not a fatal error
EOF
    then
        log_success "PyTorch verification passed"
    else
        log_warning "PyTorch verification skipped or incomplete"
    fi
    log_substep "Verifying TensorRT..."
    if python3 -c "import tensorrt; print(f'TensorRT Version: {tensorrt.__version__}')" 2>/dev/null; then
        log_success "TensorRT verification passed"
    else
        log_warning "TensorRT not available (optional for ONNX-only usage)"
    fi
    echo ""
    if [[ "$all_passed" == "true" ]]; then
        log_success "All critical verifications passed!"
    else
        log_error "Some verifications failed. Check the output above."
    fi
}

# ==========================================================================
# Run Simple Inference Test
# ==========================================================================
run_inference_test() {
    if [[ "$SKIP_VERIFY" == "true" ]]; then
        return 0
    fi
    log_step "Step 7: Running Inference Test"
    log_substep "Testing ONNX Runtime CUDA inference..."
    if python3 << 'EOF'
import numpy as np
import onnxruntime as ort

# Create a simple test model
print("Creating test ONNX model...")

import onnx
from onnx import helper, TensorProto

# Simple identity model
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 224, 224])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3, 224, 224])
node = helper.make_node('Identity', ['X'], ['Y'])
graph = helper.make_graph([node], 'test_graph', [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])

# Save model
model_path = '/tmp/test_model.onnx'
onnx.save(model, model_path)

# Test inference with CUDA
print("Running inference with CUDA provider...")
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)

# Check which provider is being used
active_providers = session.get_providers()
print(f"Active providers: {active_providers}")

# Run inference
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = session.run(None, {'X': input_data})

print(f"Input shape: {input_data.shape}")
print(f"Output shape: {output[0].shape}")
print("✓ CUDA inference test passed!")

import os
os.remove(model_path)
EOF
    then
        log_success "Inference test passed"
    else
        log_error "Inference test failed"
    fi
}

# ==========================================================================
# Print Summary
# ==========================================================================
print_summary() {
    log_step "Installation Summary"
    echo -e "${BOLD}Installed Components:${NC}"
    echo ""
    local ort_info
    ort_info=$(python3 -c "
import onnxruntime as ort
print(f'  Version: {ort.__version__}')
print(f'  Providers: {ort.get_available_providers()}')
" 2>/dev/null || echo "  Status: Not installed")
    echo -e "${CYAN}ONNX Runtime GPU:${NC}"
    echo "$ort_info"
    echo ""
    local onnx_info
    onnx_info=$(python3 -c "import onnx; print(f'  Version: {onnx.__version__}')" 2>/dev/null || echo "  Status: Not installed")
    echo -e "${CYAN}ONNX:${NC}"
    echo "$onnx_info"
    echo ""
    local onnxsim_info
    onnxsim_info=$(python3 -c "import onnxsim; print(f'  Version: {onnxsim.__version__}')" 2>/dev/null || echo "  Status: Not installed")
    echo -e "${CYAN}ONNX Simplifier:${NC}"
    echo "$onnxsim_info"
    echo ""
    local flask_info
    flask_info=$(python3 -c "import flask; print(f'  Version: {flask.__version__}')" 2>/dev/null || echo "  Status: Not installed")
    echo -e "${CYAN}Flask:${NC}"
    echo "$flask_info"
    echo ""
    if [[ $WARNING_COUNT -gt 0 ]] || [[ $ERROR_COUNT -gt 0 ]]; then
        echo -e "${YELLOW}Warnings: $WARNING_COUNT${NC}"
        echo -e "${RED}Errors: $ERROR_COUNT${NC}"
        echo ""
    fi
    echo -e "${BOLD}Next Steps:${NC}"
    echo "  1. Download a model:"
    echo "     python3 src/advantech-coe-model-load.py --task detection --size n"
    echo ""
    echo "  2. Export to optimized format:"
    echo "     python3 src/advantech-coe-model-export.py --task detection --size n --format onnx"
    echo ""
    echo "  3. Run inference:"
    echo "     python3 src/advantech_pipeline.py --model yolov8n.onnx --source /dev/video0"
    echo ""    
    if [[ $ERROR_COUNT -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}✓ Environment initialization completed successfully!${NC}"
    else
        echo -e "${RED}${BOLD}✗ Environment initialization completed with errors.${NC}"
        echo -e "${RED}  Please check the output above and refer to troubleshooting.md${NC}"
    fi
}

# ==========================================================================
# Main
# ==========================================================================
main() {
    parse_args "$@"

    print_banner

    check_system_requirements
    check_current_status
    uninstall_conflicting_packages
    install_onnxruntime_gpu
    install_python_packages
    verify_installation
    run_inference_test
    print_summary
    exit $ERROR_COUNT
}

main "$@"
