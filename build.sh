#!/bin/bash
# ==========================================================================
# Jetson GPU Passthrough Docker Compose Build Script
# ==========================================================================
# Version:      2.1.0
# Author:       Samir Singh <samir.singh@nvantech.com> and Apoorv Saxena<apoorv.saxena@advantech.com>
# Created:      January 10, 2025
# Last Updated: July 11, 2025
# Description:
#   This script prepares a Docker environment optimized for GPU and display
#   passthrough on Advantech edge AI platforms. It:
#     • Creates standard project directories (src, models, data, diagnostics)
#     • Configures X11 or Wayland forwarding for GUI applications
#     • Sets up NVIDIA GPU device access and permissions in containers
#     • Enables display passthrough for accelerated rendering
#     • Launches containers with hardware acceleration support
#
# Terms and Conditions:
#   1. Provided by Advantech Corporation "as is," with no express or implied
#      warranties of merchantability or fitness for a particular purpose.
#   2. Advantech Corporation shall not be liable for any direct, indirect,
#      incidental, special, exemplary, or consequential damages arising from
#      the use of this software.
#   3. Redistribution and use in source or binary form, with or without
#      modification, are permitted provided this notice appears in all copies.
#
# Copyright (c) 2025 Advantech Corporation. All rights reserved.
# ==========================================================================

set -euo pipefail

readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CONTAINER_NAME="advantech-yolo-vision"
readonly PROJECT_DIRS=("src" "models" "data")
readonly XAUTH_FILE="${HOME}/.docker.xauth"  
readonly COMPOSE_TIMEOUT=60
readonly RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m' CYAN='\033[0;36m' WHITE='\033[1;37m' NC='\033[0m'

log() { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }
log_error() { echo -e "${RED}[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $*${NC}" >&2; }
log_success() { echo -e "${GREEN}[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') $*${NC}" >&2; }
log_warning() { echo -e "${YELLOW}[WARN] $(date '+%Y-%m-%d %H:%M:%S') $*${NC}" >&2; }

error_handler() {
    local line_no=$1
    log_error "Build script failed at line ${line_no}"
    exit 1
}
trap 'error_handler ${LINENO}' ERR

display_banner() {
    clear
    echo -e "${BLUE}"
    cat << 'EOF'
       █████╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗ ██████╗██╗  ██╗     ██████╗ ██████╗ ███████╗
      ██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔════╝██║  ██║    ██╔════╝██╔═══██╗██╔════╝
      ███████║██║  ██║██║   ██║███████║██╔██╗ ██║   ██║   █████╗  ██║     ███████║    ██║     ██║   ██║█████╗  
      ██╔══██║██║  ██║╚██╗ ██╔╝██╔══██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██╔══██║    ██║     ██║   ██║██╔══╝  
      ██║  ██║██████╔╝ ╚████╔╝ ██║  ██║██║ ╚████║   ██║   ███████╗╚██████╗██║  ██║    ╚██████╗╚██████╔╝███████╗
      ╚═╝  ╚═╝╚═════╝   ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝     ╚═════╝ ╚═════╝ ╚══════╝
EOF
    echo -e "${WHITE}                                  Center of Excellence${NC}"
    echo
    echo -e "${CYAN}Initializing AI Development Environment...${NC}\n"
    sleep 2
}

command_exists() { command -v "$1" &>/dev/null; }

check_prerequisites() {
    log "Verifying prerequisites..."
    local missing_deps=()
    if ! command_exists docker; then missing_deps+=("docker"); fi
    if ! command_exists docker-compose && ! (command_exists docker && docker compose version &>/dev/null); then
        missing_deps+=("docker-compose")
    fi
    if ! command_exists xhost; then log_warning "xhost not found - X11 forwarding may not work properly"; fi
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_error "Please install the missing dependencies and try again"
        return 1
    fi
    if ! docker info &>/dev/null; then log_error "Docker daemon is not running"; return 1; fi
    if ! docker info 2>/dev/null | grep -q nvidia; then
        log_warning "NVIDIA Docker runtime not detected - GPU acceleration may not work"
    fi
    log_success "Prerequisites verified"
    return 0
}

init_project_structure() {
    log "Checking project directory structure..."
    for dir in "${PROJECT_DIRS[@]}"; do
        if [[ ! -d "$dir" ]]; then mkdir -p "$dir"; log "Created directory: $dir"
        else log "Directory already exists: $dir"; fi
    done
    for dir in "${PROJECT_DIRS[@]}"; do
        if [[ -d "$dir" && -z "$(ls -A "$dir")" ]]; then touch "$dir/.gitkeep"; fi
    done
    log_success "Project structure verified"
}

setup_x11_forwarding() {
    log "Configuring X11 forwarding..."
    if [[ -n "${SSH_CONNECTION:-}" ]]; then
        log_warning "SSH session detected - X11 forwarding may require additional configuration"
    fi
    local x11_configured=false
    if [[ -n "${DISPLAY:-}" ]]; then log "DISPLAY=${DISPLAY}"; x11_configured=true
    else log_warning "DISPLAY variable not set"; fi
    if [[ -z "${XAUTHORITY:-}" ]]; then
        if command_exists xauth; then
            local xauth_path
            xauth_path=$(xauth info 2>/dev/null | grep "Authority file" | awk '{print $3}')
            if [[ -n "$xauth_path" ]]; then export XAUTHORITY="$xauth_path"; log "XAUTHORITY set to: ${XAUTHORITY}"; fi
        fi
    else log "XAUTHORITY=${XAUTHORITY}"; fi
    if [[ -z "${XDG_RUNTIME_DIR:-}" ]]; then
        export XDG_RUNTIME_DIR="/run/user/$(id -u)"
        log "XDG_RUNTIME_DIR set to: ${XDG_RUNTIME_DIR}"
    fi
    if command_exists xhost && [[ "$x11_configured" == "true" ]]; then
        log "Configuring xhost access..."
        xhost +local:docker &>/dev/null || log_warning "Failed to configure xhost"
        log "Creating X authentication file..."
        rm -f "${XAUTH_FILE}" 2>/dev/null || true
        touch "${XAUTH_FILE}"
        if command_exists xauth; then
            xauth nlist "${DISPLAY}" 2>/dev/null | sed -e 's/^..../ffff/' | \
                xauth -f "${XAUTH_FILE}" nmerge - &>/dev/null || \
                log_warning "Failed to merge X authentication data"
        fi
        chmod 666 "${XAUTH_FILE}" 2>/dev/null || true
    else log_warning "X11 forwarding not configured - GUI applications may not work"; fi
    log_success "X11 configuration completed"
}

start_containers() {
    log "Starting Docker containers..."
    if [[ ! -f "${SCRIPT_DIR}/docker-compose.yml" ]]; then
        log_error "docker-compose.yml not found in ${SCRIPT_DIR}"
        return 1
    fi
    local compose_cmd
    if command_exists docker-compose; then compose_cmd="docker-compose"
    elif command_exists docker && docker compose version &>/dev/null; then compose_cmd="docker compose"
    else log_error "No valid Docker Compose command found"; return 1; fi
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log "Stopping existing container..."
        ${compose_cmd} down --timeout "${COMPOSE_TIMEOUT}" || true
    fi
    log "Starting container with ${compose_cmd}..."
    if ! ${compose_cmd} up -d --timeout "${COMPOSE_TIMEOUT}"; then
        log_error "Failed to start containers"
        ${compose_cmd} logs --tail=50
        return 1
    fi
    log "Waiting for container to be ready..."
    local retries=30
    while [[ $retries -gt 0 ]]; do
        if docker exec "${CONTAINER_NAME}" echo "ready" &>/dev/null; then
            log_success "Container is ready"
            return 0
        fi
        sleep 1
        ((retries--))
    done
    log_error "Container failed to become ready"
    return 1
}

run_post_start_scripts() {
    log "Running post-start initialization..."
    local onnx_script="/advantech/init.sh"
    if docker exec "${CONTAINER_NAME}" test -f "$onnx_script" 2>/dev/null; then
        log "Found Initialization script inside container"
        log "Installing ONNX Runtime GPU..."
        if docker exec -it "${CONTAINER_NAME}" bash -c "$onnx_script --force"; then
            log_success "ONNX Runtime GPU installation completed"
        else
            log_warning "ONNX Runtime GPU installation failed - continuing anyway"
        fi
    else
        log_warning "ONNX Runtime GPU installation script not found inside container"
        log_warning "To install later, run inside container: ./init.sh --force"
    fi
}

connect_to_container() {
    log "Connecting to container..."
    echo
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           Container Successfully Started!                      ║${NC}"
    echo -e "${GREEN}╠════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${GREEN}║  Container: ${CONTAINER_NAME}                                  ║${NC}"
    echo -e "${GREEN}║  GPU Support: Enabled                                          ║${NC}"
    echo -e "${GREEN}║  Working Directory: /advantech                                 ║${NC}"
    echo -e "${GREEN}║  Mounted Volumes:                                              ║${NC}"
    echo -e "${GREEN}║    ./src → /app/src                                            ║${NC}"
    echo -e "${GREEN}║    ./models → /app/models                                      ║${NC}"
    echo -e "${GREEN}║    ./data → /app/data                                          ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo
    exec docker exec -it "${CONTAINER_NAME}" bash
}

main() {
    display_banner
    cd "${SCRIPT_DIR}"
    check_prerequisites || exit 1
    init_project_structure
    setup_x11_forwarding
    start_containers || exit 1
    run_post_start_scripts
    connect_to_container
}

main "$@"
