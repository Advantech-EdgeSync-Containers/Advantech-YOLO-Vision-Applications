#!/bin/bash
# Advantech YOLO Vision Applications - Docker Build and Launch Script
# Version: 2.0.0
# Author: Samir Singh <samir.singh@advantech.com>
# Copyright (c) 2024-2025 Advantech Corporation. All rights reserved.
set -euo pipefail

readonly SCRIPT_VERSION="2.0.0"
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CONTAINER_NAME="advantech-yolo-vision"
readonly COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
readonly RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m' CYAN='\033[0;36m' BOLD='\033[1m' NC='\033[0m'

log() { echo -e "${CYAN}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

print_banner() {
    clear
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════════════════════════════╗"
    echo "║     █████╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗ ██████╗██╗  ██╗          ║"
    echo "║    ██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔════╝██║  ██║          ║"
    echo "║    ███████║██║  ██║╚██╗ ██╔╝███████║██╔██╗ ██║   ██║   █████╗  ██║     ███████║          ║"
    echo "║    ██╔══██║██║  ██║ ╚████╔╝ ██╔══██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██╔══██║          ║"
    echo "║    ██║  ██║██████╔╝  ╚██╔╝  ██║  ██║██║ ╚████║   ██║   ███████╗╚██████╗██║  ██║          ║"
    echo "║    ╚═╝  ╚═╝╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝          ║"
    echo "║                         YOLO Vision Applications Container                               ║"
    echo "║                              Center of Excellence                                        ║"
    echo "║                                Version ${SCRIPT_VERSION}                                         ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

show_help() {
    cat << EOF
${BOLD}Advantech YOLO Vision Applications - Build Script${NC}
${BOLD}Usage:${NC} $SCRIPT_NAME [options]
${BOLD}Options:${NC}
    --no-cache     Force pull latest container image
    --detach       Run container in background
    --restart      Restart existing container
    --stop         Stop running container
    --logs         Show container logs
    --status       Show container status
    --shell        Attach to running container
    --help         Show this help message
EOF
}

check_prerequisites() {
    log "Checking prerequisites..."
    command -v docker &>/dev/null || { log_error "Docker is not installed"; exit 1; }
    if command -v docker-compose &>/dev/null; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version &>/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        log_error "Docker Compose is not installed"; exit 1
    fi
    docker info 2>/dev/null | grep -q "nvidia" || log_warning "NVIDIA runtime not detected"
    [[ -f "$COMPOSE_FILE" ]] || { log_error "docker-compose.yml not found"; exit 1; }
    log_success "Prerequisites check passed"
}

create_directories() {
    log "Creating directories..."
    for dir in src models data results diagnostics; do
        [[ ! -d "${SCRIPT_DIR}/${dir}" ]] && mkdir -p "${SCRIPT_DIR}/${dir}"
    done
    log_success "Directories ready"
}

setup_x11() {
    log "Setting up X11..."
    [[ -z "${DISPLAY:-}" ]] && export DISPLAY=:0
    if [[ -z "${XAUTHORITY:-}" ]]; then
        [[ -f "$HOME/.Xauthority" ]] && export XAUTHORITY="$HOME/.Xauthority"
    fi
    if command -v xhost &>/dev/null; then
        xhost +local:docker 2>/dev/null || true
        xhost +local:root 2>/dev/null || true
        if [[ -n "${XAUTHORITY:-}" ]] && [[ -f "$XAUTHORITY" ]]; then
            touch /tmp/.docker.xauth 2>/dev/null || true
            xauth nlist "$DISPLAY" 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge - 2>/dev/null || true
            chmod 644 /tmp/.docker.xauth 2>/dev/null || true
        fi
    fi
    export DISPLAY XAUTHORITY="${XAUTHORITY:-/tmp/.docker.xauth}"
    log_success "X11 ready"
}

is_container_running() { docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; }
is_container_exists() { docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; }

stop_container() {
    log "Stopping container..."
    is_container_running && $COMPOSE_CMD -f "$COMPOSE_FILE" down && log_success "Stopped" || log "Not running"
}

start_container() {
    local detach="${1:-false}"
    log "Starting container..."
    [[ "${NO_CACHE:-false}" == "true" ]] && $COMPOSE_CMD -f "$COMPOSE_FILE" pull
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d
    local retries=30
    while [[ $retries -gt 0 ]]; do
        is_container_running && break
        sleep 1; ((retries--))
    done
    [[ $retries -eq 0 ]] && { log_error "Failed to start"; docker logs "$CONTAINER_NAME" 2>&1 | tail -20; exit 1; }
    log_success "Container started"
    if [[ "$detach" == "false" ]]; then
        log "Waiting for initialization..."
        local elapsed=0
        while [[ $elapsed -lt 120 ]]; do
            docker exec "$CONTAINER_NAME" test -f /tmp/.advantech_initialized 2>/dev/null && break
            echo -n "."; sleep 2; ((elapsed+=2))
        done
        echo ""
        log_success "Ready"
        echo -e "${YELLOW}Environment initialized. Type 'exit' to leave container.${NC}"
        docker exec -it "$CONTAINER_NAME" bash
    else
        log_success "Running in background"
        echo "Attach: docker exec -it ${CONTAINER_NAME} bash"
    fi
}

restart_container() { stop_container; sleep 2; start_container "${1:-false}"; }
show_logs() { docker logs -f "$CONTAINER_NAME"; }
show_status() {
    echo -e "${BOLD}Container Status:${NC}"
    is_container_exists && docker ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}" || echo -e "${RED}Not found${NC}"
}
attach_shell() { is_container_running && docker exec -it "$CONTAINER_NAME" bash || { log_error "Not running"; exit 1; }; }

main() {
    local action="start" detach=false
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-cache) NO_CACHE=true; shift;;
            --detach|-d) detach=true; shift;;
            --restart) action="restart"; shift;;
            --stop) action="stop"; shift;;
            --logs) action="logs"; shift;;
            --status) action="status"; shift;;
            --shell) action="shell"; shift;;
            --help|-h) show_help; exit 0;;
            *) log_error "Unknown: $1"; show_help; exit 1;;
        esac
    done
    [[ "$action" != "logs" ]] && [[ "$action" != "status" ]] && print_banner
    cd "$SCRIPT_DIR"
    case $action in
        start)
            check_prerequisites
            create_directories
            setup_x11
            if is_container_running; then
                log "Already running"
                read -p "Restart? (y/N): " -n 1 -r
                echo ""
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    restart_container "$detach"
                else
                    attach_shell
                fi
            else
                start_container "$detach"
            fi
            ;;
        restart)
            check_prerequisites
            create_directories
            setup_x11
            restart_container "$detach"
            ;;
        stop)
            stop_container
            ;;
        logs)
            show_logs
            ;;
        status)
            show_status
            ;;
        shell)
            attach_shell
            ;;
    esac
}

main "$@"
