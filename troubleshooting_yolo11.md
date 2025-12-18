# Troubleshooting Guide for YOLO11 on JetPack 6

## Table of Contents
- [What to do if neither docker-compose nor docker compose commands are available?](#what-to-do-if-neither-docker-compose-nor-docker-compose-commands-are-available)
- [How do I resolve GStreamer warnings when running in a container?](#how-do-i-resolve-gstreamer-warnings-when-running-in-a-container)
- [How do I fix X11 display issues when running GUI applications in containers?](#how-do-i-fix-x11-display-issues-when-running-gui-applications-in-containers)
- [How do I resolve issues with the NVIDIA Container Toolkit?](#how-do-i-resolve-issues-with-the-nvidia-container-toolkit)
- [Why do I get "permission denied" errors when running Docker commands?](#why-do-i-get-permission-denied-errors-when-running-docker-commands)
- [How do I fix CUDA or TensorRT errors with YOLO11 models?](#how-do-i-fix-cuda-or-tensorrt-errors-with-yolo11-models)
- [Why is my model inference running slowly or on CPU instead of GPU?](#why-is-my-model-inference-running-slowly-or-on-cpu-instead-of-gpu)
- [How do I resolve Ultralytics version compatibility issues?](#how-do-i-resolve-ultralytics-version-compatibility-issues)

---

## What to do if neither docker-compose nor docker compose commands are available?

**Issue Description**:
If you try to run `docker-compose` or `docker compose` commands and receive an error indicating that the command is not found, it means Docker Compose is not installed or not properly configured on your system.

**Quick Solution**:
First, check if you have either command available:
```bash
docker-compose --version
```
or
```bash
docker compose version
```

If neither command works, follow the installation steps below.

**Installation Steps (recommended)**:

For Docker Compose V2 (recommended for JetPack 6):
1. Ensure you have Docker Engine installed.
2. Install Docker Compose V2 by following the official [Docker Compose installation guide](https://docs.docker.com/compose/install/).
3. After installation, verify it with:

```bash
docker compose version
```

Alternatively, install Docker Compose plugin directly:

```bash
sudo apt update
sudo apt install ca-certificates curl gnupg lsb-release

sudo mkdir -p /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update

sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

---

## How do I resolve GStreamer warnings when running in a container?

**Issue Description**:
When running YOLO11 applications with GStreamer in containers, you may encounter warning messages related to pipeline initialization, camera access, or hardware acceleration, particularly when using camera input with `--input 0`.

**Quick Solution**:
```bash
sudo systemctl restart nvargus-daemon
```
This restarts the NVIDIA camera service that handles hardware access for camera and multimedia operations on JetPack 6.

**Why This Works**:
The `nvargus-daemon` manages hardware resources for camera and video processing on Jetson platforms. Containers sometimes lose proper communication with this service, causing GStreamer warnings. JetPack 6 relies on this daemon for camera access, and restarting it re-establishes the connection.

**Additional Steps for Persistent Issues**:
If restarting the daemon doesn't resolve the issue, verify that the container has proper access to video devices:
```bash
# Check if video devices are accessible
ls -l /dev/video*

# Ensure your user is in the video group
sudo usermod -aG video $USER
```

**Verification**:
After restarting the daemon, launch your container again and run your YOLO11 application with camera input:
```bash
python3 src/advantech-yolo.py --input 0 --task detect --model yolo11n.pt --show
```
The GStreamer warnings should now be resolved and camera feed should work correctly.

---

## How do I fix X11 display issues when running GUI applications in containers?

**Issue Description**:
When attempting to run YOLO11 applications with the `--show` flag inside Docker containers, you may encounter "Cannot open display" errors, X11 authentication issues, or blank windows.

**Quick Solution**:
```bash
# Remove the existing file if it exists
sudo rm -f /tmp/.docker.xauth

# Create a proper .docker.xauth file
sudo touch /tmp/.docker.xauth

# Set proper permissions so your user can access it
sudo chown $(whoami):$(whoami) /tmp/.docker.xauth
sudo chmod 644 /tmp/.docker.xauth

# Now populate it with X11 authentication data
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge -
```

**Why This Works**:
This script creates a proper X11 authentication file that can be shared with the container. It ensures the container has the correct permissions to communicate with the host's X server, allowing GUI applications (like YOLO11 visualization windows) to display properly.

**Additional Configuration**:
Ensure your `docker-compose.yml` includes the necessary X11 volume mounts and environment variables:
```yaml
volumes:
  - /tmp/.X11-unix:/tmp/.X11-unix
  - /tmp/.docker.xauth:/tmp/.docker.xauth
environment:
  - DISPLAY=${DISPLAY}
  - XAUTHORITY=/tmp/.docker.xauth
```

**Verification**:
After running these commands, launch your container and test with a YOLO11 application:
```bash
python3 src/advantech-yolo.py --input data/test.mp4 --task detect --model yolo11n.pt --show
```
Your visualization window should now display correctly with real-time detection results.

---

## How do I resolve issues with the NVIDIA Container Toolkit?

**Issue Description**:
If you encounter errors related to GPU access, CUDA, or the NVIDIA Container Toolkit when running YOLO11 inference, it may be due to improper installation or configuration of the toolkit for JetPack 6.

**Quick Solution**:
1. Ensure that the NVIDIA drivers are installed on the host system and JetPack 6 is properly configured.
2. Verify that the NVIDIA Container Toolkit is installed:
```bash
sudo dpkg -l | grep nvidia-container-toolkit
```

3. If it's not installed, install it with:
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

4. Configure Docker to use the NVIDIA runtime:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Why This Works**:
The NVIDIA Container Toolkit is essential for enabling GPU access within Docker containers on JetPack 6. Proper installation and configuration ensure that containers can utilize the host's GPU resources for CUDA and TensorRT acceleration, which is critical for YOLO11 inference performance.

**Verification**:
Test GPU access inside the container:
```bash
# Inside the container
nvidia-smi

# Or check CUDA availability with Python
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If `nvidia-smi` displays GPU information and PyTorch reports CUDA availability, the NVIDIA Container Toolkit is working correctly.

**Additional Resources**:
- [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [JetPack 6 Documentation](https://developer.nvidia.com/embedded/jetpack)

---

## Why do I get "permission denied" errors when running Docker commands?

**Issue Description**:
When attempting to run Docker commands like `./build.sh` or `docker compose up` without `sudo`, you may encounter permission errors like "Got permission denied while trying to connect to the Docker daemon socket" or "docker: permission denied".

**Quick Solution**:
Add your user to the docker group:
```bash
sudo usermod -aG docker $USER
```
Then log out and log back in for the changes to take effect. You may need to reboot for some systems.

Alternatively, you can continue using Docker with sudo prefix:
```bash
sudo docker <command>
sudo ./build.sh
```

**Why This Works**:
Docker daemon runs with root privileges. Adding your user to the docker group allows you to run Docker commands without sudo, which is more convenient and safer than running every command with sudo privileges.

**Verification**:
After logging back in, run a simple Docker command without sudo:
```bash
docker ps
```
If it runs without permission errors, you've successfully configured Docker to work without sudo. You can now run `./build.sh` directly.

---

## How do I fix CUDA or TensorRT errors with YOLO11 models?

**Issue Description**:
When running YOLO11 inference with TensorRT engines or CUDA acceleration, you may encounter errors like "CUDA error", "TensorRT engine build failed", or "Invalid device ordinal".

**Common Causes and Solutions**:

**1. CUDA Out of Memory**:
```bash
# Error: CUDA out of memory
```
**Solution**: Use smaller models or reduce batch size:
```bash
# Use nano model instead of small
python3 src/advantech-yolo.py --input data/test.mp4 --task detect --model yolo11n.pt
```

**2. TensorRT Engine Version Mismatch**:
```bash
# Error: TensorRT engine was built with a different version
```
**Solution**: Re-export the model with the current TensorRT version:
```bash
# Remove old engine files
rm *.engine

# Re-export the model
python3 src/advantech-coe-model-export.py
```

**3. Invalid Device Configuration**:
```bash
# Error: Invalid device ordinal
```
**Solution**: Verify GPU availability and specify correct device:
```bash
# Check available GPUs
python3 -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# Use CPU if GPU is unavailable
python3 src/advantech-yolo.py --input data/test.mp4 --task detect --model yolo11n.pt --device cpu
```

**Why This Works**:
JetPack 6 includes specific versions of CUDA and TensorRT. TensorRT engines are tied to the TensorRT version they were built with and must be regenerated if versions change. Memory management is also critical on edge devices with limited GPU memory.

---

## Why is my model inference running slowly or on CPU instead of GPU?

**Issue Description**:
YOLO11 inference is running slower than expected, or you notice that the model is running on CPU instead of GPU, resulting in poor real-time performance.

**Quick Diagnostic**:
```bash
# Check if CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

**Common Causes and Solutions**:

**1. Model Not Optimized for GPU**:
If using `.pt` models without optimization:
```bash
# Export to TensorRT engine for best performance
python3 src/advantech-coe-model-export.py
```

**2. Device Parameter Not Set**:
```bash
# Explicitly specify GPU device
python3 src/advantech-yolo.py --input data/test.mp4 --task detect --model yolo11n.pt --device 0
```

**3. Using Large Models on Limited Hardware**:
```bash
# Switch to nano model for better real-time performance
python3 src/advantech-yolo.py --input data/test.mp4 --task detect --model yolo11n.pt --device 0
```

**4. NVIDIA Runtime Not Configured**:
Ensure the container is using the NVIDIA runtime:
```bash
# Check docker-compose.yml includes:
runtime: nvidia
```

**Performance Tips**:
- Use TensorRT engine format (`.engine`) for maximum speed
- Use FP16 half-precision for 2x speedup: `--half` flag during export
- Use smaller models (n, s) optimized for edge devices
- Ensure proper cooling for sustained performance

**Verification**:
Monitor GPU utilization during inference:
```bash
# In another terminal
watch -n 1 nvidia-smi
```
You should see GPU utilization increase during model inference.

---

## How do I resolve Ultralytics version compatibility issues?

**Issue Description**:
When running YOLO11 applications, you may encounter import errors, attribute errors, or unexpected behavior due to version incompatibilities with the Ultralytics package.

**Quick Solution**:
Install the specific tested version of Ultralytics:
```bash
pip install ultralytics==8.3.0 --no-deps
```

**Why This Works**:
The YOLO11 toolkit has been tested and validated with Ultralytics version 8.3.0. Using `--no-deps` prevents pip from automatically upgrading or downgrading dependencies that might conflict with JetPack 6's pre-installed packages.

**Common Version-Related Errors**:

**1. Import Error**:
```bash
# Error: cannot import name 'YOLO' from 'ultralytics'
```
**Solution**: Reinstall the correct version:
```bash
pip uninstall ultralytics -y
pip install ultralytics==8.3.0 --no-deps
```

**2. Model Loading Error**:
```bash
# Error: 'YOLO11' model not found
```
**Solution**: Ensure you're using the correct model name format:
```bash
# Correct format for YOLO11
python3 src/advantech-yolo.py --model yolo11n.pt  # Not yolov11n.pt
```

**3. Export Format Not Supported**:
```bash
# Error: Export format 'engine' not supported
```
**Solution**: Verify PyTorch and ONNX versions are compatible:
```bash
pip install torch torchvision --no-deps
pip install onnx onnxruntime --no-deps
```

**Verification**:
Test the installation:
```bash
python3 -c "from ultralytics import YOLO; print(YOLO.__version__)"
```

**Additional Notes**:
- Always check the [Acknowledgments](README_YOLO11.md#acknowledgments) section in the README for required package versions
- Avoid using `pip install --upgrade` as it may break compatibility with JetPack 6
- If you encounter persistent issues, rebuild the Docker container to ensure a clean environment

---

## Additional Resources

For further assistance:
- [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/)
- [NVIDIA JetPack 6 Documentation](https://developer.nvidia.com/embedded/jetpack)
- [Advantech Edge AI Solutions](https://www.advantech.com/en/products/intelligent-systems)
- [GitHub Issues](https://github.com/Advantech-EdgeSync-Containers/Advantech-YOLO-Vision-Applications/issues)

If your issue persists after trying these troubleshooting steps, please open an issue on the GitHub repository with:
- Detailed error messages
- Your device model and JetPack version
- Steps to reproduce the issue
- Output of `nvidia-smi` and `docker info`
