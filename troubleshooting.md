# Troubleshooting Guide

## Table of Contents
- [What to do if neither docker-compose nor docker compose commands are available?](#what-to-do-if-neither-docker-compose-nor-docker-compose-commands-are-available)
- [How do I resolve Gstreamer warnings when running in a container?](#how-do-i-resolve-gstreamer-warnings-when-running-in-a-container)
- [How do I fix X11 display issues when running GUI applications in containers?](#how-do-i-fix-x11-display-issues-when-running-gui-applications-in-containers)
- [How do I resolve issues with the NVIDIA Container Toolkit?](#how-do-i-resolve-issues-with-the-nvidia-container-toolkit)
- [Why do I get "permission denied" errors when running Docker commands?](#why-do-i-get-permission-denied-errors-when-running-docker-commands)


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

   **Installation Steps(recommended)**:

   For Docker Compose V2 (recommended):
        1. Ensure you have Docker Engine installed.
        2. Install Docker Compose V2 by following the official [Docker Compose installation guide](https://docs.docker.com/compose/install/).
        3. After installation, verify it with:
 

   ```bash
  docker compose version
   ```   

Also can run :

   ```bash

sudo apt updatesudo apt install ca-certificates curl gnupg lsb-release 

sudo mkdir -p /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update 

sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```


## How do I resolve Gstreamer warnings when running in a container?

   **Issue Description**:  
     When running applications with Gstreamer in containers, you may encounter warning messages related to pipeline initialization or hardware acceleration.

   **Quick Solution**:  
   ```bash
     sudo systemctl restart nvargus-daemon
   ```
   This restarts the NVIDIA camera service that handles hardware access for camera and multimedia operations.

 **Why This Works**:  
     The nvargus-daemon manages hardware resources for camera and video processing. Containers sometimes lose proper communication with this service, causing Gstreamer warnings.

   **Verification**:  
   After restarting the daemon, launch your container again and check if the warnings have disappeared. Your Gstreamer pipelines should now function correctly without warnings.


## How do I fix X11 display issues when running GUI applications in containers?
   **Issue Description**:  
   When attempting to run GUI applications inside Docker containers, you may encounter "Cannot open display" errors or X11 authentication issues.

   **Quick Solution**:  
   ```bash
     # Remove the existing file if it exists
     sudo rm -f /tmp/.docker.xauth
      
     # Create a proper .docker.xauth file
     sudo touch /tmp/.docker.xauth
      
     # Set proper permissions so your user can access it
     sudo chown $(whoami):$(whoami) /tmp/.docker.xauth
     sudo chmod 644 /tmp/.docker.xauth
      
     # Now populate it
     xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge -
   ```

  **Why This Works**:  
  This script creates a proper X11 authentication file that can be shared with the container. It ensures the container has the correct permissions to communicate with the host's X server, allowing GUI applications to display properly.

   **Verification**:  
  After running these commands, launch your container with the appropriate volume mounts for X11 (typically `-v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp/.docker.xauth:/tmp/.docker.xauth -e XAUTHORITY=/tmp/.docker.xauth`). Your GUI applications should now display correctly.



## How do I resolve issues with the NVIDIA Container Toolkit?

   **Issue Description**:  
     If you encounter errors related to the NVIDIA Container Toolkit, it may be due to improper installation or configuration.

   **Quick Solution**:  
      Ensure that the NVIDIA drivers are installed on the host system.
      Verify that the NVIDIA Container Toolkit is installed:
  ```bash
  sudo dpkg -l | grep nvidia-container-toolkit
   ```
   If it's not installed, follow the official [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

   **Why This Works**:  
          The NVIDIA Container Toolkit is essential for enabling GPU access within Docker containers. Proper installation and configuration ensure that containers can utilize the host's GPU resources.


## Why do I get "permission denied" errors when running Docker commands?

  **Issue Description**:  
     When attempting to run Docker commands without `sudo`, you may encounter permission errors like "Got permission denied while trying to connect to the Docker daemon socket" or "docker: permission denied".

   **Quick Solution**:  
     Add your user to the docker group:
  ```bash
 sudo usermod -aG docker $USER
   ```
   Then log out and log back in for the changes to take effect.

   Alternatively, you can continue using Docker with sudo prefix:
  ```bash
 sudo docker <command>
  ```

  **Why This Works**:  
     Docker daemon runs with root privileges. Adding your user to the docker group allows you to run Docker commands without sudo, which is more convenient and safer than running every command with sudo privileges.

   **Verification**:  
     After logging back in, run a simple Docker command without sudo:
   ```bash
   docker ps
  ```
   If it runs without permission errors, you've successfully configured Docker to work without sudo.