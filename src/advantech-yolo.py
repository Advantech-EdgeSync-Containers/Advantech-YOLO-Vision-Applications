#!/usr/bin/env python3

# Enhanced YOLO Application with Hardware Acceleration

# Maintains core detection and segmentation functionality from original app

# Adds better hardware acceleration and fixed classification



import sys

import os

import time

import shutil

import signal

import subprocess

import gc

from pathlib import Path



# Add system packages explicitly to path

print("== Setting up system module paths ==")

system_paths = [

    '/usr/lib/python3/dist-packages',

    '/usr/local/lib/python3.8/dist-packages'

]



for path in system_paths:

    if os.path.exists(path) and path not in sys.path:

        sys.path.insert(0, path)

        print(f"Added {path} to sys.path")



# Test six import

try:

    import six

    print(f"Found six module: version {six.__version__} at {six.__file__}")

except ImportError:

    print("❌ Cannot find six module despite adding system paths!")

    sys.exit(1)



# Fix NumPy compatibility issues

def fix_numpy_path():

    """Ensure we're using a compatible NumPy version and patch missing attributes"""

    print("== Fixing NumPy paths and compatibility issues ==")

    

    # Force newer NumPy to be first in path

    dist_packages = '/usr/local/lib/python3.8/dist-packages'

    if os.path.exists(dist_packages):

        if dist_packages in sys.path:

            sys.path.remove(dist_packages)

        sys.path.insert(0, dist_packages)

        print(f"Added dist-packages to beginning of path: {dist_packages}")

    

    # Force NumPy to reload if already imported

    if 'numpy' in sys.modules:

        print("NumPy already imported, forcing reload")

        del sys.modules['numpy']

        if 'numpy.random' in sys.modules:

            del sys.modules['numpy.random']

    

    try:

        # Now import NumPy and check

        import numpy as np

        print(f"Using NumPy {np.__version__} from {np.__file__}")

        

        # Fix BitGenerator if missing

        patched = False

        if not hasattr(np.random, 'BitGenerator'):

            print("❌ NumPy is missing BitGenerator attribute.")

            print("Adding BitGenerator mock...")

            

            # If BitGenerator is missing, try to add it

            class DummyBitGenerator:

                def __init__(self, seed=None):

                    self.seed = seed

            

            # Add BitGenerator to numpy.random

            np.random.BitGenerator = DummyBitGenerator

            print("Added dummy BitGenerator to numpy.random")

            patched = True

        

        # Fix mtrand attribute if missing

        if not hasattr(np.random, 'mtrand'):

            print("❌ NumPy is missing mtrand attribute.")

            print("Adding mtrand mock...")

            

            # Create a mock mtrand module with _rand attribute

            class DummyRand:

                def __init__(self):

                    pass

                

                def __call__(self, *args, **kwargs):

                    return np.random.random(*args, **kwargs)

            

            # Create mock mtrand module

            class DummyMtrand:

                def __init__(self):

                    self._rand = DummyRand()

            

            # Add mtrand to numpy.random

            np.random.mtrand = DummyMtrand()

            print("Added dummy mtrand to numpy.random")

            patched = True

        

        # Check if patches were successful

        if patched:

            has_bitgen = hasattr(np.random, 'BitGenerator')

            has_mtrand = hasattr(np.random, 'mtrand')

            print(f"NumPy patched: BitGenerator={has_bitgen}, mtrand={has_mtrand}")

            return True

        else:

            print("✅ NumPy has all required attributes, no patching needed")

            return True

            

    except ImportError as e:

        print(f"❌ Error importing NumPy: {e}")

        return False



# Test if hardware acceleration is available

def test_hardware_acceleration():

    print("== Testing hardware acceleration capabilities ==")

    

    # Check for V4L2 codecs

    try:

        result = subprocess.run(

            ['v4l2-ctl', '--list-devices'],

            capture_output=True, text=True, check=False

        )

        if result.returncode == 0:

            print("V4L2 devices found:")

            print(result.stdout.strip())

            v4l2_available = True

        else:

            print("V4L2 devices not found or v4l2-ctl not available")

            v4l2_available = False

    except Exception as e:

        print(f"Error checking V4L2: {e}")

        v4l2_available = False

    

    # Check for NVDEC/NVENC

    try:

        # Look for Gstreamer NVIDIA plugins

        result = subprocess.run(

            ['gst-inspect-1.0', 'nvv4l2decoder'],

            capture_output=True, text=True, check=False

        )

        if "Plugin Details" in result.stdout:

            print("✅ NVIDIA hardware decoder (nvv4l2decoder) is available")

            nvdec_available = True

        else:

            print("NVIDIA hardware decoder not found")

            nvdec_available = False

            

        # Check for encoder

        result = subprocess.run(

            ['gst-inspect-1.0', 'nvv4l2h264enc'],

            capture_output=True, text=True, check=False

        )

        if "Plugin Details" in result.stdout:

            print("✅ NVIDIA hardware encoder (nvv4l2h264enc) is available")

            nvenc_available = True

        else:

            print("NVIDIA hardware encoder not found")

            nvenc_available = False

    except Exception as e:

        print(f"Error checking NVIDIA codecs: {e}")

        nvdec_available = False

        nvenc_available = False

    

    # Check for FFmpeg hwaccel

    ffmpeg_hwaccels = []

    try:

        result = subprocess.run(

            ['ffmpeg', '-hwaccels'],

            capture_output=True, text=True, check=False

        )

        if result.returncode == 0:

            hwaccels = [line.strip() for line in result.stdout.split('\n') if line.strip() and "Hardware acceleration methods:" not in line]

            if hwaccels:

                print("FFmpeg hardware acceleration available:")

                for hwaccel in hwaccels:

                    print(f" - {hwaccel}")

                ffmpeg_hwaccel = True

                ffmpeg_hwaccels = hwaccels

            else:

                print("No FFmpeg hardware acceleration found")

                ffmpeg_hwaccel = False

        else:

            print("Error checking FFmpeg hardware acceleration")

            ffmpeg_hwaccel = False

    except Exception as e:

        print(f"Error checking FFmpeg: {e}")

        ffmpeg_hwaccel = False

        

    return {

        "v4l2": v4l2_available,

        "nvdec": nvdec_available,

        "nvenc": nvenc_available,

        "ffmpeg_hwaccel": ffmpeg_hwaccel,

        "ffmpeg_hwaccels": ffmpeg_hwaccels

    }



# Configure paths and fix imports before anything else

def setup_environment():

    print("== Setting up environment ==")

    

    # Clear any __pycache__ directories which might have old imports cached

    for root, dirs, files in os.walk('.'):

        for d in dirs:

            if d == '__pycache__':

                cache_dir = os.path.join(root, d)

                print(f"Removing cache directory: {cache_dir}")

                shutil.rmtree(cache_dir)

    

    # Remove conflicting OpenCV paths

    cv2_paths = [p for p in sys.path if 'cv2' in p]

    for p in cv2_paths:

        if p in sys.path:

            sys.path.remove(p)

            print(f"Removed potentially conflicting CV2 path: {p}")

    

    # Fix NumPy path

    if not fix_numpy_path():

        print("❌ Failed to fix NumPy path. Cannot continue.")

        return False

    

    # Set CUDA memory allocation variables - critical for Jetson

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'  # Limit memory chunk size

    os.environ['CUDA_CACHE_DISABLE'] = '0'  # Enable CUDA cache

    

    # Library paths

    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')

    

    # For display capability - ENABLE display (don't set DISPLAY to empty)

    if 'DISPLAY' not in os.environ:

        os.environ['DISPLAY'] = ':0'  # Try to use the default display

    

    # Disable watchdog timer for NVIDIA driver to prevent timeout issues

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    

    # Set XDG_RUNTIME_DIR for ffplay

    os.environ['XDG_RUNTIME_DIR'] = '/tmp'

    

    # Enable direct import of system modules

    os.environ['PYTHONPATH'] = '/usr/lib/python3/dist-packages:' + os.environ.get('PYTHONPATH', '')

    

    # Print the updated Python path

    print("\nFinal Python path:")

    for p in sys.path:

        print(f"  {p}")

    

    try:

        # Create missing functionality in torch.distributed if needed

        import torch

        if not hasattr(torch.distributed, 'is_initialized'):

            torch.distributed.is_initialized = lambda: False

            print("Patched torch.distributed.is_initialized")

        

        # Apply torch optimization settings

        torch.backends.cudnn.benchmark = True

        torch.backends.cudnn.enabled = True

    except ImportError:

        print("Warning: Could not import torch")

    

    # Test hardware acceleration

    hw_accel = test_hardware_acceleration()

    

    print("✅ Environment setup complete")

    return hw_accel



# Register signal handler to handle Ctrl+C gracefully

def signal_handler(sig, frame):

    print("\nReceived interrupt. Cleaning up and exiting...")

    sys.exit(0)



signal.signal(signal.SIGINT, signal_handler)



# Pre-process the video using FFmpeg with hardware acceleration

def preprocess_video(input_file, output_file, scale=None, fps=None, hw_accel=None):

    """Pre-process video using hardware acceleration"""

    print(f"Pre-processing video: {input_file} -> {output_file}")

    

    # Base command

    cmd = ['ffmpeg', '-y']

    

    # Add hardware acceleration if available

    if hw_accel and hw_accel.get('ffmpeg_hwaccel', False):

        # Get list of available accelerators

        hwaccels = hw_accel.get('ffmpeg_hwaccels', [])

        

        # Try with hardware decoding if available

        if 'vaapi' in hwaccels:

            cmd.extend(['-hwaccel', 'vaapi', '-hwaccel_output_format', 'vaapi'])

        elif 'nvdec' in hwaccels:

            cmd.extend(['-hwaccel', 'nvdec'])

        elif 'cuda' in hwaccels:

            cmd.extend(['-hwaccel', 'cuda'])

        elif 'vdpau' in hwaccels:

            cmd.extend(['-hwaccel', 'vdpau'])

        elif 'drm' in hwaccels:

            cmd.extend(['-hwaccel', 'drm'])

    

    # Input file

    cmd.extend(['-i', input_file])

    

    # Scaling if requested

    if scale:

        width, height = scale

        cmd.extend(['-vf', f'scale={width}:{height}'])

    

    # Framerate if requested

    if fps:

        cmd.extend(['-r', str(fps)])

    

    # Use hardware encoding if available

    if hw_accel and hw_accel.get('nvenc', False):

        # Check if h264_nvenc is available

        try:

            check_cmd = ['ffmpeg', '-hide_banner', '-encoders']

            result = subprocess.run(check_cmd, capture_output=True, text=True, check=False)

            if 'h264_nvenc' in result.stdout:

                cmd.extend(['-c:v', 'h264_nvenc', '-preset', 'p4', '-tune', 'fastdecode', 

                            '-b:v', '2M', '-maxrate', '2M', '-bufsize', '2M', '-pix_fmt', 'yuv420p'])

            else:

                print("h264_nvenc encoder not available, falling back to h264_v4l2m2m")

                if hw_accel.get('v4l2', False):

                    cmd.extend(['-c:v', 'h264_v4l2m2m', '-b:v', '2M', '-pix_fmt', 'yuv420p'])

                else:

                    cmd.extend(['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-pix_fmt', 'yuv420p'])

        except Exception as e:

            print(f"Error checking for h264_nvenc: {e}")

            cmd.extend(['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-pix_fmt', 'yuv420p'])

    elif hw_accel and hw_accel.get('v4l2', False):

        # Try hardware encoding with V4L2

        cmd.extend(['-c:v', 'h264_v4l2m2m', '-b:v', '2M', '-pix_fmt', 'yuv420p'])

    else:

        # Software encoding fallback

        cmd.extend(['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-pix_fmt', 'yuv420p'])

    

    # Output file

    cmd.append(output_file)

    

    print(f"Running command: {' '.join(cmd)}")

    try:

        process = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if process.returncode != 0:

            print(f"❌ FFmpeg preprocessing failed: {process.stderr}")

            return False

        else:

            print("✅ Video preprocessing complete")

            return True

    except Exception as e:

        print(f"❌ Error during preprocessing: {e}")

        return False



# Post-process results using FFmpeg with hardware acceleration

def create_video_from_frames(frame_dir, output_file, fps=30, hw_accel=None):

    """Create video from frames using hardware acceleration"""

    print(f"Creating video from frames in {frame_dir} -> {output_file}")

    

    # Ensure frames directory exists

    if not os.path.exists(frame_dir):

        print(f"❌ Frames directory not found: {frame_dir}")

        return False

    

    # Count frames to ensure we have something to work with

    frames = sorted(list(Path(frame_dir).glob("*.jpg")))

    if not frames:

        print(f"❌ No frames found in {frame_dir}")

        return False

    

    print(f"Found {len(frames)} frames to process")

    

    # Base command

    cmd = ['ffmpeg', '-y']

    

    # Input frames pattern

    first_frame = frames[0].name

    frame_pattern = first_frame.split('_')[0] + "_%06d.jpg"

    cmd.extend(['-framerate', str(fps), '-i', f"{frame_dir}/{frame_pattern}"])

    

    # Use hardware encoding if available

    if hw_accel and hw_accel.get('nvenc', False):

        # Check if h264_nvenc is available

        try:

            check_cmd = ['ffmpeg', '-hide_banner', '-encoders']

            result = subprocess.run(check_cmd, capture_output=True, text=True, check=False)

            if 'h264_nvenc' in result.stdout:

                cmd.extend(['-c:v', 'h264_nvenc', '-preset', 'p4', '-tune', 'fastdecode', 

                            '-b:v', '2M', '-maxrate', '2M', '-bufsize', '2M', '-pix_fmt', 'yuv420p'])

            else:

                print("h264_nvenc encoder not available, falling back to h264_v4l2m2m")

                if hw_accel.get('v4l2', False):

                    cmd.extend(['-c:v', 'h264_v4l2m2m', '-b:v', '2M', '-pix_fmt', 'yuv420p'])

                else:

                    cmd.extend(['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-pix_fmt', 'yuv420p'])

        except Exception as e:

            print(f"Error checking for h264_nvenc: {e}")

            cmd.extend(['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-pix_fmt', 'yuv420p'])

    elif hw_accel and hw_accel.get('v4l2', False):

        # Try hardware encoding with V4L2

        cmd.extend(['-c:v', 'h264_v4l2m2m', '-b:v', '2M', '-pix_fmt', 'yuv420p'])

    else:

        # Software encoding fallback

        cmd.extend(['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-pix_fmt', 'yuv420p'])

    

    # Output file

    cmd.append(output_file)

    

    print(f"Running command: {' '.join(cmd)}")

    try:

        process = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if process.returncode != 0:

            print(f"❌ FFmpeg video creation failed: {process.stderr}")

            return False

        else:

            print("✅ Video creation complete")

            return True

    except Exception as e:

        print(f"❌ Error during video creation: {e}")

        return False



# YOLO Memory-Optimized Processing Class for Jetson

class YOLOHWAccelerated:

    def __init__(self, hw_acceleration=None):

        print("\n== Initializing YOLO with Hardware Acceleration ==")

        self.hw_accel = hw_acceleration or {}

        

        # Import dependencies

        try:

            import numpy as np

            self.np = np

            print(f"Using NumPy {np.__version__}")

            

            import torch

            self.torch = torch

            

            # Configure PyTorch for better memory usage on Jetson

            torch.backends.cudnn.benchmark = True  # Use cuDNN autotuner

            torch.backends.cudnn.enabled = True    # Enable cuDNN

            

            # Print CUDA information

            print(f"Using PyTorch {torch.__version__}")

            print(f"CUDA available: {torch.cuda.is_available()}")

            if torch.cuda.is_available():

                print(f"CUDA device: {torch.cuda.get_device_name(0)}")

                print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024 / 1024:.2f} MB")

                print(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024 / 1024:.2f} MB")

            

            # Import OpenCV with correct settings

            try:

                # First, temporarily clear path to avoid recursive imports

                old_path = sys.path.copy()

                sys.path = ['/usr/lib/python3.8', '/usr/local/lib/python3.8/dist-packages']

                

                import cv2

                self.cv2 = cv2

                print(f"Using OpenCV {cv2.__version__}")

                

                # Check for GPU acceleration in OpenCV

                if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:

                    print(f"✅ OpenCV CUDA acceleration available: {cv2.cuda.getCudaEnabledDeviceCount()} devices")

                    self.opencv_cuda = True

                else:

                    print("OpenCV CUDA acceleration not available")

                    self.opencv_cuda = False

                

                # Try to check display availability

                try:

                    test_window_name = "TestWindow"

                    cv2.namedWindow(test_window_name, cv2.WINDOW_NORMAL)

                    cv2.resizeWindow(test_window_name, 320, 240)

                    cv2.imshow(test_window_name, np.zeros((240, 320, 3), dtype=np.uint8))

                    cv2.waitKey(1)  # Only wait for 1ms

                    cv2.destroyWindow(test_window_name)

                    print("✅ Display is available! Real-time visualization will be enabled.")

                    self.display_available = True

                except Exception as disp_e:

                    print(f"❌ Display test failed: {disp_e}")

                    print("Real-time visualization will be disabled.")

                    self.display_available = False

                

                # Restore path

                sys.path = old_path

            except ImportError as e:

                print(f"❌ OpenCV import error: {e}")

                print("Trying alternative OpenCV import method...")

                self.display_available = False

                self.opencv_cuda = False

                

                # Try alternative approach

                if 'cv2' in sys.modules:

                    del sys.modules['cv2']

                

                import ctypes

                try:

                    ctypes.CDLL("libopencv_core.so.4.5")

                    # Try importing again

                    import cv2

                    self.cv2 = cv2

                    print(f"Using OpenCV {cv2.__version__} (loaded via alternative method)")

                except Exception as cv_e:

                    print(f"❌ Failed to load OpenCV: {cv_e}")

                    print("Will continue without OpenCV but functionality will be limited")

                    self.cv2 = None

            

            # Import ultralytics with proper path settings

            old_sys_path = list(sys.path)

            

            # Temporarily prepend system paths to start of sys.path

            for path in ['/usr/lib/python3/dist-packages', '/usr/local/lib/python3.8/dist-packages']:

                if path not in sys.path:

                    sys.path.insert(0, path)

            

            try:

                from ultralytics import YOLO

                self.YOLO = YOLO

                import ultralytics

                print(f"Using Ultralytics {ultralytics.__version__}")

                

                # Store the ultralytics version

                self.ultralytics_version = ultralytics.__version__

                

                # Configure ultralytics for performance

                from ultralytics.yolo.utils import ops

                print("Configured ultralytics for Jetson optimization")

            except ImportError as e:

                print(f"❌ Error importing ultralytics: {e}")

                import traceback

                traceback.print_exc()

                raise

            finally:

                # Restore original sys.path

                sys.path = old_sys_path

            

            self.models = {}

            

        except ImportError as e:

            print(f"❌ Error importing dependencies: {e}")

            import traceback

            traceback.print_exc()

            raise

    

    def load_model(self, model_type='detection'):

        """Load a YOLO model for detection, segmentation, or classification"""

        if model_type in self.models:

            print(f"Model for {model_type} already loaded")

            return self.models[model_type]

        

        # Use nano models (smallest) for better performance on Jetson

        model_paths = {

            'detection': 'yolov8n.pt',        # Nano detection model

            'detection-s': 'yolov8s.pt',      # Small detection model

            'segmentation': 'yolov8n-seg.pt', # Nano segmentation model

            'classification': 'yolov8n-cls.pt' # Nano classification model

        }

        

        if model_type not in model_paths:

            print(f"❌ Invalid model type: {model_type}")

            print(f"Available types: {', '.join(model_paths.keys())}")

            return None

        

        model_path = model_paths[model_type]

        

        # Download the model if needed

        if not os.path.exists(model_path):

            print(f"Downloading {model_path}...")

            # Create progress indicator

            print("[", end="", flush=True)

            

            def download_progress(block_num, block_size, total_size):

                downloaded = block_num * block_size

                percentage = int((downloaded / total_size) * 30)

                sys.stdout.write("\r[" + "=" * percentage + " " * (30 - percentage) + f"] {percentage*3}%")

                sys.stdout.flush()

            

            try:

                import urllib.request

                url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_path}"

                urllib.request.urlretrieve(url, model_path, reporthook=download_progress)

                print("\n✅ Download complete")

            except Exception as e:

                print(f"\n❌ Download failed: {e}")

                return None

        

        # Load the model with memory optimization for Jetson

        try:

            print(f"Loading {model_type} model from {model_path}...")

            device = 0 if self.torch.cuda.is_available() else 'cpu'

            

            # Force garbage collection before loading model

            gc.collect()

            

            # For CUDA devices

            if device == 0:

                self.torch.cuda.empty_cache()

                print(f"Pre-load CUDA memory: {self.torch.cuda.memory_allocated(0) / 1024 / 1024:.2f} MB allocated")

            

            # Load with half precision for better performance on Jetson

            model = self.YOLO(model_path)

            

            # Store the model

            self.models[model_type] = model

            

            # Print memory usage after loading

            if device == 0:

                print(f"Post-load CUDA memory: {self.torch.cuda.memory_allocated(0) / 1024 / 1024:.2f} MB allocated")

            

            print(f"✅ {model_type.capitalize()} model loaded successfully on {'CUDA' if device == 0 else 'CPU'}")

            return model

        except Exception as e:

            print(f"❌ Error loading model: {e}")

            import traceback

            traceback.print_exc()

            return None

    

    def process_video_with_hwaccel(self, source, mode='detection', conf=0.25, 

                                  output_dir="./results", frame_skip=0, 

                                  preprocess=True, max_frames=None,

                                  batch_size=4, enable_display=True):

        """Process video with hardware acceleration pipeline with optional real-time display"""

        # Select model

        if mode == 'detection':

            model = self.load_model('detection')

        elif mode == 'segmentation':

            model = self.load_model('segmentation')

        elif mode == 'classification':

            model = self.load_model('classification')

        else:

            print(f"❌ Invalid mode: {mode}")

            return None

        

        if not model:

            return None

        

        source_path = Path(source)

        if not source_path.exists():

            print(f"❌ File not found: {source}")

            return None

        

        # Create output directory

        if not os.path.exists(output_dir):

            os.makedirs(output_dir)

            print(f"Created output directory: {output_dir}")

        

        # Create frames directory

        frames_dir = f"{output_dir}/frames_{mode}_{source_path.stem}"

        if not os.path.exists(frames_dir):

            os.makedirs(frames_dir)

        

        print(f"\nProcessing video: {source} with {mode} mode...")

        

        # Step 1: Pre-process the video with hardware acceleration if enabled

        if preprocess:

            temp_video = f"{output_dir}/temp_{source_path.stem}.mp4"

            if not preprocess_video(source, temp_video, hw_accel=self.hw_accel):

                print("Skipping pre-processing, using original video...")

                temp_video = source

        else:

            temp_video = source

        

        # Step 2: Process the video

        try:

            # Configure hardware device for inference

            device = 0 if self.torch.cuda.is_available() else 'cpu'

            

            # Start timing

            start_time = time.time()

            

            # Open the video

            cap = self.cv2.VideoCapture(temp_video)

            if not cap.isOpened():

                print(f"❌ Failed to open video: {temp_video}")

                return None

            

            # Get video properties

            width = int(cap.get(self.cv2.CAP_PROP_FRAME_WIDTH))

            height = int(cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT))

            fps = cap.get(self.cv2.CAP_PROP_FPS)

            total_frames = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))

            

            # Limit frames if specified

            if max_frames and max_frames > 0:

                total_frames = min(total_frames, max_frames)

                print(f"Processing limited to {max_frames} frames")

            

            print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

            

            # Initialize display window if enabled and available

            display_enabled = enable_display and self.display_available

            if display_enabled:

                window_name = f"YOLO {mode.capitalize()} - {source_path.name}"

                self.cv2.namedWindow(window_name, self.cv2.WINDOW_NORMAL)

                

                # Calculate window size to maintain aspect ratio

                if width > height:

                    display_width = min(1280, width)

                    display_height = int((display_width / width) * height)

                else:

                    display_height = min(720, height)

                    display_width = int((display_height / height) * width)

                

                self.cv2.resizeWindow(window_name, display_width, display_height)

                print(f"✅ Display window created with size {display_width}x{display_height}")

            

            # Process frames

            frame_count = 0

            processed_count = 0

            saved_frames = 0

            

            # Statistics

            total_objects = 0

            class_counts = {}

            processing_times = []

            

            print("\nProcessing frames...")

            

            # Batch processing for better performance

            batch_frames = []

            batch_indices = []

            

            while frame_count < total_frames:

                # Track progress

                if frame_count % 10 == 0:

                    percent = int((frame_count / total_frames) * 100)

                    elapsed = time.time() - start_time

                    if processed_count > 0 and elapsed > 0:

                        fps_achieved = processed_count / elapsed

                        eta_seconds = (total_frames - frame_count) / fps_achieved if fps_achieved > 0 else 0

                        eta_min = eta_seconds / 60

                        print(f"\rProgress: {percent}% ({frame_count}/{total_frames}) | "

                              f"Speed: {fps_achieved:.1f} FPS | ETA: {eta_min:.1f} min", end="", flush=True)

                

                # Read the frame

                ret, frame = cap.read()

                if not ret:

                    break

                

                frame_count += 1

                

                # Apply frame skipping if enabled

                if frame_skip > 0 and (frame_count % (frame_skip + 1)) != 1:

                    continue

                

                # Add frame to batch

                batch_frames.append(frame)

                batch_indices.append(frame_count)

                

                # Process when batch is full or at end of video

                if len(batch_frames) >= batch_size or frame_count == total_frames:

                    try:

                        # Process batch

                        batch_start = time.time()

                        

                        results = model.predict(

                            source=batch_frames,

                            conf=conf,

                            verbose=False,

                            stream=False,

                            device=device,

                            half=True  # Use half precision for better performance

                        )

                        

                        batch_time = time.time() - batch_start

                        processing_times.append(batch_time / len(batch_frames))

                        

                        # Process and save each result

                        for i, r in enumerate(results):

                            processed_count += 1

                            

                            # Extract detection results

                            if mode == 'detection':

                                # Extract detection results

                                if hasattr(r, 'boxes'):

                                    boxes = r.boxes

                                    objects_in_frame = len(boxes)

                                    total_objects += objects_in_frame

                                    

                                    # Update class counts

                                    for box in boxes:

                                        cls = int(box.cls[0])

                                        class_name = model.names[cls]

                                        

                                        if class_name in class_counts:

                                            class_counts[class_name] += 1

                                        else:

                                            class_counts[class_name] = 1

                            

                            elif mode == 'segmentation':

                                # Extract segmentation results

                                if hasattr(r, 'masks') and r.masks is not None:

                                    objects_in_frame = len(r.masks)

                                    total_objects += objects_in_frame

                                    

                                    # Update class counts from associated boxes

                                    if hasattr(r, 'boxes'):

                                        for j in range(min(len(r.boxes), objects_in_frame)):

                                            cls = int(r.boxes[j].cls[0])

                                            class_name = model.names[cls]

                                            

                                            if class_name in class_counts:

                                                class_counts[class_name] += 1

                                            else:

                                                class_counts[class_name] = 1

                            

                            elif mode == 'classification':

                                # Handle classification (fixed for the version issue)

                                if hasattr(r, 'probs'):

                                    probs = r.probs

                                    if probs is not None:

                                        if isinstance(probs, self.torch.Tensor):

                                            # Direct tensor handling - find max value

                                            if len(probs.shape) > 0:

                                                max_idx = probs.argmax().item()

                                                confidence = probs[max_idx].item()

                                                class_name = model.names[max_idx]

                                                

                                                if class_name in class_counts:

                                                    class_counts[class_name] += 1

                                                else:

                                                    class_counts[class_name] = 1

                                                

                                                total_objects += 1

                                        elif hasattr(probs, 'top1'):

                                            # Handle case with top1 attribute

                                            top_idx = int(probs.top1)

                                            class_name = model.names[top_idx]

                                            

                                            if class_name in class_counts:

                                                class_counts[class_name] += 1

                                            else:

                                                class_counts[class_name] = 1

                                            

                                            total_objects += 1

                            

                            # Get annotated frame and save

                            annotated_frame = r.plot()

                            frame_index = batch_indices[i]

                            

                            # Save frame

                            frame_path = f"{frames_dir}/frame_{frame_index:06d}.jpg"

                            self.cv2.imwrite(frame_path, annotated_frame)

                            saved_frames += 1

                            

                            # Display the frame if enabled

                            if display_enabled:

                                # Add progress text to the displayed frame

                                progress_text = f"Progress: {int((frame_count / total_frames) * 100)}% | FPS: {1/processing_times[-1]:.1f}"

                                self.cv2.putText(annotated_frame, progress_text, (10, 30), 

                                                self.cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                                

                                # Display the frame

                                self.cv2.imshow(window_name, annotated_frame)

                                

                                # Process UI events and check for exit key

                                key = self.cv2.waitKey(1) & 0xFF

                                if key == 27 or key == ord('q'):  # ESC or q key

                                    print("\n⚠️ Processing interrupted by user")

                                    break

                        

                        # Clear batch

                        batch_frames = []

                        batch_indices = []

                        

                        # Periodic memory cleanup

                        if processed_count % 100 == 0:

                            gc.collect()

                            if device == 0:

                                self.torch.cuda.empty_cache()

                    except Exception as e:

                        print(f"\n❌ Error processing batch: {e}")

                        import traceback

                        traceback.print_exc()

                        # Continue with next batch

                        batch_frames = []

                        batch_indices = []

            

            # Cleanup

            cap.release()

            

            # Close display window if opened

            if display_enabled:

                self.cv2.destroyAllWindows()

            

            # Final cleanup

            gc.collect()

            if device == 0:

                self.torch.cuda.empty_cache()

            

            # Step 3: Create final video from frames using FFmpeg with hardware encoding

            output_video = f"{output_dir}/{mode}_{source_path.stem}.mp4"

            print(f"\nCreating final video with FFmpeg...")

            if saved_frames > 0:

                # Calculate the effective frame rate after skipping

                effective_fps = fps

                if frame_skip > 0:

                    effective_fps = fps / (frame_skip + 1)

                create_video_from_frames(frames_dir, output_video, fps=effective_fps, hw_accel=self.hw_accel)

            else:

                print("❌ No frames were processed or saved")

            

            # Print statistics

            end_time = time.time()

            total_time = end_time - start_time

            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

            

            print(f"\n\nVideo Processing Complete:")

            print(f"Processed {processed_count} frames in {total_time:.2f} seconds")

            print(f"Total frames in video: {total_frames}")

            print(f"Frame skip: Every {frame_skip + 1} frames")

            print(f"Average processing time per frame: {avg_processing_time*1000:.1f} ms")

            print(f"Effective FPS: {processed_count/total_time:.2f}")

            

            print(f"\nTotal objects detected: {total_objects}")

            

            if class_counts:

                print("\nObjects by class:")

                for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):

                    print(f" - {cls}: {count}")

            

            # Save summary

            summary_file = f"{output_dir}/{mode}_{source_path.stem}_summary.txt"

            with open(summary_file, 'w') as f:

                f.write(f"Summary for {source}\n")

                f.write(f"Mode: {mode}\n")

                f.write(f"Frames processed: {processed_count}\n")

                f.write(f"Processing time: {total_time:.2f} seconds\n")

                f.write(f"Effective FPS: {processed_count/total_time:.2f}\n\n")

                f.write(f"Total objects detected: {total_objects}\n\n")

                f.write("Objects by class:\n")

                for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):

                    f.write(f" - {cls}: {count}\n")

            

            print(f"\nResults saved to: {output_dir}")

            print(f"Output video: {output_video}")

            print(f"Summary file: {summary_file}")

            

            return True

        except Exception as e:

            print(f"\n❌ Error processing video: {e}")

            import traceback

            traceback.print_exc()

            return None

        finally:

            # Ensure cleanup

            try:

                if 'cap' in locals() and cap is not None:

                    cap.release()

                

                # Close any open windows

                if self.display_available:

                    self.cv2.destroyAllWindows()

                

                # Clean up temporary files if they exist and are not the source

                if preprocess and 'temp_video' in locals() and temp_video != source and os.path.exists(temp_video):

                    print(f"Cleaning up temporary file: {temp_video}")

                    os.remove(temp_video)

            except Exception as cleanup_error:

                print(f"Error during cleanup: {cleanup_error}")



# Add a new option to play the video with ffplay

def play_video_with_ffplay(video_path):

    """Play a video file using ffplay"""

    if not os.path.exists(video_path):

        print(f"❌ Video file not found: {video_path}")

        return False

    

    try:

        print(f"Playing video: {video_path}")

        cmd = [

            'ffplay', 

            '-autoexit',      # Exit when the video ends

            '-loglevel', 'error',  # Only show errors

            '-window_title', f"Playing: {os.path.basename(video_path)}",

            video_path

        ]

        

        print(f"Running command: {' '.join(cmd)}")

        process = subprocess.Popen(cmd)

        

        print("Press Ctrl+C to stop playback")

        process.wait()

        return True

    except KeyboardInterrupt:

        print("\nPlayback interrupted by user")

        if process.poll() is None:  # If process is still running

            process.terminate()

        return True

    except Exception as e:

        print(f"❌ Error playing video: {e}")

        return False



# Interactive menu

def show_menu():

    print("\n" + "="*50)

    print("        YOLO HARDWARE ACCELERATED PROCESSOR")

    print("="*50)

    print("1. Process Video (Detection - Default with Real-time Display)")

    print("2. Process Video (Detection - High Performance with Real-time Display)")

    print("3. Process Video (Segmentation with Real-time Display)")

    print("4. Process Video (Classification with Real-time Display)")

    print("5. Play Last Processed Video")

    print("6. Test Hardware Acceleration")

    print("7. Exit")

    choice = input("\nEnter your choice (1-7): ").strip()

    return choice



# Main function

def main():

    # Setup environment and test hardware acceleration

    hw_accel = setup_environment()

    

    # Initialize the processor

    try:

        processor = YOLOHWAccelerated(hw_accel)

    except Exception as e:

        print(f"❌ Failed to initialize YOLO processor: {e}")

        import traceback

        traceback.print_exc()

        return

    

    # Set up the results directory

    results_dir = "./results"

    if not os.path.exists(results_dir):

        os.makedirs(results_dir)

        print(f"Created results directory: {results_dir}")

    

    # Track the last processed video

    last_processed_video = None

    

    while True:

        choice = show_menu()

        

        if choice == '1':

            # Detection - Default

            source = input("\nEnter video path: ").strip()

            if not os.path.exists(source):

                print(f"❌ File not found: {source}")

                continue

            

            conf = float(input("Enter confidence threshold (0.1-1.0): ").strip() or "0.25")

            

            success = processor.process_video_with_hwaccel(

                source=source, 

                mode='detection', 

                conf=conf,

                output_dir=results_dir,

                preprocess=True,

                frame_skip=0,

                batch_size=4,

                enable_display=True  # Always enable display for real-time visualization

            )

            

            if success:

                last_processed_video = f"{results_dir}/detection_{Path(source).stem}.mp4"

            

        elif choice == '2':

            # Detection - High Performance

            source = input("\nEnter video path: ").strip()

            if not os.path.exists(source):

                print(f"❌ File not found: {source}")

                continue

            

            conf = float(input("Enter confidence threshold (0.1-1.0): ").strip() or "0.25")

            frame_skip = int(input("Frame skip (0=process all, 1=every other frame, etc): ").strip() or "2")

            

            success = processor.process_video_with_hwaccel(

                source=source, 

                mode='detection', 

                conf=conf,

                output_dir=results_dir,

                preprocess=True,

                frame_skip=frame_skip,

                batch_size=8,

                enable_display=True  # Always enable display for real-time visualization

            )

            

            if success:

                last_processed_video = f"{results_dir}/detection_{Path(source).stem}.mp4"

            

        elif choice == '3':

            # Segmentation

            source = input("\nEnter video path: ").strip()

            if not os.path.exists(source):

                print(f"❌ File not found: {source}")

                continue

            

            conf = float(input("Enter confidence threshold (0.1-1.0): ").strip() or "0.25")

            frame_skip = int(input("Frame skip (0=process all, 1=every other frame, etc): ").strip() or "1")

            

            success = processor.process_video_with_hwaccel(

                source=source, 

                mode='segmentation', 

                conf=conf,

                output_dir=results_dir,

                preprocess=True,

                frame_skip=frame_skip,

                batch_size=2,  # Smaller batch for segmentation

                enable_display=True  # Always enable display for real-time visualization

            )

            

            if success:

                last_processed_video = f"{results_dir}/segmentation_{Path(source).stem}.mp4"

            

        elif choice == '4':

            # Classification

            source = input("\nEnter video path: ").strip()

            if not os.path.exists(source):

                print(f"❌ File not found: {source}")

                continue

            

            conf = float(input("Enter confidence threshold (0.1-1.0): ").strip() or "0.25")

            frame_skip = int(input("Frame skip (0=process all, 1=every other frame, etc): ").strip() or "2")

            

            success = processor.process_video_with_hwaccel(

                source=source, 

                mode='classification', 

                conf=conf,

                output_dir=results_dir,

                preprocess=True,

                frame_skip=frame_skip,

                batch_size=4,  # Good batch size for classification

                enable_display=True  # Always enable display for real-time visualization

            )

            

            if success:

                last_processed_video = f"{results_dir}/classification_{Path(source).stem}.mp4"

            

        elif choice == '5':

            # Play last processed video

            if last_processed_video and os.path.exists(last_processed_video):

                play_video_with_ffplay(last_processed_video)

            else:

                video_path = input("\nNo recent video. Enter video path to play: ").strip()

                if os.path.exists(video_path):

                    play_video_with_ffplay(video_path)

                else:

                    print(f"❌ File not found: {video_path}")

            

        elif choice == '6':

            # Test hardware acceleration

            print("\n== Testing Hardware Acceleration ==")

            hw_accel = test_hardware_acceleration()

            print("\nHardware acceleration capabilities:")

            for k, v in hw_accel.items():

                print(f" - {k}: {v}")

                

        elif choice == '7':

            print("\nExiting YOLO Video Processor. Goodbye!")

            break

            

        else:

            print("\n❌ Invalid choice. Please try again.")



if __name__ == "__main__":

    main()
