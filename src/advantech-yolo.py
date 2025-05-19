#!/usr/bin/env python3
# ==========================================================================
# Enhanced YOLO Application with Hardware Acceleration
# ==========================================================================
# Version:      3.0.0
# Author:       Samir Singh <samir.singh@advantech.com> and Apoorv Saxena<apoorv.saxena@advantech.com>
# Created:      March 25, 2025
# Last Updated: May 19, 2025
# 
# Description:
#   This application provides hardware-accelerated YOLOv8 inference for
#   object detection, segmentation, and classification tasks on Advantech
#   edge AI devices. It automatically detects hardware capabilities and
#   optimizes performance for NVIDIA Jetson platforms.
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
import time
import shutil
import signal
import subprocess
import gc
from pathlib import Path

print("== Setting up system module paths ==")
system_paths = [
    '/usr/lib/python3/dist-packages',
    '/usr/local/lib/python3.8/dist-packages'
]

for path in system_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        print(f"Added {path} to sys.path")

try:
    import six
    print(f"Found six module: version {six.__version__} at {six.__file__}")
except ImportError:
    print("❌ Cannot find six module despite adding system paths!")
    sys.exit(1)

def fix_numpy_path():
    print("== Fixing NumPy paths and compatibility issues ==")
    
    dist_packages = '/usr/local/lib/python3.8/dist-packages'
    if os.path.exists(dist_packages):
        if dist_packages in sys.path:
            sys.path.remove(dist_packages)
        sys.path.insert(0, dist_packages)
        print(f"Added dist-packages to beginning of path: {dist_packages}")
    
    if 'numpy' in sys.modules:
        print("NumPy already imported, forcing reload")
        del sys.modules['numpy']
        if 'numpy.random' in sys.modules:
            del sys.modules['numpy.random']
    
    try:
        import numpy as np
        print(f"Using NumPy {np.__version__} from {np.__file__}")
        
        patched = False
        if not hasattr(np.random, 'BitGenerator'):
            print("❌ NumPy is missing BitGenerator attribute.")
            print("Adding BitGenerator mock...")
            
            class DummyBitGenerator:
                def __init__(self, seed=None):
                    self.seed = seed
            
            np.random.BitGenerator = DummyBitGenerator
            print("Added dummy BitGenerator to numpy.random")
            patched = True
        
        if not hasattr(np.random, 'mtrand'):
            print("❌ NumPy is missing mtrand attribute.")
            print("Adding mtrand mock...")
            
            class DummyRand:
                def __init__(self):
                    pass
                
                def __call__(self, *args, **kwargs):
                    return np.random.random(*args, **kwargs)
            
            class DummyMtrand:
                def __init__(self):
                    self._rand = DummyRand()
            
            np.random.mtrand = DummyMtrand()
            print("Added dummy mtrand to numpy.random")
            patched = True
        
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

def test_hardware_acceleration():
    print("== Testing hardware acceleration capabilities ==")
    
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
    
    try:
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

def setup_environment():
    print("== Setting up environment ==")
    
    for root, dirs, files in os.walk('.'):
        for d in dirs:
            if d == '__pycache__':
                cache_dir = os.path.join(root, d)
                print(f"Removing cache directory: {cache_dir}")
                shutil.rmtree(cache_dir)
    
    cv2_paths = [p for p in sys.path if 'cv2' in p]
    for p in cv2_paths:
        if p in sys.path:
            sys.path.remove(p)
            print(f"Removed potentially conflicting CV2 path: {p}")
    
    if not fix_numpy_path():
        print("❌ Failed to fix NumPy path. Cannot continue.")
        return False
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
    os.environ['CUDA_CACHE_DISABLE'] = '0'
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')
    
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':0'
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['XDG_RUNTIME_DIR'] = '/tmp'
    os.environ['PYTHONPATH'] = '/usr/lib/python3/dist-packages:' + os.environ.get('PYTHONPATH', '')
    
    print("\nFinal Python path:")
    for p in sys.path:
        print(f"  {p}")
    
    try:
        import torch
        if not hasattr(torch.distributed, 'is_initialized'):
            torch.distributed.is_initialized = lambda: False
            print("Patched torch.distributed.is_initialized")
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    except ImportError:
        print("Warning: Could not import torch")
    
    hw_accel = test_hardware_acceleration()
    
    print("✅ Environment setup complete")
    return hw_accel

def signal_handler(sig, frame):
    print("\nReceived interrupt. Cleaning up and exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def preprocess_video(input_file, output_file, scale=None, fps=None, hw_accel=None):
    print(f"Pre-processing video: {input_file} -> {output_file}")
    gst_cmd = ['gst-launch-1.0']
    gst_cmd.extend([f'filesrc location={input_file}', '!', 'decodebin', '!', 'nvvidconv'])
    if scale: width, height = scale; gst_cmd.append(f'! video/x-raw,width={width},height={height}')
    if fps: gst_cmd.append(f'! videorate ! video/x-raw,framerate={fps}/1')
    gst_cmd.extend(['!', 'nvv4l2h264enc', '!', 'h264parse', '!', 'qtmux', '!', f'filesink location={output_file}'])
    gst_command = " ".join(gst_cmd)
    print(f"Running GStreamer command: {gst_command}")
    try:
        process = subprocess.run(gst_command, shell=True, capture_output=True, text=True, check=False)
        if process.returncode == 0: print("✅ GStreamer video preprocessing complete"); return True
        else: print(f"❌ GStreamer preprocessing failed: {process.stderr}"); return False
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return False

def create_video_from_frames(frame_dir, output_file, fps=30, hw_accel=None):
    print(f"Creating video from frames in {frame_dir} -> {output_file}")
    if not os.path.exists(frame_dir): print(f"❌ Frames directory not found: {frame_dir}"); return False
    frames = sorted(list(Path(frame_dir).glob("*.jpg")))
    if not frames: print(f"❌ No frames found in {frame_dir}"); return False
    num_frames = len(frames)
    print(f"Found {num_frames} frames to process")
    
    # Get first frame dimensions
    try:
        import cv2
        first_frame = cv2.imread(str(frames[0]))
        if first_frame is None:
            print(f"❌ Could not read first frame: {frames[0]}")
            height, width = 720, 1280  # Default fallback dimensions
        else:
            height, width = first_frame.shape[:2]
            print(f"Frame dimensions: {width}x{height}")
    except Exception as e:
        print(f"❌ Error getting frame dimensions: {e}")
        height, width = 720, 1280  # Default fallback dimensions
    
    temp_dir = os.path.join(frame_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    for i, frame in enumerate(frames): shutil.copy(frame, os.path.join(temp_dir, f"img_{i:06d}.jpg"))
    frame_pattern = f"{temp_dir}/img_%06d.jpg"
    
    # Convert fps to a proper fraction for GStreamer
    from fractions import Fraction
    fps_fraction = Fraction(fps).limit_denominator(100)
    gst_fps = f"{fps_fraction.numerator}/{fps_fraction.denominator}"
    print(f"Using framerate: {fps} → {gst_fps}")
    
    # GStreamer pipeline with full caps specification (including width and height)
    gst_cmd = [
        'gst-launch-1.0',
        'multifilesrc',
        f'location={frame_pattern}',
        'index=0',
        '!',
        f'image/jpeg,framerate={gst_fps},width={width},height={height}',
        '!',
        'jpegdec',
        '!',
        'videoconvert',
        '!',
        'nvvidconv',
        '!',
        'nvv4l2h264enc',
        '!',
        'h264parse',
        '!',
        'qtmux',
        '!',
        f'filesink location={output_file}'
    ]
    gst_command = " ".join(gst_cmd)
    
    print(f"Running GStreamer command: {gst_command}")
    try:
        process = subprocess.run(gst_command, shell=True, capture_output=True, text=True, check=False)
        success = process.returncode == 0
        shutil.rmtree(temp_dir, ignore_errors=True)
        if success: print("✅ GStreamer video creation complete"); return True
        else:
            print(f"❌ GStreamer video creation failed: {process.stderr}")
            original_frame_pattern = f"{frame_dir}/frame_%06d.jpg"
            
            # Alternative approach with complete caps filter
            alt_cmd = [
                'gst-launch-1.0',
                'multifilesrc',
                f'location={original_frame_pattern}',
                'index=1',
                '!',
                f'image/jpeg,framerate={gst_fps},width={width},height={height}',
                '!',
                'jpegdec',
                '!',
                'videoconvert',
                '!',
                'nvvidconv',
                '!',
                'nvv4l2h264enc',
                '!',
                'h264parse',
                '!',
                'qtmux',
                '!',
                f'filesink location={output_file}'
            ]
            alt_command = " ".join(alt_cmd)
            
            print(f"Trying alternative GStreamer approach: {alt_command}")
            alt_process = subprocess.run(alt_command, shell=True, capture_output=True, text=True, check=False)
            if alt_process.returncode == 0: print("✅ Alternative GStreamer approach successful"); return True
            else:
                print(f"❌ All GStreamer approaches failed, trying fallback method with ffmpeg")
                ffmpeg_command = (f'ffmpeg -y -framerate {fps} -i "{frame_dir}/frame_%06d.jpg" -c:v libx264 -preset ultrafast -crf 28 -pix_fmt yuv420p "{output_file}"')
                print(f"Running ffmpeg fallback: {ffmpeg_command}")
                ffmpeg_process = subprocess.run(ffmpeg_command, shell=True, capture_output=True, text=True, check=False)
                if ffmpeg_process.returncode == 0: print("✅ Fallback ffmpeg approach successful"); return True
                else: print(f"❌ All approaches failed to create video from frames"); return False
    except Exception as e:
        print(f"❌ Error during video creation: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False

class YOLOHWAccelerated:
    def __init__(self, hw_acceleration=None):
        print("\n== Initializing YOLO with Hardware Acceleration ==")
        self.hw_accel = hw_acceleration or {}
        
        try:
            import numpy as np
            self.np = np
            print(f"Using NumPy {np.__version__}")
            
            import torch
            self.torch = torch
            
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            print(f"Using PyTorch {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
                print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024 / 1024:.2f} MB")
                print(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024 / 1024:.2f} MB")
            
            try:
                old_path = sys.path.copy()
                sys.path = ['/usr/lib/python3.8', '/usr/local/lib/python3.8/dist-packages']
                
                import cv2
                self.cv2 = cv2
                print(f"Using OpenCV {cv2.__version__}")
                
                if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    print(f"✅ OpenCV CUDA acceleration available: {cv2.cuda.getCudaEnabledDeviceCount()} devices")
                    self.opencv_cuda = True
                else:
                    print("OpenCV CUDA acceleration not available")
                    self.opencv_cuda = False
                
                try:
                    test_window_name = "TestWindow"
                    cv2.namedWindow(test_window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(test_window_name, 320, 240)
                    cv2.imshow(test_window_name, np.zeros((240, 320, 3), dtype=np.uint8))
                    cv2.waitKey(1)
                    cv2.destroyWindow(test_window_name)
                    print("✅ Display is available! Real-time visualization will be enabled.")
                    self.display_available = True
                except Exception as disp_e:
                    print(f"❌ Display test failed: {disp_e}")
                    print("Real-time visualization will be disabled.")
                    self.display_available = False
                
                sys.path = old_path
            except ImportError as e:
                print(f"❌ OpenCV import error: {e}")
                print("Trying alternative OpenCV import method...")
                self.display_available = False
                self.opencv_cuda = False
                
                if 'cv2' in sys.modules:
                    del sys.modules['cv2']
                
                import ctypes
                try:
                    ctypes.CDLL("libopencv_core.so.4.5")
                    import cv2
                    self.cv2 = cv2
                    print(f"Using OpenCV {cv2.__version__} (loaded via alternative method)")
                except Exception as cv_e:
                    print(f"❌ Failed to load OpenCV: {cv_e}")
                    print("Will continue without OpenCV but functionality will be limited")
                    self.cv2 = None
            
            old_sys_path = list(sys.path)
            
            for path in ['/usr/lib/python3/dist-packages', '/usr/local/lib/python3.8/dist-packages']:
                if path not in sys.path:
                    sys.path.insert(0, path)
            
            try:
                from ultralytics import YOLO
                self.YOLO = YOLO
                import ultralytics
                print(f"Using Ultralytics {ultralytics.__version__}")
                
                self.ultralytics_version = ultralytics.__version__
                
                from ultralytics.yolo.utils import ops
                print("Configured ultralytics for Jetson optimization")
            except ImportError as e:
                print(f"❌ Error importing ultralytics: {e}")
                import traceback
                traceback.print_exc()
                raise
            finally:
                sys.path = old_sys_path
            
            self.models = {}
            
        except ImportError as e:
            print(f"❌ Error importing dependencies: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_model(self, model_type='detection'):
        if model_type in self.models:
            print(f"Model for {model_type} already loaded")
            return self.models[model_type]
        
        model_paths = {
            'detection': 'yolov8n.pt',
            'detection-s': 'yolov8s.pt',
            'segmentation': 'yolov8n-seg.pt',
            'classification': 'yolov8n-cls.pt'
        }
        
        if model_type not in model_paths:
            print(f"❌ Invalid model type: {model_type}")
            print(f"Available types: {', '.join(model_paths.keys())}")
            return None
        
        model_path = model_paths[model_type]
        
        if not os.path.exists(model_path):
            print(f"Downloading {model_path}...")
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
        
        try:
            print(f"Loading {model_type} model from {model_path}...")
            device = 0 if self.torch.cuda.is_available() else 'cpu'
            
            gc.collect()
            
            if device == 0:
                self.torch.cuda.empty_cache()
                print(f"Pre-load CUDA memory: {self.torch.cuda.memory_allocated(0) / 1024 / 1024:.2f} MB allocated")
            
            model = self.YOLO(model_path)
            
            self.models[model_type] = model
            
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
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        frames_dir = f"{output_dir}/frames_{mode}_{source_path.stem}"
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)
        
        print(f"\nProcessing video: {source} with {mode} mode...")
        
        if preprocess:
            temp_video = f"{output_dir}/temp_{source_path.stem}.mp4"
            if not preprocess_video(source, temp_video, hw_accel=self.hw_accel):
                print("Skipping pre-processing, using original video...")
                temp_video = source
        else:
            temp_video = source
        
        try:
            device = 0 if self.torch.cuda.is_available() else 'cpu'
            
            start_time = time.time()
            
            cap = self.cv2.VideoCapture(temp_video)
            if not cap.isOpened():
                print(f"❌ Failed to open video: {temp_video}")
                return None
            
            width = int(cap.get(self.cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(self.cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
            
            if max_frames and max_frames > 0:
                total_frames = min(total_frames, max_frames)
                print(f"Processing limited to {max_frames} frames")
            
            print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
            
            display_enabled = enable_display and self.display_available
            if display_enabled:
                window_name = f"YOLO {mode.capitalize()} - {source_path.name}"
                self.cv2.namedWindow(window_name, self.cv2.WINDOW_NORMAL)
                
                if width > height:
                    display_width = min(1280, width)
                    display_height = int((display_width / width) * height)
                else:
                    display_height = min(720, height)
                    display_width = int((display_height / height) * width)
                
                self.cv2.resizeWindow(window_name, display_width, display_height)
                print(f"✅ Display window created with size {display_width}x{display_height}")
            
            frame_count = 0
            processed_count = 0
            saved_frames = 0
            
            total_objects = 0
            class_counts = {}
            processing_times = []
            
            print("\nProcessing frames...")
            
            batch_frames = []
            batch_indices = []
            
            while frame_count < total_frames:
                if frame_count % 10 == 0:
                    percent = int((frame_count / total_frames) * 100)
                    elapsed = time.time() - start_time
                    if processed_count > 0 and elapsed > 0:
                        fps_achieved = processed_count / elapsed
                        eta_seconds = (total_frames - frame_count) / fps_achieved if fps_achieved > 0 else 0
                        eta_min = eta_seconds / 60
                        print(f"\rProgress: {percent}% ({frame_count}/{total_frames}) | "
                              f"Speed: {fps_achieved:.1f} FPS | ETA: {eta_min:.1f} min", end="", flush=True)
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_skip > 0 and (frame_count % (frame_skip + 1)) != 1:
                    continue
                
                batch_frames.append(frame)
                batch_indices.append(frame_count)
                
                if len(batch_frames) >= batch_size or frame_count == total_frames:
                    try:
                        batch_start = time.time()
                        
                        results = model.predict(
                            source=batch_frames,
                            conf=conf,
                            verbose=False,
                            stream=False,
                            device=device,
                            half=True
                        )
                        
                        batch_time = time.time() - batch_start
                        processing_times.append(batch_time / len(batch_frames))
                        
                        for i, r in enumerate(results):
                            processed_count += 1
                            
                            if mode == 'detection':
                                if hasattr(r, 'boxes'):
                                    boxes = r.boxes
                                    objects_in_frame = len(boxes)
                                    total_objects += objects_in_frame
                                    
                                    for box in boxes:
                                        cls = int(box.cls[0])
                                        class_name = model.names[cls]
                                        
                                        if class_name in class_counts:
                                            class_counts[class_name] += 1
                                        else:
                                            class_counts[class_name] = 1
                            
                            elif mode == 'segmentation':
                                if hasattr(r, 'masks') and r.masks is not None:
                                    objects_in_frame = len(r.masks)
                                    total_objects += objects_in_frame
                                    
                                    if hasattr(r, 'boxes'):
                                        for j in range(min(len(r.boxes), objects_in_frame)):
                                            cls = int(r.boxes[j].cls[0])
                                            class_name = model.names[cls]
                                            
                                            if class_name in class_counts:
                                                class_counts[class_name] += 1
                                            else:
                                                class_counts[class_name] = 1
                            
                            elif mode == 'classification':
                                if hasattr(r, 'probs'):
                                    probs = r.probs
                                    if probs is not None:
                                        if isinstance(probs, self.torch.Tensor):
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
                                            top_idx = int(probs.top1)
                                            class_name = model.names[top_idx]
                                            
                                            if class_name in class_counts:
                                                class_counts[class_name] += 1
                                            else:
                                                class_counts[class_name] = 1
                                            
                                            total_objects += 1
                            
                            annotated_frame = r.plot()
                            frame_index = batch_indices[i]
                            
                            frame_path = f"{frames_dir}/frame_{frame_index:06d}.jpg"
                            self.cv2.imwrite(frame_path, annotated_frame)
                            saved_frames += 1
                            
                            if display_enabled:
                                progress_text = f"Progress: {int((frame_count / total_frames) * 100)}% | FPS: {1/processing_times[-1]:.1f}"
                                self.cv2.putText(annotated_frame, progress_text, (10, 30), 
                                                self.cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                
                                self.cv2.imshow(window_name, annotated_frame)
                                
                                key = self.cv2.waitKey(1) & 0xFF
                                if key == 27 or key == ord('q'):
                                    print("\n⚠️ Processing interrupted by user")
                                    break
                        
                        batch_frames = []
                        batch_indices = []
                        
                        if processed_count % 100 == 0:
                            gc.collect()
                            if device == 0:
                                self.torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"\n❌ Error processing batch: {e}")
                        import traceback
                        traceback.print_exc()
                        batch_frames = []
                        batch_indices = []
            
            cap.release()
            
            if display_enabled:
                self.cv2.destroyAllWindows()
            
            gc.collect()
            if device == 0:
                self.torch.cuda.empty_cache()
            
            output_video = f"{output_dir}/{mode}_{source_path.stem}.mp4"
            print(f"\nCreating final video with FFmpeg...")
            if saved_frames > 0:
                effective_fps = fps
                if frame_skip > 0:
                    effective_fps = fps / (frame_skip + 1)
                create_video_from_frames(frames_dir, output_video, fps=effective_fps, hw_accel=self.hw_accel)
            else:
                print("❌ No frames were processed or saved")
            
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
            try:
                if 'cap' in locals() and cap is not None:
                    cap.release()
                
                if self.display_available:
                    self.cv2.destroyAllWindows()
                
                if preprocess and 'temp_video' in locals() and temp_video != source and os.path.exists(temp_video):
                    print(f"Cleaning up temporary file: {temp_video}")
                    os.remove(temp_video)
            except Exception as cleanup_error:
                print(f"Error during cleanup: {cleanup_error}")

def play_video_with_ffplay(video_path):
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    try:
        print(f"Playing video: {video_path}")
        cmd = [
            'ffplay', 
            '-autoexit',
            '-loglevel', 'error',
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
        if process.poll() is None:
            process.terminate()
        return True
    except Exception as e:
        print(f"❌ Error playing video: {e}")
        return False

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

def main():
    hw_accel = setup_environment()
    
    try:
        processor = YOLOHWAccelerated(hw_accel)
    except Exception as e:
        print(f"❌ Failed to initialize YOLO processor: {e}")
        import traceback
        traceback.print_exc()
        return
    
    results_dir = "./results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    
    last_processed_video = None
    
    while True:
        choice = show_menu()
        
        if choice == '1':
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
                enable_display=True
            )
            
            if success:
                last_processed_video = f"{results_dir}/detection_{Path(source).stem}.mp4"
            
        elif choice == '2':
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
                enable_display=True
            )
            
            if success:
                last_processed_video = f"{results_dir}/detection_{Path(source).stem}.mp4"
            
        elif choice == '3':
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
                batch_size=2,
                enable_display=True
            )
            
            if success:
                last_processed_video = f"{results_dir}/segmentation_{Path(source).stem}.mp4"
            
        elif choice == '4':
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
                batch_size=4,
                enable_display=True
            )
            
            if success:
                last_processed_video = f"{results_dir}/classification_{Path(source).stem}.mp4"
            
        elif choice == '5':
            if last_processed_video and os.path.exists(last_processed_video):
                play_video_with_ffplay(last_processed_video)
            else:
                video_path = input("\nNo recent video. Enter video path to play: ").strip()
                if os.path.exists(video_path):
                    play_video_with_ffplay(video_path)
                else:
                    print(f"❌ File not found: {video_path}")
            
        elif choice == '6':
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
