#!/usr/bin/env python3
# ==========================================================================
# Enhanced YOLO Application with Hardware Acceleration
# ==========================================================================
# Version:      2.0.0
# Author:       Samir Singh <samir.singh@advantech.com> and Apoorv Saxena<apoorv.saxena@advantech.com>
# Created:      March 25, 2025
# Last Updated: Dec 03, 2025
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
__version__="2.0.0"
__author__="Advantech Co., Ltd"
__build_date__="2024-12"
__copyright__="Copyright (c) 2024 Advantech Co., Ltd. All Rights Reserved."

import sys,os,signal,threading,time,atexit
from pathlib import Path
from typing import Optional,List,Dict,Any,Tuple,Union
from dataclasses import dataclass,field
from enum import Enum
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['GLOG_minloglevel']='3'
import warnings
warnings.filterwarnings('ignore')

WINDOW_NAME="Advantech Yolo Vision"
_GLOBAL_SHUTDOWN=threading.Event()
_FORCE_EXIT_COUNT=0

def force_shutdown_handler(signum,frame):
    global _FORCE_EXIT_COUNT
    _FORCE_EXIT_COUNT+=1
    _GLOBAL_SHUTDOWN.set()
    if _FORCE_EXIT_COUNT==1:
        print("\n\n[!] Shutdown requested (Ctrl+C again to force quit)...")
    elif _FORCE_EXIT_COUNT>=2:
        print("\n[!] Force quitting...")
        try:
            cv2.destroyAllWindows()
        except:
            pass
        os._exit(1)

signal.signal(signal.SIGINT,force_shutdown_handler)
signal.signal(signal.SIGTERM,force_shutdown_handler)

def is_shutdown()->bool:
    return _GLOBAL_SHUTDOWN.is_set()

def request_shutdown():
    _GLOBAL_SHUTDOWN.set()

PYCUDA_AVAILABLE=False
TENSORRT_AVAILABLE=False
ONNX_AVAILABLE=False
TORCH_AVAILABLE=False
FLASK_AVAILABLE=False
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE=True
except ImportError:
    cuda=None
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE=True
except ImportError:
    pass
try:
    import onnxruntime as ort
    ONNX_AVAILABLE=True
except ImportError:
    pass
try:
    import torch
    TORCH_AVAILABLE=True
except ImportError:
    pass
try:
    from flask import Flask,jsonify
    FLASK_AVAILABLE=True
except ImportError:
    pass

try:
    from advantech_core import (
        AdvantechConfig,AdvantechLogger,AdvantechTaskType,AdvantechModelFormat,
        AdvantechInputSource,AdvantechPrecision,AdvantechDetection,
        AdvantechClassification,AdvantechEngine,AdvantechTensorRTEngine,
        AdvantechOnnxEngine,AdvantechPyTorchEngine,AdvantechModelLoader,
        AdvantechMetricsCollector,AdvantechMetrics,AdvantechOverlayRenderer,
        AdvantechMemoryManager,AdvantechRingBuffer,AdvantechFrame,
        ensure_cuda_context,cleanup_cuda_context
    )
    CORE_IMPORTED=True
except ImportError as e:
    print(f"Warning: Could not import advantech_core: {e}")
    CORE_IMPORTED=False

@dataclass
class CameraFormat:
    pixel_format:str
    width:int
    height:int
    fps:List[int]=field(default_factory=list)

class AdvantechCameraDiscovery:
    @staticmethod
    def _suppress_errors():
        os.environ['GST_DEBUG']='0'
        os.environ['OPENCV_LOG_LEVEL']='OFF'
    @staticmethod
    def _get_formats(device_num:int)->List[CameraFormat]:
        formats=[]
        try:
            import subprocess
            result=subprocess.run(['v4l2-ctl','-d',f'/dev/video{device_num}','--list-formats-ext'],capture_output=True,text=True,timeout=5)
            if result.returncode!=0:
                return formats
            current_format=None
            current_resolution=None
            for line in result.stdout.split('\n'):
                line=line.strip()
                if "'" in line and 'Pixel Format' in line:
                    parts=line.split("'")
                    if len(parts)>=2:
                        current_format=parts[1]
                elif 'Size:' in line and 'Discrete' in line:
                    parts=line.split()
                    for p in parts:
                        if 'x' in p and p[0].isdigit():
                            try:
                                w,h=map(int,p.split('x'))
                                current_resolution=(w,h)
                            except:
                                pass
                elif 'Interval:' in line and current_format and current_resolution:
                    fps_val=0
                    if '(' in line and 'fps' in line:
                        try:
                            fps_str=line.split('(')[1].split()[0]
                            fps_val=int(float(fps_str))
                        except:
                            pass
                    existing=None
                    for f in formats:
                        if f.pixel_format==current_format and f.width==current_resolution[0] and f.height==current_resolution[1]:
                            existing=f
                            break
                    if existing:
                        if fps_val>0 and fps_val not in existing.fps:
                            existing.fps.append(fps_val)
                    else:
                        fps_list=[fps_val] if fps_val>0 else []
                        formats.append(CameraFormat(current_format,current_resolution[0],current_resolution[1],fps_list))
        except:
            pass
        return formats
    @staticmethod
    def _test_capture(device_num:int)->bool:
        if is_shutdown():
            return False
        try:
            stderr_fd=os.dup(2)
            devnull=os.open(os.devnull,os.O_WRONLY)
            os.dup2(devnull,2)
            try:
                cap=cv2.VideoCapture(device_num,cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap=cv2.VideoCapture(device_num)
                if not cap.isOpened():
                    cap=cv2.VideoCapture(f'/dev/video{device_num}')
                if not cap.isOpened():
                    return False
                cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
                for _ in range(3):
                    if is_shutdown():
                        cap.release()
                        return False
                    ret,frame=cap.read()
                    if ret and frame is not None:
                        cap.release()
                        return True
                cap.release()
                return False
            finally:
                os.dup2(stderr_fd,2)
                os.close(stderr_fd)
                os.close(devnull)
        except:
            return False
    @classmethod
    def discover_cameras(cls)->List[Dict]:
        cls._suppress_errors()
        cameras=[]
        for i in range(10):
            if is_shutdown():
                break
            device_path=f'/dev/video{i}'
            if not Path(device_path).exists():
                continue
            name=f"Camera {i}"
            try:
                import subprocess
                result=subprocess.run(['v4l2-ctl','-d',device_path,'--info'],capture_output=True,text=True,timeout=2)
                for line in result.stdout.split('\n'):
                    if 'Card type' in line:
                        name=line.split(':')[1].strip()
                        break
            except:
                pass
            formats=cls._get_formats(i)
            can_capture=cls._test_capture(i)
            if formats or can_capture:
                cameras.append({'device':i,'path':device_path,'name':name,'formats':formats if formats else [],'tested':can_capture})
        return cameras

class AdvantechVideoSource:
    def __init__(self):
        self._cap:Optional[cv2.VideoCapture]=None
        self._source_type="unknown"
        self.width=0
        self.height=0
        self.fps=0
        self.total_frames=0
        self.is_video_file=False
        self._frame_count=0
        self._start_time=0
    def open_camera(self,device:int,width:int=1280,height:int=720,pixel_format:str="MJPG",fps:int=30)->bool:
        if is_shutdown():
            return False
        try:
            stderr_fd=os.dup(2)
            devnull=os.open(os.devnull,os.O_WRONLY)
            os.dup2(devnull,2)
            try:
                if device==-1:
                    gst=(f"nvarguscamerasrc ! video/x-raw(memory:NVMM),width={width},height={height},"
                         f"framerate={fps}/1,format=NV12 ! nvvidconv ! video/x-raw,format=BGRx ! "
                         f"videoconvert ! video/x-raw,format=BGR ! appsink drop=true sync=false")
                    self._cap=cv2.VideoCapture(gst,cv2.CAP_GSTREAMER)
                elif pixel_format.upper() in ['MJPG','MJPEG']:
                    gst=(f"v4l2src device=/dev/video{device} ! "
                         f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
                         f"jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1")
                    self._cap=cv2.VideoCapture(gst,cv2.CAP_GSTREAMER)
                else:
                    self._cap=cv2.VideoCapture(device,cv2.CAP_V4L2)
                    if self._cap.isOpened():
                        fourcc=cv2.VideoWriter_fourcc(*pixel_format[:4])
                        self._cap.set(cv2.CAP_PROP_FOURCC,fourcc)
                        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
                        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
                        self._cap.set(cv2.CAP_PROP_FPS,fps)
            finally:
                os.dup2(stderr_fd,2)
                os.close(stderr_fd)
                os.close(devnull)
            if self._cap and self._cap.isOpened():
                self._source_type="camera"
                self.width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
                self.height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
                self.fps=int(self._cap.get(cv2.CAP_PROP_FPS)) or fps
                self._start_time=time.time()
                return True
        except:
            pass
        return False
    def open_rtsp(self,url:str)->bool:
        if is_shutdown():
            return False
        try:
            gst=(f"rtspsrc location={url} latency=100 ! "
                 f"rtph264depay ! h264parse ! nvv4l2decoder ! "
                 f"nvvidconv ! video/x-raw,format=BGRx ! "
                 f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1")
            self._cap=cv2.VideoCapture(gst,cv2.CAP_GSTREAMER)
            if not self._cap.isOpened():
                self._cap=cv2.VideoCapture(url)
            if self._cap.isOpened():
                self._source_type="rtsp"
                self.width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps=int(self._cap.get(cv2.CAP_PROP_FPS)) or 30
                self._start_time=time.time()
                return True
        except:
            pass
        return False
    def open_file(self,path:str)->bool:
        if is_shutdown():
            return False
        if not Path(path).exists():
            return False
        try:
            self._cap=cv2.VideoCapture(path)
            if self._cap.isOpened():
                self._source_type="file"
                self.is_video_file=True
                self.width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps=self._cap.get(cv2.CAP_PROP_FPS) or 30
                self.total_frames=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self._start_time=time.time()
                return True
        except:
            pass
        return False
    def read(self)->Optional[np.ndarray]:
        if is_shutdown() or self._cap is None:
            return None
        try:
            ret,frame=self._cap.read()
            if ret and frame is not None:
                self._frame_count+=1
                return frame
        except:
            pass
        return None
    def release(self):
        if self._cap:
            try:
                self._cap.release()
            except:
                pass
            self._cap=None

class AdvantechPipeline:
    def __init__(self,config:'AdvantechConfig',logger:'AdvantechLogger',source:AdvantechVideoSource,engine:'AdvantechEngine'):
        self.config=config
        self.logger=logger
        self.source=source
        self.engine=engine
        self._running=False
        self._stop_event=threading.Event()
        self._capture_thread=None
        self._output_thread=None
        self._frame_buffer=None
        self._frame_lock=threading.Lock()
        self._result_buffer=None
        self._result_lock=threading.Lock()
        self._total_frames=0
        self._start_time=0
        self._encoder=None
        self._renderer=AdvantechOverlayRenderer(config)
    def start(self):
        self._running=True
        self._stop_event.clear()
        self._start_time=time.perf_counter()
        if self.config.save_video and self.config.output_path:
            try:
                output_file=Path(self.config.output_path)
                if output_file.is_dir():
                    output_file=output_file/f"output_{int(time.time())}.mp4"
                fourcc=cv2.VideoWriter_fourcc(*'mp4v')
                self._encoder=cv2.VideoWriter(str(output_file),fourcc,self.source.fps or 30,(self.source.width,self.source.height))
            except Exception as e:
                print(f"Warning: Could not create video writer: {e}")
        self._capture_thread=threading.Thread(target=self._capture_loop,daemon=True)
        self._capture_thread.start()
        self._output_thread=threading.Thread(target=self._output_loop,daemon=True)
        self._output_thread.start()
        self._inference_loop()
    def _capture_loop(self):
        frame_interval=1.0/(self.source.fps or 30) if self.source.is_video_file else 0
        last_time=time.perf_counter()
        while self._running and not self._stop_event.is_set() and not is_shutdown():
            frame=self.source.read()
            if frame is None:
                if self.source.is_video_file:
                    self._stop_event.set()
                    break
                continue
            if self.source.is_video_file and frame_interval>0:
                elapsed=time.perf_counter()-last_time
                if elapsed<frame_interval:
                    time.sleep(frame_interval-elapsed)
                last_time=time.perf_counter()
            with self._frame_lock:
                self._frame_buffer=(frame,time.perf_counter())
    def _inference_loop(self):
        while self._running and not self._stop_event.is_set() and not is_shutdown():
            frame_data=None
            with self._frame_lock:
                if self._frame_buffer is not None:
                    frame_data=self._frame_buffer
                    self._frame_buffer=None
            if frame_data is None:
                time.sleep(0.001)
                continue
            frame,timestamp=frame_data
            try:
                result=self.engine.infer(frame)
                with self._result_lock:
                    self._result_buffer={'frame':frame,'result':result,'timestamp':timestamp}
            except:
                pass
    def _output_loop(self):
        window_created=False
        while self._running and not self._stop_event.is_set() and not is_shutdown():
            result_data=None
            with self._result_lock:
                if self._result_buffer is not None:
                    result_data=self._result_buffer
                    self._result_buffer=None
            if result_data is None:
                if self.config.show_display and window_created:
                    key=cv2.waitKey(1)&0xFF
                    if key==ord('q') or key==27:
                        self._stop_event.set()
                        request_shutdown()
                time.sleep(0.001)
                continue
            self._total_frames+=1
            frame=result_data['frame']
            result=result_data['result']
            elapsed=time.perf_counter()-self._start_time
            fps=self._total_frames/elapsed if elapsed>0 else 0
            metrics=AdvantechMetrics(fps=fps,total_frames=self._total_frames)
            try:
                if isinstance(result,AdvantechClassification):
                    frame=self._renderer.render(frame,[],metrics,result)
                else:
                    frame=self._renderer.render(frame,result or [],metrics)
            except:
                pass
            if self._encoder:
                try:
                    self._encoder.write(frame)
                except:
                    pass
            if self.config.show_display:
                try:
                    if not window_created:
                        cv2.namedWindow(WINDOW_NAME,cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(WINDOW_NAME,1280,720)
                        window_created=True
                    cv2.imshow(WINDOW_NAME,frame)
                    key=cv2.waitKey(1)&0xFF
                    if key==ord('q') or key==27:
                        self._stop_event.set()
                        request_shutdown()
                except:
                    pass
    def stop(self):
        self._running=False
        self._stop_event.set()
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
        if self._output_thread:
            self._output_thread.join(timeout=1.0)
        self.source.release()
        if self._encoder:
            try:
                self._encoder.release()
            except:
                pass
        try:
            cv2.destroyAllWindows()
        except:
            pass
    def is_running(self)->bool:
        return self._running and not self._stop_event.is_set() and not is_shutdown()
    def get_total_frames(self)->int:
        return self._total_frames
    def get_total_fps(self)->float:
        elapsed=time.perf_counter()-self._start_time
        return self._total_frames/elapsed if elapsed>0 else 0

def print_banner():
    banner=f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║     █████╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗ ██████╗██╗  ██╗  ║
║    ██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔════╝██║  ██║  ║
║    ███████║██║  ██║╚██╗ ██╔╝███████║██╔██╗ ██║   ██║   █████╗  ██║     ███████║  ║
║    ██╔══██║██║  ██║ ╚████╔╝ ██╔══██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██╔══██║  ║
║    ██║  ██║██████╔╝  ╚██╔╝  ██║  ██║██║ ╚████║   ██║   ███████╗╚██████╗██║  ██║  ║
║    ╚═╝  ╚═╝╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝  ║
║                        YOLO Inference Pipeline v{__version__}                            ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Author: {__author__:<18}  Build: {__build_date__:<14}                               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║       {__copyright__}                ║
╚══════════════════════════════════════════════════════════════════════════════════╝"""
    print(banner)

class AdvantechCLI:
    def __init__(self):
        self.config=AdvantechConfig.from_env()
        self.logger=AdvantechLogger(self.config)
        self._pipeline:Optional[AdvantechPipeline]=None
    def _get_choice(self,prompt:str,valid:List[str],default:str="1")->str:
        try:
            if is_shutdown():
                sys.exit(0)
            choice=input(f"{prompt} [{default}]: ").strip() or default
            return choice if choice in valid else default
        except (EOFError,KeyboardInterrupt):
            sys.exit(0)
    def _get_input(self,prompt:str,default:str="")->str:
        try:
            if is_shutdown():
                sys.exit(0)
            value=input(f"{prompt} [{default}]: ").strip()
            return value if value else default
        except (EOFError,KeyboardInterrupt):
            sys.exit(0)
    def list_cameras(self):
        print_banner()
        print("\nDiscovering cameras...")
        print("  Scanning /dev/video0-9...")
        import subprocess
        for i in range(10):
            device_path=f'/dev/video{i}'
            if Path(device_path).exists():
                print(f"  Found: {device_path}")
        cameras=AdvantechCameraDiscovery.discover_cameras()
        if not cameras:
            print("\n  No cameras found that can capture video.")
            print("\n  Troubleshooting:")
            print("    - Check if camera is connected: ls -la /dev/video*")
            print("    - Check permissions: sudo chmod 666 /dev/video*")
            print("    - Install v4l-utils: sudo apt install v4l-utils")
            print("    - Test manually: v4l2-ctl --list-devices")
            return
        print(f"\nFound {len(cameras)} camera(s):\n")
        for cam in cameras:
            status="[OK]" if cam.get('tested',False) else "[?]"
            print(f"  {status} [{cam['device']}] {cam['name']}")
            print(f"      Path: {cam['path']}")
            if cam['formats']:
                print(f"      Formats:")
                for fmt in cam['formats'][:5]:
                    fps_str=', '.join(map(str,fmt.fps)) if fmt.fps else 'N/A'
                    print(f"        {fmt.pixel_format} {fmt.width}x{fmt.height} @ {fps_str} fps")
            else:
                print(f"      Formats: None detected (v4l2-ctl may not be installed)")
            print()
    def run_interactive(self):
        print_banner()
        print("\n[1] Detection  [2] Classification  [3] Segmentation")
        task_choice=self._get_choice("Select task",["1","2","3"],"1")
        self.config.task_type={"1":AdvantechTaskType.DETECTION,"2":AdvantechTaskType.CLASSIFICATION,"3":AdvantechTaskType.SEGMENTATION}[task_choice]
        print("\n[1] TensorRT  [2] ONNX  [3] PyTorch")
        format_choice=self._get_choice("Select format",["1","2","3"],"1")
        self.config.model_format={"1":AdvantechModelFormat.TENSORRT,"2":AdvantechModelFormat.ONNX,"3":AdvantechModelFormat.PYTORCH}[format_choice]
        ext_map={"1":".engine/.trt","2":".onnx","3":".pt"}
        self.config.model_path=self._get_input(f"Model path ({ext_map[format_choice]})","")
        if not self.config.model_path:
            print("Error: Model path required")
            return
        print("\n[1] Webcam  [2] RTSP  [3] Video File")
        source_choice=self._get_choice("Select source",["1","2","3"],"1")
        self.config.input_source={"1":AdvantechInputSource.CAMERA,"2":AdvantechInputSource.RTSP,"3":AdvantechInputSource.FILE}[source_choice]
        if self.config.input_source==AdvantechInputSource.CAMERA:
            cameras=AdvantechCameraDiscovery.discover_cameras()
            if cameras:
                print("\nAvailable cameras:")
                for cam in cameras:
                    status="[OK]" if cam.get('tested',False) else "[?]"
                    print(f"  {status} [{cam['device']}] {cam['name']}")
            else:
                print("\nNo cameras auto-detected. You can still enter device manually.")
                print("  Check: ls /dev/video*")
            device=self._get_input("Camera device (0, 1, etc)","0")
            self.config.camera_device=device if device.startswith('/dev/') else f"/dev/video{device}"
            self.config.camera_width=1280
            self.config.camera_height=720
            self.config.camera_format="MJPG"
            self.config.camera_fps=30
        elif self.config.input_source==AdvantechInputSource.RTSP:
            self.config.input_path=self._get_input("RTSP URL","rtsp://192.168.1.100:554/stream")
        else:
            self.config.input_path=self._get_input("Video file path","input.mp4")
        save_choice=self._get_choice("Save output video? (y/n)",["y","n","Y","N"],"n")
        self.config.save_video=save_choice.lower()=="y"
        if self.config.save_video:
            self.config.output_path=self._get_input("Output path","./output.mp4")
        display_choice=self._get_choice("Show display? (y/n)",["y","n","Y","N"],"y")
        self.config.show_display=display_choice.lower()=="y"
        self._run_pipeline()
    def run_dryrun(self,model_path:Optional[str]=None):
        print_banner()
        print("\n"+"="*66)
        print("  DRY RUN - System Verification")
        print("="*66)
        if model_path and Path(model_path).exists():
            print(f"\nModel: {model_path}")
            loader=AdvantechModelLoader(self.config,self.logger)
            try:
                engine=loader.load(model_path)
                engine.warmup(iterations=2)
                print(f"  Status: SUCCESS")
                print(f"  Task: {engine.task_type.name}")
                print(f"  Input: {engine._model_input_width}x{engine._model_input_height}")
                engine.cleanup()
            except Exception as e:
                print(f"  Status: FAILED - {e}")
        else:
            print(f"\nModel: Not specified or not found")
        print("\nCameras:")
        cameras=AdvantechCameraDiscovery.discover_cameras()
        if cameras:
            for cam in cameras:
                print(f"  [{cam['device']}] {cam['name']} - OK")
        else:
            print("  No working cameras found")
        print("\n"+"="*66)
    def run_benchmark(self,model_path:str):
        print_banner()
        print("\n"+"="*66)
        print("  BENCHMARK")
        print("="*66)
        if not Path(model_path).exists():
            print(f"Error: Model not found: {model_path}")
            return
        print(f"\nModel: {model_path}")
        loader=AdvantechModelLoader(self.config,self.logger)
        engine=loader.load(model_path)
        input_h=engine._model_input_height
        input_w=engine._model_input_width
        dummy=np.random.randint(0,255,(input_h,input_w,3),dtype=np.uint8)
        print("Warming up...")
        for _ in range(10):
            if is_shutdown():
                engine.cleanup()
                return
            engine.infer(dummy)
        print("Running benchmark (100 iterations)...")
        latencies=[]
        for i in range(100):
            if is_shutdown():
                break
            start=time.perf_counter()
            engine.infer(dummy)
            latencies.append((time.perf_counter()-start)*1000)
        if latencies:
            sorted_lat=sorted(latencies)
            n=len(sorted_lat)
            avg=sum(latencies)/n
            print(f"\nResults:")
            print(f"  Iterations: {n}")
            print(f"  Avg: {avg:.2f}ms")
            print(f"  Min: {min(latencies):.2f}ms")
            print(f"  Max: {max(latencies):.2f}ms")
            print(f"  P50: {sorted_lat[int(n*0.5)]:.2f}ms")
            print(f"  P95: {sorted_lat[int(n*0.95)]:.2f}ms")
            print(f"  P99: {sorted_lat[min(int(n*0.99),n-1)]:.2f}ms")
            print(f"  FPS: {1000.0/avg:.1f}")
        engine.cleanup()
        print("\n"+"="*66)
    def _run_pipeline(self):
        if not Path(self.config.model_path).exists():
            print(f"Error: Model not found: {self.config.model_path}")
            return
        print(f"\nLoading model: {self.config.model_path}")
        loader=AdvantechModelLoader(self.config,self.logger)
        try:
            engine=loader.load(self.config.model_path,self.config.model_format)
            engine.warmup(iterations=self.config.warmup_iterations)
            print(f"  Task: {engine.task_type.value}")
            print(f"  Format: {loader.detect_format(self.config.model_path).value}")
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return
        if is_shutdown():
            engine.cleanup()
            return
        source=AdvantechVideoSource()
        try:
            if self.config.input_source==AdvantechInputSource.CAMERA:
                device=self.config.camera_device
                if isinstance(device,int):
                    device_num=device
                elif device.startswith('/dev/video'):
                    device_num=int(device.replace('/dev/video',''))
                elif device.startswith('csi://'):
                    device_num=-1
                elif device.isdigit():
                    device_num=int(device)
                else:
                    device_num=0
                print(f"\nOpening camera: {'CSI' if device_num==-1 else f'/dev/video{device_num}'}")
                if not source.open_camera(device_num,self.config.camera_width,self.config.camera_height,self.config.camera_format,self.config.camera_fps):
                    engine.cleanup()
                    print("Error: Failed to open camera")
                    return
            elif self.config.input_source==AdvantechInputSource.RTSP:
                print(f"\nOpening RTSP: {self.config.input_path}")
                if not source.open_rtsp(self.config.input_path):
                    engine.cleanup()
                    print("Error: Failed to open RTSP stream")
                    return
            elif self.config.input_source==AdvantechInputSource.FILE:
                print(f"\nOpening file: {self.config.input_path}")
                if not source.open_file(self.config.input_path):
                    engine.cleanup()
                    print("Error: Failed to open video file")
                    return
        except Exception as e:
            engine.cleanup()
            print(f"Error opening source: {e}")
            return
        print(f"  Resolution: {source.width}x{source.height}")
        if source.fps>0:
            print(f"  FPS: {source.fps}")
        if source.is_video_file:
            print(f"  Frames: {source.total_frames}")
        if is_shutdown():
            source.release()
            engine.cleanup()
            return
        self._pipeline=AdvantechPipeline(self.config,self.logger,source,engine)
        if source.is_video_file and self.config.show_display:
            print(f"\n{'='*66}")
            print("  VIDEO FILE MODE - Window opens in 3 seconds")
            print("  Maximize window now. Press 'q' to quit.")
            print("="*66)
            cv2.namedWindow(WINDOW_NAME,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME,1280,720)
            for i in range(3,0,-1):
                if is_shutdown():
                    source.release()
                    engine.cleanup()
                    return
                wait_frame=np.zeros((720,1280,3),dtype=np.uint8)
                cv2.putText(wait_frame,f"Starting in {i}...",(450,360),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),3)
                cv2.imshow(WINDOW_NAME,wait_frame)
                cv2.waitKey(1000)
        print(f"\n{'='*66}")
        print("  Pipeline started. Press 'q' in window or Ctrl+C to stop.")
        print("="*66+"\n")
        self._pipeline.start()
        while self._pipeline.is_running() and not is_shutdown():
            time.sleep(0.1)
        total_fps=self._pipeline.get_total_fps()
        total_frames=self._pipeline.get_total_frames()
        self._pipeline.stop()
        engine.cleanup()
        print(f"\n{'='*66}")
        print(f"  COMPLETE")
        print(f"  Frames: {total_frames}")
        print(f"  Average FPS: {total_fps:.2f}")
        if self.config.save_video and self.config.output_path:
            print(f"  Output: {self.config.output_path}")
        print("="*66+"\n")

def main():
    import argparse
    parser=argparse.ArgumentParser(description="Advantech YOLO Inference Pipeline",formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list-cameras
  %(prog)s --model yolov8n.engine --source 0
  %(prog)s --model yolov8n.onnx --source video.mp4
  %(prog)s --model yolov8n-seg.engine --source 0 --task segmentation
  %(prog)s --benchmark --model yolov8n.engine
  %(prog)s --dryrun --model yolov8n.engine
""")
    parser.add_argument("--list-cameras",action="store_true",help="List available cameras")
    parser.add_argument("--dryrun",action="store_true",help="Verify system setup")
    parser.add_argument("--benchmark",action="store_true",help="Run inference benchmark")
    parser.add_argument("--model","-m",type=str,help="Model path")
    parser.add_argument("--format","-f",choices=["pt","onnx","trt"],help="Model format")
    parser.add_argument("--source","-s",type=str,help="Input source")
    parser.add_argument("--task","-t",choices=["detection","classification","segmentation"],default="detection",help="Task type")
    parser.add_argument("--precision",choices=["fp32","fp16","int8"],default="fp16",help="Precision")
    parser.add_argument("--device",type=int,default=0,help="GPU device ID")
    parser.add_argument("--save-video",action="store_true",help="Save output video")
    parser.add_argument("--output","-o",type=str,default="./output",help="Output path")
    parser.add_argument("--no-display",action="store_true",help="Disable display")
    parser.add_argument("--conf",type=float,default=0.25,help="Confidence threshold")
    parser.add_argument("--iou",type=float,default=0.45,help="IoU threshold")
    parser.add_argument("--max-memory",type=int,default=6000,help="Max GPU memory (MB)")
    args=parser.parse_args()
    cli=AdvantechCLI()
    cli.config.precision=AdvantechPrecision(args.precision)
    cli.config.gpu_device=args.device
    cli.config.save_video=args.save_video
    cli.config.output_path=args.output
    cli.config.show_display=not args.no_display
    cli.config.max_memory_mb=args.max_memory
    cli.config.confidence_threshold=args.conf
    cli.config.iou_threshold=args.iou
    cli.config.task_type={"detection":AdvantechTaskType.DETECTION,"classification":AdvantechTaskType.CLASSIFICATION,"segmentation":AdvantechTaskType.SEGMENTATION}[args.task]
    if args.format:
        format_map={"pt":AdvantechModelFormat.PYTORCH,"onnx":AdvantechModelFormat.ONNX,"trt":AdvantechModelFormat.TENSORRT}
        cli.config.model_format=format_map.get(args.format)
    if args.list_cameras:
        cli.list_cameras()
    elif args.dryrun:
        cli.run_dryrun(args.model)
    elif args.benchmark and args.model:
        cli.run_benchmark(args.model)
    elif args.model and args.source:
        cli.config.model_path=args.model
        cli.config.input_path=args.source
        if args.source.startswith("rtsp://"):
            cli.config.input_source=AdvantechInputSource.RTSP
        elif args.source.startswith("/dev/video") or args.source.isdigit():
            cli.config.input_source=AdvantechInputSource.CAMERA
            if args.source.isdigit():
                cli.config.camera_device=f"/dev/video{args.source}"
            else:
                cli.config.camera_device=args.source
            cli.config.camera_width=1280
            cli.config.camera_height=720
            cli.config.camera_format="MJPG"
            cli.config.camera_fps=30
        elif args.source.startswith("csi://"):
            cli.config.input_source=AdvantechInputSource.CAMERA
            cli.config.camera_device=args.source
            cli.config.camera_width=1280
            cli.config.camera_height=720
            cli.config.camera_format="RG10"
            cli.config.camera_fps=30
        else:
            cli.config.input_source=AdvantechInputSource.FILE
        print_banner()
        cli._run_pipeline()
    else:
        cli.run_interactive()

if __name__=="__main__":
    main()
