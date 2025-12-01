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
__title__="Advantech YOLO Inference Pipeline"
__author__="Samir Singh"
__copyright__="Copyright (c) 2024-2025 Advantech Corporation. All Rights Reserved."
__license__="Proprietary - Advantech Corporation"
__version__="2.0.1"
__build_date__="2025-01-15"
__maintainer__="Samir Singh"
import os,sys,time,signal,hashlib,threading,glob,subprocess,select
from pathlib import Path
from typing import Optional,Dict,Any,List,Tuple,Union
import numpy as np
from advantech_core import (TORCH_AVAILABLE,TRT_AVAILABLE,PYCUDA_AVAILABLE,ONNX_AVAILABLE,GST_AVAILABLE,FLASK_AVAILABLE,cv2,torch,trt,cuda,ort,Gst,GstApp,GLib,Flask,jsonify,init_pycuda,COCO_CLASSES,IMAGENET_CLASSES,get_class_name,get_class_color,get_class_names,MAX_MEMORY_MB,AdvantechTaskType,AdvantechModelFormat,AdvantechInputSource,AdvantechPrecision,AdvantechDropPolicy,AdvantechCameraFormat,AdvantechCameraInfo,AdvantechConfig,AdvantechMetrics,AdvantechDetection,AdvantechClassification,AdvantechFrame,AdvantechMemoryManager,AdvantechLogger,AdvantechMetricsCollector,AdvantechRingBuffer,AdvantechEngine)
class AdvantechModelLoader:
    def __init__(self,config:AdvantechConfig,logger:AdvantechLogger):
        self.config=config
        self.logger=logger
        Path(self.config.trt_cache_path).mkdir(parents=True,exist_ok=True)
        self._memory_manager=AdvantechMemoryManager()
    def detect_format(self,model_path:str)->AdvantechModelFormat:
        suffix=Path(model_path).suffix.lower()
        if suffix in [".pt",".pth"]:
            return AdvantechModelFormat.PYTORCH
        elif suffix==".onnx":
            return AdvantechModelFormat.ONNX
        elif suffix in [".trt",".engine",".plan"]:
            return AdvantechModelFormat.TENSORRT
        raise ValueError(f"Unsupported model format: {suffix}")
    def _compute_model_hash(self,model_path:str)->str:
        hasher=hashlib.md5()
        with open(model_path,'rb') as f:
            for chunk in iter(lambda:f.read(65536),b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]
    def _get_compute_capability(self)->str:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            major,minor=torch.cuda.get_device_capability(self.config.gpu_device)
            return f"{major}{minor}"
        return "87"
    def _get_engine_path(self,model_path:str)->str:
        model_hash=self._compute_model_hash(model_path)
        cc=self._get_compute_capability()
        model_name=Path(model_path).stem
        return str(Path(self.config.trt_cache_path)/f"{model_name}_{model_hash}_{self.config.precision.value}_sm{cc}.engine")
    def _detect_task_from_filename(self,model_path:str)->Optional[AdvantechTaskType]:
        model_name=Path(model_path).stem.lower()
        if '-seg' in model_name or '_seg' in model_name or model_name.endswith('seg'):
            return AdvantechTaskType.SEGMENTATION
        elif '-cls' in model_name or '_cls' in model_name or model_name.endswith('cls'):
            return AdvantechTaskType.CLASSIFICATION
        return None
    def load(self,model_path:str,model_format:Optional[AdvantechModelFormat]=None)->"AdvantechEngine":
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if model_format is None:
            model_format=self.detect_format(model_path)
        detected_task=self._detect_task_from_filename(model_path)
        if detected_task and self.config.task_type==AdvantechTaskType.DETECTION:
            self.config.task_type=detected_task
        self._memory_manager.force_cleanup()
        if model_format==AdvantechModelFormat.TENSORRT:
            return self._load_as_tensorrt(model_path)
        elif model_format==AdvantechModelFormat.ONNX:
            return self._load_as_onnx(model_path)
        elif model_format==AdvantechModelFormat.PYTORCH:
            return self._load_as_pytorch(model_path)
        raise RuntimeError(f"Failed to load model: {model_path}")
    def _load_as_tensorrt(self,model_path:str)->"AdvantechEngine":
        source_format=self.detect_format(model_path)
        if source_format==AdvantechModelFormat.TENSORRT:
            return AdvantechTensorRTEngine(self.config,self.logger,model_path)
        elif source_format==AdvantechModelFormat.ONNX:
            if not TRT_AVAILABLE:
                raise RuntimeError("TensorRT not available")
            engine_path=self._get_engine_path(model_path)
            if Path(engine_path).exists():
                try:
                    return AdvantechTensorRTEngine(self.config,self.logger,engine_path)
                except:
                    pass
            return self._convert_onnx_to_trt(model_path,engine_path)
        elif source_format==AdvantechModelFormat.PYTORCH:
            onnx_path=self._convert_pt_to_onnx(model_path)
            engine_path=self._get_engine_path(onnx_path)
            return self._convert_onnx_to_trt(onnx_path,engine_path)
        raise RuntimeError("Cannot convert to TensorRT")
    def _load_as_onnx(self,model_path:str)->"AdvantechEngine":
        source_format=self.detect_format(model_path)
        if source_format==AdvantechModelFormat.ONNX:
            if not ONNX_AVAILABLE:
                raise RuntimeError("ONNX Runtime not available")
            return AdvantechOnnxEngine(self.config,self.logger,model_path)
        elif source_format==AdvantechModelFormat.PYTORCH:
            onnx_path=self._convert_pt_to_onnx(model_path)
            return AdvantechOnnxEngine(self.config,self.logger,onnx_path)
        raise RuntimeError("Cannot run as ONNX")
    def _load_as_pytorch(self,model_path:str)->"AdvantechEngine":
        source_format=self.detect_format(model_path)
        if source_format==AdvantechModelFormat.PYTORCH:
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch not available")
            return AdvantechPyTorchEngine(self.config,self.logger,model_path)
        raise RuntimeError("Cannot run as PyTorch")
    def _convert_pt_to_onnx(self,pt_path:str)->str:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        onnx_path=str(Path(self.config.trt_cache_path)/f"{Path(pt_path).stem}.onnx")
        if Path(onnx_path).exists():
            return onnx_path
        try:
            from ultralytics import YOLO
            model=YOLO(pt_path)
            imgsz=224 if self.config.task_type==AdvantechTaskType.CLASSIFICATION else self.config.input_width
            model.export(format="onnx",imgsz=imgsz,opset=12,simplify=True)
            exported_path=str(Path(pt_path).with_suffix('.onnx'))
            if Path(exported_path).exists():
                import shutil
                shutil.move(exported_path,onnx_path)
            return onnx_path
        except ImportError:
            raise RuntimeError("Ultralytics required for .pt to .onnx conversion")
    def _convert_onnx_to_trt(self,onnx_path:str,engine_path:str)->"AdvantechEngine":
        if not TRT_AVAILABLE:
            return AdvantechOnnxEngine(self.config,self.logger,onnx_path)
        trt_logger=trt.Logger(trt.Logger.WARNING)
        builder=trt.Builder(trt_logger)
        network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser=trt.OnnxParser(network,trt_logger)
        with open(onnx_path,'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    self.logger.error(f"ONNX parse error: {parser.get_error(i)}","ModelLoader")
                raise RuntimeError("Failed to parse ONNX model")
        config=builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,512<<20)
        if self.config.precision==AdvantechPrecision.FP16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.config.precision==AdvantechPrecision.INT8 and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
        input_tensor=network.get_input(0)
        input_shape=input_tensor.shape
        if -1 in input_shape:
            profile=builder.create_optimization_profile()
            h=224 if self.config.task_type==AdvantechTaskType.CLASSIFICATION else self.config.input_height
            w=224 if self.config.task_type==AdvantechTaskType.CLASSIFICATION else self.config.input_width
            profile.set_shape(input_tensor.name,(1,3,h,w),(1,3,h,w),(1,3,h,w))
            config.add_optimization_profile(profile)
        serialized_engine=builder.build_serialized_network(network,config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        with open(engine_path,'wb') as f:
            f.write(serialized_engine)
        del serialized_engine,network,parser,builder
        self._memory_manager.force_cleanup()
        return AdvantechTensorRTEngine(self.config,self.logger,engine_path)
class AdvantechTensorRTEngine(AdvantechEngine):
    def __init__(self,config:AdvantechConfig,logger:AdvantechLogger,engine_path:str):
        super().__init__(config,logger)
        self.engine_path=engine_path
        self._stream=None
        self._context=None
        self._engine=None
        self._inputs=[]
        self._outputs=[]
        self._load_engine()
    def _load_engine(self):
        trt_logger=trt.Logger(trt.Logger.WARNING)
        runtime=trt.Runtime(trt_logger)
        with open(self.engine_path,'rb') as f:
            self._engine=runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {self.engine_path}")
        self._context=self._engine.create_execution_context()
        if init_pycuda() and cuda is not None:
            self._stream=cuda.Stream()
        for i in range(self._engine.num_io_tensors):
            name=self._engine.get_tensor_name(i)
            dtype=trt.nptype(self._engine.get_tensor_dtype(name))
            shape=self._engine.get_tensor_shape(name)
            size=abs(trt.volume(shape))
            host_mem=self._memory_manager.allocate_pinned_buffer((size,),dtype)
            if host_mem is None:
                host_mem=np.empty(size,dtype=dtype)
            device_mem=None
            if PYCUDA_AVAILABLE and cuda is not None:
                try:
                    device_mem=cuda.mem_alloc(host_mem.nbytes)
                except:
                    pass
            binding={"name":name,"dtype":dtype,"shape":list(shape),"host":host_mem,"device":device_mem}
            if self._engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT:
                self._inputs.append(binding)
                if len(shape)==4:
                    self._model_input_height=shape[2]
                    self._model_input_width=shape[3]
            else:
                self._outputs.append(binding)
        if len(self._outputs)>=2 and self.task_type==AdvantechTaskType.DETECTION:
            self.task_type=AdvantechTaskType.SEGMENTATION
    def infer(self,frame:np.ndarray)->Union[List[AdvantechDetection],AdvantechClassification]:
        original_shape=frame.shape[:2]
        input_data=self.preprocess(frame)
        expected_dtype=self._inputs[0]["dtype"]
        if input_data.dtype!=expected_dtype:
            input_data=input_data.astype(expected_dtype)
        np.copyto(self._inputs[0]["host"],input_data.ravel())
        if PYCUDA_AVAILABLE and self._stream is not None and self._inputs[0]["device"] is not None:
            cuda.memcpy_htod_async(self._inputs[0]["device"],self._inputs[0]["host"],self._stream)
            self._context.set_tensor_address(self._inputs[0]["name"],int(self._inputs[0]["device"]))
            for output in self._outputs:
                if output["device"] is not None:
                    self._context.set_tensor_address(output["name"],int(output["device"]))
            self._context.execute_async_v3(stream_handle=self._stream.handle)
            for output in self._outputs:
                if output["device"] is not None:
                    cuda.memcpy_dtoh_async(output["host"],output["device"],self._stream)
            self._stream.synchronize()
        else:
            self._context.execute_v2([inp["host"] for inp in self._inputs]+[out["host"] for out in self._outputs])
        det_output=self._outputs[0]["host"].reshape(self._outputs[0]["shape"])
        if self.task_type==AdvantechTaskType.CLASSIFICATION:
            return self.postprocess_classification(det_output)
        elif self.task_type==AdvantechTaskType.SEGMENTATION and len(self._outputs)>=2:
            mask_output=self._outputs[1]["host"].reshape(self._outputs[1]["shape"])
            return self.postprocess_segmentation(det_output,mask_output,original_shape)
        return self.postprocess(det_output,original_shape,num_masks=0)
    def warmup(self,iterations:int=5):
        if self._warmed_up:
            return
        dummy=np.random.randint(0,255,(self._model_input_height,self._model_input_width,3),dtype=np.uint8)
        for _ in range(iterations):
            self.infer(dummy)
        self._warmed_up=True
    def cleanup(self):
        for inp in self._inputs:
            if inp["device"] is not None:
                try:
                    inp["device"].free()
                except:
                    pass
        for out in self._outputs:
            if out["device"] is not None:
                try:
                    out["device"].free()
                except:
                    pass
        self._context=None
        self._engine=None
        self._stream=None
        self._memory_manager.force_cleanup()
class AdvantechOnnxEngine(AdvantechEngine):
    def __init__(self,config:AdvantechConfig,logger:AdvantechLogger,onnx_path:str):
        super().__init__(config,logger)
        self.onnx_path=onnx_path
        self._session=None
        self._input_name=None
        self._input_dtype=np.float32
        self._num_outputs=1
        self._is_seg_model=False
        self._load_model()
    def _load_model(self):
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available")
        sess_options=ort.SessionOptions()
        sess_options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads=2
        sess_options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.log_severity_level=3
        providers=[]
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append(('CUDAExecutionProvider',{'device_id':self.config.gpu_device,'arena_extend_strategy':'kSameAsRequested','gpu_mem_limit':self.config.max_memory_mb*1024*1024}))
        providers.append('CPUExecutionProvider')
        self._session=ort.InferenceSession(self.onnx_path,sess_options,providers=providers)
        input_info=self._session.get_inputs()[0]
        self._input_name=input_info.name
        input_shape=input_info.shape
        if len(input_shape)==4 and isinstance(input_shape[2],int) and isinstance(input_shape[3],int):
            self._model_input_height=input_shape[2]
            self._model_input_width=input_shape[3]
        onnx_dtype=input_info.type
        self._input_dtype=np.float16 if 'float16' in onnx_dtype.lower() else np.float32
        outputs=self._session.get_outputs()
        self._num_outputs=len(outputs)
        if self._num_outputs>=2:
            out0_shape=outputs[0].shape
            out1_shape=outputs[1].shape
            if len(out1_shape)==4 and out1_shape[1]==32:
                self._is_seg_model=True
                if self.task_type==AdvantechTaskType.DETECTION:
                    self.task_type=AdvantechTaskType.SEGMENTATION
        if self._num_outputs==1 and len(outputs[0].shape)==2:
            out_shape=outputs[0].shape
            if out_shape[1] is not None and out_shape[1]>=100:
                if self.task_type!=AdvantechTaskType.CLASSIFICATION:
                    self.task_type=AdvantechTaskType.CLASSIFICATION
    def infer(self,frame:np.ndarray)->Union[List[AdvantechDetection],AdvantechClassification]:
        original_shape=frame.shape[:2]
        input_data=self.preprocess(frame)
        if input_data.dtype!=self._input_dtype:
            input_data=input_data.astype(self._input_dtype)
        outputs=self._session.run(None,{self._input_name:input_data})
        if self.task_type==AdvantechTaskType.CLASSIFICATION:
            return self.postprocess_classification(outputs[0])
        elif self.task_type==AdvantechTaskType.SEGMENTATION and self._is_seg_model and len(outputs)>=2:
            return self.postprocess_segmentation(outputs[0],outputs[1],original_shape)
        return self.postprocess(outputs[0],original_shape,num_masks=0)
    def warmup(self,iterations:int=5):
        if self._warmed_up:
            return
        dummy=np.random.randint(0,255,(self._model_input_height,self._model_input_width,3),dtype=np.uint8)
        for _ in range(iterations):
            self.infer(dummy)
        self._warmed_up=True
    def cleanup(self):
        self._session=None
        self._memory_manager.force_cleanup()
class AdvantechPyTorchEngine(AdvantechEngine):
    def __init__(self,config:AdvantechConfig,logger:AdvantechLogger,model_path:str):
        super().__init__(config,logger)
        self.model_path=model_path
        self._model=None
        self._device=None
        self._use_ultralytics=False
        self._load_model()
    def _load_model(self):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        torch.backends.cudnn.benchmark=True
        self._device=torch.device(f"cuda:{self.config.gpu_device}" if torch.cuda.is_available() else "cpu")
        try:
            from ultralytics import YOLO
            self._model=YOLO(self.model_path)
            self._use_ultralytics=True
            if hasattr(self._model,'task'):
                if self._model.task=='classify':
                    self.task_type=AdvantechTaskType.CLASSIFICATION
                    self._model_input_height=224
                    self._model_input_width=224
                elif self._model.task=='segment':
                    self.task_type=AdvantechTaskType.SEGMENTATION
        except ImportError:
            self._model=torch.load(self.model_path,map_location=self._device)
            if isinstance(self._model,dict):
                self._model=self._model.get('model',self._model.get('ema',self._model))
            self._model=self._model.float().fuse().eval().to(self._device)
            self._use_ultralytics=False
    def infer(self,frame:np.ndarray)->Union[List[AdvantechDetection],AdvantechClassification]:
        if self._use_ultralytics:
            results=self._model(frame,conf=self.config.conf_threshold,iou=self.config.iou_threshold,verbose=False)
            class_names=self.get_class_names()
            for r in results:
                if hasattr(r,'probs') and r.probs is not None:
                    probs=r.probs.data.cpu().numpy()
                    top_indices=np.argsort(probs)[::-1][:5]
                    return AdvantechClassification(top_classes=[(int(idx),class_names[idx] if idx<len(class_names) else f"class_{idx}",float(probs[idx])) for idx in top_indices])
                detections=[]
                if hasattr(r,'boxes') and r.boxes is not None:
                    boxes=r.boxes
                    masks=r.masks if hasattr(r,'masks') and r.masks is not None else None
                    for i in range(len(boxes)):
                        xyxy=boxes.xyxy[i].cpu().numpy()
                        conf=float(boxes.conf[i].cpu().numpy())
                        cls=int(boxes.cls[i].cpu().numpy())
                        mask=masks.data[i].cpu().numpy() if masks is not None and i<len(masks) else None
                        detections.append(AdvantechDetection(bbox=tuple(xyxy.tolist()),confidence=conf,class_id=cls,class_name=class_names[cls] if cls<len(class_names) else f"class_{cls}",mask=mask))
                return detections
            return []
        original_shape=frame.shape[:2]
        input_data=self.preprocess(frame)
        input_tensor=torch.from_numpy(input_data).to(self._device)
        with torch.no_grad():
            if self.config.precision==AdvantechPrecision.FP16:
                with torch.cuda.amp.autocast():
                    outputs=self._model(input_tensor)
            else:
                outputs=self._model(input_tensor)
        if isinstance(outputs,(tuple,list)):
            outputs=outputs[0]
        outputs_np=outputs.cpu().numpy()
        if self.task_type==AdvantechTaskType.CLASSIFICATION:
            return self.postprocess_classification(outputs_np)
        return self.postprocess(outputs_np,original_shape,num_masks=0)
    def warmup(self,iterations:int=5):
        if self._warmed_up:
            return
        dummy=np.random.randint(0,255,(self._model_input_height,self._model_input_width,3),dtype=np.uint8)
        for _ in range(iterations):
            self.infer(dummy)
        self._warmed_up=True
    def cleanup(self):
        self._model=None
        self._memory_manager.force_cleanup()
class AdvantechVideoSource:
    def __init__(self,config:AdvantechConfig,logger:AdvantechLogger):
        self.config=config
        self.logger=logger
        self._capture=None
        self._width=0
        self._height=0
        self._fps=0
        self._frame_id=0
    def discover_cameras(self)->List[AdvantechCameraInfo]:
        cameras=[]
        if self._check_csi_camera():
            cameras.append(AdvantechCameraInfo(device_path="csi://0",name="CSI Camera (nvarguscamerasrc)",formats=[AdvantechCameraFormat("NV12",1920,1080,[30,60]),AdvantechCameraFormat("NV12",1280,720,[30,60,120]),AdvantechCameraFormat("NV12",640,480,[30,60,120])],is_working=True,is_csi=True))
        import re
        video_devices=sorted(glob.glob("/dev/video*"))
        for dev in video_devices:
            info=self._probe_v4l2_device(dev)
            if info and info.formats:
                info.is_working=self._test_camera_working(dev)
                if info.is_working:
                    cameras.append(info)
        return cameras
    def _check_csi_camera(self)->bool:
        if not GST_AVAILABLE:
            return False
        try:
            pipeline=Gst.parse_launch("nvarguscamerasrc sensor-id=0 ! fakesink")
            pipeline.set_state(Gst.State.PAUSED)
            time.sleep(0.3)
            state=pipeline.get_state(500*Gst.MSECOND)[1]
            pipeline.set_state(Gst.State.NULL)
            return state==Gst.State.PAUSED
        except:
            return False
    def _probe_v4l2_device(self,device_path:str)->Optional[AdvantechCameraInfo]:
        try:
            result=subprocess.run(["v4l2-ctl","-d",device_path,"--list-formats-ext"],capture_output=True,text=True,timeout=3)
            if result.returncode!=0:
                return None
            name_result=subprocess.run(["v4l2-ctl","-d",device_path,"--info"],capture_output=True,text=True,timeout=3)
            name=device_path
            if name_result.returncode==0:
                for line in name_result.stdout.split('\n'):
                    if 'Card type' in line and ':' in line:
                        name=line.split(':',1)[1].strip()
                        break
            formats=self._parse_v4l2_formats(result.stdout)
            return AdvantechCameraInfo(device_path=device_path,name=name,formats=formats,is_working=False,is_csi=False)
        except:
            return None
    def _parse_v4l2_formats(self,output:str)->List[AdvantechCameraFormat]:
        import re
        formats=[]
        current_format=None
        for line in output.split('\n'):
            fmt_match=re.search(r"'(\w+)'",line)
            if fmt_match and any(x in line for x in ['YUYV','MJPG','MJPEG','YU12','NV12']):
                current_format=fmt_match.group(1)
                if current_format=='MJPG':
                    current_format='MJPEG'
            size_match=re.search(r'Size:.*?(\d+)x(\d+)',line)
            if size_match and current_format:
                w,h=int(size_match.group(1)),int(size_match.group(2))
                fps_matches=re.findall(r'(\d+)\.000 fps',output[output.find(f"{w}x{h}"):output.find(f"{w}x{h}")+500])
                fps_list=[int(fps) for fps in fps_matches[:5]] or [30]
                formats.append(AdvantechCameraFormat(pixel_format=current_format,width=w,height=h,fps=fps_list))
        return formats
    def _test_camera_working(self,device_path:str)->bool:
        try:
            cap=cv2.VideoCapture(device_path,cv2.CAP_V4L2)
            if not cap.isOpened():
                return False
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            ret,frame=cap.read()
            cap.release()
            return ret and frame is not None
        except:
            return False
    def open_webcam(self,device:str,width:int,height:int,pixel_format:str,fps:int)->bool:
        self._width,self._height,self._fps=width,height,fps
        if device.startswith("csi://"):
            return self._open_csi_camera(width,height,fps)
        return self._open_v4l2_camera(device,width,height,pixel_format,fps)
    def _open_csi_camera(self,width:int,height:int,fps:int)->bool:
        if not GST_AVAILABLE:
            return False
        gst_str=f"nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1,format=NV12 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=2 drop=true"
        self._capture=cv2.VideoCapture(gst_str,cv2.CAP_GSTREAMER)
        return self._capture.isOpened()
    def _open_v4l2_camera(self,device:str,width:int,height:int,pixel_format:str,fps:int)->bool:
        if GST_AVAILABLE and pixel_format=="MJPEG":
            gst_str=f"v4l2src device={device} ! image/jpeg,width={width},height={height},framerate={fps}/1 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=2 drop=true"
            self._capture=cv2.VideoCapture(gst_str,cv2.CAP_GSTREAMER)
            if self._capture.isOpened():
                return True
        if GST_AVAILABLE:
            gst_str=f"v4l2src device={device} ! video/x-raw,width={width},height={height},framerate={fps}/1 ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=2 drop=true"
            self._capture=cv2.VideoCapture(gst_str,cv2.CAP_GSTREAMER)
            if self._capture.isOpened():
                return True
        self._capture=cv2.VideoCapture(device,cv2.CAP_V4L2)
        if self._capture.isOpened():
            if pixel_format=="MJPEG":
                self._capture.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH,width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
            self._capture.set(cv2.CAP_PROP_FPS,fps)
            self._width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return True
        return False
    def open_rtsp(self,url:str)->bool:
        if GST_AVAILABLE:
            pipeline_hw=f"rtspsrc location={url} latency=100 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
            try:
                self._capture=cv2.VideoCapture(pipeline_hw,cv2.CAP_GSTREAMER)
                if self._capture.isOpened():
                    self._probe_stream_properties()
                    return True
            except:
                if self._capture:
                    self._capture.release()
                    self._capture=None
            pipeline_sw=f"rtspsrc location={url} latency=100 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
            try:
                self._capture=cv2.VideoCapture(pipeline_sw,cv2.CAP_GSTREAMER)
                if self._capture.isOpened():
                    self._probe_stream_properties()
                    return True
            except:
                if self._capture:
                    self._capture.release()
                    self._capture=None
        self._capture=cv2.VideoCapture(url)
        if self._capture.isOpened():
            self._probe_stream_properties()
            return True
        return False
    def open_file(self,file_path:str)->bool:
        if not Path(file_path).exists():
            return False
        self._capture=cv2.VideoCapture(file_path)
        if self._capture.isOpened():
            self._probe_stream_properties()
            return True
        return False
    def _probe_stream_properties(self):
        if self._capture:
            self._width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._fps=int(self._capture.get(cv2.CAP_PROP_FPS)) or 30
    def read(self)->Optional[AdvantechFrame]:
        if self._capture is None or not self._capture.isOpened():
            return None
        ret,frame=self._capture.read()
        if not ret or frame is None:
            return None
        self._frame_id+=1
        return AdvantechFrame(data=frame,timestamp=time.perf_counter(),frame_id=self._frame_id,width=frame.shape[1],height=frame.shape[0])
    @property
    def width(self)->int:
        return self._width
    @property
    def height(self)->int:
        return self._height
    @property
    def fps(self)->int:
        return self._fps
    def release(self):
        if self._capture:
            self._capture.release()
            self._capture=None
class AdvantechGstEncoder:
    def __init__(self,config:AdvantechConfig,logger:AdvantechLogger,width:int,height:int,fps:int):
        self.config=config
        self.logger=logger
        self.width=width
        self.height=height
        self.fps=fps
        self._writer=None
        self._output_path=None
    def start_file_output(self,output_path:str)->bool:
        self._output_path=output_path
        Path(output_path).parent.mkdir(parents=True,exist_ok=True)
        if GST_AVAILABLE:
            gst_str=f"appsrc ! videoconvert ! video/x-raw,format=I420 ! nvv4l2h264enc bitrate=4000000 ! h264parse ! mp4mux ! filesink location={output_path}"
            self._writer=cv2.VideoWriter(gst_str,cv2.CAP_GSTREAMER,0,self.fps,(self.width,self.height))
            if self._writer.isOpened():
                return True
        self._writer=cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'),self.fps,(self.width,self.height))
        return self._writer.isOpened()
    def write(self,frame:np.ndarray):
        if self._writer and self._writer.isOpened():
            self._writer.write(frame)
    def release(self):
        if self._writer:
            self._writer.release()
            self._writer=None
class AdvantechOverlayRenderer:
    def __init__(self,config:AdvantechConfig):
        self.config=config
    def render(self,frame:np.ndarray,detections:List[AdvantechDetection],metrics:Optional[AdvantechMetrics]=None,classification:Optional[AdvantechClassification]=None)->np.ndarray:
        output=frame.copy()
        for det in detections:
            color=get_class_color(det.class_id)
            x1,y1,x2,y2=map(int,det.bbox)
            if det.mask is not None:
                try:
                    mask=det.mask
                    if mask.shape[:2]!=output.shape[:2]:
                        mask=cv2.resize(mask,(output.shape[1],output.shape[0]))
                    mask_bool=mask>0.5
                    overlay=output.copy()
                    overlay[mask_bool]=(overlay[mask_bool]*0.5+np.array(color)*0.5).astype(np.uint8)
                    output=overlay
                except:
                    pass
            cv2.rectangle(output,(x1,y1),(x2,y2),color,2)
            label=f"{det.class_name} {det.confidence:.2f}"
            (lw,lh),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
            cv2.rectangle(output,(x1,y1-lh-8),(x1+lw,y1),color,-1)
            cv2.putText(output,label,(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        if classification is not None:
            y_offset=60 if metrics else 25
            h,w=output.shape[:2]
            overlay=output.copy()
            cv2.rectangle(overlay,(w-320,y_offset-25),(w-10,y_offset+len(classification.top5)*28+5),(0,0,0),-1)
            output=cv2.addWeighted(overlay,0.6,output,0.4,0)
            cv2.putText(output,"Classification:",(w-310,y_offset),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            y_offset+=28
            for i,(class_id,class_name,prob) in enumerate(classification.top5):
                color=(0,255,0) if i==0 else (200,200,200)
                cv2.putText(output,f"{i+1}. {class_name}: {prob*100:.1f}%",(w-310,y_offset+i*28),cv2.FONT_HERSHEY_SIMPLEX,0.55,color,1)
        if metrics:
            info=[f"FPS: {metrics.fps:.1f}",f"Latency: {metrics.avg_latency_ms:.1f}ms",f"GPU: {metrics.gpu_memory_used_mb:.0f}MB"]
            y=25
            for line in info:
                cv2.putText(output,line,(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                y+=22
        return output
class AdvantechPipeline:
    def __init__(self,config:AdvantechConfig,logger:AdvantechLogger,engine:AdvantechEngine,source:AdvantechVideoSource):
        self.config=config
        self.logger=logger
        self.engine=engine
        self.source=source
        self.metrics=AdvantechMetricsCollector()
        self.renderer=AdvantechOverlayRenderer(config)
        self._running=False
        self._capture_queue=AdvantechRingBuffer(config.queue_size,config.drop_policy)
        self._output_queue=AdvantechRingBuffer(config.queue_size,config.drop_policy)
        self._threads:List[threading.Thread]=[]
        self._encoder:Optional[AdvantechGstEncoder]=None
        self._memory_manager=AdvantechMemoryManager()
        self._stop_event=threading.Event()
    def start(self):
        self._running=True
        self._stop_event.clear()
        self.metrics.set_model_status("running")
        if self.config.save_video:
            output_file=str(Path(self.config.output_path)/f"output_{int(time.time())}.mp4")
            self._encoder=AdvantechGstEncoder(self.config,self.logger,self.source.width,self.source.height,self.source.fps)
            self._encoder.start_file_output(output_file)
        capture_thread=threading.Thread(target=self._capture_loop,daemon=True)
        capture_thread.start()
        self._threads.append(capture_thread)
        inference_thread=threading.Thread(target=self._inference_loop,daemon=True)
        inference_thread.start()
        self._threads.append(inference_thread)
        output_thread=threading.Thread(target=self._output_loop,daemon=True)
        output_thread.start()
        self._threads.append(output_thread)
    def _capture_loop(self):
        while self._running and not self._stop_event.is_set():
            frame=self.source.read()
            if frame is None:
                time.sleep(0.001)
                continue
            self._capture_queue.put(frame)
            self.metrics.update_queue_size("capture",self._capture_queue.size())
    def _inference_loop(self):
        while self._running and not self._stop_event.is_set():
            frame=self._capture_queue.get(timeout=0.1)
            if frame is None:
                continue
            if self._memory_manager.get_memory_usage_mb()>self.config.max_memory_mb*0.9:
                self._memory_manager.force_cleanup()
                self.metrics.record_drop()
                continue
            start_time=time.perf_counter()
            try:
                result=self.engine.infer(frame.data)
                if isinstance(result,AdvantechClassification):
                    frame.classification=result
                    frame.detections=[]
                else:
                    frame.detections=result
                    frame.classification=None
            except Exception as e:
                self.logger.error(f"Inference error: {e}","Pipeline")
                frame.detections=[]
                frame.classification=None
            frame.inference_time_ms=(time.perf_counter()-start_time)*1000
            frame.total_latency_ms=(time.perf_counter()-frame.timestamp)*1000
            self.metrics.record_latency(frame.total_latency_ms)
            self._output_queue.put(frame)
    def _output_loop(self):
        window_name="Advantech YOLO"
        if self.config.show_display:
            try:
                cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
            except:
                self.config.show_display=False
        while self._running and not self._stop_event.is_set():
            frame=self._output_queue.get(timeout=0.1)
            if frame is None:
                continue
            self.metrics.record_frame()
            current_metrics=self.metrics.get_metrics()
            rendered=self.renderer.render(frame.data,frame.detections,current_metrics,frame.classification)
            if self._encoder:
                self._encoder.write(rendered)
            if self.config.show_display:
                try:
                    cv2.imshow(window_name,rendered)
                    key=cv2.waitKey(1)&0xFF
                    if key==ord('q') or key==27:
                        self._stop_event.set()
                        self._running=False
                except:
                    pass
    def stop(self):
        self._running=False
        self._stop_event.set()
        self._capture_queue.close()
        self._output_queue.close()
        for thread in self._threads:
            thread.join(timeout=2.0)
        if self._encoder:
            self._encoder.release()
        self.source.release()
        self.engine.cleanup()
        if self.config.show_display:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        self.metrics.set_model_status("stopped")
    def is_running(self)->bool:
        return self._running and not self._stop_event.is_set()
    def get_metrics(self)->AdvantechMetrics:
        return self.metrics.get_metrics()
    def get_total_fps(self)->float:
        return self.metrics.get_total_fps()
    def get_total_frames(self)->int:
        return self.metrics.get_total_frames()
class AdvantechHealthServer:
    def __init__(self,config:AdvantechConfig,logger:AdvantechLogger,pipeline:AdvantechPipeline):
        self.config=config
        self.logger=logger
        self.pipeline=pipeline
    def start(self):
        if not FLASK_AVAILABLE:
            return
        app=Flask("AdvantechHealth")
        app.logger.disabled=True
        import logging as log
        log.getLogger('werkzeug').disabled=True
        @app.route('/health')
        def health():
            m=self.pipeline.get_metrics()
            return jsonify({"status":"healthy","fps":m.fps,"latency_ms":m.avg_latency_ms,"gpu_memory_mb":m.gpu_memory_used_mb})
        def run():
            try:
                app.run(host='0.0.0.0',port=self.config.health_port,threaded=True,use_reloader=False)
            except:
                pass
        threading.Thread(target=run,daemon=True).start()
class AdvantechBenchmark:
    def __init__(self,config:AdvantechConfig,logger:AdvantechLogger,engine:AdvantechEngine):
        self.config=config
        self.logger=logger
        self.engine=engine
    def run(self)->Dict[str,Any]:
        input_h,input_w=self.engine._model_input_height,self.engine._model_input_width
        dummy=np.random.randint(0,255,(input_h,input_w,3),dtype=np.uint8)
        for _ in range(self.config.warmup_iterations):
            self.engine.infer(dummy)
        latencies=[]
        for _ in range(self.config.benchmark_iterations):
            start=time.perf_counter()
            self.engine.infer(dummy)
            latencies.append((time.perf_counter()-start)*1000)
        sorted_lat=sorted(latencies)
        n=len(sorted_lat)
        return {"iterations":n,"avg_ms":sum(latencies)/n,"min_ms":min(latencies),"max_ms":max(latencies),"p50_ms":sorted_lat[int(n*0.5)],"p90_ms":sorted_lat[int(n*0.9)],"p95_ms":sorted_lat[int(n*0.95)],"p99_ms":sorted_lat[min(int(n*0.99),n-1)],"fps":1000.0/(sum(latencies)/n)}
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
║  {__copyright__}             ║
╚══════════════════════════════════════════════════════════════════════════════════╝"""
    print(banner)
class AdvantechCLI:
    def __init__(self):
        self.config=AdvantechConfig.from_env()
        self.logger=AdvantechLogger(self.config)
        self._shutdown_event=threading.Event()
        self._pipeline:Optional[AdvantechPipeline]=None
        signal.signal(signal.SIGINT,self._signal_handler)
        signal.signal(signal.SIGTERM,self._signal_handler)
    def _signal_handler(self,signum,frame):
        self._shutdown_event.set()
        if self._pipeline:
            self._pipeline.stop()
    def _get_choice(self,prompt:str,valid:List[str],default:str="1")->str:
        try:
            choice=input(f"{prompt} [{default}]: ").strip() or default
            return choice if choice in valid else default
        except (EOFError,KeyboardInterrupt):
            sys.exit(0)
    def _get_input(self,prompt:str,default:str="")->str:
        try:
            value=input(f"{prompt} [{default}]: ").strip()
            return value if value else default
        except (EOFError,KeyboardInterrupt):
            sys.exit(0)
    def run_interactive(self):
        print_banner()
        print("\n[1] Detection  [2] Classification  [3] Segmentation")
        task_choice=self._get_choice("Select task",["1","2","3"],"1")
        self.config.task_type={
            "1":AdvantechTaskType.DETECTION,
            "2":AdvantechTaskType.CLASSIFICATION,
            "3":AdvantechTaskType.SEGMENTATION
        }[task_choice]
        print("\n[1] PyTorch  [2] ONNX  [3] TensorRT")
        format_choice=self._get_choice("Select format",["1","2","3"],"1")
        self.config.model_format={
            "1":AdvantechModelFormat.PYTORCH,
            "2":AdvantechModelFormat.ONNX,
            "3":AdvantechModelFormat.TENSORRT
        }[format_choice]
        ext_map={"1":".pt","2":".onnx","3":".trt/.engine"}
        self.config.model_path=self._get_input(f"Model path ({ext_map[format_choice]})","")
        if not self.config.model_path:
            print("Error: Model path required")
            return
        print("\n[1] Webcam  [2] RTSP  [3] File")
        source_choice=self._get_choice("Select source",["1","2","3"],"1")
        self.config.input_source={
            "1":AdvantechInputSource.WEBCAM,
            "2":AdvantechInputSource.RTSP,
            "3":AdvantechInputSource.FILE
        }[source_choice]
        if self.config.input_source==AdvantechInputSource.WEBCAM:
            if not self._select_webcam():
                return
        elif self.config.input_source==AdvantechInputSource.RTSP:
            self.config.input_path=self._get_input("RTSP URL","rtsp://192.168.1.100:554/stream")
        elif self.config.input_source==AdvantechInputSource.FILE:
            self.config.input_path=self._get_input("Video file path","input.mp4")
        save_choice=self._get_choice("Save output? (y/n)",["y","n","Y","N"],"n")
        self.config.save_video=save_choice.lower()=="y"
        if self.config.save_video:
            self.config.output_path=self._get_input("Output directory","./output")
        display_choice=self._get_choice("Show display? (y/n)",["y","n","Y","N"],"y")
        self.config.show_display=display_choice.lower()=="y"
        self._run_pipeline()
    def _select_webcam(self)->bool:
        print("\nDiscovering cameras...")
        source=AdvantechVideoSource(self.config,self.logger)
        cameras=source.discover_cameras()
        if not cameras:
            print("No cameras found!")
            return False
        print(f"\nFound {len(cameras)} camera(s):")
        for i,cam in enumerate(cameras,1):
            status="[OK]" if cam.is_working else "[?]"
            cam_type="[CSI]" if cam.is_csi else "[USB]"
            print(f"  {i}) {cam.device_path} - {cam.name} {cam_type} {status}")
        cam_choice=self._get_choice("Select camera",[str(i) for i in range(1,len(cameras)+1)],"1")
        selected_camera=cameras[int(cam_choice)-1]
        self.config.camera_device=selected_camera.device_path
        resolutions=selected_camera.get_resolutions()
        if not resolutions:
            print("No resolutions available!")
            return False
        print(f"\nResolutions:")
        for i,(w,h) in enumerate(resolutions[:10],1):
            formats_at_res=selected_camera.get_formats_for_resolution(w,h)
            fmt_strs=", ".join([f.pixel_format for f in formats_at_res])
            print(f"  {i}) {w}x{h} [{fmt_strs}]")
        res_choice=self._get_choice("Select resolution",[str(i) for i in range(1,min(len(resolutions)+1,11))],"1")
        selected_res=resolutions[int(res_choice)-1]
        self.config.camera_width,self.config.camera_height=selected_res
        formats_at_res=selected_camera.get_formats_for_resolution(self.config.camera_width,self.config.camera_height)
        if len(formats_at_res)>1:
            print(f"\nFormats:")
            for i,fmt in enumerate(formats_at_res,1):
                fps_str="/".join(map(str,fmt.fps[:3]))
                print(f"  {i}) {fmt.pixel_format} @{fps_str}fps")
            fmt_choice=self._get_choice("Select format",[str(i) for i in range(1,len(formats_at_res)+1)],"1")
            selected_fmt=formats_at_res[int(fmt_choice)-1]
        else:
            selected_fmt=formats_at_res[0]
        self.config.camera_format=selected_fmt.pixel_format
        if len(selected_fmt.fps)>1:
            print(f"\nFramerates:")
            for i,fps in enumerate(selected_fmt.fps[:5],1):
                print(f"  {i}) {fps} fps")
            fps_choice=self._get_choice("Select framerate",[str(i) for i in range(1,min(len(selected_fmt.fps)+1,6))],"1")
            self.config.camera_fps=selected_fmt.fps[int(fps_choice)-1]
        else:
            self.config.camera_fps=selected_fmt.fps[0]
        print(f"\nSelected: {self.config.camera_device} @ {self.config.camera_width}x{self.config.camera_height} {self.config.camera_format} {self.config.camera_fps}fps")
        return True
    def _run_pipeline(self):
        if not Path(self.config.model_path).exists():
            print(f"Error: Model not found: {self.config.model_path}")
            return
        loader=AdvantechModelLoader(self.config,self.logger)
        try:
            engine=loader.load(self.config.model_path,self.config.model_format)
            engine.warmup(iterations=self.config.warmup_iterations)
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        source=AdvantechVideoSource(self.config,self.logger)
        try:
            if self.config.input_source==AdvantechInputSource.WEBCAM:
                if not source.open_webcam(self.config.camera_device,self.config.camera_width,self.config.camera_height,self.config.camera_format,self.config.camera_fps):
                    engine.cleanup()
                    print("Error: Failed to open camera")
                    return
            elif self.config.input_source==AdvantechInputSource.RTSP:
                if not source.open_rtsp(self.config.input_path):
                    engine.cleanup()
                    print("Error: Failed to open RTSP stream")
                    return
            elif self.config.input_source==AdvantechInputSource.FILE:
                if not source.open_file(self.config.input_path):
                    engine.cleanup()
                    print("Error: Failed to open video file")
                    return
        except Exception as e:
            engine.cleanup()
            print(f"Error opening source: {e}")
            return
        self._pipeline=AdvantechPipeline(self.config,self.logger,engine,source)
        health_server=AdvantechHealthServer(self.config,self.logger,self._pipeline)
        health_server.start()
        self._pipeline.start()
        print("\n" + "="*66)
        print("  Pipeline started. Press 'q' in display window to stop.")
        print("="*66 + "\n")
        while not self._shutdown_event.is_set() and self._pipeline.is_running():
            time.sleep(0.1)
        total_fps=self._pipeline.get_total_fps()
        total_frames=self._pipeline.get_total_frames()
        self._pipeline.stop()
        print("\n" + "="*66)
        print(f"  Inference Pipeline Stopped")
        print(f"  Total Frames: {total_frames}")
        print(f"  Average FPS:  {total_fps:.2f}")
        print("="*66 + "\n")
    def run_dryrun(self,model_path:Optional[str]=None):
        print_banner()
        print("\n=== Dry Run ===\n")
        if model_path and Path(model_path).exists():
            loader=AdvantechModelLoader(self.config,self.logger)
            try:
                engine=loader.load(model_path)
                engine.warmup(iterations=2)
                print(f"Model load: SUCCESS")
                engine.cleanup()
            except Exception as e:
                print(f"Model load: FAILED - {e}")
        source=AdvantechVideoSource(self.config,self.logger)
        cameras=source.discover_cameras()
        print(f"\nFound {len(cameras)} camera(s):")
        for cam in cameras:
            status="OK" if cam.is_working else "NOT WORKING"
            print(f"  {cam.device_path}: {cam.name} [{status}]")
            for fmt in cam.formats[:5]:
                print(f"    - {fmt}")
    def run_benchmark(self,model_path:str):
        print_banner()
        print("\n=== Benchmark ===\n")
        if not Path(model_path).exists():
            print(f"Error: Model not found: {model_path}")
            return
        loader=AdvantechModelLoader(self.config,self.logger)
        engine=loader.load(model_path)
        benchmark=AdvantechBenchmark(self.config,self.logger,engine)
        results=benchmark.run()
        print(f"Iterations: {results['iterations']}")
        print(f"Avg: {results['avg_ms']:.2f}ms  Min: {results['min_ms']:.2f}ms  Max: {results['max_ms']:.2f}ms")
        print(f"P50: {results['p50_ms']:.2f}ms  P95: {results['p95_ms']:.2f}ms  P99: {results['p99_ms']:.2f}ms")
        print(f"FPS: {results['fps']:.1f}")
        engine.cleanup()
def main():
    import argparse
    parser=argparse.ArgumentParser(description="Advantech YOLO Inference Pipeline")
    parser.add_argument("--dryrun",action="store_true",help="Verify setup")
    parser.add_argument("--benchmark",action="store_true",help="Run benchmark")
    parser.add_argument("--model",type=str,help="Model path")
    parser.add_argument("--format",choices=["pt","onnx","trt"],help="Model format")
    parser.add_argument("--source",type=str,help="Input source")
    parser.add_argument("--task",choices=["detection","classification","segmentation"],default="detection")
    parser.add_argument("--precision",choices=["fp32","fp16","int8"],default="fp16")
    parser.add_argument("--device",type=int,default=0,help="GPU device")
    parser.add_argument("--save-video",action="store_true")
    parser.add_argument("--output",type=str,default="./output")
    parser.add_argument("--no-display",action="store_true")
    parser.add_argument("--max-memory",type=int,default=6000,help="Max GPU memory MB")
    args=parser.parse_args()
    cli=AdvantechCLI()
    cli.config.precision=AdvantechPrecision(args.precision)
    cli.config.gpu_device=args.device
    cli.config.save_video=args.save_video
    cli.config.output_path=args.output
    cli.config.show_display=not args.no_display
    cli.config.max_memory_mb=args.max_memory
    cli.config.task_type={"detection":AdvantechTaskType.DETECTION,"classification":AdvantechTaskType.CLASSIFICATION,"segmentation":AdvantechTaskType.SEGMENTATION}[args.task]
    if args.format:
        cli.config.model_format=AdvantechModelFormat(args.format)
    if args.dryrun:
        cli.run_dryrun(args.model)
    elif args.benchmark and args.model:
        cli.run_benchmark(args.model)
    elif args.model and args.source:
        cli.config.model_path=args.model
        cli.config.input_path=args.source
        if args.source.startswith("rtsp://"):
            cli.config.input_source=AdvantechInputSource.RTSP
        elif args.source.startswith("/dev/video") or args.source.startswith("csi://"):
            cli.config.input_source=AdvantechInputSource.WEBCAM
            cli.config.camera_device=args.source
            cli.config.camera_width=1280
            cli.config.camera_height=720
            cli.config.camera_format="MJPEG"
            cli.config.camera_fps=30
        else:
            cli.config.input_source=AdvantechInputSource.FILE
        print_banner()
        cli._run_pipeline()
    else:
        cli.run_interactive()
if __name__=="__main__":
    main()
