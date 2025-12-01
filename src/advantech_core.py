#!/usr/bin/env python3
__title__="Advantech YOLO Core Module"
__author__="Samir Singh"
__copyright__="Copyright (c) 2024-2025 Advantech Corporation. All Rights Reserved."
__license__="Proprietary - Advantech Corporation"
__version__="2.0.1"
__build_date__="2025-01-15"
__maintainer__="Samir Singh"
__status__="Production"
import os,sys,time,threading,logging,gc,hashlib,subprocess,glob,re
from pathlib import Path
from typing import Optional,Dict,Any,List,Tuple,Union
from dataclasses import dataclass,field
from enum import Enum,auto
from logging.handlers import RotatingFileHandler
from collections import deque
from abc import ABC,abstractmethod
import numpy as np
try:
    import cv2
except ImportError:
    cv2=None
try:
    import torch
    TORCH_AVAILABLE=True
except ImportError:
    torch=None
    TORCH_AVAILABLE=False
try:
    import tensorrt as trt
    TRT_AVAILABLE=True
except ImportError:
    trt=None
    TRT_AVAILABLE=False
PYCUDA_AVAILABLE=False
PYCUDA_INITIALIZED=False
cuda=None
def init_pycuda():
    global PYCUDA_AVAILABLE,PYCUDA_INITIALIZED,cuda
    if PYCUDA_INITIALIZED:
        return PYCUDA_AVAILABLE
    try:
        import pycuda.driver as cuda_driver
        cuda_driver.init()
        device=cuda_driver.Device(0)
        ctx=device.make_context()
        ctx.pop()
        cuda=cuda_driver
        PYCUDA_AVAILABLE=True
    except:
        PYCUDA_AVAILABLE=False
    PYCUDA_INITIALIZED=True
    return PYCUDA_AVAILABLE
try:
    import onnxruntime as ort
    ONNX_AVAILABLE=True
except ImportError:
    ort=None
    ONNX_AVAILABLE=False
try:
    import gi
    gi.require_version('Gst','1.0')
    gi.require_version('GstApp','1.0')
    from gi.repository import Gst,GstApp,GLib
    GST_AVAILABLE=True
    os.environ['GST_DEBUG']='0'
    Gst.init(None)
    Gst.debug_set_active(False)
    Gst.debug_set_default_threshold(0)
except:
    Gst=None
    GstApp=None
    GLib=None
    GST_AVAILABLE=False
try:
    from flask import Flask,jsonify
    FLASK_AVAILABLE=True
except ImportError:
    Flask=None
    jsonify=None
    FLASK_AVAILABLE=False
from advantech_classes import COCO_CLASSES,IMAGENET_CLASSES,get_class_name,get_class_color,get_class_names
MAX_MEMORY_MB=int(os.environ.get("ADVANTECH_MAX_MEMORY_MB","6000"))
class AdvantechTaskType(Enum):
    DETECTION=auto()
    CLASSIFICATION=auto()
    SEGMENTATION=auto()
class AdvantechModelFormat(Enum):
    PYTORCH="pt"
    ONNX="onnx"
    TENSORRT="trt"
class AdvantechInputSource(Enum):
    WEBCAM=auto()
    RTSP=auto()
    FILE=auto()
class AdvantechPrecision(Enum):
    FP32="fp32"
    FP16="fp16"
    INT8="int8"
class AdvantechDropPolicy(Enum):
    DROP_OLDEST=auto()
    SKIP_TO_LATEST=auto()
@dataclass
class AdvantechCameraFormat:
    pixel_format:str
    width:int
    height:int
    fps:List[int]
    def __str__(self)->str:
        return f"{self.pixel_format} {self.width}x{self.height} @{'/'.join(map(str,self.fps[:3]))}fps"
@dataclass
class AdvantechCameraInfo:
    device_path:str
    name:str
    formats:List[AdvantechCameraFormat]
    is_working:bool=False
    is_csi:bool=False
    def get_resolutions(self)->List[Tuple[int,int]]:
        seen=set()
        resolutions=[]
        for fmt in self.formats:
            key=(fmt.width,fmt.height)
            if key not in seen:
                seen.add(key)
                resolutions.append(key)
        return sorted(resolutions,key=lambda x:x[0]*x[1],reverse=True)
    def get_formats_for_resolution(self,width:int,height:int)->List[AdvantechCameraFormat]:
        return [f for f in self.formats if f.width==width and f.height==height]
@dataclass
class AdvantechConfig:
    task_type:AdvantechTaskType=AdvantechTaskType.DETECTION
    model_format:Optional[AdvantechModelFormat]=None
    model_path:str=""
    input_source:AdvantechInputSource=AdvantechInputSource.WEBCAM
    input_path:str=""
    camera_device:str=""
    camera_width:int=1280
    camera_height:int=720
    camera_format:str="MJPEG"
    camera_fps:int=30
    save_video:bool=False
    output_path:str="./output"
    gpu_device:int=0
    precision:AdvantechPrecision=AdvantechPrecision.FP16
    trt_cache_path:str="./trt_cache"
    prefetch_threads:int=2
    queue_size:int=4
    batch_size:int=1
    conf_threshold:float=0.25
    iou_threshold:float=0.45
    max_det:int=300
    input_width:int=640
    input_height:int=640
    target_fps:int=30
    drop_policy:AdvantechDropPolicy=AdvantechDropPolicy.SKIP_TO_LATEST
    latency_threshold_ms:float=100.0
    show_display:bool=True
    rtsp_output:bool=False
    rtsp_output_port:int=8554
    health_port:int=8080
    log_level:str="WARNING"
    log_file:str="./advantech_yolo.log"
    benchmark_iterations:int=100
    warmup_iterations:int=10
    adaptive_skip:bool=True
    max_frame_skip:int=3
    max_memory_mb:int=MAX_MEMORY_MB
    @classmethod
    def from_env(cls)->"AdvantechConfig":
        config=cls()
        config.gpu_device=int(os.environ.get("ADVANTECH_GPU_DEVICE","0"))
        precision_str=os.environ.get("ADVANTECH_PRECISION","fp16").lower()
        config.precision=AdvantechPrecision(precision_str) if precision_str in ["fp32","fp16","int8"] else AdvantechPrecision.FP16
        config.trt_cache_path=os.environ.get("ADVANTECH_TRT_CACHE","./trt_cache")
        config.prefetch_threads=int(os.environ.get("ADVANTECH_PREFETCH_THREADS","2"))
        config.queue_size=int(os.environ.get("ADVANTECH_QUEUE_SIZE","4"))
        config.output_path=os.environ.get("ADVANTECH_OUTPUT_PATH","./output")
        config.log_level=os.environ.get("ADVANTECH_LOG_LEVEL","WARNING")
        config.max_memory_mb=int(os.environ.get("ADVANTECH_MAX_MEMORY_MB","6000"))
        return config
@dataclass
class AdvantechMetrics:
    fps:float=0.0
    avg_latency_ms:float=0.0
    p50_latency_ms:float=0.0
    p90_latency_ms:float=0.0
    p95_latency_ms:float=0.0
    p99_latency_ms:float=0.0
    queue_sizes:Dict[str,int]=field(default_factory=dict)
    model_status:str="not_loaded"
    gpu_memory_used_mb:float=0.0
    frames_processed:int=0
    frames_dropped:int=0
@dataclass
class AdvantechDetection:
    bbox:Tuple[float,float,float,float]
    confidence:float
    class_id:int
    class_name:str=""
    mask:Optional[np.ndarray]=None
@dataclass
class AdvantechClassification:
    top_classes:List[Tuple[int,str,float]]
    @property
    def top1(self)->Tuple[int,str,float]:
        return self.top_classes[0] if self.top_classes else (0,"unknown",0.0)
    @property
    def top5(self)->List[Tuple[int,str,float]]:
        return self.top_classes[:5]
@dataclass
class AdvantechFrame:
    data:np.ndarray
    timestamp:float
    frame_id:int
    width:int
    height:int
    detections:List[AdvantechDetection]=field(default_factory=list)
    classification:Optional[AdvantechClassification]=None
    inference_time_ms:float=0.0
    total_latency_ms:float=0.0
class AdvantechMemoryManager:
    _instance=None
    _lock=threading.Lock()
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance=super().__new__(cls)
                    cls._instance._initialized=False
        return cls._instance
    def __init__(self):
        if self._initialized:
            return
        self._initialized=True
        self._max_memory_mb=MAX_MEMORY_MB
        self._allocated_buffers:List[np.ndarray]=[]
        self._lock=threading.Lock()
    def set_max_memory(self,max_mb:int):
        self._max_memory_mb=max_mb
    def get_memory_usage_mb(self)->float:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                return torch.cuda.memory_allocated()/(1024*1024)
            except:
                pass
        return 0.0
    def check_memory_available(self,required_mb:float)->bool:
        return (self.get_memory_usage_mb()+required_mb)<self._max_memory_mb
    def force_cleanup(self):
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                pass
    def allocate_pinned_buffer(self,shape:tuple,dtype=np.float32)->Optional[np.ndarray]:
        size_mb=np.prod(shape)*np.dtype(dtype).itemsize/(1024*1024)
        if not self.check_memory_available(size_mb):
            self.force_cleanup()
            if not self.check_memory_available(size_mb):
                return None
        with self._lock:
            if init_pycuda() and cuda is not None:
                try:
                    buffer=cuda.pagelocked_empty(shape,dtype)
                    self._allocated_buffers.append(buffer)
                    return buffer
                except:
                    pass
            buffer=np.empty(shape,dtype=dtype)
            self._allocated_buffers.append(buffer)
            return buffer
class AdvantechLogger:
    _instance:Optional["AdvantechLogger"]=None
    _lock=threading.Lock()
    def __new__(cls,*args,**kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance=super().__new__(cls)
        return cls._instance
    def __init__(self,config:Optional[AdvantechConfig]=None):
        if hasattr(self,'_initialized') and self._initialized:
            return
        self._initialized=True
        self.config=config or AdvantechConfig()
        self.logger=logging.getLogger("AdvantechYOLO")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        try:
            Path(self.config.log_file).parent.mkdir(parents=True,exist_ok=True)
            file_handler=RotatingFileHandler(self.config.log_file,maxBytes=5*1024*1024,backupCount=3)
            file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',datefmt='%H:%M:%S'))
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)
        except:
            pass
    def info(self,msg:str,component:str="Main"):
        self.logger.info(f"[{component}] {msg}")
    def warning(self,msg:str,component:str="Main"):
        self.logger.warning(f"[{component}] {msg}")
    def error(self,msg:str,component:str="Main"):
        self.logger.error(f"[{component}] {msg}")
    def debug(self,msg:str,component:str="Main"):
        self.logger.debug(f"[{component}] {msg}")
class AdvantechMetricsCollector:
    def __init__(self,window_size:int=120):
        self._lock=threading.Lock()
        self._latencies:deque=deque(maxlen=window_size)
        self._frame_times:deque=deque(maxlen=window_size)
        self._last_frame_time=time.perf_counter()
        self._frames_processed=0
        self._frames_dropped=0
        self._queue_sizes:Dict[str,int]={}
        self._model_status="not_loaded"
        self._start_time=time.perf_counter()
    def record_latency(self,latency_ms:float):
        with self._lock:
            self._latencies.append(latency_ms)
    def record_frame(self):
        current_time=time.perf_counter()
        with self._lock:
            self._frame_times.append(current_time-self._last_frame_time)
            self._last_frame_time=current_time
            self._frames_processed+=1
    def record_drop(self):
        with self._lock:
            self._frames_dropped+=1
    def update_queue_size(self,name:str,size:int):
        with self._lock:
            self._queue_sizes[name]=size
    def set_model_status(self,status:str):
        with self._lock:
            self._model_status=status
    def get_metrics(self)->AdvantechMetrics:
        with self._lock:
            metrics=AdvantechMetrics()
            metrics.frames_processed=self._frames_processed
            metrics.frames_dropped=self._frames_dropped
            metrics.queue_sizes=dict(self._queue_sizes)
            metrics.model_status=self._model_status
            if self._frame_times:
                avg_frame_time=sum(self._frame_times)/len(self._frame_times)
                metrics.fps=1.0/avg_frame_time if avg_frame_time>0 else 0.0
            if self._latencies:
                sorted_lat=sorted(self._latencies)
                n=len(sorted_lat)
                metrics.avg_latency_ms=sum(sorted_lat)/n
                metrics.p50_latency_ms=sorted_lat[int(n*0.5)]
                metrics.p90_latency_ms=sorted_lat[min(int(n*0.9),n-1)]
                metrics.p95_latency_ms=sorted_lat[min(int(n*0.95),n-1)]
                metrics.p99_latency_ms=sorted_lat[min(int(n*0.99),n-1)]
            metrics.gpu_memory_used_mb=AdvantechMemoryManager().get_memory_usage_mb()
            return metrics
    def get_total_fps(self)->float:
        with self._lock:
            elapsed=time.perf_counter()-self._start_time
            return self._frames_processed/elapsed if elapsed>0 else 0.0
    def get_total_frames(self)->int:
        with self._lock:
            return self._frames_processed
class AdvantechRingBuffer:
    def __init__(self,capacity:int,drop_policy:AdvantechDropPolicy=AdvantechDropPolicy.SKIP_TO_LATEST):
        self._capacity=capacity
        self._drop_policy=drop_policy
        self._buffer:deque=deque(maxlen=capacity)
        self._lock=threading.Lock()
        self._not_empty=threading.Condition(self._lock)
        self._closed=False
        self._dropped_count=0
    def put(self,item:Any,timeout:Optional[float]=None)->bool:
        with self._not_empty:
            if self._closed:
                return False
            if len(self._buffer)>=self._capacity:
                if self._drop_policy==AdvantechDropPolicy.DROP_OLDEST:
                    self._buffer.popleft()
                else:
                    self._buffer.clear()
                self._dropped_count+=1
            self._buffer.append(item)
            self._not_empty.notify()
            return True
    def get(self,timeout:Optional[float]=None)->Optional[Any]:
        with self._not_empty:
            while len(self._buffer)==0 and not self._closed:
                if not self._not_empty.wait(timeout=timeout):
                    return None
            if self._closed and len(self._buffer)==0:
                return None
            if self._drop_policy==AdvantechDropPolicy.SKIP_TO_LATEST and len(self._buffer)>1:
                while len(self._buffer)>1:
                    self._buffer.popleft()
                    self._dropped_count+=1
            return self._buffer.popleft() if self._buffer else None
    def size(self)->int:
        with self._lock:
            return len(self._buffer)
    def close(self):
        with self._lock:
            self._closed=True
            self._not_empty.notify_all()
class AdvantechEngine(ABC):
    def __init__(self,config:AdvantechConfig,logger:AdvantechLogger):
        self.config=config
        self.logger=logger
        self.task_type=config.task_type
        self._warmed_up=False
        self._memory_manager=AdvantechMemoryManager()
        self._model_input_height=config.input_height
        self._model_input_width=config.input_width
    def get_class_names(self)->List[str]:
        return IMAGENET_CLASSES if self.task_type==AdvantechTaskType.CLASSIFICATION else COCO_CLASSES
    @abstractmethod
    def infer(self,frame:np.ndarray)->Union[List[AdvantechDetection],AdvantechClassification]:
        pass
    @abstractmethod
    def warmup(self,iterations:int=5):
        pass
    @abstractmethod
    def cleanup(self):
        pass
    def preprocess(self,frame:np.ndarray)->np.ndarray:
        h,w=frame.shape[:2]
        target_h,target_w=self._model_input_height,self._model_input_width
        scale=min(target_w/w,target_h/h)
        new_w,new_h=int(round(w*scale)),int(round(h*scale))
        resized=cv2.resize(frame,(new_w,new_h),interpolation=cv2.INTER_LINEAR)
        padded=np.full((target_h,target_w,3),114,dtype=np.uint8)
        pad_x,pad_y=(target_w-new_w)//2,(target_h-new_h)//2
        padded[pad_y:pad_y+new_h,pad_x:pad_x+new_w]=resized
        blob=padded.transpose(2,0,1).astype(np.float32)/255.0
        return np.expand_dims(blob,axis=0)
    def postprocess(self,outputs:np.ndarray,original_shape:Tuple[int,int],num_masks:int=0)->List[AdvantechDetection]:
        outputs=outputs.astype(np.float32)
        if outputs.ndim==3:
            outputs=outputs[0]
        if outputs.shape[0]<outputs.shape[1]:
            outputs=outputs.T
        boxes=outputs[:,:4]
        if num_masks>0:
            scores=outputs[:,4:-num_masks]
        else:
            scores=outputs[:,4:]
        if scores.ndim==1 or scores.shape[1]==0:
            confidences=scores.flatten() if scores.ndim==1 else np.zeros(len(boxes))
            class_ids=np.zeros(len(boxes),dtype=np.int32)
        else:
            confidences=np.max(scores,axis=1)
            class_ids=np.argmax(scores,axis=1)
        mask=confidences>=self.config.conf_threshold
        boxes,confidences,class_ids=boxes[mask],confidences[mask],class_ids[mask]
        if len(boxes)==0:
            return []
        h_orig,w_orig=original_shape
        target_h,target_w=self._model_input_height,self._model_input_width
        scale=min(target_w/w_orig,target_h/h_orig)
        new_w,new_h=int(round(w_orig*scale)),int(round(h_orig*scale))
        pad_x,pad_y=(target_w-new_w)/2,(target_h-new_h)/2
        boxes_xyxy=np.zeros_like(boxes)
        boxes_xyxy[:,0]=(boxes[:,0]-boxes[:,2]/2-pad_x)/scale
        boxes_xyxy[:,1]=(boxes[:,1]-boxes[:,3]/2-pad_y)/scale
        boxes_xyxy[:,2]=(boxes[:,0]+boxes[:,2]/2-pad_x)/scale
        boxes_xyxy[:,3]=(boxes[:,1]+boxes[:,3]/2-pad_y)/scale
        boxes_xyxy[:,0]=np.clip(boxes_xyxy[:,0],0,w_orig)
        boxes_xyxy[:,1]=np.clip(boxes_xyxy[:,1],0,h_orig)
        boxes_xyxy[:,2]=np.clip(boxes_xyxy[:,2],0,w_orig)
        boxes_xyxy[:,3]=np.clip(boxes_xyxy[:,3],0,h_orig)
        indices=self._nms(boxes_xyxy,confidences,self.config.iou_threshold)[:self.config.max_det]
        class_names=self.get_class_names()
        detections=[]
        for i in indices:
            class_id=int(class_ids[i])
            detections.append(AdvantechDetection(bbox=tuple(boxes_xyxy[i].tolist()),confidence=float(confidences[i]),class_id=class_id,class_name=class_names[class_id] if class_id<len(class_names) else f"class_{class_id}"))
        return detections
    def _nms(self,boxes:np.ndarray,scores:np.ndarray,iou_threshold:float)->List[int]:
        boxes,scores=boxes.astype(np.float32),scores.astype(np.float32)
        x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas=(x2-x1)*(y2-y1)
        order=scores.argsort()[::-1]
        keep=[]
        while len(order)>0:
            i=order[0]
            keep.append(i)
            if len(order)==1:
                break
            xx1,yy1=np.maximum(x1[i],x1[order[1:]]),np.maximum(y1[i],y1[order[1:]])
            xx2,yy2=np.minimum(x2[i],x2[order[1:]]),np.minimum(y2[i],y2[order[1:]])
            w,h=np.maximum(0,xx2-xx1),np.maximum(0,yy2-yy1)
            iou=(w*h)/(areas[i]+areas[order[1:]]-w*h+1e-6)
            order=order[np.where(iou<=iou_threshold)[0]+1]
        return keep
    def postprocess_classification(self,outputs:np.ndarray,top_k:int=5)->AdvantechClassification:
        outputs=outputs.astype(np.float32)
        probs=outputs[0] if outputs.ndim==2 else outputs
        if probs.max()>1.0 or probs.min()<0.0:
            exp_probs=np.exp(probs-np.max(probs))
            probs=exp_probs/exp_probs.sum()
        top_indices=np.argsort(probs)[::-1][:top_k]
        class_names=self.get_class_names()
        return AdvantechClassification(top_classes=[(int(idx),class_names[idx] if idx<len(class_names) else f"class_{idx}",float(probs[idx])) for idx in top_indices])
    def postprocess_segmentation(self,det_output:np.ndarray,mask_protos:np.ndarray,original_shape:Tuple[int,int])->List[AdvantechDetection]:
        det_output=det_output.astype(np.float32)
        mask_protos=mask_protos.astype(np.float32)
        if det_output.ndim==3:
            det_output=det_output[0]
        if det_output.shape[0]<det_output.shape[1]:
            det_output=det_output.T
        num_preds=det_output.shape[0]
        num_cols=det_output.shape[1]
        num_mask_coeffs=32
        num_classes=num_cols-4-num_mask_coeffs
        if num_classes<=0:
            return self.postprocess(det_output,original_shape,num_masks=0)
        boxes=det_output[:,:4]
        class_scores=det_output[:,4:4+num_classes]
        mask_coeffs=det_output[:,4+num_classes:4+num_classes+num_mask_coeffs]
        confidences=np.max(class_scores,axis=1)
        class_ids=np.argmax(class_scores,axis=1)
        valid_mask=confidences>=self.config.conf_threshold
        valid_indices=np.where(valid_mask)[0]
        if len(valid_indices)==0:
            return []
        boxes=boxes[valid_mask]
        confidences=confidences[valid_mask]
        class_ids=class_ids[valid_mask]
        mask_coeffs=mask_coeffs[valid_mask]
        h_orig,w_orig=original_shape
        target_h,target_w=self._model_input_height,self._model_input_width
        scale=min(target_w/w_orig,target_h/h_orig)
        new_w,new_h=int(round(w_orig*scale)),int(round(h_orig*scale))
        pad_x,pad_y=(target_w-new_w)/2,(target_h-new_h)/2
        boxes_xyxy=np.zeros_like(boxes)
        boxes_xyxy[:,0]=(boxes[:,0]-boxes[:,2]/2-pad_x)/scale
        boxes_xyxy[:,1]=(boxes[:,1]-boxes[:,3]/2-pad_y)/scale
        boxes_xyxy[:,2]=(boxes[:,0]+boxes[:,2]/2-pad_x)/scale
        boxes_xyxy[:,3]=(boxes[:,1]+boxes[:,3]/2-pad_y)/scale
        boxes_xyxy[:,0]=np.clip(boxes_xyxy[:,0],0,w_orig)
        boxes_xyxy[:,1]=np.clip(boxes_xyxy[:,1],0,h_orig)
        boxes_xyxy[:,2]=np.clip(boxes_xyxy[:,2],0,w_orig)
        boxes_xyxy[:,3]=np.clip(boxes_xyxy[:,3],0,h_orig)
        indices=self._nms(boxes_xyxy,confidences,self.config.iou_threshold)[:self.config.max_det]
        if mask_protos.ndim==4:
            protos=mask_protos[0]
        else:
            protos=mask_protos
        proto_h,proto_w=protos.shape[1],protos.shape[2]
        class_names=self.get_class_names()
        detections=[]
        for idx in indices:
            class_id=int(class_ids[idx])
            x1,y1,x2,y2=boxes_xyxy[idx]
            det=AdvantechDetection(bbox=(float(x1),float(y1),float(x2),float(y2)),confidence=float(confidences[idx]),class_id=class_id,class_name=class_names[class_id] if class_id<len(class_names) else f"class_{class_id}")
            try:
                coeffs=mask_coeffs[idx]
                mask_pred=np.einsum('i,ihw->hw',coeffs,protos)
                mask_pred=1.0/(1.0+np.exp(-mask_pred))
                scale_x=target_w/proto_w
                scale_y=target_h/proto_h
                x1_proto=int(max(0,(boxes[idx,0]-boxes[idx,2]/2)/scale_x))
                y1_proto=int(max(0,(boxes[idx,1]-boxes[idx,3]/2)/scale_y))
                x2_proto=int(min(proto_w,(boxes[idx,0]+boxes[idx,2]/2)/scale_x))
                y2_proto=int(min(proto_h,(boxes[idx,1]+boxes[idx,3]/2)/scale_y))
                mask_resized=cv2.resize(mask_pred,(w_orig,h_orig),interpolation=cv2.INTER_LINEAR)
                ix1,iy1,ix2,iy2=int(max(0,x1)),int(max(0,y1)),int(min(w_orig,x2)),int(min(h_orig,y2))
                full_mask=np.zeros((h_orig,w_orig),dtype=np.float32)
                if ix2>ix1 and iy2>iy1:
                    crop=mask_resized[iy1:iy2,ix1:ix2]
                    full_mask[iy1:iy2,ix1:ix2]=(crop>0.5).astype(np.float32)
                det.mask=full_mask
            except:
                pass
            detections.append(det)
        return detections
__all__=['TORCH_AVAILABLE','TRT_AVAILABLE','PYCUDA_AVAILABLE','ONNX_AVAILABLE','GST_AVAILABLE','FLASK_AVAILABLE','cv2','torch','trt','cuda','ort','Gst','GstApp','GLib','Flask','jsonify','init_pycuda','COCO_CLASSES','IMAGENET_CLASSES','get_class_name','get_class_color','get_class_names','MAX_MEMORY_MB','AdvantechTaskType','AdvantechModelFormat','AdvantechInputSource','AdvantechPrecision','AdvantechDropPolicy','AdvantechCameraFormat','AdvantechCameraInfo','AdvantechConfig','AdvantechMetrics','AdvantechDetection','AdvantechClassification','AdvantechFrame','AdvantechMemoryManager','AdvantechLogger','AdvantechMetricsCollector','AdvantechRingBuffer','AdvantechEngine']
