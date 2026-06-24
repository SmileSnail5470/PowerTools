import os
import sys
import subprocess
import onnxruntime as ort
from PySide6.QtCore import QObject
from app.ui.common.config import cfg


def get_file_type(input_file: str):
    images_suffix = ["png", "jpg", "jpeg", "bmp", "avif", "webp"]
    videos_sufffix = ["mp4", "avi", "mov", "mkv"]
    ext = input_file.lower().split(".")[-1]
    if ext in images_suffix:
        return "image"
    if ext in videos_sufffix:
        return "video"
    else:
        return None
    

def verify_gpu_environment():
    if sys.platform != "win32":
        return "CPU 运行", ""
    if not _has_cuda_gpu():
        return "CPU 运行", ""
    try:
        ppt_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "license", "fusion.onnx")
        if not os.path.exists(ppt_onnx_path):
            ppt_onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "library", "fusion.onnx")
        session = ort.InferenceSession(ppt_onnx_path, providers=['CUDAExecutionProvider'])
        providers = session.get_providers()
        if "CUDAExecutionProvider" in providers:
            return "GPU 运行", _get_gpu_hardware_name()
        else:
            return "CPU 运行", ""
    except Exception:
        return "CPU 运行", ""
    
def _has_cuda_gpu():
    if sys.platform != "win32":
        return False
    cuda_path = r"C:\Program Files\NVIDIA Corporation"
    if os.path.exists(cuda_path):
        return True
    return False

def _get_gpu_hardware_name() -> str:
    try:
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],  encoding='utf-8', **kwargs)
        return result.strip().split('\n')[0]
    except:
        return "NVIDIA GPU"


class BackendInfoCacheManager(QObject):
    def __init__(self):
        super().__init__()
        self._cache = {}

    def get(self, key: str = "backend_info"):
        if key in self._cache:
            return self._cache[key]
        if key not in self._cache:
            self._cache[key] = verify_gpu_environment()
        return self._cache[key]

    def clear(self, key=None):
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()

global_backend_info_cache = BackendInfoCacheManager()
global_backend_info_cache.get()