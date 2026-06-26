import json
import os
import pathlib
import sys
import platform
import threading
import numpy as np
import onnxruntime as ort
import app.library._model_loader as model_loader
from app.ui.common.utils import global_backend_info_cache
from app.ui.common.config import cfg



def is_gpu_device():
    hw_type = cfg.get(cfg.hardwareOptimizationType)
    if hw_type == "CPU":
        return False
    if hw_type == "GPU":
        return True
    status, _ = global_backend_info_cache.get()
    return "GPU" in status


_ONNX_TO_NP_DTYPE = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


class _AutoCastSession:
    def __init__(self, session):
        self._session = session
        self._input_dtypes = {}
        for inp in session.get_inputs():
            dtype = _ONNX_TO_NP_DTYPE.get(inp.type)
            if dtype is not None:
                self._input_dtypes[inp.name] = dtype

    def run(self, output_names, input_feed, **kwargs):
        casted_feed = {}
        for name, value in input_feed.items():
            expected = self._input_dtypes.get(name)
            if expected is not None and isinstance(value, np.ndarray) and value.dtype != expected:
                casted_feed[name] = value.astype(expected)
            else:
                casted_feed[name] = value
        return self._session.run(output_names, casted_feed, **kwargs)

    def __getattr__(self, name):
        return getattr(self._session, name)


class ORTEnvironment:
    _initialized = False
    _lock = threading.Lock()

    @classmethod
    def initialize(cls):
        if cls._initialized:
            return
        with cls._lock:
            if cls._initialized:
                return
            
            cuda_info = ort.OrtMemoryInfo("Cuda", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0, ort.OrtMemType.DEFAULT)
            arena_cfg = ort.OrtArenaCfg(0, 1, -1, -1)
            ort.create_and_register_allocator_v2("CUDAExecutionProvider", cuda_info, {}, arena_cfg)

            info = ort.OrtMemoryInfo("Cpu", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0, ort.OrtMemType.DEFAULT)
            ort.create_and_register_allocator(info, None)
            cls._initialized = True


def general_inference_session(model_path: str, sess_options, providers, provider_options):
    model_name = os.environ["_feature_name_"]
    lic_path = os.path.join(os.path.join(pathlib.Path.home(), ".PowerTools", "license"), "license.lic")
    with open(lic_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    license_json = json.dumps(raw_data)
    old_cwd = os.getcwd()
    os.chdir(os.path.dirname(model_path))
    sess = model_loader.load_model_auto(
        model_path,
        model_name, 
        license_json, 
        session_options=sess_options, 
        providers=providers, 
        provider_options=provider_options
    )
    os.chdir(old_cwd)
    return _AutoCastSession(sess)



def general_provider():
    available = ort.get_available_providers()
    is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    if is_apple_silicon:
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
    elif "CUDAExecutionProvider" in available and is_gpu_device():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        provider_options = [{"arena_extend_strategy": "kSameAsRequested"}, {}]
    else:
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
    return providers, provider_options


def general_session():
    sess = ort.SessionOptions()
    sess.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess.add_session_config_entry("session.use_env_allocators", "1")
    return sess