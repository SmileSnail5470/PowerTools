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


_NP_TO_ORT_DTYPE = {
    np.dtype(np.float16): "tensor(float16)",
    np.dtype(np.float32): "tensor(float)",
    np.dtype(np.float64): "tensor(double)",
    np.dtype(np.int32): "tensor(int32)",
    np.dtype(np.int64): "tensor(int64)",
    np.dtype(np.int8): "tensor(int8)",
    np.dtype(np.uint8): "tensor(uint8)",
    np.dtype(np.bool_): "tensor(bool)",
}


def _is_cuda_session(session):
    try:
        providers = session.get_providers()
        return "CUDAExecutionProvider" in providers
    except Exception:
        return False


class IOBindingSession:
    def __init__(self, session):
        self._session = session
        self._use_cuda = _is_cuda_session(session)
        self._input_dtypes = {}
        self._input_names = [inp.name for inp in session.get_inputs()]
        self._output_names = [out.name for out in session.get_outputs()]
        for inp in session.get_inputs():
            dtype = _ONNX_TO_NP_DTYPE.get(inp.type)
            if dtype is not None:
                self._input_dtypes[inp.name] = dtype

    @property
    def use_cuda(self):
        return self._use_cuda

    def _cast_input(self, name, value):
        expected = self._input_dtypes.get(name)
        if expected is not None and isinstance(value, np.ndarray) and value.dtype != expected:
            return value.astype(expected)
        return value

    def run(self, output_names, input_feed, **kwargs):
        casted_feed = {}
        for name, value in input_feed.items():
            casted_feed[name] = self._cast_input(name, value)
        return self._session.run(output_names, casted_feed, **kwargs)

    def run_with_iobinding(self, input_feed, run_options=None):
        if not self._use_cuda:
            numpy_feed = {}
            for name, value in input_feed.items():
                if isinstance(value, ort.OrtValue):
                    numpy_feed[name] = value.numpy()
                else:
                    numpy_feed[name] = self._cast_input(name, value)
            results = self._session.run(None, numpy_feed, run_options=run_options)
            return [ort.OrtValue.ortvalue_from_numpy(r) for r in results]

        io_binding = self._session.io_binding()
        _input_ort_values = []
        for name, value in input_feed.items():
            if isinstance(value, ort.OrtValue):
                expected = self._input_dtypes.get(name)
                if expected is not None:
                    ort_dtype = _NP_TO_ORT_DTYPE.get(np.dtype(expected))
                    actual_dtype = value.data_type()
                    if ort_dtype and actual_dtype != ort_dtype:
                        arr = value.numpy().astype(expected)
                        arr = np.ascontiguousarray(arr)
                        value = ort.OrtValue.ortvalue_from_numpy(arr, device_type="cuda", device_id=0)
                        _input_ort_values.append(value)
                io_binding.bind_ortvalue_input(name, value)
            elif isinstance(value, np.ndarray):
                value = self._cast_input(name, value)
                value = np.ascontiguousarray(value)
                ort_value = ort.OrtValue.ortvalue_from_numpy(value, device_type="cuda", device_id=0)
                io_binding.bind_ortvalue_input(name, ort_value)
                _input_ort_values.append(ort_value)
            else:
                raise TypeError(f"Unsupported input type for IOBinding: {type(value)}")
        for out_name in self._output_names:
            io_binding.bind_output(out_name, device_type="cuda", device_id=0)
        if run_options:
            self._session.run_with_iobinding(io_binding, run_options)
        else:
            self._session.run_with_iobinding(io_binding)

        outputs = io_binding.get_outputs()
        _input_ort_values.clear()
        return outputs

    def run_with_iobinding_numpy(self, input_feed, run_options=None):
        if not self._use_cuda:
            numpy_feed = {}
            for name, value in input_feed.items():
                if isinstance(value, ort.OrtValue):
                    numpy_feed[name] = value.numpy()
                else:
                    numpy_feed[name] = self._cast_input(name, value)
            return self._session.run(None, numpy_feed, run_options=run_options)
        ort_outputs = self.run_with_iobinding(input_feed, run_options=run_options)
        return [o.numpy() for o in ort_outputs]

    def get_inputs(self):
        return self._session.get_inputs()

    def get_outputs(self):
        return self._session.get_outputs()

    def get_providers(self):
        return self._session.get_providers()

    def __getattr__(self, name):
        return getattr(self._session, name)


def ortvalue_from_numpy(arr, use_cuda=True):
    arr = np.ascontiguousarray(arr)
    if use_cuda:
        return ort.OrtValue.ortvalue_from_numpy(arr, device_type="cuda", device_id=0)
    return ort.OrtValue.ortvalue_from_numpy(arr)


def ortvalue_to_numpy(ort_value):
    if isinstance(ort_value, np.ndarray):
        return ort_value
    return ort_value.numpy()


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
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available and is_gpu_device():
                try:
                    cuda_info = ort.OrtMemoryInfo("Cuda", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0, ort.OrtMemType.DEFAULT)
                    arena_cfg = ort.OrtArenaCfg(0, 0, 256 * 1024 * 1024, -1)
                    ort.create_and_register_allocator_v2("CUDAExecutionProvider", cuda_info, {}, arena_cfg)
                except Exception:
                    pass
            try:
                info = ort.OrtMemoryInfo("Cpu", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0, ort.OrtMemType.DEFAULT)
                ort.create_and_register_allocator(info, None)
            except Exception:
                pass
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
    return IOBindingSession(sess)



def general_provider():
    available = ort.get_available_providers()
    is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    if is_apple_silicon:
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
    elif "CUDAExecutionProvider" in available and is_gpu_device():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        provider_options = [{"arena_extend_strategy": "kNextPowerOfTwo"},{}]
    else:
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
    return providers, provider_options


def general_session():
    sess = ort.SessionOptions()
    sess.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess.add_session_config_entry("session.use_env_allocators", "1")
    sess.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess.inter_op_num_threads = 0
    sess.intra_op_num_threads = 0
    return sess