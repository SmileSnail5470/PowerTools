import json
import os
import pathlib
import sys
import platform
import threading
import numpy as np
import onnxruntime as ort
ort.preload_dlls(directory="")
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
        self._persist_inputs = {}

    @property
    def use_cuda(self):
        return self._use_cuda

    def expected_dtype(self, name):
        return self._input_dtypes.get(name)

    def to_device(self, name, value):
        if isinstance(value, ort.OrtValue):
            return value
        value = self._cast_input(name, value)
        value = np.ascontiguousarray(value)
        if self._use_cuda:
            return ort.OrtValue.ortvalue_from_numpy(value, device_type="cuda", device_id=0)
        return ort.OrtValue.ortvalue_from_numpy(value)

    def _bind_numpy_input(self, io_binding, name, value):
        value = self._cast_input(name, value)
        value = np.ascontiguousarray(value)
        buf = self._persist_inputs.get(name)
        ort_dtype = _NP_TO_ORT_DTYPE.get(np.dtype(value.dtype))
        if (
            buf is not None
            and tuple(buf.shape()) == value.shape
            and buf.data_type() == ort_dtype
        ):
            buf.update_inplace(value)
        else:
            buf = ort.OrtValue.ortvalue_from_numpy(value, device_type="cuda", device_id=0)
            self._persist_inputs[name] = buf
        io_binding.bind_ortvalue_input(name, buf)

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
                io_binding.bind_ortvalue_input(name, value)
            elif isinstance(value, np.ndarray):
                self._bind_numpy_input(io_binding, name, value)
            else:
                raise TypeError(f"Unsupported input type for IOBinding: {type(value)}")
        for out_name in self._output_names:
            io_binding.bind_output(out_name, device_type="cuda", device_id=0)
        if run_options:
            self._session.run_with_iobinding(io_binding, run_options)
        else:
            self._session.run_with_iobinding(io_binding)

        outputs = io_binding.get_outputs()
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


class CudaGraphRunner:
    """针对“固定输入/输出形状、反复调用”的单个 session,
    使用 CUDA Graph 捕获-重放, 消除大量逐 kernel 的 launch 开销。

    使用前提 (启用前务必在目标 GPU 上验证输出正确性):
      - 关联的 session 必须以 provider option enable_cuda_graph='1' 创建
        (即 general_provider(enable_cuda_graph=True));
      - 每次调用各输入的形状/dtype 必须完全一致;
      - 模型需能完整运行在 CUDA 上 (无 CPU 回退算子)。

    output_specs: {output_name: (shape_tuple, np_dtype)}。CUDA Graph 必须在首次
        (捕获) 运行前就把输出绑定到固定显存, 因此需要预先知道输出形状。

    若构建/捕获/重放过程中抛出异常, 会永久回退到普通 run_with_iobinding_numpy,
    保证不会因为 CUDA Graph 不可用而中断业务。
    注意: 回退只能拦截“异常”, 无法拦截 CUDA Graph 误用导致的“静默错误结果”——
    这正是必须先在 GPU 上验证的原因。
    """

    def __init__(self, session: "IOBindingSession", output_specs, device_id: int = 0):
        self._s = session
        self._device_id = device_id
        self._output_specs = output_specs  # {name: (shape, np_dtype)}
        self._io = None
        self._in_bufs = {}
        self._out_bufs = {}
        self._fallback = False

    def _build(self, input_feed):
        io = self._s._session.io_binding()
        for name, value in input_feed.items():
            value = self._s._cast_input(name, value)
            value = np.ascontiguousarray(value)
            buf = ort.OrtValue.ortvalue_from_numpy(value, "cuda", self._device_id)
            self._in_bufs[name] = buf
            io.bind_ortvalue_input(name, buf)
        for name, (shape, dt) in self._output_specs.items():
            buf = ort.OrtValue.ortvalue_from_shape_and_type(list(shape), dt, "cuda", self._device_id)
            self._out_bufs[name] = buf
            io.bind_ortvalue_output(name, buf)
        self._io = io

    def run(self, input_feed):
        if self._fallback or not self._s.use_cuda:
            outs = self._s.run_with_iobinding_numpy(input_feed)
            return {n: outs[i] for i, n in enumerate(self._output_specs.keys())}
        try:
            if self._io is None:
                self._build(input_feed)
                self._s._session.run_with_iobinding(self._io)
            else:
                for name, value in input_feed.items():
                    value = self._s._cast_input(name, value)
                    value = np.ascontiguousarray(value)
                    self._in_bufs[name].update_inplace(value)
                self._s._session.run_with_iobinding(self._io)
            return {n: b.numpy() for n, b in self._out_bufs.items()}
        except Exception:
            print("CudaGraphRunner failed and start fallback")
            self._fallback = True
            self._io = None
            self._in_bufs.clear()
            self._out_bufs.clear()
            outs = self._s.run_with_iobinding_numpy(input_feed)
            return {n: outs[i] for i, n in enumerate(self._output_specs.keys())}


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



def general_provider(enable_cuda_graph: bool = False, use_cpu: bool = False):
    if use_cpu:
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
        return providers, provider_options
    available = ort.get_available_providers()
    is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    if is_apple_silicon:
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
    elif "CUDAExecutionProvider" in available and is_gpu_device():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        cuda_opts = {
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": "1",
        }
        if enable_cuda_graph:
            cuda_opts["enable_cuda_graph"] = "1"
        provider_options = [cuda_opts, {}]
    else:
        providers = ["CPUExecutionProvider"]
        provider_options = [{}]
    return providers, provider_options


def general_session():
    sess = ort.SessionOptions()
    sess.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess.add_session_config_entry("session.use_env_allocators", "1")
    sess.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess.inter_op_num_threads = 0
    sess.intra_op_num_threads = 0
    return sess