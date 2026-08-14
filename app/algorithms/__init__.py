import json
import logging
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
try:
    import cupy
except Exception:
    pass


algorithms_logger = logging.getLogger("algorithms")


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


_SESSION_CACHE = {}
_SESSION_CACHE_LOCK = threading.Lock()


def _is_cuda_session(session):
    try:
        providers = session.get_providers()
        return "CUDAExecutionProvider" in providers
    except Exception:
        return False


def _is_cupy_array(value):
    return type(value).__module__.partition(".")[0] == "cupy"


class _BindingState(threading.local):
    def __init__(self, registry, lock):
        self.binding = None
        self.input_bufs = {}       # name -> [OrtValue, shape, np.dtype]
        self.bound_inputs = {}     # name -> 当前已绑定的对象(避免重复 bind)
        self.feed_keys = None      # 上一次喂入的输入名签名
        self.output_bufs = {}      # (name, shape, is_cupy) -> buffer
        self.output_mode = None    # None | "auto" | 固定输出形状 key
        self.output_current = None  # 固定输出模式下 {name: buffer}
        with lock:
            registry.append(self)


class IOBindingSession:
    def __init__(self, session):
        self._session = session
        self._use_cuda = _is_cuda_session(session)
        self._device_type = "cuda" if self._use_cuda else "cpu"
        self._device_id = 0
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        self._input_names = [inp.name for inp in inputs]
        self._output_names = [out.name for out in outputs]
        self._input_dtypes = {}
        self._input_ort_dtypes = {}
        self._input_shapes = {}
        for inp in inputs:
            dtype = _ONNX_TO_NP_DTYPE.get(inp.type)
            if dtype is not None:
                self._input_dtypes[inp.name] = np.dtype(dtype)
                self._input_ort_dtypes[inp.name] = inp.type
            self._input_shapes[inp.name] = inp.shape
        self._output_dtypes = {}
        self._output_shapes = {}
        for out in outputs:
            dtype = _ONNX_TO_NP_DTYPE.get(out.type)
            if dtype is not None:
                self._output_dtypes[out.name] = np.dtype(dtype)
            self._output_shapes[out.name] = out.shape
        self._static_inputs = {}
        self._states = []
        self._states_lock = threading.Lock()
        self._state = _BindingState(self._states, self._states_lock)

    @property
    def use_cuda(self):
        return self._use_cuda

    @property
    def device_type(self):
        return self._device_type

    @property
    def device_id(self):
        return self._device_id

    @property
    def input_names(self):
        return self._input_names

    @property
    def output_names(self):
        return self._output_names

    @property
    def input_dtypes(self):
        return self._input_dtypes

    @property
    def output_dtypes(self):
        return self._output_dtypes

    @property
    def input_shapes(self):
        return self._input_shapes

    @property
    def output_shapes(self):
        return self._output_shapes

    @property
    def provider(self):
        providers = self._session.get_providers()
        return providers[0] if providers else "CPUExecutionProvider"

    def expected_dtype(self, name):
        return self._input_dtypes.get(name)

    def output_dtype(self, name):
        return self._output_dtypes.get(name)

    def get_inputs(self):
        return self._session.get_inputs()

    def get_outputs(self):
        return self._session.get_outputs()

    def get_providers(self):
        return self._session.get_providers()

    def __getattr__(self, name):
        try:
            session = self.__dict__["_session"]
        except KeyError:
            raise AttributeError(name)
        return getattr(session, name)

    def _prepare_numpy(self, name, value):
        expected = self._input_dtypes.get(name)
        if expected is not None and value.dtype != expected:
            value = value.astype(expected, copy=False)
        if not value.flags.c_contiguous:
            value = np.ascontiguousarray(value)
        return value

    def _cast_input(self, name, value):
        if isinstance(value, np.ndarray):
            return self._prepare_numpy(name, value)
        return value

    def _to_numpy_input(self, name, value):
        if isinstance(value, np.ndarray):
            return self._prepare_numpy(name, value)
        if isinstance(value, ort.OrtValue):
            return self._prepare_numpy(name, value.numpy())
        if _is_cupy_array(value):
            return self._prepare_numpy(name, value.get())
        return value

    def to_device(self, name, value):
        if isinstance(value, ort.OrtValue):
            return value
        if _is_cupy_array(value):
            if not self._use_cuda:
                value = value.get()
            else:
                return self._ortvalue_from_cupy(name, value)
        value = self._prepare_numpy(name, value)
        if self._use_cuda:
            return ort.OrtValue.ortvalue_from_numpy(value, device_type=self._device_type, device_id=self._device_id)
        return ort.OrtValue.ortvalue_from_numpy(value)

    def _ortvalue_from_cupy(self, name, value):
        value = cupy.ascontiguousarray(value)
        try:
            return ort.OrtValue.from_dlpack(value.toDlpack())
        except Exception as e:
            algorithms_logger.warning(f"OrtValue from_dlpack failed {e} and fallback ortvalue_from_numpy")
            return ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(value.get()), device_type="cuda", device_id=self._device_id)

    def set_static_input(self, name, value):
        if self._use_cuda:
            self._static_inputs[name] = self.to_device(name, value)
        else:
            self._static_inputs[name] = self._to_numpy_input(name, value)

    def clear_static_inputs(self):
        self._static_inputs.clear()

    def clear_persistent_inputs(self):
        with self._states_lock:
            states = list(self._states)
        for state in states:
            state.input_bufs.clear()
            state.bound_inputs.clear()
            state.feed_keys = None
            state.output_bufs.clear()
            state.output_current = None
            state.output_mode = None
            state.binding = None

    def _binding(self, state):
        binding = state.binding
        if binding is None:
            binding = state.binding = self._session.io_binding()
        return binding

    def _bind_input(self, binding, state, name, value, bound):
        if isinstance(value, np.ndarray):
            value = self._prepare_numpy(name, value)
            if self._device_type != "cuda":
                binding.bind_cpu_input(name, value)
                if bound is not None:
                    bound[name] = value
                return False
            entry = state.input_bufs.get(name)
            if entry is not None and entry[1] == value.shape and entry[2] == value.dtype:
                buf = entry[0]
                buf.update_inplace(value)
            else:
                buf = ort.OrtValue.ortvalue_from_numpy(value, device_type=self._device_type, device_id=self._device_id)
                state.input_bufs[name] = [buf, value.shape, value.dtype]
            if bound is None:
                binding.bind_ortvalue_input(name, buf)
            elif bound.get(name) is not buf:
                binding.bind_ortvalue_input(name, buf)
                bound[name] = buf
            return False
        if isinstance(value, ort.OrtValue):
            expected = self._input_ort_dtypes.get(name)
            if expected is not None and value.data_type() != expected:
                value = ort.OrtValue.ortvalue_from_numpy(
                    self._prepare_numpy(name, value.numpy()),
                    device_type=self._device_type,
                    device_id=self._device_id,
                )
            if bound is None:
                binding.bind_ortvalue_input(name, value)
            elif bound.get(name) is not value:
                binding.bind_ortvalue_input(name, value)
                bound[name] = value
            return False
        if _is_cupy_array(value):
            expected = self._input_dtypes.get(name)
            if expected is not None and value.dtype != expected:
                value = value.astype(expected, copy=False)
            if not value.flags.c_contiguous:
                value = cupy.ascontiguousarray(value)
            binding.bind_input(
                name,
                "cuda",
                self._device_id,
                value.dtype,
                value.shape,
                value.data.ptr,
            )
            if bound is not None:
                bound[name] = value
            return True
        raise TypeError(f"Unsupported input type for IOBinding: {type(value)}")

    def _bind_inputs(self, binding, state, input_feed, bound):
        if bound is not None:
            keys = frozenset(input_feed)
            if state.feed_keys != keys:
                if state.feed_keys is not None:
                    binding.clear_binding_inputs()
                    bound.clear()
                state.feed_keys = keys
        needs_sync = False
        for name, value in input_feed.items():
            if self._bind_input(binding, state, name, value, bound):
                needs_sync = True
        if self._static_inputs:
            for name, value in self._static_inputs.items():
                if name in input_feed:
                    continue
                if self._bind_input(binding, state, name, value, bound):
                    needs_sync = True
        return needs_sync

    def _bind_auto_outputs(self, binding, state):
        if state.output_mode is not None:
            binding.clear_binding_outputs()
        device_type = self._device_type
        device_id = self._device_id
        for name in self._output_names:
            binding.bind_output(name, device_type, device_id)
        state.output_mode = "auto"
        state.output_current = None

    def _alloc_output(self, name, shape, use_cupy):
        dtype = self._output_dtypes.get(name, np.float32)
        if use_cupy:
            return cupy.empty(shape, dtype=dtype)
        return ort.OrtValue.ortvalue_from_shape_and_type(list(shape), dtype, self._device_type, self._device_id)

    def _bind_fixed_outputs(self, binding, state, output_shapes, use_cupy):
        key = (use_cupy, tuple(tuple(output_shapes[name]) for name in self._output_names))
        if state.output_mode == key:
            return state.output_current
        if state.output_mode is not None:
            binding.clear_binding_outputs()
        buffers = {}
        for name in self._output_names:
            shape = tuple(output_shapes[name])
            cache_key = (name, shape, use_cupy)
            buf = state.output_bufs.get(cache_key)
            if buf is None:
                buf = self._alloc_output(name, shape, use_cupy)
                state.output_bufs[cache_key] = buf
            buffers[name] = buf
            if use_cupy:
                binding.bind_output(name, "cuda", self._device_id, buf.dtype, buf.shape, buf.data.ptr)
            else:
                binding.bind_ortvalue_output(name, buf)
        state.output_mode = key
        state.output_current = buffers
        return buffers

    def run(self, output_names, input_feed, **kwargs):
        casted_feed = {}
        for name, value in input_feed.items():
            casted_feed[name] = self._to_numpy_input(name, value)
        if self._static_inputs:
            for name, value in self._static_inputs.items():
                if name not in casted_feed:
                    casted_feed[name] = self._to_numpy_input(name, value)
        return self._session.run(output_names, casted_feed, **kwargs)

    def _run_numpy_fallback(self, input_feed, run_options=None):
        numpy_feed = {}
        for name, value in input_feed.items():
            numpy_feed[name] = self._to_numpy_input(name, value)
        if self._static_inputs:
            for name, value in self._static_inputs.items():
                if name not in numpy_feed:
                    numpy_feed[name] = self._to_numpy_input(name, value)
        return self._session.run(None, numpy_feed, run_options=run_options)

    def _run_bound(self, input_feed, run_options, output_shapes=None, use_cupy=False):
        state = self._state
        binding = self._binding(state)
        needs_sync = self._bind_inputs(binding, state, input_feed, state.bound_inputs)
        if output_shapes:
            buffers = self._bind_fixed_outputs(binding, state, output_shapes, use_cupy)
        else:
            buffers = None
            self._bind_auto_outputs(binding, state)
        if needs_sync or use_cupy:
            binding.synchronize_inputs()
        self._session.run_with_iobinding(binding, run_options)
        if needs_sync or use_cupy:
            binding.synchronize_outputs()
        return binding, buffers

    def _run_fresh_outputs(self, input_feed, run_options):
        state = self._state
        binding = self._session.io_binding()
        needs_sync = self._bind_inputs(binding, state, input_feed, None)
        device_type = self._device_type
        device_id = self._device_id
        for name in self._output_names:
            binding.bind_output(name, device_type, device_id)
        if needs_sync:
            binding.synchronize_inputs()
        self._session.run_with_iobinding(binding, run_options)
        if needs_sync:
            binding.synchronize_outputs()
        return binding

    def run_with_iobinding(self, input_feed, run_options=None):
        if not self._use_cuda:
            results = self._run_numpy_fallback(input_feed, run_options=run_options)
            return [ort.OrtValue.ortvalue_from_numpy(r) for r in results]
        binding = self._run_fresh_outputs(input_feed, run_options)
        return binding.get_outputs()

    def run_with_iobinding_numpy(self, input_feed, run_options=None):
        if not self._use_cuda:
            return self._run_numpy_fallback(input_feed, run_options=run_options)
        binding, _ = self._run_bound(input_feed, run_options)
        return binding.copy_outputs_to_cpu()

    def run_ortvalues(self, input_feed, run_options=None):
        if not self._use_cuda:
            results = self._run_numpy_fallback(input_feed, run_options=run_options)
            return [ort.OrtValue.ortvalue_from_numpy(r) for r in results]
        binding, _ = self._run_bound(input_feed, run_options)
        return binding.get_outputs()

    def run_dict(
        self,
        input_feed,
        output_shapes=None,
        run_options=None,
        prefer_cupy=False,
        use_io_binding=None,
    ):
        if use_io_binding is None:
            use_io_binding = self._use_cuda
        if not use_io_binding or not self._use_cuda:
            outputs = self._run_numpy_fallback(input_feed, run_options=run_options)
            return dict(zip(self._output_names, outputs))

        if output_shapes is not None and not all(name in output_shapes for name in self._output_names):
            output_shapes = None
        use_cupy = bool(prefer_cupy) and output_shapes is not None
        binding, buffers = self._run_bound(input_feed, run_options, output_shapes=output_shapes, use_cupy=use_cupy)
        if buffers is None:
            return dict(zip(self._output_names, binding.copy_outputs_to_cpu()))
        if use_cupy:
            return dict(buffers)
        return {name: buf.numpy() for name, buf in buffers.items()}


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
            value = self._s._to_numpy_input(name, value)
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
                    value = self._s._to_numpy_input(name, value)
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


def _session_cache_key(model_path, feature_name, providers, provider_options):
    try:
        opts_key = json.dumps(provider_options, sort_keys=True, default=str)
    except Exception:
        opts_key = repr(provider_options)
    prov_key = tuple(providers) if providers else ()
    return (os.path.abspath(model_path), feature_name, prov_key, opts_key)


def general_inference_session(model_path: str, sess_options, providers, provider_options):
    model_name = os.environ["_feature_name_"]
    cache_key = _session_cache_key(model_path, model_name, providers, provider_options)

    cached = _SESSION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _SESSION_CACHE_LOCK:
        cached = _SESSION_CACHE.get(cache_key)
        if cached is not None:
            return cached
        lic_path = os.path.join(os.path.join(pathlib.Path.home(), ".PowerTools", "license"), "license.lic")
        with open(lic_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        license_json = json.dumps(raw_data)
        old_cwd = os.getcwd()
        os.chdir(os.path.dirname(model_path))
        try:
            sess = model_loader.load_model_auto(
                model_path,
                model_name,
                license_json,
                session_options=sess_options,
                providers=providers,
                provider_options=provider_options
            )
        finally:
            os.chdir(old_cwd)
        wrapped = IOBindingSession(sess)
        _SESSION_CACHE[cache_key] = wrapped
        return wrapped


def evict_session_cache(model_paths):
    normalized_paths = {os.path.abspath(os.fspath(path)) for path in model_paths}
    removed = []
    with _SESSION_CACHE_LOCK:
        keys = [key for key in _SESSION_CACHE if key[0] in normalized_paths]
        for key in keys:
            wrapped = _SESSION_CACHE.pop(key)
            wrapped.clear_persistent_inputs()
            removed.append(wrapped)
    count = len(removed)
    removed.clear()
    return count


def clear_session_cache():
    removed = []
    with _SESSION_CACHE_LOCK:
        for wrapped in _SESSION_CACHE.values():
            wrapped.clear_persistent_inputs()
            removed.append(wrapped)
        _SESSION_CACHE.clear()
    removed.clear()



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


def general_session(
    intra_op_num_threads: int | None = None,
    inter_op_num_threads: int = 1,
    graph_optimization_level: str = "all",
    enable_cpu_mem_arena: bool = True,
    enable_mem_pattern: bool = True,
    optimized_model_path: str | None = None,
    log_severity_level: int = 2,
    free_dim_overrides: dict[str, int] | None = None,
    ) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.log_severity_level = log_severity_level
    options.graph_optimization_level = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[graph_optimization_level]
    options.add_session_config_entry("session.use_env_allocators", "1")
    options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if intra_op_num_threads:
        options.intra_op_num_threads = int(intra_op_num_threads)
    options.inter_op_num_threads = int(inter_op_num_threads)
    options.enable_cpu_mem_arena = bool(enable_cpu_mem_arena)
    options.enable_mem_pattern = bool(enable_mem_pattern)
    if optimized_model_path:
        options.optimized_model_filepath = str(optimized_model_path)
    for name, value in (free_dim_overrides or {}).items():
        options.add_free_dimension_override_by_name(name, int(value))
    return options