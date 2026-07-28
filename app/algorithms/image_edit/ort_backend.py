import os
from pathlib import Path
from typing import Any
import numpy as np
import onnxruntime as ort
from app.algorithms import general_provider, general_session, general_inference_session, ORTEnvironment, is_gpu_device
ORTEnvironment.initialize()


_ORT_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(bfloat16)": np.float32,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}

_CUDA_PROVIDERS = ("TensorrtExecutionProvider", "CUDAExecutionProvider")
_DEVICE_PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    "rocm": ["ROCMExecutionProvider", "CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}


def cupy_available() -> bool:
    try:
        import cupy  # noqa: F401
    except Exception:
        return False
    return True


def resolve_device(device: str = "auto") -> str:
    """Map a user facing device string to one of the keys of `_DEVICE_PROVIDERS`."""
    if device not in (None, "auto"):
        return device
    available = set(ort.get_available_providers())
    for name, providers in (
        ("cuda", ["CUDAExecutionProvider"]),
        ("rocm", ["ROCMExecutionProvider"]),
        ("dml", ["DmlExecutionProvider"]),
    ):
        if providers[0] in available:
            return name
    return "cpu"


def array_module(device: str, use_cupy: bool | None = None):
    if device in ("cuda", "tensorrt", "rocm") and use_cupy is not False and cupy_available():
        import cupy
        return cupy, True
    return np, False


def asnumpy(array) -> np.ndarray:
    """`cupy`/`numpy` -> `numpy` without importing cupy eagerly."""
    if isinstance(array, np.ndarray):
        return array
    get = getattr(array, "get", None)
    if get is not None:
        return get()
    return np.asarray(array)


def _is_cupy(array) -> bool:
    return type(array).__module__.split(".")[0] == "cupy"


class OnnxModule:
    def __init__(
        self,
        path: str | Path,
        device: str = "auto",
        device_id: int = 0,
        use_io_binding: bool | None = None,
        session_options: ort.SessionOptions | None = None,
        providers: list | None = None,
        use_cupy: bool | None = None,
        **session_option_kwargs,
    ):
        self.path = Path(path)
        self.path = self.path / "model.onnx" if self.path.is_dir() else self.path
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.device = resolve_device(device)
        self.device_id = int(device_id)
        self.options = session_options or general_session(**session_option_kwargs)
        self.providers = providers if providers is not None else build_providers(self.device, self.device_id)
        self.session = ort.InferenceSession(str(self.path), sess_options=self.options, providers=self.providers)
        self.provider = self.session.get_providers()[0]
        self.is_gpu = is_gpu_device()
        self.ort_device_type = "cuda" if self.provider in _CUDA_PROVIDERS else (
            "hip" if self.provider == "ROCMExecutionProvider" else "cpu"
        )

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_dtypes = {i.name: _ORT_TO_NP[i.type] for i in self.session.get_inputs()}
        self.output_dtypes = {o.name: _ORT_TO_NP[o.type] for o in self.session.get_outputs()}
        self.input_shapes = {i.name: i.shape for i in self.session.get_inputs()}
        self.output_shapes = {o.name: o.shape for o in self.session.get_outputs()}

        self.xp, self.uses_cupy = array_module(self.device if self.is_gpu else "cpu", use_cupy)
        use_io_binding = self.is_gpu if use_io_binding is None else use_io_binding
        self.use_io_binding = bool(use_io_binding)
        self._binding = self.session.io_binding() if self.use_io_binding else None
        self._static: dict[str, Any] = {}
        self._out_buffers: dict[tuple, Any] = {}

    def __repr__(self) -> str:
        return (
            f"OnnxModule({self.path.parent.name}, provider={self.provider}, "
            f"io_binding={self.use_io_binding}, cupy={self.uses_cupy})"
        )

    def cast(self, name: str, array) -> Any:
        want = self.input_dtypes[name]
        xp = self.xp if _is_cupy(array) else np
        if array.dtype != want:
            array = array.astype(want, copy=False)
        if not array.flags["C_CONTIGUOUS"]:
            array = xp.ascontiguousarray(array)
        return array

    def set_static_input(self, name: str, array) -> None:
        """Upload an input once and reuse it for every subsequent `run`."""
        array = self.cast(name, array)
        if not self.use_io_binding:
            self._static[name] = array
            return
        if _is_cupy(array):
            self._static[name] = array
        else:
            self._static[name] = ort.OrtValue.ortvalue_from_numpy(
                np.ascontiguousarray(array), self.ort_device_type, self.device_id
            )

    def clear_static_inputs(self) -> None:
        self._static.clear()

    def empty(self, shape, dtype) -> Any:
        return self.xp.empty(shape, dtype=dtype)

    def _output_buffer(self, name: str, shape: tuple) -> Any:
        key = (name, tuple(shape))
        buf = self._out_buffers.get(key)
        if buf is not None:
            return buf
        dtype = self.output_dtypes[name]
        if self.uses_cupy:
            buf = self.xp.empty(shape, dtype=dtype)
        else:
            buf = ort.OrtValue.ortvalue_from_shape_and_type(
                list(shape), dtype, self.ort_device_type, self.device_id
            )
        self._out_buffers[key] = buf
        return buf

    def _bind_input(self, name: str, value) -> None:
        binding = self._binding
        if isinstance(value, ort.OrtValue):
            binding.bind_ortvalue_input(name, value)
        elif _is_cupy(value):
            binding.bind_input(
                name,
                self.ort_device_type,
                self.device_id,
                self.input_dtypes[name],
                tuple(value.shape),
                value.data.ptr,
            )
        elif self.ort_device_type == "cpu":
            binding.bind_cpu_input(name, value)
        else:
            binding.bind_ortvalue_input(
                name, ort.OrtValue.ortvalue_from_numpy(value, self.ort_device_type, self.device_id)
            )

    def _bind_output(self, name: str, buf) -> None:
        if _is_cupy(buf):
            self._binding.bind_output(
                name,
                self.ort_device_type,
                self.device_id,
                self.output_dtypes[name],
                tuple(buf.shape),
                buf.data.ptr,
            )
        else:
            self._binding.bind_ortvalue_output(name, buf)

    def run(self, feed: dict[str, Any], output_shapes: dict[str, tuple] | None = None) -> dict[str, Any]:
        if not self.use_io_binding:
            inputs = {}
            for name in self.input_names:
                value = feed.get(name, self._static.get(name))
                if value is None:
                    raise ValueError(f"missing input {name!r} for {self.path.parent.name}")
                inputs[name] = np.ascontiguousarray(asnumpy(self.cast(name, value)))
            outputs = self.session.run(None, inputs)
            return dict(zip(self.output_names, outputs))

        binding = self._binding
        binding.clear_binding_inputs()
        binding.clear_binding_outputs()
        for name in self.input_names:
            value = feed.get(name)
            if value is None:
                value = self._static.get(name)
                if value is None:
                    raise ValueError(f"missing input {name!r} for {self.path.parent.name}")
            else:
                value = self.cast(name, value)
            self._bind_input(name, value)

        if output_shapes:
            buffers = {name: self._output_buffer(name, output_shapes[name]) for name in self.output_names}
            for name, buf in buffers.items():
                self._bind_output(name, buf)
            if self.uses_cupy:
                # cupy buffers live on cupy's stream, ORT runs on its own: the
                # documented safe pattern is to synchronize around the call.
                binding.synchronize_inputs()
            self.session.run_with_iobinding(binding)
            if self.uses_cupy:
                binding.synchronize_outputs()
            return {
                name: (buf if _is_cupy(buf) else buf.numpy()) for name, buf in buffers.items()
            }

        # let ORT allocate the outputs (shape unknown to the caller)
        for name in self.output_names:
            binding.bind_output(name, self.ort_device_type, self.device_id)
        self.session.run_with_iobinding(binding)
        return dict(zip(self.output_names, binding.copy_outputs_to_cpu()))

    def __call__(self, output_shapes: dict[str, tuple] | None = None, **feed) -> dict[str, Any]:
        return self.run(feed, output_shapes=output_shapes)

    def unload(self) -> None:
        self._static.clear()
        self._out_buffers.clear()
        self._binding = None
        self.session = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.unload()
        return False
