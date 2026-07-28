from pathlib import Path
from typing import Any
import numpy as np
import onnxruntime as ort
from app.algorithms import general_provider, general_session, general_inference_session, ORTEnvironment
ORTEnvironment.initialize()


def cupy_available() -> bool:
    try:
        import cupy
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
        options = session_options or general_session(**session_option_kwargs)
        providers, provider_options = general_provider()
        self.session = general_inference_session(
            model_path=str(self.path),
            sess_options=options,
            providers=providers,
            provider_options=provider_options,
        )
        self.is_gpu = self.session.use_cuda
        self.ort_device_type = self.session.device_type

        self.input_names = list(self.session.input_names)
        self.output_names = list(self.session.output_names)
        self.input_dtypes = dict(self.session.input_dtypes)
        self.output_dtypes = dict(self.session.output_dtypes)
        self.input_shapes = dict(self.session.input_shapes)
        self.output_shapes = dict(self.session.output_shapes)

        self.xp, self.uses_cupy = array_module("cuda" if self.is_gpu else "cpu", use_cupy)
        use_io_binding = self.is_gpu if use_io_binding is None else use_io_binding
        self.use_io_binding = bool(use_io_binding) and self.is_gpu

    @property
    def provider(self) -> str:
        return self.session.provider

    def cast(self, name: str, array) -> Any:
        want = self.input_dtypes.get(name)
        xp = self.xp if _is_cupy(array) else np
        if want is not None and array.dtype != want:
            array = array.astype(want, copy=False)
        if not array.flags["C_CONTIGUOUS"]:
            array = xp.ascontiguousarray(array)
        return array

    def set_static_input(self, name: str, array) -> None:
        self.session.set_static_input(name, self.cast(name, array))

    def clear_static_inputs(self) -> None:
        self.session.clear_static_inputs()

    def empty(self, shape, dtype) -> Any:
        return self.xp.empty(shape, dtype=dtype)

    def run(self, feed: dict[str, Any], output_shapes: dict[str, tuple] | None = None) -> dict[str, Any]:
        return self.session.run_dict(
            feed,
            output_shapes=output_shapes,
            prefer_cupy=self.uses_cupy,
            use_io_binding=self.use_io_binding,
        )

    def __call__(self, output_shapes: dict[str, tuple] | None = None, **feed) -> dict[str, Any]:
        return self.run(feed, output_shapes=output_shapes)

    def unload(self) -> None:
        session = self.session
        if session is not None:
            session.clear_static_inputs()
            session.clear_persistent_inputs()
        self.session = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.unload()
        return False
