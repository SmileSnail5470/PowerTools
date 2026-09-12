import gc
import logging
import time
from pathlib import Path
from typing import Any
import numpy as np
import onnxruntime as ort
from app.algorithms import (
    ORTEnvironment,
    evict_session_cache,
    general_inference_session,
    general_provider,
    general_session,
)
ORTEnvironment.initialize()
try:
    import cupy  # type: ignore
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False


image_edit_logger = logging.getLogger("ImageEdit")
WEIGHT_FILE_NAMES = ("model.encmodel", "model.onnx")


def cupy_available() -> bool:
    return _HAS_CUPY


def array_module(device: str, use_cupy: bool | None = None):
    if device in ("cuda", "tensorrt", "rocm") and use_cupy is not False and cupy_available():
        return cupy, True
    return np, False


def asnumpy(array) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    get = getattr(array, "get", None)
    if get is not None:
        return get()
    return np.asarray(array)


def _is_cupy(array) -> bool:
    return type(array).__module__.split(".")[0] == "cupy"


def resolve_weights(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        for name in WEIGHT_FILE_NAMES:
            candidate = path / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"no onnx model found in {path}")
    if path.is_file():
        return path
    raise FileNotFoundError(path)


class OnnxModule:
    def __init__(
        self,
        path: str | Path,
        use_io_binding: bool | None = None,
        session_options: ort.SessionOptions | None = None,
        use_cupy: bool = True,
        verbose: bool = True,
        **session_option_kwargs,
    ):
        self.path = resolve_weights(path)
        self.verbose = verbose
        self._session_options = session_options
        self._session_option_kwargs = session_option_kwargs
        self._requested_io_binding = use_io_binding
        self._requested_cupy = use_cupy
        self._session = None
        self._is_gpu = False
        self._use_io_binding = False
        self._uses_cupy = False
        self._xp = np
        self._input_dtypes: dict[str, np.dtype] = {}
        self._run_options = None

    @property
    def name(self) -> str:
        return self.path.parent.name or self.path.stem

    @property
    def loaded(self) -> bool:
        return self._session is not None

    @property
    def session(self):
        if self._session is None:
            self._load()
        return self._session

    @property
    def is_gpu(self) -> bool:
        self.session
        return self._is_gpu

    @property
    def use_io_binding(self) -> bool:
        self.session
        return self._use_io_binding

    @property
    def uses_cupy(self) -> bool:
        self.session
        return self._uses_cupy

    @property
    def xp(self):
        self.session
        return self._xp

    @property
    def input_dtypes(self) -> dict[str, np.dtype]:
        self.session
        return self._input_dtypes

    @property
    def run_options(self):
        self.session
        return self._run_options

    @property
    def provider(self) -> str:
        return self.session.provider

    def _build_run_options(self, shrink_memory=True, use_cuda=False):
        options = ort.RunOptions()
        if shrink_memory:
            device = "gpu:0" if use_cuda else "cpu"
            options.add_run_config_entry("memory.enable_memory_arena_shrinkage", device)
        return options

    def _load(self) -> None:
        options = self._session_options or general_session(**self._session_option_kwargs)
        providers, provider_options = general_provider()
        start = time.perf_counter()
        session = general_inference_session(
            model_path=str(self.path),
            sess_options=options,
            providers=providers,
            provider_options=provider_options,
        )
        self._session = session
        self._is_gpu = session.use_cuda
        self._input_dtypes = dict(session.input_dtypes)
        use_io_binding = self._is_gpu if self._requested_io_binding is None else self._requested_io_binding
        self._use_io_binding = bool(use_io_binding) and self._is_gpu
        self._xp, self._uses_cupy = array_module("cuda" if self._is_gpu else "cpu", self._requested_cupy)
        self._run_options = self._build_run_options(shrink_memory=True, use_cuda=self._is_gpu)
        if self.verbose:
            image_edit_logger.info(f"[onnx] loaded {self.name} in {time.perf_counter() - start:.1f}s ({session.provider})")

    def cast(self, name: str, array) -> Any:
        want = self.input_dtypes.get(name)
        xp = self._xp if _is_cupy(array) else np
        if want is not None and array.dtype != want:
            array = array.astype(want, copy=False)
        if not array.flags["C_CONTIGUOUS"]:
            array = xp.ascontiguousarray(array)
        return array

    def set_static_input(self, name: str, array) -> None:
        self.session.set_static_input(name, self.cast(name, array))

    def clear_static_inputs(self) -> None:
        if self._session is not None:
            self._session.clear_static_inputs()

    def empty(self, shape, dtype) -> Any:
        return self.xp.empty(shape, dtype=dtype)

    def run(self, feed: dict[str, Any], output_shapes: dict[str, tuple] | None = None) -> dict[str, Any]:
        session = self.session
        return session.run_dict(
            feed,
            output_shapes=output_shapes,
            run_options=self._run_options,
            prefer_cupy=self._uses_cupy,
            use_io_binding=self._use_io_binding,
        )

    def __call__(self, output_shapes: dict[str, tuple] | None = None, **feed) -> dict[str, Any]:
        return self.run(feed, output_shapes=output_shapes)

    def unload(self) -> None:
        session, self._session = self._session, None
        was_gpu = self._is_gpu
        self._is_gpu = False
        self._use_io_binding = False
        self._uses_cupy = False
        self._xp = np
        self._input_dtypes = {}
        self._run_options = None
        if session is None:
            return
        session.clear_static_inputs()
        session.clear_persistent_inputs()
        evict_session_cache([str(self.path)])
        del session
        gc.collect()
        if was_gpu and _HAS_CUPY:
            try:
                cupy.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        if self.verbose:
            image_edit_logger.info(f"[onnx] released {self.name}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.unload()
        return False
