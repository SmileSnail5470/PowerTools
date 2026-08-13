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


sr_edit_logger = logging.getLogger("ImageSR")
WEIGHT_FILE_NAMES = ("model.encmodel", "model.onnx")


def cupy_available() -> bool:
    return _HAS_CUPY


def asnumpy(array) -> Any:
    if array is None or isinstance(array, np.ndarray):
        return array
    if isinstance(array, ort.OrtValue):
        return array.numpy()
    get = getattr(array, "get", None)
    if get is not None:
        return get()
    return np.asarray(array)


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
    for suffix in (".encmodel", ".onnx"):
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path)


class OnnxModule:
    def __init__(
        self,
        path: str | Path,
        use_io_binding: bool | None = None,
        use_cupy: bool = False,
        session_options: ort.SessionOptions | None = None,
        verbose: bool = True,
        **session_option_kwargs,
    ):
        self.path = resolve_weights(path)
        self.verbose = verbose
        self._session_options = session_options
        self._session_option_kwargs = session_option_kwargs
        self._requested_io_binding = use_io_binding
        self._requested_cupy = bool(use_cupy)
        self._session = None
        self.is_gpu = False
        self.use_io_binding = False
        self.uses_cupy = False
        self.xp = np
        self.input_dtypes: dict[str, np.dtype] = {}
        self.output_dtypes: dict[str, np.dtype] = {}

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
    def provider(self) -> str:
        return self.session.provider

    @property
    def input_names(self) -> list[str]:
        return self.session.input_names

    @property
    def output_names(self) -> list[str]:
        return self.session.output_names

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
        self.is_gpu = session.use_cuda
        self.input_dtypes = dict(session.input_dtypes)
        self.output_dtypes = dict(session.output_dtypes)
        use_io_binding = self.is_gpu if self._requested_io_binding is None else self._requested_io_binding
        self.use_io_binding = bool(use_io_binding) and self.is_gpu
        self.uses_cupy = self._requested_cupy and self.is_gpu and cupy_available()
        self.xp = cupy if self.uses_cupy else np
        if self.verbose:
            sr_edit_logger.info(f"[onnx] loaded {self.name} in {time.perf_counter() - start:.1f}s ({session.provider})")

    def expected_dtype(self, name: str) -> np.dtype | None:
        return self.session.input_dtypes.get(name)

    def declared_input_shape(self, name: str) -> list:
        return list(self.session.input_shapes.get(name) or [])

    def declared_output_shape(self, name: str) -> list:
        return list(self.session.output_shapes.get(name) or [])

    def cast(self, name: str, array):
        want = self.input_dtypes.get(name)
        if isinstance(array, ort.OrtValue):
            return array
        xp = self.xp if type(array).__module__.split(".")[0] == "cupy" else np
        array = xp.asarray(array)
        if want is not None and array.dtype != want:
            array = array.astype(want, copy=False)
        if not array.flags["C_CONTIGUOUS"]:
            array = xp.ascontiguousarray(array)
        return array

    def set_static_input(self, name: str, array) -> None:
        session = self.session
        session.set_static_input(name, self.cast(name, array))

    def clear_static_inputs(self) -> None:
        if self._session is not None:
            self._session.clear_static_inputs()

    def run(self, feed: dict[str, Any], output_shapes: dict[str, tuple] | None = None) -> dict[str, Any]:
        session = self.session
        casted = {name: self.cast(name, value) for name, value in feed.items()}
        return session.run_dict(
            casted,
            output_shapes=output_shapes,
            prefer_cupy=self.uses_cupy,
            use_io_binding=self.use_io_binding,
        )

    def __call__(self, output_shapes: dict[str, tuple] | None = None, **feed) -> dict[str, Any]:
        return self.run(feed, output_shapes=output_shapes)

    def unload(self) -> None:
        session, self._session = self._session, None
        if session is None:
            return
        session.clear_static_inputs()
        session.clear_persistent_inputs()
        evict_session_cache([str(self.path)])
        del session
        gc.collect()
        if self.verbose:
            sr_edit_logger.info(f"[onnx] released {self.name}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.unload()
        return False
