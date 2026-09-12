import os
import cv2
import shutil
import tempfile
import numpy as np
from pathlib import Path
from typing import Any



_ORIGINAL_VIDEO_CAPTURE = cv2.VideoCapture


def imread(filename: str | os.PathLike[str], flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    try:
        with open(os.fspath(filename), "rb") as image_file:
            encoded = np.frombuffer(image_file.read(), dtype=np.uint8)
        if encoded.size == 0:
            return None
        return cv2.imdecode(encoded, flags)
    except (OSError, TypeError, ValueError):
        return None


def imwrite(filename: str | os.PathLike[str], img: np.ndarray, params: list[int] | tuple[int, ...] | None = None) -> bool:
    try:
        suffix = Path(os.fspath(filename)).suffix
        if not suffix:
            return False
        encode_params = list(params) if params is not None else []
        success, encoded = cv2.imencode(suffix, img, encode_params)
        if not success:
            return False
        with open(os.fspath(filename), "wb") as image_file:
            image_file.write(encoded.tobytes())
        return True
    except (cv2.error, OSError, TypeError, ValueError):
        return False


class _TemporaryVideoCapture:
    def __init__(self, capture: Any, temporary_directory: str):
        self._capture = capture
        self._temporary_directory = temporary_directory
        self._released = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._capture, name)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            self._capture.release()
        finally:
            shutil.rmtree(self._temporary_directory, ignore_errors=True)

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


def open_video_capture(path: str | os.PathLike[str], *args: Any, **kwargs: Any) -> Any:
    source = os.fspath(path)
    capture = _ORIGINAL_VIDEO_CAPTURE(source, *args, **kwargs)
    if capture.isOpened() or not os.path.isfile(source):
        return capture
    capture.release()
    temporary_directory = tempfile.mkdtemp(prefix="powertools-video-")
    temporary_path = os.path.join(temporary_directory, f"input{Path(source).suffix}")
    try:
        shutil.copy2(source, temporary_path)
        fallback_capture = _ORIGINAL_VIDEO_CAPTURE(temporary_path, *args, **kwargs)
        if fallback_capture.isOpened():
            return _TemporaryVideoCapture(fallback_capture, temporary_directory)
        fallback_capture.release()
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    shutil.rmtree(temporary_directory, ignore_errors=True)
    return _ORIGINAL_VIDEO_CAPTURE(source, *args, **kwargs)


def install_unicode_opencv_io() -> None:
    if getattr(cv2.imread, "_powertools_unicode_io", False):
        return
    imread._powertools_unicode_io = True
    imwrite._powertools_unicode_io = True
    cv2.imread = imread
    cv2.imwrite = imwrite
