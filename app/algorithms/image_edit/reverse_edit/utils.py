import json
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tokenizers import Tokenizer


ONNX_WEIGHTS_NAME = "model.onnx"
PIPELINE_CONFIG_NAME = "model_index.json"
SCHEDULER_CONFIG_NAME = "scheduler_config.json"
COMPONENT_CONFIG_NAME = "config.json"
TOKENIZER_FILE_NAME = "tokenizer.json"
HF_TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
SPECIAL_TOKENS_MAP_NAME = "special_tokens_map.json"


def load_json(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_load_json(path: str | os.PathLike) -> dict:
    path = Path(path)
    return load_json(path) if path.is_file() else {}


def pil_to_numpy(image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    arr = arr.astype(np.float32) / 255.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])


def preprocess_image(image, width: int | None = None, height: int | None = None) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim == 3:  # HWC
            arr = image
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            if arr.max() > 1.5:
                arr = arr / 255.0
            arr = np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
        elif image.ndim == 4:  # NCHW
            arr = image.astype(np.float32, copy=False)
        else:
            raise ValueError(f"unsupported image ndim: {image.ndim}")
    else:
        if width and height and image.size != (width, height):
            image = image.resize((width, height))
        arr = pil_to_numpy(image)
    if width and height and (arr.shape[-1] != width or arr.shape[-2] != height):
        arr = resize_nchw(arr, width, height)
    return np.ascontiguousarray(2.0 * arr - 1.0)


def resize_nchw(x: np.ndarray, width: int, height: int, interp: int = cv2.INTER_LANCZOS4) -> np.ndarray:
    n, c, _, _ = x.shape
    out = np.empty((n, c, height, width), dtype=x.dtype)
    for i in range(n):
        hwc = x[i].transpose(1, 2, 0)
        resized = cv2.resize(hwc, (width, height), interpolation=interp)
        if resized.ndim == 2:
            resized = resized[:, :, None]
        out[i] = resized.transpose(2, 0, 1)
    return out


def denormalize(x: np.ndarray) -> np.ndarray:
    return np.clip(x / 2.0 + 0.5, 0.0, 1.0)


def numpy_to_pil(images: np.ndarray) -> list:
    arr = (images.transpose(0, 2, 3, 1) * 255.0).round().astype(np.uint8)
    return [Image.fromarray(a[:, :, 0], mode="L") if a.shape[-1] == 1 else Image.fromarray(a) for a in arr]


class DiagonalGaussianDistribution:
    def __init__(self, parameters: np.ndarray):
        self.parameters = parameters
        self.mean, self.logvar = np.split(parameters.astype(np.float32, copy=False), 2, axis=1)
        self.logvar = np.clip(self.logvar, -30.0, 20.0)
        self.std = np.exp(0.5 * self.logvar)

    def sample(self, generator: np.random.Generator | None = None) -> np.ndarray:
        rng = generator if generator is not None else np.random.default_rng()
        noise = rng.standard_normal(self.mean.shape, dtype=np.float32)
        return self.mean + self.std * noise

    def mode(self) -> np.ndarray:
        return self.mean


def randn(shape, generator: np.random.Generator | None = None, dtype=np.float32) -> np.ndarray:
    rng = generator if generator is not None else np.random.default_rng()
    return rng.standard_normal(tuple(shape), dtype=np.float32).astype(dtype, copy=False)


def _token_text(value) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("content")
    return None


class CLIPTokenizer:
    def __init__(self, tokenizer_dir: str | os.PathLike, model_max_length: int | None = None):
        tokenizer_dir = Path(tokenizer_dir)
        self._tokenizer = Tokenizer.from_file(str(tokenizer_dir / TOKENIZER_FILE_NAME))
        cfg = maybe_load_json(tokenizer_dir / HF_TOKENIZER_CONFIG_NAME)
        special = maybe_load_json(tokenizer_dir / SPECIAL_TOKENS_MAP_NAME)
        max_len = model_max_length or cfg.get("model_max_length") or 77
        self.model_max_length = int(max_len) if 0 < int(max_len) <= 10_000 else 77
        self.pad_token = (_token_text(cfg.get("pad_token")) or _token_text(special.get("pad_token")) or "<|endoftext|>")
        pad_id = self._tokenizer.token_to_id(self.pad_token)
        self.pad_token_id = int(pad_id) if pad_id is not None else 49407
        self._tokenizer.enable_truncation(max_length=self.model_max_length)
        self._tokenizer.enable_padding(
            pad_id=self.pad_token_id,
            pad_token=self.pad_token,
            length=self.model_max_length,
        )

    def __call__(self, text: str | list[str], dtype=np.int64) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
        encodings = self._tokenizer.encode_batch(text)
        return np.asarray([e.ids for e in encodings], dtype=dtype)
