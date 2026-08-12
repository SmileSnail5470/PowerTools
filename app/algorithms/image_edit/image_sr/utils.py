import json
import os
import cv2
from pathlib import Path
from PIL import Image
import numpy as np
from onnxruntime import InferenceSession, OrtValue
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import RobertaProcessing


_ORT_TYPE_TO_NP = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    "tensor(bool)": np.bool_,
}


def ort_type_to_numpy_type(ort_type: str) -> np.dtype:
    return _ORT_TYPE_TO_NP.get(ort_type, np.float32)


class ORTSessionWrapper:
    def __init__(self, session: InferenceSession, use_io_binding: bool | None = None):
        self.session = session
        self.path = Path(session._model_path)
        self.input_names = {inp.name: idx for idx, inp in enumerate(session.get_inputs())}
        self.output_names = {out.name: idx for idx, out in enumerate(session.get_outputs())}
        self.input_dtypes = {inp.name: inp.type for inp in session.get_inputs()}
        self.output_dtypes = {out.name: out.type for out in session.get_outputs()}
        self.input_shapes = {inp.name: inp.shape for inp in session.get_inputs()}
        self._dtype = np.float32
        for inp in session.get_inputs():
            np_type = ort_type_to_numpy_type(inp.type)
            if np_type in (np.float16, np.float32, np.float64):
                self._dtype = np_type
                break

        self._provider = session.get_providers()[0]
        if use_io_binding is None:
            use_io_binding = self._provider == "CUDAExecutionProvider"
        self.use_io_binding = use_io_binding

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def device_type(self) -> str:
        return "cuda" if self._provider == "CUDAExecutionProvider" else "cpu"

    @property
    def device_id(self) -> int:
        if self._provider == "CUDAExecutionProvider":
            opts = self.session.get_provider_options().get("CUDAExecutionProvider", {})
            return int(opts.get("device_id", 0))
        return 0

    def _prepare_input(self, name: str, val) -> OrtValue:
        if isinstance(val, OrtValue):
            return val
        expected_dtype = ort_type_to_numpy_type(self.input_dtypes[name])
        if val.dtype != expected_dtype:
            val = val.astype(expected_dtype)
        return OrtValue.ortvalue_from_numpy(val, self.device_type, self.device_id)

    def _run_io_binding_gpu(self, kwargs: dict) -> dict[str, OrtValue]:
        io_binding = self.session.io_binding()
        input_ortvals = []
        for name in self.input_names:
            ort_val = self._prepare_input(name, kwargs[name])
            io_binding.bind_ortvalue_input(name, ort_val)
            input_ortvals.append(ort_val)
        # Bind outputs to GPU
        device = self.device_type
        device_id = self.device_id
        for name in self.output_names:
            io_binding.bind_output(name, device, device_id)
        self.session.run_with_iobinding(io_binding)
        ort_outputs = io_binding.get_outputs()
        return {name: ort_outputs[idx] for name, idx in self.output_names.items()}

    def run_gpu(self, **kwargs) -> dict[str, OrtValue]:
        for name in self.input_names:
            if name not in kwargs or kwargs[name] is None:
                raise ValueError(f"Input '{name}' is required but not provided.")
        if self.use_io_binding:
            return self._run_io_binding_gpu(kwargs)
        # Fallback: non-GPU path returns wrapped numpy
        return self(**kwargs)

    def __call__(self, **kwargs) -> dict[str, np.ndarray]:
        onnx_inputs = {}
        for name in self.input_names:
            if name not in kwargs or kwargs[name] is None:
                raise ValueError(f"Input '{name}' is required but not provided.")
            arr = kwargs[name]
            if isinstance(arr, OrtValue):
                arr = arr.numpy()
            expected_dtype = ort_type_to_numpy_type(self.input_dtypes[name])
            if arr.dtype != expected_dtype:
                arr = arr.astype(expected_dtype)
            onnx_inputs[name] = arr

        if self.use_io_binding:
            io_binding = self.session.io_binding()
            device = self.device_type
            device_id = self.device_id
            for name, arr in onnx_inputs.items():
                ort_val = OrtValue.ortvalue_from_numpy(arr, device, device_id)
                io_binding.bind_ortvalue_input(name, ort_val)
            for name in self.output_names:
                io_binding.bind_output(name, device, device_id)
            self.session.run_with_iobinding(io_binding)
            ort_outputs = io_binding.get_outputs()
            return {name: ort_outputs[idx].numpy() for name, idx in self.output_names.items()}

        outputs = self.session.run(None, onnx_inputs)
        return {name: outputs[idx] for name, idx in self.output_names.items()}


class DiagonalGaussianDistribution:
    def __init__(self, parameters: np.ndarray):
        self.parameters = parameters
        self.mean, self.logvar = np.split(parameters, 2, axis=1)
        self.logvar = np.clip(self.logvar, -30.0, 20.0)
        self.std = np.exp(0.5 * self.logvar)
        self.var = np.exp(self.logvar)

    def sample(self, generator=None) -> np.ndarray:
        noise = np.random.randn(*self.mean.shape).astype(self.mean.dtype)
        return self.mean + self.std * noise

    def mode(self) -> np.ndarray:
        return self.mean


def load_config(model_path: str | os.PathLike) -> dict:
    model_path = Path(model_path)
    for config_name in ["model_index.json", "config.json"]:
        config_path = model_path / config_name
        if config_path.is_file():
            with open(config_path, "r") as f:
                return json.load(f)
    raise FileNotFoundError(f"No configuration file found in {model_path}")


def numpy_to_pil(images: np.ndarray):
    if images.ndim == 3:
        images = images[np.newaxis, ...]

    if images.shape[1] in (1, 3, 4) and images.shape[3] not in (1, 3, 4):
        images = np.transpose(images, (0, 2, 3, 1))

    images = np.clip(images * 255.0, 0, 255).astype(np.uint8)

    pil_images = []
    for img in images:
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
        pil_images.append(Image.fromarray(img))
    return pil_images


def pil_to_numpy(images) -> np.ndarray:
    if not isinstance(images, (list, tuple)):
        images = [images]

    np_images = []
    for img in images:
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
        arr = 2.0 * arr - 1.0
        np_images.append(arr)

    return np.stack(np_images, axis=0)


def prepare_mask(mask_image, height: int, width: int, vae_scale_factor: int = 8) -> np.ndarray:
    if hasattr(mask_image, 'convert'):
        mask_image = mask_image.convert("L")
        mask = np.array(mask_image).astype(np.float32) / 255.0
    elif isinstance(mask_image, np.ndarray):
        if mask_image.ndim == 3:
            mask = mask_image[:, :, 0].astype(np.float32)
        else:
            mask = mask_image.astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
    else:
        raise ValueError(f"Unsupported mask type: {type(mask_image)}")

    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor
    mask = cv2.resize(mask, (latent_w, latent_h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.float32)
    # Shape: [1, 1, H, W]
    return mask[np.newaxis, np.newaxis, :, :]


def prepare_image(image, height: int, width: int) -> np.ndarray:
    if hasattr(image, 'convert'):
        image = image.convert("RGB")
        img = np.array(image).astype(np.float32)
    elif isinstance(image, np.ndarray):
        img = image.astype(np.float32)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    if img.max() > 1.0:
        img = img / 255.0

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)

    # HWC -> CHW, normalize to [-1, 1]
    img = np.transpose(img, (2, 0, 1))
    img = 2.0 * img - 1.0
    return img[np.newaxis, :, :, :].astype(np.float32)


class Tokenizer:
    def __init__(self, tokenizer_path: str | Path):
        self.tokenizer_path = Path(tokenizer_path)
        config_path = self.tokenizer_path / "tokenizer_config.json"
        self._config = {}
        if config_path.is_file():
            with open(config_path) as f:
                self._config = json.load(f)
        special_tokens_path = self.tokenizer_path / "special_tokens_map.json"
        self._special_tokens_map = {}
        if special_tokens_path.is_file():
            with open(special_tokens_path) as f:
                self._special_tokens_map = json.load(f)
        tokenizer_json = self.tokenizer_path / "tokenizer.json"
        vocab_json = self.tokenizer_path / "vocab.json"
        merges_txt = self.tokenizer_path / "merges.txt"

        if tokenizer_json.is_file():
            self._tokenizer = HFTokenizer.from_file(str(tokenizer_json))
        elif vocab_json.is_file() and merges_txt.is_file():
            self._tokenizer = HFTokenizer(BPE.from_file(str(vocab_json), str(merges_txt)))
            self._tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            bos_token = self._resolve_special_token("bos_token", "<|startoftext|>")
            eos_token = self._resolve_special_token("eos_token", "<|endoftext|>")
            bos_id = self._tokenizer.token_to_id(bos_token)
            eos_id = self._tokenizer.token_to_id(eos_token)
            if bos_id is not None and eos_id is not None:
                self._tokenizer.post_processor = RobertaProcessing(
                    sep=(eos_token, eos_id),
                    cls=(bos_token, bos_id),
                    add_prefix_space=False,
                    trim_offsets=True,
                )
        else:
            raise FileNotFoundError(f"No tokenizer.json or vocab.json+merges.txt found in {tokenizer_path}")
        
        self.model_max_length = self._config.get("model_max_length", 77)

    def _resolve_special_token(self, token_key: str, default: str) -> str:
        val = self._special_tokens_map.get(token_key)
        if val is not None:
            if isinstance(val, dict):
                return val.get("content", default)
            return val
        val = self._config.get(token_key)
        if val is not None:
            if isinstance(val, dict):
                return val.get("content", default)
            return val
        return default

    def __call__(self, text: str | list[str], padding: str = "max_length", max_length: int | None = None, truncation: bool = True, return_attention_mask: bool = True) -> dict[str, np.ndarray]:
        if isinstance(text, str):
            text = [text]
        max_len = max_length or self.model_max_length
        pad_token = self._resolve_special_token("pad_token", "<|endoftext|>")
        pad_id = self._tokenizer.token_to_id(pad_token) or 0
        self._tokenizer.enable_padding(
            length=max_len if padding == "max_length" else None,
            pad_id=pad_id,
        )
        if truncation:
            self._tokenizer.enable_truncation(max_length=max_len)
        encodings = self._tokenizer.encode_batch(text)
        input_ids = np.array([enc.ids[:max_len] for enc in encodings], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask[:max_len] for enc in encodings], dtype=np.int64)
        result = {"input_ids": input_ids}
        if return_attention_mask:
            result["attention_mask"] = attention_mask
        return result
