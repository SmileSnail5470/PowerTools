import json
import math
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from app.algorithms.image_edit.general_edit.backend import OnnxModule, asnumpy


image_edit_logger = logging.getLogger("ImageEdit")


@dataclass
class PipelineOutput:
    images: list[Image.Image] | np.ndarray
    latents: np.ndarray | None = None
    timings: dict[str, float] = field(default_factory=dict)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


class FlowMatchEulerScheduler:
    def __init__(self, config: dict):
        self.num_train_timesteps = int(config.get("num_train_timesteps", 1000))
        self.shift = float(config.get("shift", 1.0))
        self.use_dynamic_shifting = bool(config.get("use_dynamic_shifting", False))
        self.time_shift_type = config.get("time_shift_type", "exponential")
        self.shift_terminal = config.get("shift_terminal")
        self.invert_sigmas = bool(config.get("invert_sigmas", False))
        for unsupported in ("use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas"):
            if config.get(unsupported):
                raise NotImplementedError(f"scheduler option {unsupported} is not supported")
        self.sigmas = np.zeros(0, dtype=np.float32)
        self.timesteps = np.zeros(0, dtype=np.float32)

    def _time_shift(self, mu: float, sigma: float, t: np.ndarray) -> np.ndarray:
        if self.time_shift_type == "exponential":
            return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)
        return mu / (mu + (1.0 / t - 1.0) ** sigma)

    def set_timesteps(
        self,
        num_inference_steps: int,
        sigmas: Sequence[float] | None = None,
        mu: float | None = None,
    ) -> np.ndarray:
        if sigmas is None:
            timesteps = np.linspace(self.num_train_timesteps, self.num_train_timesteps / num_inference_steps, num_inference_steps)
            sigmas = timesteps / self.num_train_timesteps
        sigmas = np.asarray(sigmas, dtype=np.float32)
        if self.use_dynamic_shifting:
            if mu is None:
                raise ValueError("`mu` must be passed when `use_dynamic_shifting=True`")
            sigmas = self._time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        if self.shift_terminal:
            one_minus_z = 1 - sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            sigmas = 1 - (one_minus_z / scale_factor)
        sigmas = np.asarray(sigmas, dtype=np.float32)
        if self.invert_sigmas:
            sigmas = 1.0 - sigmas
            self.timesteps = sigmas * self.num_train_timesteps
            self.sigmas = np.concatenate([sigmas, np.ones(1, dtype=np.float32)])
        else:
            self.timesteps = sigmas * self.num_train_timesteps
            self.sigmas = np.concatenate([sigmas, np.zeros(1, dtype=np.float32)])
        return self.timesteps

    def step(self, sample, model_output, index: int):
        dt = float(self.sigmas[index + 1]) - float(self.sigmas[index])
        return sample + dt * model_output


def rope_embeddings(ids: np.ndarray, theta: int, axes_dim: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(ids, dtype=np.float64)
    cos_out, sin_out = [], []
    for axis, dim in enumerate(axes_dim):
        freqs = 1.0 / (float(theta) ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
        angles = np.outer(pos[:, axis], freqs)
        cos_out.append(np.repeat(np.cos(angles), 2, axis=1))
        sin_out.append(np.repeat(np.sin(angles), 2, axis=1))
    return (
        np.concatenate(cos_out, axis=-1).astype(np.float32),
        np.concatenate(sin_out, axis=-1).astype(np.float32),
    )


def cartesian_ids(t_values: Sequence[int], height: int, width: int, length: int = 1) -> np.ndarray:
    t = np.asarray(t_values, dtype=np.int64).reshape(-1)
    grid = np.meshgrid(t, np.arange(height), np.arange(width), np.arange(length), indexing="ij")
    return np.stack(grid, axis=-1).reshape(-1, 4).astype(np.int64)


def patchify_latents(latents: np.ndarray) -> np.ndarray:
    b, c, h, w = latents.shape
    x = latents.reshape(b, c, h // 2, 2, w // 2, 2)
    x = x.transpose(0, 1, 3, 5, 2, 4)
    return x.reshape(b, c * 4, h // 2, w // 2)


def unpatchify_latents(latents: np.ndarray) -> np.ndarray:
    b, c, h, w = latents.shape
    x = latents.reshape(b, c // 4, 2, 2, h, w)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(b, c // 4, h * 2, w * 2)


def pack_latents(latents: np.ndarray) -> np.ndarray:
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).transpose(0, 2, 1)


def unpack_latents_with_ids(x: np.ndarray, ids: np.ndarray, height: int, width: int) -> np.ndarray:
    batch, _, channels = x.shape
    out = np.zeros((batch, height * width, channels), dtype=x.dtype)
    flat = (ids[:, 1].astype(np.int64) * width + ids[:, 2].astype(np.int64))
    for i in range(batch):
        out[i, flat] = x[i]
    return out.reshape(batch, height, width, channels).transpose(0, 3, 1, 2)


def check_image_input(image: Image.Image, max_aspect_ratio: int = 8, min_side_length: int = 64) -> None:
    if not isinstance(image, Image.Image):
        raise ValueError(f"Image must be a PIL.Image.Image, got {type(image)}")
    width, height = image.size
    if width < min_side_length or height < min_side_length:
        raise ValueError(f"Image too small: {width}x{height}. Both dimensions must be at least {min_side_length}px")
    ratio = max(width / height, height / width)
    if ratio > max_aspect_ratio:
        raise ValueError(
            f"Aspect ratio too extreme: {width}x{height} (ratio: {ratio:.1f}:1). "
            f"Maximum allowed ratio is {max_aspect_ratio}:1"
        )


def preprocess_condition_image(image: Image.Image, max_area: int = 1024 * 1024, multiple_of: int = 16) -> tuple[np.ndarray, int, int]:
    image = image.convert("RGB")
    width, height = image.size
    if width * height > max_area:
        scale = math.sqrt(max_area / (width * height))
        width, height = int(width * scale), int(height * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    target_w = (width // multiple_of) * multiple_of
    target_h = (height // multiple_of) * multiple_of
    left = (width - target_w) // 2
    top = (height - target_h) // 2
    image = image.crop((left, top, left + target_w, top + target_h))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = array.transpose(2, 0, 1)[None]
    return array * 2.0 - 1.0, target_w, target_h


def postprocess_image(sample: np.ndarray, output_type: str = "pil"):
    images = (sample / 2.0 + 0.5).clip(0.0, 1.0)
    images = images.transpose(0, 2, 3, 1)
    if output_type == "np":
        return images
    uint8 = (images * 255).round().astype(np.uint8)
    return [Image.fromarray(frame) for frame in uint8]


class Pipeline:
    def __init__(
        self,
        model_dir: str | Path,
        use_io_binding: bool | None = None,
        use_cupy: bool = True,
        low_memory: bool = True,
        intra_op_num_threads: int | None = None,
        enable_cpu_mem_arena: bool = True,
        verbose: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "pipeline_config.json").read_text())
        self.use_io_binding = use_io_binding
        self.use_cupy = use_cupy
        self.low_memory = low_memory
        self.verbose = verbose
        self._session_kwargs = dict(
            intra_op_num_threads=intra_op_num_threads,
            enable_cpu_mem_arena=enable_cpu_mem_arena,
        )
        self._modules: dict[str, OnnxModule] = {}

        self.scheduler = FlowMatchEulerScheduler(self.config["scheduler"])
        self.vae_scale_factor = int(self.config["vae_scale_factor"])
        self.latent_channels = int(self.config["latent_channels"])
        self.in_channels = int(self.config["in_channels"])
        self.axes_dims_rope = list(self.config["axes_dims_rope"])
        self.rope_theta = int(self.config["rope_theta"])
        self.max_sequence_length = int(self.config["max_sequence_length"])
        self.text_seq_len = int(self.config.get("text_seq_len", self.max_sequence_length))
        self.default_sample_size = int(self.config.get("default_sample_size", 128))
        self.image_ids_scale = int(self.config.get("image_ids_scale", 10))
        self.max_condition_area = int(self.config.get("max_condition_area", 1024 * 1024))
        self.pad_token_id = int(self.config.get("pad_token_id", 151643))
        self.chat_template = self.config["chat_template"]

        bn = np.load(self.model_dir / "vae_bn.npz")
        eps = float(self.config.get("batch_norm_eps", 1e-4))
        self.latent_mean = bn["running_mean"].astype(np.float32).reshape(1, -1, 1, 1)
        self.latent_std = np.sqrt(bn["running_var"].astype(np.float32) + eps).reshape(1, -1, 1, 1)
        self._tokenizer = None

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, **kwargs):
        return cls(model_dir, **kwargs)

    def _log(self, message: str) -> None:
        if self.verbose:
            image_edit_logger.info(message)

    def _module(self, name: str) -> OnnxModule:
        module = self._modules.get(name)
        if module is None:
            path = self.model_dir / name
            start = time.perf_counter()
            module = OnnxModule(path, use_io_binding=self.use_io_binding, use_cupy=self.use_cupy, **self._session_kwargs)
            self._modules[name] = module
            self._log(f"[onnx] loaded {name} in {time.perf_counter() - start:.1f}s ({module.provider})")
        return module

    def unload(self, *names: str) -> None:
        for name in names or tuple(self._modules):
            module = self._modules.pop(name, None)
            if module is None:
                continue
            module.unload()
            self._log(f"[onnx] released {name}")

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer" / "tokenizer.json"))
        return self._tokenizer

    def encode_prompt(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        text = self.chat_template.format(prompt=prompt or "")
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        length = self.max_sequence_length
        ids = ids[:length]
        attention_mask = np.zeros((1, length), dtype=np.int64)
        attention_mask[0, : len(ids)] = 1
        input_ids = np.full((1, length), self.pad_token_id, dtype=np.int64)
        input_ids[0, : len(ids)] = ids

        module = self._module("text_encoder")
        outputs = module.run({"input_ids": input_ids, "attention_mask": attention_mask})
        prompt_embeds = asnumpy(outputs["prompt_embeds"])
        text_ids = cartesian_ids([0], 1, 1, length)
        return prompt_embeds, text_ids

    def encode_condition_images(self, images: Sequence[Image.Image]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        module = self._module("vae_encoder")
        packed, all_ids, first_size = [], [], None
        for index, image in enumerate(images):
            check_image_input(image)
            pixel_values, width, height = preprocess_condition_image(image, self.max_condition_area, self.vae_scale_factor * 2)
            if first_size is None:
                first_size = (width, height)
            latent_h = height // self.vae_scale_factor
            latent_w = width // self.vae_scale_factor
            outputs = module.run({"pixel_values": pixel_values}, output_shapes={"latent": (1, self.latent_channels, latent_h, latent_w)})
            latent = asnumpy(outputs["latent"]).astype(np.float32)
            latent = patchify_latents(latent)
            latent = (latent - self.latent_mean) / self.latent_std
            packed.append(pack_latents(latent)[0])
            all_ids.append(cartesian_ids([self.image_ids_scale * (index + 1)], latent.shape[2], latent.shape[3]))
        image_latents = np.concatenate(packed, axis=0)[None]
        image_ids = np.concatenate(all_ids, axis=0)
        return image_latents, image_ids, first_size

    def prepare_latents(self, height: int, width: int, seed: int | None, latents: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int, int]:
        num_latent_channels = self.in_channels // 4
        latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (1, num_latent_channels * 4, latent_height // 2, latent_width // 2)
        if latents is None:
            rng = np.random.default_rng(seed)
            latents = rng.standard_normal(shape, dtype=np.float32)
        else:
            latents = latents.astype(np.float32).reshape(shape)
        latent_ids = cartesian_ids([0], shape[2], shape[3])
        return pack_latents(latents), latent_ids, latent_height, latent_width

    def unpack_and_denormalize(self, latents: np.ndarray, latent_ids: np.ndarray, latent_height: int, latent_width: int) -> np.ndarray:
        latents = unpack_latents_with_ids(latents, latent_ids, latent_height // 2, latent_width // 2)
        latents = latents * self.latent_std + self.latent_mean
        return unpatchify_latents(latents)

    def decode_latents(self, latents: np.ndarray) -> np.ndarray:
        module = self._module("vae_decoder")
        outputs = module.run(
            {"latent": latents},
            output_shapes={
                "sample": (
                    1,
                    3,
                    latents.shape[2] * self.vae_scale_factor,
                    latents.shape[3] * self.vae_scale_factor,
                )
            },
        )
        return asnumpy(outputs["sample"]).astype(np.float32)

    def __call__(
        self,
        prompt: str,
        image: Image.Image | Sequence[Image.Image] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 4,
        sigmas: Sequence[float] | None = None,
        seed: int | None = 42,
        latents: np.ndarray | None = None,
        prompt_embeds: np.ndarray | None = None,
        output_type: str = "pil",
        callback_on_step_end=None,
    ) -> PipelineOutput:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        if prompt_embeds is None:
            prompt_embeds, text_ids = self.encode_prompt(prompt)
        else:
            prompt_embeds = np.asarray(prompt_embeds)
            text_ids = cartesian_ids([0], 1, 1, prompt_embeds.shape[1])
        if prompt_embeds.shape[1] != self.text_seq_len:
            raise ValueError(f"the transformer graph was exported for text_seq_len={self.text_seq_len}, got {prompt_embeds.shape[1]}")
        timings["text_encoder"] = time.perf_counter() - start
        if self.low_memory:
            self.unload("text_encoder")

        images = [image] if isinstance(image, Image.Image) else (list(image) if image else [])
        image_latents = image_ids = None
        if images:
            mark = time.perf_counter()
            image_latents, image_ids, first_size = self.encode_condition_images(images)
            width = width or first_size[0]
            height = height or first_size[1]
            timings["vae_encoder"] = time.perf_counter() - mark
            if self.low_memory:
                self.unload("vae_encoder")

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        latents, latent_ids, latent_height, latent_width = self.prepare_latents(height, width, seed, latents)
        num_latent_tokens = latents.shape[1]
        img_ids = latent_ids if image_ids is None else np.concatenate([latent_ids, image_ids], axis=0)
        rope_cos, rope_sin = rope_embeddings(np.concatenate([text_ids, img_ids], axis=0), self.rope_theta, self.axes_dims_rope)
        if sigmas is None:
            sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        mu = compute_empirical_mu(num_latent_tokens, num_inference_steps)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, mu=mu)
        step_sigmas = self.scheduler.sigmas
        mark = time.perf_counter()
        transformer = self._module("transformer")
        xp = transformer.xp
        hidden_dtype = transformer.input_dtypes["hidden_states"]
        total_tokens = num_latent_tokens + (0 if image_latents is None else image_latents.shape[1])

        transformer.set_static_input("encoder_hidden_states", prompt_embeds)
        transformer.set_static_input("rope_cos", rope_cos)
        transformer.set_static_input("rope_sin", rope_sin)

        model_input = xp.empty((1, total_tokens, self.in_channels), dtype=hidden_dtype)
        if image_latents is not None:
            model_input[:, num_latent_tokens:] = xp.asarray(image_latents.astype(hidden_dtype))
        sample = xp.asarray(latents)
        timestep = xp.empty((1,), dtype=transformer.input_dtypes["timestep"])
        out_shapes = {"noise_pred": (1, total_tokens, self.in_channels)}
        for index in range(num_inference_steps):
            model_input[:, :num_latent_tokens] = sample.astype(hidden_dtype)
            timestep[0] = step_sigmas[index]
            outputs = transformer.run({"hidden_states": model_input, "timestep": timestep}, output_shapes=out_shapes)
            noise_pred = outputs["noise_pred"][:, :num_latent_tokens]
            dt = float(step_sigmas[index + 1]) - float(step_sigmas[index])
            sample = sample + dt * noise_pred.astype(np.float32)
            if callback_on_step_end is not None:
                callback_on_step_end(self, index, float(self.scheduler.timesteps[index]), sample)
            self._log(f"[onnx] step {index + 1}/{num_inference_steps}")
        latents = asnumpy(sample).astype(np.float32)
        timings["transformer"] = time.perf_counter() - mark
        transformer.clear_static_inputs()
        if self.low_memory:
            self.unload("transformer")

        latents = self.unpack_and_denormalize(latents, latent_ids, latent_height, latent_width)
        if output_type == "latent":
            timings["total"] = time.perf_counter() - start
            return PipelineOutput(images=latents, latents=latents, timings=timings)
        mark = time.perf_counter()
        sample = self.decode_latents(latents)
        timings["vae_decoder"] = time.perf_counter() - mark
        if self.low_memory:
            self.unload("vae_decoder")
        timings["total"] = time.perf_counter() - start
        return PipelineOutput(images=postprocess_image(sample, output_type), latents=latents, timings=timings)
