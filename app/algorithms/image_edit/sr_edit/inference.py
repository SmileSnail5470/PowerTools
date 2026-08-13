import logging
import math
import time
from pathlib import Path
import cv2
import numpy as np
from app.algorithms.image_edit.sr_edit.backend import OnnxModule, asnumpy
from app.algorithms.image_edit.sr_edit.scheduling import Scheduler
from app.algorithms.image_edit.sr_edit.utils import DiagonalGaussianDistribution, Tokenizer


sr_edit_logger = logging.getLogger("ImageEdit")

_COMPONENTS = ("text_encoder", "vae_encoder", "unet", "vae_decoder")
NOISE = "noise"
NOISE_WEIGHT_NAMES = ("noise_model.encmodel", "noise_model.onnx")
_POSITIVE_PROMPT = (
    "Cinematic, high-contrast, photo-realistic, 8k, ultra HD, "
    "meticulous detailing, hyper sharpness, perfect without deformations"
)
_NEGATIVE_PROMPT = (
    "Low quality, blurring, jpeg artifacts, deformed, over-smooth, cartoon, noisy,"
    "painting, drawing, sketch, oil painting"
)


def default_configs() -> dict:
    return {
        "seed": 12345,
        "basesr": {
            "sf": 4,
            "chopping": {
                "pch_size": 128,
                "extra_bs": 1,
                "weight_type": "Gaussian",
            },
        },
        "timesteps": [200],
        "cfg_scale": 1.0,
        "start_timesteps": 200,
    }


class ImageSpliterNp:
    def __init__(self, im, pch_size, stride, sf=1, extra_bs=1, weight_type="Gaussian"):
        assert weight_type in ["Gaussian", "ones"]
        assert stride <= pch_size
        self.weight_type = weight_type
        self.stride = stride
        self.pch_size = pch_size
        self.sf = sf
        self.extra_bs = max(1, int(extra_bs))
        bs, chn, height, width = im.shape
        self.true_bs = bs
        self.height_starts_list = self._extract_starts(height)
        self.width_starts_list = self._extract_starts(width)
        self.starts_list = []
        for ii in self.height_starts_list:
            for jj in self.width_starts_list:
                self.starts_list.append([ii, jj])
        self.length = len(self.starts_list)
        if self.length == 1:
            self.weight_type = "ones"
        self.count_pchs = 0
        self.im_ori = im
        self.im_res = np.zeros([bs, chn, height * sf, width * sf], dtype=np.float32)
        self.pixel_count = np.zeros([1, 1, height * sf, width * sf], dtype=np.float32)
        self._weight_cache = {}

    def _extract_starts(self, length):
        if length <= self.pch_size:
            return [0]
        starts = list(range(0, length, self.stride))
        for ii in range(len(starts)):
            if starts[ii] + self.pch_size > length:
                starts[ii] = length - self.pch_size
        seen = set()
        unique = []
        for s in starts:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    def __len__(self):
        return self.length

    @property
    def num_groups(self) -> int:
        return math.ceil(self.length / self.extra_bs)

    def __iter__(self):
        self.count_pchs = 0
        return self

    def __next__(self):
        if self.count_pchs >= self.length:
            raise StopIteration()
        index_infos = []
        patches = []
        current_starts_list = self.starts_list[self.count_pchs : self.count_pchs + self.extra_bs]
        for h_start, w_start in current_starts_list:
            h_end = h_start + self.pch_size
            w_end = w_start + self.pch_size
            patches.append(self.im_ori[:, :, h_start:h_end, w_start:w_end])
            index_infos.append([h_start * self.sf, h_end * self.sf, w_start * self.sf, w_end * self.sf])
        pch = patches[0] if len(patches) == 1 else np.concatenate(patches, axis=0)
        self.count_pchs += len(current_starts_list)
        return pch, index_infos

    def update(self, pch_res, index_infos):
        assert pch_res.shape[0] % self.true_bs == 0
        pch_list = np.split(pch_res, pch_res.shape[0] // self.true_bs, axis=0)
        assert len(pch_list) == len(index_infos)
        for ii, (h_start, h_end, w_start, w_end) in enumerate(index_infos):
            current_pch = np.asarray(pch_list[ii], dtype=np.float32)
            weight = self._get_weight(current_pch.shape[-2], current_pch.shape[-1])
            self.im_res[:, :, h_start:h_end, w_start:w_end] += current_pch * weight
            self.pixel_count[:, :, h_start:h_end, w_start:w_end] += weight

    @staticmethod
    def _generate_kernel_1d(ksize):
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        if ksize % 2 == 0:
            kernel = cv2.getGaussianKernel(ksize=ksize + 1, sigma=sigma, ktype=cv2.CV_32F)
            kernel = kernel[1:]
        else:
            kernel = cv2.getGaussianKernel(ksize=ksize, sigma=sigma, ktype=cv2.CV_32F)
        return kernel

    def _get_weight(self, height, width):
        cache_key = (height, width)
        weight = self._weight_cache.get(cache_key)
        if weight is not None:
            return weight
        if self.weight_type == "ones":
            weight = np.ones((1, 1, height, width), dtype=np.float32)
        elif self.weight_type == "Gaussian":
            kernel_h = self._generate_kernel_1d(height).reshape(-1, 1)
            kernel_w = self._generate_kernel_1d(width).reshape(1, -1)
            weight = (kernel_h @ kernel_w).reshape(1, 1, height, width).astype(np.float32)
        else:
            raise ValueError(f"Unsupported weight type: {self.weight_type}")
        self._weight_cache[cache_key] = weight
        return weight

    def gather(self):
        assert np.all(self.pixel_count != 0)
        self.im_res /= self.pixel_count
        return self.im_res


def retrieve_timesteps_invsr(scheduler, timesteps):
    num_inference_steps = len(timesteps)
    timesteps_arr = np.array(timesteps, dtype=np.float32) - 1
    scheduler.timesteps = timesteps_arr
    sigmas_all = ((1 - scheduler.alphas_cumprod) / scheduler.alphas_cumprod) ** 0.5
    timestep_indices = timesteps_arr.astype(np.int64)
    sigmas = sigmas_all[timestep_indices]
    if hasattr(scheduler.config, "final_sigmas_type"):
        final_sigmas_type = scheduler.config.final_sigmas_type
    else:
        final_sigmas_type = "zero"
    if final_sigmas_type == "sigma_min":
        sigma_last = float(((1 - scheduler.alphas_cumprod[0]) / scheduler.alphas_cumprod[0]) ** 0.5)
    elif final_sigmas_type == "zero":
        sigma_last = 0.0
    else:
        raise ValueError(f"`final_sigmas_type` must be 'zero' or 'sigma_min', got {final_sigmas_type}")
    sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
    scheduler.sigmas = sigmas
    scheduler._step_index = None
    scheduler._begin_index = None
    return scheduler.timesteps, num_inference_steps


def bicubic_resize(image, target_h, target_w):
    n, c, h, w = image.shape
    if (h, w) == (target_h, target_w):
        return image
    result = np.empty((n, c, target_h, target_w), dtype=np.float32)
    for i in range(n):
        img_hwc = np.ascontiguousarray(image[i].transpose(1, 2, 0))
        resized = cv2.resize(img_hwc, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]
        result[i] = resized.transpose(2, 0, 1)
    return result


class ImageSRInference:
    def __init__(
        self,
        model_dir,
        noise_path=None,
        low_memory: bool = True,
        use_io_binding: bool = True,
        use_cupy: bool = True,
        intra_op_num_threads: int | None = None,
        patch_size: int | None = None,
        patch_batch_size: int = 1,
        timesteps=None,
        cfg_scale: float | None = None,
        seed: int = 12345,
        verbose: bool = True,
    ):
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"model dir not found: {self.model_dir}")
        self.configs = default_configs()
        self.configs["seed"] = int(seed)
        if patch_size:
            self.configs["basesr"]["chopping"]["pch_size"] = int(patch_size)
        if timesteps:
            self.configs["timesteps"] = list(timesteps)
        if cfg_scale is not None:
            self.configs["cfg_scale"] = float(cfg_scale)
        self.low_memory = bool(low_memory)
        self.verbose = bool(verbose)
        self._rng = np.random.default_rng(seed)
        self._module_kwargs = dict(
            use_io_binding=use_io_binding,
            use_cupy=use_cupy,
            verbose=verbose,
            intra_op_num_threads=intra_op_num_threads,
        )
        self._paths = {name: self.model_dir / name for name in _COMPONENTS}
        missing = [str(path) for path in self._paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("missing onnx component: " + ", ".join(missing))
        noise_path = self._resolve_noise(noise_path)
        if noise_path is not None:
            self._paths[NOISE] = noise_path
        self._has_noise = noise_path is not None
        if not self._has_noise:
            sr_edit_logger.warning("[sr_edit] noise predictor not found, fallback to random noise initialization")
        self._modules: dict[str, OnnxModule] = {}
        self._pinned: set[str] = set()
        self._graph_checked = False
        self._patch_batch = int(patch_batch_size)
        self.vae_scale_factor = 8
        self.vae_scaling_factor = 0.18215
        self.latent_channels = 4
        self.tokenizer = Tokenizer(self.model_dir / "tokenizer")
        self.scheduler = Scheduler.from_pretrained()
        self._prompt_embeds = None
        self._negative_embeds = None
        self._embeds_cache: dict[int, np.ndarray] = {}
        self.last_timings: dict[str, float] = {}

    def _resolve_noise(self, noise_path) -> Path | None:
        if noise_path is not None:
            path = Path(noise_path)
            return path if path.exists() else None
        for name in NOISE_WEIGHT_NAMES:
            candidate = self.model_dir / name
            if candidate.is_file():
                return candidate
        candidate = self.model_dir / NOISE
        return candidate if candidate.is_dir() else None

    def _module(self, name: str) -> OnnxModule:
        module = self._modules.get(name)
        if module is None:
            path = self._paths.get(name)
            if path is None:
                raise KeyError(f"unknown onnx component: {name}")
            module = OnnxModule(path, **self._module_kwargs)
            self._modules[name] = module
        return module

    def _release(self, name: str) -> None:
        module = self._modules.get(name)
        if module is None:
            return
        module.clear_static_inputs()
        if not self.low_memory or name in self._pinned:
            return
        self._modules.pop(name, None)
        module.unload()

    def release(self) -> None:
        self._pinned.clear()
        for name in list(self._modules):
            self._modules.pop(name).unload()

    def _graph_limits(self) -> tuple[int, int]:
        chopping = self.configs["basesr"]["chopping"]
        if self._graph_checked:
            return int(chopping["pch_size"]), int(chopping["extra_bs"])
        module = self._module("unet")
        self._pinned.add("unet")
        sf = int(self.configs["basesr"]["sf"])
        pch_size = int(chopping["pch_size"])
        batch = self._patch_batch
        shape = module.declared_input_shape("sample")
        if len(shape) == 4:
            latent_h, latent_w = shape[2], shape[3]
            if isinstance(latent_h, int) and isinstance(latent_w, int) and latent_h > 0 and latent_w > 0:
                required = int(min(latent_h, latent_w)) * self.vae_scale_factor // sf
                if required != pch_size:
                    sr_edit_logger.info(f"[sr_edit] unet expects static latent {latent_h}x{latent_w}, patch size {pch_size} -> {required}")
                    pch_size = required
            if isinstance(shape[0], int) and shape[0] > 0:
                allowed = max(1, int(shape[0]) // (2 if self._do_cfg else 1))
                batch = min(batch, allowed)
        chopping["pch_size"] = pch_size
        chopping["extra_bs"] = batch
        self._graph_checked = True
        return pch_size, batch

    @property
    def _do_cfg(self) -> bool:
        return float(self.configs["cfg_scale"]) > 1.0

    def _latent_size(self, height: int, width: int, sf: int) -> tuple[int, int]:
        return height * sf // self.vae_scale_factor, width * sf // self.vae_scale_factor

    @staticmethod
    def _pick_output(module: OnnxModule, prefer: tuple[str, ...]) -> str:
        names = module.output_names
        for name in prefer:
            if name in names:
                return name
        return names[0]

    def _out_shapes(self, module: OnnxModule, templates: dict[str, tuple]) -> dict[str, tuple] | None:
        shapes = {}
        for name, template in templates.items():
            declared = module.declared_output_shape(name)
            dims = []
            for index, value in enumerate(template):
                static = declared[index] if index < len(declared) else None
                if isinstance(static, int) and static > 0:
                    dims.append(static)
                elif value is None:
                    return None
                else:
                    dims.append(int(value))
            shapes[name] = tuple(dims)
        return shapes

    @staticmethod
    def _timestep_array(module: OnnxModule, name: str, timestep: float, batch: int) -> np.ndarray:
        declared = module.declared_input_shape(name)
        if not declared:
            return np.asarray(float(timestep), dtype=np.float32)
        dim = declared[0]
        size = int(dim) if isinstance(dim, int) and dim > 0 else batch
        return np.full((size,), float(timestep), dtype=np.float32)

    def _sample_distribution(self, params: np.ndarray) -> np.ndarray:
        if params.shape[1] == 2 * self.latent_channels:
            return DiagonalGaussianDistribution(params).sample(self._rng)
        return params

    def _reset_scheduler(self) -> None:
        self.scheduler._step_index = None
        self.scheduler._begin_index = None

    def _encode_prompt(self) -> None:
        if self._prompt_embeds is not None:
            return
        module = self._module("text_encoder")

        def _encode(texts):
            tokens = self.tokenizer(
                texts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            outputs = module.run({"input_ids": tokens["input_ids"]})
            embeds = outputs.get("last_hidden_state")
            if embeds is None:
                embeds = next(iter(outputs.values()))
            return np.asarray(asnumpy(embeds), dtype=np.float32)

        self._prompt_embeds = _encode([_POSITIVE_PROMPT])
        if self._do_cfg:
            self._negative_embeds = _encode([_NEGATIVE_PROMPT])
        self._release("text_encoder")

    def _embeds_for(self, batch: int) -> np.ndarray:
        cached = self._embeds_cache.get(batch)
        if cached is not None:
            return cached
        embeds = self._prompt_embeds if batch == 1 else np.repeat(self._prompt_embeds, batch, axis=0)
        if self._negative_embeds is not None:
            negative = self._negative_embeds if batch == 1 else np.repeat(self._negative_embeds, batch, axis=0)
            embeds = np.concatenate([negative, embeds], axis=0)
        embeds = np.ascontiguousarray(embeds)
        self._embeds_cache[batch] = embeds
        return embeds

    def _predict_noise(self, module: OnnxModule, patch: np.ndarray, timestep: float, latent_size) -> np.ndarray:
        batch = patch.shape[0]
        name = self._pick_output(module, ("latent_parameters", "noise", "sample"))
        outputs = module.run(
            {
                "sample": patch * 2.0 - 1.0,
                "timestep": self._timestep_array(module, "timestep", timestep, batch),
            },
            output_shapes=self._out_shapes(module, {name: (batch, None, latent_size[0], latent_size[1])}),
        )
        params = np.asarray(asnumpy(outputs[name]), dtype=np.float32)
        return np.asarray(self._sample_distribution(params), dtype=np.float32)

    def _encode_latent(self, module: OnnxModule, image: np.ndarray) -> np.ndarray:
        batch = image.shape[0]
        latent_h = image.shape[-2] // self.vae_scale_factor
        latent_w = image.shape[-1] // self.vae_scale_factor
        name = self._pick_output(module, ("latent_sample", "latent_parameters", "latent"))
        outputs = module.run(
            {"sample": image},
            output_shapes=self._out_shapes(module, {name: (batch, None, latent_h, latent_w)}),
        )
        params = np.asarray(asnumpy(outputs[name]), dtype=np.float32)
        latent = self._sample_distribution(params)
        return np.asarray(latent * self.vae_scaling_factor, dtype=np.float32)

    def _decode_latent(self, module: OnnxModule, latents: np.ndarray) -> np.ndarray:
        batch, _, latent_h, latent_w = latents.shape
        name = self._pick_output(module, ("sample", "image"))
        template = (batch, 3, latent_h * self.vae_scale_factor, latent_w * self.vae_scale_factor)
        outputs = module.run(
            {"latent_sample": latents / self.vae_scaling_factor},
            output_shapes=self._out_shapes(module, {name: template}),
        )
        images = np.asarray(asnumpy(outputs[name]), dtype=np.float32)
        images = (images + 1.0) * 0.5
        return np.clip(images, 0.0, 1.0, out=images)

    def _denoise(self, module: OnnxModule, latents: np.ndarray, timesteps) -> np.ndarray:
        do_cfg = self._do_cfg
        cfg_scale = float(self.configs["cfg_scale"])
        name = self._pick_output(module, ("out_sample", "noise_pred"))
        for timestep in timesteps:
            model_input = np.concatenate([latents, latents], axis=0) if do_cfg else latents
            model_input = self.scheduler.scale_model_input(model_input, timestep)
            outputs = module.run(
                {
                    "sample": model_input,
                    "timestep": self._timestep_array(module, "timestep", timestep, model_input.shape[0]),
                },
                output_shapes=self._out_shapes(module, {name: model_input.shape}),
            )
            noise_pred = np.asarray(asnumpy(outputs[name]), dtype=np.float32)
            if do_cfg:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2, axis=0)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        return np.asarray(latents, dtype=np.float32)

    def _run_tiles(self, spliter: ImageSpliterNp, sf: int) -> np.ndarray:
        timings: dict[str, float] = {}
        timesteps, _ = retrieve_timesteps_invsr(self.scheduler, self.configs["timesteps"])
        latent_timestep = float(timesteps[0])
        sigma_init = float(self.scheduler.sigmas[0])
        groups = spliter.num_groups
        index_infos_list: list[list] = []
        noises: list[np.ndarray | None] = []
        latents: list[np.ndarray | None] = []
        if self._has_noise:
            mark = time.perf_counter()
            module = self._module(NOISE)
            for patch, _infos in spliter:
                latent_size = self._latent_size(patch.shape[-2], patch.shape[-1], sf)
                noises.append(self._predict_noise(module, patch, latent_timestep, latent_size))
            self._release(NOISE)
            timings["noise"] = time.perf_counter() - mark
        mark = time.perf_counter()
        module = self._module("vae_encoder")
        for patch, infos in spliter:
            index_infos_list.append(infos)
            image_up = bicubic_resize(patch, patch.shape[-2] * sf, patch.shape[-1] * sf)
            latents.append(self._encode_latent(module, image_up * 2.0 - 1.0))
        self._release("vae_encoder")
        timings["vae_encoder"] = time.perf_counter() - mark
        mark = time.perf_counter()
        module = self._module("unet")
        static_batch = None
        for index, latent in enumerate(latents):
            if noises:
                noise = noises[index]
                noises[index] = None
            else:
                noise = self._rng.standard_normal(latent.shape, dtype=np.float32)
            sample = latent + noise * sigma_init
            batch = sample.shape[0]
            if batch != static_batch:
                module.set_static_input("encoder_hidden_states", self._embeds_for(batch))
                static_batch = batch
            self._reset_scheduler()
            latents[index] = self._denoise(module, sample, timesteps)
        self._pinned.discard("unet")
        self._release("unet")
        timings["unet"] = time.perf_counter() - mark
        mark = time.perf_counter()
        module = self._module("vae_decoder")
        for index, latent in enumerate(latents):
            latents[index] = None
            spliter.update(self._decode_latent(module, latent), index_infos_list[index])
        self._release("vae_decoder")
        timings["vae_decoder"] = time.perf_counter() - mark
        timings["total"] = sum(timings.values())
        self.last_timings = timings
        if self.verbose:
            detail = " ".join(f"{key}={value:.2f}s" for key, value in timings.items())
            sr_edit_logger.info(f"[sr_edit] {len(spliter)} patches in {groups} groups: {detail}")
        return spliter.gather()

    def _pad_to_patch(self, im_cond: np.ndarray, pch_size: int) -> tuple[np.ndarray, int, int]:
        pad_top = pad_left = 0
        while min(im_cond.shape[-2:]) < pch_size:
            height, width = im_cond.shape[-2:]
            top = max(min((pch_size - height) // 2, height - 1), 0)
            bottom = max(min(pch_size - height - top, height - 1), 0)
            left = max(min((pch_size - width) // 2, width - 1), 0)
            right = max(min(pch_size - width - left, width - 1), 0)
            im_cond = np.pad(im_cond, ((0, 0), (0, 0), (top, bottom), (left, right)), mode="reflect")
            pad_top += top
            pad_left += left
        return im_cond, pad_top, pad_left

    @staticmethod
    def _pad_to_multiple(im_cond: np.ndarray, mod: int) -> np.ndarray:
        height, width = im_cond.shape[-2:]
        pad_h = math.ceil(height / mod) * mod - height
        pad_w = math.ceil(width / mod) * mod - width
        if pad_h == 0 and pad_w == 0:
            return im_cond
        return np.pad(im_cond, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

    def sample_func(self, im_cond: np.ndarray) -> np.ndarray:
        sf = int(self.configs["basesr"]["sf"])
        chopping = self.configs["basesr"]["chopping"]
        ori_h, ori_w = im_cond.shape[-2:]
        self._encode_prompt()
        pch_size, extra_bs = self._graph_limits()
        try:
            im_cond, pad_top, pad_left = self._pad_to_patch(im_cond, pch_size)
            im_cond = self._pad_to_multiple(im_cond, max(1, self.vae_scale_factor // sf * 8))
            spliter = ImageSpliterNp(
                im_cond,
                pch_size=pch_size,
                stride=max(1, int(pch_size * 0.5)),
                sf=sf,
                extra_bs=extra_bs,
                weight_type=chopping["weight_type"],
            )
            res_sr = self._run_tiles(spliter, sf)
        finally:
            self._pinned.discard("unet")
            self._release("unet")
        top = pad_top * sf
        left = pad_left * sf
        res_sr = res_sr[:, :, top : ori_h * sf + top, left : ori_w * sf + left]
        res_sr = res_sr.transpose(0, 2, 3, 1)
        return (np.clip(res_sr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def inference(self, img: np.ndarray):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = None
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            img = img[:, :, :3]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_nchw = (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
        result = self.sample_func(img_nchw)[0]
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        if alpha is not None:
            alpha = cv2.resize(alpha, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_CUBIC)
            result = np.dstack([result, alpha])
        return result
