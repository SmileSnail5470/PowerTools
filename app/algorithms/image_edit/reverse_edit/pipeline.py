import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from app.algorithms.image_edit.reverse_edit.ddim import DDIMScheduler
from app.algorithms.image_edit.reverse_edit.renoise import inversion_step
from app.algorithms.image_edit.reverse_edit.utils import (
    COMPONENT_CONFIG_NAME,
    ONNX_WEIGHTS_NAME,
    CLIPTokenizer,
    denormalize,
    maybe_load_json,
    numpy_to_pil,
    preprocess_image,
    randn,
    DiagonalGaussianDistribution
)
from app.algorithms import general_provider, general_session, general_inference_session, ORTEnvironment, evict_session_cache
ORTEnvironment.initialize()


image_edit_logger = logging.getLogger("ImageEdit")
_COMPONENTS = ("text_encoder", "unet", "vae_encoder", "vae_decoder")


@dataclass
class RunConfig:
    seed: int = 7865
    num_inference_steps: int = 4
    num_inversion_steps: int = 4
    guidance_scale: float = 0.0
    num_renoise_steps: int = 9
    max_num_renoise_steps_first_step: int = 5
    inversion_max_step: float = 1.0
    
    average_latent_estimations: bool = True
    average_first_step_range: tuple = (0, 5)
    average_step_range: tuple = (8, 10)

    noise_regularization_lambda_ac: float = 20.0
    noise_regularization_lambda_kl: float = 0.065
    noise_regularization_num_reg_steps: int = 4
    noise_regularization_num_ac_rolls: int = 5
    perform_noise_correction: bool = False


class ORTPipeline:
    def __init__(
        self,
        model_dir: str | Path,
        use_io_binding: bool | None = None,
        use_cupy: bool = True,
        low_memory: bool | None = None,
        intra_op_num_threads: int | None = None,
        skip_unused_optimal_pass: bool = True,
        identity_add_noise: bool = True,
        vae_sample_mode: str = "sample",
        verbose: bool = True,
    ):
        self.model_dir = Path(model_dir).resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"model dir not found: {self.model_dir}")
        self.use_io_binding = bool(use_io_binding) if use_io_binding is not None else None
        self.use_cupy = bool(use_cupy) if use_cupy is not None else None
        self.low_memory = bool(low_memory) if low_memory is not None else None
        self.intra_op_num_threads = intra_op_num_threads
        self.skip_unused_optimal_pass = bool(skip_unused_optimal_pass)
        self.identity_add_noise = bool(identity_add_noise)
        if vae_sample_mode not in ("sample", "mode"):
            raise ValueError("vae_sample_mode can only be 'sample' or 'mode'")
        self.vae_sample_mode = vae_sample_mode
        self.verbose = bool(verbose)
        self._paths = {name: self.model_dir / name / ONNX_WEIGHTS_NAME for name in _COMPONENTS}
        missing = [str(p) for p in self._paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError("no onnx model: " + ", ".join(missing))
        self._sessions: dict[str, object] = {}
        self._pinned: set[str] = set()
        vae_cfg = maybe_load_json(self.model_dir / "vae_encoder" / COMPONENT_CONFIG_NAME) or maybe_load_json(self.model_dir / "vae_decoder" / COMPONENT_CONFIG_NAME)
        unet_cfg = maybe_load_json(self.model_dir / "unet" / COMPONENT_CONFIG_NAME)
        self.vae_scaling_factor = float(vae_cfg.get("scaling_factor", 0.18215))
        block_out_channels = vae_cfg.get("block_out_channels") or [128, 256, 512, 512]
        self.vae_scale_factor = 2 ** (len(block_out_channels) - 1)
        sample_size = vae_cfg.get("sample_size")
        if not sample_size and unet_cfg.get("sample_size"):
            sample_size = int(unet_cfg["sample_size"]) * self.vae_scale_factor
        self.sample_size = int(sample_size or 512)
        self.tokenizer = CLIPTokenizer(self.model_dir / "tokenizer")
        self.scheduler = DDIMScheduler.from_pretrained(self.model_dir, subfolder="scheduler")
        self._timestep_rank: int | None = None
        self.cfg: RunConfig | None = None
        self.z_0: np.ndarray | None = None
        self.noise: np.ndarray | None = None
        self.guidance_scale: float = 0.0
        self.do_classifier_free_guidance: bool = False

    def _log(self, msg: str):
        if self.verbose:
            image_edit_logger.info(msg)

    def _session(self, name: str):
        sess = self._sessions.get(name)
        if sess is not None:
            return sess
        self._log(f"[ort] loading {name} ...")
        kwargs = {}
        if self.intra_op_num_threads:
            kwargs["intra_op_num_threads"] = self.intra_op_num_threads
        options = general_session(**kwargs)
        providers, provider_options = general_provider()
        sess = general_inference_session(
            model_path=str(self.path),
            sess_options=options,
            providers=providers,
            provider_options=provider_options,
        )
        self._sessions[name] = sess
        return sess

    def _release(self, name: str):
        if not self.low_memory or name in self._pinned:
            return
        sess = self._sessions.pop(name, None)
        if sess is None:
            return
        sess.clear_static_inputs()
        evict_session_cache([str(self._paths[name])])
        del sess
        gc.collect()

    def release_all(self):
        for name in list(self._sessions):
            self._sessions.pop(name).clear_static_inputs()
        evict_session_cache([str(p) for p in self._paths.values()])
        gc.collect()

    @property
    def unet_dtype(self) -> np.dtype:
        return self._session("unet").expected_dtype("sample") or np.dtype(np.float32)

    def encode_prompt(self, prompt: str | list[str], negative_prompt: str | list[str] | None, do_cfg: bool) -> np.ndarray:
        sess = self._session("text_encoder")
        ids_dtype = sess.expected_dtype("input_ids") or np.dtype(np.int64)

        def _encode(text):
            ids = self.tokenizer(text, dtype=ids_dtype)
            out = sess.run_dict({"input_ids": ids}, prefer_cupy=self.use_cupy, use_io_binding=self.use_io_binding)
            return np.asarray(out["last_hidden_state"], dtype=np.float32)

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        embeds = _encode(prompts)
        if do_cfg:
            if negative_prompt is None:
                negatives = [""] * len(prompts)
            elif isinstance(negative_prompt, str):
                negatives = [negative_prompt] * len(prompts)
            else:
                negatives = list(negative_prompt)
            embeds = np.concatenate([_encode(negatives), embeds], axis=0)
        self._release("text_encoder")
        return np.ascontiguousarray(embeds)

    def vae_encode(self, image: np.ndarray, generator: np.random.Generator | None = None) -> np.ndarray:
        sess = self._session("vae_encoder")
        out = sess.run_dict({"sample": image}, prefer_cupy=self.use_cupy, use_io_binding=self.use_io_binding)
        params = np.asarray(out["latent_parameters"], dtype=np.float32)
        self._release("vae_encoder")
        dist = DiagonalGaussianDistribution(params)
        latents = dist.mode() if self.vae_sample_mode == "mode" else dist.sample(generator)
        return (self.vae_scaling_factor * latents).astype(np.float32)

    def vae_decode(self, latents: np.ndarray) -> np.ndarray:
        sess = self._session("vae_decoder")
        out = sess.run_dict({"latent_sample": (latents / self.vae_scaling_factor).astype(np.float32)}, prefer_cupy=self.use_cupy, use_io_binding=self.use_io_binding)
        image = np.asarray(out["sample"], dtype=np.float32)
        self._release("vae_decoder")
        return image

    def forward_noised_latent(self, timestep: int) -> np.ndarray:
        if self.identity_add_noise:
            return self.z_0
        return self.scheduler.add_noise(self.z_0, self.noise, np.asarray([int(timestep)]))

    def unet_pass(self, latents: np.ndarray, timestep: int, prompt_embeds: np.ndarray) -> np.ndarray:
        sess = self._session("unet")
        model_input = np.concatenate([latents] * 2, axis=0) if self.do_classifier_free_guidance else latents
        model_input = self.scheduler.scale_model_input(model_input, timestep)
        feed = {
            "sample": np.ascontiguousarray(model_input, dtype=np.float32),
            "timestep": self._timestep_input(timestep, model_input.shape[0]),
            "encoder_hidden_states": prompt_embeds,
        }
        out = sess.run_dict(feed, output_shapes={"out_sample": model_input.shape}, prefer_cupy=self.use_cupy, use_io_binding=self.use_io_binding)
        return np.asarray(out["out_sample"], dtype=np.float32)

    def _timestep_input(self, timestep: int, batch: int) -> np.ndarray:
        if self._timestep_rank is None:
            shape = self._session("unet").input_shapes.get("timestep") or []
            self._timestep_rank = len(shape)
        if self._timestep_rank == 0:
            return np.asarray(float(timestep), dtype=np.float32)
        return np.full((batch,), float(timestep), dtype=np.float32)

    def invert(
        self,
        image,
        prompt: str,
        cfg: RunConfig,
        negative_prompt: str | None = None,
        return_all_latents: bool = False,
    ):
        self.cfg = cfg
        generator = np.random.default_rng(cfg.seed)
        self.guidance_scale = float(cfg.guidance_scale)
        self.do_classifier_free_guidance = self.guidance_scale > 1.0

        prompt_embeds = self.encode_prompt(prompt, negative_prompt, self.do_classifier_free_guidance)

        image = preprocess_image(image, self.sample_size, self.sample_size)
        latents = self.vae_encode(image, generator)

        self.scheduler.set_timesteps(cfg.num_inversion_steps)
        timesteps, _ = self.scheduler.get_timesteps(cfg.num_inversion_steps, cfg.inversion_max_step)

        self.z_0 = latents.copy()
        self.noise = randn(self.z_0.shape, generator)

        all_latents = [latents.copy()] if return_all_latents else None
        self._pinned.add("unet")
        try:
            total = len(timesteps)
            t0 = time.perf_counter()
            for i, t in enumerate(reversed(timesteps.tolist())):
                latents = inversion_step(
                    self,
                    latents,
                    t,
                    prompt_embeds,
                    num_renoise_steps=cfg.num_renoise_steps,
                    generator=generator,
                )
                if all_latents is not None:
                    all_latents.append(latents.copy())
                if self.verbose:
                    elapsed = time.perf_counter() - t0
                    self._log(f"inversion {i + 1}/{total}  t={t:4d}  {elapsed:.1f}s ({elapsed / (i + 1):.2f}s/step)")
        finally:
            self._pinned.discard("unet")
        return latents, all_latents

    def reconstruct(
        self,
        latents: np.ndarray,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float = 1.0,
        negative_prompt: str | None = None,
        strength: float = 1.0,
        eta: float = 0.0,
        output_type: str = "pil",
        generator: np.random.Generator | None = None,
    ):
        self.guidance_scale = float(guidance_scale)
        self.do_classifier_free_guidance = self.guidance_scale > 1.0

        prompt_embeds = self.encode_prompt(prompt, negative_prompt, self.do_classifier_free_guidance)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps, _ = self.scheduler.get_timesteps(num_inference_steps, strength)

        latents = latents.astype(np.float32, copy=True)
        self._pinned.add("unet")
        try:
            total = len(timesteps)
            t0 = time.perf_counter()
            for i, t in enumerate(timesteps.tolist()):
                noise_pred = self.unet_pass(latents, t, prompt_embeds)
                if self.do_classifier_free_guidance:
                    uncond, text = np.split(noise_pred, 2, axis=0)
                    noise_pred = uncond + self.guidance_scale * (text - uncond)
                latents = self.scheduler.step(noise_pred, t, latents, eta=eta, generator=generator)
                if self.verbose:
                    elapsed = time.perf_counter() - t0
                    self._log(f"denoise {i + 1}/{total}  t={t:4d}  {elapsed:.1f}s ({elapsed / (i + 1):.2f}s/step)")
        finally:
            self._pinned.discard("unet")
            self._release("unet")
        image = self.vae_decode(latents)
        image = denormalize(image)
        if output_type == "np":
            return image
        return numpy_to_pil(image)

    def __call__(
        self,
        image,
        prompt: str,
        cfg: RunConfig,
        do_reconstruction: bool = True,
        edit_prompt: str | None = None,
        reconstruction_guidance_scale: float = 1.0,
        return_all_latents: bool = False,
    ):
        inv_latent, all_latents = self.invert(image, prompt, cfg, return_all_latents=return_all_latents)
        rec_image = None
        if do_reconstruction:
            rec_image = self.reconstruct(
                inv_latent,
                edit_prompt if edit_prompt is not None else prompt,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=reconstruction_guidance_scale,
            )[0]
        return rec_image, inv_latent, all_latents
