import os
from pathlib import Path
import numpy as np
from app.algorithms.image_edit.reverse_edit.utils import config_dict


def betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta: float = 0.999) -> np.ndarray:
    def alpha_bar(t):
        return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.asarray(betas, dtype=np.float64)


def rescale_zero_terminal_snr(betas: np.ndarray) -> np.ndarray:
    alphas_cumprod = np.cumprod(1.0 - betas)
    alphas_bar_sqrt = np.sqrt(alphas_cumprod)
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].copy()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].copy()
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = np.concatenate([alphas_bar[0:1], alphas])
    return 1.0 - alphas


class DDIMScheduler:
    order = 1
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas=None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        rescale_betas_zero_snr: bool = False,
        **ignored,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.beta_schedule = beta_schedule
        self.clip_sample = bool(clip_sample)
        self.clip_sample_range = float(clip_sample_range)
        self.prediction_type = prediction_type
        self.thresholding = bool(thresholding)
        self.dynamic_thresholding_ratio = float(dynamic_thresholding_ratio)
        self.sample_max_value = float(sample_max_value)
        self.timestep_spacing = timestep_spacing
        self.steps_offset = int(steps_offset)

        if trained_betas is not None:
            betas = np.asarray(trained_betas, dtype=np.float64)
        elif beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, self.num_train_timesteps, dtype=np.float64)
        elif beta_schedule == "scaled_linear":
            betas = np.linspace(beta_start**0.5, beta_end**0.5, self.num_train_timesteps, dtype=np.float64) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            betas = betas_for_alpha_bar(self.num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        if rescale_betas_zero_snr:
            betas = rescale_zero_terminal_snr(betas)

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.final_alpha_cumprod = np.float64(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        self.init_noise_sigma = 1.0

        self.num_inference_steps: int | None = None
        self.timesteps = np.arange(0, self.num_train_timesteps)[::-1].copy()

    @classmethod
    def from_pretrained(cls, pretrained_path: str | os.PathLike, subfolder: str | None = None, **kwargs):
        path = Path(pretrained_path)
        if subfolder:
            path = path / subfolder
        config = config_dict["scheduler_config"]
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)
        config.update(kwargs)
        return cls(**config)

    def set_timesteps(self, num_inference_steps: int):
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(f"num_inference_steps ({num_inference_steps}) > num_train_timesteps ({self.num_train_timesteps})")
        self.num_inference_steps = int(num_inference_steps)
        if self.timestep_spacing == "linspace":
            timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps).round()[::-1].copy()
            timesteps = timesteps.astype(np.int64)
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_train_timesteps // self.num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += self.steps_offset
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / self.num_inference_steps
            timesteps = np.round(np.arange(self.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(f"{self.timestep_spacing} is not supported.")
        self.timesteps = timesteps
        return self.timesteps

    def get_timesteps(self, num_inference_steps: int, strength: float):
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        return self.timesteps[t_start * self.order :], num_inference_steps - t_start

    def scale_model_input(self, sample: np.ndarray, timestep=None) -> np.ndarray:
        return sample

    def _alphas(self, timestep: int):
        prev_timestep = int(timestep) - self.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[int(timestep)]
        alpha_prod_t_prev = (self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod)
        return prev_timestep, alpha_prod_t, alpha_prod_t_prev

    def _get_variance(self, timestep: int, prev_timestep: int) -> np.float64:
        alpha_prod_t = self.alphas_cumprod[int(timestep)]
        alpha_prod_t_prev = (self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        return (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    def _threshold_sample(self, sample: np.ndarray) -> np.ndarray:
        batch, channels = sample.shape[0], sample.shape[1]
        flat = sample.reshape(batch, -1).astype(np.float32, copy=False)
        abs_flat = np.abs(flat)
        s = np.quantile(abs_flat, self.dynamic_thresholding_ratio, axis=1, keepdims=True)
        s = np.clip(s, 1.0, self.sample_max_value)
        return (np.clip(flat, -s, s) / s).reshape(sample.shape)

    def _pred_original_and_epsilon(self, model_output, sample, alpha_prod_t, beta_prod_t):
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
            pred_epsilon = model_output
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
            pred_epsilon = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
        else:
            raise ValueError(f"unsupported prediction_type: {self.prediction_type}")
        if self.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.clip_sample:
            pred_original_sample = np.clip(pred_original_sample, -self.clip_sample_range, self.clip_sample_range)
        return pred_original_sample, pred_epsilon

    def step(
        self,
        model_output: np.ndarray,
        timestep: int,
        sample: np.ndarray,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        variance_noise: np.ndarray | None = None,
        generator: np.random.Generator | None = None,
    ) -> np.ndarray:
        if self.num_inference_steps is None:
            raise ValueError("call set_timesteps() first")

        prev_timestep, alpha_prod_t, alpha_prod_t_prev = self._alphas(timestep)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample, pred_epsilon = self._pred_original_and_epsilon(model_output, sample, alpha_prod_t, beta_prod_t)
        std_dev_t = eta * self._get_variance(timestep, prev_timestep) ** 0.5
        if use_clipped_model_output:
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * pred_epsilon
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        if eta > 0:
            if variance_noise is None:
                rng = generator if generator is not None else np.random.default_rng()
                variance_noise = rng.standard_normal(model_output.shape, dtype=np.float32)
            prev_sample = prev_sample + std_dev_t * variance_noise
        return prev_sample.astype(np.float32, copy=False)

    def inv_step(
        self,
        model_output: np.ndarray,
        timestep: int,
        sample: np.ndarray,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        variance_noise: np.ndarray | None = None,
        generator: np.random.Generator | None = None,
    ) -> np.ndarray:
        if self.num_inference_steps is None:
            raise ValueError("call set_timesteps() first")

        prev_timestep, alpha_prod_t, alpha_prod_t_prev = self._alphas(timestep)
        beta_prod_t = 1 - alpha_prod_t
        _, pred_epsilon = self._pred_original_and_epsilon(model_output, sample, alpha_prod_t, beta_prod_t)
        std_dev_t = eta * self._get_variance(timestep, prev_timestep) ** 0.5
        if use_clipped_model_output:
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * pred_epsilon
        sqrt_a_t = alpha_prod_t ** 0.5
        sqrt_a_prev = alpha_prod_t_prev ** 0.5
        prev_sample = (
            (sqrt_a_t * sample) / sqrt_a_prev
            + (sqrt_a_prev * beta_prod_t ** 0.5 * model_output) / sqrt_a_prev
            - (sqrt_a_t * pred_sample_direction) / sqrt_a_prev
        )
        if eta > 0:
            if variance_noise is None:
                rng = generator if generator is not None else np.random.default_rng()
                variance_noise = rng.standard_normal(model_output.shape, dtype=np.float32)
            prev_sample = prev_sample + std_dev_t * variance_noise

        return prev_sample.astype(np.float32, copy=False)

    def add_noise(self, original_samples: np.ndarray, noise: np.ndarray, timesteps) -> np.ndarray:
        timesteps = np.atleast_1d(np.asarray(timesteps, dtype=np.int64))
        alphas = self.alphas_cumprod[timesteps]
        sqrt_alpha_prod = (alphas ** 0.5).reshape(-1, *([1] * (original_samples.ndim - 1)))
        sqrt_one_minus = ((1 - alphas) ** 0.5).reshape(-1, *([1] * (original_samples.ndim - 1)))
        noisy = sqrt_alpha_prod * original_samples + sqrt_one_minus * noise
        return noisy.astype(np.float32, copy=False)
