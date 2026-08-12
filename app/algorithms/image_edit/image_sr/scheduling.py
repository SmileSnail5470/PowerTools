import numpy as np
from app.algorithms.image_edit.image_sr.scheduling_utils import SchedulerMixin, SchedulerOutput


class Scheduler(SchedulerMixin):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon",
        timestep_spacing: str = "linspace",
        rescale_betas_zero_snr: bool = False,
        use_karras_sigmas: bool = False,
        use_exponential_sigmas: bool = False,
        use_beta_sigmas: bool = False,
        timestep_type: str = "discrete",
        interpolation_type: str = "linear",
        steps_offset: int = 0,
        final_sigmas_type: str = "zero",
        **kwargs,
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            timestep_spacing=timestep_spacing,
            use_karras_sigmas=use_karras_sigmas,
            use_exponential_sigmas=use_exponential_sigmas,
            use_beta_sigmas=use_beta_sigmas,
            timestep_type=timestep_type,
            interpolation_type=interpolation_type,
            steps_offset=steps_offset,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
            final_sigmas_type=final_sigmas_type,
            **kwargs,
        )
        if beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float64)
        elif beta_schedule == "scaled_linear":
            betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float64) ** 2
        else:
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0).astype(np.float32)
        if rescale_betas_zero_snr:
            self.alphas_cumprod[-1] = 2**-24
        sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        sigmas = sigmas[::-1].copy()
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        self.num_inference_steps = None
        self.timesteps = timesteps
        self.sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.is_scale_input_called = False
        self._step_index = None
        self._begin_index = None

    @property
    def init_noise_sigma(self) -> float:
        max_sigma = float(np.max(self.sigmas))
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            return max_sigma
        return (max_sigma**2 + 1) ** 0.5

    @property
    def step_index(self):
        return self._step_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def scale_model_input(self, sample: np.ndarray, timestep: float) -> np.ndarray:
        if self._step_index is None:
            self._init_step_index(timestep)
        sigma = self.sigmas[self._step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)
        self.is_scale_input_called = True
        return sample

    def set_timesteps(self, num_inference_steps: int, **kwargs):
        self.num_inference_steps = num_inference_steps
        num_train_timesteps = self.config.num_train_timesteps
        timestep_spacing = self.config.timestep_spacing
        steps_offset = self.config.steps_offset
        if timestep_spacing == "linspace":
            timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps, dtype=np.float32)[::-1].copy()
        elif timestep_spacing == "leading":
            step_ratio = num_train_timesteps // num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
            timesteps += steps_offset
        elif timestep_spacing == "trailing":
            step_ratio = num_train_timesteps / num_inference_steps
            timesteps = np.round(np.arange(num_train_timesteps, 0, -step_ratio)).astype(np.float32)
            timesteps -= 1
        else:
            raise ValueError(f"Unsupported timestep_spacing: {timestep_spacing}")
        sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        log_sigmas = np.log(sigmas)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas).astype(np.float32)
        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(sigmas, num_inference_steps)
            timesteps = np.array([self._sigma_to_t(s, log_sigmas) for s in sigmas]).astype(np.float32)
        self.sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.timesteps = timesteps
        self._step_index = None
        self._begin_index = None

    def _sigma_to_t(self, sigma, log_sigmas):
        log_sigma = np.log(max(sigma, 1e-10))
        dists = log_sigma - log_sigmas
        low_idx = np.where(dists >= 0)[0]
        if len(low_idx) == 0:
            return 0
        low_idx = low_idx[-1]
        high_idx = low_idx + 1
        if high_idx >= len(log_sigmas):
            return len(log_sigmas) - 1
        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)
        return float(low_idx + w)

    def _convert_to_karras(self, sigmas: np.ndarray, num_inference_steps: int) -> np.ndarray:
        sigma_min = sigmas[-1].item() if sigmas[-1] > 0 else 0.01
        sigma_max = sigmas[0].item()
        rho = 7.0
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas.astype(np.float32)

    def _init_step_index(self, timestep):
        index_candidates = np.where(self.timesteps == timestep)[0]
        if len(index_candidates) > 1:
            self._step_index = index_candidates[1]
        elif len(index_candidates) == 1:
            self._step_index = index_candidates[0]
        else:
            self._step_index = int(np.argmin(np.abs(self.timesteps - timestep)))

    def step(
            self,
            model_output: np.ndarray,
            timestep: float,
            sample: np.ndarray,
            s_churn: float = 0.0,
            s_tmin: float = 0.0,
            s_tmax: float = float("inf"),
            s_noise: float = 1.0,
            noise: "np.ndarray | None" = None,
            generator=None,
            return_dict: bool = True,
        ) -> "SchedulerOutput | tuple":
        if self._step_index is None:
            self._init_step_index(timestep)
        sample = sample.astype(np.float32)
        sigma = self.sigmas[self._step_index]
        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0
        if noise is None:
            noise_eps = np.random.randn(*model_output.shape).astype(np.float32)
        else:
            noise_eps = noise.astype(np.float32)
        eps = noise_eps * s_noise
        sigma_hat = sigma * (gamma + 1)
        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5
        sigma_next = self.sigmas[self._step_index + 1]
        prediction_type = self.config.prediction_type
        if prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
        elif prediction_type == "v_prediction":
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        elif prediction_type == "sample":
            pred_original_sample = model_output
        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")
        derivative = (sample - pred_original_sample) / sigma_hat
        dt = sigma_next - sigma_hat
        prev_sample = sample + derivative * dt
        prev_sample = prev_sample.astype(model_output.dtype)
        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    def add_noise(
            self,
            original_samples: np.ndarray,
            noise: np.ndarray,
            timesteps: np.ndarray,
        ) -> np.ndarray:
        timesteps = np.atleast_1d(timesteps).astype(np.float32)
        if self._begin_index is None:
            step_indices = [self._index_for_timestep(t) for t in timesteps]
        elif self._step_index is not None:
            step_indices = [self._step_index] * len(timesteps)
        else:
            step_indices = [self._begin_index] * len(timesteps)
        sigma = self.sigmas[step_indices].reshape(-1)
        while sigma.ndim < original_samples.ndim:
            sigma = sigma[..., np.newaxis]

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def _index_for_timestep(self, timestep):
        indices = np.where(self.timesteps == timestep)[0]
        if len(indices) > 1:
            return int(indices[1])
        elif len(indices) == 1:
            return int(indices[0])
        else:
            return int(np.argmin(np.abs(self.timesteps - timestep)))
