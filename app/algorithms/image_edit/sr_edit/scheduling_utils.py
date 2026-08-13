import numpy as np


SCHEDULER_CONFIG_NAME = "scheduler_config.json"

SCHEDULER_CONFIG = {
  "beta_end": 0.012,
  "beta_schedule": "scaled_linear",
  "beta_start": 0.00085,
  "clip_sample": False,
  "final_sigmas_type": "zero",
  "interpolation_type": "linear",
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon",
  "rescale_betas_zero_snr": False,
  "sample_max_value": 1.0,
  "set_alpha_to_one": False,
  "sigma_max": None,
  "sigma_min": None,
  "skip_prk_steps": True,
  "steps_offset": 1,
  "timestep_spacing": "trailing",
  "timestep_type": "discrete",
  "trained_betas": None,
  "use_beta_sigmas": False,
  "use_exponential_sigmas": False,
  "use_karras_sigmas": False
}


class SchedulerOutput:
    def __init__(self, prev_sample: np.ndarray):
        self.prev_sample = prev_sample


class SchedulerMixin:
    def __init__(self, **kwargs):
        self.config = type("Config", (), kwargs)()

    @classmethod
    def from_pretrained(cls, **kwargs):
        config = SCHEDULER_CONFIG
        config.update(kwargs)
        return cls(**config)
