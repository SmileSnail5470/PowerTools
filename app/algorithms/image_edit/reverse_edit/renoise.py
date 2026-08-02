import numpy as np


EPSILON = 1e-6



def _patchify(x: np.ndarray, patch_size: int, num_channels: int) -> np.ndarray:
    b, c, h, w = x.shape
    if c != num_channels:
        raise ValueError(f"expected {num_channels} channels, got {c}")
    if h % patch_size or w % patch_size:
        raise ValueError(f"spatial size {(h, w)} not divisible by patch_size {patch_size}")
    return (
        x.reshape(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
        .transpose(0, 2, 4, 1, 3, 5)
        .reshape(-1, c, patch_size, patch_size)
    )


def _unpatchify(p: np.ndarray, shape: tuple[int, ...], patch_size: int) -> np.ndarray:
    b, c, h, w = shape
    return (
        p.reshape(b, h // patch_size, w // patch_size, c, patch_size, patch_size)
        .transpose(0, 3, 1, 4, 2, 5)
        .reshape(b, c, h, w)
    )


def patchify_latents_kl_divergence(x0: np.ndarray, x1: np.ndarray, patch_size: int = 4, num_channels: int = 4) -> float:
    a0 = _patchify(x0, patch_size, num_channels).reshape(-1, num_channels, patch_size * patch_size)
    a1 = _patchify(x1, patch_size, num_channels).reshape(-1, num_channels, patch_size * patch_size)
    kl = _kl(a0, a1)[0]
    return float(np.abs(kl).sum())


def _kl(a0: np.ndarray, a1: np.ndarray):
    a0 = a0.astype(np.float64, copy=False)
    a1 = a1.astype(np.float64, copy=False)
    mu0 = a0.mean(axis=-1)
    mu1 = a1.mean(axis=-1)
    var0 = a0.var(axis=-1, ddof=1)
    var1 = a1.var(axis=-1, ddof=1)
    kl = np.log((var1 + EPSILON) / (var0 + EPSILON)) + (var0 + (mu0 - mu1) ** 2) / (var1 + EPSILON) - 1.0
    return kl, mu0, mu1, var0, var1


def patchify_kl_divergence_grad(x0: np.ndarray, x1: np.ndarray, patch_size: int = 4, num_channels: int = 4) -> np.ndarray:
    shape = x0.shape
    n = patch_size * patch_size
    a0 = _patchify(x0, patch_size, num_channels).reshape(-1, num_channels, n)
    a1 = _patchify(x1, patch_size, num_channels).reshape(-1, num_channels, n)

    kl, mu0, mu1, var0, var1 = _kl(a0, a1)
    sign = np.sign(kl)
    d_var0 = (1.0 / (var1 + EPSILON) - 1.0 / (var0 + EPSILON)) * sign
    d_mu0 = (2.0 * (mu0 - mu1) / (var1 + EPSILON)) * sign

    a0f = a0.astype(np.float64, copy=False)
    grad = d_var0[..., None] * (2.0 * (a0f - mu0[..., None]) / (n - 1)) + d_mu0[..., None] / n
    grad = grad.astype(np.float32).reshape(-1, num_channels, patch_size, patch_size)
    return _unpatchify(grad, shape, patch_size)

def _avg_pool2d(y: np.ndarray) -> np.ndarray:
    n, c, h, w = y.shape
    return y.reshape(n, c, h // 2, 2, w // 2, 2).mean(axis=(3, 5))


def _avg_pool2d_backward(g: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(g, 2, axis=2), 2, axis=3) * 0.25


def _shift_terms(y: np.ndarray, roll_amount: int):
    loss = 0.0
    grad = np.zeros_like(y)
    inv_n = 1.0 / y.size
    for axis in (2, 3):
        z = np.roll(y, roll_amount, axis=axis)
        m = float((y * z).mean())
        loss += m * m
        grad += (2.0 * m * inv_n) * (z + np.roll(y, -roll_amount, axis=axis))
    return loss, grad


def auto_corr_loss_and_grad(x: np.ndarray, random_shift: bool = True, generator: np.random.Generator | None = None):
    if x.shape[0] != 1:
        raise AssertionError("auto_corr_loss expects batch size 1")
    rng = generator if generator is not None else np.random.default_rng()

    total_loss = 0.0
    grad = np.zeros_like(x, dtype=np.float32)

    for ch in range(x.shape[1]):
        levels: list[tuple[np.ndarray, int]] = []
        y = x[0:1, ch : ch + 1].astype(np.float32, copy=True)
        while True:
            roll_amount = int(rng.integers(0, y.shape[2] // 2)) if random_shift else 1
            levels.append((y, roll_amount))
            if y.shape[2] <= 8:
                break
            y = _avg_pool2d(y)

        g_deeper = None
        for y_level, roll_amount in reversed(levels):
            loss, g = _shift_terms(y_level, roll_amount)
            total_loss += loss
            if g_deeper is not None:
                g = g + _avg_pool2d_backward(g_deeper)
            g_deeper = g
        grad[0, ch] = g_deeper[0, 0]

    return total_loss, grad


def noise_regularization(
    e_t: np.ndarray,
    noise_pred_optimal: np.ndarray | None,
    lambda_kl: float,
    lambda_ac: float,
    num_reg_steps: int,
    num_ac_rolls: int,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    e_t = e_t.astype(np.float32, copy=True)
    for _ in range(num_reg_steps):
        if lambda_kl > 0:
            if noise_pred_optimal is None:
                raise ValueError("lambda_kl > 0 requires noise_pred_optimal")
            grad = patchify_kl_divergence_grad(e_t, noise_pred_optimal)
            grad = np.clip(grad, -100.0, 100.0)
            e_t = e_t - lambda_kl * grad
        if lambda_ac > 0:
            for _ in range(num_ac_rolls):
                _, grad = auto_corr_loss_and_grad(e_t, generator=generator)
                e_t = e_t - lambda_ac * (grad / num_ac_rolls)
    return e_t


def _apply_cfg(noise_pred: np.ndarray, guidance_scale: float) -> np.ndarray:
    uncond, text = np.split(noise_pred, 2, axis=0)
    return uncond + guidance_scale * (text - uncond)


def inversion_step(
    pipe,
    z_t: np.ndarray,
    t: int,
    prompt_embeds: np.ndarray,
    num_renoise_steps: int = 100,
    first_step_max_timestep: int = 250,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    cfg = pipe.cfg
    t = int(t)
    is_first_range = t < first_step_max_timestep
    avg_range = cfg.average_first_step_range if is_first_range else cfg.average_step_range
    if is_first_range:
        num_renoise_steps = min(cfg.max_num_renoise_steps_first_step, num_renoise_steps)

    total_iters = num_renoise_steps + 1
    reg_enabled = cfg.noise_regularization_num_reg_steps > 0
    reg_will_run = reg_enabled and any((i >= avg_range[0]) or (not cfg.average_latent_estimations and i > 0) for i in range(total_iters))
    avg_will_run = cfg.average_latent_estimations and any(avg_range[0] <= i < avg_range[1] for i in range(total_iters))
    need_optimal = reg_enabled and (reg_will_run or avg_will_run or not pipe.skip_unused_optimal_pass)

    noise_pred_avg = None
    noise_pred_optimal = None
    z_tp1_forward = pipe.forward_noised_latent(t)

    approximated_z_tp1 = z_t.copy()
    for i in range(total_iters):
        double_batch = need_optimal and i == 0
        if double_batch:
            model_input = np.concatenate([z_tp1_forward, approximated_z_tp1], axis=0)
            prompt_embeds_in = np.concatenate([prompt_embeds, prompt_embeds], axis=0)
        else:
            model_input = approximated_z_tp1
            prompt_embeds_in = prompt_embeds
        noise_pred = pipe.unet_pass(model_input, t, prompt_embeds_in)
        if double_batch:
            noise_pred_optimal, noise_pred = np.split(noise_pred, 2, axis=0)
            if pipe.do_classifier_free_guidance:
                noise_pred_optimal = _apply_cfg(noise_pred_optimal, pipe.guidance_scale)
        if pipe.do_classifier_free_guidance:
            noise_pred = _apply_cfg(noise_pred, pipe.guidance_scale)
        if avg_range[0] <= i < avg_range[1]:
            j = i - avg_range[0]
            if noise_pred_avg is None:
                noise_pred_avg = noise_pred.copy()
            else:
                noise_pred_avg = j * noise_pred_avg / (j + 1) + noise_pred / (j + 1)
        if i >= avg_range[0] or (not cfg.average_latent_estimations and i > 0):
            noise_pred = noise_regularization(
                noise_pred,
                noise_pred_optimal,
                lambda_kl=cfg.noise_regularization_lambda_kl,
                lambda_ac=cfg.noise_regularization_lambda_ac,
                num_reg_steps=cfg.noise_regularization_num_reg_steps,
                num_ac_rolls=cfg.noise_regularization_num_ac_rolls,
                generator=generator,
            )
        approximated_z_tp1 = pipe.scheduler.inv_step(noise_pred, t, z_t)
    if cfg.average_latent_estimations and noise_pred_avg is not None:
        noise_pred_avg = noise_regularization(
            noise_pred_avg,
            noise_pred_optimal,
            lambda_kl=cfg.noise_regularization_lambda_kl,
            lambda_ac=cfg.noise_regularization_lambda_ac,
            num_reg_steps=cfg.noise_regularization_num_reg_steps,
            num_ac_rolls=cfg.noise_regularization_num_ac_rolls,
            generator=generator,
        )
        approximated_z_tp1 = pipe.scheduler.inv_step(noise_pred_avg, t, z_t)
    if cfg.perform_noise_correction:
        raise NotImplementedError("perform_noise_correction only for Euler not for")
    return approximated_z_tp1
