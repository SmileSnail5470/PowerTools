from typing import Tuple
import cv2
import numpy as np
from PIL import Image


_KERNEL = np.array(
    [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ],
    dtype=np.float64,
)


def _to_tensor(image: Image.Image) -> np.ndarray:
    if image.mode not in ("RGB", "L", "RGBA"):
        image = image.convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:
        array = array[:, :, None]
    array = array.astype(np.float32) / 255.0
    return array.transpose(2, 0, 1)[None, ...] # HWC -> CHW -> NCHW


def _to_pil_image(tensor: np.ndarray) -> Image.Image:
    """(1, C, H, W) float array in [0, 1] -> PIL image.
    
    """
    array = tensor[0].transpose(1, 2, 0)
    array = (array * 255.0).astype(np.uint8)
    if array.shape[2] == 1:
        return Image.fromarray(array[:, :, 0], mode="L")
    if array.shape[2] == 4:
        return Image.fromarray(array, mode="RGBA")
    return Image.fromarray(array, mode="RGB")


def wavelet_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Apply wavelet blur to the input array of shape (N, C, H, W).

    """
    height, width = image.shape[2], image.shape[3]
    rows = [np.clip(np.arange(height) + (j - 1) * radius, 0, height - 1) for j in range(3)]
    cols = [np.clip(np.arange(width) + (i - 1) * radius, 0, width - 1) for i in range(3)]
    output = np.zeros(image.shape, dtype=np.float64)
    for j in range(3):
        shifted_rows = image[:, :, rows[j], :]
        for i in range(3):
            output += _KERNEL[j, i] * shifted_rows[:, :, :, cols[i]]
    return output.astype(image.dtype, copy=False)


def wavelet_decomposition(image: np.ndarray, levels: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Wavelet decomposition returning only the summed high frequency and the final low frequency band.
    
    """
    high_freq = np.zeros_like(image)
    low_freq = image
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq = high_freq + (image - low_freq)
        image = low_freq
    return high_freq, low_freq


def wavelet_reconstruction(content_feat: np.ndarray, style_feat: np.ndarray) -> np.ndarray:
    """Give `content_feat` the color (low frequency) of `style_feat`.
    
    """
    if content_feat.shape != style_feat.shape:
        raise ValueError(f"shape mismatch: content {content_feat.shape} vs style {style_feat.shape}")
    content_high_freq, _ = wavelet_decomposition(content_feat)
    _, style_low_freq = wavelet_decomposition(style_feat)
    return content_high_freq + style_low_freq


def calc_mean_std(feat: np.ndarray, mask: np.ndarray = None, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    assert feat.ndim == 4, "The input feature should be 4D array."
    # Reshape to (N, C, H*W) and compute stats along spatial dims
    n, c = feat.shape[:2]
    if mask is None:
        feat_flat = feat.reshape(n, c, -1)
        feat_var = feat_flat.var(axis=2, ddof=0) + eps  # (N, C)
        feat_std = np.sqrt(feat_var).reshape(n, c, 1, 1)
        feat_mean = feat_flat.mean(axis=2).reshape(n, c, 1, 1)
        return feat_mean, feat_std 
    else:
        valid_mask = (mask < 0.5).astype(feat.dtype)
        valid_count = valid_mask.sum(axis=(2, 3), keepdims=True)
        valid_count = np.maximum(valid_count, 1.0)
        feat_mean = (feat * valid_mask).sum(axis=(2, 3), keepdims=True) / valid_count
        feat_var = (((feat - feat_mean) ** 2) * valid_mask).sum(axis=(2, 3), keepdims=True) / valid_count
        feat_std = np.sqrt(feat_var + eps)
        return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: np.ndarray, style_feat: np.ndarray, style_mask: np.ndarray = None) -> np.ndarray:
    style_mean, style_std = calc_mean_std(style_feat, mask=style_mask)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean) / content_std
    return normalized_feat * style_std + style_mean


def adain_color_fix(target: Image.Image, source: Image.Image, mask_gray: Image.Image = None) -> Image.Image:
    target_tensor = _to_tensor(target)
    source_tensor = _to_tensor(source)
    if source_tensor.shape != target_tensor.shape:
        source = source.convert(target.mode).resize(target.size, Image.LANCZOS)
        source_tensor = _to_tensor(source)
    mask_tensor = None
    if mask_gray is not None:
        if mask_gray.size != source.size:
            mask_gray = mask_gray.resize(source.size, Image.NEAREST)
        mask_tensor = _to_tensor(mask_gray)
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor, style_mask=mask_tensor)
    np.clip(result_tensor, 0.0, 1.0, out=result_tensor)
    return _to_pil_image(result_tensor)


def wavelet_color_fix(target: Image.Image, source: Image.Image) -> Image.Image:
    target_tensor = _to_tensor(target)
    source_tensor = _to_tensor(source)
    if source_tensor.shape != target_tensor.shape:
        source = source.convert(target.mode).resize(target.size, Image.LANCZOS)
        source_tensor = _to_tensor(source)
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)
    np.clip(result_tensor, 0.0, 1.0, out=result_tensor)
    return _to_pil_image(result_tensor)

def lab_color_fix(original_img: Image.Image, generated_img: Image.Image, mask_gray: np.ndarray = None) -> Image.Image:
    orig_rgb = np.array(original_img.convert("RGB"))
    gen_rgb = np.array(generated_img.convert("RGB"))
    orig_lab = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gen_lab = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    if mask_gray is None:
        bg_mask = np.ones(orig_rgb.shape[:2], dtype=bool)
    else:
        bg_mask = (mask_gray == 0)
        if not np.any(bg_mask):
            bg_mask = np.ones_like(mask_gray, dtype=bool)
    calibrated_lab = gen_lab.copy()
    orig_l_bg = orig_lab[:, :, 0][bg_mask]
    gen_l_bg = gen_lab[:, :, 0][bg_mask]
    mean_orig, std_orig = np.mean(orig_l_bg), np.std(orig_l_bg)
    mean_gen, std_gen = np.mean(gen_l_bg), np.std(gen_l_bg)
    std_gen = max(std_gen, 1e-5)
    calibrated_lab[:, :, 0] = (gen_lab[:, :, 0] - mean_gen) * (std_orig / std_gen) + mean_orig
    calibrated_lab = np.clip(calibrated_lab, 0, 255).astype(np.uint8)
    calibrated_rgb = cv2.cvtColor(calibrated_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(calibrated_rgb)
