from typing import Tuple
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


def wavelet_color_fix(target: Image.Image, source: Image.Image) -> Image.Image:
    """Transfer the color of `source` onto `target` via wavelet decomposition.
    
    """
    target_tensor = _to_tensor(target)
    source_tensor = _to_tensor(source)

    if source_tensor.shape != target_tensor.shape:
        source = source.convert(target.mode).resize(target.size, Image.LANCZOS)
        source_tensor = _to_tensor(source)

    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)
    np.clip(result_tensor, 0.0, 1.0, out=result_tensor)

    return _to_pil_image(result_tensor)
