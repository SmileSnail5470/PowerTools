import numpy as np

RESCALE_FACTOR = 0.00392156862745098  # 1/255
IMG_MEAN = 0.5
IMG_STD = 0.5


def _nearest_index_map(src_size: int, dst_size: int) -> np.ndarray:
    dst = np.arange(dst_size, dtype=np.int64)
    src = (src_size * dst) // dst_size
    np.minimum(src, src_size - 1, out=src)
    return src


def preprocess_bgr(image_bgr: np.ndarray, dst_width: int, dst_height: int) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[2] < 3:
        raise ValueError("Expected an HxWx3 BGR image")
    src_h, src_w = image_bgr.shape[:2]
    x_map = _nearest_index_map(src_w, dst_width)
    y_map = _nearest_index_map(src_h, dst_height)

    resized = image_bgr[np.ix_(y_map, x_map)][:, :, :3]

    b = resized[:, :, 0].astype(np.float32)
    g = resized[:, :, 1].astype(np.float32)
    r = resized[:, :, 2].astype(np.float32)

    r = (r * RESCALE_FACTOR - IMG_MEAN) / IMG_STD
    g = (g * RESCALE_FACTOR - IMG_MEAN) / IMG_STD
    b = (b * RESCALE_FACTOR - IMG_MEAN) / IMG_STD
    chw = np.stack([r, g, b], axis=0)
    return chw[np.newaxis, ...].astype(np.float32, copy=False)
