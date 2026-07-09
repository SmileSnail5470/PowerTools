import numpy as np


SAM3_OUTMASK_WIDTH = 288
SAM3_OUTMASK_HEIGHT = 288


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def _mask_index_map(src_size: int, mask_size: int) -> np.ndarray:
    idx = (np.arange(src_size, dtype=np.int64) * mask_size) // src_size
    np.minimum(idx, mask_size - 1, out=idx)
    return idx


def save_mask(src_bgr: np.ndarray, semantic_seg: np.ndarray, prob_threshold: float) -> np.ndarray:
    h, w = src_bgr.shape[:2]
    seg = np.asarray(semantic_seg, dtype=np.float32).reshape(SAM3_OUTMASK_HEIGHT, SAM3_OUTMASK_WIDTH,)
    x_map = _mask_index_map(w, SAM3_OUTMASK_WIDTH)
    y_map = _mask_index_map(h, SAM3_OUTMASK_HEIGHT)
    prob = _sigmoid(seg[np.ix_(y_map, x_map)])
    mask = (prob > prob_threshold).astype(np.uint8) * 255
    return mask
