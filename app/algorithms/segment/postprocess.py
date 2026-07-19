import cv2
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


def save_mask(src_bgr: np.ndarray, semantic_seg: np.ndarray, prob_threshold: float) -> np.ndarray:
    h, w = src_bgr.shape[:2]
    seg = np.asarray(semantic_seg, dtype=np.float32).reshape(SAM3_OUTMASK_HEIGHT, SAM3_OUTMASK_WIDTH)
    seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_LINEAR)
    prob = _sigmoid(seg)
    mask = (prob > prob_threshold).astype(np.uint8) * 255
    return mask
