import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class TrackResult:
    frame_index: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    score: float                     # TrackerNano confidence in [0, 1]
    reliable: bool                   # score >= threshold and box valid


class MaskTrackerNano:
    def __init__(
        self,
        tacker_onnx_dir: Optional[str] = None,
        score_threshold: float = 0.3,
        backend: int = cv2.dnn.DNN_BACKEND_DEFAULT,
        target: int = cv2.dnn.DNN_TARGET_CPU,
    ):
        if tacker_onnx_dir is None:
            raise ValueError("tacker_onnx_dir must be specified.")
        
        if not hasattr(cv2, "TrackerNano_create"):
            raise RuntimeError(
                "cv2.TrackerNano is unavailable. Install opencv-contrib-python."
            )
        backbone_path = os.path.join(tacker_onnx_dir, "nanotrack_backbone_sim.onnx")
        head_path = os.path.join(tacker_onnx_dir, "nanotrack_head_sim.onnx")
        params = cv2.TrackerNano_Params()
        params.backbone = backbone_path
        params.neckhead = head_path
        params.backend = backend
        params.target = target

        self._params = params
        self.score_threshold = float(score_threshold)
        self._tracker: Optional[cv2.TrackerNano] = None
        self._ref_mask_crop: Optional[np.ndarray] = None 
        self._init_bbox: Optional[Tuple[int, int, int, int]] = None
        self._frame_shape: Optional[Tuple[int, int]] = None  # (H, W)
        self._frame_index = -1

    @staticmethod
    def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            raise ValueError("Input mask is empty (no non-zero pixels).")
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return x0, y0, x1 - x0 + 1, y1 - y0 + 1

    def init(self, frame: np.ndarray, mask: np.ndarray) -> TrackResult:
        if frame.ndim != 3:
            raise ValueError("frame must be a BGR image with 3 channels.")
        mask = self._prepare_mask(mask, frame.shape[:2])

        bbox = self.bbox_from_mask(mask)
        self._tracker = cv2.TrackerNano_create(self._params)
        self._tracker.init(frame, bbox)

        x, y, w, h = bbox
        self._init_bbox = bbox
        self._ref_mask_crop = mask[y : y + h, x : x + w].copy()
        self._frame_shape = frame.shape[:2]
        self._frame_index = 0

        result = TrackResult(0, bbox, 1.0, True)
        self._last_mask = mask
        return result

    def update(self, frame: np.ndarray) -> Tuple[np.ndarray, TrackResult]:
        if self._tracker is None:
            raise RuntimeError("Call init() before update().")

        self._frame_index += 1
        ok, box = self._tracker.update(frame)
        score = float(self._tracker.getTrackingScore())

        H, W = frame.shape[:2]
        x, y, w, h = (int(round(v)) for v in box)
        # Clamp box to frame bounds.
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        bbox = (x, y, w, h)

        reliable = bool(ok) and score >= self.score_threshold
        mask = self._warp_mask_to_bbox(bbox, (H, W))
        if not reliable:
            print(f"Warning: Tracking unreliable at frame {self._frame_index}, score={score:.3f}", flush=True)
            return False, None, None

        self._last_mask = mask
        return True, mask, TrackResult(self._frame_index, bbox, score, reliable)

    def _warp_mask_to_bbox(self, bbox: Tuple[int, int, int, int], shape: Tuple[int, int]) -> np.ndarray:
        H, W = shape
        x, y, w, h = bbox
        canvas = np.zeros((H, W), np.uint8)
        if self._ref_mask_crop is None or w <= 0 or h <= 0:
            return canvas
        resized = cv2.resize(self._ref_mask_crop, (w, h), interpolation=cv2.INTER_NEAREST)
        canvas[y : y + h, x : x + w] = resized
        return canvas

    @staticmethod
    def _prepare_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[:2] != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        binm = (mask > 0).astype(np.uint8) * 255
        return binm
    
    def iter_frames(self, frames_dir: str) -> Iterator[Tuple[str, np.ndarray]]:
        file_names = sorted(f for f in os.listdir(frames_dir))
        for name in file_names:
            img = cv2.imread(os.path.join(frames_dir, name), cv2.IMREAD_COLOR)
            if img is not None:
                yield name, img
    
    def inference(self, mask_path: str, frames_dir: str) -> List[TrackResult]:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask image: {mask_path}")

        mask_out = os.path.dirname(mask_path)
        os.makedirs(mask_out, exist_ok=True)

        initialized = False
        for _, (name, frame) in enumerate(self.iter_frames(frames_dir)):
            base = os.path.splitext(name)[0]
            if not initialized:
                res = self.init(frame, mask)
                out_mask = self._last_mask
                initialized = True
            else:
                res, out_mask, res = self.update(frame)
            if res:
                cv2.imwrite(os.path.join(mask_out, f"{base}.png"), out_mask)
            else:
                return self._frame_index
        return -1
