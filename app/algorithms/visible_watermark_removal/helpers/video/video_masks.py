import logging
import os
import shutil
import tempfile
import numpy as np
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol
from app.algorithms.tracker.mask_tracker_edgetam import MaskTrackerEdgeTAM
from app.algorithms.private.object_segmentation.video.inference import VideoObjectSegmentationInference
from app.algorithms.visible_watermark_removal.params_config import VideoWatermarkRemoveConfig
from app.algorithms.visible_watermark_removal.watermark_removal_image import WatermarkSegment


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
logger = logging.getLogger("subprocess")


class Tracker(Protocol):
    def inference(self, mask_path: str, frames_dir: str, reverse: bool = False, output_dir: str | None = None) -> int: 
        ...


@dataclass(frozen=True, slots=True)
class MaskPreparationResult:
    mask_dir: Path
    frame_mask_map: dict[str, str]


class MaskPreparer:
    def __init__(self, config: VideoWatermarkRemoveConfig) -> None:
        self.config = config
        self._segmentor: WatermarkSegment | None = None

    def _segment_frame(self, frame_file: Path) -> np.ndarray:
        config = self.config
        detection, segment = config.detection, config.segment
        if self._segmentor is None:
            self._segmentor = WatermarkSegment(detection)
        mask = self._segmentor.segment(
            image_path=str(frame_file),
            sr_onnx_path=segment.sr_segment_onnx_path,
            pt_onnx_path=segment.pt_segment_onnx_path,
            text_detection_onnx_path=segment.text_detection_onnx_path,
            yolo_detection_onnx_path=segment.yolo_detection_onnx_path,
            segment_onnx_dir=segment.segment_onnx_dir,
            **detection.to_segment_kwargs(),
        )
        return np.asarray(mask, dtype=np.uint8)

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        detection = self.config.detection
        if not detection.watermark_ai_detect_pos_boxes:
            return mask
        result = np.zeros(mask.shape, dtype=np.uint8)
        for xmin, ymin, xmax, ymax in detection.watermark_ai_detect_pos_boxes:
            result[ymin:ymax, xmin:xmax] = mask[ymin:ymax, xmin:xmax]
        return result

    def _save_mask(self, frame_file: Path, mask: np.ndarray, mask_dir: Path, mapping: dict[str, str]) -> str:
        path = mask_dir / frame_file.name
        Image.fromarray(self._postprocess_mask(mask)).convert("L").save(path)
        mapping[str(frame_file)] = str(path)
        return str(path)

    def prepare(self, mask_dir: Path, frame_files: list[Path]) -> MaskPreparationResult:
        mask_dir.mkdir(parents=True, exist_ok=True)
        if not frame_files:
            raise ValueError("Video contains no frames")
        mapping: dict[str, str] = {}
        mask_config = self.config.mask
        mask_result = None
        frame_files = sorted(frame_files, key=lambda p: int(Path(p).stem))
        if mask_config.use_cache_mask and not mask_config.mask_path:
            mask = self._postprocess_mask(self._segment_frame(frame_files[mask_config.video_watermark_ai_detect_anchor_frame-1]))
            start_frame_idx, end_frame_idx = mask_config.video_watermark_ai_detect_frame_range
            shared_mask_path = mask_dir / "mask.png"
            Image.fromarray(mask).convert("L").save(shared_mask_path)
            mapping.update({str(frame_file): str(shared_mask_path) for frame_file in frame_files[start_frame_idx-1:end_frame_idx]})
            mask_result = MaskPreparationResult(mask_dir, mapping)
        elif mask_config.mask_path and not Path(mask_config.mask_path).is_dir():
            shared_mask_path = mask_dir / "mask.png"
            shutil.copy2(mask_config.mask_path, shared_mask_path)
            mapping.update({str(frame_file): str(shared_mask_path) for frame_file in frame_files})
            mask_result = MaskPreparationResult(mask_dir, mapping)
        elif mask_config.mask_path and Path(mask_config.mask_path).is_dir():
            external_masks = sorted(path for path in Path(mask_config.mask_path).iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
            if len(external_masks) != len(frame_files):
                raise ValueError(f"Mask directory contains {len(external_masks)} image(s) for {len(frame_files)} video frame(s)")
            mapping.update({
                str(frame_file): str(mask_file)
                for frame_file, mask_file in zip(frame_files, external_masks)
            })
            mask_result = MaskPreparationResult(mask_dir, mapping)
        else:
            if mask_config.video_watermark_ai_detect_strategy == "frame_by_frame":
                start_frame_idx, end_frame_idx = mask_config.video_watermark_ai_detect_frame_range
                for frame_file in frame_files[start_frame_idx-1:end_frame_idx]:
                    self._save_mask(frame_file, self._segment_frame(frame_file), mask_dir, mapping)
                mask_result = MaskPreparationResult(mask_dir, mapping)
            else:
                tracking_data = mask_config.watermark_tracking_data
                if tracking_data["enable_preview_mode"]:
                    tracking_mask_dir = tracking_data["tracking_detect_mask_path"]
                    tracking_mask_files = sorted((p for p in tracking_mask_dir.iterdir() if p.is_file()), key=lambda p: p.stem)
                    frame_files = sorted(frame_files, key=lambda x: Path(x).stem)
                    mapping.update({str(frame): str(mask) for frame, mask in zip(frame_files, tracking_mask_files, strict=True)})
                    mask_result = MaskPreparationResult(tracking_mask_dir, mapping)
                else:
                    mask_dir = tracking_data["tracking_detect_mask_path"] or mask_dir
                    self._prepare_tracked_masks(mask_dir, frame_files, mapping, tracking_data)
                    mask_result = MaskPreparationResult(mask_dir, mapping)
        return mask_result

    def _prepare_tracked_masks(self, mask_dir: Path, frame_files: list[Path], mapping: dict[str, str], tracking_data: dict) -> None:
        tracking_algorithm = tracking_data["tracking_algorithm"]
        if tracking_algorithm == "edgetam":
            self._prepare_tracked_masks_edgetam(mask_dir, frame_files, mapping, tracking_data)
        else:
            self._prepare_tracked_masks_sam3(mask_dir, frame_files, mapping, tracking_data)

    def _prepare_tracked_masks_edgetam(self, mask_dir: Path, frame_files: list[Path], mapping: dict[str, str], tracking_data: dict) -> None:
        config = self.config
        model_dir = config.video_extra.tacker_onnx_dir
        tracking_detect_anchor_frame = int(tracking_data["tracking_detect_anchor_frame"])
        tracking_detect_direction = tracking_data["tracking_detect_direction"]
        tracking_detect_frame_range = tracking_data["tracking_detect_frame_range"]
        start_frame_idx, end_frame_idx = tracking_detect_frame_range[0], tracking_detect_frame_range[1]
        reverse_tracking_frames = max(0, int(tracking_data.get("reverse_tracking_frames", 8)))
        logger.info(
            "Start fast track mask with anchorframe %s and start frame %d to end frame %d and track direction %s, reverse_tracking_frames %d",
            tracking_detect_anchor_frame, start_frame_idx, end_frame_idx, tracking_detect_direction, reverse_tracking_frames,
        )
        total_frames = len(frame_files)
        keyframe_index = tracking_detect_anchor_frame - 1
        if keyframe_index < 0 or keyframe_index >= total_frames:
            raise Exception("Anchor frame exceed total_frames")

        def on_fail_frame(frame_file: Path) -> Optional[str]:
            mask = self._segment_frame(frame_file)
            if np.count_nonzero(mask) == 0:
                return None
            return self._save_mask(frame_file, mask, mask_dir, mapping)

        keyframe_file = frame_files[keyframe_index]
        keyframe_mask = self._segment_frame(keyframe_file)
        self._save_mask(keyframe_file, keyframe_mask, mask_dir, mapping)

        def run_pass(ordered_frames: list[Path], label: str) -> None:
            if len(ordered_frames) < 2:
                logger.info("%s tracking pass skipped: no frames to propagate", label)
                return
            logger.info("%s tracking pass: %s → %s (%d frames)", label, ordered_frames[0].name, ordered_frames[-1].name, len(ordered_frames))
            tracker = MaskTrackerEdgeTAM(model_dir=model_dir)
            try:
                result = tracker.track(
                    ordered_frames=ordered_frames,
                    output_dir=mask_dir,
                    on_fail_frame=on_fail_frame,
                    reverse_tracking_frames=reverse_tracking_frames,
                )
                mapping.update(result)
            finally:
                release = getattr(tracker, "release", None)
                if callable(release):
                    release()

        if tracking_detect_direction in ("forward", "both"):
            run_pass(frame_files[keyframe_index: end_frame_idx], "Forward")

        if tracking_detect_direction in ("backward", "both"):
            run_pass(list(reversed(frame_files[start_frame_idx - 1: keyframe_index + 1])), "Backward")

    def _prepare_tracked_masks_sam3(self, mask_dir: Path, frame_files: list[Path], mapping: dict[str, str], tracking_data: dict) -> None:
        config = self.config
        model_dir = config.video_extra.object_segmentation_model_dir
        tracking_detect_prompt = tracking_data["tracking_detect_prompt"]
        tracking_detect_pos_boxes = tracking_data["tracking_detect_pos_boxes"]
        tracking_detect_neg_boxes = tracking_data["tracking_detect_neg_boxes"]
        tracking_detect_confidence = tracking_data["tracking_detect_confidence"]
        tracking_detect_anchor_frame = int(tracking_data["tracking_detect_anchor_frame"])
        tracking_detect_direction = tracking_data["tracking_detect_direction"]
        tracking_detect_frame_range = tracking_data["tracking_detect_frame_range"]
        start_frame_idx, end_frame_idx = tracking_detect_frame_range[0], tracking_detect_frame_range[1]
        tracker = VideoObjectSegmentationInference(
        model_dir=model_dir, 
        video_model_dir=model_dir,
        max_mem_attn_batch=None
        )
        if tracking_detect_direction == "forward":
            max_frames = end_frame_idx - tracking_detect_anchor_frame
        elif tracking_detect_direction == "backward":
            max_frames = tracking_detect_anchor_frame - start_frame_idx
        else:
            max_frames = end_frame_idx - start_frame_idx
        # frame_files 和 mask_dir 里面的帧编号都是以索引 0 开始。
        tracker.infer(
            input_path=str(frame_files[0].parent),
            output_path=str(mask_dir),
            text=tracking_detect_prompt if tracking_detect_prompt else None,
            threshold=tracking_detect_confidence,
            box=tracking_detect_pos_boxes,
            negative_box=tracking_detect_neg_boxes,
            frame_index=tracking_detect_anchor_frame-1,
            max_frames=max_frames,
            propagation_direction=tracking_detect_direction,
        )
        mask_index = {p.stem: str(p) for p in Path(mask_dir).glob("*.png")}
        mapping.update({str(f): str(mask_index[s]) for f in frame_files if (s := Path(f).stem) in mask_index})