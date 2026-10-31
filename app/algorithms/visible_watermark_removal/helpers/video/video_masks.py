import logging
import shutil
import tempfile
import numpy as np
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol
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
    def __init__(self, config: VideoWatermarkRemoveConfig, tracker_factory: Callable[[str, str], Tracker] | None = None) -> None:
        self.config = config
        self._tracker_factory = tracker_factory or self._create_tracker
        self._segmentor: WatermarkSegment | None = None

    @staticmethod
    def _create_tracker(model_dir: str, algorithm: str) -> Tracker:
        if algorithm == "sam3":
            return VideoObjectSegmentationInference(model_dir=model_dir, video_model_dir=model_dir)
        return MaskTrackerEdgeTAM(model_dir=model_dir)

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

        def segment_tracking_frame(frame_file: Path) -> tuple[np.ndarray, str]:
            mask = self._segment_frame(frame_file)
            return mask, self._save_mask(frame_file, mask, mask_dir, mapping)

        def merge_reverse_transition_masks(tracker: Tracker, segment_frames: list[Path], fail_frame_idx: int, reference_mask_path: str) -> None:
            if reverse_tracking_frames <= 0 or fail_frame_idx <= 0:
                return
            reverse_start_index = max(0, fail_frame_idx - reverse_tracking_frames)
            reverse_frames = segment_frames[reverse_start_index:fail_frame_idx + 1]
            if len(reverse_frames) <= 1:
                return
            try:
                with tempfile.TemporaryDirectory(prefix="track_reverse_frames_") as frames_tmp, \
                        tempfile.TemporaryDirectory(prefix="track_reverse_masks_") as masks_tmp:
                    for frame_file in reverse_frames:
                        shutil.copy2(frame_file, Path(frames_tmp) / frame_file.name)
                    logger.info(
                        "Reverse tracker from frame %s over %d previous frames to repair transition",
                        reverse_frames[-1].name, len(reverse_frames) - 1,
                    )
                    reverse_fail_idx = tracker.inference(reference_mask_path, frames_tmp, reverse=True, output_dir=masks_tmp)
                    reverse_traversal = list(reversed(reverse_frames))
                    reliable_frames = (
                        reverse_traversal if reverse_fail_idx == -1
                        else reverse_traversal[:reverse_fail_idx]
                    )
                    merged_count = 0
                    for frame_file in reliable_frames:
                        reverse_mask_path = Path(masks_tmp) / frame_file.name
                        if not reverse_mask_path.exists():
                            continue
                        reverse_mask = np.asarray(Image.open(reverse_mask_path).convert("L"), dtype=np.uint8)
                        output_mask_path = mask_dir / frame_file.name
                        if output_mask_path.exists():
                            forward_mask = np.asarray(Image.open(output_mask_path).convert("L"), dtype=np.uint8)
                            merged_mask = np.maximum(forward_mask, reverse_mask)
                        else:
                            merged_mask = reverse_mask
                        Image.fromarray(merged_mask).convert("L").save(output_mask_path)
                        mapping[str(frame_file)] = str(output_mask_path)
                        merged_count += 1
                    logger.info(
                        "Reverse tracker repair completed at frame %s: merged %d frames%s",
                        reverse_frames[-1].name,
                        merged_count,
                        " before reverse tracking became unreliable" if reverse_fail_idx != -1 else "",
                    )
            except Exception:
                logger.exception("Reverse tracker repair failed at frame %s; continuing with forward restart", reverse_frames[-1].name)

        for segment_index, keyframe in enumerate(keyframes):
            segment_end = keyframes[segment_index + 1] if segment_index + 1 < len(keyframes) else effective_end + 1
            keyframe_index = keyframe - 1
            if keyframe_index < 0 or keyframe_index >= total_frames:
                continue
            keyframe_file = frame_files[keyframe_index]
            keyframe_mask, keyframe_mask_path = segment_tracking_frame(keyframe_file)
            segment_frame_start = keyframe_index + 1
            segment_frame_end = min(segment_end - 1, total_frames)
            if segment_frame_start >= segment_frame_end:
                continue
            current_track_start = segment_frame_start
            if np.count_nonzero(keyframe_mask) == 0:
                tracking_state = "ABSENT"
                current_reference_mask_path = None
                current_reference_frame = None
                logger.info("No mask at keyframe %s; entering ABSENT state", keyframe_file.name)
            else:
                tracking_state = "TRACKING"
                current_reference_mask_path = keyframe_mask_path
                current_reference_frame = keyframe_file
            while current_track_start < segment_frame_end:
                if tracking_state == "ABSENT":
                    probe_frame = frame_files[current_track_start]
                    probe_mask, probe_mask_path = segment_tracking_frame(probe_frame)
                    current_track_start += 1
                    if np.count_nonzero(probe_mask) == 0:
                        logger.info("Mask remains absent at frame %s", probe_frame.name)
                        continue
                    current_reference_mask_path = probe_mask_path
                    current_reference_frame = probe_frame
                    logger.info("Mask reacquired at frame %s; restarting tracker", probe_frame.name)
                    tracking_state = "TRACKING"
                    continue
                remaining_frames = frame_files[current_track_start:segment_frame_end]
                if not remaining_frames:
                    break
                segment_frames = [current_reference_frame] + remaining_frames
                with tempfile.TemporaryDirectory(prefix="track_seg_") as segment_frames_dir:
                    for frame_file in segment_frames:
                        shutil.copy2(frame_file, Path(segment_frames_dir) / frame_file.name)
                    tracker = self._tracker_factory(model_dir, algorithm)
                    try:
                        fail_frame_idx = tracker.inference(current_reference_mask_path, segment_frames_dir)
                        if fail_frame_idx == -1:
                            for frame_file in remaining_frames:
                                tracked_mask_path = mask_dir / frame_file.name
                                if tracked_mask_path.exists():
                                    mapping[str(frame_file)] = str(tracked_mask_path)
                            break
                        successful_count = max(0, fail_frame_idx - 1)
                        for frame_file in remaining_frames[:successful_count]:
                            tracked_mask_path = mask_dir / frame_file.name
                            if tracked_mask_path.exists():
                                mapping[str(frame_file)] = str(tracked_mask_path)
                        failed_frame_offset = successful_count
                        if failed_frame_offset >= len(remaining_frames):
                            break
                        failed_frame = remaining_frames[failed_frame_offset]
                        logger.warning("Tracking failed at frame %s; verifying with segmentation", failed_frame.name)
                        verified_mask, verified_mask_path = segment_tracking_frame(failed_frame)
                        current_track_start += failed_frame_offset + 1
                        if np.count_nonzero(verified_mask) == 0:
                            tracking_state = "ABSENT"
                            current_reference_mask_path = None
                            current_reference_frame = None
                            logger.info("No mask found at frame %s; entering ABSENT state", failed_frame.name)
                            continue
                        merge_reverse_transition_masks(tracker, segment_frames, fail_frame_idx, verified_mask_path)
                        current_reference_mask_path = verified_mask_path
                        current_reference_frame = failed_frame
                        logger.info("Mask verified at frame %s; restarting tracker", failed_frame.name)
                        tracking_state = "TRACKING"
                    finally:
                        release = getattr(tracker, "release", None)
                        if callable(release):
                            release()

    def _prepare_tracked_masks_sam3(self, mask_dir: Path, frame_files: list[Path], mapping: dict[str, str], tracking_data: dict) -> None:
        pass