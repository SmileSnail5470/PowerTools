from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class InpaintModelPaths:
    pt_inpaint_onnx_path: str
    cf_onnx_path: str
    lama_onnx_path: str
    emdf_onnx_path: str
    grig_onnx_path: str
    general_edit_onnx_dir: str
    object_remove_onnx_dir: str = ""


@dataclass(frozen=True, slots=True)
class SegmentModelPaths:
    sr_segment_onnx_path: str
    pt_segment_onnx_path: str
    text_detection_onnx_path: str
    yolo_detection_onnx_path: str
    segment_onnx_dir: str


@dataclass(frozen=True, slots=True)
class VideoExtraModelPaths:
    ppt_onnx_basedir: str
    tacker_onnx_dir: str


@dataclass(frozen=True, slots=True)
class RestorationModelPaths:
    image_restoration_onnx_dir: str


@dataclass(slots=True)
class DetectionConfig:
    watermark_type: Literal["text", "subtitle", "all"] = "all"
    ai_detect_type: Literal[
        "ai_interactive_detect",
        "ai_auto_detect",
    ] = "ai_interactive_detect"
    ai_interactive_type: Literal[
        "semantic_detect",
        "space_detect",
    ] = "semantic_detect"
    ai_interactive_prompt: str = "watermark"
    ai_interactive_boxes: list = field(default_factory=list)
    watermark_confidence: float = 0.5
    watermark_boxes: list = field(default_factory=list)

    # OCR 检测超参
    ocr_limit_side_len: int = 960
    ocr_limit_type: str = "max"
    ocr_det_thresh: float = 0.3
    ocr_det_box_thresh: float = 0.6
    ocr_unclip_ratio: float = 1.5
    ocr_score_mode: str = "fast"
    ocr_det_box_type: str = "quad"

    # YOLO 检测超参
    yolo_confidence: float = 0.03
    yolo_iou_threshold: float = 0.2

    def to_segment_kwargs(self) -> dict[str, Any]:
        return {
            "limit_side_len": self.ocr_limit_side_len,
            "limit_type": self.ocr_limit_type,
            "det_thresh": self.ocr_det_thresh,
            "det_box_thresh": self.ocr_det_box_thresh,
            "unclip_ratio": self.ocr_unclip_ratio,
            "score_mode": self.ocr_score_mode,
            "det_box_type": self.ocr_det_box_type,
            "confidence": self.yolo_confidence,
            "iou_threshold": self.yolo_iou_threshold,
        }

@dataclass(slots=True)
class MaskConfig:
    mask_path: str = ""
    refine_type: str = "patchwiper"
    dilate_num: int = 2


@dataclass(slots=True)
class VideoMaskConfig(MaskConfig):
    refine_type: str = "coordfill"
    use_cache_mask: bool = False
    watermark_tracking_data: dict = field(default_factory=dict)


@dataclass(slots=True)
class RestorationConfig:
    refine_type: str = "image_restoration"
    restoration_prompt: str = ""
    restoration_watermark_type: str = "general"


@dataclass(slots=True)
class RuntimeConfig:
    progress_cb: Callable[[str, str], None] | None = None
    ffmpeg_path: str = ""
    callback_func: Callable[[int, int], None] | None = None


@dataclass(slots=True)
class ImageWatermarkRemoveConfig:
    segment: SegmentModelPaths
    inpaint: InpaintModelPaths
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass(slots=True)
class VideoWatermarkRemoveConfig:
    segment: SegmentModelPaths
    inpaint: InpaintModelPaths
    video_extra: VideoExtraModelPaths
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    mask: VideoMaskConfig = field(default_factory=VideoMaskConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass(slots=True)
class RestorationRemoveConfig:
    restoration: RestorationModelPaths
    params: RestorationConfig = field(default_factory=RestorationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)