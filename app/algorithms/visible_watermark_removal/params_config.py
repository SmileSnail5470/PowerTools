from typing import Literal, Any
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class ModelPathsConfig:
    # 检测
    sr_segment_onnx_path: Path
    pt_segment_onnx_path: Path
    text_detection_onnx_path: Path
    yolo_detection_onnx_path: Path

    tacker_onnx_dir: Path
    segment_onnx_dir: Path

    # 修复
    pt_inpaint_onnx_path: Path
    cf_onnx_path: Path
    lama_onnx_path: Path
    emdf_onnx_path: Path
    grig_onnx_path: Path
    general_edit_onnx_dir: Path
    object_remove_onnx_dir: Path | None = None

    ppt_onnx_basedir: Path


@dataclass(slots=True)
class DetectionConfig:
    watermark_type: str = "all"
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
    watermark_tracking_data: dict = field(default_factory=dict)

    # OCR
    ocr_limit_side_len = 960
    ocr_limit_type = "max"
    ocr_det_thresh = 0.3
    ocr_det_box_thresh = 0.6
    ocr_unclip_ratio = 1.5
    ocr_score_mode = "fast"
    ocr_det_box_type = "quad"

    # YOLO
    yolo_confidence = 0.03
    yolo_iou_threshold = 0.2


@dataclass(slots=True)
class MaskConfig:
    mask_path: str | None = None
    use_cache_mask: bool = False
    refine_type = "coordfill"
    dilate_num: int = 2


@dataclass(slots=True)
class RuntimeConfig:
    ffmpeg_path: str | None = None
    callback_func: Callable[..., Any] | None = None


@dataclass(slots=True)
class VideoProcessConfig:
    models: ModelPathsConfig
    detection: DetectionConfig = field(
        default_factory=DetectionConfig
    )
    mask: MaskConfig = field(
        default_factory=MaskConfig
    )
    runtime: RuntimeConfig = field(
        default_factory=RuntimeConfig
    )