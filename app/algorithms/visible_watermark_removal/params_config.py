from collections.abc import Callable
from dataclasses import dataclass, field
import os
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
    object_segmentation_model_dir: str = ""


@dataclass(frozen=True, slots=True)
class RestorationModelPaths:
    image_restoration_onnx_dir: str

@dataclass(frozen=True, slots=True)
class VideoEngineModelPaths:
    video_engine_model_dir: str


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


TrackingAlgorithm = Literal["edgetam", "sam3"]


@dataclass(slots=True)
class VideoMaskConfig(MaskConfig):
    refine_type: str = "coordfill"
    use_cache_mask: bool = False
    watermark_tracking_data: dict = field(default_factory=dict)
    tracking_algorithm: TrackingAlgorithm = "sam3"


@dataclass(slots=True)
class RestorationConfig:
    refine_type: str = "image_restoration"
    restoration_prompt: str = ""
    restoration_watermark_type: str = "general"

@dataclass(slots=True)
class VideoEngineConfig:
    model_paths: VideoEngineModelPaths
    prompt: str | None = None
    task_type: str | None = None
    keyframe_stride: int = 2
    is_static_watermark: bool = False
    mask_path: str = ""
    auto_detect_mask: bool = False


@dataclass(slots=True)
class RuntimeConfig:
    progress_cb: Callable[[str, str], None] | None = None
    ffmpeg_path: str = ""
    callback_func: Callable[[int, int], None] | None = None
    should_cancel: Callable[[], bool] | None = None

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
class VideoEngineRemoveConfig:

    engine: VideoEngineConfig
    segment: SegmentModelPaths
    inpaint: InpaintModelPaths
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    video_extra: VideoExtraModelPaths | None = None
    mask: VideoMaskConfig | None = None


@dataclass(slots=True)
class RestorationRemoveConfig:
    restoration: RestorationModelPaths
    params: RestorationConfig = field(default_factory=RestorationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def get_ppt_onnx_path(ppt_onnx_basedir: str):
    ONNX_PATHS = {
        "raft": os.path.join(ppt_onnx_basedir, "raft", "raft_iter20.encmodel"),
        "recurrent_flow_complete": os.path.join(ppt_onnx_basedir, "recurrent_flow_completion"),
        "ppt": {
            'encoder': os.path.join(ppt_onnx_basedir, "propagation_transformer", "encoder.encmodel"),
            'decoder': os.path.join(ppt_onnx_basedir, "propagation_transformer", "decoder.encmodel"),
            "image_prop_step": os.path.join(ppt_onnx_basedir, "propagation_transformer", "img_prop_step.encmodel"),
            'ss': os.path.join(ppt_onnx_basedir, "propagation_transformer", "soft_split.encmodel"),
            'sc': os.path.join(ppt_onnx_basedir, "propagation_transformer", "soft_comp.encmodel"),
            'bp_backward_step': os.path.join(ppt_onnx_basedir, "propagation_transformer", "backward_step.encmodel"),
            'bp_forward_step': os.path.join(ppt_onnx_basedir, "propagation_transformer", "forward_step.encmodel"),
            'bp_backward_first': os.path.join(ppt_onnx_basedir, "propagation_transformer", "backward_first.encmodel"),
            'bp_forward_first': os.path.join(ppt_onnx_basedir, "propagation_transformer", "forward_first.encmodel"),
            'bp_fusion': os.path.join(ppt_onnx_basedir, "propagation_transformer", "fusion.encmodel"),
            'transformer': {
                'core': [os.path.join(ppt_onnx_basedir, 'propagation_transformer', f"attention_{i}_core.encmodel") for i in range(8)],
                'attn': [os.path.join(ppt_onnx_basedir, 'propagation_transformer', f"attention_{i}_comp.encmodel") for i in range(8)],
                'proj': [os.path.join(ppt_onnx_basedir, 'propagation_transformer', f"output_{i}_proj.encmodel") for i in range(8)],
                'norm1': [os.path.join(ppt_onnx_basedir, 'propagation_transformer', f"norm1_{i}_comp.encmodel") for i in range(8)],
                'norm2': [os.path.join(ppt_onnx_basedir, 'propagation_transformer', f"norm2_{i}_comp.encmodel") for i in range(8)],
                'mlp': [os.path.join(ppt_onnx_basedir, 'propagation_transformer', f"mlp_{i}_comp.encmodel") for i in range(8)],
            }
        }
    }
    return ONNX_PATHS