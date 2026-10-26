import logging
import os
from app.ui.common.config import cfg
from app.workers.work_base import BaseWorker, _resolve_hardware_variant
from app.utils.logger.decorators import log_exception
from app.algorithms.visible_watermark_removal.watermark_removal_image import ImageWatermarkRemove
from app.algorithms.visible_watermark_removal.watermark_removal_video import VideoWatermarkRemover
from app.algorithms.visible_watermark_removal.params_config import (
    DetectionConfig,
    ImageWatermarkRemoveConfig,
    InpaintModelPaths,
    MaskConfig,
    RestorationConfig,
    RestorationModelPaths,
    RestorationRemoveConfig,
    RuntimeConfig,
    SegmentModelPaths,
    VideoEngineConfig,
    VideoEngineModelPaths,
    VideoEngineRemoveConfig,
    VideoExtraModelPaths,
    VideoMaskConfig,
    VideoWatermarkRemoveConfig,
)
from dataclasses import dataclass, field



@dataclass
class _IOParams:
    input_path: str = ""
    output_path: str = ""
    output_format: str = "保持原格式"


@dataclass
class _DetectParams:
    watermark_detect_type: str = "ai_detect"
    video_watermark_format: str = "static_watermark"
    video_watermark_ai_detect_strategy: str = "tracking"
    video_watermark_ai_detect_frame_range: list = [1, 2**31 - 1]
    video_watermark_ai_detect_anchor_frame: int = 1
    manual_watermark_mask_path: str = ""
    watermark_ai_detect_method: str = "semantic_detect"
    watermark_ai_detect_prompt: str = ""
    watermark_ai_detect_pos_boxes: list = field(default_factory=list)
    watermark_ai_detect_neg_boxes: list = field(default_factory=list)
    watermark_ai_detect_confidence: float = 0.3
    video_watermark_tracking_data: dict = field(default_factory=lambda: {
        "keyframes": [], 
        "end_frame": None, 
        "reverse_tracking_frames": 8
    })


@dataclass
class _ModelParams:
    model_name: str = ""
    mask_dilate: int = 2
    # restoration 模型参数
    restoration_watermark_type: str = "general"
    restoration_prompt: str = ""


@dataclass
class WatermarkRemoveUIParams:
    io: _IOParams = field(default_factory=_IOParams)
    detect: _DetectParams = field(default_factory=_DetectParams)
    model: _ModelParams = field(default_factory=_ModelParams)
    _feature_name_: str = ""

    def validate(self) -> str:
        pass



class WatermarkRemoveWork(BaseWorker):
    _image_instance = None
    _video_instance = None

    @classmethod
    def _get_image_instance(cls):
        if cls._image_instance is None:
            cls._image_instance = ImageWatermarkRemove()
        return cls._image_instance

    @classmethod
    def _get_video_instance(cls):
        if cls._video_instance is None:
            cls._video_instance = VideoWatermarkRemover()
        return cls._video_instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deps_path = cfg.get(cfg.localAIModelDeps)


    @log_exception(logger=logging.getLogger('WatermarkRemove'), reraise=True, log_args=True, log_result=True)
    def run_algorithm(self, progress_cb, cancel_requested, *args, **kwargs):
        input_path = kwargs["input_path"]
        output_path = kwargs["output_path"]
        output_format = kwargs["output_format"]
        base_name, source_ext = os.path.splitext(os.path.basename(input_path))
        output_ext = source_ext.lstrip(".") if output_format == "保持原格式" else output_format.lower()
        output_file = os.path.join(output_path, f"{base_name}_wm.{output_ext}")

        file_type = self.file_type(input_file=input_path)
        if file_type is None:
            raise Exception("Not support file {0}".format(source_ext.lstrip(".")))
        onnx_model_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "visible_watermark_removal")
        segment_model_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "segment")
        general_edit_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "image_edit", "general_edit")
        image_restoration_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "image_restoration", "general_restoration")
        object_remove_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "image_edit", "object_remove")
        if "_feature_name_" in kwargs:
            os.environ["_feature_name_"] = kwargs["_feature_name_"]
        if "POWERTOOLS_TMPE_DIR" not in os.environ:
            os.environ["POWERTOOLS_TMPE_DIR"] = cfg.get(cfg.cachePath)

        segment_paths = SegmentModelPaths(
            sr_segment_onnx_path=os.path.join(onnx_model_dir, "sr_segment.encmodel"),
            pt_segment_onnx_path=os.path.join(onnx_model_dir, "pt_segment.encmodel"),
            text_detection_onnx_path=os.path.join(onnx_model_dir, "pp_ocr_det.encmodel"),
            yolo_detection_onnx_path=os.path.join(onnx_model_dir, "yolo.encmodel"),
            segment_onnx_dir=segment_model_dir,
        )
        inpaint_paths = InpaintModelPaths(
            pt_inpaint_onnx_path=os.path.join(onnx_model_dir, "pt_inpaint.encmodel"),
            cf_onnx_path=os.path.join(onnx_model_dir, "cf.encmodel"),
            lama_onnx_path=os.path.join(onnx_model_dir, "lama_fp32.encmodel"),
            emdf_onnx_path=os.path.join(onnx_model_dir, "emdf_inpaint.encmodel"),
            grig_onnx_path=os.path.join(onnx_model_dir, "grig_inpaint.encmodel"),
            general_edit_onnx_dir=general_edit_dir,
            object_remove_onnx_dir=object_remove_dir,
        )
        detection_cfg = DetectionConfig(
            watermark_type=(
                "text" if kwargs.get("watermark_content") == "text_watermark"
                else "subtitle" if kwargs.get("watermark_content") == "subtitle"
                else "all"
            ),
            ai_detect_type=kwargs["watermark_detect_type"],
            ai_interactive_type=kwargs.get("watermark_ai_interactive_type", "semantic_detect"),
            ai_interactive_prompt=kwargs.get("watermark_detect_prompt", "watermark"),
            ai_interactive_boxes=kwargs.get("watermark_boxes", []),
            watermark_confidence=kwargs.get("watermark_confidence", 0.5),
            watermark_boxes=kwargs.get("image_boxes", []),
        )
        runtime_cfg = RuntimeConfig(
            progress_cb=progress_cb,
            ffmpeg_path=os.getenv("POWERTOOLS_FFMPEG_BIN", ""),
            should_cancel=cancel_requested,
        )
        if file_type == "image":
            if kwargs["model_name"] not in ["image_restoration"]:
                image_cfg = ImageWatermarkRemoveConfig(
                    segment=segment_paths,
                    inpaint=inpaint_paths,
                    detection=detection_cfg,
                    mask=MaskConfig(
                        mask_path=kwargs["manual_watermark_mask_path"] if kwargs.get("manual_watermark_mask_path") else "",
                        refine_type=kwargs["model_name"],
                        dilate_num=int(kwargs["mask_dilate"]),
                    ),
                    runtime=runtime_cfg,
                )
                self._get_image_instance().run(input_path, output_file, config=image_cfg)
            else:
                restoration_cfg = RestorationRemoveConfig(
                    restoration=RestorationModelPaths(image_restoration_onnx_dir=image_restoration_dir),
                    params=RestorationConfig(
                        refine_type="image_restoration",
                        restoration_prompt=kwargs.get("restoration_prompt", ""),
                        restoration_watermark_type=kwargs.get("restoration_watermark_type", "general"),
                    ),
                    runtime=runtime_cfg,
                )
                self._get_image_instance().run_without_mask(input_path, output_file, config=restoration_cfg)
        elif kwargs["model_name"] != "video_engine":
            if kwargs["model_name"] not in ["image_restoration"]:
                ppt_onnx_basedir = os.path.join(self.deps_path, _resolve_hardware_variant(), "video_inpainting", "ppt")
                tacker_onnx_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "tracker")
                video_cfg = VideoWatermarkRemoveConfig(
                    segment=segment_paths,
                    inpaint=inpaint_paths,
                    video_extra=VideoExtraModelPaths(
                        ppt_onnx_basedir=ppt_onnx_basedir,
                        tacker_onnx_dir=tacker_onnx_dir,
                    ),
                    detection=detection_cfg,
                    mask=VideoMaskConfig(
                        mask_path=kwargs["manual_watermark_mask_path"] if kwargs.get("manual_watermark_mask_path") else "",
                        refine_type=kwargs["model_name"],
                        dilate_num=int(kwargs["mask_dilate"]),
                        use_cache_mask=(kwargs.get("watermark_format") == "static_watermark"),
                        watermark_tracking_data=kwargs.get("watermark_tracking_data", {}),
                    ),
                    runtime=RuntimeConfig(
                        progress_cb=progress_cb,
                        ffmpeg_path=os.getenv("POWERTOOLS_FFMPEG_BIN", ""),
                        callback_func=None,
                        should_cancel=cancel_requested,
                    ),
                )
                self._get_video_instance().process_video(input_path, output_file, config=video_cfg)
            else:
                restoration_cfg = RestorationRemoveConfig(
                    restoration=RestorationModelPaths(image_restoration_onnx_dir=image_restoration_dir),
                    params=RestorationConfig(
                        refine_type="image_restoration",
                        restoration_prompt=kwargs.get("restoration_prompt", ""),
                        restoration_watermark_type=kwargs.get("restoration_watermark_type", "general"),
                    ),
                    runtime=RuntimeConfig(
                        progress_cb=progress_cb,
                        ffmpeg_path=os.getenv("POWERTOOLS_FFMPEG_BIN", ""),
                        should_cancel=cancel_requested,
                    ),
                )
                self._get_video_instance().process_video_without_mask(input_path, output_file, config=restoration_cfg)
        else:
            engine_remove_cfg = VideoEngineRemoveConfig(
                engine=VideoEngineConfig(
                    model_paths=VideoEngineModelPaths(
                        video_engine_model_dir=os.path.join(self.deps_path, _resolve_hardware_variant(), "video_engine"),
                    ),
                    prompt=kwargs.get("prompt"),
                    task_type=kwargs.get("task_type"),
                    keyframe_stride=int(kwargs.get("keyframe_stride", 2)),
                    is_static_watermark=(kwargs.get("watermark_format") == "static_watermark"),
                    mask_path=kwargs["manual_watermark_mask_path"] if kwargs.get("manual_watermark_mask_path") else "",
                ),
                segment=segment_paths,
                inpaint=inpaint_paths,
                detection=detection_cfg,
                runtime=RuntimeConfig(
                    progress_cb=progress_cb,
                    ffmpeg_path=kwargs.get("ffmpeg_path") or os.getenv("POWERTOOLS_FFMPEG_BIN", ""),
                    should_cancel=cancel_requested,
                ),
            )
            self._get_video_instance().process_video_engine(input_path, output_file, config=engine_remove_cfg)
        return output_file