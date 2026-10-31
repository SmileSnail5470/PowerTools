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
from dataclasses import dataclass, field, asdict



@dataclass
class _IOParams:
    input_path: str = ""
    output_path: str = ""
    output_format: str = "保持原格式"


@dataclass
class _VideoTrackingParams:
    enable_preview_mode: bool = True
    tracking_algorithm: str | None = None
    tracking_detect_prompt: str | None = None
    tracking_detect_pos_boxes: list = field(default_factory=list)
    tracking_detect_neg_boxes: list = field(default_factory=list)
    tracking_detect_anchor_frame: int | None = None
    tracking_detect_direction: str | None = None
    tracking_detect_frame_range: list = field(default_factory=lambda: [1, 2**31 - 1])
    tracking_detect_confidence: float = 0.3
    tracking_detect_mask_path: str | None = None
    # edgetam 算法使用
    reverse_tracking_frames: int = 8


@dataclass
class _DetectParams:
    watermark_detect_type: str = "ai_detect"
    video_watermark_format: str = "static_watermark"
    video_watermark_ai_detect_strategy: str = "tracking"
    video_watermark_ai_detect_frame_range: list = field(default_factory=lambda: [1, 2**31 - 1])
    video_watermark_ai_detect_anchor_frame: int = 1
    manual_watermark_mask_path: str = ""
    watermark_ai_detect_method: str = "semantic_detect"
    watermark_ai_detect_prompt: str = ""
    watermark_ai_detect_pos_boxes: list = field(default_factory=list)
    watermark_ai_detect_neg_boxes: list = field(default_factory=list)
    watermark_ai_detect_confidence: float = 0.3
    video_watermark_tracking_data: _VideoTrackingParams = field(default_factory=_VideoTrackingParams)


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
    _feature_name_: str | None = None

    def validate(self) -> str:
        if not self.io.input_path:
            return "请选择要处理的文件或目录"
        if not self.io.output_path:
            return "请设置文件保存位置"
        if not self.model.model_name:
            return "请选择水印移除算法"
        if self.model.model_name in ["image_restoration"]:
            # 文档图片模型不需要检测流程
            return ""
        if self.detect.watermark_detect_type == "manual_detect":
            return ""
        if self.detect.watermark_ai_detect_method == "semantic_detect" and not self.detect.watermark_ai_detect_prompt:
            return "已选【语义检测】请输入水印语义检测提示词"
        if self.detect.watermark_ai_detect_method == "space_detect" and not self.detect.watermark_ai_detect_pos_boxes:
            return "已选【框选区域】请框选检测的区域"
        return ""


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
        p: WatermarkRemoveUIParams = args[0].params
        input_path = p.io.input_path
        output_path = p.io.output_path
        output_format = p.io.output_format
        base_name, source_ext = os.path.splitext(os.path.basename(input_path))
        output_ext = source_ext.lstrip(".") if output_format == "保持原格式" else output_format.lower()
        output_file = os.path.join(output_path, f"{base_name}_wm.{output_ext}")

        file_type = self.file_type(input_file=input_path)
        if file_type is None:
            raise Exception("Not support file {0}".format(source_ext.lstrip(".")))
        onnx_model_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "visible_watermark_removal")
        segment_model_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "object_segmentation")
        general_edit_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "image_edit", "general_edit")
        image_restoration_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "image_restoration", "general_restoration")
        object_remove_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "image_edit", "object_remove")
        if p._feature_name_ is not None:
            os.environ["_feature_name_"] = p._feature_name_
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
        detect = p.detect
        tracking_data = detect.video_watermark_tracking_data
        detection_cfg = DetectionConfig(
            watermark_ai_detect_method=detect.watermark_ai_detect_method,
            watermark_ai_detect_prompt=detect.watermark_ai_detect_prompt,
            watermark_ai_detect_pos_boxes=detect.watermark_ai_detect_pos_boxes,
            watermark_ai_detect_neg_boxes=detect.watermark_ai_detect_neg_boxes,
            watermark_ai_detect_confidence=detect.watermark_ai_detect_confidence,
        )
        runtime_cfg = RuntimeConfig(
            progress_cb=progress_cb,
            ffmpeg_path=os.getenv("POWERTOOLS_FFMPEG_BIN", ""),
            callback_func=None,
            should_cancel=cancel_requested,
        )
        tracking_dict = asdict(tracking_data)

        if file_type == "image":
            if p.model.model_name not in ["image_restoration"]:
                image_cfg = ImageWatermarkRemoveConfig(
                    segment=segment_paths,
                    inpaint=inpaint_paths,
                    detection=detection_cfg,
                    mask=MaskConfig(
                        mask_path=detect.manual_watermark_mask_path,
                        refine_type=p.model.model_name,
                        dilate_num=int(p.model.mask_dilate),
                    ),
                    runtime=runtime_cfg,
                )
                self._get_image_instance().run(input_path, output_file, config=image_cfg)
            else:
                restoration_cfg = RestorationRemoveConfig(
                    restoration=RestorationModelPaths(image_restoration_onnx_dir=image_restoration_dir),
                    params=RestorationConfig(
                        refine_type=p.model.model_name,
                        restoration_prompt=p.model.restoration_prompt,
                        restoration_watermark_type=p.model.restoration_watermark_type,
                    ),
                    runtime=runtime_cfg,
                )
                self._get_image_instance().run_without_mask(input_path, output_file, config=restoration_cfg)
        elif p.model.model_name != "video_engine":
            if p.model.model_name not in ["image_restoration"]:
                ppt_onnx_basedir = os.path.join(self.deps_path, _resolve_hardware_variant(), "video_inpainting", "ppt")
                tacker_onnx_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "tracker")
                video_cfg = VideoWatermarkRemoveConfig(
                    segment=segment_paths,
                    inpaint=inpaint_paths,
                    video_extra=VideoExtraModelPaths(
                        ppt_onnx_basedir=ppt_onnx_basedir,
                        tacker_onnx_dir=tacker_onnx_dir,
                        object_segmentation_model_dir=segment_model_dir
                    ),
                    detection=detection_cfg,
                    mask=VideoMaskConfig(
                        mask_path=detect.manual_watermark_mask_path,
                        refine_type=p.model.model_name,
                        dilate_num=int(p.model.mask_dilate),
                        use_cache_mask=(detect.video_watermark_format == "static_watermark"),
                        video_watermark_ai_detect_strategy=detect.video_watermark_ai_detect_strategy,
                        video_watermark_ai_detect_anchor_frame=detect.video_watermark_ai_detect_anchor_frame,
                        video_watermark_ai_detect_frame_range=detect.video_watermark_ai_detect_frame_range,
                        watermark_tracking_data=tracking_dict,
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
                        refine_type=p.model.model_name,
                        restoration_prompt=p.model.restoration_prompt,
                        restoration_watermark_type=p.model.restoration_watermark_type,
                    ),
                    runtime=RuntimeConfig(
                        progress_cb=progress_cb,
                        ffmpeg_path=os.getenv("POWERTOOLS_FFMPEG_BIN", ""),
                        should_cancel=cancel_requested,
                        callback_func=None,
                    ),
                )
                self._get_video_instance().process_video_without_mask(input_path, output_file, config=restoration_cfg)
        else:
            engine_remove_cfg = VideoEngineRemoveConfig(
                engine=VideoEngineConfig(
                    model_paths=VideoEngineModelPaths(
                        video_engine_model_dir=os.path.join(self.deps_path, _resolve_hardware_variant(), "video_engine"),
                    ),
                    prompt=None,
                    task_type=None,
                    keyframe_stride=2,
                    is_static_watermark=(detect.video_watermark_format == "static_watermark"),
                    mask_path=detect.manual_watermark_mask_path,
                ),
                segment=segment_paths,
                inpaint=inpaint_paths,
                detection=detection_cfg,
                runtime=RuntimeConfig(
                    progress_cb=progress_cb,
                    ffmpeg_path=os.getenv("POWERTOOLS_FFMPEG_BIN", ""),
                    should_cancel=cancel_requested,
                ),
            )
            self._get_video_instance().process_video_engine(input_path, output_file, config=engine_remove_cfg)
        return output_file