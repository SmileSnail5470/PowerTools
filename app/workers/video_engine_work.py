import logging
import os
from app.ui.common.config import cfg
from app.workers.work_base import BaseWorker, _resolve_hardware_variant
from app.utils.logger.decorators import log_exception
from app.algorithms.private.video_engine.config import EngineConfig
from app.algorithms.private.video_engine.engine import process_video


# 关键帧任务插件 -> (开始阶段, 完成阶段)，与其它 worker 的进度语义保持一致
PLUGIN_STAGES = {
    "watermark_removal": ("MaskStart", "WaterRemoved"),
    "reverse_edit": ("BlindWatermarkRemoveStart", "BlindWatermarkRemoveCompleted"),
    "image_edit": ("EditStart", "EditDone"),
}
DEFAULT_PLUGIN = "watermark_removal"


class VideoEngineWork(BaseWorker):
    """视频编辑引擎（分镜 + 关键帧图片模型 + RIFE 插帧）的调用入口。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deps_path = cfg.get(cfg.localAIModelDeps)

    def _model_dir(self, *parts: str) -> str:
        return os.path.join(self.deps_path, _resolve_hardware_variant(), *parts)

    def _output_file(self, input_path: str, output_path: str, output_format: str) -> str:
        name, _, suffix = os.path.basename(input_path).rpartition(".")
        if output_format and output_format != "保持原格式":
            suffix = output_format.lower()
        return os.path.join(output_path, "{0}_wm.{1}".format(name or "output", suffix or "mp4"))

    def _watermark_removal_kwargs(self, kwargs: dict) -> dict:
        onnx_model_dir = self._model_dir("visible_watermark_removal")
        refine_type = (
            kwargs.get("refine_type")
            or kwargs.get("keyframe_model_name")
            or kwargs.get("model_name")
        )
        if refine_type in (None, "", "video_engine"):
            # 该入口的定位是"图片模型处理关键帧"，默认走通用图像编辑模型
            refine_type = "general_edit"
        return {
            "sr_segment_onnx_path": os.path.join(onnx_model_dir, "sr_segment.encmodel"),
            "pt_segment_onnx_path": os.path.join(onnx_model_dir, "pt_segment.encmodel"),
            "pt_inpaint_onnx_path": os.path.join(onnx_model_dir, "pt_inpaint.encmodel"),
            "cf_onnx_path": os.path.join(onnx_model_dir, "cf.encmodel"),
            "lama_onnx_path": os.path.join(onnx_model_dir, "lama_fp32.encmodel"),
            "emdf_onnx_path": os.path.join(onnx_model_dir, "emdf_inpaint.encmodel"),
            "grig_onnx_path": os.path.join(onnx_model_dir, "grig_inpaint.encmodel"),
            "text_detection_onnx_path": os.path.join(onnx_model_dir, "pp_ocr_det.encmodel"),
            "yolo_detection_onnx_path": os.path.join(onnx_model_dir, "yolo.encmodel"),
            "segment_onnx_dir": self._model_dir("segment"),
            "general_edit_onnx_dir": self._model_dir("image_edit", "general_edit"),
            "refine_type": refine_type,
            "watermark_type": (
                "text" if kwargs.get("watermark_content") == "text_watermark"
                else "subtitle" if kwargs.get("watermark_content") == "subtitle"
                else "all"
            ),
            "ai_detect_type": kwargs.get("watermark_detect_type", "ai_interactive_detect"),
            "ai_interactive_type": kwargs.get("watermark_ai_interactive_type", "semantic_detect"),
            "ai_interactive_prompt": kwargs.get("watermark_detect_prompt", "watermark"),
            "ai_interactive_boxes": kwargs.get("watermark_boxes", []),
            "watermark_confidence": kwargs.get("watermark_confidence", 0.5),
            "dilate_num": int(kwargs.get("mask_dilate", 2)),
        }

    def _reverse_edit_kwargs(self, kwargs: dict) -> dict:
        return {
            "model_dir": self._model_dir("image_edit", "reverse_edit"),
            "prompt": kwargs.get("prompt", ""),
            "edit_prompt": kwargs.get("edit_prompt"),
            "use_color_fix": kwargs.get("use_color_fix", True),
            "high_quality_output": kwargs.get("high_quality_output", False),
            "reserve_region": kwargs.get("reserve_region"),
        }

    def _image_edit_kwargs(self, kwargs: dict) -> dict:
        return {
            "model_dir": self._model_dir("image_edit", "general_edit"),
            "prompt": kwargs.get("prompt"),
            "task_type": kwargs.get("task_type"),
            "dilate_num": int(kwargs.get("mask_dilate", 2)),
        }

    def _plugin_kwargs(self, plugin_name: str, kwargs: dict) -> dict:
        if plugin_name == "watermark_removal":
            return self._watermark_removal_kwargs(kwargs)
        if plugin_name == "reverse_edit":
            return self._reverse_edit_kwargs(kwargs)
        if plugin_name == "image_edit":
            return self._image_edit_kwargs(kwargs)
        raise ValueError(f"Unsupported video engine plugin: {plugin_name}")

    @log_exception(logger=logging.getLogger('VideoEngine'), reraise=True, log_args=True, log_result=True)
    def run_algorithm(self, progress_cb, cancel_requested, *args, **kwargs):
        input_path = kwargs["input_path"]
        output_path = kwargs["output_path"]
        output_format = kwargs.get("output_format", "保持原格式")
        plugin_name = kwargs.get("plugin_name") or DEFAULT_PLUGIN

        if cancel_requested and cancel_requested():
            raise InterruptedError("Task was cancelled before start")
        if self.file_type(input_file=input_path) != "video":
            raise Exception("Video engine only supports video input: {0}".format(input_path))
        if "_feature_name_" in kwargs:
            os.environ["_feature_name_"] = kwargs["_feature_name_"]

        os.makedirs(output_path, exist_ok=True)
        output_file = self._output_file(input_path, output_path, output_format)
        start_stage, done_stage = PLUGIN_STAGES.get(plugin_name, ("VideoEngineStart", "VideoEngineCompleted"))
        reported = {"percent": -1}

        def report(done: int, total: int) -> None:
            if progress_cb is None:
                return
            percent = int(done * 100 / total) if total else 0
            if percent == reported["percent"]:
                return
            reported["percent"] = percent
            progress_cb("VideoEngineProgress", "{0}/{1}".format(done, total))

        config = EngineConfig(
            prompt=kwargs.get("prompt"),
            task_type=kwargs.get("task_type", "general_edit"),
            rife_model_path=kwargs.get("rife_model_path") or self._model_dir("video_engine", "rife"),
            keyframe_min_stride=int(kwargs.get("keyframe_min_stride", 2)),
            keyframe_max_stride=int(kwargs.get("keyframe_max_stride", 8)),
            feather_radius=int(kwargs.get("mask_feather", 6)),
            ffmpeg_path=kwargs.get("ffmpeg_path") or os.getenv("POWERTOOLS_FFMPEG_BIN", ""),
            temp_dir=os.getenv("POWERTOOLS_TASK_TMPDIR") or None,
            callback_func=report,
            should_cancel=cancel_requested,
        )
        if progress_cb:
            progress_cb(start_stage, "")
        stats = process_video(
            input_video=input_path,
            output_video=output_file,
            plugin=plugin_name,
            mask=kwargs.get("mask_path") or kwargs.get("manual_watermark_mask_path") or None,
            config=config,
            plugin_kwargs=self._plugin_kwargs(plugin_name, kwargs),
        )
        logging.getLogger('VideoEngine').info("Video engine stats: %s", stats)
        if progress_cb:
            progress_cb(done_stage, "")
        return output_file
