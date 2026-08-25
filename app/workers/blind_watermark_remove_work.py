import logging
import os
from app.ui.common.config import cfg
from app.workers.work_base import BaseWorker, _resolve_hardware_variant
from app.utils.logger.decorators import log_exception
from app.algorithms.private.video_engine.config import EngineConfig
from app.algorithms.private.video_engine.engine import process_video
from app.algorithms.private.reverse_edit.inference import ReverseEditInference
from app.workers.watermark_extract_work import WatermarkExtractInternal


class BlindWatermarkRemoveWork(BaseWorker):
    _instance = None
    _instance_model_dir = None

    @classmethod
    def _get_instance(cls, model_dir, use_color_fix=True):
        if cls._instance is None or cls._instance_model_dir != model_dir:
            cls._instance = ReverseEditInference(model_dir=model_dir, use_color_fix=use_color_fix)
            cls._instance_model_dir = model_dir
        return cls._instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deps_path = cfg.get(cfg.localAIModelDeps)

    @log_exception(logger=logging.getLogger('BlindWatermarkRemove'), reraise=True, log_args=True, log_result=True)
    def run_algorithm(self, progress_cb, cancel_requested, *args, **kwargs):
        input_path = kwargs["input_path"]
        output_path = kwargs["output_path"]
        use_color_fix = kwargs.get("use_color_fix", True)
        high_quality_output = kwargs.get("high_quality_output", False)
        reserve_region = kwargs.get("reserve_region")

        if cancel_requested and cancel_requested():
            raise InterruptedError("Task was cancelled before start")

        file_type = self.file_type(input_file=input_path)
        if file_type is None:
            raise Exception(f"Not support file {input_path.split('.')[-1]}")
        if file_type == "image" and WatermarkExtractInternal(file_type="image").has_powertools_blind_watermark(input_path=input_path):
            raise Exception(f"Reject process {input_path}")
        
        model_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "image_edit", "reverse_edit")
        output_dir = output_path
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            f"{os.path.basename(input_path).rsplit(".", 1)[0]}_cleaned.{os.path.basename(input_path).rsplit(".", 1)[1]}"
        )
        if "_feature_name_" in kwargs:
            os.environ["_feature_name_"] = kwargs["_feature_name_"]

        if file_type == "video":
            video_config = EngineConfig(
                prompt=kwargs.get("prompt", ""),
                task_type=kwargs.get("task_type", "general_edit"),
                rife_model_path=kwargs.get("rife_model_path") or os.path.join(self.deps_path, _resolve_hardware_variant(), "video_engine", "rife"),
                keyframe_stride=int(kwargs.get("keyframe_stride", 0)),
                ffmpeg_path=kwargs.get("ffmpeg_path") or os.getenv("POWERTOOLS_FFMPEG_BIN", ""),
                should_cancel=cancel_requested,
            )
            if progress_cb:
                progress_cb("BlindWatermarkRemoveStart", "")
            process_video(
                input_video=input_path,
                output_video=output_file,
                plugin="reverse_edit",
                config=video_config,
                plugin_kwargs={
                    "model_dir": model_dir,
                    "prompt": kwargs.get("prompt", ""),
                    "edit_prompt": kwargs.get("edit_prompt"),
                    "use_color_fix": use_color_fix,
                    "high_quality_output": high_quality_output,
                    "reserve_region": reserve_region,
                },
            )
            if progress_cb:
                progress_cb("BlindWatermarkRemoveCompleted", "")
            return (output_file, {})
        else:
            if progress_cb:
                progress_cb("BlindWatermarkRemoveStart", "")
            inference = self._get_instance(model_dir=model_dir, use_color_fix=use_color_fix)
            inference.use_color_fix = use_color_fix
            inference.infer(
                input_path=input_path,
                output_path=output_file,
                prompt="",
                region=reserve_region,
                high_quality_output=high_quality_output,
            )
            if progress_cb:
                progress_cb("BlindWatermarkRemoveCompleted", "")
            return (output_file, {})
