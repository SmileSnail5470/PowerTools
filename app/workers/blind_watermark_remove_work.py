import logging
import os
from dataclasses import dataclass, field
from app.ui.common.config import cfg
from app.workers.work_base import BaseWorker, _resolve_hardware_variant
from app.utils.logger.decorators import log_exception
from app.algorithms.private.video_engine.config import EngineConfig
from app.algorithms.private.video_engine.engine import process_video
from app.algorithms.private.reverse_edit.inference import ReverseEditInference
from app.workers.watermark_extract_work import WatermarkExtractInternal


@dataclass
class _BlindRemoveIOParams:
    input_path: str = ""
    output_path: str = ""


@dataclass
class _BlindRemoveModelParams:
    model_name: str = ""


@dataclass
class _BlindRemoveSettingsParams:
    use_color_fix: bool = True
    high_quality_output: bool = False
    reserve_region: dict = None


@dataclass
class BlindWatermarkRemoveUIParams:
    io: _BlindRemoveIOParams = field(default_factory=_BlindRemoveIOParams)
    model: _BlindRemoveModelParams = field(default_factory=_BlindRemoveModelParams)
    settings: _BlindRemoveSettingsParams = field(default_factory=_BlindRemoveSettingsParams)
    _feature_name_: str | None = None

    def validate(self) -> str:
        if not self.io.input_path:
            return "请选择要处理的文件或目录"
        if not self.model.model_name:
            return "请选择去除模型"
        if not self.io.output_path:
            return "请设置文件保存位置"
        return ""


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
        p: BlindWatermarkRemoveUIParams = args[0].params
        input_path = p.io.input_path
        output_path = p.io.output_path
        use_color_fix = p.settings.use_color_fix
        high_quality_output = p.settings.high_quality_output
        reserve_region = p.settings.reserve_region

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
            f"{os.path.basename(input_path).rsplit('.', 1)[0]}_cleaned.{os.path.basename(input_path).rsplit('.', 1)[1]}"
        )
        if p._feature_name_ is not None:
            os.environ["_feature_name_"] = p._feature_name_

        if file_type == "video":
            video_config = EngineConfig(
                prompt="",
                task_type="general_edit",
                rife_model_path=os.path.join(self.deps_path, _resolve_hardware_variant(), "video_engine"),
                keyframe_stride=0,
                ffmpeg_path=os.getenv("POWERTOOLS_FFMPEG_BIN", ""),
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
                    "prompt": "",
                    "edit_prompt": None,
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
