import logging
import os
from app.ui.common.config import cfg
from app.workers.work_base import BaseWorker, _resolve_hardware_variant
from app.utils.logger.decorators import log_exception
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
        self.cache_path = cfg.get(cfg.cachePath)

    @log_exception(logger=logging.getLogger('BlindWatermarkRemove'), reraise=True, log_args=True, log_result=True)
    def run_algorithm(self, progress_cb, cancel_requested, *args, **kwargs):
        input_path = kwargs["input_path"]
        output_path = kwargs["output_path"]
        use_color_fix = kwargs.get("use_color_fix", True)
        high_quality_output = kwargs.get("high_quality_output", False)
        reserve_region = kwargs.get("reserve_region")

        if cancel_requested and cancel_requested():
            raise InterruptedError("Task was cancelled before start")

        if WatermarkExtractInternal(file_type="image").has_powertools_blind_watermark(input_path=input_path):
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
