import logging
import os
import tempfile
import numpy as np
from PIL import Image
from app.ui.common.config import cfg
from app.workers.work_base import BaseWorker, _resolve_hardware_variant
from app.utils.logger.decorators import log_exception
from app.algorithms.private.image_restoration.inference import ImageRestorationInference


restoration_logger = logging.getLogger('ImageRestoration')
SAVE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


class ImageRestorationWork(BaseWorker):
    _instance = None
    _instance_model_dir = None

    @classmethod
    def _get_instance(cls, model_dir, **kwargs):
        if cls._instance is None or cls._instance_model_dir != model_dir:
            cls._instance = ImageRestorationInference(model_dir=model_dir, **kwargs)
            cls._instance_model_dir = model_dir
        return cls._instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deps_path = cfg.get(cfg.localAIModelDeps)

    @staticmethod
    def _output_file(input_path: str, output_dir: str) -> str:
        basename = os.path.basename(input_path)
        stem, ext = os.path.splitext(basename)
        if ext.lower() not in SAVE_SUFFIXES:
            ext = ".png"
        return os.path.join(output_dir, f"{stem}_restored{ext}")

    @staticmethod
    def _pre_upscale(input_path: str, upscale: int) -> str:
        image = Image.open(input_path).convert("RGB")
        width, height = image.size
        image = image.resize((width * upscale, height * upscale), resample=Image.Resampling.LANCZOS)
        tmp_dir = os.environ.get("POWERTOOLS_TASK_TMPDIR") or tempfile.gettempdir()
        os.makedirs(tmp_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(input_path))[0]
        tmp_path = os.path.join(tmp_dir, f"{stem}_upscaled_x{upscale}.png")
        image.save(tmp_path)
        return tmp_path

    @log_exception(logger=restoration_logger, reraise=True, log_args=True, log_result=True)
    def run_algorithm(self, progress_cb, cancel_requested, *args, **kwargs):
        input_path = kwargs["input_path"]
        output_path = kwargs["output_path"]
        task_type = kwargs.get("task_type", "restoration")
        prompt = kwargs.get("prompt", "").strip()
        num_inference_steps = kwargs.get("num_inference_steps")
        guidance = kwargs.get("guidance")
        seed = kwargs.get("seed", 42)
        low_memory = kwargs.get("low_memory", True)
        upscale = int(kwargs.get("upscale", 1) or 1)

        if cancel_requested and cancel_requested():
            raise InterruptedError("Task was cancelled before start")

        file_type = self.file_type(input_file=input_path)
        if file_type is None:
            raise Exception(f"Not support file {os.path.basename(input_path)}")
        if file_type != "image":
            raise Exception("图像修复暂不支持视频文件，视频算法接入中")

        if "_feature_name_" in kwargs:
            os.environ["_feature_name_"] = kwargs["_feature_name_"]

        model_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "image_restoration", "general_restoration")
        output_dir = output_path
        os.makedirs(output_dir, exist_ok=True)
        output_file = self._output_file(input_path, output_dir)

        if progress_cb:
            progress_cb("RestorationStart", "")

        infer_path = input_path
        tmp_path = ""
        if task_type == "super_resolution" and upscale > 1:
            tmp_path = self._pre_upscale(input_path, upscale)
            infer_path = tmp_path

        try:
            inference = self._get_instance(
                model_dir,
                low_memory=low_memory,
                num_inference_steps=num_inference_steps,
                guidance=guidance,
                seed=seed,
            )
            inference.num_inference_steps = int(num_inference_steps or inference.num_inference_steps)
            inference.guidance = guidance
            inference.seed = seed
            result_np = inference.infer(prompt=prompt, input_path=infer_path, task_type=task_type)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        if cancel_requested and cancel_requested():
            raise InterruptedError("Task was cancelled after execution")

        result_img = Image.fromarray(np.asarray(result_np).astype(np.uint8))
        if output_file.lower().endswith((".jpg", ".jpeg")):
            result_img = result_img.convert("RGB")
        result_img.save(output_file)

        if progress_cb:
            progress_cb("RestorationCompleted", "")
        return (output_file, {})
