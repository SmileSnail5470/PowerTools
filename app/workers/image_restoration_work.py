import logging
import os
import cv2
import numpy as np
from PIL import Image
from app.ui.common.config import cfg
from app.workers.work_base import BaseWorker, _resolve_hardware_variant
from app.utils.logger.decorators import log_exception
from app.algorithms.private.sr_edit.inference import ImageSRInference
from app.algorithms.private.image_restoration.inference import ImageRestorationInference


restoration_logger = logging.getLogger('ImageRestoration')
SAVE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


class ImageRestorationWork(BaseWorker):
    _instance = None
    _instance_model_dir = None

    @classmethod
    def _get_restoration_instance(cls, model_dir, **kwargs):
        if cls._instance is None or cls._instance_model_dir != model_dir:
            cls._instance = ImageRestorationInference(model_dir=model_dir, **kwargs)
            cls._instance_model_dir = model_dir
        return cls._instance

    @classmethod
    def _get_sr_instance(cls, model_dir, **kwargs):
        if cls._instance is None or cls._instance_model_dir != model_dir:
            cls._instance = ImageSRInference(model_dir=model_dir, **kwargs)
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

    def _post_upscale(self, image: Image.Image, upscale: int) -> Image.Image:
        if upscale == 4:
            return image
        width, height = image.size
        out_w = int(width // 4 * upscale)
        out_h = int(height // 4 * upscale)
        return image.resize((out_w, out_h), Image.Resampling.LANCZOS)

    @log_exception(logger=restoration_logger, reraise=True, log_args=True, log_result=True)
    def run_algorithm(self, progress_cb, cancel_requested, *args, **kwargs):
        input_path = kwargs["input_path"]
        output_path = kwargs["output_path"]
        task_type = kwargs.get("task_type", "restoration")
        prompt = kwargs.get("prompt", "").strip()
        seed = kwargs.get("seed", 42)
        low_memory = kwargs.get("low_memory", True)
        upscale = int(kwargs.get("upscale", 1) or 1)
        model_name = kwargs.get("model_name")

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

        if task_type != "super_resolution" or model_name != "image_sr":
            inference = self._get_restoration_instance(model_dir, low_memory=low_memory, seed=seed)
            result_np = inference.infer(prompt=prompt, input_path=input_path, task_type=task_type)
            result_img = Image.fromarray(np.asarray(result_np).astype(np.uint8))
        else:
            sr_model_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "image_edit", "general_edit", "sr_edit")
            sr_edit_instance = self._get_sr_instance(sr_model_dir, low_memory=low_memory)
            image = Image.open(input_path)
            arr = np.array(image.convert("RGB"))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            result_bgr = sr_edit_instance.inference(img=bgr)
            result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            result_img = self._post_upscale(image=Image.fromarray(result), upscale=upscale)

        if cancel_requested and cancel_requested():
            raise InterruptedError("Task was cancelled after execution")
        
        result_img.save(output_file)
        if progress_cb:
            progress_cb("RestorationCompleted", "")
        return (output_file, {})
