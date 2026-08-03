import logging
import os
import numpy as np
from PIL import Image
from app.ui.common.config import cfg
from app.workers.work_base import BaseWorker, _resolve_hardware_variant
from app.utils.logger.decorators import log_exception
from app.algorithms.image_edit.general_edit.inference import ImageEditInference


class ImageEditWork(BaseWorker):
    _instance = None
    _instance_model_dir = None

    @classmethod
    def _get_instance(cls, model_dir):
        if cls._instance is None or cls._instance_model_dir != model_dir:
            cls._instance = ImageEditInference(model_dir=model_dir)
            cls._instance_model_dir = model_dir
        return cls._instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deps_path = cfg.get(cfg.localAIModelDeps)

    @log_exception(logger=logging.getLogger('ImageEdit'), reraise=True, log_args=True, log_result=True)
    def run_algorithm(self, progress_cb, cancel_requested, *args, **kwargs):
        input_path = kwargs["input_path"]
        output_path = kwargs["output_path"]
        prompt = kwargs.get("prompt", "")
        mask_boxes = kwargs.get("mask_boxes", [])

        if cancel_requested and cancel_requested():
            raise InterruptedError("Task was cancelled before start")

        model_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "image_edit", "general_edit")
        if "_feature_name_" in kwargs:
            os.environ["_feature_name_"] = kwargs["_feature_name_"]
        if progress_cb:
            progress_cb("EditStart", "")
        inference = self._get_instance(model_dir)
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        mask = None
        if mask_boxes:
            image = Image.open(input_path).convert("RGB")
            img_width, img_height = image.size
            mask_np = np.zeros((img_height, img_width), dtype=np.uint8)
            for box in mask_boxes:
                x = int(box.get('x', 0))
                y = int(box.get('y', 0))
                w = int(box.get('w', 0))
                h = int(box.get('h', 0))
                mask_np[y:y+h, x:x+w] = 255
            mask = mask_np
        result_np = inference.infer(prompt=prompt, input_path=input_path, mask=mask)
        result_img = Image.fromarray(result_np.astype(np.uint8))
        result_img.save(output_path)
        if progress_cb:
            progress_cb("EditDone", "")
        return output_path
