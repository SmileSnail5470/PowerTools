import logging
import os
from app.ui.common.config import cfg
from app.workers.work_base import BaseWorker
from app.utils.logger.decorators import log_exception
from app.algorithms.ocr.pp_ocr import OCR



class OCRWork(BaseWorker):
    _ocr_instance = None

    @classmethod
    def _get_ocr_instance(cls):
        if cls._ocr_instance is None:
            cls._ocr_instance = OCR()
        return cls._ocr_instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deps_path = cfg.get(cfg.localAIModelDeps)


    @log_exception(logger=logging.getLogger('OCR'), reraise=True, log_args=True, log_result=True)
    def run_algorithm(self, progress_cb, cancel_requested, *args, **kwargs):
        input_path = kwargs["input_path"]
        onnx_model_dir = os.path.join(self.deps_path, "ocr")
        if "_feature_name_" in kwargs:
            os.environ["_feature_name_"] = kwargs["_feature_name_"]
        params = {
            "limit_side_len": 960,
            "limit_type": "max",
            "det_thresh": 0.3,
            "det_box_thresh": 0.6,
            "unclip_ratio": 1.5,
            "score_mode": "fast",
            "det_box_type": "quad",
            "drop_score": float(kwargs["drop_score"]),
            "det_onnx_path": os.path.join(onnx_model_dir, "pp_ocr_det.onnx"),
            "rec_onnx_path": os.path.join(onnx_model_dir, "pp_ocr_rec.onnx"),
            "cls_onnx_path": os.path.join(onnx_model_dir, "pp_lcnet_x1_0_textline_ori.onnx"),
            "progress_cb": progress_cb
        }
        ocr = self._get_ocr_instance()
        ocr.prepare(**params)
        ocr_result = ocr.predict(image_path=input_path, use_cls=bool(kwargs["use_textline_ori"]) if "use_textline_ori" in kwargs else False)
        return ocr_result