import logging
import os
from dataclasses import dataclass, field
from app.ui.common.config import cfg
from app.workers.work_base import BaseWorker, _resolve_hardware_variant
from app.utils.logger.decorators import log_exception
from app.algorithms.ocr.pp_ocr import OCR


@dataclass
class _OCRIOParams:
    input_path: str = ""


@dataclass
class _OCRModelParams:
    model_name: str = "pp_ocr"


@dataclass
class _OCRSettingsParams:
    ocr_rec_language: str = "zh-jp-en"
    use_textline_ori: bool = False
    drop_score: float = 0.5


@dataclass
class OCRUIParams:
    io: _OCRIOParams = field(default_factory=_OCRIOParams)
    model: _OCRModelParams = field(default_factory=_OCRModelParams)
    settings: _OCRSettingsParams = field(default_factory=_OCRSettingsParams)
    _feature_name_: str = ""

    def validate(self) -> str:
        if not self.io.input_path:
            return "请选择要处理的文件或目录"
        if not self.model.model_name:
            return "请选择OCR识别算法"
        if not self.settings.ocr_rec_language:
            return "请设置要识别的目标语言"
        return ""


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
        p: OCRUIParams = args[0].params
        input_path = p.io.input_path
        onnx_model_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "ocr")
        if p._feature_name_:
            os.environ["_feature_name_"] = p._feature_name_
        ocr_params = {
            "limit_side_len": 960,
            "limit_type": "max",
            "det_thresh": 0.3,
            "det_box_thresh": 0.6,
            "unclip_ratio": 1.5,
            "score_mode": "fast",
            "det_box_type": "quad",
            "drop_score": float(p.settings.drop_score),
            "det_onnx_path": os.path.join(onnx_model_dir, "pp_ocr_det.encmodel"),
            "rec_onnx_path": os.path.join(onnx_model_dir, "pp_ocr_rec.encmodel"),
            "cls_onnx_path": os.path.join(onnx_model_dir, "pp_lcnet_x1_0_textline_ori.encmodel"),
            "progress_cb": progress_cb
        }
        ocr = self._get_ocr_instance()
        ocr.prepare(**ocr_params)
        ocr_result = ocr.predict(image_path=input_path, use_cls=bool(p.settings.use_textline_ori))
        return ocr_result