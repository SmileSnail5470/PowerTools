import logging
import os
from app.ui.common.config import cfg
from app.workers.work_base import BaseWorker, _resolve_hardware_variant
from app.utils.logger.decorators import log_exception
from app.algorithms.visible_watermark_removal.watermark_removal_image import ImageWatermarkRemove
from app.algorithms.visible_watermark_removal.watermark_removal_video import VideoWatermarkRemover



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
        input_path = kwargs["input_path"]
        output_path = kwargs["output_path"]
        output_format = kwargs["output_format"]
        if output_format == "保持原格式":
            output_file = os.path.join(output_path, "{0}_wm.{1}".format(os.path.basename(input_path).split(".")[0], os.path.basename(input_path).split(".")[1]))
        else:
            output_file = os.path.join(output_path, "{0}_wm.{1}".format(os.path.basename(input_path).split(".")[0], output_format.lower()))

        file_type = self.file_type(input_file=input_path)
        if file_type is None:
            raise Exception("Not support file {0}".format(input_path.split(".")[-1]))
        onnx_model_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "visible_watermark_removal")
        segment_model_dir = os.path.join(self.deps_path, _resolve_hardware_variant(), "segment")
        if "_feature_name_" in kwargs:
            os.environ["_feature_name_"] = kwargs["_feature_name_"]
        if file_type == "image":
            params = {
                "image_path": input_path, 
                "output_path": output_file,
                "sr_segment_onnx_path": os.path.join(onnx_model_dir, "sr_segment.encmodel"),
                "pt_segment_onnx_path": os.path.join(onnx_model_dir, "pt_segment.encmodel"),
                "pt_inpaint_onnx_path": os.path.join(onnx_model_dir, "pt_inpaint.encmodel"), 
                "cf_onnx_path": os.path.join(onnx_model_dir, "cf.encmodel"), 
                "lama_onnx_path": os.path.join(onnx_model_dir, "lama_fp32.encmodel"),
                "emdf_onnx_path": os.path.join(onnx_model_dir, "emdf_inpaint.encmodel"),
                "grig_onnx_path": os.path.join(onnx_model_dir, "grig_inpaint.encmodel"),
                "text_detection_onnx_path": os.path.join(onnx_model_dir, "pp_ocr_det.encmodel"),
                "yolo_detection_onnx_path": os.path.join(onnx_model_dir, "yolo.encmodel"),
                "segment_onnx_dir": segment_model_dir,
                "mask_path": kwargs["manual_watermark_mask_path"] if "manual_watermark_mask_path" in kwargs and kwargs["manual_watermark_mask_path"] else "",
                "refine_type": kwargs["model_name"],
                "watermark_type": "text" if "watermark_content" in kwargs and kwargs["watermark_content"] == "text_watermark" else "subtitle" if "watermark_content" in kwargs and kwargs["watermark_content"] == "subtitle" else "all",
                "ai_detect_type": kwargs["watermark_detect_type"],
                "ai_interactive_type": kwargs["watermark_ai_interactive_type"] if "watermark_ai_interactive_type" in kwargs else "semantic_detect",
                "ai_interactive_prompt": kwargs["watermark_detect_prompt"] if "watermark_detect_prompt" in kwargs else "watermark",
                "ai_interactive_boxes": kwargs["watermark_boxes"] if "watermark_boxes" in kwargs else [],
                "watermark_confidence": kwargs["watermark_confidence"] if "watermark_confidence" in kwargs else 0.5,
                "dilate_num": int(kwargs["mask_dilate"]),
                "progress_cb": progress_cb,
            }
            self._get_image_instance().run(**params)
        else:
            ppt_onnx_basedir = os.path.join(self.deps_path, _resolve_hardware_variant(), "video_inpainting", "ppt")
            params = {
                "input_video_path": input_path, 
                "output_video_path": output_file,
                "sr_segment_onnx_path": os.path.join(onnx_model_dir, "sr_segment.encmodel"),
                "pt_segment_onnx_path": os.path.join(onnx_model_dir, "pt_segment.encmodel"),
                "pt_inpaint_onnx_path": os.path.join(onnx_model_dir, "pt_inpaint.encmodel"), 
                "cf_onnx_path": os.path.join(onnx_model_dir, "cf.encmodel"), 
                "lama_onnx_path": os.path.join(onnx_model_dir, "lama_fp32.encmodel"),
                "emdf_onnx_path": os.path.join(onnx_model_dir, "emdf_inpaint.encmodel"),
                "grig_onnx_path": os.path.join(onnx_model_dir, "grig_inpaint.encmodel"),
                "text_detection_onnx_path": os.path.join(onnx_model_dir, "pp_ocr_det.encmodel"),
                "yolo_detection_onnx_path": os.path.join(onnx_model_dir, "yolo.encmodel"),
                "segment_onnx_dir": segment_model_dir,
                "ppt_onnx_basedir": ppt_onnx_basedir,
                "mask_path": kwargs["manual_watermark_mask_path"] if "manual_watermark_mask_path" in kwargs and kwargs["manual_watermark_mask_path"] else "",
                "refine_type": kwargs["model_name"],
                "use_cache_mask": True if "watermark_format" in kwargs and kwargs["watermark_format"] == "static_watermark" else False,
                "watermark_type": "text" if "watermark_content" in kwargs and kwargs["watermark_content"] == "text_watermark" else "subtitle" if "watermark_content" in kwargs and kwargs["watermark_content"] == "subtitle" else "all",
                "ai_detect_type": kwargs["watermark_detect_type"],
                "ai_interactive_type": kwargs["watermark_ai_interactive_type"] if "watermark_ai_interactive_type" in kwargs else "semantic_detect",
                "ai_interactive_prompt": kwargs["watermark_detect_prompt"] if "watermark_detect_prompt" in kwargs else "watermark",
                "ai_interactive_boxes": kwargs["watermark_boxes"] if "watermark_boxes" in kwargs else [],
                "watermark_confidence": kwargs["watermark_confidence"] if "watermark_confidence" in kwargs else 0.5,
                "dilate_num": int(kwargs["mask_dilate"]),
                "ffmpeg_path": os.getenv("POWERTOOLS_FFMPEG_BIN"),
                "progress_cb": progress_cb,
            }
            self._get_video_instance().process_video(**params)
        return output_file