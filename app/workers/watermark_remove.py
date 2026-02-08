import logging
import os
from app.ui.common.config import cfg
from app.workers.work_base import BaseWorker
from app.utils.logger.decorators import log_exception
from app.algorithms.visible_watermark_removal.watermark_removal_image import ImageWatermarkRemove
from app.algorithms.visible_watermark_removal.watermark_removal_video import VideoWatermarkRemover



class WatermarkRemoveWork(BaseWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deps_path = cfg.get(cfg.localAIModelDeps)
        self.watermark_remove_image_instance = ImageWatermarkRemove()
        self.watermark_remove_video_instance = VideoWatermarkRemover()


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
        onnx_model_dir = os.path.join(self.deps_path, "visible_watermark_removal")
        if file_type == "image":
            params = {
                "image_path": input_path, 
                "output_path": output_file,
                "sr_segment_onnx_path": os.path.join(onnx_model_dir, "sr_segment.onnx"),
                "pt_segment_onnx_path": os.path.join(onnx_model_dir, "pt_segment.onnx"),
                "pt_inpaint_onnx_path": os.path.join(onnx_model_dir, "pt_inpaint.onnx"), 
                "cf_onnx_path": os.path.join(onnx_model_dir, "cf.onnx"), 
                "lama_onnx_path": os.path.join(onnx_model_dir, "lama_fp32.onnx"),
                "emdf_onnx_path": os.path.join(onnx_model_dir, "emdf_inpaint.onnx"),
                "grig_onnx_path": os.path.join(onnx_model_dir, "grig_inpaint.onnx"),
                "text_detection_onnx_path": os.path.join(onnx_model_dir, "pp_ocr_det.onnx"),
                "yolo_detection_onnx_path": os.path.join(onnx_model_dir, "yolo.onnx"),
                "mask_path": kwargs["mask_path"] if "mask_path" in kwargs and kwargs["mask_path"] else "",
                "refine_type": kwargs["model_name"],
                "watermark_type": "all" if "watermark_content" in kwargs and kwargs["watermark_content"] == "通用水印" else "text",
                "dilate_num": int(kwargs["mask_dilate"]),
            }
            self.watermark_remove_image_instance.run(**params)
        else:
            params = {
                "input_video_path": input_path, 
                "output_video_path": output_file,
                "sr_segment_onnx_path": os.path.join(onnx_model_dir, "sr_segment.onnx"),
                "pt_segment_onnx_path": os.path.join(onnx_model_dir, "pt_segment.onnx"),
                "pt_inpaint_onnx_path": os.path.join(onnx_model_dir, "pt_inpaint.onnx"), 
                "cf_onnx_path": os.path.join(onnx_model_dir, "cf.onnx"), 
                "lama_onnx_path": os.path.join(onnx_model_dir, "lama_fp32.onnx"),
                "emdf_onnx_path": os.path.join(onnx_model_dir, "emdf_inpaint.onnx"),
                "grig_onnx_path": os.path.join(onnx_model_dir, "grig_inpaint.onnx"),
                "text_detection_onnx_path": os.path.join(onnx_model_dir, "pp_ocr_det.onnx"),
                "yolo_detection_onnx_path": os.path.join(onnx_model_dir, "yolo.onnx"),
                "mask_path": kwargs["mask_path"] if "mask_path" in kwargs and kwargs["mask_path"] else "",
                "image_refine_type": kwargs["model_name"],
                "use_cache_mask": True if "watermark_format" in kwargs and kwargs["watermark_format"] == "静态水印" else False,
                "watermark_type": "all" if "watermark_content" in kwargs and kwargs["watermark_content"] == "通用水印" else "text",
                "dilate_num": int(kwargs["mask_dilate"]),
                "ffmpeg_path": os.getenv("POWERTOOLS_FFMPEG_BIN"),
                "callback_func": progress_cb,
            }
            self.watermark_remove_video_instance.process_video(**params)
        return output_file