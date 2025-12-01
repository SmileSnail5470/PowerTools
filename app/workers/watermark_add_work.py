import os
import logging
from app.workers.work_base import BaseWorker
from app.workers.algorithm_manager import global_algorithm_manager
from app.utils.logger.decorators import log_exception


class WatermarkAddWork(BaseWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _hex_to_rgba(self, hex_color: str, alpha: int = 255):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            raise ValueError("hex_color must be #RRGGBB style")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b, alpha)

    @log_exception(logger=logging.getLogger('WatermarkAdd'), reraise=False, log_args=True, log_result=True)
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
        
        is_visible_watermark = True if kwargs["watermark_type"] == "visible" else False
        if is_visible_watermark and kwargs["watermark_content"] == "ImageSettings":
            if file_type == "image":
                watermark_add_instance, metadata = global_algorithm_manager.create_instance(name="visible_watermark", capability="image.watermark_img.visible")
                params = {
                    "input_image_path": input_path,
                    "watermark_image_path": kwargs["watermark_image"],
                    "output_image_path": output_file,
                    "position": kwargs["watermark_location"],
                    "opacity": kwargs["watermark_opacity"] / 100,
                    "rotation": kwargs["watermark_rotation"],
                    "scale": kwargs["watermark_zoom"] / 100,
                    "margin": 10,
                    "relative_to": "watermark",
                    "jpeg_quality": 95,
                }
                self.call_algorithm(instance=watermark_add_instance, method_metadata=metadata, input_args=params)
            else:
                watermark_add_instance, metadata = global_algorithm_manager.create_instance(name="visible_watermark", capability="video.watermark_img.visible")
                params = {
                    "input_video_path": input_path,
                    "watermark_image_path": kwargs["watermark_image"],
                    "output_video_path": output_file,
                    "position": kwargs["watermark_location"],
                    "opacity": kwargs["watermark_opacity"] / 100,
                    "scale": kwargs["watermark_zoom"] / 100,
                    "rotation": kwargs["watermark_rotation"],
                    "margin": 10,
                    "codec": "libx264",
                    "crf": 18,
                    "hardware_accel": True
                }
                self.call_algorithm(instance=watermark_add_instance, method_metadata=metadata, input_args=params)
        elif is_visible_watermark and kwargs["watermark_content"] == "TextSettings":
            if file_type == "image":
                watermark_add_instance, metadata = global_algorithm_manager.create_instance(name="visible_watermark", capability="image.watermark_text.visible")
                params = {
                    "input_image_path": input_path,
                    "output_image_path": output_file,
                    "text": kwargs["watermark_text"],
                    "font_name": kwargs["font"],
                    "font_size": kwargs["font_size"],
                    "color": self._hex_to_rgba(kwargs["font_color"], int(255 * kwargs["watermark_opacity"] / 100)),
                    "position": kwargs["watermark_location"],
                    "rotation": kwargs["watermark_rotation"],
                    "scale": kwargs["watermark_zoom"] / 100,
                    "margin": 10,
                    "max_width_ratio": 0.8,
                    "outline": True,
                    "outline_width": 2,
                    "shadow": True,
                    "shadow_offset": (2,2),
                    "jpeg_quality": 95
                }
                self.call_algorithm(instance=watermark_add_instance, method_metadata=metadata, input_args=params)
            else:
                watermark_add_instance, metadata = global_algorithm_manager.create_instance(name="visible_watermark", capability="video.watermark_text.visible")
                params = {
                    "input_video_path": input_path,
                    "output_video_path": output_file,
                    "text": kwargs["watermark_text"],
                    "font_name": kwargs["font"],
                    "font_size": kwargs["font_size"],
                    "scale": kwargs["watermark_zoom"] / 100,
                    "color": self._hex_to_rgba(kwargs["font_color"], int(255 * kwargs["watermark_opacity"] / 100)),
                    "position": kwargs["watermark_location"],
                    "margin": 10,
                    "rotation": kwargs["watermark_rotation"],
                    "shadow": True,
                    "shadow_offset": (1, 1),
                    "hardware_accel": True,
                    "codec": "libx264",
                    "crf": 18
                }
                self.call_algorithm(instance=watermark_add_instance, method_metadata=metadata, input_args=params)

        elif not is_visible_watermark:
            if file_type == "image":
                watermark_add_instance, metadata = global_algorithm_manager.create_instance(name="blind_watermark_addition", capability="image.watermark.blind.addition")
                params = {
                    "input_image_path": input_path,
                    "output_file": output_file,
                    "message": kwargs["watermark_text"]
                }
                self.call_algorithm(instance=watermark_add_instance, method_metadata=metadata, input_args=params)
            else:
                watermark_add_instance, metadata = global_algorithm_manager.create_instance(name="blind_watermark_addition", capability="video.watermark.blind.addition")
                params = {
                    "input_path": input_path,
                    "output_file": output_file,
                    "message": kwargs["watermark_text"],
                    "chunk_size": 8
                }
                self.call_algorithm(instance=watermark_add_instance, method_metadata=metadata, input_args=params)
        return output_file