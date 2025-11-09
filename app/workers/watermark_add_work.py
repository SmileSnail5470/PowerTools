import os
from app.workers.work_base import BaseWorker
from app.algorithms.visible_watermark_addition import VisibleWatermarkAddition


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

    def run_algorithm(self, progress_cb, cancel_requested, *args, **kwargs):
        watermark_add_instance = VisibleWatermarkAddition()
        input_path = kwargs["input_path"]
        output_path = kwargs["output_path"]
        output_format = kwargs["output_format"]
        if output_format == "保持原格式":
            output_file = os.path.join(output_path, os.path.basename(input_path))
        else:
            output_file = os.path.join(output_path, "{0}.{1}".format(os.path.basename(input_path).split(".")[0], output_format.lower()))
        if "watermark_content" in kwargs and kwargs["watermark_content"] == "ImageSettings":
            watermark_add_instance.add_image_watermark(
                input_image_path=input_path,
                watermark_image_path=kwargs["watermark_image"],
                output_image_path=output_file,
                position = kwargs["watermark_location"],
                opacity = kwargs["watermark_opacity"] / 100,
                rotation = kwargs["watermark_rotation"],
                scale = kwargs["watermark_zoom"] / 100,
                margin = 10,
                relative_to = "watermark",
                jpeg_quality = 95,
            )
        else:
            watermark_add_instance.add_text_watermark(
                input_image_path=input_path,
                output_image_path=output_file,
                text=kwargs["watermark_text"],
                font_name=kwargs["font"],
                font_size=kwargs["font_size"],
                color=self._hex_to_rgba(kwargs["font_color"], int(255 * kwargs["watermark_opacity"] / 100)),
                position=kwargs["watermark_location"],
                rotation=kwargs["watermark_rotation"],
                scale=kwargs["watermark_zoom"] / 100,
                margin = 10,
                max_width_ratio=0.8,
                outline=True,
                outline_width=2,
                shadow=True,
                shadow_offset=(2,2),
                jpeg_quality=95,
            )
        return output_file