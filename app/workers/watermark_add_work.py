import os
import logging
from dataclasses import dataclass, field
from app.ui.common.config import cfg
from app.workers.work_base import BaseWorker, _resolve_hardware_variant
from app.utils.logger.decorators import log_exception

from app.algorithms.visible_watermark_addition.watermark_addition import VisibleWatermarkAddition
from app.algorithms.blind_watermark_addition.blind_watermark_addition_image import ImageBlindWatermarkEmbed
from app.algorithms.blind_watermark_addition.blind_watermark_addition_video import VideoBlindWatermarkEmbed


@dataclass
class _WatermarkAddIOParams:
    input_path: str = ""
    output_path: str = ""
    output_format: str = "保持原格式"


@dataclass
class _WatermarkAddTypeParams:
    watermark_type: str = "visible"
    watermark_content: str = "TextSettings"
    blind_watermark_task_type: str = "add_blind_watermark"


@dataclass
class _WatermarkAddVisibleParams:
    watermark_text: str = ""
    watermark_image: str = ""
    font: str = ""
    font_size: int = 15
    font_color: str = "#000000"
    watermark_location: str = "center"
    watermark_rotation: int = 0
    watermark_zoom: int = 100
    watermark_opacity: int = 70


@dataclass
class _WatermarkAddBlindParams:
    blind_watermark_model_name: str = ""
    watermark_text: str = "POWERTOOLS"
    custom_characters: list = field(default_factory=lambda: list("ABCDEFGHIJKLMNOPQRSTUVWXYZ,1234"))


@dataclass
class WatermarkAddUIParams:
    io: _WatermarkAddIOParams = field(default_factory=_WatermarkAddIOParams)
    type: _WatermarkAddTypeParams = field(default_factory=_WatermarkAddTypeParams)
    visible: _WatermarkAddVisibleParams = field(default_factory=_WatermarkAddVisibleParams)
    blind: _WatermarkAddBlindParams = field(default_factory=_WatermarkAddBlindParams)
    _feature_name_: str = ""

    def validate(self) -> str:
        if not self.io.input_path:
            return "请选择要处理的文件或目录"
        if self.type.blind_watermark_task_type == "extract_blind_watermark":
            # 提取任务只需要 model
            if not self.blind.blind_watermark_model_name or self.blind.blind_watermark_model_name == "placeholder_model":
                return "请选择盲水印算法"
            return ""
        if self.type.watermark_type == "visible":
            if not self.io.output_path:
                return "请设置文件保存位置"
            if self.type.watermark_content == "ImageSettings" and not self.visible.watermark_image:
                return "请选择水印图片"
            if self.type.watermark_content == "TextSettings" and not self.visible.watermark_text:
                return "请设置水印文本"
        else:
            if not self.blind.watermark_text:
                return "请设置水印文本"
            if not self.blind.blind_watermark_model_name or self.blind.blind_watermark_model_name == "placeholder_model":
                return "请选择盲水印算法"
            if not self.io.output_path:
                return "请设置文件保存位置"
        return ""


class WatermarkAddWork(BaseWorker):
    _visible_instance = None
    _blind_image_instance = None
    _blind_video_instance = None

    @classmethod
    def _get_visible_instance(cls):
        if cls._visible_instance is None:
            cls._visible_instance = VisibleWatermarkAddition()
        return cls._visible_instance

    @classmethod
    def _get_blind_image_instance(cls):
        if cls._blind_image_instance is None:
            cls._blind_image_instance = ImageBlindWatermarkEmbed()
        return cls._blind_image_instance

    @classmethod
    def _get_blind_video_instance(cls):
        if cls._blind_video_instance is None:
            cls._blind_video_instance = VideoBlindWatermarkEmbed()
        return cls._blind_video_instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deps_path = cfg.get(cfg.localAIModelDeps)

    def _hex_to_rgba(self, hex_color: str, alpha: int = 255):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            raise ValueError("hex_color must be #RRGGBB style")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b, alpha)

    @log_exception(logger=logging.getLogger('WatermarkAdd'), reraise=True, log_args=True, log_result=True)
    def run_algorithm(self, progress_cb, cancel_requested, *args, **kwargs):
        p: WatermarkAddUIParams = args[0].params
        input_path = p.io.input_path
        output_path = p.io.output_path
        output_format = p.io.output_format
        base_name, source_ext = os.path.splitext(os.path.basename(input_path))
        output_ext = source_ext.lstrip(".") if output_format == "保持原格式" else output_format.lower()
        output_file = os.path.join(output_path, f"{base_name}_wm.{output_ext}")
        file_type = self.file_type(input_file=input_path)

        if file_type is None:
            raise Exception("Not support file {0}".format(source_ext.lstrip(".")))

        is_visible_watermark = (p.type.watermark_type == "visible")
        if progress_cb is not None:
            progress_cb("VisibleWatermarkAddStart", "")
        if is_visible_watermark and p.type.watermark_content == "ImageSettings":
            if file_type == "image":
                watermark_params = {
                    "input_image_path": input_path,
                    "watermark_image_path": p.visible.watermark_image,
                    "output_image_path": output_file,
                    "position": p.visible.watermark_location,
                    "opacity": p.visible.watermark_opacity / 100,
                    "rotation": p.visible.watermark_rotation,
                    "scale": p.visible.watermark_zoom / 100,
                    "margin": 10,
                    "relative_to": "watermark",
                    "jpeg_quality": 95,
                }
                self._get_visible_instance().image_add_image_watermark(**watermark_params)
            else:
                watermark_params = {
                    "input_video_path": input_path,
                    "watermark_image_path": p.visible.watermark_image,
                    "output_video_path": output_file,
                    "position": p.visible.watermark_location,
                    "opacity": p.visible.watermark_opacity / 100,
                    "scale": p.visible.watermark_zoom / 100,
                    "rotation": p.visible.watermark_rotation,
                    "margin": 10,
                    "codec": "libx264",
                    "crf": 18,
                    "hardware_accel": True
                }
                self._get_visible_instance().video_add_image_watermark(**watermark_params)
        elif is_visible_watermark and p.type.watermark_content == "TextSettings":
            if file_type == "image":
                watermark_params = {
                    "input_image_path": input_path,
                    "output_image_path": output_file,
                    "text": p.visible.watermark_text,
                    "font_name": p.visible.font,
                    "font_size": p.visible.font_size,
                    "color": self._hex_to_rgba(p.visible.font_color, int(255 * p.visible.watermark_opacity / 100)),
                    "position": p.visible.watermark_location,
                    "rotation": p.visible.watermark_rotation,
                    "scale": p.visible.watermark_zoom / 100,
                    "margin": 10,
                    "max_width_ratio": 0.8,
                    "outline": True,
                    "outline_width": 2,
                    "shadow": False,
                    "shadow_offset": (2, 2),
                    "jpeg_quality": 95
                }
                self._get_visible_instance().image_add_text_watermark(**watermark_params)
            else:
                watermark_params = {
                    "input_video_path": input_path,
                    "output_video_path": output_file,
                    "text": p.visible.watermark_text,
                    "font_name": p.visible.font,
                    "font_size": p.visible.font_size,
                    "scale": p.visible.watermark_zoom / 100,
                    "color": self._hex_to_rgba(p.visible.font_color, int(255 * p.visible.watermark_opacity / 100)),
                    "position": p.visible.watermark_location,
                    "margin": 10,
                    "rotation": p.visible.watermark_rotation,
                    "shadow": False,
                    "shadow_offset": (1, 1),
                    "hardware_accel": True,
                    "codec": "libx264",
                    "crf": 18
                }
                self._get_visible_instance().video_add_text_watermark(**watermark_params)
        elif not is_visible_watermark:
            if p._feature_name_:
                os.environ["_feature_name_"] = p._feature_name_
            custom_characters = p.blind.custom_characters or list("ABCDEFGHIJKLMNOPQRSTUVWXYZ,1234")
            os.environ["BLIND_WATERMARK_CHARSET"] = "".join(custom_characters) + "`"
            if file_type == "image":
                blind_params = {
                    "input_image_path": input_path,
                    "output_file": output_file,
                    "message": p.blind.watermark_text
                }
                self._get_blind_image_instance().prepare(
                    onnx_path=os.path.join(self.deps_path, _resolve_hardware_variant(), "blind_watermark_addition", "{0}_image_embed.encmodel".format(p.blind.blind_watermark_model_name))
                )
                self._get_blind_image_instance().watermark_addition(**blind_params)
            else:
                blind_params = {
                    "input_path": input_path,
                    "output_file": output_file,
                    "message": p.blind.watermark_text,
                    "chunk_size": 8
                }
                self._get_blind_video_instance().prepare(
                    onnx_path=os.path.join(self.deps_path, _resolve_hardware_variant(), "blind_watermark_addition", "{0}_video_embed.encmodel".format(p.blind.blind_watermark_model_name)),
                    ffmpeg_path=os.getenv("POWERTOOLS_FFMPEG_BIN")
                )
                self._get_blind_video_instance().watermark_addition(**blind_params)
        if progress_cb is not None:
            progress_cb("VisibleWatermarkAddCompleted", "")
        return output_file