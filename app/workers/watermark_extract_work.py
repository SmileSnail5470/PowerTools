import os
import logging
import platform
from PIL import Image, ImageDraw, ImageFont
import app.utils.ffmpeg as ffmpeg
from app.ui.common.config import cfg
from app.workers.work_base import BaseWorker
from app.utils.logger.decorators import log_exception

from app.algorithms.blind_watermark_addition.blind_watermark_addition_image import ImageBlindWatermarkDetect
from app.algorithms.blind_watermark_addition.blind_watermark_addition_video import VideoBlindWatermarkDetect


class WatermarkExtractWork(BaseWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_path = cfg.get(cfg.cachePath)
        self.deps_path = cfg.get(cfg.localAIModelDeps)
        self.blind_watermark_extract_image_instance = ImageBlindWatermarkDetect()
        self.blind_watermark_extract_video_instance = VideoBlindWatermarkDetect()

    def _text_to_image(self, input_path, text):
        w, h = Image.open(input_path).size
        bg_color = (24, 24, 24)
        text_color = (255, 255, 255)

        img = Image.new("RGB", (w, h), bg_color)
        draw = ImageDraw.Draw(img)

        font_size = max(64, min(w, h) // 10)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default(font_size)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        x = (w - text_w) // 2
        y = (h - text_h) // 2

        draw.text(
            (x, y),
            text,
            fill=text_color,
            font=font
        )

        output_dir = os.path.join(self.cache_path, "blind_watermark_extract")
        os.makedirs(output_dir, exist_ok=True)
        base, ext = os.path.basename(input_path).split(".")
        output_path = os.path.join(output_dir, "{0}_wm_extract.{1}".format(base, ext))
        img.save(output_path)
        return output_path

    def _text_to_video(self, input_path, text):
        ffmpeg_exe = os.path.join(os.getenv("POWERTOOLS_FFMPEG_BIN"), "ffmpeg.exe" if platform.system().lower() == "windows" else "ffmpeg")
        ffprobe_exe = os.path.join(os.getenv("POWERTOOLS_FFMPEG_BIN"), "ffprobe.exe" if platform.system().lower() == "windows" else "ffprobe")

        probe = ffmpeg.probe(input_path, cmd=ffprobe_exe)
        vinfo = next(s for s in probe["streams"] if s["codec_type"] == "video")

        width = int(vinfo["width"])
        height = int(vinfo["height"])
        duration = float(vinfo.get("duration", probe["format"]["duration"]))

        output_dir = os.path.join(self.cache_path, "blind_watermark_extract")
        os.makedirs(output_dir, exist_ok=True)
        base, ext = os.path.basename(input_path).split(".")
        output_path = os.path.join(output_dir, "{0}_wm_extract.{1}".format(base, ext))

        (
            ffmpeg
            .input(
                f"color=c=black:s={width}x{height}:d={duration}",
                f='lavfi'
            )
            .drawtext(
                text=text,
                fontcolor='white',
                fontsize=max(64, min(width, height) // 8),
                x='(w-text_w)/2',
                y='(h-text_h)/2'
            )
            .output(
                output_path,
                vcodec='libx264',
                pix_fmt='yuv420p',
                an=None
            )
            .overwrite_output()
            .run(quiet=True, cmd=ffmpeg_exe)
        )
        return output_path

    @log_exception(logger=logging.getLogger('WatermarkExtract'), reraise=True, log_args=True, log_result=True)
    def run_algorithm(self, progress_cb, cancel_requested, *args, **kwargs):
        input_path = kwargs["input_path"]
        file_type = self.file_type(input_file=input_path)
        if file_type is None:
            raise Exception("Not support file {0}".format(input_path.split(".")[-1]))
        if "watermark_type" not in kwargs or kwargs["watermark_type"] != "blind":
            raise Exception("Error watermark type value, watermark type must be blind")
        if "blind_watermark_task_type" not in kwargs or kwargs["blind_watermark_task_type"] != "extract_blind_watermark":
            raise Exception("Error blind watermark task type, must be extract_blind_watermark")
        if file_type == "image":
            params = {
                "input_image_path": input_path
            }
            self.blind_watermark_extract_image_instance.prepare(
                onnx_path=os.path.join(self.deps_path, "blind_watermark_addition", "pixelseal_image_detect.onnx")
            )
            text = self.blind_watermark_extract_image_instance.watermark_extraction(**params)["preds"]
            output_file = self._text_to_image(input_path=input_path, text=text)
        else:
            params = {
                "input_path": input_path,
                "chunk_size": 8
            }
            self.blind_watermark_extract_video_instance.prepare(
                onnx_path=os.path.join(self.deps_path, "blind_watermark_addition", "pixelseal_video_detect.onnx"),
                ffmpeg_path=os.getenv("POWERTOOLS_FFMPEG_BIN")
            )
            text = self.blind_watermark_extract_video_instance.watermark_extraction(**params)["preds"]
            output_file = self._text_to_video(input_path=input_path, text=text)
        return output_file