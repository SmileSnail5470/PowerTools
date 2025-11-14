from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
import os
from typing import Tuple, Union, Optional, List, Dict
import platform
import subprocess
from pathlib import Path
import threading
import tempfile
import ffmpeg

Position = Union[str, Tuple[int, int]]

class VisibleWatermarkAddition:
    _font_cache: Dict[Tuple[Optional[str], int], ImageFont.FreeTypeFont] = {}
    _sys_fonts_cache: Optional[List[str]] = None
    _font_cache_lock = threading.Lock()

    def __init__(self):
        ffmpeg_bin = os.getenv(
            "POWERTOOLS_FFMPEG_BIN", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "ffmpeg", "bin")
        )
        self.ffmpeg_exe = os.path.join(ffmpeg_bin, "ffmpeg.exe" if platform.system().lower() == "windows" else "ffmpeg")
        self.ffprobe_exe = os.path.join(ffmpeg_bin, "ffprobe.exe" if platform.system().lower() == "windows" else "ffprobe")

    def _find_system_fonts_once(self) -> List[str]:
        if self._sys_fonts_cache is not None:
            return self._sys_fonts_cache
        fonts = []
        system = platform.system()
        try:
            output = subprocess.check_output(['fc-list', ':', 'file'], text=True, stderr=subprocess.DEVNULL)
            for line in output.splitlines():
                path = line.split(':')[0].strip()
                if os.path.exists(path):
                    fonts.append(path)
        except Exception:
            pass
        dirs = []
        if system == "Windows":
            dirs = [os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")]
        elif system == "Darwin":
            dirs = ["/System/Library/Fonts", "/Library/Fonts", str(Path.home() / "Library/Fonts")]
        else:
            dirs = ["/usr/share/fonts", "/usr/local/share/fonts", str(Path.home() / ".fonts")]

        for d in dirs:
            if os.path.isdir(d):
                for root, _, files in os.walk(d):
                    for f in files:
                        p = os.path.join(root, f)
                        if p not in fonts and (f.lower().endswith(('.ttf', '.otf', '.ttc'))):
                            fonts.append(p)
        self._sys_fonts_cache = fonts
        return fonts

    def _get_font_path(self, font_name: Optional[str]) -> Optional[str]:
        if font_name:
            target = font_name.lower()
        else:
            target = None
        fonts = self._find_system_fonts_once()
        if target:
            for p in fonts:
                if not p.endswith((".ttf", ".otf", ".ttc")):
                    continue
                if target in os.path.basename(p).lower() or target in p.lower() or target.split(" ")[0] in p.lower():
                    return p
            if platform.system() == "Windows":
                try:
                    import winreg
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts") as key:
                        for i in range(winreg.QueryInfoKey(key)[1]):
                            name, path, _ = winreg.EnumValue(key, i)
                            if target in name.lower() or target in path.lower():
                                candidate = os.path.join(os.environ['WINDIR'], 'Fonts', path)
                                if os.path.exists(candidate):
                                    return candidate
                except Exception:
                    pass
        fallback_names = [
            "DejaVuSans.ttf", "Arial.ttf", "NotoSansCJK-Regular.ttc", "PingFang.ttc", "NotoSans-Regular.ttf"
        ]
        for fname in fallback_names:
            for p in fonts:
                if os.path.basename(p).lower().startswith(fname.lower()):
                    return p
        for p in fonts:
            return p
        return None

    def _load_font(self, font_name: Optional[str], size: int) -> ImageFont.FreeTypeFont:
        key = (font_name, size)
        with self._font_cache_lock:
            if key in self._font_cache:
                return self._font_cache[key]
            path = self._get_font_path(font_name)
            font_obj = None
            try:
                if path and os.path.exists(path):
                    font_obj = ImageFont.truetype(path, size)
                else:
                    path2 = self._get_font_path("DejaVuSans")
                    if path2 and os.path.exists(path2):
                        font_obj = ImageFont.truetype(path2, size)
            except Exception:
                font_obj = None

            if font_obj is None:
                font_obj = ImageFont.load_default()
            self._font_cache[key] = font_obj
            return font_obj

    def _calc_position(self, base_size: Tuple[int,int], overlay_size: Tuple[int,int], position: Position, margin=10) -> Tuple[int,int]:
        bw, bh = base_size
        ow, oh = overlay_size

        if isinstance(position, tuple):
            return int(position[0]), int(position[1])

        pos = (position or "bottom-right").lower()
        x = y = margin

        if 'left' in pos:
            x = margin
        elif 'right' in pos:
            x = bw - ow - margin
        else:
            x = (bw - ow) // 2

        if 'top' in pos:
            y = margin
        elif 'bottom' in pos:
            y = bh - oh - margin
        else:
            y = (bh - oh) // 2

        return int(x), int(y)

    def _wrap_text(self, text: str, font: ImageFont.ImageFont, draw: ImageDraw.ImageDraw, max_w: int) -> list:
        if not text:
            return ['']

        if ' ' in text.strip():
            words = text.split(' ')
            lines = []
            cur = ""
            for w in words:
                test = (cur + (" " if cur else "") + w).strip()
                bbox = draw.textbbox((0,0), test, font=font)
                tw = bbox[2] - bbox[0]
                if tw <= max_w or cur == "":
                    cur = test
                else:
                    lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
            return lines
        else:
            lines = []
            cur = ""
            for ch in text:
                test = cur + ch
                bbox = draw.textbbox((0,0), test, font=font)
                if bbox[2] - bbox[0] <= max_w or cur == "":
                    cur = test
                else:
                    lines.append(cur)
                    cur = ch
            if cur:
                lines.append(cur)
            return lines

    def image_add_text_watermark(
        self,
        input_image_path: str,
        output_image_path: str,
        text: str,
        font_name: str = None,
        font_size: int = 36,
        color: Tuple[int,int,int,int] = (255,255,255,180),
        position: Position = "bottom-right",
        rotation: float = 0.0,
        scale: float = 1.0,
        margin: int = 10,
        max_width_ratio: float = 0.8,
        outline: bool = True,
        outline_width: int = 2,
        shadow: bool = True,
        shadow_offset: Tuple[int,int] = (2,2),
        jpeg_quality: int = 100,
    ):
        im = Image.open(input_image_path)
        im = ImageOps.exif_transpose(im).convert("RGBA")
        iw, ih = im.size

        ref = max(1, min(iw, ih) / 512.0)
        size_px = max(6, int(font_size * scale * ref))

        font = self._load_font(font_name, size_px)

        tmp = Image.new("RGBA", (1, 1), (0,0,0,0))
        draw_tmp = ImageDraw.Draw(tmp)

        max_w = int(iw * max_width_ratio)

        lines = self._wrap_text(text, font, draw_tmp, max_w)

        sizes = [draw_tmp.textbbox((0,0), line, font=font) for line in lines]
        widths = [s[2]-s[0] for s in sizes] if sizes else [0]
        heights = [s[3]-s[1] for s in sizes] if sizes else [0]
        max_line_w = max(widths) if widths else 0
        spacing = int(0.18 * size_px)

        total_h = sum(heights) + (len(lines)-1) * spacing

        extra_w = outline_width*2 + abs(shadow_offset[0])
        extra_h = outline_width*2 + abs(shadow_offset[1])
        tw = max(1, max_line_w + extra_w)
        th = max(1, total_h + extra_h)
        text_img = Image.new("RGBA", (tw, th), (255,255,255,0))
        td = ImageDraw.Draw(text_img)

        y = outline_width
        for i, line in enumerate(lines):
            bbox = draw_tmp.textbbox((0, 0), line, font=font)
            baseline_correction = -bbox[1]
            x = outline_width
            if shadow:
                td.text(
                    (x+shadow_offset[0], y+shadow_offset[1]+baseline_correction),
                    line, font=font,
                    fill=(0,0,0,int(color[3]*0.6))
                )
            td.text(
                (x, y+baseline_correction),
                line, font=font,
                fill=color,
                stroke_width=outline_width if outline else 0,
                stroke_fill=(0,0,0,int(color[3]*0.85)) if outline else None
            )

            y += heights[i] + spacing

        if rotation:
            text_img = text_img.rotate(rotation + 360, expand=True)

        tx, ty = self._calc_position((iw, ih), text_img.size, position, margin=margin)

        layer = Image.new("RGBA", im.size, (255,255,255,0))
        layer.paste(text_img, (tx, ty), text_img)
        try:
            out = Image.alpha_composite(im, layer)
        except ValueError:
            out = im.copy()
            out.paste(layer, (0,0), layer)

        out_dir = os.path.dirname(output_image_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        exif_data = im.info.get("exif")
        pnginfo = im.info.get("pnginfo")

        if output_image_path.lower().endswith(('.jpg', '.jpeg')):
            bg = Image.new("RGB", out.size, (255, 255, 255))
            bg.paste(out, mask=out.split()[3])
            if exif_data:
                bg.save(output_image_path, "JPEG", quality=jpeg_quality, optimize=True, exif=exif_data)
            else:
                bg.save(output_image_path, "JPEG", quality=jpeg_quality, optimize=True)
        else:
            if pnginfo:
                out.save(output_image_path, pnginfo=pnginfo)
            elif exif_data:
                out.save(output_image_path, exif=exif_data)
            else:
                out.save(output_image_path)

    def image_add_image_watermark(
        self,
        input_image_path: str,
        watermark_image_path: str,
        output_image_path: str,
        position: Position = "bottom-right",
        opacity: float = 0.5,
        rotation: float = 0.0,
        scale: float = 1.0,
        margin: int = 10,
        relative_to: str = "watermark",
        jpeg_quality: int = 100,
    ):
        base = Image.open(input_image_path)
        base = ImageOps.exif_transpose(base).convert("RGBA")
        wm = Image.open(watermark_image_path).convert("RGBA")
        bw, bh = base.size
        ww, wh = wm.size

        if relative_to == "base":
            target_w = max(1, int(bw * scale))
            if ww:
                target_h = max(1, int((target_w / ww) * wh))
            else:
                target_h = wh
            wm = wm.resize((target_w, target_h), Image.LANCZOS)
        else:
            target_w = max(1, int(ww * scale))
            target_h = max(1, int(wh * scale))
            wm = wm.resize((target_w, target_h), Image.LANCZOS)

        if rotation:
            wm = wm.rotate(rotation + 360, expand=True)

        opacity = max(0.0, min(1.0, opacity))
        if opacity < 1.0:
            alpha = wm.split()[-1]
            alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
            wm.putalpha(alpha)

        x, y = self._calc_position((bw, bh), wm.size, position, margin=margin)
        layer = Image.new("RGBA", base.size, (255,255,255,0))
        layer.paste(wm, (x, y), wm)
        out = Image.alpha_composite(base, layer)

        out_dir = os.path.dirname(output_image_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        exif_data = base.info.get("exif")
        pnginfo = base.info.get("pnginfo")

        if output_image_path.lower().endswith(('.jpg', '.jpeg')):
            bg = Image.new("RGB", out.size, (255, 255, 255))
            bg.paste(out, mask=out.split()[3])
            if exif_data:
                bg.save(output_image_path, "JPEG", quality=jpeg_quality, optimize=True, exif=exif_data)
            else:
                bg.save(output_image_path, "JPEG", quality=jpeg_quality, optimize=True)
        else:
            if pnginfo:
                out.save(output_image_path, pnginfo=pnginfo)
            elif exif_data:
                out.save(output_image_path, exif=exif_data)
            else:
                out.save(output_image_path)

    def video_add_text_watermark(
        self,
        input_video_path: str,
        output_video_path: str,
        text: str,
        font_name: Optional[str] = None,
        font_size: int = 36,
        scale: float = 1.0,
        color: Tuple[int, int, int, float] = (255, 255, 255, 0.8),
        position: Position = "bottom-right",
        margin: int = 20,
        rotation: float = 0.0,
        shadow: bool = True,
        shadow_offset: Tuple[int, int] = (2, 2),
        hardware_accel: bool = True,
        codec: str = "libx264",
        crf: int = 18,
    ):
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        font_path = self._get_font_path(font_name)
        if not font_path:
            raise RuntimeError("Cannot find system font")
        
        # 适配 windows 路径
        font_path = font_path.replace("\\", "/").replace(":", "\\:")
        input_video_path = os.path.abspath(input_video_path).replace("\\", "/")
        output_video_path = os.path.abspath(output_video_path).replace("\\", "/")


        rgba = color
        r, g, b, a = rgba
        alpha = max(0.0, min(a, 1.0)) if isinstance(a, float) else (a / 255.0)

        position_expr = self._get_ffmpeg_position_expr(position, margin, is_text=True)

        font_size = max(6, int(font_size * scale))

        fontcolor_expr = f"#{r:02x}{g:02x}{b:02x}@{alpha}"

        drawtext_kwargs = dict(
            text=text,
            fontfile=font_path,
            fontsize=font_size,
            fontcolor=fontcolor_expr,
            x=position_expr[0],
            y=position_expr[1],
        )
        if shadow:
            drawtext_kwargs.update(
                shadowcolor="black",
                shadowx=shadow_offset[0],
                shadowy=shadow_offset[1]
            )

        stream = ffmpeg.input(input_video_path)
        stream = stream.drawtext(**drawtext_kwargs)

        if rotation:
            stream = stream.filter("rotate", f"{rotation}*PI/180", ow="rotw(iw)", oh="roth(ih)", c="none")

        stream = ffmpeg.output(
            stream,
            output_video_path,
            vcodec=codec,
            crf=crf,
            acodec="copy",
            movflags="+faststart"
        )

        run_kwargs = {"cmd": self.ffmpeg_exe}
        run_kwargs["capture_stdout"] = True
        run_kwargs["capture_stderr"] = True
        if hardware_accel and platform.system() != "Windows":
            run_kwargs["cmd"] = [self.ffmpeg_exe, "-hwaccel", "auto"]

        ffmpeg.run(stream, overwrite_output=True, **run_kwargs)

    def video_add_image_watermark(
        self,
        input_video_path: str,
        watermark_image_path: str,
        output_video_path: str,
        position: Position = "bottom-right",
        opacity: float = 0.5,
        scale: float = 1.0,
        rotation: float = 0.0,
        margin: int = 20,
        codec: str = "libx264",
        crf: int = 18,
        hardware_accel: bool = True
    ):
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")
        if not os.path.exists(watermark_image_path):
            raise FileNotFoundError(f"Watermark image not found: {watermark_image_path}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_wm = os.path.join(tmp_dir, "wm.png")

            wm = Image.open(watermark_image_path).convert("RGBA")
            if rotation:
                wm = wm.rotate(rotation + 360, expand=True)
            if scale != 1.0:
                w, h = wm.size
                wm = wm.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            if opacity < 1.0:
                alpha = wm.split()[-1]
                alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
                wm.putalpha(alpha)
            wm.save(tmp_wm)

            # 计算 overlay 位置表达式
            x_expr, y_expr = self._get_ffmpeg_position_expr(position, margin, is_text=False)

            video = ffmpeg.input(input_video_path)
            watermark = ffmpeg.input(tmp_wm)
            stream = ffmpeg.overlay(video, watermark, x=x_expr, y=y_expr)

            stream = ffmpeg.output(
                stream,
                output_video_path,
                vcodec=codec,
                crf=crf,
                acodec="copy",
                movflags="+faststart"
            )

            run_kwargs = {"cmd": self.ffmpeg_exe}
            run_kwargs["capture_stdout"] = True
            run_kwargs["capture_stderr"] = True
            if hardware_accel and platform.system() != "Windows":
                run_kwargs["cmd"] = [self.ffmpeg_exe, "-hwaccel", "auto"]

            ffmpeg.run(stream, overwrite_output=True, **run_kwargs)

    def _get_ffmpeg_position_expr(self, position: Position, margin: int, is_text: bool = False) -> Tuple[str, str]:
        if isinstance(position, tuple):
            return str(position[0]), str(position[1])
        pos = position.lower()
        w_var = "text_w" if is_text else "overlay_w"
        h_var = "text_h" if is_text else "overlay_h"

        if "left" in pos:
            x = f"{margin}"
        elif "right" in pos:
            x = f"main_w-{w_var}-{margin}"
        else:
            x = f"(main_w-{w_var})/2"

        if "top" in pos:
            y = f"{margin}"
        elif "bottom" in pos:
            y = f"main_h-{h_var}-{margin}"
        else:
            y = f"(main_h-{h_var})/2"

        return x, y