import os
import cv2
import json
import math
import shutil
from typing import Optional
from dataclasses import dataclass, field
import app.utils.ffmpeg as ffmpeg
from PySide6.QtCore import Qt, QRectF, QPointF, QSize, Signal, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QPixmap, QImage, QPainterPath, QCursor, QFontMetrics
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QFrame, QSizePolicy, QButtonGroup,
    QScrollArea, QGridLayout,
    QGraphicsDropShadowEffect, QSlider, QMessageBox, QProgressDialog,
    QApplication, QFileDialog, QLineEdit,
)
from app.ui.library.qfluentwidgets import CardWidget, FlowLayout, MaskDialogBase, PrimaryPushButton, FluentIcon, Theme, MessageBox, setFont
from app.ui.common.config import cfg


C_BG          = QColor("#F4F6F9")
C_BRAND       = QColor("#0078D4")
C_BRAND_LIGHT = QColor("#E0EFFE")
C_BORDER      = QColor("#E2E8F0")
C_BORDER_HV   = QColor("#CBD5E1")
C_TEXT        = QColor("#0F172A")
C_MUTED       = QColor("#64748B")
C_SEL_BG      = QColor(0, 120, 212, 45)
C_HOVER_BG    = QColor(0, 120, 212, 18)
C_UNSEL_MASK  = QColor(15, 23, 42, 110)


@dataclass
class SplitRegion:
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    output_path: Optional[str] = None

    def __repr__(self):
        return f"SplitRegion(row={self.row}, col={self.col}, x={self.x}, y={self.y}, w={self.width}, h={self.height})"


@dataclass
class SplitResult:
    rows: int = 1
    cols: int = 1
    regions: list[SplitRegion] = field(default_factory=list)
    media_width: int = 0
    media_height: int = 0
    source_path: Optional[str] = None
    fps: float = 0.0
    total_frames: int = 0
    export_dir: Optional[str] = None
    metadata_path: Optional[str] = None


class _OpaqueCard(CardWidget):
    def _normalBackgroundColor(self):
        return QColor("#FFFFFF")

    def _hoverBackgroundColor(self):
        return self._normalBackgroundColor()

    def _pressedBackgroundColor(self):
        return self._normalBackgroundColor()


def _suggest_grid(w: int, h: int, is_video: bool) -> tuple[int, int]:
    if is_video:
        if w <= 1080:
            return 1, 1
        cols = math.ceil(w / 1080)
        rows = max(1, math.ceil(h / (w / cols)))
    else:
        if w <= 1536 and h <= 1536:
            return 1, 1
        cols = math.ceil(w / 1536)
        rows = math.ceil(h / 1536)
    return max(1, rows), max(1, cols)


def _is_video_path(file_path: str) -> bool:
    return os.path.splitext(file_path)[-1].lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def write_split_metadata(output_dir: str, result: "SplitResult") -> str:
    metadata = {
        "version": 1,
        "source_path": result.source_path,
        "export_dir": output_dir,
        "is_video": _is_video_path(result.source_path),
        "rows": result.rows,
        "cols": result.cols,
        "media_width": result.media_width,
        "media_height": result.media_height,
        "fps": result.fps,
        "total_frames": result.total_frames,
        "regions": [
            {
                "index": index + 1,
                "row": region.row,
                "col": region.col,
                "x": region.x,
                "y": region.y,
                "width": region.width,
                "height": region.height,
                "time_start": region.time_start,
                "time_end": region.time_end,
                "crop_output_path": region.output_path,
                "processed_output_path": None,
            }
            for index, region in enumerate(result.regions)
        ],
    }
    metadata_path = os.path.join(output_dir, "crop_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
    return metadata_path


def compose_split_outputs(metadata_path: str, processed_paths: Optional[list[str]], output_path: Optional[str] = None) -> str:
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    regions = metadata.get("regions", [])
    if len(processed_paths) != len(regions):
        raise ValueError("processed_paths 数量与裁剪区域数量不一致")
    for region, processed_path in zip(regions, processed_paths):
        region["processed_output_path"] = processed_path
    if not all(region.get("processed_output_path") for region in regions):
        raise ValueError("裁剪文件尚未处理完成")
    export_dir = metadata.get("export_dir") or os.path.dirname(metadata_path)
    source_path = metadata.get("source_path")
    stem = os.path.splitext(os.path.basename(source_path or "composed"))[0]
    if metadata.get("is_video"):
        source_ext = os.path.splitext(source_path or "")[-1] or ".mp4"
        final_path = output_path or os.path.join(export_dir, f"{stem}_composed{source_ext}")
        stream = _probe_video_stream(source_path)
        source = ffmpeg.input(source_path)
        composite = source.video.filter("setpts", "PTS-STARTPTS")
        for region in regions:
            processed = ffmpeg.input(region["processed_output_path"])
            processed_video = processed.video.filter(
                "scale",
                int(region["width"]),
                int(region["height"]),
                flags="lanczos",
            ).filter("setpts", "PTS-STARTPTS")
            composite = ffmpeg.overlay(
                composite,
                processed_video,
                x=int(region["x"]),
                y=int(region["y"]),
                shortest=1,
            )

        def build(audio_args):
            outputs = ffmpeg.output(
                composite,
                source.audio,
                final_path,
                **_video_encode_kwargs(stream),
                acodec=audio_args["acodec"],
                **({"b:a": audio_args["b:a"]} if "b:a" in audio_args else {}),
                fps_mode="passthrough",
                shortest=None,
                map_metadata="0",
                map_chapters="0",
            )
            return outputs.overwrite_output().global_args("-hide_banner", "-loglevel", "error")

        try:
            ffmpeg.run(build({"acodec": "copy"}), cmd=_ffmpeg_executable(), capture_stderr=True)
        except ffmpeg.Error:
            ffmpeg.run(build({"acodec": "aac", "b:a": "192k"}), cmd=_ffmpeg_executable(), capture_stderr=True)
    else:
        final_path = output_path or os.path.join(export_dir, f"{stem}_composed.png")
        source = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
        if source is None:
            raise RuntimeError(f"无法读取原始图片：{source_path}")
        for region in regions:
            processed = cv2.imread(region["processed_output_path"], cv2.IMREAD_UNCHANGED)
            if processed is None:
                raise RuntimeError(f"无法读取处理后的裁剪图片：{region['processed_output_path']}")
            width, height = int(region["width"]), int(region["height"])
            if processed.shape[:2] != (height, width):
                processed = cv2.resize(processed, (width, height), interpolation=cv2.INTER_LANCZOS4)
            if processed.ndim == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            if source.ndim == 2:
                source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            if processed.shape[2] != source.shape[2]:
                if source.shape[2] == 4:
                    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2BGRA) if processed.shape[2] == 3 else processed
                else:
                    processed = processed[:, :, :3]
            processed = processed.astype(source.dtype, copy=False)
            x, y = int(region["x"]), int(region["y"])
            source[y:y + height, x:x + width] = processed
        if not cv2.imwrite(final_path, source):
            raise RuntimeError(f"无法写入合成图片：{final_path}")
    metadata["final_output_path"] = final_path
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
    return final_path


def _ffmpeg_executable() -> str:
    configured = os.getenv("POWERTOOLS_FFMPEG_BIN", "")
    if configured:
        if os.path.isdir(configured):
            executable = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
            return os.path.join(configured, executable)
        return configured
    else:
        raise Exception("Not found ffmppeg from env 'POWERTOOLS_FFMPEG_BIN'")


def _ffprobe_executable() -> str:
    ffmpeg = _ffmpeg_executable()
    name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    return os.path.join(os.path.dirname(ffmpeg), name)


def _probe_video_stream(path: str) -> dict:
    try:
        streams = ffmpeg.probe(path, cmd=_ffprobe_executable(),  v="error", select_streams="v:0").get("streams", [])
        return streams[0] if streams else {}
    except Exception:
        return {}


def _video_encode_kwargs(stream: dict, fallback_pix_fmt_444: bool = False) -> dict:
    codec = "libx265" if stream.get("codec_name") == "hevc" else "libx264"
    kwargs = {
        "vcodec": codec,
        "preset": "medium",
        "crf": "18",
        "pix_fmt": "yuv444p" if fallback_pix_fmt_444 else (stream.get("pix_fmt") or "yuv420p"),
    }
    for key, option in (
        ("color_space", "colorspace"), 
        ("color_transfer", "color_trc"), 
        ("color_primaries", "color_primaries"), 
        ("color_range", "color_range")
    ):
        value = stream.get(key)
        if value and value != "unknown":
            kwargs[option] = value
    if codec == "libx265":
        kwargs["tag:v"] = "hvc1"
    return kwargs


def _prepare_export_dir(base_dir: str, source_path: str) -> str:
    stem = os.path.splitext(os.path.basename(source_path))[0]
    export_dir = os.path.join(base_dir, f"{stem}_crop")
    if os.path.lexists(export_dir):
        if os.path.isdir(export_dir) and not os.path.islink(export_dir):
            shutil.rmtree(export_dir)
        else:
            os.remove(export_dir)
    os.makedirs(export_dir, exist_ok=True)
    return export_dir


class _VideoFrameReader:
    def __init__(self, file_path: str):
        self._capture = cv2.VideoCapture(file_path)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"无法打开视频文件：{file_path}")
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.width <= 0 or self.height <= 0 or self.total_frames <= 0:
            self._capture.release()
            raise RuntimeError(f"无法读取视频分辨率或总帧数：{file_path}")
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))

    def read_frame(self, frame_index: int) -> QPixmap:
        frame_index = max(0, min(frame_index, self.total_frames - 1))
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError(f"无法读取视频第 {frame_index + 1} 帧")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        return QPixmap.fromImage(image)

    def release(self):
        self._capture.release()


def _load_preview_pixmap(file_path: str) -> tuple[QPixmap, int, int, bool]:
    pix = QPixmap(file_path)
    if pix.isNull():
        pix = QPixmap(800, 600)
        pix.fill(QColor("#f0f0f0"))
        return pix, 800, 600, False
    return pix, pix.width(), pix.height(), False


class _GridCanvas(QWidget):
    grid_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAttribute(Qt.WA_StyledBackground, False)

        self._pixmap      = QPixmap()
        self._orig_w      = 0
        self._orig_h      = 0
        self._is_video    = False
        self._video_reader: Optional[_VideoFrameReader] = None
        self._video_fps = 0.0
        self._total_frames = 0
        self._current_frame = 0
        self._rows        = 1
        self._cols        = 1
        self._selected: set[tuple[int, int]] = set()
        self._custom_mode = False
        self._custom_rects: list[QRectF] = []
        self._custom_selected: set[int] = set()
        self._scale       = 1.0
        self._offset      = QPointF(0, 0)
        self._drag_start: Optional[QPointF] = None
        self._drag_end:   Optional[QPointF] = None
        self._is_dragging = False
        self._hover_cell: Optional[tuple[int, int]] = None

    def load(self, file_path: str):
        if _is_video_path(file_path):
            self._video_reader = _VideoFrameReader(file_path)
            self._video_fps = self._video_reader.fps
            self._total_frames = self._video_reader.total_frames
            self._current_frame = 0
            pix = self._video_reader.read_frame(0)
            w, h, is_vid = self._video_reader.width, self._video_reader.height, True
        else:
            pix, w, h, is_vid = _load_preview_pixmap(file_path)
        self._pixmap   = pix
        self._orig_w   = w
        self._orig_h   = h
        self._is_video = is_vid
        self._custom_rects.clear()
        self._custom_selected.clear()
        self._fit_to_view()
        rows, cols = _suggest_grid(w, h, is_vid)
        self.set_grid(rows, cols)

    def seek_video_frame(self, frame_index: int):
        if self._video_reader is None:
            return
        self._current_frame = max(0, min(frame_index, self._total_frames - 1))
        self._pixmap = self._video_reader.read_frame(self._current_frame)
        self.update()

    def release_video(self):
        if self._video_reader is not None:
            self._video_reader.release()
            self._video_reader = None

    def set_grid(self, rows: int, cols: int):
        self._rows = max(1, rows)
        self._cols = max(1, cols)
        self._selected = {(r, c) for r in range(self._rows) for c in range(self._cols)}
        self.update()
        self.grid_changed.emit()

    def set_custom_mode(self, enabled: bool):
        self._custom_mode = enabled
        self.setCursor(QCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor))
        self.update()
        self.grid_changed.emit()

    def set_cell_selected(self, row: int, col: int, sel: bool):
        if self._custom_mode:
            if 0 <= col < len(self._custom_rects):
                if sel:
                    self._custom_selected.add(col)
                else:
                    self._custom_selected.discard(col)
            self.update()
            self.grid_changed.emit()
            return
        if sel:
            self._selected.add((row, col))
        else:
            self._selected.discard((row, col))
        self.update()
        self.grid_changed.emit()

    def select_all(self):
        if self._custom_mode:
            self._custom_selected = set(range(len(self._custom_rects)))
            self.update()
            self.grid_changed.emit()
            return
        self._selected = {(r, c) for r in range(self._rows) for c in range(self._cols)}
        self.update()
        self.grid_changed.emit()

    def deselect_all(self):
        if self._custom_mode:
            self._custom_selected.clear()
            self.update()
            self.grid_changed.emit()
            return
        self._selected.clear()
        self.update()
        self.grid_changed.emit()

    def get_split_result(self) -> SplitResult:
        if self._custom_mode:
            result = SplitResult(
                rows=1,
                cols=max(1, len(self._custom_rects)),
                media_width=self._orig_w, 
                media_height=self._orig_h,
                fps=self._video_fps, 
                total_frames=self._total_frames
            )
            for index, region in enumerate(self.custom_regions):
                if index in self._custom_selected:
                    result.regions.append(region)
            return result
        result = SplitResult(
            rows=self._rows, 
            cols=self._cols,
            media_width=self._orig_w, 
            media_height=self._orig_h,
            fps=self._video_fps, 
            total_frames=self._total_frames
        )
        for r in range(self._rows):
            for c in range(self._cols):
                if (r, c) not in self._selected:
                    continue
                x  = round(c * self._orig_w / self._cols)
                y  = round(r * self._orig_h / self._rows)
                rw = round((c + 1) * self._orig_w / self._cols) - x
                rh = round((r + 1) * self._orig_h / self._rows) - y
                result.regions.append(SplitRegion(row=r, col=c, x=x, y=y, width=rw, height=rh))
        return result

    def get_cell_thumbnail(self, row: int, col: int, size: int = 90) -> QPixmap:
        if self._pixmap.isNull() or self._orig_w == 0:
            return QPixmap()
        x  = round(col * self._orig_w / self._cols)
        y  = round(row * self._orig_h / self._rows)
        rw = round((col + 1) * self._orig_w / self._cols) - x
        rh = round((row + 1) * self._orig_h / self._rows) - y
        pw, ph = self._pixmap.width(), self._pixmap.height()
        sx = int(x * pw / self._orig_w)
        sy = int(y * ph / self._orig_h)
        sw = int(rw * pw / self._orig_w)
        sh = int(rh * ph / self._orig_h)
        sub = self._pixmap.copy(sx, sy, max(1, sw), max(1, sh))
        return sub.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def get_custom_thumbnail(self, index: int, size: int = 74) -> QPixmap:
        regions = self.custom_regions
        if not 0 <= index < len(regions) or self._pixmap.isNull():
            return QPixmap()
        region = regions[index]
        pw, ph = self._pixmap.width(), self._pixmap.height()
        sx = round(region.x * pw / self._orig_w)
        sy = round(region.y * ph / self._orig_h)
        ex = round((region.x + region.width) * pw / self._orig_w)
        ey = round((region.y + region.height) * ph / self._orig_h)
        cropped = self._pixmap.copy(
            sx, sy, max(1, ex - sx), max(1, ey - sy))
        return cropped.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    @property
    def custom_regions(self) -> list[SplitRegion]:
        return [
            SplitRegion(
                row=0, col=index,
                x=int(rect.x()), y=int(rect.y()),
                width=int(rect.width()), height=int(rect.height()))
            for index, rect in enumerate(self._custom_rects)
        ]

    @property
    def rows(self): return self._rows
    @property
    def cols(self): return self._cols
    @property
    def selected_count(self):
        return len(self._custom_selected) if self._custom_mode else len(self._selected)
    @property
    def total_cells(self):
        return len(self._custom_rects) if self._custom_mode else self._rows * self._cols
    @property
    def selected_cells(self):
        return {(0, index) for index in self._custom_selected} if self._custom_mode else set(self._selected)
    @property
    def custom_mode(self): return self._custom_mode
    @property
    def orig_w(self): return self._orig_w
    @property
    def orig_h(self): return self._orig_h
    @property
    def is_video(self): return self._is_video
    @property
    def fps(self): return self._video_fps
    @property
    def total_frames(self): return self._total_frames
    @property
    def current_frame(self): return self._current_frame

    def _fit_to_view(self):
        if self._pixmap.isNull():
            return
        vw, vh = self.width(), self.height()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw == 0 or ph == 0:
            return
        margin = 32
        self._scale = max(0.05, min((vw - margin * 2) / pw, (vh - margin * 2) / ph))
        self._center_image()

    def _center_image(self):
        if self._pixmap.isNull():
            return
        pw = self._pixmap.width() * self._scale
        ph = self._pixmap.height() * self._scale
        self._offset = QPointF((self.width() - pw) / 2, (self.height() - ph) / 2)
        self.update()

    def _image_rect(self) -> QRectF:
        if self._pixmap.isNull():
            return QRectF()
        return QRectF(self._offset.x(), self._offset.y(),
                      self._pixmap.width() * self._scale,
                      self._pixmap.height() * self._scale)

    def _cell_rect_in_view(self, row: int, col: int) -> QRectF:
        ir = self._image_rect()
        cw = ir.width() / self._cols
        ch = ir.height() / self._rows
        return QRectF(ir.left() + col * cw, ir.top() + row * ch, cw, ch)

    def _view_to_cell(self, pos: QPointF) -> Optional[tuple[int, int]]:
        ir = self._image_rect()
        if not ir.contains(pos):
            return None
        col = int((pos.x() - ir.left()) / ir.width() * self._cols)
        row = int((pos.y() - ir.top()) / ir.height() * self._rows)
        return max(0, min(row, self._rows - 1)), max(0, min(col, self._cols - 1))

    def _view_rect_to_source(self, view_rect: QRectF) -> Optional[QRectF]:
        image_rect = self._image_rect()
        clipped = view_rect.normalized().intersected(image_rect)
        if clipped.width() < 1 or clipped.height() < 1:
            return None
        left = max(0, math.floor((clipped.left() - image_rect.left()) / image_rect.width() * self._orig_w))
        top = max(0, math.floor((clipped.top() - image_rect.top()) / image_rect.height() * self._orig_h))
        right = min(self._orig_w, math.ceil((clipped.right() - image_rect.left()) / image_rect.width() * self._orig_w))
        bottom = min(self._orig_h, math.ceil((clipped.bottom() - image_rect.top()) / image_rect.height() * self._orig_h))
        if right <= left or bottom <= top:
            return None
        return QRectF(left, top, right - left, bottom - top)

    def _source_rect_to_view(self, source_rect: QRectF) -> QRectF:
        image_rect = self._image_rect()
        return QRectF(
            image_rect.left() + source_rect.x() / self._orig_w * image_rect.width(),
            image_rect.top() + source_rect.y() / self._orig_h * image_rect.height(),
            source_rect.width() / self._orig_w * image_rect.width(),
            source_rect.height() / self._orig_h * image_rect.height())

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), C_BG)
        dot_color = QColor("#CBD5E1")
        dot_step = 24
        for x in range(0, self.width(), dot_step):
            for y in range(0, self.height(), dot_step):
                p.fillRect(x, y, 2, 2, dot_color)
        if self._pixmap.isNull():
            p.setPen(C_MUTED)
            setFont(p, 13)
            p.drawText(self.rect(), Qt.AlignCenter, "正在加载预览…")
            return
        ir = self._image_rect()
        shadow_rect = ir.adjusted(-3, -3, 3, 6)
        for i in range(6, 0, -1):
            alpha = int(40 * i / 6)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(15, 23, 42, alpha))
            p.drawRoundedRect(shadow_rect.adjusted(i, i, -i, -i), 4, 4)
        p.drawPixmap(ir.toRect(), self._pixmap)
        p.setPen(QPen(QColor("#94A3B8"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(ir.adjusted(-0.5, -0.5, 0.5, 0.5))
        if self._custom_mode:
            crop_views = [self._source_rect_to_view(rect) for rect in self._custom_rects]
            if self._is_dragging and self._drag_start and self._drag_end:
                preview = self._view_rect_to_source(QRectF(self._drag_start, self._drag_end))
                if preview is not None:
                    crop_views.append(self._source_rect_to_view(preview))
            outside = QPainterPath()
            outside.addRect(ir)
            for crop_view in crop_views:
                crop_path = QPainterPath()
                crop_path.addRect(crop_view)
                outside = outside.subtracted(crop_path)
            p.fillPath(outside, QColor(15, 23, 42, 105))
            for index, crop_view in enumerate(crop_views):
                is_selected = index < len(self._custom_rects) and index in self._custom_selected
                if is_selected:
                    p.fillRect(crop_view, QColor(0, 120, 212, 38))
                p.setPen(QPen(C_BRAND if is_selected else C_MUTED, 2.0, Qt.SolidLine if is_selected else Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(crop_view, 3, 3)
                p.setPen(Qt.white)
                p.setBrush(C_BRAND if is_selected else C_MUTED)
                badge = QRectF(crop_view.right() - 23, crop_view.top() + 8, 18, 18)
                p.drawEllipse(badge)
                setFont(p, 8)
                p.drawText(badge, Qt.AlignCenter, str(index + 1))
            return
        if self._rows == 1 and self._cols == 1:
            return
        cw = ir.width() / self._cols
        ch = ir.height() / self._rows
        for r in range(self._rows):
            for c in range(self._cols):
                cr = QRectF(ir.left() + c * cw, ir.top() + r * ch, cw, ch)
                is_sel = (r, c) in self._selected
                is_hov = self._hover_cell == (r, c)
                if not is_sel:
                    p.fillRect(cr, C_UNSEL_MASK)
                elif is_hov:
                    p.fillRect(cr, C_HOVER_BG)
                if is_sel:
                    pen = QPen(C_BRAND, 2.0)
                    p.setPen(pen)
                    p.setBrush(C_SEL_BG)
                    p.drawRect(cr.adjusted(1, 1, -1, -1))
                else:
                    pen = QPen(QColor(255, 255, 255, 60), 1.0)
                    p.setPen(pen)
                    p.setBrush(Qt.NoBrush)
                    p.drawRect(cr.adjusted(0.5, 0.5, -0.5, -0.5))
                bx, by, br = cr.right() - 13, cr.top() + 13, 9
                badge = QPainterPath()
                badge.addEllipse(QPointF(bx, by), br, br)
                if is_sel:
                    p.fillPath(badge, C_BRAND)
                    p.setPen(QPen(Qt.white, 1.8, Qt.SolidLine, Qt.RoundCap))
                    p.drawLine(QPointF(bx - 4, by), QPointF(bx - 1.5, by + 3.5))
                    p.drawLine(QPointF(bx - 1.5, by + 3.5), QPointF(bx + 5, by - 4))
                else:
                    p.fillPath(badge, QColor(255, 255, 255, 180))
                    p.setPen(QPen(QColor("#94A3B8"), 1.2))
                    p.drawPath(badge)
                setFont(p, 8)
                p.setPen(QColor(255, 255, 255, 200) if is_sel else QColor(255, 255, 255, 100))
                p.drawText(cr.adjusted(6, 6, -6, -6), Qt.AlignLeft | Qt.AlignTop, f"{r + 1},{c + 1}")
        if self._is_dragging and self._drag_start and self._drag_end:
            rb = QRectF(self._drag_start, self._drag_end).normalized()
            p.setPen(QPen(C_BRAND, 1.5, Qt.DashLine))
            p.setBrush(QBrush(QColor(0, 120, 212, 25)))
            p.drawRect(rb)

    def resizeEvent(self, event):
        self._fit_to_view()
        super().resizeEvent(event)

    def wheelEvent(self, event):
        factor = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
        pos = QPointF(event.position())
        self._scale = max(0.05, min(self._scale * factor, 20.0))
        self._offset = QPointF(
            pos.x() - (pos.x() - self._offset.x()) * factor,
            pos.y() - (pos.y() - self._offset.y()) * factor,
        )
        self.update()
        self.grid_changed.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = QPointF(event.position())
            if self._custom_mode and not self._image_rect().contains(pos):
                return
            self._drag_start = pos
            self._drag_end   = pos
            self._is_dragging = False

    def mouseMoveEvent(self, event):
        pos = QPointF(event.position())
        if self._drag_start is not None:
            if (pos - self._drag_start).manhattanLength() > 4:
                self._is_dragging = True
            self._drag_end = pos
            self.update()
        else:
            cell = self._view_to_cell(pos)
            if cell != self._hover_cell:
                self._hover_cell = cell
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        if self._drag_start is None:
            return
        pos = QPointF(event.position())
        if self._custom_mode:
            if self._is_dragging:
                source_rect = self._view_rect_to_source(
                    QRectF(self._drag_start, pos))
                if source_rect is not None:
                    self._custom_rects.append(source_rect)
                    self._custom_selected.add(len(self._custom_rects) - 1)
                    self.grid_changed.emit()
            self._drag_start = None
            self._drag_end = None
            self._is_dragging = False
            self.update()
            return

        if self._is_dragging and self._drag_start:
            rb = QRectF(self._drag_start, pos).normalized()
            changed = False
            for r in range(self._rows):
                for c in range(self._cols):
                    if rb.intersects(self._cell_rect_in_view(r, c)):
                        if (r, c) in self._selected:
                            self._selected.discard((r, c))
                        else:
                            self._selected.add((r, c))
                        changed = True
            if changed:
                self.grid_changed.emit()
        else:
            cell = self._view_to_cell(pos)
            if cell:
                r, c = cell
                if (r, c) in self._selected:
                    self._selected.discard((r, c))
                else:
                    self._selected.add((r, c))
                self.grid_changed.emit()

        self._drag_start  = None
        self._drag_end    = None
        self._is_dragging = False
        self.update()

class _ThumbCard(QWidget):
    toggled = Signal(int, int, bool)
    _SIZE = 84

    def __init__(self, row: int, col: int, pixmap: QPixmap, sub_w: int, sub_h: int, index: int, parent=None):
        super().__init__(parent)
        self.row = row
        self.col = col
        self._selected = True
        self._hovered  = False
        self._pixmap   = pixmap
        self._size_lbl = f"{sub_w}×{sub_h}"
        self._idx_lbl  = f"{index:02d}"
        self.setFixedSize(152, self._SIZE + 42)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setMouseTracking(True)
        self.setToolTip(f"切片 {self._idx_lbl} · 第 {row + 1} 行，第 {col + 1} 列\n{self._size_lbl} px")

    def set_selected(self, v: bool):
        if self._selected != v:
            self._selected = v
            self.update()

    def enterEvent(self, e):
        self._hovered = True
        self.update()

    def leaveEvent(self, e):
        self._hovered = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._selected = not self._selected
            self.toggled.emit(self.row, self.col, self._selected)
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        W, H = self.width(), self.height()
        pad_x = 6
        thumb_y = 4
        thumb_w = W - pad_x * 2
        thumb_h = self._SIZE
        card_path = QPainterPath()
        card_path.addRoundedRect(QRectF(0, 0, W, H), 10, 10)
        if self._selected:
            bg = C_BRAND_LIGHT
            border_color = C_BRAND
            border_w = 1.5
        elif self._hovered:
            bg = QColor("#F1F5F9")
            border_color = C_BORDER_HV
            border_w = 1.0
        else:
            bg = QColor("#F8FAFC")
            border_color = C_BORDER
            border_w = 1.0
        p.fillPath(card_path, bg)
        p.setPen(QPen(border_color, border_w))
        p.setBrush(Qt.NoBrush)
        p.drawPath(card_path)
        if not self._pixmap.isNull():
            scaled = self._pixmap.scaled(thumb_w, thumb_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            ix = pad_x + (thumb_w - scaled.width()) // 2
            iy = thumb_y + (thumb_h - scaled.height()) // 2
            clip = QPainterPath()
            clip.addRoundedRect(QRectF(pad_x, thumb_y, thumb_w, thumb_h), 6, 6)
            p.save()
            p.setClipPath(clip)
            p.drawPixmap(ix, iy, scaled)
            p.restore()
            p.setPen(QPen(QColor(0, 0, 0, 20), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(pad_x, thumb_y, thumb_w, thumb_h), 6, 6)
        br = 9
        bx = pad_x + thumb_w - br - 2
        by = thumb_y + br + 2
        badge = QPainterPath()
        badge.addEllipse(QPointF(bx, by), br, br)
        if self._selected:
            p.fillPath(badge, C_BRAND)
            p.setPen(QPen(Qt.white, 1.8, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(bx - 4, by), QPointF(bx - 1.5, by + 3.5))
            p.drawLine(QPointF(bx - 1.5, by + 3.5), QPointF(bx + 5, by - 4))
        else:
            p.fillPath(badge, QColor(255, 255, 255, 220))
            p.setPen(QPen(QColor("#CBD5E1"), 1.2))
            p.drawPath(badge)
        label_top = thumb_y + thumb_h + 3
        p.setPen(C_BRAND if self._selected else C_TEXT)
        setFont(p, 8, weight=QFont.Normal if not self._selected else QFont.Bold)
        p.drawText(QRectF(0, label_top, W, 12), Qt.AlignCenter, f"#{self._idx_lbl}")
        p.setPen(C_MUTED)
        setFont(p, 7)
        p.drawText(QRectF(0, label_top + 12, W, 11), Qt.AlignCenter, self._size_lbl)

class _ZoomDock(QWidget):
    zoom_changed           = Signal(float)
    fit_requested          = Signal()
    select_all_requested   = Signal()
    deselect_all_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scale = 1.0
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(0)
        self.setFixedHeight(36)
        btn_minus = self._icon_btn("−")
        btn_minus.clicked.connect(lambda: self._adjust(-0.1))
        layout.addWidget(btn_minus)
        self._zoom_lbl = QLabel("—")
        self._zoom_lbl.setFixedWidth(50)
        self._zoom_lbl.setAlignment(Qt.AlignCenter)
        self._zoom_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #0F172A;")
        layout.addWidget(self._zoom_lbl)

        btn_plus = self._icon_btn("+")
        btn_plus.clicked.connect(lambda: self._adjust(0.1))
        layout.addWidget(btn_plus)

        btn_fit = self._text_btn("适应窗口")
        btn_fit.clicked.connect(self.fit_requested)
        layout.addWidget(btn_fit)

        layout.addWidget(self._sep_widget())

        btn_all  = self._text_btn("全选")
        btn_none = self._text_btn("清空")
        btn_all.clicked.connect(self.select_all_requested)
        btn_none.clicked.connect(self.deselect_all_requested)
        layout.addWidget(btn_all)
        self._add_slash(layout)
        layout.addWidget(btn_none)

    def set_scale(self, v: float):
        self._scale = v
        self._zoom_lbl.setText(f"{int(v * 100)}%")

    def _adjust(self, delta: float):
        new_scale = max(0.1, min(self._scale + delta, 10.0))
        self.zoom_changed.emit(new_scale)

    def _icon_btn(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedSize(24, 24)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet("""
            QPushButton {
                background: transparent; border: none; border-radius: 12px;
                font-size: 15px; font-weight: bold; color: #334155;
            }
            QPushButton:hover { background: #F1F5F9; }
            QPushButton:pressed { background: #E2E8F0; }
        """)
        return btn

    def _text_btn(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(24)
        btn.setStyleSheet("""
            QPushButton {
                background: transparent; border: none;
                font-size: 11px; color: #64748B; padding: 0 6px;
            }
            QPushButton:hover { color: #0078D4; }
        """)
        return btn

    def _sep_widget(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.VLine)
        f.setFixedWidth(1)
        f.setFixedHeight(16)
        f.setStyleSheet("background: #E2E8F0; border: none;")
        return f

    def _add_slash(self, layout: QHBoxLayout):
        lbl = QLabel("/")
        lbl.setStyleSheet("color: #CBD5E1; font-size: 11px;")
        layout.addWidget(lbl)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, self.width(), self.height()), 18, 18)
        p.fillPath(path, QColor("#FFFFFF"))
        p.setPen(QPen(C_BORDER, 1))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

class _LeftPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(272)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("""
            _LeftPanel {
                background: #FFFFFF;
                border: none;
            }
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { width: 5px; background: transparent; }
            QScrollBar::handle:vertical { background: #CBD5E1; border-radius: 3px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        body = QWidget()
        body.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(14)
        layout.addWidget(self._section_label("分割模式配置"))

        switcher = QWidget()
        switcher.setFixedHeight(36)
        switcher.setStyleSheet("""
            QWidget {
                background: #F1F5F9; border-radius: 10px; border: 1px solid #E2E8F0;
            }
        """)
        sw_layout = QHBoxLayout(switcher)
        sw_layout.setContentsMargins(4, 4, 4, 4)
        sw_layout.setSpacing(2)

        self.btn_suggest = self._tab_btn("建议裁剪", active=True)
        self.btn_custom  = self._tab_btn("自定义裁剪", active=False)
        grp = QButtonGroup(self)
        grp.setExclusive(True)
        grp.addButton(self.btn_suggest)
        grp.addButton(self.btn_custom)
        sw_layout.addWidget(self.btn_suggest)
        sw_layout.addWidget(self.btn_custom)
        layout.addWidget(switcher)

        # Suggested panel
        self.panel_suggest = QWidget()
        self.panel_suggest.setStyleSheet("background: transparent;")
        ps_layout = QVBoxLayout(self.panel_suggest)
        ps_layout.setContentsMargins(0, 0, 0, 0)
        ps_layout.setSpacing(10)

        info_card = self._card()
        ic_layout = QVBoxLayout(info_card)
        ic_layout.setContentsMargins(14, 12, 14, 12)
        ic_layout.setSpacing(8)

        badge_row = QHBoxLayout()
        dot = QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet("background: #0078D4; border-radius: 4px;")
        badge_row.addWidget(dot)
        badge_row.addWidget(self._label("智能裁剪建议", bold=True, size=11))
        badge_row.addStretch()
        self.badge_threshold = QLabel("规则: 1536px")
        self.badge_threshold.setStyleSheet(
            "font-size: 10px; font-weight: 600; "
            "color: #0078D4; background: #E0EFFE; border: 1px solid #BFDBFE; "
            "border-radius: 10px; padding: 1px 7px;")
        badge_row.addWidget(self.badge_threshold)
        ic_layout.addLayout(badge_row)
        ic_layout.addWidget(self._hdiv())

        _, self.lbl_raw       = self._info_row("原始分辨率:", ic_layout)
        _, self.lbl_grid      = self._info_row("切分方案:", ic_layout)
        _, self.lbl_cell_size = self._info_row("单块尺寸:", ic_layout)
        self.lbl_grid.setStyleSheet(
            "font-size: 11px; font-weight: 600; color: #0078D4; "
            "background: #E0EFFE; border: 1px solid #BFDBFE; "
            "border-radius: 6px; padding: 1px 7px;")

        self.lbl_strategy = QLabel("")
        self.lbl_strategy.setWordWrap(True)
        self.lbl_strategy.setStyleSheet(
            "font-size: 10px; color: #64748B; padding-top: 4px; "
            "border-top: 1px solid #F1F5F9;")
        ic_layout.addWidget(self.lbl_strategy)
        ps_layout.addWidget(info_card)

        stepper_card = self._card()
        sc_layout = QVBoxLayout(stepper_card)
        sc_layout.setContentsMargins(14, 12, 14, 12)
        sc_layout.setSpacing(10)

        sc_header = QHBoxLayout()
        sc_header.addWidget(self._label("切片布局", size=11, color="#64748B"))
        sc_header.addStretch()
        self._reset_btn = QPushButton("重置")
        self._reset_btn.setCursor(Qt.PointingHandCursor)
        self._reset_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; font-size: 11px; "
            "color: #0078D4; } QPushButton:hover { text-decoration: underline; }")
        sc_header.addWidget(self._reset_btn)
        sc_layout.addLayout(sc_header)

        for label, attr in [("行数 (Rows)", "_rows_spin"), ("列数 (Cols)", "_cols_spin")]:
            row_w = QWidget()
            row_w.setStyleSheet("background: transparent;")
            rl = QHBoxLayout(row_w)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(8)
            rl.addWidget(self._label(label, size=11))
            rl.addStretch()

            sl = QHBoxLayout()
            sl.setContentsMargins(0, 0, 0, 0)
            sl.setSpacing(4)
            btn_m = self._stepper_btn("−")
            spin  = self._make_spin(1, 16)
            btn_p = self._stepper_btn("+")
            setattr(self, attr, spin)
            btn_m.clicked.connect((lambda s: lambda: s.setValue(max(1, s.value() - 1)))(spin))
            btn_p.clicked.connect((lambda s: lambda: s.setValue(min(16, s.value() + 1)))(spin))
            sl.addWidget(btn_m); sl.addWidget(spin); sl.addWidget(btn_p)
            rl.addLayout(sl)

            if attr == "_cols_spin":
                sc_layout.addWidget(self._hdiv())
            sc_layout.addWidget(row_w)

        layout.addWidget(self.panel_suggest)

        # Custom panel
        self.panel_custom = QWidget()
        self.panel_custom.setStyleSheet("background: transparent;")
        pc_layout = QVBoxLayout(self.panel_custom)
        pc_layout.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("在预览区重复拖拽，可添加多个裁剪框；点击右侧缩略图可切换是否导出。")
        hint.setWordWrap(True)
        hint.setStyleSheet(
            "font-size: 11px; color: #475569; background: #F8FAFC; "
            "border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px;")
        pc_layout.addWidget(hint)
        self.panel_custom.setVisible(False)
        layout.addWidget(self.panel_custom)
        self._stepper_card = stepper_card
        layout.addWidget(stepper_card)

        layout.addWidget(self._divider())

        # Info section
        layout.addWidget(self._section_label("分割信息"))
        self.lbl_info_orig = self._label("原始尺寸：—", size=11, color="#334155")
        self.lbl_info_sub  = self._label("子块尺寸：—", size=11, color="#334155")
        layout.addWidget(self.lbl_info_orig)
        layout.addWidget(self.lbl_info_sub)

        layout.addWidget(self._divider())

        layout.addStretch()

        rule_lbl = QLabel(
            "图片 ≤ 1536 px · 视频 ≤ 1080 px")
        rule_lbl.setWordWrap(True)
        rule_lbl.setStyleSheet(
            "font-size: 10px; color: #0078D4; background: #E0EFFE; "
            "border: 1px solid #BFDBFE; border-radius: 8px; padding: 8px;")
        layout.addWidget(rule_lbl)

        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        footer = QLabel("拖拽裁剪 · 滚轮缩放")
        footer.setAlignment(Qt.AlignCenter)
        footer.setFixedHeight(28)
        footer.setStyleSheet(
            "font-size: 10px; color: #94A3B8; background: rgba(255,255,255,0.5); "
            "border-top: 1px solid #E2E8F0;")
        outer.addWidget(footer)

        self.btn_suggest.toggled.connect(self._on_mode_toggle)

    def _on_mode_toggle(self, suggest_checked: bool):
        self.panel_suggest.setVisible(suggest_checked)
        self.panel_custom.setVisible(not suggest_checked)
        self._stepper_card.setVisible(suggest_checked)

    def update_info(self, orig_w: int, orig_h: int, rows: int, cols: int, is_video: bool):
        sw = math.ceil(orig_w / cols)
        sh = math.ceil(orig_h / rows)
        threshold = "1080px" if is_video else "1536px"
        self.badge_threshold.setText(f"规则: {threshold}")
        self.lbl_raw.setText(f"{orig_w} × {orig_h} px")
        self.lbl_grid.setText(f"{cols} 列 × {rows} 行  ({rows * cols} 块)")
        self.lbl_cell_size.setText(f"{sw} × {sh} px")
        needs_split = rows > 1 or cols > 1
        if is_video:
            self.lbl_strategy.setText("视频超过 1080px，自动均匀裁剪。" if needs_split else "视频 ≤ 1080px，无需裁剪。")
        else:
            self.lbl_strategy.setText("图片超过 1536px，自动均匀裁剪。" if needs_split else "图片 ≤ 1536px，无需裁剪。")
        self.lbl_info_orig.setText(f"原始尺寸：{orig_w} × {orig_h} px")
        self.lbl_info_sub.setText(f"子块尺寸：{sw} × {sh} px")

    def update_custom_info(self, orig_w: int, orig_h: int, region_count: int):
        self.lbl_info_orig.setText(f"原图：{orig_w} × {orig_h} px")
        self.lbl_info_sub.setText(f"自定义裁剪区域：{region_count} 个")

    def set_grid_values(self, r: int, c: int):
        for s, v in [(self._rows_spin, r), (self._cols_spin, c)]:
            s.blockSignals(True); s.setValue(v); s.blockSignals(False)

    def _tab_btn(self, text, active):
        btn = QPushButton(text); btn.setCheckable(True); btn.setChecked(active)
        btn.setCursor(Qt.PointingHandCursor); btn.setFixedHeight(28)
        btn.setStyleSheet("""
            QPushButton { background: transparent; border: none; border-radius: 7px;
                font-size: 12px; font-weight: 500; color: #64748B; padding: 0 10px; }
            QPushButton:checked { background: white; color: #0078D4; font-weight: 600; }
            QPushButton:hover:!checked { color: #334155; }""")
        return btn

    def _card(self):
        w = QWidget()
        w.setStyleSheet("QWidget { background: white; border: 1px solid #E2E8F0; border-radius: 10px; }")
        sh = QGraphicsDropShadowEffect(); sh.setBlurRadius(10)
        sh.setOffset(0, 2); sh.setColor(QColor(15, 23, 42, 10)); w.setGraphicsEffect(sh)
        return w

    def _hdiv(self):
        f = QFrame(); f.setFrameShape(QFrame.HLine)
        f.setStyleSheet("color: #F1F5F9; background: #F1F5F9; border: none; max-height: 1px;")
        return f

    def _divider(self):
        f = QFrame(); f.setFrameShape(QFrame.HLine)
        f.setStyleSheet("color: #E2E8F0; background: #E2E8F0; border: none; max-height: 1px;")
        return f

    def _section_label(self, text):
        lbl = QLabel(text.upper())
        lbl.setStyleSheet(
            "font-size: 10px; font-weight: 600; color: #94A3B8; "
            "letter-spacing: 0.8px; background: transparent;")
        return lbl

    def _label(self, t, bold=False, size=11, color="#334155"):
        l = QLabel(t)
        l.setStyleSheet(f"font-size: {size}px; font-weight: {'600' if bold else '400'}; "
                        f"color: {color}; background: transparent;")
        return l

    def _info_row(self, label, layout):
        row = QHBoxLayout(); row.setContentsMargins(0, 0, 0, 0)
        lbl = self._label(label, size=11, color="#64748B")
        val = QLabel("—")
        val.setStyleSheet("font-size: 11px; font-weight: 600; color: #0F172A; background: transparent;")
        row.addWidget(lbl); row.addStretch(); row.addWidget(val)
        layout.addLayout(row)
        return row, val

    def _stepper_btn(self, t):
        btn = QPushButton(t); btn.setFixedSize(26, 26); btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet("""QPushButton { background: white; border: 1px solid #E2E8F0;
            border-radius: 7px; font-size: 14px; font-weight: bold; color: #334155; }
            QPushButton:hover { background: #F1F5F9; border-color: #CBD5E1; }
            QPushButton:pressed { background: #E2E8F0; }""")
        return btn

    def _make_spin(self, lo, hi):
        s = QSpinBox(); s.setRange(lo, hi); s.setValue(1)
        s.setFixedSize(36, 26); s.setAlignment(Qt.AlignCenter)
        s.setButtonSymbols(QSpinBox.NoButtons)
        s.setStyleSheet("""QSpinBox { border: 1px solid #E2E8F0; border-radius: 7px;
            background: #F8FAFC; color: #0078D4; font-size: 13px; font-weight: 700;}
            QSpinBox:focus { border-color: #0078D4; background: white; }""")
        return s


class _RightPanel(QWidget):
    confirm_clicked = Signal()
    cell_toggled    = Signal(int, int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(360)
        self.setStyleSheet("""
            _RightPanel {
                background: #FFFFFF;
                border: none;
            }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QWidget()
        header.setFixedHeight(48)
        header.setStyleSheet("background: #FFFFFF; border: none;")
        hdr_l = QHBoxLayout(header)
        hdr_l.setContentsMargins(14, 0, 14, 0)

        title_v = QVBoxLayout()
        title_v.setContentsMargins(0, 0, 0, 0)
        title_v.setSpacing(0)
        t1 = QLabel("分割结果预览")
        t1.setStyleSheet("font-size: 12px; font-weight: 600; color: #0F172A;")
        title_v.addWidget(t1)
        hdr_l.addLayout(title_v); hdr_l.addStretch()

        self._count_lbl = QLabel("已选: —")
        self._count_lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._count_lbl.setAlignment(Qt.AlignCenter)
        self._count_lbl.setStyleSheet(
            "font-size: 10px; font-weight: 600; "
            "color: #0369A1; background: #E0F2FE; border: 1px solid #BAE6FD; "
            "border-radius: 10px; padding: 3px 8px;")
        hdr_l.addWidget(self._count_lbl)

        btn_all  = self._link_btn("全选")
        btn_none = self._link_btn("取消选择")
        btn_all.clicked.connect(self._on_select_all)
        btn_none.clicked.connect(self._on_deselect_all)
        hdr_l.addSpacing(8); hdr_l.addWidget(btn_all)
        hdr_l.addWidget(btn_none)
        root.addWidget(header)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { width: 5px; background: transparent; }
            QScrollBar::handle:vertical { background: #CBD5E1; border-radius: 3px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        self._thumb_w = QWidget()
        self._thumb_w.setStyleSheet("background: transparent;")
        self._flow = FlowLayout(self._thumb_w, needAni=False, isTight=True)
        self._flow.setContentsMargins(8, 8, 8, 8)
        self._flow.setHorizontalSpacing(8)
        self._flow.setVerticalSpacing(8)
        scroll.setWidget(self._thumb_w)
        root.addWidget(scroll, 1)

        footer = QWidget()
        footer.setStyleSheet("background: #FFFFFF;")
        ft_l = QVBoxLayout(footer)
        ft_l.setContentsMargins(8, 4, 8, 4)

        export_title = QLabel("导出路径")
        export_title.setStyleSheet(
            "font-size: 10px; font-weight: 600; color: #64748B; "
            "letter-spacing: 0.3px; background: transparent;")
        ft_l.addWidget(export_title)

        export_row = QHBoxLayout()
        export_row.setContentsMargins(0, 0, 0, 0)
        export_row.setSpacing(6)

        self.export_path_edit = QLineEdit()
        self.export_path_edit.setObjectName("cropExportPath")
        self.export_path_edit.setPlaceholderText("选择裁剪结果保存目录...")
        self.export_path_edit.setFixedHeight(28)
        cache_path = cfg.get(cfg.cachePath) or os.path.join(os.path.expanduser("~"), "PowerToolsCache")
        self.export_path_edit.setText(os.path.join(cache_path, "tmpCrop"))
        self.export_path_edit.setStyleSheet(
            "QLineEdit#cropExportPath {"
            "  min-height: 28px; max-height: 28px; color: #1e293b; background: #f8fafc;"
            "  border: 1px solid #e2e8f0; border-radius: 10px; padding: 0 12px;"
            "  font-size: 11px;"
            "}"
            "QLineEdit#cropExportPath:hover { background: #ffffff; border-color: #cbd5e1; }"
            "QLineEdit#cropExportPath:focus { background: #ffffff; border: 1px solid #818cf8; }")

        browse_btn = QPushButton()
        browse_btn.setObjectName("cropExportBrowse")
        browse_btn.setIcon(FluentIcon.FOLDER.icon(Theme.LIGHT))
        browse_btn.setIconSize(QSize(16, 16))
        browse_btn.setToolTip("浏览裁剪导出目录")
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.setFixedSize(28, 28)
        browse_btn.setStyleSheet(
            "QPushButton#cropExportBrowse { color: #475569; background: #f8fafc;"
            "  border: 1px solid #e2e8f0; border-radius: 10px; }"
            "QPushButton#cropExportBrowse:hover { background: #f1f5f9; border-color: #cbd5e1; }"
            "QPushButton#cropExportBrowse:pressed { background: #e2e8f0; }")
        browse_btn.clicked.connect(self._choose_export_path)

        export_row.addWidget(self.export_path_edit, 1)
        export_row.addWidget(browse_btn)
        ft_l.addLayout(export_row)

        summary_row = QHBoxLayout()
        sum_lbl = QLabel("已选编号:")
        sum_lbl.setStyleSheet("font-size: 10px; color: #64748B; font-weight: 500;")
        self._sel_summary = QLabel("—")
        self._sel_summary.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._sel_summary.setAlignment(Qt.AlignCenter)
        self._sel_summary.setStyleSheet(
            "font-size: 9px; font-weight: 600; color: #0369A1; background: #E0F2FE; "
            "border: 1px solid #BAE6FD; border-radius: 9px; padding: 3px 7px;")
        summary_row.addWidget(sum_lbl)
        summary_row.addSpacing(4)
        summary_row.addWidget(self._sel_summary)
        summary_row.addStretch()
        ft_l.addLayout(summary_row)

        self._confirm_btn = PrimaryPushButton("确认裁剪", footer)
        self._confirm_btn.setFixedHeight(34)
        self._confirm_btn.setCursor(Qt.PointingHandCursor)
        self._confirm_btn.setEnabled(False)
        self._confirm_btn.clicked.connect(self.confirm_clicked)
        ft_l.addWidget(self._confirm_btn)
        root.addWidget(footer)

        self._cards: dict[tuple[int, int], _ThumbCard] = {}
        self._canvas_ref = None

    def export_path(self) -> str:
        return self.export_path_edit.text().strip()

    def _choose_export_path(self):
        directory = QFileDialog.getExistingDirectory(
            self, "选择裁剪导出目录", self.export_path(),
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog,
        )
        if directory:
            self.export_path_edit.setText(directory)

    def rebuild(self, canvas: _GridCanvas):
        self._canvas_ref = canvas
        self._flow.takeAllWidgets()
        self._cards.clear()

        rows, cols = canvas.rows, canvas.cols
        if canvas.custom_mode:
            for index, region in enumerate(canvas.custom_regions):
                thumb = canvas.get_custom_thumbnail(index, size=_ThumbCard._SIZE)
                card = _ThumbCard(0, index, thumb, region.width, region.height, index + 1, self._thumb_w)
                card.set_selected((0, index) in canvas.selected_cells)
                card.toggled.connect(self._on_card_toggled)
                self._flow.addWidget(card)
                self._cards[(0, index)] = card
            self._refresh_status(canvas)
            QTimer.singleShot(0, self._reflow_thumbnails)
            return

        if rows == 1 and cols == 1:
            self._count_lbl.setText("已选: —")
            self._sel_summary.setText("—")
            self._confirm_btn.setEnabled(False)
            self._confirm_btn.setText("确认裁剪")
            QTimer.singleShot(0, self._reflow_thumbnails)
            return

        sw = math.ceil(canvas.orig_w / cols)
        sh = math.ceil(canvas.orig_h / rows)
        for r in range(rows):
            for c in range(cols):
                thumb = canvas.get_cell_thumbnail(r, c, size=_ThumbCard._SIZE)
                index = r * cols + c + 1
                card = _ThumbCard(r, c, thumb, sw, sh, index, self._thumb_w)
                card.set_selected((r, c) in canvas.selected_cells)
                card.toggled.connect(self._on_card_toggled)
                self._flow.addWidget(card)
                self._cards[(r, c)] = card
        self._refresh_status(canvas)
        QTimer.singleShot(0, self._reflow_thumbnails)

    def _reflow_thumbnails(self):
        self._flow.invalidate()
        self._flow.activate()
        self._flow.setGeometry(self._thumb_w.rect())
        self._thumb_w.updateGeometry()

    def sync_selection(self, canvas: _GridCanvas):
        for (r, c), card in self._cards.items():
            card.set_selected((r, c) in canvas.selected_cells)
        self._refresh_status(canvas)

    def _on_card_toggled(self, row, col, sel):
        self.cell_toggled.emit(row, col, sel)

    def _on_select_all(self):
        if self._canvas_ref: self._canvas_ref.select_all(); self.sync_selection(self._canvas_ref)

    def _on_deselect_all(self):
        if self._canvas_ref: self._canvas_ref.deselect_all(); self.sync_selection(self._canvas_ref)

    def _refresh_status(self, canvas: _GridCanvas):
        sel = canvas.selected_count
        total = canvas.total_cells
        self._count_lbl.setText(
            f"已选: {sel}" if not canvas.custom_mode or total else "已选: 未绘制")
        selected_ids = []
        if canvas.custom_mode:
            for index, region in enumerate(canvas.custom_regions):
                if (0, index) in canvas.selected_cells:
                    selected_ids.append(f"{index + 1:02d}")
        else:
            selected_ids = [f"{row * canvas.cols + col + 1:02d}" for row, col in sorted(canvas.selected_cells)]
        selected_text = "、".join(selected_ids) if selected_ids else "无"
        self._sel_summary.setText(selected_text)
        self._sel_summary.setToolTip(f"已选编号：{selected_text}")
        self._confirm_btn.setEnabled(sel > 0)
        self._confirm_btn.setText(f"确认裁剪 ({sel})" if sel > 0 else "确认裁剪")

    def _link_btn(self, t):
        btn = QPushButton(t); btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet("""QPushButton { background: transparent; border: none;
            font-size: 11px; font-weight: 600; color: #0078D4; padding: 0 2px; }
            QPushButton:hover { text-decoration: underline; }""")
        return btn


class MediaSplitDialog(MaskDialogBase):
    def __init__(self, file_path: str, parent=None):
        super().__init__(parent=parent)

        self._file_path = file_path
        self._result: Optional[SplitResult] = None

        self.widget.setFixedSize(1160, 700)
        self.widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._hBoxLayout.setAlignment(self.widget, Qt.AlignCenter)
        self.widget.setAttribute(Qt.WA_StyledBackground, True)
        self.widget.setStyleSheet(
            "QFrame#centerWidget { background: #FFFFFF; border-radius: 14px; }")
        self.setShadowEffect(60, (0, 12), QColor(15, 23, 42, 60))
        self.setMaskColor(QColor(15, 23, 42, 100))

        root = QVBoxLayout(self.widget)
        root.setContentsMargins(12, 8, 12, 12)
        root.setSpacing(8)

        root.addWidget(self._build_header())

        body = _OpaqueCard()
        body.setBorderRadius(12)
        body_l = QHBoxLayout(body)
        body_l.setContentsMargins(10, 8, 10, 8)
        body_l.setSpacing(10)
        root.addWidget(body, 1)

        self._left = _LeftPanel()
        body_l.addWidget(self._left, 0)

        canvas_wrapper = QWidget()
        canvas_wrapper.setObjectName("canvasPreview")
        canvas_wrapper.setStyleSheet(
            "QWidget#canvasPreview { background: #F4F6F9; "
            "border: 1px solid #E2E8F0; border-radius: 10px; }")
        cw_l = QVBoxLayout(canvas_wrapper)
        cw_l.setContentsMargins(6, 6, 6, 6)
        cw_l.setSpacing(6)
        self._canvas = _GridCanvas()
        cw_l.addWidget(self._canvas, 1)

        self._video_controls = QWidget()
        video_controls_layout = QHBoxLayout(self._video_controls)
        video_controls_layout.setContentsMargins(6, 0, 6, 2)
        video_controls_layout.setSpacing(8)
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.setEnabled(False)
        self._frame_slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 4px; background: #CBD5E1; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #0078D4; border-radius: 2px; }
            QSlider::handle:horizontal {
                width: 12px; margin: -5px 0; border-radius: 6px;
                background: #0078D4; border: 2px solid white;
            }
        """)
        self._frame_counter = QLabel("—")
        self._frame_counter.setFixedWidth(88)
        self._frame_counter.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._frame_counter.setStyleSheet("font-size: 10px; color: #64748B;")
        video_controls_layout.addWidget(self._frame_slider, 1)
        video_controls_layout.addWidget(self._frame_counter)
        self._video_controls.hide()
        cw_l.addWidget(self._video_controls)
        body_l.addWidget(canvas_wrapper, 1)

        self._right = _RightPanel()
        body_l.addWidget(self._right, 0)

        self._dock = _ZoomDock(canvas_wrapper)
        self._dock.raise_()

        # Connections
        self._canvas.grid_changed.connect(self._on_grid_changed)
        self._frame_slider.valueChanged.connect(self._on_frame_changed)
        self._left.btn_suggest.toggled.connect(self._on_mode_toggle)
        self._left._rows_spin.valueChanged.connect(self._on_spin_changed)
        self._left._cols_spin.valueChanged.connect(self._on_spin_changed)
        self._left._reset_btn.clicked.connect(self._reset_suggested)
        self._right.confirm_clicked.connect(self._on_confirm)
        self._right.cell_toggled.connect(lambda r, c, sel: self._canvas.set_cell_selected(r, c, sel))
        self._dock.zoom_changed.connect(self._on_zoom_changed)
        self._dock.fit_requested.connect(self._fit_canvas)
        self._dock.select_all_requested.connect(self._canvas.select_all)
        self._dock.deselect_all_requested.connect(self._canvas.deselect_all)

        QTimer.singleShot(80, self._load_file)

    def _build_header(self) -> QWidget:
        h = QWidget(); h.setFixedHeight(60)
        h.setStyleSheet(
            "background: #FFFFFF; border: none;")
        hl = QGridLayout(h)
        hl.setContentsMargins(8, 0, 8, 0)
        hl.setHorizontalSpacing(12)
        hl.setColumnStretch(0, 1)
        hl.setColumnStretch(1, 0)
        hl.setColumnStretch(2, 1)

        badge = QWidget(); badge.setFixedSize(32, 32)
        badge.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            "stop:0 #0078D4,stop:1 #38BDF8); border-radius: 10px; border: none;")
        bl = QHBoxLayout(badge); bl.setContentsMargins(0, 0, 0, 0)
        ic = QLabel("✂"); ic.setAlignment(Qt.AlignCenter)
        ic.setStyleSheet("color: white; font-size: 14px; background: transparent; border: none;")
        bl.addWidget(ic)

        title_block = QWidget()
        title_layout = QVBoxLayout(title_block)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(1)
        t = QLabel("图片/视频裁剪工具")
        t.setStyleSheet("font-size: 14px; font-weight: 600; color: #0F172A; letter-spacing: -0.2px;")
        title_layout.addWidget(t)

        file_row = QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(5)
        file_caption = QLabel("文件名:")
        file_caption.setStyleSheet("font-size: 10px; color: #94A3B8;")
        file_row.addWidget(file_caption)
        fn_text = os.path.basename(self._file_path)
        self._file_name_lbl = QLabel(QFontMetrics(self.font()).elidedText(fn_text, Qt.ElideMiddle, 250))
        self._file_name_lbl.setMaximumWidth(270)
        self._file_name_lbl.setToolTip(self._file_path)
        self._file_name_lbl.setStyleSheet("font-size: 11px; color: #475569;")
        file_row.addWidget(self._file_name_lbl)
        title_layout.addLayout(file_row)

        identity = QWidget()
        identity_layout = QHBoxLayout(identity)
        identity_layout.setContentsMargins(0, 0, 0, 0)
        identity_layout.setSpacing(10)
        identity_layout.addWidget(badge)
        identity_layout.addWidget(title_block)
        hl.addWidget(identity, 0, 0, Qt.AlignLeft | Qt.AlignVCenter)

        info_pills = QHBoxLayout()
        info_pills.setContentsMargins(0, 0, 0, 0)
        info_pills.setSpacing(8)
        self._media_type_lbl = QLabel("—")
        self._media_type_lbl.setObjectName("mediaTypePill")
        self._media_type_lbl.setAlignment(Qt.AlignCenter)
        self._media_type_lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._media_type_lbl.setStyleSheet(
            "QLabel#mediaTypePill { font-size: 10px; font-weight: 600; color: #475569; "
            "background: #F1F5F9; border: 1px solid #E2E8F0; "
            "border-radius: 11px; padding: 4px 9px; }")
        info_pills.addWidget(self._media_type_lbl)

        self._res_lbl = QLabel("—")
        self._res_lbl.setObjectName("resolutionPill")
        self._res_lbl.setAlignment(Qt.AlignCenter)
        self._res_lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._res_lbl.setStyleSheet(
            "QLabel#resolutionPill { font-size: 11px;"
            "font-weight: 600; color: #0369A1; background: #E0F2FE; "
            "border: 1px solid #BAE6FD; border-radius: 11px; padding: 4px 9px; }")
        info_pills.addWidget(self._res_lbl)

        self._frame_separator = QFrame()
        self._frame_separator.setFrameShape(QFrame.VLine)
        self._frame_separator.setFixedSize(1, 18)
        self._frame_separator.setStyleSheet("background: #CBD5E1; border: none;")
        self._frame_separator.hide()
        info_pills.addWidget(self._frame_separator)

        self._frame_count_lbl = QLabel("总帧数: —")
        self._frame_count_lbl.setObjectName("frameCountPill")
        self._frame_count_lbl.setAlignment(Qt.AlignCenter)
        self._frame_count_lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._frame_count_lbl.setStyleSheet(
            "QLabel#frameCountPill { font-size: 11px; "
            "font-weight: 600; color: #0369A1; background: #E0F2FE; "
            "border: 1px solid #BAE6FD; border-radius: 11px; padding: 4px 9px; }")
        self._frame_count_lbl.hide()
        info_pills.addWidget(self._frame_count_lbl)
        hl.addLayout(info_pills, 0, 0, 1, 3, Qt.AlignCenter)

        cb = QPushButton("✕"); cb.setFixedSize(30, 30)
        cb.setCursor(Qt.PointingHandCursor)
        cb.setStyleSheet("""QPushButton {
                background: #F1F5F9; color: #64748B;
                border: 1px solid #E2E8F0; border-radius: 15px; font-size: 13px;
            }
            QPushButton:hover {
                background: #FEE2E2; color: #B91C1C; border-color: #FECACA;
            }
            QPushButton:pressed { background: #FECACA; }""")
        cb.clicked.connect(self.reject)
        hl.addWidget(cb, 0, 2, Qt.AlignRight | Qt.AlignVCenter)
        return h

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._reposition_dock)
        QTimer.singleShot(0, self._sync_dock_scale)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(10, self._reposition_dock)
        QTimer.singleShot(10, self._sync_dock_scale)

    def _sync_dock_scale(self):
        if self._canvas.orig_w:
            self._dock.set_scale(self._canvas._scale)

    def _reposition_dock(self):
        cw = self._canvas.parent()
        if cw and self._dock:
            dw = 300
            x = (cw.width() - dw) // 2
            y = self._canvas.y() + self._canvas.height() - self._dock.height() - 10
            self._dock.setFixedWidth(dw); self._dock.move(x, max(0, y))

    def _load_file(self):
        if not self._file_path or not os.path.isfile(self._file_path):
            QMessageBox.critical(self, "加载失败", "文件不存在，无法加载媒体预览。")
            return
        try:
            self._canvas.load(self._file_path)
        except (OSError, RuntimeError, cv2.error) as exc:
            self._canvas.release_video()
            QMessageBox.critical(self, "加载失败", f"无法加载媒体预览：\n{exc}")
            return
        if self._canvas.orig_w:
            self._res_lbl.setText(
                f"分辨率: {self._canvas.orig_w} × {self._canvas.orig_h} px")
            self._media_type_lbl.setText(
                f"媒体类型: {'视频' if self._canvas.is_video else '图片'}")
            self._dock.set_scale(self._canvas._scale)
        if self._canvas.is_video:
            self._frame_count_lbl.setText(f"总帧数: {self._canvas.total_frames:,}")
            self._frame_count_lbl.show()
            self._frame_separator.show()
            self._frame_slider.setRange(0, self._canvas.total_frames - 1)
            self._frame_slider.setEnabled(True)
            self._frame_slider.setValue(0)
            self._frame_counter.setText(f"1 / {self._canvas.total_frames:,}")
            self._video_controls.show()
        self._sync_all(); self._reposition_dock()

    def _on_frame_changed(self, frame_index: int):
        if not self._canvas.is_video:
            return
        try:
            self._canvas.seek_video_frame(frame_index)
        except (OSError, RuntimeError, cv2.error) as exc:
            QMessageBox.critical(self, "预览失败", f"无法读取视频帧：\n{exc}")
            self._frame_slider.setEnabled(False)
            return
        self._frame_counter.setText(
            f"{frame_index + 1} / {self._canvas.total_frames:,}")
        self._right.rebuild(self._canvas)

    def _on_mode_toggle(self, suggest_active: bool):
        self._canvas.set_custom_mode(not suggest_active)
        if suggest_active: self._reset_suggested()

    def _reset_suggested(self):
        if self._canvas.orig_w:
            rows, cols = _suggest_grid(self._canvas.orig_w, self._canvas.orig_h,
                                       self._canvas.is_video)
            self._canvas.set_grid(rows, cols)

    def _on_spin_changed(self):
        self._canvas.set_grid(self._left._rows_spin.value(), self._left._cols_spin.value())

    def _on_grid_changed(self):
        self._sync_all()

    def _sync_all(self):
        rows, cols = self._canvas.rows, self._canvas.cols
        self._left.set_grid_values(rows, cols)
        if self._canvas.orig_w:
            if self._canvas.custom_mode:
                self._left.update_custom_info(self._canvas.orig_w, self._canvas.orig_h, len(self._canvas.custom_regions))
            else:
                self._left.update_info(self._canvas.orig_w, self._canvas.orig_h,
                                       rows, cols, self._canvas.is_video)
        self._right.rebuild(self._canvas)

    def _on_zoom_changed(self, scale: float):
        self._canvas._scale = scale
        self._canvas._center_image()
        self._dock.set_scale(scale)

    def _fit_canvas(self):
        self._canvas._fit_to_view()
        self._dock.set_scale(self._canvas._scale)

    def _on_confirm(self):
        result = self._canvas.get_split_result()
        result.source_path = self._file_path
        output_dir = self._right.export_path()
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        if not result.regions:
            QMessageBox.warning(self, "裁剪失败", "请至少选择一个裁剪区域。")
            return
        try:
            output_dir = _prepare_export_dir(output_dir, self._file_path)
            stem = os.path.splitext(os.path.basename(self._file_path))[0]
            progress = QProgressDialog("正在生成裁剪结果…", None, 0, len(result.regions), self)
            progress.setWindowTitle("生成裁剪结果")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()
            QApplication.processEvents()
            if self._canvas.is_video:
                stream = _probe_video_stream(self._file_path)
                chroma_subsampled = (stream.get("pix_fmt") or "yuv420p").startswith(("yuv420", "yuv422", "nv12", "nv21", "yuvj420", "yuvj422"))
                for index, region in enumerate(result.regions):
                    progress.setLabelText(f"正在生成第 {index + 1}/{len(result.regions)} 个视频…")
                    progress.setValue(index)
                    QApplication.processEvents()
                    output_path = os.path.join(output_dir, f"{stem}_crop_{index + 1:03d}.mp4")
                    need_444 = chroma_subsampled and not all(value % 2 == 0 for value in (region.x, region.y, region.width, region.height))
                    video_kwargs = _video_encode_kwargs(stream, fallback_pix_fmt_444=need_444)
                    source = ffmpeg.input(self._file_path)
                    cropped = source.video.filter(
                        "crop",
                        region.width,
                        region.height,
                        region.x,
                        region.y,
                        exact=1,
                    )
                    try:
                        ffmpeg.run(
                            ffmpeg.output(cropped, source.audio, output_path,
                                          acodec="copy", fps_mode="passthrough",
                                          map_metadata="0", map_chapters="0", **video_kwargs)
                            .overwrite_output().global_args("-hide_banner", "-loglevel", "error"),
                            cmd=_ffmpeg_executable(), capture_stderr=True,
                        )
                    except ffmpeg.Error:
                        ffmpeg.run(
                            ffmpeg.output(cropped, source.audio, output_path,
                                          acodec="aac", **{"b:a": "192k"}, fps_mode="passthrough",
                                          map_metadata="0", map_chapters="0", **video_kwargs)
                            .overwrite_output().global_args("-hide_banner", "-loglevel", "error"),
                            cmd=_ffmpeg_executable(), capture_stderr=True,
                        )
                    region.output_path = output_path
            else:
                source = cv2.imread(self._file_path)
                if source is None:
                    raise RuntimeError(f"无法读取原始图片：{self._file_path}")
                for index, region in enumerate(result.regions):
                    progress.setLabelText(f"正在生成第 {index + 1}/{len(result.regions)} 张图片…")
                    progress.setValue(index)
                    QApplication.processEvents()
                    crop = source[
                        region.y:region.y + region.height,
                        region.x:region.x + region.width,
                    ]
                    output_path = os.path.join(output_dir, f"{stem}_crop_{index + 1:03d}.png")
                    if not cv2.imwrite(output_path, crop):
                        raise RuntimeError(f"无法写入裁剪图片：{output_path}")
                    region.output_path = output_path
            progress.setValue(len(result.regions))
        except Exception as exc:
            detail = getattr(exc, "stderr", None)
            detail = detail.decode("utf-8", "replace") if isinstance(detail, bytes) else (detail or str(exc))
            QMessageBox.critical(self, "裁剪失败", f"生成裁剪结果失败：\n{detail}")
            return
        finally:
            if 'progress' in locals():
                progress.close()
        result.export_dir = output_dir
        result.metadata_path = write_split_metadata(output_dir, result)
        self._canvas.release_video()
        self._result = result
        completion_box = MessageBox(
            title="裁剪完成",
            content=(f"裁剪结果已保存至：\n{output_dir}\n\n您可以在软件中重新选择裁剪后的文件进行处理。"),
            parent=self,
        )
        completion_box.cancelButton.hide()
        completion_box.yesButton.setText("知道了")
        completion_box.yesButton.setFixedWidth(280)
        completion_box.exec()
        self.accept()

    @property
    def result(self) -> Optional[SplitResult]:
        return self._result

    def closeEvent(self, event):
        self._canvas.release_video()
        super().closeEvent(event)