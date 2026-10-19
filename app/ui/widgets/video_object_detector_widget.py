import os
import sys
import cv2
from pathlib import Path
from PySide6.QtCore import QEasingCurve, QPointF, QRectF, QSize, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QFrame, QGraphicsOpacityEffect, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea,
    QProgressBar, QSizePolicy, QSlider, QSpinBox, QVBoxLayout, QWidget
)
from app.ui.library.qfluentwidgets import FluentIcon, Theme, getFont, setFont
from app.ui.common.utils import global_backend_info_cache
from app.utils.image_io import open_video_capture
from app.ui.common.config import cfg
from app.ui.widgets.image_preview_widget import ScrollBar


class VideoFrameCanvas(QFrame):
    promptAdded = Signal(tuple)
    promptStateChanged = Signal(bool, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("videoCanvas")
        self.setMouseTracking(True)
        self._frame = QImage()
        self._placeholder = self.tr("打开视频以预览画面")
        self._view_mode = "overlay"
        self._show_detection_overlay = True
        self._show_prompts = True
        self._prompt_mode = "pos"
        self._drag_start = None
        self._drag_rect = None
        self._prompts = []
        self._selected_prompt_index = None
        self._prompt_history = []
        self.setFocusPolicy(Qt.StrongFocus)
        self._hint_opacity = 0.0
        self._hint_animation = QVariantAnimation(self)
        self._hint_animation.setDuration(150)
        self._hint_animation.setEasingCurve(QEasingCurve.OutCubic)
        self._hint_animation.valueChanged.connect(self._set_hint_opacity)

    def set_frame(self, frame: QImage, placeholder: str = "", show_detection_overlay: bool = True, show_prompts: bool = True):
        self._frame = frame
        self._placeholder = placeholder or self.tr("打开视频以预览画面")
        self._show_detection_overlay = show_detection_overlay
        self._show_prompts = show_prompts
        self.update()

    def set_view_mode(self, mode: str):
        self._view_mode = mode
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        background_clip = QPainterPath()
        background_clip.addRoundedRect(QRectF(self.rect()), 16, 16)
        painter.save()
        painter.setClipPath(background_clip)
        painter.fillRect(self.rect(), QColor("#000000"))
        painter.restore()
        if self._frame.isNull():
            painter.setPen(QColor("#94a3b8"))
            painter.setFont(getFont(14, QFont.DemiBold))
            painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
            if self._placeholder == self.tr("打开视频以预览画面"):
                painter.setPen(QColor("#64748b"))
                painter.setFont(getFont(10))
                painter.drawText(self.rect().adjusted(0, 38, 0, 0), Qt.AlignHCenter | Qt.AlignTop, self.tr("支持 MP4、MOV、AVI 等常见视频格式"))
            event.accept()
            return
        target = self._image_target_rect()
        clip = QPainterPath()
        clip.addRoundedRect(target, 16, 16)
        painter.save()
        painter.setClipPath(clip)
        painter.drawImage(target, self._frame)
        if self._show_prompts:
            for index, (prompt_rect, prompt_mode) in enumerate(self._prompts):
                prompt = QRectF(
                    target.x() + prompt_rect.x() * target.width(),
                    target.y() + prompt_rect.y() * target.height(),
                    prompt_rect.width() * target.width(),
                    prompt_rect.height() * target.height(),
                )
                self._paint_prompt(painter, prompt, prompt_mode, index == self._selected_prompt_index)
            if self._drag_rect is not None:
                self._paint_prompt(painter, self._drag_rect, self._prompt_mode)
        if self._view_mode == "original":
            painter.restore()
            self._paint_hover_hint(painter)
            event.accept()
            return
        if self._show_detection_overlay:
            bounds = QRectF(
                target.x() + target.width() * 0.266,
                target.y() + target.height() * 0.222,
                target.width() * 0.125,
                target.height() * 0.389,
            )
            if self._view_mode == "mask":
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(99, 102, 241, 225))
                mask = QRectF(
                    bounds.x() + bounds.width() * 0.2,
                    bounds.y() + bounds.height() * 0.08,
                    bounds.width() * 0.62,
                    bounds.height() * 0.86,
                )
                painter.drawRoundedRect(mask, 12, 12)
            else:
                painter.setBrush(QColor(99, 102, 241, 42))
                painter.setPen(QPen(QColor("#6366f1"), 2, Qt.DashLine))
                painter.drawRoundedRect(bounds, 4, 4)
            painter.setPen(Qt.NoPen)
            if self._view_mode != "mask":
                painter.setBrush(QColor("#4f46e5"))
                label_rect = QRectF(bounds.x(), bounds.y() - 22, 128, 20)
                painter.drawRoundedRect(label_rect, 4, 4)
                painter.setPen(Qt.white)
                painter.setFont(getFont(9, QFont.DemiBold))
                painter.drawText(label_rect, Qt.AlignCenter, self.tr("#1 person (SAM3)"))
        painter.restore()
        self._paint_hover_hint(painter)
        event.accept()

    def _paint_hover_hint(self, painter: QPainter):
        if self._hint_opacity <= 0:
            return
        hint_width = max(0, min(400, self.width() - 32))
        hint_rect = QRectF(self.rect().left() + 16, self.rect().bottom() - 48, hint_width, 32)
        hint_text = self.tr("✎ 拖拽添加提示框，点击选中后可删除，或撤销上一步操作。")
        hint_font = getFont(10)
        hint_text = QFontMetrics(hint_font).elidedText(hint_text, Qt.ElideRight, max(0, hint_width - 20))
        painter.save()
        painter.setOpacity(self._hint_opacity)
        painter.setPen(QPen(QColor(51, 65, 85, 153), 1))
        painter.setBrush(QColor(15, 23, 42, 204))
        painter.drawRoundedRect(hint_rect, 16, 16)
        painter.setPen(QColor("#cbd5e1"))
        painter.setFont(hint_font)
        painter.drawText(hint_rect.adjusted(14, 0, -10, 0), Qt.AlignVCenter, hint_text)
        painter.restore()

    def enterEvent(self, event):
        self._animate_hint(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animate_hint(0.0)
        super().leaveEvent(event)

    def _animate_hint(self, opacity: float):
        self._hint_animation.stop()
        self._hint_animation.setStartValue(self._hint_opacity)
        self._hint_animation.setEndValue(opacity)
        self._hint_animation.start()

    def _set_hint_opacity(self, opacity: float | None):
        if opacity is None:
            return
        self._hint_opacity = opacity
        self.update()

    def set_prompt_mode(self, mode: str):
        self._prompt_mode = mode

    def clear_prompts(self):
        self._prompts.clear()
        self._selected_prompt_index = None
        self._prompt_history.clear()
        self._drag_start = None
        self._drag_rect = None
        self.update()
        self._emit_prompt_state()

    def undo_prompt(self):
        if not self._prompt_history:
            return
        self._prompts, self._selected_prompt_index = self._prompt_history.pop()
        self.update()
        self._emit_prompt_state()

    def delete_selected_prompt(self):
        if self._selected_prompt_index is None:
            return
        self._save_prompt_state()
        del self._prompts[self._selected_prompt_index]
        self._selected_prompt_index = None
        self.update()
        self._emit_prompt_state()

    def _save_prompt_state(self):
        self._prompt_history.append((self._prompts.copy(), self._selected_prompt_index))

    def _emit_prompt_state(self):
        self.promptStateChanged.emit(bool(self._prompt_history), self._selected_prompt_index is not None)

    def _select_prompt_at(self, position: QPointF, target: QRectF):
        self._selected_prompt_index = None
        for index in range(len(self._prompts) - 1, -1, -1):
            prompt_rect, _ = self._prompts[index]
            prompt = QRectF(
                target.x() + prompt_rect.x() * target.width(),
                target.y() + prompt_rect.y() * target.height(),
                prompt_rect.width() * target.width(),
                prompt_rect.height() * target.height(),
            )
            if prompt.adjusted(-6, -6, 6, 6).contains(position):
                self._selected_prompt_index = index
                break
        self.update()
        self._emit_prompt_state()

    def mousePressEvent(self, event):
        target = self._image_target_rect() if not self._frame.isNull() else QRectF()
        if event.button() == Qt.LeftButton and target.contains(event.position()):
            self.setFocus()
            self._drag_start = QPointF(event.position())
            self._drag_rect = QRectF(self._drag_start, self._drag_start)
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            target = self._image_target_rect()
            position = self._clamp_to_target(QPointF(event.position()), target)
            self._drag_rect = QRectF(self._drag_start, position).normalized()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_start is not None:
            target = self._image_target_rect()
            position = self._clamp_to_target(QPointF(event.position()), target)
            rect = QRectF(self._drag_start, position).normalized()
            rect = rect.intersected(target)
            if rect.width() >= 4 and rect.height() >= 4:
                normalized = QRectF(
                    (rect.x() - target.x()) / max(target.width(), 1),
                    (rect.y() - target.y()) / max(target.height(), 1),
                    rect.width() / max(target.width(), 1),
                    rect.height() / max(target.height(), 1),
                )
                self._save_prompt_state()
                self._prompts.append((normalized, self._prompt_mode))
                self._selected_prompt_index = len(self._prompts) - 1
                self.promptAdded.emit((normalized, self._prompt_mode))
                self._emit_prompt_state()
            else:
                self._select_prompt_at(position, target)
            self._drag_start = None
            self._drag_rect = None
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    @staticmethod
    def _paint_prompt(painter: QPainter, rect: QRectF, mode: str, selected: bool = False):
        color = QColor("#10b981") if mode == "pos" else QColor("#ef4444")
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 55))
        pen_color = QColor("#facc15") if selected else color
        painter.setPen(QPen(pen_color, 3 if selected else 2, Qt.DashLine if selected else Qt.SolidLine))
        painter.drawRoundedRect(rect, 4, 4)

    def _image_target_rect(self) -> QRectF:
        target = QRectF(self.rect())
        source_ratio = self._frame.width() / max(self._frame.height(), 1)
        target_ratio = target.width() / max(target.height(), 1)
        if target_ratio > source_ratio:
            width = target.height() * source_ratio
            target.setX((self.width() - width) / 2)
            target.setWidth(width)
        else:
            height = target.width() / source_ratio
            target.setY((self.height() - height) / 2)
            target.setHeight(height)
        return target

    @staticmethod
    def _clamp_to_target(point: QPointF, target: QRectF) -> QPointF:
        return QPointF(min(max(point.x(), target.left()), target.right()), min(max(point.y(), target.top()), target.bottom()))


class VideoPreviewViewport(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = None
        self.toolbar = None
        self.toolbar_layout = None
        self.view_group = None
        self.prompt_actions = None
        self.status_label = None
        self._compact_toolbar = None

    def set_content(self, canvas: VideoFrameCanvas, toolbar: QWidget, view_group: QWidget, prompt_actions: QWidget, status_label: QLabel):
        self.canvas = canvas
        self.toolbar = toolbar
        self.toolbar_layout = toolbar.layout()
        self.view_group = view_group
        self.prompt_actions = prompt_actions
        self.status_label = status_label
        self.toolbar_layout.addWidget(view_group, 0, 0, Qt.AlignLeft)
        self.toolbar_layout.addWidget(prompt_actions, 0, 1, Qt.AlignCenter)
        self.toolbar_layout.addWidget(status_label, 0, 2, Qt.AlignRight)
        self.toolbar_layout.setColumnStretch(0, 1)
        self._layout_content()

    def resizeEvent(self, event):
        self._layout_content()
        super().resizeEvent(event)

    def refresh_layout(self):
        self._layout_content()

    def _layout_content(self):
        if self.canvas is None or self.toolbar is None or self.view_group is None or self.prompt_actions is None or self.status_label is None:
            return
        toolbar_width = max(0, self.width() - 32)
        margins = self.toolbar_layout.contentsMargins()
        required_width = (
            self.view_group.sizeHint().width()
            + self.prompt_actions.sizeHint().width()
            + self.status_label.sizeHint().width()
            + self.toolbar_layout.horizontalSpacing() * 2
            + margins.left()
            + margins.right()
        )
        compact = toolbar_width < required_width
        if compact != self._compact_toolbar:
            self.toolbar_layout.removeWidget(self.status_label)
            if compact:
                self.toolbar_layout.addWidget(self.status_label, 1, 0, 1, 3, Qt.AlignRight)
            else:
                self.toolbar_layout.addWidget(self.status_label, 0, 2, Qt.AlignRight)
            self._compact_toolbar = compact
        toolbar_height = 72 if compact else 40
        self.toolbar.setGeometry(16, 16, toolbar_width, toolbar_height)
        canvas_top = self.toolbar.y() + toolbar_height + 8
        self.canvas.setGeometry(8, canvas_top, max(0, self.width() - 16), max(0, self.height() - canvas_top - 8))


class VideoObjectDetectorWidget(QDialog):
    ISSUE_FRAME = 1
    taskTerminateRequested = Signal()

    def __init__(self, file_path: str | None = None, parent=None):
        super().__init__(parent)
        self.file_path = None
        self.cap = None
        self.fps = 30.0
        self.total_frames = 120
        self.current_frame = 14
        self.current_model = "sam3"
        self.current_view = "original"
        self.prompt_tool = "pos"
        self.is_playing = False
        self._task_running = False
        self._task_progress = 0
        self._playback_compact = False
        self._playback_legend_labels = []
        self.result_video_captures: dict[str, dict[str, cv2.VideoCapture | None]] = {model: {"mask": None, "overlay": None} for model in ("sam3", "edgetam")}
        self._frame_detection_results: dict[int, tuple[float | None, dict[str, int]]] = {}

        self.setWindowTitle(self.tr("视频分割与跟踪工作台"))
        self.setObjectName("videoObjectDetector")
        self.setFixedSize(1400, 800)

        self._build_ui()
        backend_type, hardware_name = global_backend_info_cache.get()
        self.device_label.setText(backend_type)
        self.device_tag.setToolTip(hardware_name or backend_type)
        self._apply_styles()

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._advance_playback)
        if file_path:
            self._load_video(file_path)
        else:
            self._update_frame()
            self._update_video_info_pill(None)
            self._update_project_subtitle(None)
            self._set_label_status(self.canvas_status, self.tr("●  等待加载视频"), "warning")

    @staticmethod
    def _pill_label(text: str, object_name: str, wrap: bool = True) -> QWidget:
        wrapper = QWidget()
        wrapper.setObjectName("pillWrapper")
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        label = QLabel(text)
        label.setObjectName(object_name)
        label.setWordWrap(wrap)
        label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        layout.addWidget(label)
        layout.addStretch()
        return wrapper

    @staticmethod
    def _info_pill(text: str) -> QWidget:
        pill = QWidget()
        pill.setObjectName("helperText")
        pill.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        layout = QHBoxLayout(pill)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        icon = QLabel("i")
        icon.setObjectName("infoIcon")
        setFont(icon, 9)
        icon.setAlignment(Qt.AlignCenter)
        icon.setFixedSize(12, 12)
        layout.addWidget(icon)

        label = QLabel(text)
        label.setObjectName("infoLabel")
        setFont(label, 10)
        label.setWordWrap(False)
        layout.addWidget(label)
        return pill

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setObjectName("topBar")
        header.setFixedHeight(64)
        header_layout = QGridLayout(header)
        header_layout.setContentsMargins(24, 0, 24, 0)
        header_layout.setHorizontalSpacing(16)
        header_layout.setVerticalSpacing(0)

        brand_container = QWidget()
        brand_layout = QHBoxLayout(brand_container)
        brand_layout.setContentsMargins(0, 0, 0, 0)
        brand_layout.setSpacing(14)

        brand_icon = QLabel()
        brand_icon.setObjectName("brandIcon")
        brand_icon.setAlignment(Qt.AlignCenter)
        brand_icon.setPixmap(FluentIcon.APPLICATION.icon(Theme.DARK).pixmap(QSize(16, 16)))
        brand_icon.setFixedSize(38, 38)
        brand_layout.addWidget(brand_icon, 0, Qt.AlignVCenter)

        brand_text = QVBoxLayout()
        brand_text.setSpacing(2)
        title = QLabel(self.tr("视频分割与跟踪工作台"))
        setFont(title, 14, QFont.Bold)
        title.setObjectName("brandTitle")
        self.brand_subtitle = QLabel(self.tr("项目文件: 未加载"))
        setFont(self.brand_subtitle, 11)
        self.brand_subtitle.setObjectName("brandSubtitle")
        brand_text.addWidget(title)
        brand_text.addWidget(self.brand_subtitle)
        brand_layout.addLayout(brand_text)
        brand_container.setFixedWidth(280)

        header_layout.addWidget(brand_container, 0, 0, Qt.AlignLeft | Qt.AlignVCenter)
        right_header_spacer = QWidget()
        right_header_spacer.setFixedWidth(228)
        header_layout.addWidget(right_header_spacer, 0, 2)

        self.header_toolbar = QFrame()
        self.header_toolbar.setObjectName("headerToolbar")
        toolbar_layout = QHBoxLayout(self.header_toolbar)
        toolbar_layout.setContentsMargins(16, 8, 16, 8)
        toolbar_layout.setSpacing(12)

        self.video_info_pill = QFrame()
        self.video_info_pill.setObjectName("videoInfoPill")
        video_info_layout = QHBoxLayout(self.video_info_pill)
        video_info_layout.setContentsMargins(10, 5, 10, 5)
        video_info_layout.setSpacing(6)
        video_icon = QLabel()
        video_icon.setPixmap(FluentIcon.VIDEO.icon(Theme.LIGHT).pixmap(QSize(14, 14)))
        video_icon.setFixedSize(14, 14)
        video_info_layout.addWidget(video_icon)
        video_info_prefix = QLabel(self.tr("视频信息:"))
        video_info_prefix.setObjectName("videoInfoTitle")
        video_info_layout.addWidget(video_info_prefix)
        self.video_resolution_label = QLabel("—")
        self.video_resolution_label.setObjectName("videoInfoValue")
        video_info_layout.addWidget(self.video_resolution_label)
        self.video_fps_label = QLabel("— fps")
        self.video_fps_label.setObjectName("videoInfoValue")
        video_info_layout.addWidget(self.video_fps_label)
        video_separator = QFrame()
        video_separator.setObjectName("videoInfoDivider")
        video_separator.setFixedSize(1, 14)
        video_info_layout.addWidget(video_separator)
        self.video_total_frames_label = QLabel(self.tr("共 0 帧"))
        self.video_total_frames_label.setObjectName("videoInfoSecondary")
        video_info_layout.addWidget(self.video_total_frames_label)
        toolbar_layout.addWidget(self.video_info_pill)

        toolbar_layout.addWidget(self._toolbar_separator())

        progress_widget = QWidget()
        progress_layout = QHBoxLayout(progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(8)
        self.task_status_icon = QLabel("●")
        self.task_status_icon.setObjectName("taskStatusIcon")
        progress_layout.addWidget(self.task_status_icon)
        self.task_status_label = QLabel(self.tr("未运行"))
        self.task_status_label.setObjectName("taskStatusLabel")
        progress_layout.addWidget(self.task_status_label)
        self.task_progress_percent = QLabel("0.0%")
        self.task_progress_percent.setObjectName("taskProgressPercent")
        self.task_progress_percent.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.task_progress_percent.setMinimumWidth(38)
        progress_layout.addWidget(self.task_progress_percent)
        self.task_progress_bar = QProgressBar()
        self.task_progress_bar.setObjectName("taskProgressBar")
        self.task_progress_bar.setRange(0, 100)
        self.task_progress_bar.setValue(0)
        self.task_progress_bar.setTextVisible(False)
        self.task_progress_bar.setFixedSize(96, 6)
        self._progress_opacity_effect = QGraphicsOpacityEffect(self.task_progress_bar)
        self._progress_opacity_effect.setOpacity(1.0)
        self.task_progress_bar.setGraphicsEffect(self._progress_opacity_effect)
        self._progress_pulse = QVariantAnimation(self)
        self._progress_pulse.setDuration(2000)
        self._progress_pulse.setStartValue(1.0)
        self._progress_pulse.setEndValue(0.6)
        self._progress_pulse.setLoopCount(-1)
        self._progress_pulse.valueChanged.connect(self._progress_opacity_effect.setOpacity)
        progress_layout.addWidget(self.task_progress_bar)
        self.task_frame_counter = QLabel("0/0 帧")
        self.task_frame_counter.setObjectName("taskFrameCounter")
        progress_layout.addWidget(self.task_frame_counter)
        toolbar_layout.addWidget(progress_widget)

        toolbar_layout.addWidget(self._toolbar_separator())

        self.device_tag = QFrame()
        self.device_tag.setObjectName("deviceTag")
        device_layout = QHBoxLayout(self.device_tag)
        device_layout.setContentsMargins(9, 4, 9, 4)
        device_layout.setSpacing(6)
        device_icon = QLabel()
        device_icon.setPixmap(FluentIcon.DEVELOPER_TOOLS.icon(Theme.LIGHT).pixmap(QSize(14, 14)))
        device_icon.setFixedSize(14, 14)
        device_layout.addWidget(device_icon)
        self.device_label = QLabel(self.tr("CPU 运行"))
        self.device_label.setObjectName("deviceLabel")
        device_layout.addWidget(self.device_label)
        toolbar_layout.addWidget(self.device_tag)

        toolbar_layout.addWidget(self._toolbar_separator())

        self.terminate_button = QPushButton()
        self.terminate_button.setObjectName("terminateButton")
        self.terminate_button.setCursor(Qt.ArrowCursor)
        self.terminate_button.setEnabled(False)
        self.terminate_button.setToolTip(self.tr("当前没有运行中的任务"))
        stop_icon = QPixmap(14, 14)
        stop_icon.fill(Qt.transparent)
        stop_painter = QPainter(stop_icon)
        stop_painter.setRenderHint(QPainter.Antialiasing)
        stop_painter.setPen(Qt.NoPen)
        stop_painter.setBrush(QColor("#dc2626"))
        stop_painter.drawRoundedRect(2, 2, 10, 10, 1, 1)
        stop_painter.end()
        self.terminate_button.setIcon(QIcon(stop_icon))
        self.terminate_button.setIconSize(QSize(14, 14))
        self.terminate_button.setText(self.tr("终止任务"))
        self.terminate_button.clicked.connect(self._confirm_task_termination)
        toolbar_layout.addWidget(self.terminate_button)

        header_layout.addWidget(self.header_toolbar, 0, 1, Qt.AlignCenter)
        header_layout.setColumnStretch(0, 1)
        header_layout.setColumnStretch(1, 0)
        header_layout.setColumnStretch(2, 1)

        root.addWidget(header)

        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)
        root.addLayout(content, 1)

        self.left_scroll = QScrollArea()
        self.left_scroll.setObjectName("sideScroll")
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setFrameShape(QFrame.NoFrame)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.left_scroll.setVerticalScrollBar(ScrollBar(Qt.Vertical, self.left_scroll))
        self.left_scroll.setFixedWidth(352)
        self.left_scroll.setWidget(self._build_settings_panel())
        content.addWidget(self.left_scroll)

        center = self._build_preview_panel()
        content.addWidget(center, 1)

        self.right_panel = self._build_objects_panel()
        self.right_panel.setFixedWidth(290)
        content.addWidget(self.right_panel)

    def _update_video_info_pill(self, resolution: tuple | None, fps: float = 0.0, total: int = 0):
        if resolution:
            self.video_resolution_label.setText(f"{resolution[0]} × {resolution[1]}")
            self.video_fps_label.setText(f"{fps:.1f} fps")
        else:
            self.video_resolution_label.setText("—")
            self.video_fps_label.setText("— fps")
        self.video_total_frames_label.setText(self.tr("共 {0} 帧").format(total))
        self.task_frame_counter.setText(self.tr("{0}/{1} 帧").format(self._task_progress, total))

    @staticmethod
    def _toolbar_separator() -> QFrame:
        separator = QFrame()
        separator.setObjectName("toolbarSeparator")
        separator.setFixedSize(1, 16)
        return separator

    def set_task_running(self, running: bool, backend_type: str | None = None, hardware_name: str = ""):
        if running and not self._task_running:
            self.set_task_progress(0)
        self._task_running = running
        if running:
            self.task_status_label.setText(self.tr("处理中"))
            self.task_status_label.setObjectName("taskStatusLabelRunning")
            self.task_status_icon.setObjectName("taskStatusIconRunning")
        else:
            self.task_status_label.setText(self.tr("未运行"))
            self.task_status_label.setObjectName("taskStatusLabel")
            self.task_status_icon.setObjectName("taskStatusIcon")
        self.task_status_label.style().unpolish(self.task_status_label)
        self.task_status_label.style().polish(self.task_status_label)
        self.task_status_icon.style().unpolish(self.task_status_icon)
        self.task_status_icon.style().polish(self.task_status_icon)
        self.task_progress_bar.style().unpolish(self.task_progress_bar)
        self.task_progress_bar.style().polish(self.task_progress_bar)
        if running:
            self._progress_pulse.start()
        else:
            self._progress_pulse.stop()
            self._progress_opacity_effect.setOpacity(1.0)
        self.terminate_button.setEnabled(running)
        self.terminate_button.setCursor(Qt.PointingHandCursor if running else Qt.ArrowCursor)
        self.terminate_button.setToolTip("" if running else self.tr("当前没有运行中的任务"))
        if backend_type:
            self.device_label.setText(backend_type)
        if hardware_name:
            self.device_tag.setToolTip(hardware_name)
        elif backend_type:
            self.device_tag.setToolTip(backend_type)

    def set_task_progress(self, current_frame: int):
        self._task_progress = max(0, min(current_frame, self.total_frames))
        percentage = (self._task_progress / self.total_frames * 100) if self.total_frames else 0
        self.task_progress_bar.setValue(round(percentage))
        self.task_progress_percent.setText(f"{percentage:.1f}%")
        self.task_frame_counter.setText(self.tr("{0}/{1} 帧").format(self._task_progress, self.total_frames))

    def _confirm_task_termination(self):
        if not self._task_running:
            return

        dialog = QDialog(self)
        dialog.setObjectName("terminateDialog")
        dialog.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(10)

        heading = QHBoxLayout()
        alert_icon = QLabel("!")
        alert_icon.setObjectName("terminateAlertIcon")
        alert_icon.setAlignment(Qt.AlignCenter)
        alert_icon.setFixedSize(34, 34)
        heading.addWidget(alert_icon)
        title = QLabel(self.tr("终止当前任务？"))
        title.setObjectName("terminateDialogTitle")
        heading.addWidget(title)
        heading.addStretch()
        layout.addLayout(heading)

        description = QLabel(self.tr("终止后，当前分割与跟踪计算将被中断，未保存的进度可能会丢失。是否确定终止？"))
        description.setObjectName("terminateDialogDescription")
        description.setWordWrap(True)
        layout.addWidget(description)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_button = QPushButton(self.tr("取消"))
        cancel_button.setObjectName("terminateCancelButton")
        cancel_button.clicked.connect(dialog.reject)
        buttons.addWidget(cancel_button)
        confirm_button = QPushButton(self.tr("确定终止"))
        confirm_button.setObjectName("terminateConfirmButton")
        confirm_button.clicked.connect(dialog.accept)
        buttons.addWidget(confirm_button)
        layout.addLayout(buttons)
        dialog.setStyleSheet(TERMINATE_DIALOG_STYLESHEET)
        dialog.setFixedWidth(360)
        dialog.adjustSize()
        if dialog.exec() == QDialog.Accepted:
            self.taskTerminateRequested.emit()
            self.set_task_running(False)
            self.task_status_label.setText(self.tr("已终止"))
            self.task_status_label.setObjectName("taskStatusLabelTerminated")
            self.task_status_icon.setObjectName("taskStatusIconTerminated")
            self.task_status_label.style().unpolish(self.task_status_label)
            self.task_status_label.style().polish(self.task_status_label)
            self.task_status_icon.style().unpolish(self.task_status_icon)
            self.task_status_icon.style().polish(self.task_status_icon)

    def _update_project_subtitle(self, path: str | None):
        if not path:
            self.brand_subtitle.setText(self.tr("项目文件: 未加载"))
            self.brand_subtitle.setToolTip("")
            return
        fm = QFontMetrics(self.brand_subtitle.font())
        elided = fm.elidedText(os.path.basename(path), Qt.ElideMiddle, 220)
        self.brand_subtitle.setText(self.tr("项目文件: ") + elided)
        self.brand_subtitle.setToolTip(path)

    def _default_save_path(self) -> str:
        return str(Path(cfg.get(cfg.cachePath)) / "VideoSegmentation" / "results")

    def notify_result_video(self, view: str, file_path: str | Path | None,  model: str | None = None):
        if view not in ("mask", "overlay"):
            raise ValueError("view must be 'mask' or 'overlay'")
        result_model = model or self.current_model
        if result_model not in self.result_video_captures:
            raise ValueError("model must be 'sam3' or 'edgetam'")
        path_value = None
        capture = None
        if file_path is not None:
            path = Path(file_path).expanduser()
            if not path.is_file():
                message = self.tr("●  检测结果文件不存在: {0}").format(path)
                self._set_label_status(self.canvas_status, message, "error")
                raise FileNotFoundError(path)
            path_value = str(path.resolve())
            capture = cv2.VideoCapture(path_value)
            if not capture.isOpened():
                capture.release()
                message = self.tr("●  无法打开检测结果视频: {0}").format(path)
                self._set_label_status(self.canvas_status, message, "error")
                raise ValueError(message)
        old_capture = self.result_video_captures[result_model][view]
        if old_capture is not None:
            old_capture.release()
        self.result_video_captures[result_model][view] = capture
        if result_model == self.current_model and view == self.current_view:
            self._update_frame()

    def _choose_save_path_for(self, line_edit: QLineEdit):
        directory = QFileDialog.getExistingDirectory(
            self,
            self.tr("选择结果保存目录"),
            line_edit.text() or str(Path.home()),
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog if sys.platform == "darwin" else QFileDialog.Option(0)
        )
        if directory:
            line_edit.setText(directory)

    def _build_save_path_section(self, attr_name: str) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._section_title(self.tr("结果保存路径")))

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        line_edit = QLineEdit()
        line_edit.setObjectName("textInput")
        line_edit.setPlaceholderText(self.tr("选择结果保存目录..."))
        line_edit.setFixedHeight(34)
        line_edit.setText(self._default_save_path())

        browse = QPushButton()
        browse.setObjectName("browseButton")
        self._set_button_icon(browse, FluentIcon.FOLDER)
        browse.setFixedSize(34, 34)
        browse.setCursor(Qt.PointingHandCursor)
        browse.setToolTip(self.tr("浏览..."))
        browse.clicked.connect(lambda: self._choose_save_path_for(line_edit))

        row_layout.addWidget(line_edit, 1)
        row_layout.addWidget(browse)
        layout.addWidget(row)

        setattr(self, attr_name, line_edit)
        return container

    def _add_anchor_frame_setting(self, layout: QVBoxLayout, spin_attr: str, button_attr: str, object_name: str):
        layout.addSpacing(4)
        layout.addWidget(self._section_title(self.tr("起始关键帧")))
        anchor_row = QHBoxLayout()
        anchor_row.setSpacing(8)
        anchor_spin = QSpinBox()
        anchor_spin.setRange(1, self.total_frames)
        anchor_spin.setValue(1)
        anchor_spin.setFixedHeight(30)
        anchor_spin.setObjectName(object_name)
        setattr(self, spin_attr, anchor_spin)
        anchor_row.addWidget(anchor_spin)

        anchor_button = QPushButton(self.tr("设当前帧为起始帧"))
        anchor_button.setObjectName("neutralButton")
        self._set_button_icon(anchor_button, FluentIcon.PIN)
        anchor_button.setFixedHeight(30)
        anchor_button.clicked.connect(lambda _checked=False, spin=anchor_spin: spin.setValue(self.current_frame))
        setattr(self, button_attr, anchor_button)
        anchor_row.addWidget(anchor_button, 1)
        layout.addLayout(anchor_row)

    def _add_tracking_direction_setting(self, layout: QVBoxLayout, buttons_attr: str):
        layout.addSpacing(4)
        layout.addWidget(self._section_title(self.tr("跟踪方向")))
        direction_group = QFrame()
        direction_group.setObjectName("segmentedGroup")
        direction_group.setFixedHeight(36)
        direction_layout = QHBoxLayout(direction_group)
        direction_layout.setContentsMargins(4, 3, 4, 3)
        direction_layout.setSpacing(2)
        direction_buttons = {}
        setattr(self, buttons_attr, direction_buttons)
        for key, text, tip in (
            ("forward", "向前", "从【起始关键帧】开始向前查找目标"),
            ("backward", "向后", "从【起始关键帧】开始向后查找目标"),
            ("bidirectional", "双向", "从【起始关键帧】开始向两边查找目标"),
        ):
            button = QPushButton(self.tr(text))
            button.setCheckable(True)
            button.setObjectName("segmentedButton")
            button.setCursor(Qt.PointingHandCursor)
            button.setToolTip(tip)
            button.clicked.connect(lambda _checked=False, direction=key: self._select_direction(direction))
            direction_buttons[key] = button
            direction_layout.addWidget(button)
        direction_buttons["forward"].setChecked(True)
        layout.addWidget(direction_group)

    def _add_frame_range_setting(self, layout: QVBoxLayout, start_attr: str, end_attr: str):
        layout.addSpacing(4)
        layout.addWidget(self._section_title(self.tr("处理帧范围")))
        range_row = QHBoxLayout()
        range_row.setSpacing(8)
        range_start = QSpinBox()
        range_start.setRange(1, self.total_frames)
        range_start.setValue(1)
        range_start.setObjectName("frameRangeInput")
        range_start.setFixedHeight(30)
        range_end = QSpinBox()
        range_end.setRange(1, self.total_frames)
        range_end.setValue(self.total_frames)
        range_end.setObjectName("frameRangeInput")
        range_end.setFixedHeight(30)
        setattr(self, start_attr, range_start)
        setattr(self, end_attr, range_end)
        range_start.valueChanged.connect(lambda frame, end=range_end: self._on_range_start_changed(frame, end))
        range_end.valueChanged.connect(lambda frame, start=range_start: self._on_range_end_changed(frame, start))
        range_row.addWidget(range_start, 0, Qt.AlignCenter)
        range_label = QLabel(self.tr("至"))
        range_label.setObjectName("inlineLabel")
        range_row.addWidget(range_label, 0, Qt.AlignCenter)
        range_row.addWidget(range_end, 0, Qt.AlignCenter)
        layout.addLayout(range_row)

    def _build_settings_panel(self):
        panel = QWidget()
        panel.setObjectName("leftPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("panelHeader")
        header.setFixedHeight(48)
        heading = QHBoxLayout(header)
        heading.setContentsMargins(16, 0, 16, 0)
        heading_label = QLabel(self.tr("☷  任务与模型设置"))
        heading_label.setObjectName("panelHeading")
        setFont(heading_label, 13, QFont.DemiBold)
        heading.addWidget(heading_label)
        heading.addStretch()
        mode_badge = QLabel(self.tr("单模型模式"))
        mode_badge.setObjectName("indigoBadge")
        setFont(mode_badge, 12)
        heading.addWidget(mode_badge, alignment=Qt.AlignmentFlag.AlignVCenter)
        panel_layout.addWidget(header)

        body = QWidget()
        body.setObjectName("panelBody")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(16, 14, 16, 18)
        layout.setSpacing(8)
        panel_layout.addWidget(body, 1)

        layout.addWidget(self._section_title(self.tr("选择算法模型")))
        model_row = QHBoxLayout()
        model_row.setSpacing(10)
        self.model_sam3 = self._model_card("精确检测", self.tr("复杂场景与目标检测"), self.tr("推荐"), "sam3")
        self.model_edgetam = self._model_card("快速跟踪", self.tr("已有掩码连续跟踪"), self.tr("轻量"), "edgetam")
        self.model_sam3.setFixedHeight(60)
        self.model_edgetam.setFixedHeight(60)
        self.model_sam3.clicked.connect(lambda: self._select_model("sam3"))
        self.model_edgetam.clicked.connect(lambda: self._select_model("edgetam"))
        model_row.addWidget(self.model_sam3, 1)
        model_row.addWidget(self.model_edgetam, 1)
        layout.addLayout(model_row)
        layout.addSpacing(4)

        self.sam3_params = QWidget()
        sam_layout = QVBoxLayout(self.sam3_params)
        sam_layout.setContentsMargins(0, 0, 0, 0)
        sam_layout.setSpacing(8)
        sam_layout.addWidget(self._section_title(self.tr("文本提示"), self.tr("多目标检测")))

        self.prompt_input = QLineEdit("watermark")
        self.prompt_input.setObjectName("promptInput")
        setFont(self.prompt_input, 12)
        self.prompt_input.setPlaceholderText(self.tr("输入目标类别，例如 watermark"))
        self.prompt_input.setClearButtonEnabled(True)
        self.prompt_input.setFixedHeight(34)
        self.prompt_input.setCursorPosition(0)
        sam_layout.addWidget(self.prompt_input)

        chips_row = QWidget()
        chips_layout = QHBoxLayout(chips_row)
        chips_layout.setContentsMargins(0, 0, 0, 0)
        chips_layout.setSpacing(6)
        chips_text = {
            self.tr("水印"): "watermark",
            self.tr("字幕"): "subtitle",
            self.tr("LOGO"): "logo",
        }
        for chip_text, chip_value in chips_text.items():
            chip = self._prompt_chip(chip_text, chip_value)
            setFont(chip, 10)
            chips_layout.addWidget(chip)
        chips_layout.addStretch()
        sam_layout.addWidget(chips_row)

        sam_layout.addWidget(self._info_pill(self.tr("文本提示在传播时会持续检测同类目标，可能增加计算量。")))

        sam_layout.addSpacing(4)
        sam_layout.addWidget(self._section_title(self.tr("框选提示 (正框 / 负框)")))
        tool_row = QHBoxLayout()
        tool_row.setSpacing(8)
        self.positive_btn = QPushButton(self.tr("正框 (去除的目标)"))
        self.negative_btn = QPushButton(self.tr("负框 (保留的目标)"))
        self.positive_btn.setObjectName("positiveButton")
        self.negative_btn.setObjectName("negativeButton")
        self.positive_btn.setCheckable(True)
        self.negative_btn.setCheckable(True)
        self.positive_btn.setIcon(self._square_plus_minus_icon(True))
        self.negative_btn.setIcon(self._square_plus_minus_icon(False))
        self.positive_btn.setIconSize(QSize(14, 14))
        self.negative_btn.setIconSize(QSize(14, 14))
        self.positive_btn.setFixedHeight(34)
        self.negative_btn.setFixedHeight(34)
        self.positive_btn.setCursor(Qt.PointingHandCursor)
        self.negative_btn.setCursor(Qt.PointingHandCursor)
        self.positive_btn.clicked.connect(lambda: self._select_prompt_tool("pos"))
        self.negative_btn.clicked.connect(lambda: self._select_prompt_tool("neg"))
        tool_row.addWidget(self.positive_btn)
        tool_row.addWidget(self.negative_btn)
        sam_layout.addLayout(tool_row)

        self._add_anchor_frame_setting(sam_layout, "anchor_spin", "anchor_button", "anchorFrameInput")
        self._add_tracking_direction_setting(sam_layout, "direction_buttons")
        self._add_frame_range_setting(sam_layout, "range_start", "range_end")
        sam_layout.addSpacing(6)
        sam_layout.addWidget(self._build_save_path_section("save_path_input"))

        sam_layout.addSpacing(6)
        self.run_sam3_button = QPushButton(self.tr("开始跟踪检测（精确检测）"))
        self.run_sam3_button.setObjectName("primaryButton")
        self._set_button_icon(self.run_sam3_button, FluentIcon.ROBOT, dark=True)
        self.run_sam3_button.setFixedHeight(38)
        self.run_sam3_button.setCursor(Qt.PointingHandCursor)
        self.run_sam3_button.clicked.connect(self._show_inference_notice)
        sam_layout.addWidget(self.run_sam3_button)
        layout.addWidget(self.sam3_params)

        self.edgetam_params = QWidget()
        edge_layout = QVBoxLayout(self.edgetam_params)
        edge_layout.setContentsMargins(0, 0, 0, 0)
        edge_layout.setSpacing(8)
        edge_layout.addWidget(self._pill_label(self.tr("ⓘ EdgeTAM 需要先提供可靠初始掩码，不通过文本搜索目标。"), "warningBox", wrap=False))
        self._add_anchor_frame_setting(edge_layout, "edge_anchor_spin", "edge_anchor_button", "anchorFrameInput")
        self._add_tracking_direction_setting(edge_layout, "edge_direction_buttons")
        self._add_frame_range_setting(edge_layout, "edge_range_start", "edge_range_end")

        edge_layout.addSpacing(6)
        edge_layout.addWidget(self._build_save_path_section("edge_save_path_input"))

        edge_layout.addSpacing(6)
        self.run_edgetam_button = QPushButton(self.tr("开始跟踪检测（快速跟踪）"))
        self.run_edgetam_button.setObjectName("violetButton")
        self._set_button_icon(self.run_edgetam_button, FluentIcon.VIDEO, dark=True)
        self.run_edgetam_button.setFixedHeight(38)
        self.run_edgetam_button.setCursor(Qt.PointingHandCursor)
        self.run_edgetam_button.clicked.connect(self._show_inference_notice)
        edge_layout.addWidget(self.run_edgetam_button)
        self.edgetam_params.hide()
        layout.addWidget(self.edgetam_params)
        layout.addStretch()
        return panel

    def _build_preview_panel(self):
        center = QFrame()
        center.setObjectName("previewPanel")
        self.preview_panel = center
        layout = QVBoxLayout(center)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        top_bar = QGridLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setHorizontalSpacing(8)
        top_bar.setVerticalSpacing(4)

        view_group = QFrame()
        view_group.setObjectName("viewGroup")
        view_layout = QHBoxLayout(view_group)
        view_layout.setContentsMargins(4, 4, 4, 4)
        view_layout.setSpacing(2)
        self.view_buttons = {}
        for key, text in (("original", "原视频"), ("overlay", "半透明叠加"), ("mask", "仅掩码")):
            button = QPushButton(self.tr(text))
            button.setCheckable(True)
            button.setObjectName("viewButton")
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, mode=key: self._select_view(mode))
            self.view_buttons[key] = button
            view_layout.addWidget(button)
        self.view_buttons["original"].setChecked(True)

        prompt_actions = QWidget()
        prompt_actions_layout = QHBoxLayout(prompt_actions)
        prompt_actions_layout.setContentsMargins(0, 0, 0, 0)
        prompt_actions_layout.setSpacing(6)
        self.undo_prompt_button = QPushButton(self.tr("撤销"))
        self.undo_prompt_button.setObjectName("promptActionButton")
        self.undo_prompt_button.setToolTip(self.tr("撤销最近添加或删除的提示框"))
        self.undo_prompt_button.setEnabled(False)
        prompt_actions_layout.addWidget(self.undo_prompt_button)
        self.delete_prompt_button = QPushButton(self.tr("删除"))
        self.delete_prompt_button.setObjectName("promptActionButton")
        self.delete_prompt_button.setToolTip(self.tr("删除选中的提示框"))
        self.delete_prompt_button.setEnabled(False)
        prompt_actions_layout.addWidget(self.delete_prompt_button)

        self.canvas_status = QLabel(self.tr("●  状态: 已分割 (Normal)"))
        self.canvas_status.setObjectName("canvasStatus")
        setFont(self.canvas_status, 12, QFont.Medium)

        viewport = VideoPreviewViewport()
        viewport.setObjectName("previewViewport")
        top_bar_widget = QWidget(viewport)
        top_bar_widget.setObjectName("previewToolbar")
        top_bar_widget.setMinimumHeight(40)
        top_bar_widget.setLayout(top_bar)

        self.canvas = VideoFrameCanvas(viewport)
        self.canvas.promptAdded.connect(self._on_prompt_added)
        self.canvas.promptStateChanged.connect(self._update_prompt_actions)
        self.undo_prompt_button.clicked.connect(self.canvas.undo_prompt)
        self.delete_prompt_button.clicked.connect(self.canvas.delete_selected_prompt)
        viewport.set_content(self.canvas, top_bar_widget, view_group, prompt_actions, self.canvas_status)
        layout.addWidget(viewport, 1)

        playback = QFrame()
        playback.setObjectName("playbackPanel")
        playback_layout = QVBoxLayout(playback)
        playback_layout.setContentsMargins(16, 12, 16, 12)
        playback_layout.setSpacing(10)
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setObjectName("timelineSlider")
        self.timeline.setRange(1, self.total_frames)
        self.timeline.setValue(self.current_frame)
        self.timeline.valueChanged.connect(self._on_timeline_changed)
        playback_layout.addWidget(self.timeline)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.prev_button = QPushButton()
        self.prev_button.setObjectName("stepButton")
        self._set_button_icon(self.prev_button, FluentIcon.SKIP_BACK, dark=True)
        self.prev_button.setFixedSize(32, 32)
        self.prev_button.clicked.connect(lambda: self._step_frame(-1))
        controls.addWidget(self.prev_button)
        self.play_button = QPushButton()
        self.play_button.setObjectName("playButton")
        self._set_button_icon(self.play_button, FluentIcon.PLAY, dark=True)
        self.play_button.setFixedSize(36, 36)
        self.play_button.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_button)
        self.next_button = QPushButton()
        self.next_button.setObjectName("stepButton")
        self._set_button_icon(self.next_button, FluentIcon.SKIP_FORWARD, dark=True)
        self.next_button.setFixedSize(32, 32)
        self.next_button.clicked.connect(lambda: self._step_frame(1))
        controls.addWidget(self.next_button)
        self.frame_counter = QLabel()
        self.frame_counter.setObjectName("frameCounter")
        setFont(self.frame_counter, 12, QFont.Medium)
        self.frame_counter.setToolTip(self.tr("Frame -- / --"))
        controls.addWidget(self.frame_counter)
        controls.addStretch()
        controls.addWidget(self._legend_dot("#10b981", self.tr("正常 (--)")))
        controls.addWidget(self._legend_dot("#f59e0b", self.tr("需要检查 (--)")))
        controls.addWidget(self._legend_dot("#475569", self.tr("未处理 (--)")))
        playback_layout.addLayout(controls)
        layout.addWidget(playback)
        return center

    def _build_objects_panel(self):
        panel = QWidget()
        panel.setObjectName("rightPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("panelHeader")
        header.setFixedHeight(48)
        heading = QHBoxLayout(header)
        heading.setContentsMargins(16, 0, 16, 0)
        icon = QLabel()
        icon.setObjectName("panelTitleIcon")
        icon.setFixedSize(16, 16)
        icon.setPixmap(self._color_title_icon("#f97316", "#fbbf24", "#f59e0b"))
        heading.addWidget(icon, 0, Qt.AlignVCenter)
        title = QLabel(self.tr("对象列表与处理状态"))
        title.setObjectName("panelHeading")
        setFont(title, 12, QFont.DemiBold)
        heading.addWidget(title)
        heading.addStretch()
        badge = QLabel(self.tr("精确检测输出"))
        badge.setObjectName("indigoBadge")
        setFont(badge, 12)
        self.result_model_badge = badge
        heading.addWidget(badge, alignment=Qt.AlignmentFlag.AlignVCenter)
        panel_layout.addWidget(header)

        body = QWidget()
        body.setObjectName("panelBody")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        panel_layout.addWidget(body, 1)

        result_card = QFrame()
        result_card.setObjectName("objectCard")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(12, 12, 12, 12)
        result_layout.setSpacing(10)
        result_header = QHBoxLayout()
        result_header.addWidget(self._color_dot("#6366f1", 12), 0, Qt.AlignVCenter)
        result_title = QLabel()
        result_title.setObjectName("objectTitle")
        setFont(result_title, 12, QFont.Bold)
        result_header.addWidget(result_title)
        result_header.addStretch()
        self.result_frame_label = result_title
        result_state = QLabel(self.tr("待检测"))
        result_state.setObjectName("indigoBadge")
        result_header.addWidget(result_state)
        self.result_state_label = result_state
        result_layout.addLayout(result_header)
        result_layout.addWidget(self._detail_row(self.tr("检测分数:"), "—", "result_score_value"))
        result_layout.addWidget(self._detail_row(self.tr("检测对象:"), self.tr("watermark"), "result_objects_value"))
        result_layout.addWidget(self._detail_row(self.tr("对象数量:"), "—", "result_count_value"))
        result_divider = QFrame()
        result_divider.setObjectName("cardDivider")
        result_divider.setFixedHeight(1)
        result_layout.addWidget(result_divider)
        result_actions = QGridLayout()
        result_actions.setContentsMargins(0, 0, 0, 0)
        result_actions.setSpacing(8)
        resegment = QPushButton(self.tr("重新分割"))
        resegment.setObjectName("smallNeutralButton")
        self._set_button_icon(resegment, FluentIcon.SYNC)
        resegment.setFixedHeight(32)
        resegment.clicked.connect(self._show_inference_notice)
        result_actions.addWidget(resegment, 0, 0)
        result_layout.addLayout(result_actions)
        layout.addWidget(result_card)

        divider = QFrame()
        divider.setObjectName("cardDivider")
        divider.setFixedHeight(1)
        layout.addWidget(divider)

        issue_header = QHBoxLayout()
        issue_title = QLabel(self.tr("需要修复"))
        issue_title.setObjectName("sectionTitle")
        setFont(issue_title, 12, QFont.DemiBold)
        issue_header.addWidget(issue_title)
        issue_header.addStretch()
        self.issue_count_label = QLabel("1")
        self.issue_count_label.setObjectName("warningBadge")
        issue_header.addWidget(self.issue_count_label)
        layout.addLayout(issue_header)

        self.issue_scroll = QScrollArea()
        self.issue_scroll.setObjectName("issueScroll")
        self.issue_scroll.setWidgetResizable(True)
        self.issue_scroll.setFrameShape(QFrame.NoFrame)
        self.issue_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.issue_scroll.setVerticalScrollBar(ScrollBar(Qt.Vertical, self.issue_scroll))
        issue_cards = QWidget()
        issue_cards.setObjectName("issueCards")
        self.issue_cards_layout = QVBoxLayout(issue_cards)
        self.issue_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.issue_cards_layout.setSpacing(10)

        second = QFrame()
        second.setObjectName("warningCard")
        second_layout = QVBoxLayout(second)
        second_layout.setContentsMargins(12, 12, 12, 12)
        second_layout.setSpacing(10)
        second_header = QHBoxLayout()
        second_header.addWidget(self._color_dot("#f59e0b", 12), 0, Qt.AlignVCenter)
        second_name = QLabel(self.tr("不可靠帧 · 第 {0} 帧").format(self.ISSUE_FRAME))
        second_name.setObjectName("objectTitle")
        setFont(second_name, 12, QFont.Bold)
        second_header.addWidget(second_name)
        second_header.addStretch()
        second_state = QLabel(self.tr("需要检查"))
        second_state.setObjectName("warningBadge")
        second_header.addWidget(second_state)
        second_layout.addLayout(second_header)
        second_layout.addWidget(self._pill_label(self.tr("遮挡导致边界模糊，建议跳转到对应帧进行人工修正或重跑。"), "warningTextPill"))
        issue_actions = QHBoxLayout()
        issue_actions.setSpacing(8)
        jump = QPushButton(self.tr("跳转到修复帧"))
        jump.setObjectName("smallNeutralButton")
        self._set_button_icon(jump, FluentIcon.SEARCH)
        jump.setFixedHeight(36)
        jump.clicked.connect(self._jump_to_issue_frame)
        repair = QPushButton(self.tr("修复当前帧"))
        repair.setObjectName("amberButton")
        repair.setFixedHeight(36)
        repair.clicked.connect(self._show_inference_notice)
        issue_actions.addWidget(jump)
        issue_actions.addWidget(repair)
        second_layout.addLayout(issue_actions)
        self.issue_cards_layout.addWidget(second)
        self.issue_cards_layout.addStretch()
        self.issue_scroll.setWidget(issue_cards)
        layout.addWidget(self.issue_scroll, 1)
        return panel

    def _prompt_chip(self, text: str, value: str) -> QPushButton:
        chip = QPushButton(text)
        chip.setObjectName("promptChip")
        chip.setCursor(Qt.PointingHandCursor)
        chip.clicked.connect(lambda _checked=False, t=value: self._append_prompt_text(t))
        return chip

    def _append_prompt_text(self, text: str):
        current = self.prompt_input.text().strip()
        tokens = [t.strip() for t in current.split(",") if t.strip()]
        if text in tokens:
            return
        tokens.append(text)
        self.prompt_input.setText(", ".join(tokens))

    def _section_title(self, text: str, badge: str = ""):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        setFont(label, 12, QFont.DemiBold)
        layout.addWidget(label)
        layout.addStretch()
        if badge:
            badge_label = QLabel(badge)
            badge_label.setObjectName("sectionBadge")
            setFont(badge_label, 10)
            layout.addWidget(badge_label)
        return row

    def _square_plus_minus_icon(self, positive: bool) -> QIcon:
        pixmap = QPixmap(18, 18)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#111827"))
        painter.drawRect(2, 2, 14, 14)
        painter.setPen(QPen(QColor("#ffffff"), 2, Qt.SolidLine))
        painter.drawLine(5, 9, 13, 9)
        if positive:
            painter.drawLine(9, 5, 9, 13)
        painter.end()
        return QIcon(pixmap)

    def _color_title_icon(self, primary: str, secondary: str, accent: str) -> QPixmap:
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(primary))
        painter.drawRoundedRect(1, 1, 14, 14, 4, 4)
        painter.setBrush(QColor(accent))
        painter.drawRect(4, 4, 8, 8)
        painter.setBrush(QColor("#ffffff"))
        painter.drawRect(7, 3, 2, 10)
        painter.drawRect(3, 7, 10, 2)
        painter.setBrush(QColor(secondary))
        painter.drawRect(4, 4, 8, 8)
        painter.end()
        return pixmap

    def _model_card(self, title: str, description: str, badge: str, model: str):
        button = QPushButton()
        button.setObjectName(f"modelCard_{model}")
        button.setCheckable(True)
        button.setCursor(Qt.PointingHandCursor)
        button.setMinimumHeight(66)
        layout = QVBoxLayout(button)
        layout.setContentsMargins(12, 8, 10, 8)
        layout.setSpacing(2)
        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        name = QLabel(title)
        name.setObjectName("modelTitle")
        setFont(name, 12, QFont.Bold)
        title_row.addWidget(name)
        title_row.addStretch()
        tag = QLabel(badge)
        tag.setObjectName("modelBadge")
        title_row.addWidget(tag)
        description_label = QLabel(description)
        description_label.setObjectName("modelDescription")
        description_label.setWordWrap(True)
        setFont(description_label, 10)
        layout.addLayout(title_row)
        layout.addWidget(description_label)
        button.setChecked(model == "sam3")
        return button

    def _detail_row(self, label_text: str, value_text: str, value_attr: str | None = None):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(label_text)
        label.setObjectName("detailLabel")
        value = QLabel(value_text)
        value.setObjectName("detailValue")
        value.setWordWrap(True)
        if value_attr:
            setattr(self, value_attr, value)
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(value)
        return row

    def _color_dot(self, color: str, size: int):
        dot = QLabel()
        dot.setFixedSize(size, size)
        dot.setStyleSheet(f"background-color: {color}; border-radius: {size / 2:g}px;")
        return dot

    def _legend_dot(self, color: str, text: str):
        item = QWidget()
        layout = QHBoxLayout(item)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._color_dot(color, 9))
        label = QLabel(text)
        label.setObjectName("legendText")
        setFont(label, 12)
        label.setToolTip(text)
        self._playback_legend_labels.append(label)
        layout.addWidget(label)
        item.setToolTip(text)
        return item

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "preview_panel"):
            self._update_playback_responsiveness()

    def _update_playback_responsiveness(self):
        compact = self.preview_panel.width() < 600
        if self._playback_compact == compact:
            return
        self._playback_compact = compact
        for label in self._playback_legend_labels:
            label.setVisible(not compact)
        self._update_frame_counter()

    def _update_frame_counter(self):
        full_text = f"Frame {self.current_frame} / {self.total_frames}"
        self.frame_counter.setText(f"{self.current_frame} / {self.total_frames}" if self._playback_compact else full_text)
        self.frame_counter.setToolTip(full_text)

    def _apply_styles(self):
        self.setStyleSheet(VIDEO_OBJECT_DETECTOR_STYLESHEET)
        self._select_prompt_tool("pos")

    def _select_model(self, model: str):
        if model not in ("sam3", "edgetam"):
            return
        self.current_model = model
        self.model_sam3.setChecked(model == "sam3")
        self.model_edgetam.setChecked(model == "edgetam")
        self.sam3_params.setVisible(model == "sam3")
        self.edgetam_params.setVisible(model == "edgetam")
        self.result_model_badge.setText(self.tr("精确检测输出" if model == "sam3" else "快速检测输出"))
        self._update_frame()

    def _select_view(self, mode: str):
        if mode not in self.view_buttons:
            return
        self.current_view = mode
        for key, button in self.view_buttons.items():
            button.setChecked(key == mode)
        self.canvas.set_view_mode(mode)
        self._update_frame()

    def set_frame_detection_result(self, frame: int, score: float | None, objects: dict[str, int]):
        self._frame_detection_results[frame] = (score, objects.copy())
        if frame == self.current_frame:
            self._update_frame_detection_summary()

    def _update_frame_detection_summary(self):
        self.result_frame_label.setText(self.tr("第 {0} 帧检测结果").format(self.current_frame))
        result = self._frame_detection_results.get(self.current_frame)
        if result is None:
            self.result_score_value.setText("检测完成后显示")
            self.result_objects_value.setText(self.tr("检测完成后显示"))
            self.result_count_value.setText("检测完成后显示")
            self.result_state_label.setText(self.tr("待检测"))
            return
        score, objects = result
        self.result_score_value.setText("—" if score is None else f"{score:.3f}")
        object_count = sum(max(0, count) for count in objects.values())
        object_text = "、".join(f"{name} × {count}" for name, count in objects.items() if count > 0)
        self.result_objects_value.setText(object_text or self.tr("未检测到对象"))
        self.result_count_value.setText(str(object_count))
        self.result_state_label.setText(self.tr("检测完成"))

    def _select_prompt_tool(self, tool: str):
        if tool not in ("pos", "neg"):
            return
        self.prompt_tool = tool
        self.positive_btn.setChecked(tool == "pos")
        self.negative_btn.setChecked(tool == "neg")
        self.canvas.set_prompt_mode(tool)

    def _on_prompt_added(self, prompt):
        prompt_mode = prompt[1]
        label = self.tr("正向") if prompt_mode == "pos" else self.tr("负向")
        self._set_label_status(self.canvas_status, self.tr("●  已添加{0}画面提示").format(label), "ready")

    def _update_prompt_actions(self, can_undo: bool, can_delete: bool):
        self.undo_prompt_button.setEnabled(can_undo)
        self.delete_prompt_button.setEnabled(can_delete)

    def _on_range_start_changed(self, frame: int, range_end: QSpinBox | None = None):
        if range_end is None:
            range_end = self.range_end
        if frame > range_end.value():
            range_end.setValue(frame)

    def _on_range_end_changed(self, frame: int, range_start: QSpinBox | None = None):
        if range_start is None:
            range_start = self.range_start
        if frame < range_start.value():
            range_start.setValue(frame)

    def _jump_to_issue_frame(self):
        self.timeline.setValue(min(self.ISSUE_FRAME, self.timeline.maximum()))

    @staticmethod
    def _set_label_status(label: QLabel, text: str, state: str):
        label.setText(text)
        label.setProperty("state", state)
        label.style().unpolish(label)
        label.style().polish(label)
        viewport = label.parentWidget().parentWidget()
        if isinstance(viewport, VideoPreviewViewport):
            viewport.refresh_layout()

    def _select_direction(self, direction: str):
        button_groups = (self.direction_buttons, self.edge_direction_buttons)
        if not any(direction in buttons for buttons in button_groups):
            return
        for buttons in button_groups:
            for key, button in buttons.items():
                button.setChecked(key == direction)

    @staticmethod
    def _set_button_icon(button: QPushButton, icon: FluentIcon, dark: bool = False):
        button.setIcon(icon.icon(Theme.DARK if dark else Theme.LIGHT))
        button.setIconSize(QSize(16, 16))

    def _sync_frame_controls(self, reset_range: bool = False):
        self.current_frame = 1 if reset_range else min(1, self.total_frames)
        self.timeline.blockSignals(True)
        self.timeline.setRange(1, self.total_frames)
        self.timeline.setValue(self.current_frame)
        self.timeline.blockSignals(False)

        for range_start, range_end in ((self.range_start, self.range_end), (self.edge_range_start, self.edge_range_end)):
            range_start.setRange(1, self.total_frames)
            if reset_range:
                range_start.setValue(1)
            else:
                range_start.setValue(min(range_start.value(), self.total_frames))
            range_end.setRange(1, self.total_frames)
            range_end.setValue(self.total_frames)
        self.anchor_spin.setRange(1, self.total_frames)
        self.anchor_spin.setValue(self.current_frame if reset_range else min(self.anchor_spin.value(), self.total_frames))
        self.edge_anchor_spin.setRange(1, self.total_frames)
        self.edge_anchor_spin.setValue(self.current_frame if reset_range else min(self.edge_anchor_spin.value(), self.total_frames))
        self._update_frame_counter()

    def _load_video(self, file_path: str):
        self._stop_playback()
        self.canvas.clear_prompts()
        self.set_task_progress(0)
        if self.cap:
            self.cap.release()
            self.cap = None

        self.file_path = file_path
        self.cap = open_video_capture(file_path)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.file_path = None
            self.total_frames = 120
            self._sync_frame_controls(reset_range=True)
            self._update_video_info_pill(None)
            self._update_project_subtitle(None)
            message = self.tr("无法读取该视频，请检查文件格式或文件是否损坏。")
            self.canvas.set_frame(QImage(), message)
            self._set_label_status(self.canvas_status, self.tr("●  无法预览视频"), "error")
            return

        reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = reported_fps if reported_fps > 0 else 30.0
        reported_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.total_frames = max(1, int(reported_frames)) if reported_frames > 0 else 1
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._sync_frame_controls()
        self._update_video_info_pill((width, height) if width and height else None, self.fps, self.total_frames)
        self._update_project_subtitle(file_path)
        self._set_label_status(self.canvas_status, self.tr("●  视频已加载"), "ready")
        self._update_frame()

    def _on_timeline_changed(self, frame: int):
        self.current_frame = frame
        self._update_frame()

    def _update_frame(self):
        if self.current_view == "original":
            capture = self.cap
            missing_placeholder = self.tr("打开视频以预览画面")
        else:
            capture = self.result_video_captures[self.current_model][self.current_view]
            if capture is None:
                self.canvas.set_frame(QImage(), self.tr("检测完成后显示"))
                self._update_frame_counter()
                self._update_frame_detection_summary()
                return
            missing_placeholder = self.tr("当前帧暂无检测结果")

        if capture is not None and capture.isOpened():
            capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame - 1)
            success, frame = capture.read()
            if not success:
                self.canvas.set_frame(QImage(), missing_placeholder)
                if self.current_view == "original":
                    self._set_label_status(self.canvas_status, self.tr("●  当前帧读取失败"), "error")
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb.shape
                image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888,).copy()
                self.canvas.set_frame(image, show_detection_overlay=False, show_prompts=self.current_view == "original")
        else:
            self.canvas.set_frame(QImage(), missing_placeholder)

        self._update_frame_counter()
        self._update_frame_detection_summary()

    def _step_frame(self, delta: int = 1):
        next_frame = max(
            self.timeline.minimum(),
            min(self.timeline.maximum(), self.current_frame + delta),
        )
        self.timeline.setValue(next_frame)

    def _advance_playback(self):
        if self.current_frame >= self.timeline.maximum():
            self._stop_playback()
            return
        self._step_frame(1)

    def _toggle_play(self):
        if self.is_playing:
            self._stop_playback()
            return
        if self.current_frame >= self.timeline.maximum():
            self.timeline.setValue(self.timeline.minimum())
        self.is_playing = True
        self._set_button_icon(self.play_button, FluentIcon.PAUSE, dark=True)
        interval = max(1, round(1000 / max(self.fps, 1)))
        self.play_timer.start(interval)

    def _stop_playback(self):
        self.is_playing = False
        if hasattr(self, "play_timer"):
            self.play_timer.stop()
        if hasattr(self, "play_button"):
            self._set_button_icon(self.play_button, FluentIcon.PLAY, dark=True)

    def _show_inference_notice(self):
        self._set_label_status(self.canvas_status, self.tr("●  当前为界面预览，尚未接入推理算法"), "warning")

    def closeEvent(self, event):
        self._stop_playback()
        if self.cap:
            self.cap.release()
            self.cap = None
        for model_captures in self.result_video_captures.values():
            for view, capture in model_captures.items():
                if capture is not None:
                    capture.release()
                    model_captures[view] = None
        super().closeEvent(event)


VIDEO_OBJECT_DETECTOR_STYLESHEET = """
            QDialog#videoObjectDetector { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8fafc, stop:1 #eef2ff); color: #0f172a; }
            QPushButton { font-size: 12px; font-weight: 500; }
            #topBar {
                background: #f3f3f3;
                border-bottom: 1px solid rgba(0, 0, 0, 0.08);
            }
            #headerToolbar {
                background: rgba(255, 255, 255, 0.85);
                border: 1px solid rgba(0, 0, 0, 0.10);
                border-radius: 12px;
            }
            #toolbarSeparator, #videoInfoDivider {
                background: #e2e8f0;
            }
            #videoInfoPill {
                background: rgba(241, 245, 249, 0.8);
                border: 1px solid rgba(226, 232, 240, 0.6);
                border-radius: 8px;
            }
            #videoInfoTitle {
                color: #64748b;
                font-size: 11px;
                font-weight: 500;
            }
            #videoInfoValue {
                color: #1e293b;
                font-size: 11px;
                font-weight: 600;
            }
            #videoInfoSecondary {
                color: #64748b;
                font-size: 11px;
            }
            #taskStatusIcon {
                color: #94a3b8;
                font-size: 11px;
            }
            #taskStatusIconRunning {
                color: #2563eb;
                font-size: 11px;
            }
            #taskStatusIconTerminated {
                color: #dc2626;
                font-size: 11px;
            }
            #taskStatusLabel {
                color: #64748b;
                font-size: 11px;
                font-weight: 600;
            }
            #taskStatusLabelRunning {
                color: #334155;
                font-size: 11px;
                font-weight: 600;
            }
            #taskStatusLabelTerminated {
                color: #dc2626;
                font-size: 11px;
                font-weight: 600;
            }
            #taskProgressPercent {
                color: #2563eb;
                font-size: 11px;
                font-weight: 600;
            }
            #taskProgressBar {
                background: #e2e8f0;
                border: none;
                border-radius: 3px;
            }
            #taskProgressBar::chunk {
                background: #2563eb;
                border-radius: 3px;
            }
            #taskFrameCounter {
                color: #64748b;
                font-family: monospace;
                font-size: 10px;
            }
            #deviceTag {
                color: #4338ca;
                background: rgba(238, 242, 255, 0.8);
                border: 1px solid rgba(199, 210, 254, 0.7);
                border-radius: 6px;
            }
            #deviceLabel {
                color: #4338ca;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton#terminateButton {
                color: #dc2626;
                background: #fef2f2;
                border: 1px solid rgba(254, 202, 202, 0.8);
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton#terminateButton:hover:enabled { background: #fee2e2; }
            QPushButton#terminateButton:pressed:enabled { background: #fecaca; }
            QPushButton#terminateButton:disabled {
                color: #94a3b8;
                background: #f1f5f9;
                border-color: #e2e8f0;
            }
            QPushButton#terminateButton:disabled::icon {
                opacity: 0.45;
            }
            #brandIcon {
                color: #fff;
                border-radius: 11px;
                background: qlineargradient(x1:0, y1:1, x2:1, y2:0, stop:0 #4f46e5, stop:1 #7c3aed);
            }
            #brandTitle, #panelHeading, #sectionTitle, #objectTitle { color: #1e293b; }
            #brandSubtitle { color: #94a3b8; }

            #leftPanel, #rightPanel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #f8fafc);
            }
            #leftPanel { border-right: 1px solid #e2e8f0; }
            #rightPanel { border-left: 1px solid #e2e8f0; }
            #panelHeader, #panelBody { background: transparent; }
            #panelHeader {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(248,250,252,0.9), stop:1 rgba(241,245,249,0.9));
                border-bottom: 1px solid #edf2f7;
            }

            QScrollBar:vertical { width: 7px; background: transparent; margin: 4px 1px; }
            QScrollBar::handle:vertical { background: #cbd5e1; border-radius: 3px; min-height: 28px; }
            QScrollBar::handle:vertical:hover { background: #94a3b8; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

            #modelCard_sam3, #modelCard_edgetam {
                text-align: left;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(255,255,255,0.92), stop:1 rgba(248,250,252,0.94));
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 2px;
            }
            #modelCard_sam3:hover, #modelCard_edgetam:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(255,255,255,0.98), stop:1 rgba(241,245,249,0.98));
                border-color: #cbd5e1;
            }
            #modelCard_sam3:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(238,242,255,0.96), stop:1 rgba(224,231,255,0.96));
                border: 1px solid #a5b4fc;
            }
            #modelCard_edgetam:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(245,243,255,0.96), stop:1 rgba(237,233,254,0.96));
                border: 1px solid #c4b5fd;
            }
            #modelTitle { color: #1e293b; }
            #modelDescription { color: #64748b; }
            #modelCard_sam3:checked #modelTitle { color: #312e81; }
            #modelCard_sam3:checked #modelDescription { color: #4f46e5; }
            #modelCard_edgetam:checked #modelTitle { color: #4c1d95; }
            #modelCard_edgetam:checked #modelDescription { color: #7c3aed; }

            #modelBadge {
                background: rgba(241, 245, 249, 0.9);
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 9px;
                font-weight: 600;
                color: #64748b;
            }
            #modelCard_sam3:checked #modelBadge {
                background: rgba(99, 102, 241, 0.12);
                border-color: rgba(99, 102, 241, 0.35);
                color: #4338ca;
            }
            #modelCard_edgetam:checked #modelBadge {
                background: rgba(139, 92, 246, 0.12);
                border-color: rgba(139, 92, 246, 0.35);
                color: #6d28d9;
            }
            #sectionBadge {
                color: #6366f1;
                background: rgba(238, 242, 255, 0.75);
                border: 1px solid rgba(165, 180, 252, 0.45);
                border-radius: 6px;
                padding: 2px 8px;
            }
            #indigoBadge {
                color: #4f46e5;
                background: rgba(238, 242, 255, 0.8);
                border: 1px solid rgba(165, 180, 252, 0.45);
                border-radius: 6px;
                padding: 2px 8px;
            }
            #panelTitleIcon {
                background: transparent;
            }
            #successBadge {
                color: #047857;
                background: rgba(236, 253, 245, 0.9);
                border: 1px solid rgba(167, 243, 208, 0.85);
                border-radius: 8px;
                padding: 3px 10px;
                font-size: 10px;
                font-weight: 600;
            }
            #warningBadge {
                color: #a16207;
                background: rgba(254, 243, 199, 0.9);
                border: 1px solid rgba(253, 230, 138, 0.85);
                border-radius: 8px;
                padding: 3px 10px;
                font-size: 10px;
                font-weight: 600;
            }

            #neutralButton, #smallNeutralButton {
                color: #334155;
                background: #fff;
                border: 1px solid #dbe3ef;
                border-radius: 12px;
                padding: 0 12px;
            }
            #neutralButton:hover, #smallNeutralButton:hover {
                background: #f8fafc;
                border-color: #cbd5e1;
            }
            #neutralButton:pressed, #smallNeutralButton:pressed { background: #eef2f7; }
            #primaryButton {
                color: #fff;
                background: #4f46e5;
                border: 1px solid #4f46e5;
                border-radius: 12px;
                padding: 0 14px;
                font-weight: 600;
            }
            #primaryButton:hover { background: #4338ca; border-color: #4338ca; }
            #primaryButton:pressed { background: #3730a3; }

            #browseButton {
                color: #475569;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
            }
            #browseButton:hover { background: #f1f5f9; border-color: #cbd5e1; }
            #browseButton:pressed { background: #e2e8f0; }

            QLineEdit#textInput, QSpinBox#textInput, QComboBox#textInput {
                min-height: 30px;
                max-height: 30px;
                color: #1e293b;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 0 12px;
                font-size: 12px;
                selection-background-color: #6366f1;
            }
            QLineEdit#textInput:hover, QSpinBox#textInput:hover, QComboBox#textInput:hover {
                background: #fff;
                border-color: #cbd5e1;
            }
            QLineEdit#textInput:focus, QSpinBox#textInput:focus, QComboBox#textInput:focus {
                background: #fff;
                border: 1px solid #818cf8;
            }
            QSpinBox#anchorFrameInput, QSpinBox#frameRangeInput {
                padding: 0 18px 0 10px;
                border-radius: 10px;
                color: #1e293b;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                font-size: 12px;
                selection-background-color: #6366f1;
            }
            QSpinBox#anchorFrameInput:hover, QSpinBox#frameRangeInput:hover {
                background: #fff;
                border-color: #cbd5e1;
            }
            QSpinBox#anchorFrameInput:focus, QSpinBox#frameRangeInput:focus {
                background: #fff;
                border-color: #818cf8;
            }
            QSpinBox#anchorFrameInput:disabled, QSpinBox#frameRangeInput:disabled {
                color: #94a3b8;
                background: #f1f5f9;
                border-color: #e2e8f0;
            }
            QSpinBox#anchorFrameInput::up-button,
            QSpinBox#anchorFrameInput::down-button,
            QSpinBox#frameRangeInput::up-button,
            QSpinBox#frameRangeInput::down-button {
                subcontrol-origin: border;
                width: 13px;
                height: 12px;
                border: none;
                background: transparent;
            }
            QSpinBox#anchorFrameInput::up-button,
            QSpinBox#frameRangeInput::up-button {
                subcontrol-position: top right;
                top: 2px;
                right: 2px;
            }
            QSpinBox#anchorFrameInput::down-button,
            QSpinBox#frameRangeInput::down-button {
                subcontrol-position: bottom right;
                bottom: 2px;
                right: 2px;
            }
            QSpinBox#anchorFrameInput::up-button:hover,
            QSpinBox#anchorFrameInput::down-button:hover,
            QSpinBox#frameRangeInput::up-button:hover,
            QSpinBox#frameRangeInput::down-button:hover {
                background: rgba(99, 102, 241, 0.08);
                border-radius: 6px;
            }
            QSpinBox#anchorFrameInput::up-arrow,
            QSpinBox#frameRangeInput::up-arrow {
                image: url(:/qfluentwidgets/images/icons/CareUpSolid_black.svg);
                width: 7px;
                height: 7px;
            }
            QSpinBox#anchorFrameInput::down-arrow,
            QSpinBox#frameRangeInput::down-arrow {
                image: url(:/qfluentwidgets/images/icons/CareDownSolid_black.svg);
                width: 7px;
                height: 7px;
            }

            QComboBox#textInput {
                padding-right: 20px;
                border-radius: 10px;
                combobox-popup: 0;
            }
            QComboBox#textInput::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 16px;
                border: none;
                background: transparent;
                margin-right: 2px;
            }
            QComboBox#textInput::down-arrow {
                image: url(:/qfluentwidgets/images/icons/ChevronDown_black.svg);
                width: 8px;
                height: 8px;
                margin-right: 2px;
            }
            QComboBox#textInput:hover::drop-down {
                background: transparent;
            }
            QComboBox#textInput QAbstractItemView {
                color: #1e293b;
                background: #fff;
                border: 1px solid #dbe3ef;
                border-radius: 10px;
                selection-background-color: #eef2ff;
                selection-color: #4338ca;
                outline: none;
                padding: 4px;
            }
            QComboBox#textInput QAbstractItemView::item {
                min-height: 28px;
                border-radius: 6px;
                padding: 0 8px;
            }

            QLineEdit#promptInput {
                min-height: 34px;
                max-height: 34px;
                color: #1e293b;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 0 12px;
                selection-background-color: #6366f1;
            }
            QLineEdit#promptInput:hover { background: #fff; border-color: #cbd5e1; }
            QLineEdit#promptInput:focus { background: #fff; border: 1px solid #818cf8; }

            QPushButton#promptChip {
                color: #475569;
                background: rgba(241, 245, 249, 0.9);
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 0 10px;
            }
            QPushButton#promptChip:hover {
                color: #4338ca;
                background: rgba(238, 242, 255, 0.95);
                border-color: #a5b4fc;
            }
            QPushButton#promptChip:pressed {
                background: rgba(224, 231, 255, 0.95);
                border-color: #818cf8;
            }

            #helperText {
                color: #64748b;
                background: #f1f5f9;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
            }
            #infoLabel {
                color: #64748b;
                background: #f1f5f9;
                border: none;
                border-radius: 8px;
            }
            #infoIcon {
                color: #ffffff;
                background: #7c3aed;
                border-radius: 6px;
            }
            #warningBox {
                color: #92400e;
                background: rgba(255, 251, 235, 0.95);
                border: 1px solid rgba(253, 230, 138, 0.9);
                border-radius: 10px;
                padding: 6px 8px;
                font-size: 10px;
            }
            #warningTextPill {
                color: rgba(120, 53, 15, 0.85);
                background: rgba(254, 243, 199, 0.65);
                border: 1px solid rgba(253, 230, 138, 0.75);
                border-radius: 8px;
                padding: 5px 8px;
                font-size: 11px;
            }
            #sideNote {
                color: #64748b;
                background: rgba(241, 245, 249, 0.9);
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 10px 14px;
                font-size: 11px;
            }
            #inlineLabel { color: #64748b; font-size: 12px; }

            QPushButton#positiveButton, QPushButton#negativeButton {
                border-radius: 12px;
                padding: 0 10px;
                font-weight: 600;
                color: #334155;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
            }
            QPushButton#positiveButton {
                qproperty-iconSize: 14px;
            }
            QPushButton#negativeButton {
                qproperty-iconSize: 14px;
            }
            QPushButton#positiveButton:checked {
                color: #047857;
                background: #ecfdf5;
                border: 1px solid #a7f3d0;
            }
            QPushButton#positiveButton:checked:hover { background: #d1fae5; }
            QPushButton#negativeButton:checked {
                color: #b91c1c;
                background: #fef2f2;
                border: 1px solid #fecaca;
            }
            QPushButton#negativeButton:checked:hover { background: #fee2e2; }
            QPushButton#positiveButton:!checked:hover,
            QPushButton#negativeButton:!checked:hover {
                background: #f1f5f9;
                border-color: #cbd5e1;
            }

            #segmentedGroup {
                background: rgba(241, 245, 249, 0.75);
                border: 1px solid #e2e8f0;
                border-radius: 12px;
            }
            QPushButton#segmentedButton {
                color: #64748b;
                background: transparent;
                border: 1px solid transparent;
                border-radius: 9px;
                padding: 5px 4px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton#segmentedButton:hover:!checked {
                color: #334155;
                background: rgba(255, 255, 255, 0.55);
            }
            QPushButton#segmentedButton:checked {
                color: #4338ca;
                background: #ffffff;
                border-color: #e0e7ff;
                font-weight: 600;
            }

            #violetButton {
                color: #fff;
                background: #7c3aed;
                border: 1px solid #7c3aed;
                border-radius: 12px;
                padding: 0 12px;
                font-weight: 600;
            }
            #violetButton:hover { background: #6d28d9; border-color: #6d28d9; }
            #amberButton {
                color: #fff;
                background: #d97706;
                border: 1px solid #d97706;
                border-radius: 12px;
                padding: 0 12px;
                font-weight: 600;
            }
            #amberButton:hover { background: #b45309; border-color: #b45309; }

            #previewPanel { background: #0f172a; }
            #viewGroup {
                background: rgba(30, 41, 59, 0.65);
                border: 1px solid rgba(51, 65, 85, 0.45);
                border-radius: 12px;
            }
            QPushButton#promptActionButton {
                color: #cbd5e1;
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 5px 10px;
                font-size: 11px;
            }
            QPushButton#promptActionButton:hover:enabled {
                color: #ffffff;
                background: #334155;
            }
            QPushButton#promptActionButton:disabled {
                color: #64748b;
                background: #172033;
                border-color: #273449;
            }
            QPushButton#viewButton {
                color: #94a3b8;
                background: transparent;
                border: none;
                border-radius: 9px;
                padding: 6px 14px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton#viewButton:hover:!checked {
                color: #e2e8f0;
                background: rgba(51, 65, 85, 0.5);
            }
            QPushButton#viewButton:checked {
                color: #ffffff;
                background: #6366f1;
                border: 1px solid #818cf8;
                font-weight: 600;
            }
            #canvasStatus {
                color: #fff;
                background: rgba(15, 23, 42, 0.8);
                border: 1px solid rgba(51, 65, 85, 0.6);
                border-radius: 10px;
                padding: 5px 10px;
            }
            #videoCanvas {
                background: #000000;
                border: 1px solid #1e293b;
                border-radius: 16px;
            }

            #playbackPanel { background: #0f172a; border-top: 1px solid #1e293b; }

            #timelineSlider {
                background: transparent;
                min-height: 22px;
            }
            #timelineSlider::groove:horizontal {
                height: 6px;
                background: #1e293b;
                border-radius: 3px;
            }
            #timelineSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #6366f1, stop:1 #a5b4fc);
                border-radius: 3px;
            }
            #timelineSlider::add-page:horizontal {
                background: #1e293b;
                border-radius: 3px;
            }
            #timelineSlider::handle:horizontal {
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
                background: #ffffff;
                border: 2px solid #6366f1;
            }
            #timelineSlider::handle:horizontal:hover {
                border-color: #a5b4fc;
            }

            #stepButton {
                color: #cbd5e1;
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 10px;
            }
            #stepButton:hover { color: #ffffff; background: #334155; }
            #playButton {
                color: #fff;
                background: #4f46e5;
                border: 1px solid #4f46e5;
                border-radius: 12px;
            }
            #playButton:hover { background: #6366f1; }
            #frameCounter {
                color: #cbd5e1;
                background: rgba(30, 41, 59, 0.75);
                border: 1px solid rgba(51, 65, 85, 0.6);
                border-radius: 8px;
                padding: 4px 10px;
                font-size: 12px;
            }
            #legendText { color: #cbd5e1; font-size: 12px; }

            #objectCard, #warningCard { border-radius: 14px; }
            #objectCard { background: #f8fafc; border: 1px solid #e2e8f0; }
            #objectCard:hover { border-color: #a5b4fc; }
            #issueScroll { background: #f1f5f9; border: none; }
            #issueCards { background: #f1f5f9; }
            #warningCard {
                background: rgba(255, 251, 235, 0.7);
                border: 1px solid #fde68a;
            }
            #cardDivider { background: rgba(226, 232, 240, 0.6); }
            #detailLabel { color: #94a3b8; font-size: 11px; }
            #detailValue {
                color: #334155;
                background: transparent;
                padding: 0;
                font-size: 11px;
            }
            #smallNeutralButton { background: #ffffff; padding: 0 8px; font-size: 12px; }
"""

TERMINATE_DIALOG_STYLESHEET = """
            QDialog#terminateDialog {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
            }
            #terminateAlertIcon {
                color: #dc2626;
                background: #fee2e2;
                border-radius: 17px;
                font-size: 18px;
                font-weight: 700;
            }
            #terminateDialogTitle {
                color: #1e293b;
                font-size: 14px;
                font-weight: 600;
            }
            #terminateDialogDescription {
                color: #475569;
                font-size: 11px;
            }
            QPushButton#terminateCancelButton {
                color: #334155;
                background: #f1f5f9;
                border: none;
                border-radius: 8px;
                padding: 6px 14px;
            }
            QPushButton#terminateCancelButton:hover { background: #e2e8f0; }
            QPushButton#terminateConfirmButton {
                color: #ffffff;
                background: #dc2626;
                border: none;
                border-radius: 8px;
                padding: 6px 14px;
            }
            QPushButton#terminateConfirmButton:hover { background: #b91c1c; }
            QPushButton#terminateConfirmButton:pressed { background: #991b1b; }
"""