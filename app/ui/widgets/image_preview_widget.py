from functools import lru_cache
import os
import ffmpeg
import platform
from PySide6.QtCore import (
    Signal, Qt, QTimer, QRect, Property, QEasingCurve, QPropertyAnimation, 
    QThreadPool, QRunnable, QBuffer, QIODevice
)
from PySide6.QtWidgets import (
    QGraphicsView, QWidget , QVBoxLayout, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem, 
    QScrollBar, QProgressBar, QPushButton, QFrame, QHBoxLayout, QScrollArea, QSizePolicy
)
from PySide6.QtGui import QPixmap, QWheelEvent, QColor, QPainter, QBrush, QPen, QLinearGradient
from app.ui.library.qfluentwidgets import setFont, qconfig, Theme 
from app.ui.common.event_bus import global_event_bus


class ScrollBar(QScrollBar):
    def __init__(self, orientation=Qt.Vertical, parent=None):
        super().__init__(orientation, parent)
        self._opacity = 1.0
        self._hover = False
        self._fade = False
        self._fade_timer = QTimer(self)
        self._fade_timer.setInterval(1500)
        self._fade_timer.timeout.connect(self._fade_out)

        self._fade_anim = QPropertyAnimation(self, b"opacity")
        self._fade_anim.setDuration(300)
        self._fade_anim.setEasingCurve(QEasingCurve.OutCubic)

        self.setMouseTracking(True)
        self.setStyleSheet("QScrollBar { background: transparent; }")
        self.setMinimumWidth(8)
        self.setMinimumHeight(8)

        self._update_colors(qconfig.theme)
        qconfig.themeChanged.connect(self._update_colors)

    def get_opacity(self):
        return self._opacity

    def set_opacity(self, value):
        self._opacity = value
        self.update()

    opacity = Property(float, get_opacity, set_opacity)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制背景
        if self.orientation() == Qt.Vertical:
            rect = QRect(self.width() // 2 - 3, 0, 6, self.height())
        else:
            rect = QRect(0, self.height() // 2 - 3, self.width(), 6)

        painter.setBrush(QBrush(self.bg_color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, 4, 4)

        # 滑块位置
        handle_rect = self._get_handle_rect()
        if not handle_rect.isNull():
            color = self.hover_color if self._hover else self.handle_color
            color.setAlphaF(self._opacity)
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(handle_rect, 4, 4)

    def _get_handle_rect(self):
        """计算滑块矩形"""
        total = self.maximum() - self.minimum() + self.pageStep()
        if total <= 0:
            return QRect()

        ratio = self.pageStep() / total
        if self.orientation() == Qt.Vertical:
            bar_len = self.height()
            handle_len = max(24, bar_len * ratio)
            pos_ratio = (self.value() - self.minimum()) / (self.maximum() - self.minimum() or 1)
            y = (bar_len - handle_len) * pos_ratio
            return QRect(self.width() // 2 - 3, int(y), 6, int(handle_len))
        else:
            bar_len = self.width()
            handle_len = max(24, bar_len * ratio)
            pos_ratio = (self.value() - self.minimum()) / (self.maximum() - self.minimum() or 1)
            x = (bar_len - handle_len) * pos_ratio
            return QRect(int(x), self.height() // 2 - 3, int(handle_len), 6)

    def enterEvent(self, event):
        super().enterEvent(event)
        self._hover = True
        self.update()
        if self._fade:
            self._fade_in()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._hover = False
        if self._fade:
            self._fade_timer.start()
        self.update()

    def mousePressEvent(self, event):
        if self._fade:
            self._fade_in()
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        if self._fade:
            self._fade_in()
            self._fade_timer.start()
        super().wheelEvent(event)

    def _fade_in(self):
        self._fade_timer.stop()
        self._fade_anim.stop()
        self._fade_anim.setStartValue(self._opacity)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.start()

    def _fade_out(self):
        self._fade_anim.stop()
        self._fade_anim.setStartValue(self._opacity)
        self._fade_anim.setEndValue(0.25)
        self._fade_anim.start()

    def setFade(self, enabled: bool):
        """启用/禁用自动隐藏"""
        self._fade = enabled
        if not enabled:
            self.set_opacity(1.0)
        else:
            self._fade_out()

    def _update_colors(self, theme: Theme):
        if theme == Theme.DARK:
            self.bg_color = QColor(255, 255, 255, 20)
            self.handle_color = QColor(255, 255, 255, 90)
            self.hover_color = QColor(255, 255, 255, 150)
        else:
            self.bg_color = QColor(0, 0, 0, 10)
            self.handle_color = QColor(120, 120, 120, 90)
            self.hover_color = QColor(80, 80, 80, 150)
        self.update()


class SyncGraphicsView(QGraphicsView):
    zoomChanged = Signal(float)
    scrollChanged = Signal(int, int)

    def __init__(self, pixmap: QPixmap = None, parent=None, sub_title: str = ""):
        super().__init__(parent)
        self.setRenderHints(self.renderHints() | self.renderHints().SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

        self._v_scroll = ScrollBar(Qt.Vertical, self)
        self._h_scroll = ScrollBar(Qt.Horizontal, self)
        self.setVerticalScrollBar(self._v_scroll)
        self.setHorizontalScrollBar(self._h_scroll)

        # 当滑条变化时发射滚动同步信号
        self.verticalScrollBar().valueChanged.connect(self._emit_scroll)
        self.horizontalScrollBar().valueChanged.connect(self._emit_scroll)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self._zoom = 1.0
        self._syncing_scroll = False
        self.pixmap_item = None
        self.placeholder = None
        self.sub_title = sub_title

        self._init_placeholder()
        if pixmap and not pixmap.isNull():
            self.set_pixmap(pixmap)

    def _init_placeholder(self):
        self.scene.clear()

        icon_item = QGraphicsTextItem("🖼️")
        icon_item.setDefaultTextColor(QColor("#cccccc"))
        setFont(icon_item, 32)
        self.scene.addItem(icon_item)

        # 主标题
        title = QGraphicsTextItem(self.tr("预览区域"))
        setFont(title, 16)
        title.setDefaultTextColor(QColor("#666666"))
        self.scene.addItem(title)

        # 副标题
        subtitle = QGraphicsTextItem(self.tr(self.sub_title) if self.sub_title else self.tr("处理完成后自动显示预览"))
        setFont(subtitle, 14)
        subtitle.setDefaultTextColor(QColor("#999999"))
        self.scene.addItem(subtitle)

        # 居中排布
        spacing = 8
        total_height = (
            icon_item.boundingRect().height()
            + spacing
            + title.boundingRect().height()
            + subtitle.boundingRect().height()
        )

        icon_item.setPos(-icon_item.boundingRect().width() / 2, -total_height / 2)
        title.setPos(-title.boundingRect().width() / 2, icon_item.pos().y() + icon_item.boundingRect().height() + spacing)
        subtitle.setPos(-subtitle.boundingRect().width() / 2, title.pos().y() + title.boundingRect().height() + 2)

        # 设置场景矩形（用于居中）
        self.scene.setSceneRect(-180, -135, 360, 270)

    def _center_placeholder(self):
        if not self.placeholder:
            return
        rect = self.scene.sceneRect()
        text_rect = self.placeholder.boundingRect()
        self.placeholder.setPos(
            rect.width() / 2 - text_rect.width() / 2,
            rect.height() / 2 - text_rect.height() / 2
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._center_placeholder()

    def set_pixmap(self, pixmap: QPixmap):
        self.scene.clear()
        if pixmap and not pixmap.isNull():
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.pixmap_item)
            self.scene.setSceneRect(self.pixmap_item.boundingRect())
        else:
            self._init_placeholder()

    def wheelEvent(self, event: QWheelEvent):
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        old_zoom = self._zoom
        self._zoom *= zoom_factor
        self._zoom = max(0.1, min(self._zoom, 10.0))

        scale_factor = self._zoom / old_zoom
        self.scale(scale_factor, scale_factor)

        self.zoomChanged.emit(self._zoom)

    def _emit_scroll(self):
        if not self._syncing_scroll:
            self.scrollChanged.emit(
                self.horizontalScrollBar().value(),
                self.verticalScrollBar().value()
            )

    def sync_scroll(self, x: int, y: int):
        self._syncing_scroll = True
        self.horizontalScrollBar().setValue(x)
        self.verticalScrollBar().setValue(y)
        self._syncing_scroll = False

    def sync_zoom(self, target_zoom: float):
        if abs(target_zoom - self._zoom) > 1e-3:
            scale_factor = target_zoom / self._zoom
            self._zoom = target_zoom
            self.scale(scale_factor, scale_factor)


class SyncImageViewer(QWidget):
    def __init__(self, img1: str = "", img2: str = "", parent=None):
        super().__init__(parent=parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        pix1 = QPixmap(img1) if img1 else None
        pix2 = QPixmap(img2) if img2 else None

        self.view1 = SyncGraphicsView(pix1, sub_title=self.tr("原图预览区域（处理完成后自动显示预览）"))
        self.view2 = SyncGraphicsView(pix2, sub_title=self.tr("添加水印后预览区域（处理完成后自动显示预览）"))

        layout.addWidget(self.view1)
        layout.addWidget(self.view2)

        # 信号互联（双向同步）
        self.view1.zoomChanged.connect(self.view2.sync_zoom)
        self.view2.zoomChanged.connect(self.view1.sync_zoom)
        self.view1.scrollChanged.connect(self.view2.sync_scroll)
        self.view2.scrollChanged.connect(self.view1.sync_scroll)

        self.setStyleSheet("""
            QWidget {
                background-color: #f3f3f3;
                border-radius: 8px;
            }
            QGraphicsView {
                background-color: #fafafa;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
            }
        """)

    def set_images(self, img1: str, img2: str):
        self.view1.set_pixmap(QPixmap(img1))
        self.view2.set_pixmap(QPixmap(img2))


class RoundedProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(6)
        self.setTextVisible(False)
        self.setRange(0, 100)
        self.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 3px;
                background-color: rgba(200, 200, 200, 0.2);
            }
            QProgressBar::chunk {
                border-radius: 3px;
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(102, 126, 234, 0.7),
                    stop:0.5 rgba(118, 75, 162, 0.8),
                    stop:1 rgba(102, 126, 234, 0.7));
            }
        """)


class AnimatedButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._scale = 1.0
        self._animation = QPropertyAnimation(self, b"scale", self)
        self._animation.setDuration(150)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)

    def get_scale(self):
        return self._scale

    def set_scale(self, scale):
        self._scale = scale
        self.update()

    scale = Property(float, get_scale, set_scale)

    def enterEvent(self, event):
        super().enterEvent(event)
        self._animation.setEndValue(1.08)
        self._animation.start()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._animation.setEndValue(1.0)
        self._animation.start()

    def mousePressEvent(self, event):
        self._animation.setEndValue(0.95)
        self._animation.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._animation.setEndValue(1.08)
        self._animation.start()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 应用缩放
        painter.translate(self.width() / 2, self.height() / 2)
        painter.scale(self._scale, self._scale)
        painter.translate(-self.width() / 2, -self.height() / 2)

        self.draw_background(painter)

        painter.setPen(QColor(85, 85, 85))
        painter.drawText(self.rect(), Qt.AlignCenter, self.text())

    def draw_background(self, painter):
        pass


class TransparentNavButton(AnimatedButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(32, 32)
        self.setCursor(Qt.PointingHandCursor)
        self._is_hovered = False

    def draw_background(self, painter):
        painter.setRenderHint(QPainter.Antialiasing)
        if self._is_hovered:
            fill_color = QColor(102, 126, 234, 60)
            border_color = QColor(102, 126, 234, 120)
        else:
            fill_color = QColor(102, 126, 234, 30)
            border_color = QColor(102, 126, 234, 80)

        painter.setBrush(fill_color)
        painter.setPen(border_color)
        painter.drawEllipse(self.rect().adjusted(2, 2, -2, -2))

    def enterEvent(self, event):
        self._is_hovered = True
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._is_hovered = False
        super().leaveEvent(event)


class LoaderWorker(QRunnable):
    def __init__(self, image_path, callback, media_type="image"):
        super().__init__()
        self.image_path = image_path
        self.callback = callback
        self.media_type = media_type

    def run(self):
        if not os.path.exists(self.image_path):
            return
        if self.media_type == "image":
            pix = QPixmap(self.image_path)
            if not pix.isNull():
                self.callback(pix)
        else:
            pix = self._extract_video_frame(video_path=self.image_path)
            if pix and not pix.isNull():
                self.callback(pix)

    @staticmethod
    @lru_cache(maxsize=512)
    def _extract_video_frame(video_path: str, size=(48, 48)) -> QPixmap:
        width, height = size

        ffmpeg_bin = os.getenv(
            "POWERTOOLS_FFMPEG_BIN", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "resources", "ffmpeg", "bin")
        )
        ffmpeg_exe = os.path.join(ffmpeg_bin, "ffmpeg.exe" if platform.system().lower() == "windows" else "ffmpeg")

        stream = (
            ffmpeg.input(video_path)
            .filter("scale", width, height, force_original_aspect_ratio="decrease")
            .output("pipe:1", vframes=1, format="image2pipe", vcodec="png")
            .global_args("-hide_banner", "-loglevel", "error")
        )
        out_bytes, _ = ffmpeg.run(
            stream,
            capture_stdout=True,
            capture_stderr=True,
            cmd=ffmpeg_exe,
            overwrite_output=True
        )

        pixmap = QPixmap()
        buffer = QBuffer()
        buffer.setData(out_bytes)
        buffer.open(QIODevice.ReadOnly)
        pixmap.loadFromData(buffer.data(), "PNG")
        buffer.close()

        return pixmap


class ThumbnailButton(AnimatedButton):
    thread_pool = QThreadPool.globalInstance()

    def __init__(self, index, image_path, media_type="image", parent=None):
        super().__init__("", parent)
        self.media_type = media_type
        self.index = index
        self.image_path = image_path
        self.pixmap = None
        self._is_hovered = False
        self.is_active = False
        self.setFixedSize(48, 48)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        scale_factor = self.scale
        rect = self.rect()
        center = rect.center()
        painter.save()
        painter.translate(center)
        painter.scale(scale_factor, scale_factor)
        painter.translate(-center)

        if self.pixmap:
            target_rect = self.rect().adjusted(3, 3, -3, -3)
            painter.drawPixmap(target_rect, self.pixmap)
            gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            gradient.setColorAt(0.0, QColor(255, 255, 255, 40))
            gradient.setColorAt(1.0, QColor(0, 0, 0, 60))
            painter.fillRect(target_rect, gradient)
        else:
            painter.setBrush(QColor(220, 220, 220, 50))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(self.rect().adjusted(3, 3, -3, -3), 6, 6)

            # 异步加载
            worker = LoaderWorker(self.image_path, self.on_loaded, media_type=self.media_type)
            self.thread_pool.start(worker)

        if self._is_hovered or self.is_active:
            overlay = QColor(255, 255, 255, 40 if self._is_hovered else 60)
            painter.setBrush(overlay)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 6, 6)

        painter.restore()

        if self.is_active:
            pen = QPen(QColor(128, 150, 255, 128), 6)  # 蓝色边框
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 6, 6)

    def on_loaded(self, pixmap: QPixmap):
        if not pixmap or pixmap.isNull():
            return
        self.pixmap = pixmap.scaled(
            self.width() - 6,
            self.height() - 6,
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation
        )
        self.update()

    def enterEvent(self, event):
        self._is_hovered = True
        if not self.is_active:
            super().enterEvent(event)

    def leaveEvent(self, event):
        self._is_hovered = False
        if not self.is_active:
            super().leaveEvent(event)

    def set_active(self, active):
        if self.is_active != active:
            self.is_active = active
            self._animation.setEndValue(1.1 if active else 1.0)
            self._animation.start()
            self.update()


class ImageNavigationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ImageNavigationWidget")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.total_images = []
        self.current_index = 0
        self.setup_ui()
        self.update_display()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 5, 10, 10)

        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(10)

        self.prev_btn = TransparentNavButton("◀")
        self.prev_btn.clicked.connect(self.prev_image)

        self.progress_bar = RoundedProgressBar()

        self.next_btn = TransparentNavButton("▶")
        self.next_btn.clicked.connect(self.next_image)

        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.progress_bar)
        nav_layout.addWidget(self.next_btn)

        main_layout.addLayout(nav_layout)

        thumbnail_frame = QFrame()
        thumbnail_frame.setObjectName("thumbnailFrame")
        thumbnail_frame.setStyleSheet("""
            QFrame#thumbnailFrame {
                background-color: rgba(250, 251, 252, 0.95);
                border-radius: 10px;
                border: 1px solid rgba(220, 220, 220, 0.5);
            }
        """)
        self.thumbnail_layout = QHBoxLayout(thumbnail_frame)
        self.thumbnail_layout.setSpacing(15)
        self.thumbnail_layout.setContentsMargins(10, 6, 10, 6)
        self.thumbnail_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_area.setHorizontalScrollBar(ScrollBar(Qt.Horizontal, scroll_area))
        scroll_area.horizontalScrollBar().setFade(True)

        scroll_area.setWidget(thumbnail_frame)
        
        main_layout.addWidget(scroll_area)

        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("""
            ImageNavigationWidget {
                background-color: white;
                border-radius: 8px;
                border: 1px solid rgba(200, 200, 200, 0.3);
            }
        """)

    def load_images(self, image_paths, media_type="image"):
        if not image_paths:
            return
        old_images_num = len(self.total_images)
        self.total_images.extend(image_paths)
        for i, path in enumerate(image_paths):
            i += old_images_num
            thumb = ThumbnailButton(i, path, media_type=media_type)
            thumb.clicked.connect(lambda checked, idx=i: self.go_to_image(idx))
            self.thumbnail_layout.insertWidget(self.thumbnail_layout.count() - 1, thumb)
        self.current_index = 0
        self.update_display()

    def clear_images(self):
        for i in reversed(range(self.thumbnail_layout.count() - 1)):
            widget = self.thumbnail_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.total_images.clear()
        self.current_index = 0
        self.update_display()

    def update_display(self):
        if not self.total_images:
            self.progress_bar.setValue(0)
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return

        progress = int(((self.current_index + 1) / len(self.total_images)) * 100)
        self.progress_bar.setValue(progress)
        for i in range(self.thumbnail_layout.count() - 1):
            thumb = self.thumbnail_layout.itemAt(i).widget()
            if thumb:
                thumb.set_active(i == self.current_index)
                if i == self.current_index:
                    global_event_bus.watermarkAdd_PreviewFile.emit(thumb.image_path)
        self.prev_btn.setEnabled(len(self.total_images) > 1)
        self.next_btn.setEnabled(len(self.total_images) > 1)

    def prev_image(self):
        if len(self.total_images) <= 1:
            return
        self.current_index = (self.current_index - 1) % len(self.total_images)
        self.update_display()

    def next_image(self):
        if len(self.total_images) <= 1:
            return
        self.current_index = (self.current_index + 1) % len(self.total_images)
        self.update_display()

    def go_to_image(self, index):
        if 0 <= index < len(self.total_images):
            self.current_index = index
            self.update_display()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prev_image()
        elif event.key() == Qt.Key_Right:
            self.next_image()