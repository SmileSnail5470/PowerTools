from PySide6.QtCore import Signal, Qt, QTimer, QRect, Property, QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import QGraphicsView, QWidget , QVBoxLayout, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem, QScrollBar
from PySide6.QtGui import QPixmap, QWheelEvent, QColor, QPainter, QBrush
from app.ui.library.qfluentwidgets import setFont, qconfig, Theme 


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

    def __init__(self, pixmap: QPixmap = None, parent=None):
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
        setFont(title, 20)
        title.setDefaultTextColor(QColor("#666666"))
        self.scene.addItem(title)

        # 副标题
        subtitle = QGraphicsTextItem(self.tr("添加文件后显示预览"))
        setFont(subtitle, 18)
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
        self.scene.setSceneRect(-200, -150, 400, 300)

    def _center_placeholder(self):
        """将占位文字居中"""
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
        """设置/更新图片"""
        self.scene.clear()
        if pixmap and not pixmap.isNull():
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.pixmap_item)
            self.scene.setSceneRect(self.pixmap_item.boundingRect())
        else:
            self._init_placeholder()

    def wheelEvent(self, event: QWheelEvent):
        """同步缩放"""
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        old_zoom = self._zoom
        self._zoom *= zoom_factor
        self._zoom = max(0.1, min(self._zoom, 10.0))

        scale_factor = self._zoom / old_zoom
        self.scale(scale_factor, scale_factor)

        self.zoomChanged.emit(self._zoom)

    def _emit_scroll(self):
        """滚动同步信号"""
        if not self._syncing_scroll:
            self.scrollChanged.emit(
                self.horizontalScrollBar().value(),
                self.verticalScrollBar().value()
            )

    def sync_scroll(self, x: int, y: int):
        """响应同步滚动"""
        self._syncing_scroll = True
        self.horizontalScrollBar().setValue(x)
        self.verticalScrollBar().setValue(y)
        self._syncing_scroll = False

    def sync_zoom(self, target_zoom: float):
        """响应同步缩放"""
        if abs(target_zoom - self._zoom) > 1e-3:
            scale_factor = target_zoom / self._zoom
            self._zoom = target_zoom
            self.scale(scale_factor, scale_factor)


class SyncImageViewer(QWidget):
    """双图同步查看器"""
    def __init__(self, img1: str = "", img2: str = ""):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        pix1 = QPixmap(img1) if img1 else None
        pix2 = QPixmap(img2) if img2 else None

        self.view1 = SyncGraphicsView(pix1)
        self.view2 = SyncGraphicsView(pix2)

        layout.addWidget(self.view1)
        layout.addWidget(self.view2)

        # 信号互联（双向同步）
        self.view1.zoomChanged.connect(self.view2.sync_zoom)
        self.view2.zoomChanged.connect(self.view1.sync_zoom)
        self.view1.scrollChanged.connect(self.view2.sync_scroll)
        self.view2.scrollChanged.connect(self.view1.sync_scroll)

        # Fluent/Win11风格
        self.setStyleSheet("""
            QWidget {
                background-color: #f3f3f3;
                border-radius: 8px;
            }
            QGraphicsView {
                background-color: #fafafa;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
            }
        """)

    def set_images(self, img1: str, img2: str):
        """动态设置图片"""
        self.view1.set_pixmap(QPixmap(img1))
        self.view2.set_pixmap(QPixmap(img2))