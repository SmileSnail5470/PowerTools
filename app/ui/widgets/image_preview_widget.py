from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGraphicsView, QWidget , QVBoxLayout, QGraphicsScene, QGraphicsPixmapItem
from PySide6.QtGui import QPixmap, QWheelEvent


class SyncGraphicsView(QGraphicsView):
    zoomChanged = Signal(float)
    scrollChanged = Signal(int, int)

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.setRenderHints(self.renderHints() | self.renderHints().SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)

        self._zoom = 1.0
        self._syncing_scroll = False

        # 连接滚动条信号
        self.horizontalScrollBar().valueChanged.connect(self._emit_scroll)
        self.verticalScrollBar().valueChanged.connect(self._emit_scroll)

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
    def __init__(self, img1: str, img2: str):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        pix1 = QPixmap(img1)
        pix2 = QPixmap(img2)

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
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
            }
        """)
