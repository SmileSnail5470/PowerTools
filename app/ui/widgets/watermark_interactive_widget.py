import time
from PySide6.QtCore import Qt, QRectF, Signal, QObject, QEvent
from PySide6.QtGui import QPixmap, QColor, QPen, QBrush, QFont, QPainter, QImage
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QSlider,
    QLabel, QPushButton, QScrollArea, QFrame, QGraphicsView,
    QGraphicsScene, QGraphicsRectItem, QGraphicsPixmapItem, QGraphicsDropShadowEffect
)
import cv2
from app.ui.widgets.image_preview_widget import ScrollBar
from app.ui.library.qfluentwidgets import setFont
from app.ui.common.utils import get_file_type


COLOR_ACCENT = "#0071e3"
COLOR_DANGER = "#ff3b30"
COLOR_SUCCESS = "#34c759"
COLOR_BG = "#f5f5f7"
COLOR_TEXT_MAIN = "#333333"
COLOR_TEXT_DIM = "#888888"
PANEL_STYLE = "background-color: rgba(255, 255, 255, 230); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.3);"


class BoxSignals(QObject):
    deleted = Signal(object)


class SelectBoxItem(QGraphicsRectItem):
    def __init__(self, box_id, rect):
        super().__init__(rect)
        self.box_id = box_id
        pen = QPen(QColor(COLOR_SUCCESS))
        pen.setWidth(2)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(QColor(52, 199, 89, 38))) 
        self.setZValue(90)


class CoordItem(QFrame):
    def __init__(self, index, box_data, parent=None):
        super().__init__(parent)
        self.box_id = box_data['id']
        self.signals = BoxSignals()
        self.setObjectName("coordItem")
        self.setFixedHeight(48)
        self.setStyleSheet(f"""
            #coordItem {{ background-color: white; border-bottom: 1px solid #f0f0f0; }}
            #coordItem:hover {{ background-color: #f9f9fb; }}
            QLabel {{ color: {COLOR_TEXT_MAIN}; background: transparent; }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(15)
        idx_label = QLabel(f"{index}")
        setFont(idx_label, 12, QFont.Bold)
        idx_label.setFixedWidth(25)
        idx_label.setStyleSheet(f"color: {COLOR_TEXT_DIM};")
        vals_layout = QHBoxLayout()
        for key in ['x', 'y', 'w', 'h']:
            label = QLabel(f"{key.upper()}: <b style='color:{COLOR_ACCENT};'>{box_data[key]}</b>")
            setFont(label, 12)
            vals_layout.addWidget(label)
        vals_layout.addStretch()
        self.del_btn = QPushButton("×")
        self.del_btn.setFixedSize(26, 26)
        self.del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.del_btn.setStyleSheet(f"QPushButton {{ background: transparent; border: none; color: {COLOR_DANGER}; font-size: 20px; line-height: 26px; }}")
        self.del_btn.clicked.connect(lambda: self.signals.deleted.emit(self.box_id))
        layout.addWidget(idx_label)
        layout.addLayout(vals_layout)
        layout.addWidget(self.del_btn)


class AreaSelectorDialog(QDialog):
    def __init__(self, file_path=None, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(self.tr("水印框选器"))
        self.setMinimumSize(900, 800)
        self.setStyleSheet(f"background-color: {COLOR_BG};")
        self.setWindowFlags(
            Qt.WindowType.Dialog | 
            Qt.WindowType.WindowTitleHint | 
            Qt.WindowType.WindowSystemMenuHint | 
            Qt.WindowType.WindowCloseButtonHint
        )
        
        self.saved_boxes = []
        self.is_drawing = False
        self.active_rect_item = None

        self.cap = None
        self.total_frames = 0
        self.fps = 30
        self.original_pixmap = QPixmap()
        self.zoom_factor = 1.0

        self.setup_ui()
        if file_path:
            self.load_file(file_path)

    def load_file(self, path):
        if not path:
            return
        media_type = get_file_type(path)
        if media_type == "image":
            self.slider.hide()
            self.original_pixmap = QPixmap(path)
            if self.original_pixmap.isNull():
                return
            self.img_item.setPixmap(self.original_pixmap)
            self.scene.setSceneRect(0, 0, self.original_pixmap.width(), self.original_pixmap.height())
            self._fit_view()
        else:
            self.slider.show()
            self._load_video(path=path)

    def _load_video(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened(): 
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.slider.setMaximum(self.total_frames - 1)
        self._update_frame(0)

    def _update_frame(self, frame_idx):
        if not self.cap: 
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = self.cap.read()
        if success:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qimg = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            self.original_pixmap = QPixmap.fromImage(qimg)
            self.img_item.setPixmap(self.original_pixmap)
            self.scene.setSceneRect(0, 0, w, h)
            self._fit_view()

    def _fit_view(self):
        self.view.resetTransform()
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.zoom_factor = self.view.transform().m11()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        main_layout.setContentsMargins(0, 40, 0, 40)
        main_layout.setSpacing(20)

        # 画布区域
        canvas_container = QFrame()
        canvas_container.setObjectName("canvasContainer")
        canvas_container.setStyleSheet("#canvasContainer { background-color: white; border-radius: 20px; padding: 12px; }")
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(0, 0, 0, 25))
        shadow.setOffset(0, 20)
        canvas_container.setGraphicsEffect(shadow)

        container_layout = QVBoxLayout(canvas_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self._v_scroll = ScrollBar(Qt.Vertical, self.view)
        self._h_scroll = ScrollBar(Qt.Horizontal, self.view)
        self.view.setVerticalScrollBar(self._v_scroll)
        self.view.setHorizontalScrollBar(self._h_scroll)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.viewport().setCursor(Qt.CursorShape.CrossCursor)
        self.view.setFrameShape(QFrame.Shape.NoFrame)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.view.setStyleSheet("border-radius: 12px; background-color: #f0f0f0;")
        
        placeholder = QPixmap(800, 600)
        placeholder.fill(QColor("#cccccc"))
        self.img_item = QGraphicsPixmapItem(placeholder)
        self.scene.addItem(self.img_item)
        self.scene.setSceneRect(0, 0, 800, 600)
        container_layout.addWidget(self.view)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setFixedHeight(24)
        self.slider.setCursor(Qt.PointingHandCursor) 
        self.slider.setStyleSheet(f"""
            QSlider {{
                background: transparent;
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                background: #E2E8F0;
                border-radius: 3px;
            }}
            QSlider::sub-page:horizontal {{
                background: {COLOR_ACCENT};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: white;
                border: 2px solid {COLOR_ACCENT};
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #F8FAFC;
                border-width: 3px;
            }}
        """)
        self.slider.valueChanged.connect(self._update_frame)
        self.slider.hide()
        container_layout.addWidget(self.slider)

        main_layout.addWidget(canvas_container, 0, Qt.AlignmentFlag.AlignHCenter)

        # 信息面板
        panel_widget = QWidget()
        panel_widget.setFixedWidth(824)
        panel_layout = QVBoxLayout(panel_widget)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(12)

        info_card = QFrame()
        info_card.setFixedHeight(64)
        info_card.setStyleSheet(PANEL_STYLE)
        card_layout = QHBoxLayout(info_card)
        card_layout.setContentsMargins(24, 0, 24, 0)
        title_label = QLabel(self.tr("已选区域列表"))
        setFont(title_label, 18, QFont.Bold)
        title_label.setStyleSheet(f"color: {COLOR_TEXT_MAIN}; background: transparent; border: none;")
        self.count_badge = QLabel("0")
        self.count_badge.setFixedSize(28, 20)
        self.count_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        setFont(self.count_badge, 12, QFont.Bold)
        self.count_badge.setStyleSheet(f"background-color: {COLOR_ACCENT}; color: white; border-radius: 10px;")
        
        btn_clear = QPushButton(self.tr("全部清空"))
        btn_clear.setFixedSize(100, 36)
        setFont(btn_clear, 13, QFont.DemiBold)
        btn_clear.setStyleSheet("QPushButton { background-color: #8e8e93; color: white; border-radius: 8px; border: none; } QPushButton:hover { background-color: #636366; }")
        btn_clear.clicked.connect(self.clear_all)

        card_layout.addWidget(title_label)
        card_layout.addWidget(self.count_badge)
        card_layout.addStretch()
        card_layout.addWidget(btn_clear)
        panel_layout.addWidget(info_card)

        self.scrollArea = QScrollArea()
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setVerticalScrollBar(ScrollBar(Qt.Vertical, self.scrollArea))
        self.scrollArea.setHorizontalScrollBar(ScrollBar(Qt.Horizontal, self.scrollArea))
        self.scrollArea.horizontalScrollBar().setFade(True)
        self.scrollArea.verticalScrollBar().setFade(True)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setFixedHeight(120)
        self.scrollArea.setFrameShape(QFrame.Shape.NoFrame)
        self.scrollArea.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("scrollContent")
        self.scroll_content.setStyleSheet("#scrollContent { background-color: white; }")
        self.list_layout = QVBoxLayout(self.scroll_content)
        self.list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(0)
        self.scrollArea.setWidget(self.scroll_content)
        panel_layout.addWidget(self.scrollArea)
        main_layout.addWidget(panel_widget, 0, Qt.AlignmentFlag.AlignHCenter)

        self.view.viewport().installEventFilter(self)
        self.render_all()

    def eventFilter(self, source, event):
        if source is self.view.viewport():
            if event.type() == QEvent.Type.Wheel:
                delta = event.angleDelta().y()
                factor = 1.25 if delta > 0 else 0.8
                self.zoom_factor *= factor
                self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
                self.view.scale(factor, factor)
                return True
            elif event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self.is_drawing = True
                self.start_pos = self.view.mapToScene(event.position().toPoint())
                self.active_rect_item = QGraphicsRectItem()
                self.active_rect_item.setPen(QPen(QColor(COLOR_ACCENT), 2))
                self.active_rect_item.setBrush(QBrush(QColor(0, 113, 227, 51)))
                self.scene.addItem(self.active_rect_item)
                return True
            elif event.type() == QEvent.Type.MouseMove and self.is_drawing:
                curr_pos = self.view.mapToScene(event.position().toPoint())
                rect = QRectF(self.start_pos, curr_pos).normalized()
                self.active_rect_item.setRect(rect.intersected(QRectF(self.img_item.pixmap().rect())))
                return True
            elif event.type() == QEvent.Type.MouseButtonRelease and self.is_drawing:
                self.is_drawing = False
                rect = self.active_rect_item.rect()
                self.scene.removeItem(self.active_rect_item)
                if rect.width() > 5:
                    real_box = {
                        "id": int(time.time()*1000), 
                        "x": int(rect.x()), 
                        "y": int(rect.y()), 
                        "w": int(rect.width()), 
                        "h": int(rect.height())
                    }
                    self.saved_boxes.append(real_box)
                    self.render_all()
                return True
        return super().eventFilter(source, event)

    def delete_box(self, box_id):
        self.saved_boxes = [b for b in self.saved_boxes if b['id'] != box_id]
        self.render_all()

    def clear_all(self):
        if self.saved_boxes: self.saved_boxes = []
        self.render_all()

    def get_results(self):
        """ 供外部调用的接口 """
        return self.saved_boxes

    def render_all(self):
        for item in self.scene.items():
            if isinstance(item, SelectBoxItem): self.scene.removeItem(item)
        while self.list_layout.count():
            child = self.list_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        if not self.saved_boxes:
            empty_label = QLabel(self.tr("暂无记录，请在上方图区拖拽选择区域"))
            setFont(empty_label, 11)
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_label.setFixedHeight(40)
            empty_label.setStyleSheet(f"color: {COLOR_TEXT_DIM}; background-color: white; border-radius: 12px;")
            self.list_layout.addWidget(empty_label)
        else:
            for i, box in enumerate(self.saved_boxes):
                ui_rect = QRectF(box['x'], box['y'], box['w'], box['h'])
                self.scene.addItem(SelectBoxItem(box['id'], ui_rect))
                item_widget = CoordItem(i + 1, box); item_widget.signals.deleted.connect(self.delete_box)
                self.list_layout.addWidget(item_widget)
        self.count_badge.setText(str(len(self.saved_boxes)))
        self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().maximum())