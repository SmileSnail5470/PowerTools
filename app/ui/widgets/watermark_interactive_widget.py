import time
from PySide6.QtCore import Qt, QRectF, Signal, QObject, QEvent, QSize
from PySide6.QtGui import QPixmap, QColor, QPen, QBrush, QFont, QPainter, QImage, QIcon
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton, QScrollArea, QFrame, QGraphicsView,
    QGraphicsScene, QGraphicsRectItem, QGraphicsPixmapItem, QGraphicsDropShadowEffect, QButtonGroup
)
import cv2
from app.ui.widgets.image_preview_widget import ScrollBar
from app.ui.library.qfluentwidgets import setFont
from app.ui.common.utils import get_file_type
from app.utils.image_io import open_video_capture


COLOR_ACCENT = "#4f46e5"
COLOR_POSITIVE = "#10b981"
COLOR_NEGATIVE = "#ef4444"
COLOR_BG = "#f5f5f7"


class BoxSignals(QObject):
    deleted = Signal(object)


class SelectBoxItem(QGraphicsRectItem):
    def __init__(self, box_id, rect, box_type="positive"):
        super().__init__(rect)
        self.box_id = box_id
        color = COLOR_NEGATIVE if box_type == "negative" else COLOR_POSITIVE
        pen = QPen(QColor(color), 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(42)
        self.setBrush(QBrush(fill))
        self.setZValue(90)


class CoordItem(QFrame):
    def __init__(self, index, box_data, parent=None):
        super().__init__(parent)
        self.box_id = box_data['id']
        self.signals = BoxSignals()
        self.setObjectName("coordItem")
        self.setFixedHeight(52)
        self.setStyleSheet("""
            #coordItem { background: #ffffff; border-bottom: 1px solid #eef2f7; }
            #coordItem:hover { background: #f8fafc; }
            QLabel { color: #334155; background: transparent; }
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(12)
        idx_label = QLabel(f"{index}")
        setFont(idx_label, 12, QFont.Bold)
        idx_label.setFixedWidth(25)
        idx_label.setStyleSheet("color: #94a3b8;")
        box_type = box_data.get("type", "positive")
        type_label = QLabel(self.tr("保留" if box_type == "negative" else "去除"))
        type_label.setFixedSize(44, 24)
        type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        setFont(type_label, 10, QFont.DemiBold)
        if box_type == "negative":
            type_label.setStyleSheet("color: #b91c1c; background: #fef2f2; border-radius: 8px;")
        else:
            type_label.setStyleSheet("color: #047857; background: #ecfdf5; border-radius: 8px;")
        vals_layout = QHBoxLayout()
        for key in ['x', 'y', 'w', 'h']:
            label = QLabel(f"{key.upper()} <b style='color:#4f46e5;'>{box_data[key]}</b>")
            setFont(label, 11)
            vals_layout.addWidget(label)
        vals_layout.addStretch()
        self.del_btn = QPushButton(self.tr("删除"))
        self.del_btn.setFixedSize(52, 30)
        self.del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.del_btn.setStyleSheet("""
            QPushButton { color: #dc2626; background: #fef2f2; border: none; border-radius: 8px; }
            QPushButton:hover { background: #fee2e2; }
        """)
        self.del_btn.clicked.connect(lambda: self.signals.deleted.emit(self.box_id))
        layout.addWidget(idx_label)
        layout.addWidget(type_label)
        layout.addLayout(vals_layout)
        layout.addWidget(self.del_btn)


class AreaSelectorDialog(QDialog):
    def __init__(self, file_path=None, single_area_only=False, parent=None, allow_negative_boxes=False):
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
        self.single_area_only = single_area_only
        self.allow_negative_boxes = allow_negative_boxes
        self.current_box_type = "positive"

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

    def _release_capture(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _load_video(self, path):
        self._release_capture()
        self.cap = open_video_capture(path)
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
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(0)
        workspace_layout = QHBoxLayout()
        workspace_layout.setSpacing(16)
        main_layout.addLayout(workspace_layout, 1)

        if self.allow_negative_boxes:
            self.setMinimumSize(1180, 800)
            control_panel = QFrame()
            control_panel.setObjectName("boxTypePanel")
            control_panel.setFixedWidth(232)
            control_panel.setStyleSheet("""
                QFrame#boxTypePanel {
                    background: #ffffff;
                    border: 1px solid #e2e8f0;
                    border-radius: 18px;
                }
                QLabel { background: transparent; border: none; }
            """)
            control_layout = QVBoxLayout(control_panel)
            control_layout.setContentsMargins(18, 20, 18, 18)
            control_layout.setSpacing(12)

            title_panel = QFrame()
            title_panel.setObjectName("boxTypeTitlePanel")
            title_panel.setStyleSheet("""
                QFrame#boxTypeTitlePanel {
                    background: #f1f5f9;
                    border: 1px solid #e2e8f0;
                    border-radius: 12px;
                }
            """)
            title_layout = QVBoxLayout(title_panel)
            title_layout.setContentsMargins(12, 10, 12, 10)
            title_layout.setSpacing(4)
            mode_title = QLabel(self.tr("框选控制"))
            setFont(mode_title, 15, QFont.DemiBold)
            mode_title.setStyleSheet("color: #1e293b;")
            mode_hint = QLabel(self.tr("选择框选类型，再在画面中拖拽绘制区域。"))
            mode_hint.setWordWrap(True)
            setFont(mode_hint, 10)
            mode_hint.setStyleSheet("color: #64748b;")
            title_layout.addWidget(mode_title)
            title_layout.addWidget(mode_hint)
            control_layout.addWidget(title_panel)

            divider = QFrame()
            divider.setFrameShape(QFrame.Shape.HLine)
            divider.setStyleSheet("color: #e2e8f0;")
            control_layout.addWidget(divider)

            self.box_type_group = QButtonGroup(self)
            self.box_type_group.setExclusive(True)
            self.positive_button = QPushButton(self.tr("正框 (去除目标)"))
            self.negative_button = QPushButton(self.tr("负框 (保留区域)"))
            self.positive_button.setIcon(self._square_plus_minus_icon(True))
            self.negative_button.setIcon(self._square_plus_minus_icon(False))
            for button, background, border, color, checked_background, checked_border in (
                (self.positive_button, "rgba(16, 185, 129, 0.10)", "#a7f3d0", "#047857", "rgba(16, 185, 129, 0.28)", "#6ee7b7"),
                (self.negative_button, "rgba(239, 68, 68, 0.09)", "#fecaca", "#b91c1c", "rgba(239, 68, 68, 0.26)", "#fca5a5"),
            ):
                button.setCheckable(True)
                button.setIconSize(QSize(18, 18))
                button.setFixedHeight(42)
                button.setCursor(Qt.PointingHandCursor)
                button.setStyleSheet(f"""
                    QPushButton {{
                        color: {color};
                        background: {background};
                        border: 1px solid {border};
                        border-radius: 11px;
                        padding: 0 12px;
                        text-align: left;
                        font-weight: 600;
                    }}
                    QPushButton:checked {{
                        color: {color};
                        background: {checked_background};
                        border-color: {checked_border};
                    }}
                    QPushButton:hover:!checked {{
                        background: {checked_background};
                        border-color: {checked_border};
                    }}
                """)
                self.box_type_group.addButton(button)
                control_layout.addWidget(button)
            self.positive_button.setChecked(True)
            self.positive_button.clicked.connect(lambda: self._set_box_type("positive"))
            self.negative_button.clicked.connect(lambda: self._set_box_type("negative"))
            control_layout.addStretch()
            workspace_layout.addWidget(control_panel, 0, Qt.AlignmentFlag.AlignTop)
        else:
            self.setMinimumSize(900, 800)

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
        container_layout.setContentsMargins(14, 14, 14, 14)
        container_layout.setSpacing(12)

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
        self.view.setStyleSheet("border-radius: 12px; background-color: #f1f5f9;")
        
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

        content_panel = QWidget()
        content_layout = QVBoxLayout(content_panel)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)
        content_layout.addWidget(canvas_container, 1)

        # 信息面板
        panel_widget = QWidget()
        panel_widget.setFixedWidth(824)
        panel_layout = QVBoxLayout(panel_widget)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(12)

        info_card = QFrame()
        info_card.setFixedHeight(60)
        info_card.setStyleSheet("""
            QFrame { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; }
            QLabel { background: transparent; border: none; color: #1e293b; }
        """)
        card_layout = QHBoxLayout(info_card)
        card_layout.setContentsMargins(18, 0, 18, 0)
        card_layout.setSpacing(10)
        title_label = QLabel(self.tr("已选区域列表"))
        setFont(title_label, 14, QFont.DemiBold)
        self.count_badge = QLabel("0")
        self.count_badge.setFixedSize(28, 22)
        self.count_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        setFont(self.count_badge, 11, QFont.Bold)
        self.count_badge.setStyleSheet("background: #eef2ff; color: #4f46e5; border-radius: 11px;")
        
        btn_clear = QPushButton(self.tr("全部清空"))
        btn_clear.setFixedSize(88, 32)
        setFont(btn_clear, 11, QFont.DemiBold)
        btn_clear.setCursor(Qt.PointingHandCursor)
        btn_clear.setStyleSheet("""
            QPushButton { color: #475569; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 9px; }
            QPushButton:hover { background: #e2e8f0; }
            QPushButton:pressed { background: #cbd5e1; }
        """)
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
        self.scrollArea.setFixedHeight(156)
        self.scrollArea.setFrameShape(QFrame.Shape.NoFrame)
        self.scrollArea.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("scrollContent")
        self.scroll_content.setStyleSheet("#scrollContent { background-color: #ffffff; }")
        self.list_layout = QVBoxLayout(self.scroll_content)
        self.list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(0)
        self.scrollArea.setWidget(self.scroll_content)
        panel_layout.addWidget(self.scrollArea)
        content_layout.addWidget(panel_widget)
        workspace_layout.addWidget(content_panel, 1)

        self.view.viewport().installEventFilter(self)
        self.render_all()

    @staticmethod
    def _square_plus_minus_icon(positive):
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
                color = COLOR_NEGATIVE if self.current_box_type == "negative" else COLOR_POSITIVE
                pen = QPen(QColor(color), 2)
                pen.setCosmetic(True)
                self.active_rect_item.setPen(pen)
                fill = QColor(color)
                fill.setAlpha(48)
                self.active_rect_item.setBrush(QBrush(fill))
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
                        "w": self._even_floor(rect.width()), 
                        "h": self._even_floor(rect.height()),
                        "type": self.current_box_type
                    }
                    if self.single_area_only:
                        self.saved_boxes = [real_box]
                    else:
                        self.saved_boxes.append(real_box)
                    self.render_all()
                return True
        return super().eventFilter(source, event)

    def _set_box_type(self, box_type):
        if self.allow_negative_boxes and box_type in ("positive", "negative"):
            self.current_box_type = box_type
    
    def _even_floor(self, v):
        # 将数值向下取整为偶数，确保宽高为偶数
        return int(v) & ~1

    def delete_box(self, box_id):
        self.saved_boxes = [b for b in self.saved_boxes if b['id'] != box_id]
        self.render_all()

    def clear_all(self):
        if self.saved_boxes: self.saved_boxes = []
        self.render_all()

    def accept(self):
        self._release_capture()
        super().accept()

    def reject(self):
        self._release_capture()
        super().reject()

    def closeEvent(self, event):
        self._release_capture()
        super().closeEvent(event)

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
            empty_label.setStyleSheet("color: #94a3b8; background: #ffffff;")
            self.list_layout.addWidget(empty_label)
        else:
            for i, box in enumerate(self.saved_boxes):
                ui_rect = QRectF(box['x'], box['y'], box['w'], box['h'])
                self.scene.addItem(SelectBoxItem(box['id'], ui_rect, box.get("type", "positive")))
                item_widget = CoordItem(i + 1, box, self)
                item_widget.signals.deleted.connect(self.delete_box)
                self.list_layout.addWidget(item_widget)
        self.count_badge.setText(str(len(self.saved_boxes)))
        self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().maximum())