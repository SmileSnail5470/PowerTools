import sys
import numpy as np
import math
from PySide6.QtWidgets import QApplication, QFrame, QWidget, QHBoxLayout,  QVBoxLayout, QLabel,  QGraphicsView, QGraphicsScene,  QGraphicsItem
from PySide6.QtCore import Qt, QRectF, Signal, QPointF
from PySide6.QtGui import QPixmap, QColor, QPainter, QPainterPath, QPen, QFont, QPolygonF

from app.ui.library.qfluentwidgets import ScrollArea, setFont


class OCRAngledHighlightItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.polygon = QPolygonF()
        self.is_danger = False
        self._rect = QRectF(0, 0, 0, 0)
        self.setZValue(10)

    def set_full_rect(self, rect):
        self.prepareGeometryChange()
        self._rect = QRectF(rect)
        self.update()

    def update_highlight(self, poly, is_danger):
        self.prepareGeometryChange()
        self.polygon = poly
        self.is_danger = is_danger
        self.update()

    def boundingRect(self):
        return self._rect

    def paint(self, painter, option, widget):
        if self.polygon.isEmpty() or self._rect.isEmpty():
            return

        painter.save()
        full_path = QPainterPath()
        full_path.addRect(self._rect)
        
        poly_path = QPainterPath()
        poly_path.addPolygon(self.polygon)
        
        # 镂空效果
        mask_path = full_path.subtracted(poly_path)
        painter.fillPath(mask_path, QColor(0, 0, 0, 120)) 
        painter.restore()

        # 2. 绘制倾斜边框
        color = QColor("#d13438") if self.is_danger else QColor("#0078d4")
        painter.setPen(QPen(color, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPolygon(self.polygon)

        # 3. 绘制标签
        p1 = self.polygon[0]
        p2 = self.polygon[1]
        angle = math.degrees(math.atan2(p2.y() - p1.y(), p2.x() - p1.x()))
        
        painter.save()
        painter.translate(p1)
        painter.rotate(angle)
        
        tag_rect = QRectF(0, -22, 80, 20)
        painter.fillRect(tag_rect, color)
        painter.setPen(Qt.white)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(tag_rect, Qt.AlignCenter, "OCR RESULT")
        painter.restore()


class OCRListItem(QWidget):
    hovered = Signal(QPolygonF, bool)
    unhovered = Signal()

    def __init__(self, text, conf, points: QPolygonF):
        super().__init__()
        self.points = points
        self.is_danger = conf < 0.8
        self.setAttribute(Qt.WA_StyledBackground, True)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        txt = QLabel(text)
        txt.setWordWrap(True)
        txt.setStyleSheet("font-size: 13px; font-weight: 500; color: #323130; background: transparent;")
        
        sub = QLabel(f"置信度: {conf:.2f}")
        sub.setStyleSheet(f"font-size: 11px; color: {'#d13438' if self.is_danger else '#605e5c'}; background: transparent;")
        
        layout.addWidget(txt)
        layout.addWidget(sub)
        self.update_style(False)

    def update_style(self, is_hover):
        bg = "#f3f9ff" if is_hover else ("#fff4f4" if self.is_danger else "#ffffff")
        border = "#0078d4" if is_hover else "transparent"
        self.setStyleSheet(f"""
            OCRListItem {{ 
                background-color: {bg}; 
                border-bottom: 1px solid #edebe9; 
                border-left: 4px solid {border}; 
            }}
        """)

    def enterEvent(self, event):
        self.update_style(True)
        self.hovered.emit(self.points, self.is_danger)

    def leaveEvent(self, event):
        self.update_style(False)
        self.unhovered.emit()


class OCRViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.show_placeholders()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        # 左侧：图形预览
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setStyleSheet("""
            QGraphicsView { 
                background-color: #f8f8f8; 
                border: none;
                border-right: 1px solid #e1dfdd;
            }
        """)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag) # 允许拖拽查看
        self.view.setRenderHint(QPainter.Antialiasing)

        self.highlighter = OCRAngledHighlightItem()
        self.scene.addItem(self.highlighter)
        layout.addWidget(self.view, 3)

        # 右侧：列表展示
        scroll = ScrollArea()
        scroll.setMinimumWidth(220)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background-color: #ffffff;")
        list_container = QWidget()
        list_container.setStyleSheet("background-color: #ffffff;")
        self.list_layout = QVBoxLayout(list_container)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(0)
        self.list_layout.addStretch()
        scroll.setWidget(list_container)
        layout.addWidget(scroll)

    def show_placeholders(self):
        text_item = self.scene.addText("原图显示区", QFont("Microsoft YaHei", 12))
        text_item.setDefaultTextColor(QColor("#888888"))
        text_item.setPos(-40, -10) 

        self.placeholder_label = QLabel("等待 OCR 识别结果...")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("color: #888888; font-size: 13px; margin-top: 300px;")
        self.list_layout.insertWidget(0, self.placeholder_label)
        self.list_layout.addStretch()

    def init_scene(self, show_placeholders=True):
        while self.list_layout.count() > 0:
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.scene.clear()
        if show_placeholders:
            self.show_placeholders()

    def set_data(self, image_path, raw_data):
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return
        self.init_scene(show_placeholders=False)
        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(pixmap.rect())
        
        self.highlighter = OCRAngledHighlightItem()
        self.highlighter.set_full_rect(pixmap.rect())
        self.scene.addItem(self.highlighter)

        coords_list, results = raw_data
        for coords, (text, conf) in zip(coords_list, results):
            poly = QPolygonF([QPointF(p[0], p[1]) for p in coords])
            list_item = OCRListItem(text, conf, poly)
            list_item.hovered.connect(self.on_hover)
            list_item.unhovered.connect(self.on_unhover)
            self.list_layout.addWidget(list_item)
        
        self.list_layout.addStretch()
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def on_hover(self, poly, is_danger):
        self.highlighter.update_highlight(poly, is_danger)
        # 自动滚动到多边形中心
        self.view.ensureVisible(poly.boundingRect(), 150, 150)

    def on_unhover(self):
        self.highlighter.update_highlight(QPolygonF(), False)

if __name__ == "__main__":
    import os
    raw_coords = [
        np.array([[547., 86.], [643., 18.], [656., 36.], [560., 105.]], dtype=np.float32),
        np.array([[959., 291.], [1178., 121.], [1208., 159.], [989., 329.]], dtype=np.float32),
        np.array([[364., 543.], [479., 454.], [500., 481.], [384., 571.]], dtype=np.float32),
        np.array([[0., 573.], [173., 402.], [209., 439.], [35., 610.]], dtype=np.float32),
        np.array([[1293., 590.], [1475., 435.], [1499., 472.], [1325., 627.]], dtype=np.float32),
        np.array([[854., 689.], [976., 605.], [992., 628.], [871., 713.]], dtype=np.float32),
        np.array([[290., 876.], [516., 699.], [549., 741.], [323., 919.]], dtype=np.float32),
        np.array([[959., 881.], [1185., 699.], [1219., 741.], [993., 923.]], dtype=np.float32)
    ]
    raw_texts = [
        ('Hes Stuidio', 0.6851480603218079), ('shutterstock', 0.9308447241783142), 
        ('Milles Studio', 0.9547773003578186), ('lutterstock', 0.8868420124053955), 
        ('shutterst', 0.9905760884284973), ('Milles Studio', 0.9662759304046631), 
        ('shutterstock', 0.947007954120636), ('shutterstack', 0.9224569797515869)
    ]

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    main_win = QWidget()
    main_layout = QVBoxLayout(main_win)
    
    ocr_widget = OCRViewerWidget()
    
    image_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assess", "test1.jpg")
    ocr_widget.set_data(image_file, (raw_coords, raw_texts))
    
    main_layout.addWidget(ocr_widget)
    main_win.show()
    
    sys.exit(app.exec())