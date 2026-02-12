import sys
import numpy as np
import math
from PySide6.QtWidgets import (
    QApplication, QFrame, QWidget, QHBoxLayout,  QVBoxLayout, QLabel,  QGraphicsView, QGraphicsScene, 
    QGraphicsItem, QTextEdit, QSizePolicy, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, QRectF, Signal, QPointF, QTimer, QSize
from PySide6.QtGui import QPixmap, QColor, QPainter, QPainterPath, QPen, QFont, QPolygonF

from app.ui.library.qfluentwidgets import ScrollArea, setFont, getFont, FluentIcon, ToolButton


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
        setFont(painter, 8, QFont.Bold)
        painter.drawText(tag_rect, Qt.AlignCenter, "OCR result")
        painter.restore()


class OCRListItem(QWidget):
    hovered = Signal(QPolygonF, bool)
    unhovered = Signal()

    def __init__(self, text, conf, points: QPolygonF):
        super().__init__()
        self.points = points
        self.is_danger = conf < 0.8
        self.setAttribute(Qt.WA_StyledBackground, True)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 12, 15, 12)
        txt = QTextEdit(text)
        txt.setReadOnly(True)
        txt.setFrameShape(QFrame.NoFrame)
        txt.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        txt.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        txt.document().setDocumentMargin(0)
        setFont(txt, 13, QFont.Medium)
        txt.setStyleSheet("color: #323130; background: transparent; border: none;")
        txt.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        txt.document().adjustSize()

        doc_height = txt.document().size().height()
        margins = txt.contentsMargins()
        txt.setFixedHeight(int(doc_height + margins.top() + margins.bottom()))
        
        sub = QLabel(f"置信度: {conf:.2f}")
        setFont(sub, 11)
        sub.setStyleSheet(f"color: {'#d13438' if self.is_danger else '#605e5c'}; background: transparent;")
        
        main_layout.addWidget(txt)
        main_layout.addWidget(sub)
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


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._zoom_level = 0

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        factor = 1.15
        
        if angle > 0:
            if self._zoom_level < 10:
                self.scale(factor, factor)
                self._zoom_level += 1
        else:
            if self._zoom_level > -5:
                self.scale(1.0 / factor, 1.0 / factor)
                self._zoom_level -= 1

class OCRViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.raw_results = []
        self.setup_ui()
        self.show_placeholders()

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(1)

        # 左侧：图形预览
        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setStyleSheet("""
            ZoomableGraphicsView { 
                background-color: #f8f8f8; 
                border: none;
                border-right: 1px solid #e1dfdd;
            }
        """)
        self.view.setRenderHint(QPainter.Antialiasing)

        self.highlighter = OCRAngledHighlightItem()
        self.scene.addItem(self.highlighter)
        main_layout.addWidget(self.view, 3)

        # 右侧：列表展示
        right_panel = QWidget()
        right_panel.setMinimumWidth(220)
        right_panel.setStyleSheet("background-color: #ffffff; border-left: 1px solid #edebe9;")
        right_vbox = QVBoxLayout(right_panel)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(0)

        self.header = QWidget()
        self.header.setFixedHeight(48)
        self.header.setStyleSheet("""
            QWidget { 
                background-color: #ffffff; 
                border-bottom: 1px solid #f3f2f1;
                border: none; 
            }
        """)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(16, 0, 8, 0)

        self.title_label = QLabel(self.tr("OCR 识别结果"))
        self.title_label.setStyleSheet("color: #323130; border: none;")
        setFont(self.title_label, 13, QFont.Bold)
        toolbar = QWidget()
        toolbar.setStyleSheet("border: none; background: transparent;")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(4)
        self.btn_copy = ToolButton(FluentIcon.COPY.qicon())
        self.btn_copy.setIconSize(QSize(16, 16))
        self.btn_copy.setStyleSheet("border-radius: 16px;")
        self.btn_copy.setToolTip("复制所有文字")
        self.btn_download = ToolButton(FluentIcon.DOWNLOAD.qicon())
        self.btn_download.setIconSize(QSize(16, 16))
        self.btn_download.setStyleSheet("border-radius: 16px;")
        self.btn_download.setToolTip("结果导出")
        self.btn_copy.clicked.connect(self.copy_all_text)
        self.btn_download.clicked.connect(self.download_result)
        toolbar_layout.addWidget(self.btn_copy)
        toolbar_layout.addWidget(self.btn_download)

        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(toolbar)

        scroll = ScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background-color: #ffffff; border: none;")
        self.list_container = QWidget()
        self.list_container.setStyleSheet("background-color: #ffffff;")
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(0)
        self.list_layout.addStretch()
        scroll.setWidget(self.list_container)
        right_vbox.addWidget(self.header)
        right_vbox.addWidget(scroll)
        main_layout.addWidget(right_panel)
        self.header.hide()

    def show_placeholders(self):
        virtual_rect = QRectF(-150, -150, 300, 300)
        self.scene.setSceneRect(virtual_rect)
        text_item = self.scene.addText(self.tr("原图显示区"), font=getFont(15))
        text_item.setDefaultTextColor(QColor("#888888"))
        rect = text_item.boundingRect()
        text_item.setPos(-rect.width() / 2, -rect.height() / 2) 
        self.view.centerOn(0, 0)

        self.placeholder_label = QLabel(self.tr("OCR 结果显示区"))
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        setFont(self.placeholder_label, 15)
        self.placeholder_label.setStyleSheet("color: #888888;")
        self.list_layout.insertWidget(0, self.placeholder_label, alignment=Qt.AlignCenter)
        self.list_layout.insertStretch(0, 1)
        self.list_layout.addStretch(1)

    def init_scene(self, show_placeholders=True):
        while self.list_layout.count() > 0:
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.scene.clear()
        self.view.resetTransform()
        self.view._zoom_level = 0
        self.scene.setSceneRect(QRectF())
        self.header.hide()
        if show_placeholders:
            self.show_placeholders()

    def set_data(self, image_path, raw_data):
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return
        self.raw_results = raw_data[1]
        self.init_scene(show_placeholders=False)
        self.header.show()
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
        
        self.list_layout.addStretch(1)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def download_result(self):
        if not self.raw_results: 
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "导出识别结果", "OCR_Result.txt", "Text Files (*.txt)")
        if not file_path:
            return
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for text, _ in self.raw_results:
                    f.write(f"{text}\n")
            self.show_toast("导出成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")

    def copy_all_text(self):
        if not self.raw_results: 
            return
        all_text = "\n".join([item[0] for item in self.raw_results])
        QApplication.clipboard().setText(all_text)
        self.show_toast("已复制到剪贴板")

    def show_toast(self, message):
        toast = QLabel(message, self.list_container)
        toast.setStyleSheet("""
            background: #333333; color: white; padding: 8px 16px; 
            border-radius: 4px; font-size: 12px;
        """)
        toast.adjustSize()
        # 居中显示在视图容器底部
        x = (self.list_container.width() - toast.width()) // 2
        y = self.list_container.height() - 60
        toast.move(x, y)
        toast.show()
        QTimer.singleShot(1500, toast.deleteLater)

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