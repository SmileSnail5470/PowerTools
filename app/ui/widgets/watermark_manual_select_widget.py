import os
from datetime import datetime
from PySide6.QtWidgets import (
    QGraphicsScene, QVBoxLayout, QLabel, QFrame, QHBoxLayout, QGraphicsItem, QGraphicsView,
    QGraphicsPixmapItem
)
from PySide6.QtCore import Qt, QSize, QPointF, QRectF
from PySide6.QtGui import QPainter, QMouseEvent, QColor, QFont, QImage, QPen, QPixmap

from app.ui.library.qfluentwidgets import(
    Action, MaskDialogBase, TeachingTip, InfoBarIcon, TeachingTipTailPosition, SubtitleLabel, CommandBar, 
    FluentIcon, FluentStyleSheet, setFont, Slider
)
from app.ui.library.qframelesswindow.titlebar import CloseButton

from app.ui.widgets.gradient_header_widget import GradientHeader
from app.ui.widgets.image_preview_widget import ScrollBar

from app.ui.common.event_bus import global_event_bus
from app.ui.common.config import cfg


class MyMessageBoxBase(MaskDialogBase):
    """ Message box base """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.vBoxLayout = QVBoxLayout(self.widget)
        self.viewLayout = QVBoxLayout()

        self.__initWidget()

    def __initWidget(self):
        self.__setQss()
        self.__initLayout()

        self.setShadowEffect(60, (0, 10), QColor(0, 0, 0, 50))
        self.setMaskColor(QColor(0, 0, 0, 76))

    def __initLayout(self):
        self._hBoxLayout.removeWidget(self.widget)
        self._hBoxLayout.addWidget(self.widget, 1, Qt.AlignCenter)

        self.vBoxLayout.setSpacing(0)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.addLayout(self.viewLayout, 1)

        self.viewLayout.setSpacing(0)
        self.viewLayout.setContentsMargins(0, 0, 0, 0)

    def __setQss(self):
        FluentStyleSheet.DIALOG.apply(self)


class MaskItem(QGraphicsItem):
    def __init__(self, size: QSize):
        super().__init__()

        self.image = QImage(size, QImage.Format_Alpha8)
        self.image.fill(0)

        self.brush_size = 20
        self.current_tool = "brush"  # brush / eraser
        self.last_point: QPointF | None = None

        self.history: list[QImage] = []
        self.history_index = -1
        self.max_history = 50
        self.save_to_history()

        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.image.width(), self.image.height())

    def paint(self, painter: QPainter, option, widget=None):
        painter.save()
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.setOpacity(0.35)

        colored = QImage(self.image.size(), QImage.Format_ARGB32)
        colored.fill(Qt.transparent)

        cp = QPainter(colored)
        cp.setCompositionMode(QPainter.CompositionMode_Source)
        cp.fillRect(colored.rect(), QColor(255, 0, 0, 255))  # 红色
        cp.setCompositionMode(QPainter.CompositionMode_DestinationIn)
        cp.drawImage(0, 0, self.image)
        cp.end()

        painter.drawImage(0, 0, colored)
        painter.restore()

    def draw_line(self, p1: QPointF, p2: QPointF):
        painter = QPainter(self.image)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen()
        pen.setWidth(self.brush_size)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)

        if self.current_tool == "brush":
            pen.setColor(QColor(255, 255, 255))
            painter.setCompositionMode(QPainter.CompositionMode_Source)
        else:
            pen.setColor(QColor(0, 0, 0))
            painter.setCompositionMode(QPainter.CompositionMode_Source)

        painter.setPen(pen)
        painter.drawLine(p1, p2)
        painter.end()

        r = QRectF(p1, p2).normalized().adjusted(
            -self.brush_size,
            -self.brush_size,
            self.brush_size,
            self.brush_size
        )
        self.update(r)

    def save_to_history(self):
        self.history = self.history[: self.history_index + 1]
        self.history.append(self.image.copy())

        if len(self.history) > self.max_history:
            self.history.pop(0)
        else:
            self.history_index += 1

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.image = self.history[self.history_index].copy()
            self.update()

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.image = self.history[self.history_index].copy()
            self.update()

    def save_mask(self, path: str):
        if self.image.isNull():
            return
        w, h = self.image.width(), self.image.height()
        gray = QImage(w, h, QImage.Format_Grayscale8)
        gray.fill(0)
        for y in range(h):
            src = self.image.constScanLine(y)
            dst = gray.scanLine(y)
            dst[:w] = src[:w]
        gray.save(path)

    def clear(self):
        self.image.fill(Qt.transparent)
        self.history.clear()
        self.history_index = -1
        self.save_to_history()
        self.update()


class CanvasView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setRenderHint(QPainter.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)

        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self.setVerticalScrollBar(ScrollBar(Qt.Vertical, self))
        self.setHorizontalScrollBar(ScrollBar(Qt.Horizontal, self))

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.setCursor(Qt.ArrowCursor)

        self.image_item: QGraphicsPixmapItem | None = None
        self.mask_item: MaskItem | None = None
  
    def load_image(self, file_path):
        self.scene.clear()

        image = QImage(file_path)
        if image.isNull():
            return

        self.image_item = QGraphicsPixmapItem(QPixmap.fromImage(image))
        self.scene.addItem(self.image_item)

        self.mask_item = MaskItem(image.size())
        self.scene.addItem(self.mask_item)

        self.scene.setSceneRect(self.image_item.boundingRect())
        self.resetTransform()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)
            
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.mask_item:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.ArrowCursor)
            p = self.mapToScene(event.pos())
            self.mask_item.last_point = p
            self.mask_item.save_to_history()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.mask_item and self.mask_item.last_point:
            p = self.mapToScene(event.pos())
            self.mask_item.draw_line(self.mask_item.last_point, p)
            self.mask_item.last_point = p
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if self.mask_item:
                self.mask_item.last_point = None
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def clear_mask(self):
        if self.mask_item:
            self.mask_item.clear()
                
    def undo(self):
        if self.mask_item:
            self.mask_item.undo()

    def redo(self):
        if self.mask_item:
            self.mask_item.redo()
            
    def set_tool(self, tool: str):
        if self.mask_item:
            self.mask_item.current_tool = tool

    def set_brush_size(self, size: int):
        if self.mask_item:
            self.mask_item.brush_size = size
        
    def save_mask(self, path: str):
        if self.mask_item:
            self.mask_item.save_mask(path)

    def sizeHint(self):
        return QSize(1000, 600)


class WatermarkMaskTool(MyMessageBoxBase):
    def __init__(self, image_path, parent=None):
        super().__init__(parent=parent)
        self._init_title_bar()
        self.setModal(True)
        self.setDraggable(True)
        self.init_ui()
        self.load_image(file_path=image_path)
        self.image_path = image_path

    def _init_title_bar(self):
        title_bar = GradientHeader(
            parent=self,
            start=QColor(0, 120, 212),
            stop=QColor(0, 90, 158),
            fixed_height=48
        )
        layout = QHBoxLayout(title_bar)
        layout.setContentsMargins(24, 0, 0, 0)
        layout.setSpacing(8)

        buttonLayout = QHBoxLayout()
        buttonLayout.setSpacing(0)
        buttonLayout.setContentsMargins(0, 0, 0, 0)
        buttonLayout.setAlignment(Qt.AlignTop)

        closeBtn = CloseButton()
        closeBtn.setNormalColor(Qt.white)
        closeBtn.clicked.connect(self.reject)
        self.titleLabel = SubtitleLabel(self.tr("水印 Mask 标注"))
        setFont(self.titleLabel, 18)
        self.titleLabel.setStyleSheet("color: white;")
        buttonLayout.addWidget(closeBtn)

        layout.addWidget(self.titleLabel)
        layout.addStretch(1)
        layout.addLayout(buttonLayout)

        self.viewLayout.addWidget(title_bar)
        
    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        commandBar = self.create_command_bar()
        main_layout.addWidget(commandBar)
        main_layout.addSpacing(10)

        showLayout = QHBoxLayout()
        showLayout.setSpacing(0)
        showLayout.setContentsMargins(0, 0, 0, 0)
        
        self.canvas = CanvasView()
        showLayout.addWidget(self.canvas)
        
        control_panel = self.create_control_panel()
        showLayout.addWidget(control_panel)
        
        main_layout.addLayout(showLayout)

        self.viewLayout.addLayout(main_layout)
        
    def create_command_bar(self):
        commandBar = CommandBar()
        commandBar.setSpaing(8)
        commandBar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.brush_btn = Action(FluentIcon.EDIT, self.tr('画笔'), triggered=lambda: self.select_tool("brush"))
        self.brush_btn.setCheckable(True)
        self.brush_btn.setChecked(True)
        self.eraser_btn = Action(FluentIcon.ERASE_TOOL, self.tr('橡皮擦'), triggered=lambda: self.select_tool("eraser"))
        self.eraser_btn.setCheckable(True)

        commandBar.addAction(self.brush_btn)
        commandBar.addAction(self.eraser_btn)
        commandBar.addAction(Action(FluentIcon.CANCEL, self.tr('撤销'), triggered=self.undo))
        commandBar.addAction(Action(FluentIcon.ROTATE, self.tr('重做'), triggered=self.redo))
        commandBar.addSeparator()
        commandBar.addAction(Action(FluentIcon.CLEAR_SELECTION, self.tr('清空Mask'), triggered=self.clear_mask))
        commandBar.addAction(Action(FluentIcon.SAVE, self.tr('保存Mask'), triggered=self.save_mask))

        return commandBar
        
    def create_control_panel(self):
        panel = QFrame()
        panel.setMinimumWidth(200)
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        layout.setContentsMargins(24, 0, 24, 0)
        layout.setAlignment(Qt.AlignTop)

        size_group = QFrame()
        size_layout = QVBoxLayout(size_group)
        size_layout.setSpacing(8)
        size_layout.setContentsMargins(0, 0, 0, 0)
        
        size_label = QLabel(self.tr("画笔大小"))
        setFont(size_label, 16, QFont.DemiBold)
        size_layout.addWidget(size_label)
        
        self.size_value_label = QLabel("20px")
        self.size_value_label.setAlignment(Qt.AlignRight)
        size_layout.addWidget(self.size_value_label)
        
        self.size_slider = Slider(Qt.Horizontal)
        self.size_slider.setThemeColor(light=QColor(0, 120, 212), dark=QColor(0, 120, 212))
        self.size_slider.setRange(1, 100)
        self.size_slider.setValue(20)
        self.size_slider.valueChanged.connect(self.on_brush_size_changed)
        size_layout.addWidget(self.size_slider)
        
        layout.addWidget(size_group)
        
        info_group = QFrame()
        info_layout = QVBoxLayout(info_group)
        info_layout.setSpacing(8)
        info_layout.setContentsMargins(0, 0, 0, 0)
        
        info_title = QLabel(self.tr("操作说明"))
        setFont(info_title, 16, QFont.DemiBold)
        info_layout.addWidget(info_title)
        
        info_text = QLabel(
            "• 左键拖动绘制\n"
            "• 画笔：绘制红色mask\n"
            "• 橡皮擦：擦除mask\n"
            "• 支持撤销/重做操作\n"
            "• 保存为PNG格式"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #aaaaaa;")
        setFont(info_text, 14)
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        
        return panel
        
    def select_tool(self, tool):
        if tool == "brush":
            self.brush_btn.setChecked(True)
            self.eraser_btn.setChecked(False)
            self.canvas.set_tool("brush")
        else:
            self.brush_btn.setChecked(False)
            self.eraser_btn.setChecked(True)
            self.canvas.set_tool("eraser")
        self.update()
            
    def on_brush_size_changed(self, value):
        self.size_value_label.setText(f"{value}px")
        self.canvas.set_brush_size(value)

    def load_image(self, file_path):
        self.canvas.load_image(file_path)
                
    def save_mask(self):
        base_path = cfg.get(cfg.cachePath)
        mask_name = f"manual_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = os.path.join(base_path, "watermark_removal", mask_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.canvas.save_mask(file_path)
        TeachingTip.create(
            target=self,
            icon=InfoBarIcon.SUCCESS,
            title=self.tr("通知"),
            content=self.tr("Mask 保存成功！"),
            isClosable=True,
            tailPosition=TeachingTipTailPosition.BOTTOM,
            duration=2000,
            parent=self
        )
        global_event_bus.watermarkRemove_ManualMaskUpdate.emit(file_path)
                
    def undo(self):
        self.canvas.undo()
        
    def redo(self):
        self.canvas.redo()
        
    def clear_mask(self):
        self.canvas.clear_mask()
