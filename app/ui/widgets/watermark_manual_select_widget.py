import os
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QHBoxLayout, QScrollArea
)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter, QMouseEvent, QColor, QFont
from PIL import Image, ImageDraw, ImageQt
import numpy as np

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


class CanvasWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        
        self.original_image = None
        self.mask_image = None
        self.display_image = None
        
        self.is_drawing = False
        self.current_tool = "brush"  # brush or eraser
        self.brush_size = 20
        self.last_point = QPoint()
        
        self.history = []
        self.history_index = -1
        self.max_history = 50
        
    def set_original_image(self, pil_image):
        self.original_image = pil_image.copy()
        self.mask_image = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        self.setFixedSize(pil_image.width, pil_image.height)
        self.update_display()
        self.clear_history()
  
    def load_image(self, file_path):
        img = Image.open(file_path).convert('RGB')
        self.set_original_image(img)
            
    def update_display(self):
        if self.original_image is None:
            return
            
        display = self.original_image.convert('RGBA')
        
        if self.mask_image:
            # 将mask以半透明红色显示
            mask_overlay = Image.new('RGBA', display.size, (255, 0, 0, 100))
            mask_overlay.putalpha(self.mask_image.getchannel('A'))
            display = Image.alpha_composite(display, mask_overlay)
            
        self.display_image = ImageQt.toqpixmap(display)
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.display_image:
            widget_size = self.size()
            image_size = self.display_image.size()
            
            x = (widget_size.width() - image_size.width()) // 2
            y = (widget_size.height() - image_size.height()) // 2
            
            painter.drawPixmap(x, y, self.display_image)
            
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.is_drawing = True
            self.last_point = self.get_image_point(event.position())
            self.save_to_history()
            
    def mouseMoveEvent(self, event: QMouseEvent):
        if self.is_drawing and self.last_point:
            current_point = self.get_image_point(event.position())
            if current_point:
                self.draw_line(self.last_point, current_point)
                self.last_point = current_point
                
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.is_drawing = False
            self.last_point = QPoint()
            
    def get_image_point(self, widget_point):
        """将控件坐标转换为图片坐标
        
        """
        if not self.display_image:
            return None
            
        widget_size = self.size()
        image_size = self.display_image.size()
        
        x = (widget_size.width() - image_size.width()) // 2
        y = (widget_size.height() - image_size.height()) // 2
        
        image_x = widget_point.x() - x
        image_y = widget_point.y() - y
        
        if 0 <= image_x < image_size.width() and 0 <= image_y < image_size.height():
            return QPoint(image_x, image_y)
        return None
        
    def draw_line(self, start_point, end_point):
        if not self.mask_image:
            return
            
        draw = ImageDraw.Draw(self.mask_image)
        
        if self.current_tool == "brush":
            # 画笔：绘制红色
            color = (255, 0, 0, 255)
        else:
            # 橡皮擦：擦除（透明）
            color = (0, 0, 0, 0)
            
        # 绘制粗线条
        draw.line([start_point.x(), start_point.y(), end_point.x(), end_point.y()], fill=color, width=self.brush_size)
        
        # 绘制端点圆圈以保持连续性
        draw.ellipse(
            [start_point.x() - self.brush_size//2, start_point.y() - self.brush_size//2, start_point.x() + self.brush_size//2, start_point.y() + self.brush_size//2], 
            fill=color
        )
        draw.ellipse(
            [end_point.x() - self.brush_size//2, end_point.y() - self.brush_size//2, end_point.x() + self.brush_size//2, end_point.y() + self.brush_size//2], 
            fill=color
        )
        self.update_display()
        
    def save_to_history(self):
        if self.mask_image:
            # 删除当前索引之后的历史记录
            self.history = self.history[:self.history_index + 1]
            
            # 添加新的历史记录
            self.history.append(self.mask_image.copy())
            
            # 限制历史记录数量
            if len(self.history) > self.max_history:
                self.history.pop(0)
            else:
                self.history_index += 1
                
    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.mask_image = self.history[self.history_index].copy()
            self.update_display()
            
    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.mask_image = self.history[self.history_index].copy()
            self.update_display()
            
    def clear_history(self):
        self.history = []
        self.history_index = -1
        if self.mask_image:
            self.save_to_history()
            
    def set_tool(self, tool):
        self.current_tool = tool
        
    def set_brush_size(self, size):
        self.brush_size = size
        
    def save_mask(self, file_path):
        if self.mask_image:
            # 创建纯黑白mask
            mask_data = np.array(self.mask_image)
            alpha = mask_data[:, :, 3]
            
            # 保存为PNG（保持透明度）
            mask_to_save = Image.fromarray(alpha).convert("L")
            mask_to_save.save(file_path)


class WatermarkMaskTool(MyMessageBoxBase):
    def __init__(self, image_path, parent=None):
        super().__init__(parent=parent)
        self._init_title_bar()
        self.setModal(True)
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
        
        scroll = QScrollArea()
        scroll.setMinimumSize(800, 600)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll.setHorizontalScrollBar(ScrollBar(Qt.Horizontal, scroll))
        scroll.horizontalScrollBar().setFade(True)
        scroll.setVerticalScrollBar(ScrollBar(Qt.Vertical, scroll))
        scroll.verticalScrollBar().setFade(True)
        self.canvas = CanvasWidget()
        scroll.setWidget(self.canvas)
        showLayout.addWidget(scroll)
        
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
        if self.canvas.original_image:
            self.canvas.mask_image = Image.new('RGBA', self.canvas.original_image.size, (0, 0, 0, 0))
            self.canvas.clear_history()
            self.canvas.update_display()
