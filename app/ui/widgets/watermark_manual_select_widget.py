import logging
import os
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QSlider, QLabel,
    QFileDialog, QMessageBox, QSplitter, QFrame, QHBoxLayout
)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter, QMouseEvent, QColor
from PIL import Image, ImageDraw, ImageQt
import numpy as np

from app.ui.library.qfluentwidgets import Action, MaskDialogBase, TeachingTip, InfoBarIcon, TeachingTipTailPosition, SubtitleLabel, CommandBar, FluentIcon, FluentStyleSheet
from app.ui.library.qframelesswindow.titlebar import CloseButton

from app.utils.logger.decorators import log_exception


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

        self.viewLayout.setSpacing(12)
        self.viewLayout.setContentsMargins(0, 0, 0, 0)

    def __setQss(self):
        FluentStyleSheet.DIALOG.apply(self)


class CanvasWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(600, 500)
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
            # 居中显示图片
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
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._init_title_bar()
        self.setModal(True)
        self.init_ui()

    def _init_title_bar(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(8)

        closeBtn = CloseButton()
        closeBtn.clicked.connect(self.reject)
        self.titleLabel = SubtitleLabel(self.tr("水印 Mask 标注"))

        layout.addWidget(self.titleLabel, 0, Qt.AlignVCenter)
        layout.addStretch(1)
        layout.addWidget(closeBtn, 0, Qt.AlignRight)
        layout.addStretch()

        self.viewLayout.addLayout(layout)
        
    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        commandBar = self.create_command_bar()
        main_layout.addWidget(commandBar)
        
        splitter = QSplitter(Qt.Horizontal)
        
        self.canvas = CanvasWidget()
        self.canvas.setStyleSheet("background-color: #2a2a3a; border-radius: 8px;")
        splitter.addWidget(self.canvas)
        
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)

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
        commandBar.addAction(Action(FluentIcon.SAVE, self.tr('保存Mask'), triggered=self.save_mask))

        return commandBar
        
    def create_control_panel(self):
        """创建控制面板"""
        panel = QFrame()
        panel.setMaximumWidth(500)
        layout = QVBoxLayout(panel)
        
        # 画笔大小控制
        size_group = QFrame()
        size_layout = QVBoxLayout(size_group)
        
        size_label = QLabel("画笔大小")
        size_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        size_layout.addWidget(size_label)
        
        self.size_value_label = QLabel("20px")
        self.size_value_label.setAlignment(Qt.AlignCenter)
        size_layout.addWidget(self.size_value_label)
        
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(1)
        self.size_slider.setMaximum(100)
        self.size_slider.setValue(20)
        self.size_slider.valueChanged.connect(self.on_brush_size_changed)
        size_layout.addWidget(self.size_slider)
        
        layout.addWidget(size_group)
        
        # 操作提示
        info_group = QFrame()
        info_layout = QVBoxLayout(info_group)
        
        info_title = QLabel("操作说明")
        info_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        info_layout.addWidget(info_title)
        
        info_text = QLabel(
            "• 左键拖动绘制\n"
            "• 画笔：绘制红色mask\n"
            "• 橡皮擦：擦除mask\n"
            "• 支持撤销/重做操作\n"
            "• 保存为PNG格式"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        
        # 快捷操作
        quick_group = QFrame()
        quick_layout = QVBoxLayout(quick_group)
        
        quick_title = QLabel("快捷操作")
        quick_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        quick_layout.addWidget(quick_title)
        
        clear_btn = QPushButton("清空Mask")
        clear_btn.clicked.connect(self.clear_mask)
        quick_layout.addWidget(clear_btn)
        
        layout.addWidget(quick_group)
        
        layout.addStretch()
        
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
        """画笔大小改变"""
        self.size_value_label.setText(f"{value}px")
        self.canvas.set_brush_size(value)
        
    def open_image(self):
        """打开图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", 
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            if self.canvas.load_image(file_path):
                self.statusBar().showMessage(f"已加载: {os.path.basename(file_path)}")
                
    def save_mask(self):
        """保存mask"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存Mask", f"mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG文件 (*.png)"
        )
        if file_path:
            if self.canvas.save_mask(file_path):
                self.statusBar().showMessage(f"已保存: {os.path.basename(file_path)}")
                QMessageBox.information(self, "成功", "Mask保存成功！")
                
    def undo(self):
        self.canvas.undo()
        self.accept()
        
    def redo(self):
        self.canvas.redo()
        self.reject()
        
    def clear_mask(self):
        """清空mask"""
        reply = QMessageBox.question(
            self, "确认", "确定要清空当前的mask吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if self.canvas.original_image:
                self.canvas.mask_image = Image.new('RGBA', 
                    self.canvas.original_image.size, (0, 0, 0, 0))
                self.canvas.clear_history()
                self.canvas.update_display()
                self.statusBar().showMessage("已清空mask")
