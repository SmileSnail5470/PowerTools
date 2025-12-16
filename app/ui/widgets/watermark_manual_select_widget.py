import sys
import os
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QSlider, QLabel, QToolBar, 
    QFileDialog, QMessageBox, QSplitter, QFrame
)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter, QMouseEvent
from PIL import Image, ImageDraw, ImageQt
import numpy as np

class CanvasWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setMouseTracking(True)
        
        # 画布状态
        self.original_image = None
        self.mask_image = None
        self.display_image = None
        
        # 绘制状态
        self.is_drawing = False
        self.current_tool = "brush"  # brush or eraser
        self.brush_size = 20
        self.last_point = QPoint()
        
        # 撤销/重做历史
        self.history = []
        self.history_index = -1
        self.max_history = 50
        
        # 加载默认图片
        self.load_default_image()
        
    def load_default_image(self):
        """创建一个默认的测试图片"""
        # 创建一个渐变背景的测试图片
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # 绘制一些测试内容
        for i in range(0, 800, 50):
            draw.line([(i, 0), (i, 600)], fill=(200, 200, 200), width=1)
        for i in range(0, 600, 50):
            draw.line([(0, i), (800, i)], fill=(200, 200, 200), width=1)
            
        # 添加一些文字作为水印示例
        try:
            # 尝试使用默认字体
            draw.text((300, 250), "Sample Watermark", fill=(100, 100, 100))
            draw.text((320, 300), "© 2024 Example", fill=(150, 150, 150))
        except:
            # 如果没有字体，跳过文字绘制
            pass
            
        self.set_original_image(img)
        
    def set_original_image(self, pil_image):
        """设置原始图片"""
        self.original_image = pil_image.copy()
        self.mask_image = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        self.update_display()
        self.clear_history()
        
    def load_image(self, file_path):
        """加载图片文件"""
        try:
            img = Image.open(file_path).convert('RGB')
            self.set_original_image(img)
            return True
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法加载图片: {str(e)}")
            return False
            
    def update_display(self):
        """更新显示图片"""
        if self.original_image is None:
            return
            
        # 合成原图和mask
        display = self.original_image.convert('RGBA')
        
        if self.mask_image:
            # 将mask以半透明红色显示
            mask_overlay = Image.new('RGBA', display.size, (255, 0, 0, 100))
            mask_overlay.putalpha(self.mask_image.getchannel('A'))
            display = Image.alpha_composite(display, mask_overlay)
            
        self.display_image = ImageQt.toqpixmap(display)
        self.update()
        
    def paintEvent(self, event):
        """绘制事件"""
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
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            self.is_drawing = True
            self.last_point = self.get_image_point(event.position())
            self.save_to_history()
            
    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件"""
        if self.is_drawing and self.last_point:
            current_point = self.get_image_point(event.position())
            if current_point:
                self.draw_line(self.last_point, current_point)
                self.last_point = current_point
                
    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            self.is_drawing = False
            self.last_point = QPoint()
            
    def get_image_point(self, widget_point):
        """将控件坐标转换为图片坐标"""
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
        """在mask上绘制线条"""
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
        draw.line([start_point.x(), start_point.y(), 
                  end_point.x(), end_point.y()], 
                 fill=color, width=self.brush_size)
        
        # 绘制端点圆圈以保持连续性
        draw.ellipse([start_point.x() - self.brush_size//2, 
                     start_point.y() - self.brush_size//2,
                     start_point.x() + self.brush_size//2, 
                     start_point.y() + self.brush_size//2], 
                    fill=color)
        draw.ellipse([end_point.x() - self.brush_size//2, 
                     end_point.y() - self.brush_size//2,
                     end_point.x() + self.brush_size//2, 
                     end_point.y() + self.brush_size//2], 
                    fill=color)
        
        self.update_display()
        
    def save_to_history(self):
        """保存到历史记录"""
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
        """撤销"""
        if self.history_index > 0:
            self.history_index -= 1
            self.mask_image = self.history[self.history_index].copy()
            self.update_display()
            
    def redo(self):
        """重做"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.mask_image = self.history[self.history_index].copy()
            self.update_display()
            
    def clear_history(self):
        """清空历史记录"""
        self.history = []
        self.history_index = -1
        if self.mask_image:
            self.save_to_history()
            
    def set_tool(self, tool):
        """设置当前工具"""
        self.current_tool = tool
        
    def set_brush_size(self, size):
        """设置画笔大小"""
        self.brush_size = size
        
    def save_mask(self, file_path):
        """保存mask"""
        if self.mask_image:
            try:
                # 创建纯黑白mask
                mask_data = np.array(self.mask_image)
                alpha = mask_data[:, :, 3]
                
                # 保存为PNG（保持透明度）
                mask_to_save = Image.fromarray(alpha).convert("L")
                mask_to_save.save(file_path)
                return True
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
                return False
        return False

class WatermarkMaskTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水印Mask标注工具")
        self.setGeometry(100, 100, 1200, 800)
        
        # 设置深色主题样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QWidget {
                background-color: #16213e;
                color: #ffffff;
            }
            QPushButton {
                background-color: #3a3a4a;
                border: 1px solid #5a5a6a;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a4a5a;
            }
            QPushButton:pressed {
                background-color: #2a2a3a;
            }
            QPushButton:checked {
                background-color: #667eea;
            }
            QSlider::groove:horizontal {
                border: 1px solid #5a5a6a;
                height: 8px;
                background: #2a2a3a;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #667eea;
                border: 2px solid #1a1a2e;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QToolBar {
                background-color: #1a1a2e;
                border: 1px solid #3a3a4a;
                spacing: 6px;
                padding: 4px;
            }
            QFrame {
                background-color: #1a1a2e;
                border: 1px solid #3a3a4a;
                border-radius: 8px;
            }
        """)
        
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 工具栏
        toolbar = self.create_toolbar()
        main_layout.addWidget(toolbar)
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：画布
        self.canvas = CanvasWidget()
        self.canvas.setStyleSheet("background-color: #2a2a3a; border-radius: 8px;")
        splitter.addWidget(self.canvas)
        
        # 右侧：控制面板
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
    def create_toolbar(self):
        """创建工具栏"""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        
        # 文件操作
        open_btn = QPushButton("打开图片")
        open_btn.clicked.connect(self.open_image)
        toolbar.addWidget(open_btn)
        
        save_btn = QPushButton("保存Mask")
        save_btn.clicked.connect(self.save_mask)
        toolbar.addWidget(save_btn)
        
        toolbar.addSeparator()
        
        # 工具选择
        self.brush_btn = QPushButton("🖌️ 画笔")
        self.brush_btn.setCheckable(True)
        self.brush_btn.setChecked(True)
        self.brush_btn.clicked.connect(lambda: self.select_tool("brush"))
        toolbar.addWidget(self.brush_btn)
        
        self.eraser_btn = QPushButton("🧹 橡皮擦")
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.clicked.connect(lambda: self.select_tool("eraser"))
        toolbar.addWidget(self.eraser_btn)
        
        toolbar.addSeparator()
        
        # 撤销/重做
        undo_btn = QPushButton("↶ 撤销")
        undo_btn.clicked.connect(self.undo)
        toolbar.addWidget(undo_btn)
        
        redo_btn = QPushButton("↷ 重做")
        redo_btn.clicked.connect(self.redo)
        toolbar.addWidget(redo_btn)
        
        return toolbar
        
    def create_control_panel(self):
        """创建控制面板"""
        panel = QFrame()
        panel.setMaximumWidth(300)
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
        """选择工具"""
        if tool == "brush":
            self.brush_btn.setChecked(True)
            self.eraser_btn.setChecked(False)
            self.canvas.set_tool("brush")
            self.statusBar().showMessage("画笔工具")
        else:
            self.brush_btn.setChecked(False)
            self.eraser_btn.setChecked(True)
            self.canvas.set_tool("eraser")
            self.statusBar().showMessage("橡皮擦工具")
            
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
        """撤销"""
        self.canvas.undo()
        self.statusBar().showMessage("已撤销")
        
    def redo(self):
        """重做"""
        self.canvas.redo()
        self.statusBar().showMessage("已重做")
        
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

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = WatermarkMaskTool()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
