from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QGridLayout, 
    QMessageBox, QProgressDialog, QWidget
)
import sys
import time
from qfluentwidgets import setFont


param_name_map = {
    "input_path": "文件路径",
    "watermark_type": "水印类型",
    "watermark_text": "水印文本",
    "font": "字体",
    "font_size": "字体大小",
    "font_color": "字体颜色",
    "watermark_image": "图片水印位置",
    "watermark_opacity": "水印透明度",
    "watermark_content": "水印类型",
    "watermark_location": "水印位置",
    "watermark_rotation": "水印旋转角度",
    "watermark_zoom": "水印缩放比例",
    "output_path": "保存位置",
    "output_format": "保存格式",
}

class TaskInfoMessageBox(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("任务信息总结")
        self.setModal(True)
        
        # 示例数据
        self.sample_data = {
            "input": "/data/projects/2025/samples/input/image001.avif",
            "output": "/data/projects/2025/samples/output/image001_watermarked.avif",
            "params": {
                "watermark_text": "Confidential",
                "font": "PingFang",
                "opacity": 0.35,
                "position": "bottom-right",
                "margin": 20,
                "size": 32,
                "rotation": 15,
                "color": "#ffffff",
                "repeat": False
            }
        }
        
        self.setup_ui()
        self.apply_styles()
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        header = QFrame()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 20, 24, 20)
        
        title_label = QLabel("📋 任务信息总结")
        title_label.setStyleSheet("color: black;")  # 设置为浅灰色
        setFont(title_label, 15, QFont.Bold)
        
        self.time_label = QLabel(time.strftime("%Y-%m-%d %H:%M"))
        self.time_label.setStyleSheet("""
            QLabel {
                background-color: #f1f3f5;
                color: #6c757d;
                padding: 6px 12px;
                border-radius: 20px;
                border: 1px solid #dee2e6;
                font-size: 14px;
            }
        """)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.time_label)
        
        body = QFrame()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(24, 20, 24, 20)
        body_layout.setSpacing(12)
        
        # 输入路径区域
        input_section = self.create_section("📁 处理文件路径")
        self.input_path = self.create_path_label(self.sample_data["input"])
        input_section.layout().addWidget(self.input_path)
        
        # 参数区域
        params_section = self.create_section("⚙️ 水印参数")
        params_grid = QGridLayout()
        params_grid.setSpacing(16)
        
        row, col = 0, 0
        for key, value in self.sample_data["params"].items():
            param_widget = self.create_param_widget(key, value)
            params_grid.addWidget(param_widget, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1
        
        params_section.layout().addLayout(params_grid)
        
        # 输出路径区域
        output_section = self.create_section("💾 输出保存路径")
        self.output_path = self.create_path_label(self.sample_data["output"])
        output_section.layout().addWidget(self.output_path)
        
        body_layout.addWidget(input_section)
        body_layout.addWidget(params_section)
        body_layout.addWidget(output_section)
        
        # 底部区域
        footer = QFrame()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(24, 20, 24, 20)
        
        tip_label = QLabel("💡 提示：点击路径或参数可复制内容")
        tip_label.setStyleSheet("color: #6c757d; font-size: 13px;")
        
        cancel_btn = QPushButton("取消")
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.clicked.connect(self.close)
        
        confirm_btn = QPushButton("确认并开始执行")
        confirm_btn.setCursor(Qt.PointingHandCursor)
        confirm_btn.clicked.connect(self.start_task)
        
        footer_layout.addWidget(tip_label)
        footer_layout.addStretch()
        footer_layout.addWidget(cancel_btn)
        footer_layout.addWidget(confirm_btn)
        
        main_layout.addWidget(header)
        main_layout.addWidget(body)
        main_layout.addWidget(footer)
        
    def create_section(self, title):
        section = QFrame()
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(20, 16, 20, 20)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setStyleSheet("""
            QLabel {
                color: #212529;
                padding-bottom: 8px;
                border-bottom: 2px solid #7c3aed;
            }
        """)
        
        section_layout.addWidget(title_label)
        return section
    
    def create_path_label(self, path):
        label = QLabel(path)
        label.setCursor(Qt.PointingHandCursor)
        label.setStyleSheet("""
            QLabel {
                font-family: "Consolas", "Monaco", monospace;
                color: #0d6efd;
                font-size: 14px;
                padding: 12px 16px;
                background-color: #f1f3f5;
                border-radius: 8px;
                border: 1px solid #dee2e6;
            }
            QLabel:hover {
                background-color: #e9ecef;
                border: 1px solid #7c3aed;
            }
        """)
        label.mousePressEvent = lambda e: self.copy_to_clipboard(path)
        return label

    def create_param_widget(self, key, value):
        widget = QFrame()
        widget.setCursor(Qt.PointingHandCursor)
        widget.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 10px;
                padding: 16px;
            }
            QFrame:hover {
                border: 1px solid #7c3aed;
            }
        """)
        
        layout = QVBoxLayout(widget)
        
        key_label = QLabel(key.replace("_", " ").title())
        key_label.setStyleSheet("""
            QLabel {
                color: #6c757d;
                font-size: 12px;
                font-weight: 600;
                letter-spacing: 0.5px;
            }
        """)
        key_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        value_label = QLabel(str(value))
        value_label.setStyleSheet("""
            QLabel {
                font-weight: 600;
                font-size: 14px;
                color: #212529;
                background-color: #f1f3f5;
                padding: 8px 10px;
                border-radius: 6px;
                border: 1px solid #e9ecef;
            }
        """)
        value_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        layout.addWidget(key_label)
        layout.addWidget(value_label)
        
        widget.mousePressEvent = lambda e: self.copy_to_clipboard(str(value))
        return widget
    
    def copy_to_clipboard(self, text):
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        self.show_copy_toast()
    
    def show_copy_toast(self):
        msg = QMessageBox(self)
        msg.setText("✓ 已复制到剪贴板")
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 8px;
            }
            QMessageBox QLabel {
                color: white;
                font-size: 14px;
                font-weight: 600;
            }
        """)
        msg.setStandardButtons(QMessageBox.NoButton)
        msg.show()
        
        # 2秒后自动关闭
        QTimer.singleShot(2000, msg.close)
    
    def start_task(self):
        # 确认对话框
        confirm = QMessageBox(self)
        confirm.setWindowTitle("任务执行确认")
        confirm.setText("确认开始执行水印任务吗？")
        confirm.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        confirm.setDefaultButton(QMessageBox.Yes)
        
        if confirm.exec() != QMessageBox.Yes:
            return
        
        # 执行进度对话框
        progress = QProgressDialog("正在处理水印任务，请稍候...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setFixedSize(300, 100)
        progress.setCancelButton(None)
        progress.show()
        
        # 模拟任务执行
        QTimer.singleShot(2000, lambda: self.task_completed(progress))
    
    def task_completed(self, progress):
        progress.close()
        
        # 完成提示
        complete = QMessageBox(self)
        complete.setWindowTitle("任务完成")
        complete.setText("✅ 水印任务已成功执行！")
        complete.setStandardButtons(QMessageBox.Ok)
        complete.exec()
        
        self.close()
    
    def apply_styles(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #fafbfc;
            }
            QFrame {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 12px;
            }
            QPushButton {
                padding: 12px 20px;
                border-radius: 10px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton#cancel {
                background-color: transparent;
                border: 1px solid #dee2e6;
                color: #6c757d;
            }
            QPushButton#cancel:hover {
                background-color: #f1f3f5;
                border: 1px solid #7c3aed;
                color: #7c3aed;
            }
            QPushButton#confirm {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #7c3aed, stop:1 #06b6d4);
                color: white;
                border: none;
            }
            QPushButton#confirm:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #8b5cf6, stop:1 #22d3ee);
            }
        """)
        
        # 设置按钮对象名称
        for btn in self.findChildren(QPushButton):
            if btn.text() == "取消":
                btn.setObjectName("cancel")
            elif btn.text() == "确认并开始执行":
                btn.setObjectName("confirm")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("任务信息演示")
        self.setFixedSize(400, 200)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        demo_btn = QPushButton("显示任务信息")
        demo_btn.setCursor(Qt.PointingHandCursor)
        demo_btn.setFixedSize(200, 50)
        demo_btn.clicked.connect(self.show_task_dialog)
        demo_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #7c3aed, stop:1 #06b6d4);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #8b5cf6, stop:1 #22d3ee);
            }
        """)
        
        layout.addWidget(demo_btn)
    
    def show_task_dialog(self):
        dialog = TaskInfoMessageBox(self)
        dialog.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())