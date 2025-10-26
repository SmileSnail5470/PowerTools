from PySide6.QtGui import QFont, QColor, QPainter, QPen, QBrush, QLinearGradient
from PySide6.QtWidgets import QWidget, QFrame, QHBoxLayout, QLabel, QGraphicsDropShadowEffect, QVBoxLayout, QListWidget, QListWidgetItem, QPushButton
from PySide6.QtCore import Qt, QTimer, QEasingCurve, QPropertyAnimation, Property, QRectF
from app.ui.library.qfluentwidgets import setFont, qconfig, Theme, theme



class ProgressRing(QWidget):
    """进度环组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(120, 120)
        self._percentage = 75
        self._animation = QPropertyAnimation(self, b"percentage")
        self._animation.setDuration(500)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        
    def get_percentage(self):
        return self._percentage
        
    def set_percentage(self, value):
        self._percentage = value
        self.update()
        
    percentage = Property(float, get_percentage, set_percentage)
    
    def set_percentage_animated(self, value):
        self._animation.setStartValue(self._percentage)
        self._animation.setEndValue(value)
        self._animation.start()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 获取中心点和半径
        center_x = self.width() // 2
        center_y = self.height() // 2
        radius = 52
        
        # 绘制背景圆环
        painter.setPen(QPen(QColor('#e5e7eb'), 8))
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # 绘制进度圆环
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor('#6b46c1'))
        gradient.setColorAt(1, QColor('#9333ea'))
        
        pen = QPen(QBrush(gradient), 8)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        
        # 计算角度
        start_angle = -90  # 从顶部开始
        span_angle = (self._percentage / 100) * 360
        
        # 绘制进度弧
        rect = QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2)
        painter.drawArc(rect, int(start_angle * 16), int(span_angle * 16))
        
        # 绘制百分比文字
        painter.setPen(QPen(QColor('#323130')))
        font = QFont('Segoe UI', 24, QFont.Bold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, f"{int(self._percentage)}%")

    
class StatCard(QFrame):
    """统计卡片组件"""
    def __init__(self, value, label, color_type="default", parent=None):
        super().__init__(parent)
        self.value = value
        self.label = label
        self.color_type = color_type
        self._scale = 1.0
        
        self.setFixedSize(100, 80)
        self.setFrameStyle(QFrame.NoFrame)
        self.setStyleSheet("""
            QFrame {
                background: transparent;
                border: none;
            }
        """)
        
        self.setup_ui()
        self.setup_shadow()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignCenter)
        
        # 数值标签
        self.value_label = QLabel(str(self.value))
        self.value_label.setAlignment(Qt.AlignCenter)
        
        # 根据类型设置颜色
        color_map = {
            "default": "#323130",
            "success": "#107c10",
            "error": "#d13438",
            "processing": "#0078d4"
        }
        color = color_map.get(self.color_type, "#323130")
        
        self.value_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 28px;
                font-weight: 700;
                background: transparent;
            }}
        """)
        self.value_label.setFont(QFont("Segoe UI", 28, QFont.Bold))
        
        # 说明标签
        self.desc_label = QLabel(self.label)
        self.desc_label.setAlignment(Qt.AlignCenter)
        self.desc_label.setStyleSheet("""
            QLabel {
                color: #605e5c;
                font-size: 12px;
                font-weight: 500;
                background: transparent;
                text-transform: uppercase;
            }
        """)
        self.desc_label.setFont(QFont("Segoe UI", 12, QFont.Medium))
        
        layout.addWidget(self.value_label)
        layout.addWidget(self.desc_label)
        
    def setup_shadow(self):
        """设置阴影效果"""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 20))
        self.setGraphicsEffect(shadow)
        
    def update_value(self, new_value):
        """更新数值"""
        self.value = new_value
        self.value_label.setText(str(new_value))
        
    def enterEvent(self, event):
        """鼠标进入事件"""
        self.animate_scale(1.05)
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        """鼠标离开事件"""
        self.animate_scale(1.0)
        super().leaveEvent(event)
        
    def mousePressEvent(self, event):
        """鼠标点击事件"""
        self.animate_scale(0.95)
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        self.animate_scale(1.05)
        if hasattr(self.parent(), 'on_stat_clicked'):
            self.parent().on_stat_clicked(self.objectName())
        super().mouseReleaseEvent(event)
        
    def animate_scale(self, target_scale):
        """缩放动画"""
        self.animation = QPropertyAnimation(self, b"scale")
        self.animation.setDuration(150)
        self.animation.setStartValue(self._scale)
        self.animation.setEndValue(target_scale)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation.start()
        
    def get_scale(self):
        return self._scale
        
    def set_scale(self, scale):
        self._scale = scale
        self.update_geometry()
        
    def update_geometry(self):
        """更新几何形状"""
        w = self.width() * self._scale
        h = self.height() * self._scale
        x = (self.width() - w) / 2
        y = (self.height() - h) / 2
        self.setGeometry(int(x), int(y), int(w), int(h))
        
    scale = Property(float, get_scale, set_scale)


class FailurePanel(QFrame):
    """失败信息面板"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_style()
        self._visible = False
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # 头部
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)
        
        # 失败图标
        self.failure_icon = QLabel("!")
        self.failure_icon.setFixedSize(20, 20)
        self.failure_icon.setAlignment(Qt.AlignCenter)
        self.failure_icon.setStyleSheet("""
            QLabel {
                background: #d13438;
                color: white;
                border-radius: 10px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        
        # 失败标题
        self.failure_title = QLabel("失败的文件列表")
        self.failure_title.setStyleSheet("""
            QLabel {
                color: #d13438;
                font-size: 14px;
                font-weight: 600;
            }
        """)
        
        header_layout.addWidget(self.failure_icon)
        header_layout.addWidget(self.failure_title)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # 失败列表
        self.failure_list = QListWidget()
        self.failure_list.setMaximumHeight(150)
        self.failure_list.setStyleSheet("""
            QListWidget {
                border: none;
                background: transparent;
            }
            QListWidget::item {
                background: white;
                border-radius: 6px;
                padding: 8px 12px;
                margin-bottom: 6px;
                font-size: 13px;
                color: #323130;
            }
            QListWidget::item:hover {
                background: #fee2e2;
            }
            QListWidget::item:selected {
                background: #fecaca;
            }
        """)
        
        layout.addWidget(self.failure_list)
        
    def setup_style(self):
        self.setStyleSheet("""
            QFrame {
                background: #fef2f2;
                border: 1px solid #fecaca;
                border-radius: 8px;
            }
        """)
        
    def set_visible(self, visible):
        """设置可见性"""
        self._visible = visible
        if visible:
            self.show()
            self.animate_show()
        else:
            self.animate_hide()
            
    def is_visible(self):
        return self._visible
        
    def animate_show(self):
        """显示动画"""
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        
        start_rect = self.geometry()
        start_rect.setHeight(0)
        end_rect = self.geometry()
        
        self.animation.setStartValue(start_rect)
        self.animation.setEndValue(end_rect)
        self.animation.start()
        
    def animate_hide(self):
        """隐藏动画"""
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QEasingCurve.InCubic)
        
        start_rect = self.geometry()
        end_rect = self.geometry()
        end_rect.setHeight(0)
        
        self.animation.setStartValue(start_rect)
        self.animation.setEndValue(end_rect)
        self.animation.finished.connect(self.hide)
        self.animation.start()
        
    def add_failure(self, filename, reason):
        """添加失败项"""
        item = QListWidgetItem(f"⚠️ {filename} - {reason}")
        self.failure_list.addItem(item)
        
    def clear_failures(self):
        """清空失败列表"""
        self.failure_list.clear()


class StatusInfoWidget(QWidget):
    """状态栏组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusBarWidget")
        
        # 数据模型
        self.status_data = {
            'total': 12,
            'processed': 8,
            'success': 6,
            'failed': 2,
            'failures': [
                ('document_001.pdf', '文件损坏'),
                ('image_045.jpg', '格式不支持')
            ]
        }
        
        self.processing_timer = None
        self.setup_ui()
        self.setup_style()
        self.update_display()
        
    def setup_ui(self):
        """设置UI布局"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 状态栏
        self.status_bar = QFrame()
        self.status_bar.setFixedHeight(120)
        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(24, 24, 24, 24)
        status_layout.setSpacing(32)
        
        # 统计卡片
        self.total_card = StatCard(12, "总任务数", "default")
        self.total_card.setObjectName("total")
        
        self.processed_card = StatCard(8, "已处理", "processing")
        self.processed_card.setObjectName("processed")
        
        self.success_card = StatCard(6, "成功数", "success")
        self.success_card.setObjectName("success")
        
        self.failed_card = StatCard(2, "失败数", "error")
        self.failed_card.setObjectName("failed")
        
        # 进度环
        self.progress_ring = ProgressRing()
        
        # 添加到布局
        status_layout.addWidget(self.total_card)
        status_layout.addWidget(self.processed_card)
        status_layout.addWidget(self.success_card)
        status_layout.addWidget(self.failed_card)
        status_layout.addStretch()
        status_layout.addWidget(self.progress_ring)
        
        # 失败信息面板
        self.failure_panel = FailurePanel()
        
        # 控制按钮
        self.controls_frame = QFrame()
        controls_layout = QHBoxLayout(self.controls_frame)
        controls_layout.setContentsMargins(24, 24, 24, 24)
        controls_layout.setSpacing(12)
        
        self.start_btn = QPushButton("▶️ 开始处理")
        self.start_btn.setObjectName("start_btn")
        
        self.reset_btn = QPushButton("🔄 重置数据")
        self.reset_btn.setObjectName("reset_btn")
        
        self.simulate_failure_btn = QPushButton("❌ 模拟失败")
        self.simulate_failure_btn.setObjectName("simulate_failure_btn")
        
        controls_layout.addStretch()
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.reset_btn)
        controls_layout.addWidget(self.simulate_failure_btn)
        controls_layout.addStretch()
        
        # 添加到主布局
        main_layout.addWidget(self.status_bar)
        main_layout.addWidget(self.failure_panel)
        main_layout.addWidget(self.controls_frame)
        
        # 初始隐藏失败面板
        self.failure_panel.hide()
        
        # 连接信号
        self.start_btn.clicked.connect(self.start_processing)
        self.reset_btn.clicked.connect(self.reset_data)
        self.simulate_failure_btn.clicked.connect(self.simulate_failure)
        
    def setup_style(self):
        """设置样式"""
        self.setStyleSheet("""
            QWidget#StatusBarWidget {
                background: white;
                border-radius: 12px;
            }
            QFrame {
                background: white;
            }
            QPushButton {
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                font-family: 'Segoe UI';
            }
            QPushButton#start_btn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6b46c1, stop:1 #9333ea);
                color: white;
            }
            QPushButton#start_btn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5a3aa8, stop:1 #822dc7);
            }
            QPushButton#reset_btn, QPushButton#simulate_failure_btn {
                background: #f3f2f1;
                color: #323130;
            }
            QPushButton#reset_btn:hover, QPushButton#simulate_failure_btn:hover {
                background: #e5e7eb;
            }
        """)
        
    def on_stat_clicked(self, stat_name):
        """统计卡片点击事件"""
        if stat_name == "failed":
            self.toggle_failure_panel()
        else:
            self.update_stat(stat_name)
            
    def toggle_failure_panel(self):
        """切换失败面板显示"""
        self.failure_panel.set_visible(not self.failure_panel.is_visible())
        
    def update_stat(self, stat_type):
        """更新统计数据"""
        if stat_type == "total":
            self.status_data['total'] = 15
        elif stat_type == "processed":
            self.status_data['processed'] = min(
                self.status_data['total'], 
                self.status_data['processed'] + 1
            )
        elif stat_type == "success":
            self.status_data['success'] = min(
                self.status_data['processed'], 
                self.status_data['success'] + 1
            )
        self.update_display()
        
    def update_display(self):
        """更新显示"""
        self.total_card.update_value(self.status_data['total'])
        self.processed_card.update_value(self.status_data['processed'])
        self.success_card.update_value(self.status_data['success'])
        self.failed_card.update_value(self.status_data['failed'])
        
        # 更新进度环
        percentage = (self.status_data['processed'] / self.status_data['total']) * 100
        self.progress_ring.set_percentage_animated(percentage)
        
        # 更新失败列表
        self.update_failure_list()
        
    def update_failure_list(self):
        """更新失败列表"""
        self.failure_panel.clear_failures()
        for filename, reason in self.status_data['failures']:
            self.failure_panel.add_failure(filename, reason)
            
    def start_processing(self):
        """开始处理"""
        if self.processing_timer:
            self.processing_timer.stop()
            self.processing_timer = None
            self.start_btn.setText("▶️ 开始处理")
            return
            
        self.start_btn.setText("⏸️ 停止处理")
        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self.process_step)
        self.processing_timer.start(1000)
        
    def process_step(self):
        """处理步骤"""
        if self.status_data['processed'] < self.status_data['total']:
            self.status_data['processed'] += 1
            
            # 随机决定成功或失败
            import random
            if random.random() > 0.2:
                self.status_data['success'] += 1
            else:
                self.status_data['failed'] += 1
                failure_names = ['report.docx', 'data.xlsx', 'presentation.pptx', 'archive.zip']
                failure_reasons = ['文件损坏', '格式不支持', '权限不足', '文件过大']
                
                random_name = failure_names[random.randint(0, len(failure_names)-1)]
                random_reason = failure_reasons[random.randint(0, len(failure_reasons)-1)]
                
                self.status_data['failures'].append((random_name, random_reason))
                
            self.update_display()
        else:
            self.processing_timer.stop()
            self.processing_timer = None
            self.start_btn.setText("▶️ 开始处理")
            self.show_notification("处理完成！")
            
    def reset_data(self):
        """重置数据"""
        if self.processing_timer:
            self.processing_timer.stop()
            self.processing_timer = None
            
        self.status_data = {
            'total': 12,
            'processed': 0,
            'success': 0,
            'failed': 0,
            'failures': []
        }
        
        self.start_btn.setText("▶️ 开始处理")
        self.update_display()
        self.failure_panel.set_visible(False)
        self.show_notification("数据已重置")
        
    def simulate_failure(self):
        """模拟失败"""
        self.status_data['failed'] = 3
        self.status_data['failures'] = [
            ('document_001.pdf', '文件损坏'),
            ('image_045.jpg', '格式不支持'),
            ('video_012.mp4', '编解码器不支持')
        ]
        self.update_display()
        self.failure_panel.set_visible(True)
        
    def show_notification(self, message):
        """显示通知"""
        # 这里可以实现一个通知组件，暂时用print代替
        print(f"通知: {message}")