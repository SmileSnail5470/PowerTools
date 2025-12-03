from PySide6.QtGui import QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, QPixmap
from PySide6.QtWidgets import (
    QWidget, QFrame, QHBoxLayout, QLabel, QGraphicsDropShadowEffect, QVBoxLayout, QListWidget, 
    QListWidgetItem, QPushButton, QApplication
)
from PySide6.QtCore import Qt, QEasingCurve, QPropertyAnimation, Property, QRectF, Signal, QRect
from app.ui.library.qfluentwidgets import setFont
from app.ui.common.task_status import TaskStatusModel


class ProgressRing(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 100)
        self._percentage = 0.0
        self._animation = QPropertyAnimation(self, b"percentage")
        self._animation.setDuration(500)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)

        # 缓存背景图层
        self._background_pixmap = None
        
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

    def resizeEvent(self, event):
        # 缓存背景图像，避免每次 paint 重画静态圆环
        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.transparent)
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#e5e7eb"), 8)
        p.setPen(pen)
        cx, cy = self.width() // 2, self.height() // 2
        radius = min(self.width(), self.height()) // 2 - 8
        p.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)
        p.end()
        self._background_pixmap = pixmap
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self._background_pixmap:
            painter.drawPixmap(0, 0, self._background_pixmap)

        # 获取中心点和半径
        center_x = self.width() // 2
        center_y = self.height() // 2
        radius = min(self.width(), self.height()) // 2 - 8
        
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
        setFont(painter, 24, QFont.Bold)
        painter.drawText(self.rect(), Qt.AlignCenter, f"{int(self._percentage)}%")
        painter.end()

    
class StatCard(QFrame):
    scaleChanged = Signal(float)
    valueTranslateChanged = Signal(float)
    clicked = Signal(str)

    def __init__(self, value, label, color_type="default", parent=None):
        super().__init__(parent)
        self.value = value
        self.label = label
        self.color_type = color_type
        self._scale = 1.0
        self._value_translate = 0.0
        self._underline_width = 0.0
        
        # self.setFixedSize(100, 80)
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
                background: transparent;
            }}
        """)
        setFont(self.value_label, 24, QFont.Bold)
        
        # 说明标签
        self.desc_label = QLabel(self.label)
        self.desc_label.setAlignment(Qt.AlignCenter)
        self.desc_label.setStyleSheet("""
            QLabel {
                color: #605e5c;
                background: transparent;
                text-transform: uppercase;
            }
        """)
        setFont(self.desc_label, 12, QFont.Medium)
        
        layout.addWidget(self.value_label)
        layout.addWidget(self.desc_label)
        
    def setup_shadow(self):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 20))
        self.setGraphicsEffect(shadow)
        
    def update_value(self, new_value):
        self.value = new_value
        self.value_label.setText(str(new_value))
        
    def enterEvent(self, event):
        self.animate_scale(1.05)
        self.animate_underline(60)
        self.animate_value_translate(-2)
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self.animate_scale(1.0)
        self.animate_underline(0)
        self.animate_value_translate(0)
        super().leaveEvent(event)
        
    def mousePressEvent(self, event):
        self.animate_scale(0.95)
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        self.animate_scale(1.05)
        self.clicked.emit(self.objectName())
        super().mouseReleaseEvent(event)
        
    def animate_scale(self, target_scale):
        self.animation = QPropertyAnimation(self, b"scale")
        self.animation.setDuration(150)
        self.animation.setStartValue(self._scale)
        self.animation.setEndValue(target_scale)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation.start()

    def animate_underline(self, target_width):
        self.underline_animation = QPropertyAnimation(self, b"underline_width")
        self.underline_animation.setDuration(300)
        self.underline_animation.setStartValue(self._underline_width)
        self.underline_animation.setEndValue(target_width)
        self.underline_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.underline_animation.start()
        
    def animate_value_translate(self, target_translate):
        self.translate_animation = QPropertyAnimation(self, b"value_translate")
        self.translate_animation.setDuration(300)
        self.translate_animation.setStartValue(self._value_translate)
        self.translate_animation.setEndValue(target_translate)
        self.translate_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.translate_animation.start()
        
    def get_scale(self):
        return self._scale
        
    def set_scale(self, scale):
        self._scale = scale
        self.scaleChanged.emit(scale)
        self.updateGeometry()

    def get_underline_width(self):
        return self._underline_width
        
    def set_underline_width(self, width):
        self._underline_width = width
        self.update()
        
    def get_value_translate(self):
        return self._value_translate
        
    def set_value_translate(self, translate):
        self._value_translate = translate
        self.valueTranslateChanged.emit(translate)
        self.update()
        
    scale = Property(float, get_scale, set_scale, notify=scaleChanged)
    underline_width = Property(float, get_underline_width, set_underline_width)
    value_translate = Property(float, get_value_translate, set_value_translate, notify=valueTranslateChanged)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制下划线
        if self._underline_width > 0:
            painter.setPen(QPen(QColor('#6b46c1'), 2))
            underline_y = self.height() - 8
            underline_x = (self.width() - self._underline_width) / 2
            painter.drawLine(int(underline_x), underline_y, int(underline_x + self._underline_width), underline_y)
        
        painter.end()


class FailurePopupWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FailurePopupWidget")
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.setup_ui()
        self.setup_shadow()
        self.hide()
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)

        self.content_widget = QWidget()
        self.content_widget.setObjectName("contentWidget")
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(16, 12, 16, 12)
        content_layout.setSpacing(10)
        
        # 头部
        header_layout = QHBoxLayout()
        header_layout.setSpacing(6)
        
        # 失败图标
        self.failure_icon = QLabel("❌")
        self.failure_icon.setFixedSize(20, 20)
        self.failure_icon.setAlignment(Qt.AlignCenter)
        self.failure_icon.setStyleSheet("""
            QLabel {
                color: #d13438;
            }
        """)
        setFont(self.failure_icon, 10)
        
        # 失败标题
        self.failure_title = QLabel(self.tr("失败的文件列表"))
        self.failure_title.setStyleSheet("""
            QLabel {
                color: #b91c1c;
            }
        """)
        setFont(self.failure_title, 12, QFont.DemiBold)

        self.close_button = QPushButton("✕")
        self.close_button.setFixedSize(24, 24)
        self.close_button.setStyleSheet("""
            QPushButton {
                border: none;
                background: transparent;
                color: #737373;
                font-size: 16px;
                border-radius: 12px;
            }
            QPushButton:hover {
                background: rgba(220, 38, 38, 0.1);
                color: #dc2626;
            }
        """)
        self.close_button.clicked.connect(self.hide)
        
        header_layout.addWidget(self.failure_icon)
        header_layout.addWidget(self.failure_title)
        header_layout.addStretch()
        header_layout.addWidget(self.close_button)
        
        content_layout.addLayout(header_layout)
        
        # 失败列表
        self.failure_list = QListWidget()
        self.failure_list.setMaximumHeight(150)
        self.failure_list.setStyleSheet("""
            QListWidget {
                border: none;
                background: transparent;
            }
            QListWidget::item {
                background: #fff;
                border-radius: 8px;
                padding: 6px 10px;
                margin-bottom: 4px;
                color: #374151;
            }
            QListWidget::item:hover {
                background: #fee2e2;
            }
        """)
        setFont(self.failure_list, 10, QFont.Normal)
        
        content_layout.addWidget(self.failure_list)

        main_layout.addWidget(self.content_widget)

        self.setStyleSheet("""
            QWidget#contentWidget {
                background: rgba(254, 242, 242, 0.95);
                border: 1px solid #fecaca;
                border-radius: 12px;
            }
        """)

    def setup_shadow(self):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.content_widget.setGraphicsEffect(shadow)
        
    def show_at(self, target_widget):
        card_rect = target_widget.rect()
        global_pos = target_widget.mapToGlobal(card_rect.bottomLeft())

        self.adjustSize()
        popup_width = self.width()
        popup_height = self.height()

        # 获取可视区域
        screen_rect = QApplication.primaryScreen().availableGeometry()

        popup_x = global_pos.x() + (target_widget.width() - popup_width) // 2
        popup_y = global_pos.y() + 8  # 下方偏移一点

        # 边界检测（防止超出屏幕）
        if popup_x + popup_width > screen_rect.right():
            popup_x = screen_rect.right() - popup_width - 8
        if popup_x < screen_rect.left():
            popup_x = screen_rect.left() + 8
        if popup_y + popup_height > screen_rect.bottom():
            # 如果太靠下，就显示在目标上方
            popup_y = global_pos.y() - popup_height - 8

        self.move(popup_x, popup_y)
        self.show()

        # 弹出动画
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(200)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        start_rect = QRect(popup_x, popup_y - 10, popup_width, popup_height)
        end_rect = QRect(popup_x, popup_y, popup_width, popup_height)
        self.animation.setStartValue(start_rect)
        self.animation.setEndValue(end_rect)
        self.animation.start()
        
    def add_failure(self, filename, reason):
        item = QListWidgetItem(f"⚠️ {filename}")
        self.failure_list.addItem(item)
        
    def clear_failures(self):
        self.failure_list.clear()


class StatusInfoWidget(QWidget):
    def __init__(self, model: TaskStatusModel, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusInfoWidget")
        self.model = model
        self.model.updated.connect(self.update_display)
        
        self.setup_ui()
        self.setup_style()
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 状态栏
        self.status_bar = QFrame()
        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(20, 0, 20, 0)
        status_layout.setSpacing(20)
        
        # 统计卡片
        self.total_card = StatCard(0, self.tr("总任务数"), "default")
        self.total_card.setObjectName("total")

        self.processed_card = StatCard(0, self.tr("已处理"), "processing")
        self.processed_card.setObjectName("processed")

        self.success_card = StatCard(0, self.tr("成功数"), "success")
        self.success_card.setObjectName("success")

        self.failed_card = StatCard(0, self.tr("失败数"), "error")
        self.failed_card.setObjectName("failed")
        
        # 进度环
        self.progress_ring = ProgressRing()

        self.total_card.clicked.connect(self.on_stat_clicked)
        self.processed_card.clicked.connect(self.on_stat_clicked)
        self.success_card.clicked.connect(self.on_stat_clicked)
        self.failed_card.clicked.connect(self.on_stat_clicked)
        
        # 添加到布局
        status_layout.addWidget(self.total_card)
        status_layout.addWidget(self.processed_card)
        status_layout.addWidget(self.success_card)
        status_layout.addWidget(self.failed_card)
        status_layout.addStretch()
        status_layout.addWidget(self.progress_ring)
        
        # 失败信息面板
        self.failure_popup = FailurePopupWidget()
        
        # 添加到主布局
        main_layout.addWidget(self.status_bar)
        
    def setup_style(self):
        self.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 8px;
                border: 1px solid rgba(200, 200, 200, 0.3);
            }
        """)
        
    def on_stat_clicked(self, stat_name):
        if stat_name == "failed":
            self.toggle_failure_popup()
            
    def toggle_failure_popup(self):
        if self.failure_popup.isVisible():
            self.failure_popup.hide()
        else:
            
            self.failure_popup.show_at(self.failed_card)
        
    def update_display(self, data):
        self.total_card.update_value(data['total'])
        self.processed_card.update_value(data['processed'])
        self.success_card.update_value(data['success'])
        self.failed_card.update_value(data['failed'])
        
        # 更新进度环
        if data['total'] == 0:
            percentage = 0
        else:
            percentage = (data['processed'] / data['total']) * 100
        self.progress_ring.set_percentage_animated(percentage)
        
        # 更新失败列表
        self.update_failure_list(data)
        
    def update_failure_list(self, data):
        self.failure_popup.clear_failures()
        for filename, reason in data['failures']:
            self.failure_popup.add_failure(filename, reason)