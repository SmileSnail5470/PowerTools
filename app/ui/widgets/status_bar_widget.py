import random
from PySide6.QtGui import QFont, QColor, QPainter, QPen, QBrush, QConicalGradient, QPixmap
from PySide6.QtWidgets import (
    QWidget, QFrame, QHBoxLayout, QLabel, QGraphicsDropShadowEffect, QVBoxLayout, QListWidget, 
    QListWidgetItem, QPushButton, QApplication, QSizePolicy
)
from PySide6.QtCore import Qt, QEasingCurve, QPropertyAnimation, Property, QRectF, Signal, QRect, QTimer, QSize
from app.ui.library.qfluentwidgets import setFont
from app.ui.common.task_status import TaskStatusModel, TaskStatus, TaskState


class HourglassIcon(QWidget):
    """旋转和脉冲动画的沙漏图标"""
    def __init__(self, hourg_size=42, parent=None):
        super().__init__(parent)
        self.setFixedSize(hourg_size, hourg_size)
        self.hourg_size = hourg_size
        self._rotation = 0.0
        self._scale = 1.0

        self.rot_anim = QPropertyAnimation(self, b"rotation")
        self.rot_anim.setStartValue(0.0)
        self.rot_anim.setEndValue(360.0)
        self.rot_anim.setDuration(2500)
        self.rot_anim.setLoopCount(-1)
        self.rot_anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.rot_anim.start()

        self.scale_anim = QPropertyAnimation(self, b"scale_factor")
        self.scale_anim.setStartValue(1.0)
        self.scale_anim.setKeyValueAt(0.5, 1.1)
        self.scale_anim.setEndValue(1.0)
        self.scale_anim.setDuration(2500)
        self.scale_anim.setLoopCount(-1)
        self.scale_anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.scale_anim.start()

    def get_rotation(self): 
        return self._rotation
    
    def set_rotation(self, val): 
        self._rotation = val
        self.update()
    
    def get_scale_factor(self): 
        return self._scale
    
    def set_scale_factor(self, val): 
        self._scale = val
        self.update()

    rotation = Property(float, get_rotation, set_rotation)
    scale_factor = Property(float, get_scale_factor, set_scale_factor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(107, 70, 193, 20))
        painter.drawEllipse(0, 0, self.hourg_size, self.hourg_size)

        painter.translate(self.hourg_size // 2, self.hourg_size // 2)
        painter.scale(self._scale, self._scale)
        painter.rotate(self._rotation)

        pen = QPen(QColor("#6b46c1"), 1.8)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)

        painter.drawLine(-4, -5, 4, -5)
        painter.drawLine(-4, 5, 4, 5)
        painter.drawLine(-3, -5, 3, 5)
        painter.drawLine(3, -5, -3, 5)
        painter.end()


class SpeedTrendWidget(QWidget):
    """实时跳动的效率波纹"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 20)
        self.bars = [40.0, 60.0, 55.0, 85.0, 95.0]
        self.target_bars = [40.0, 60.0, 55.0, 85.0, 95.0]
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_bars)
        self.timer.start(120)

    def update_bars(self):
        for i in range(4):
            if abs(self.bars[i] - self.target_bars[i]) < 5:
                self.target_bars[i] = random.randint(30, 95)
            self.bars[i] += (self.target_bars[i] - self.bars[i]) * 0.2
        if self.bars[4] >= 90:
            self.target_bars[4] = 30
        elif self.bars[4] <= 35:
            self.target_bars[4] = 95
        self.bars[4] += (self.target_bars[4] - self.bars[4]) * 0.3
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        bar_width = 3
        spacing = 2
        x = 0
        for i, h_pct in enumerate(self.bars):
            h = max(2, int(20 * (h_pct / 100.0)))
            y = 20 - h
            if i == 4:
                painter.setBrush(QColor("#6b46c1"))
            else:
                painter.setBrush(QColor(107, 70, 193, 51))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(x, y, bar_width, h, 1.5, 1.5)
            x += bar_width + spacing
        painter.end()


class ProgressRing(QWidget):
    def __init__(self, ring_size=100, parent=None):
        super().__init__(parent)
        self._ring_size = ring_size
        self.setFixedSize(ring_size, ring_size)
        self._percentage = 0.0
        self._animation = QPropertyAnimation(self, b"percentage")
        self._animation.setDuration(500)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._rotation_angle = 0.0
        self._spin_animation = QPropertyAnimation(self, b"rotation_angle")
        self._spin_animation.setDuration(2000)
        self._spin_animation.setStartValue(0.0)
        self._spin_animation.setEndValue(360.0)
        self._spin_animation.setLoopCount(-1)
        self._spin_animation.setEasingCurve(QEasingCurve.Linear)
        self._background_pixmap = None
        if ring_size >= 80:
            self._font_size = 24
            self._pen_width = 8
        elif ring_size >= 60:
            self._font_size = 16
            self._pen_width = 6
        else:
            self._font_size = 12
            self._pen_width = 5

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

    def get_rotation_angle(self):
        return self._rotation_angle

    def set_rotation_angle(self, value):
        self._rotation_angle = value
        self.update()

    rotation_angle = Property(float, get_rotation_angle, set_rotation_angle)

    def start_spin(self):
        if self._spin_animation.state() != QPropertyAnimation.Running:
            self._spin_animation.start()

    def stop_spin(self):
        if self._spin_animation.state() == QPropertyAnimation.Running:
            self._spin_animation.stop()
        self._rotation_angle = 0.0
        self.update()

    def resizeEvent(self, event):
        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.transparent)
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#e5e7eb"), self._pen_width)
        p.setPen(pen)
        cx, cy = self.width() // 2, self.height() // 2
        radius = min(self.width(), self.height()) // 2 - self._pen_width
        p.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)
        p.end()
        self._background_pixmap = pixmap

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self._background_pixmap:
            painter.drawPixmap(0, 0, self._background_pixmap) 

        is_running = self._spin_animation.state() == QPropertyAnimation.Running
        if self._percentage < 0.1 and not is_running:
            painter.setPen(QPen(QColor('#323130')))
            setFont(painter, self._font_size, QFont.Bold)
            painter.drawText(self.rect(), Qt.AlignCenter, "0%")
            painter.end()
            return

        center_x = self.width() // 2
        center_y = self.height() // 2
        radius = min(self.width(), self.height()) // 2 - self._pen_width
        rect = QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2)
        painter.save()
        painter.translate(center_x, center_y)
        painter.rotate(self._rotation_angle)
        painter.translate(-center_x, -center_y)
        gradient = QConicalGradient(center_x, center_y, 90)
        gradient.setColorAt(0.0, QColor('#6b46c1'))
        gradient.setColorAt(0.5, QColor('#9333ea'))
        gradient.setColorAt(1.0, QColor('#6b46c1'))
        start_angle = 90
        if self._percentage < 0.5:
            span_angle = -18
        else:
            span_angle = -(self._percentage / 100) * 360
        pen = QPen(QBrush(gradient), self._pen_width)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, int(start_angle * 16), int(span_angle * 16))
        painter.restore()
        painter.setPen(QPen(QColor('#323130')))
        setFont(painter, self._font_size, QFont.Bold)
        painter.drawText(self.rect(), Qt.AlignCenter, f"{int(self._percentage)}%")
        painter.end()


class BatchPipelineBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TimeEstimator")
        self.setMinimumWidth(240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setup_ui()

    def sizeHint(self):
        return QSize(max(280, self.width()), 40)

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 8, 0, 0)
        main_layout.setSpacing(0)

        left_layout = QHBoxLayout()
        left_layout.setSpacing(6)

        self.icon = HourglassIcon()
        left_layout.addWidget(self.icon)

        meta_left = QVBoxLayout()
        meta_left.setSpacing(8)
        meta_left.setAlignment(Qt.AlignVCenter)
        meta_left.setContentsMargins(0, 0, 0, 0)

        self.lbl_eta_title = QLabel(self.tr("预计剩余时间"))
        self.lbl_eta_title.setStyleSheet("color: #605e5c; background: transparent;")
        setFont(self.lbl_eta_title, 9, QFont.DemiBold)

        self.lbl_eta_value = QLabel(self.tr("00:00:00"))
        self.lbl_eta_value.setStyleSheet("color: #6b46c1; background: transparent;")
        setFont(self.lbl_eta_value, 12, QFont.Bold)

        meta_left.addWidget(self.lbl_eta_title)
        meta_left.addWidget(self.lbl_eta_value)
        left_layout.addLayout(meta_left)

        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFixedSize(1, 24)
        divider.setStyleSheet("background-color: rgba(200, 200, 200, 0.4); border: none;")

        right_layout = QHBoxLayout()
        right_layout.setSpacing(12)
        right_layout.setAlignment(Qt.AlignRight)

        meta_right = QVBoxLayout()
        meta_right.setSpacing(8)
        meta_right.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        meta_right.setContentsMargins(0, 0, 0, 0)

        self.lbl_spd_title = QLabel(self.tr("当前处理速率"))
        self.lbl_spd_title.setStyleSheet("color: #605e5c; background: transparent;")
        self.lbl_spd_title.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        setFont(self.lbl_spd_title, 9, QFont.DemiBold)

        self.lbl_spd_value = QLabel("-- 分钟/个")
        self.lbl_spd_value.setStyleSheet("color: #323130; background: transparent;")
        self.lbl_spd_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        setFont(self.lbl_spd_value, 11, QFont.Bold)

        meta_right.addWidget(self.lbl_spd_title)
        meta_right.addWidget(self.lbl_spd_value)

        self.trend_chart = SpeedTrendWidget()

        right_layout.addLayout(meta_right)
        right_layout.addWidget(self.trend_chart, 0, Qt.AlignVCenter)

        main_layout.addLayout(left_layout)
        main_layout.addStretch()
        main_layout.addWidget(divider)
        main_layout.addStretch()
        main_layout.addLayout(right_layout)

    def update_eta(self, seconds: int):
        if seconds <= 0:
            self.lbl_eta_value.setText(self.tr("00:00:00"))
            self.lbl_eta_value.setStyleSheet("color: #107c10; background: transparent;")
            self.icon.rot_anim.pause()
            self.icon.scale_anim.pause()
            self.trend_chart.timer.stop()
            for i in range(5): 
                self.trend_chart.bars[i] = 15.0
            self.trend_chart.update()
        else:
            if self.icon.rot_anim.state() == QPropertyAnimation.Paused:
                self.icon.rot_anim.resume()
                self.icon.scale_anim.resume()
                self.trend_chart.timer.start()
            hrs = int(seconds) // 3600
            mins = (int(seconds) % 3600) // 60
            secs = int(seconds) % 60
            self.lbl_eta_value.setText(f"{hrs:02d}:{mins:02d}:{secs:02d}")
            self.lbl_eta_value.setStyleSheet("color: #6b46c1; background: transparent;")

    def update_speed(self, speed_str: str):
        self.lbl_spd_value.setText(speed_str)


class PipelineBar(QWidget):
    def __init__(self, steps=None, parent=None):
        super().__init__(parent)
        # steps: [{'name': str, 'status': 'completed'|'running'|'pending', 'duration': str}]
        self.steps = steps or []
        self.setMinimumWidth(240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def sizeHint(self):
        return QSize(max(240, len(self.steps) * 70), 54)

    def set_steps(self, steps):
        self.steps = steps
        self.update()

    def update_step(self, index, status, duration=None):
        if 0 <= index < len(self.steps):
            self.steps[index]['status'] = status
            if duration is not None:
                self.steps[index]['duration'] = duration
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        n = len(self.steps)
        if n == 0:
            painter.end()
            return
        step_width = self.width() / n
        dot_y = 14
        dot_radius = 7
        color_map = {
            "completed": QColor("#107c10"),
            "running": QColor("#0078d4"),
            "pending": QColor("#d1d5db"),
        }
        for i, step in enumerate(self.steps):
            cx = int(step_width * i + step_width / 2)
            color = color_map.get(step['status'], QColor("#d1d5db"))
            # 连接线
            if i < n - 1:
                next_cx = int(step_width * (i + 1) + step_width / 2)
                if step['status'] == 'completed':
                    line_color = QColor("#107c10")
                elif step['status'] == 'running':
                    line_color = QColor("#93c5fd")
                else:
                    line_color = QColor("#e5e7eb")
                pen = QPen(line_color, 2)
                pen.setCapStyle(Qt.RoundCap)
                painter.setPen(pen)
                painter.drawLine(cx + dot_radius + 4, dot_y, next_cx - dot_radius - 4, dot_y)
            # 圆点
            painter.setPen(Qt.NoPen)
            if step['status'] == 'completed':
                painter.setBrush(QBrush(color))
                painter.drawEllipse(cx - dot_radius, dot_y - dot_radius, dot_radius * 2, dot_radius * 2)
                # 白色对勾
                painter.setPen(QPen(QColor("white"), 1.8))
                painter.drawLine(cx - 2, dot_y + 0.5, cx, dot_y + 2.5)
                painter.drawLine(cx, dot_y + 2.5, cx + 3, dot_y - 2)
                painter.setPen(Qt.NoPen)
            elif step['status'] == 'running':
                # 外圈脉冲 + 内实心
                painter.setBrush(QBrush(QColor("#dbeafe")))
                painter.drawEllipse(cx - dot_radius - 3, dot_y - dot_radius - 3, (dot_radius + 3) * 2, (dot_radius + 3) * 2)
                painter.setBrush(QBrush(color))
                painter.drawEllipse(cx - dot_radius + 1, dot_y - dot_radius + 1, (dot_radius - 1) * 2, (dot_radius - 1) * 2)
            else:
                # 空心灰色圆
                painter.setBrush(QBrush(QColor("#f3f4f6")))
                painter.drawEllipse(cx - dot_radius, dot_y - dot_radius, dot_radius * 2, dot_radius * 2)
                painter.setPen(QPen(QColor("#d1d5db"), 1.5))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(cx - dot_radius, dot_y - dot_radius, dot_radius * 2, dot_radius * 2)
                painter.setPen(Qt.NoPen)

            # 步骤名称
            name_color = color if step['status'] != 'pending' else QColor("#9ca3af")
            painter.setPen(QPen(name_color))
            setFont(painter, 9, QFont.Medium)
            name_rect = QRect(int(step_width * i), dot_y + dot_radius + 6, int(step_width), 14)
            painter.drawText(name_rect, Qt.AlignCenter, step['name'])
            # 耗时
            if step['status'] == 'pending':
                dur_color = QColor("#d1d5db")
            elif step['status'] == 'running':
                dur_color = QColor("#60a5fa")
            else:
                dur_color = QColor("#6b7280")
            painter.setPen(QPen(dur_color))
            setFont(painter, 10, QFont.Normal)
            dur_rect = QRect(int(step_width * i), dot_y + dot_radius + 20, int(step_width), 12)
            painter.drawText(dur_rect, Qt.AlignCenter, step.get('duration', '--'))
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
        self.setFrameStyle(QFrame.NoFrame)
        self.setStyleSheet("""
            QFrame {
                background: transparent;
                border: none;
            }
        """)
        self.setup_ui()

    def _format_value(self, value):
        try:
            return f"{int(value):,}"
        except (ValueError, TypeError):
            return str(value)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 2)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignCenter)

        self.value_label = QLabel(self._format_value(self.value))
        self.value_label.setAlignment(Qt.AlignCenter)
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
        setFont(self.value_label, 16, QFont.Bold)

        self.desc_label = QLabel(self.label)
        self.desc_label.setAlignment(Qt.AlignCenter)
        self.desc_label.setStyleSheet("""
            QLabel {
                color: #605e5c;
                background: transparent;
            }
        """)
        setFont(self.desc_label, 9, QFont.Medium)

        layout.addWidget(self.value_label)
        layout.addWidget(self.desc_label)

    def update_value(self, new_value):
        self.value = new_value
        self.value_label.setText(self._format_value(new_value))

    def enterEvent(self, event):
        self.animate_scale(1.05)
        self.animate_underline(26)
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
        if self._underline_width > 0:
            painter.setPen(QPen(QColor('#6b46c1'), 2))
            underline_y = self.height() - 1
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

        self.failure_icon = QLabel("❌")
        self.failure_icon.setFixedSize(20, 20)
        self.failure_icon.setAlignment(Qt.AlignCenter)
        self.failure_icon.setStyleSheet("QLabel { color: #d13438; }")
        setFont(self.failure_icon, 10)

        self.failure_title = QLabel(self.tr("失败的文件列表"))
        self.failure_title.setStyleSheet("QLabel { color: #b91c1c; }")
        setFont(self.failure_title, 12, QFont.DemiBold)

        self.close_button = QPushButton("✕")
        self.close_button.setFixedSize(24, 24)
        self.close_button.setStyleSheet("""
            QPushButton {
                border: none; background: transparent;
                color: #737373; font-size: 16px; border-radius: 12px;
            }
            QPushButton:hover {
                background: rgba(220, 38, 38, 0.1); color: #dc2626;
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
            QListWidget { border: none; background: transparent; }
            QListWidget::item {
                background: #fff; border-radius: 8px;
                padding: 6px 10px; margin-bottom: 4px; color: #374151;
            }
            QListWidget::item:hover { background: #fee2e2; }
        """)
        setFont(self.failure_list, 10, QFont.Normal)
        content_layout.addWidget(self.failure_list)

        main_layout.addWidget(self.content_widget)

        self.setStyleSheet("""
            QWidget#contentWidget {
                background: rgba(254, 242, 242, 0.95);
                border: 1px solid #fecaca; border-radius: 12px;
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
        screen_rect = QApplication.primaryScreen().availableGeometry()
        popup_x = global_pos.x() + (target_widget.width() - popup_width) // 2
        popup_y = global_pos.y() + 8

        if popup_x + popup_width > screen_rect.right():
            popup_x = screen_rect.right() - popup_width - 8
        if popup_x < screen_rect.left():
            popup_x = screen_rect.left() + 8
        if popup_y + popup_height > screen_rect.bottom():
            popup_y = global_pos.y() - popup_height - 8

        self.move(popup_x, popup_y)
        self.show()

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


class BackendInfoWidget(QWidget):
    def __init__(self, gpu_type="GPU 运行", elapsed="00:00:00", parent=None):
        super().__init__(parent)
        self.setup_ui(gpu_type, elapsed)

    def setup_ui(self, gpu_type, elapsed):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)

        self.gpu_tag = QLabel(gpu_type)
        self.gpu_tag.setAlignment(Qt.AlignCenter)
        self.gpu_tag.setStyleSheet("""
            QLabel {
                color: #6b46c1;
                background: rgba(107, 70, 193, 0.1);
                border-radius: 9px;
                padding: 2px 12px;
            }
        """)
        setFont(self.gpu_tag, 10, QFont.Medium)

        self.time_label = QLabel(self.tr("耗时: ") + elapsed)
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("color: #323130; background: transparent;")
        setFont(self.time_label, 10, QFont.Bold)

        layout.addWidget(self.gpu_tag)
        layout.setSpacing(6)
        layout.addWidget(self.time_label)

    def set_elapsed(self, elapsed):
        self.time_label.setText(self.tr("耗时: ") + elapsed)

    def set_gpu_type(self, gpu_type):
        self.gpu_tag.setText(gpu_type)


class VerticalSeparator(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.VLine)
        self.setFixedWidth(1)
        self.setStyleSheet("color: #e5e7eb;")

class HorizontalSeparator(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFixedHeight(1)
        self.setStyleSheet("color: #e5e7eb;")


class StatusInfoWidget(QFrame):
    def __init__(self, model: TaskStatusModel, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusInfoWidget")
        self.model = model
        self.model.updated.connect(self.update_display)
        self.setup_ui()
        self.setup_style()

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 6, 0, 6)
        main_layout.setSpacing(0)

        self.info_bar = QWidget()
        info_layout = QVBoxLayout(self.info_bar)
        info_layout.setContentsMargins(10, 0, 10, 0)
        info_layout.setAlignment(Qt.AlignVCenter)
        info_layout.setSpacing(0)

        self.status_bar = QWidget()
        status_layout = QVBoxLayout(self.status_bar)
        status_layout.setContentsMargins(10, 0, 10, 0)
        status_layout.setAlignment(Qt.AlignVCenter)
        status_layout.setSpacing(0)

        self.task_bar = QWidget()
        task_layout = QHBoxLayout(self.task_bar)
        task_layout.setContentsMargins(10, 0, 10, 0)
        task_layout.setAlignment(Qt.AlignVCenter)

        self.pipeline_widget = PipelineBar()
        self.batch_pipeline_widget = BatchPipelineBar()
        self.batch_pipeline_widget.hide()

        h_separator = HorizontalSeparator()

        self.total_card = StatCard(0, self.tr("总任务"), "default")
        self.total_card.setObjectName("total")
        self.processed_card = StatCard(0, self.tr("已处理"), "processing")
        self.processed_card.setObjectName("processed")
        self.success_card = StatCard(0, self.tr("成功数"), "success")
        self.success_card.setObjectName("success")
        self.failed_card = StatCard(0, self.tr("失败数"), "error")
        self.failed_card.setObjectName("failed")

        v_separator = VerticalSeparator()

        self.gpu_info = BackendInfoWidget()

        self.progress_ring = ProgressRing(ring_size=66)

        self.total_card.clicked.connect(self.on_stat_clicked)
        self.processed_card.clicked.connect(self.on_stat_clicked)
        self.success_card.clicked.connect(self.on_stat_clicked)
        self.failed_card.clicked.connect(self.on_stat_clicked)

        info_layout.addWidget(self.pipeline_widget)
        info_layout.addWidget(self.batch_pipeline_widget)
        info_layout.addStretch(1)
        info_layout.addWidget(h_separator)

        task_layout.addWidget(self.total_card)
        task_layout.addWidget(self.processed_card)
        task_layout.addWidget(self.success_card)
        task_layout.addWidget(self.failed_card)

        info_layout.addWidget(self.task_bar)

        status_layout.addWidget(self.gpu_info)
        status_layout.addStretch(1)
        status_layout.addWidget(self.progress_ring)

        self.failure_popup = FailurePopupWidget()

        main_layout.addWidget(self.info_bar)
        main_layout.addWidget(v_separator)
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

    def set_backend_info(self, backend_type, elapsed):
        self.gpu_info.set_gpu_type(backend_type)
        self.gpu_info.set_elapsed(elapsed)

    def show_pipeline_widget(self):
        self.pipeline_widget.show()
        self.batch_pipeline_widget.hide()

    def show_batch_pipeline_widget(self):
        self.batch_pipeline_widget.show()
        self.pipeline_widget.hide()

    def update_failure_list(self, data):
        self.failure_popup.clear_failures()
        for filename, reason in data['failures']:
            self.failure_popup.add_failure(filename, reason)

    def update_display(self, status: TaskStatus):
        batch = status.batch
        self.total_card.update_value(batch.total)
        self.processed_card.update_value(batch.processed)
        self.success_card.update_value(batch.success)
        self.failed_card.update_value(batch.failed)

        self.progress_ring.set_percentage_animated(self.model.progress_percent)
        if status.state == TaskState.RUNNING:
            self.progress_ring.start_spin()
        else:
            self.progress_ring.stop_spin()
        self.batch_pipeline_widget.update_eta(int(status.performance.eta_seconds))
        avg = status.performance.avg_seconds_per_file
        if avg > 60:
            speed_text = f"{avg / 60:.1f} 分钟/个"
        else:
            speed_text = f"{avg:.1f} 秒/个"
        self.batch_pipeline_widget.update_speed(speed_text)
        elapsed = int(status.performance.elapsed_seconds)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        self.set_backend_info(backend_type=status.backend.backend_type, elapsed=f"{h:02}:{m:02}:{s:02}")
        self.pipeline_widget.set_steps([
            {
                "name": step.name,
                "status": step.state.value,
                "duration": "--" if step.display_duration == 0 else f"{step.display_duration:.1f}s"
            }
            for step in status.pipeline_steps
        ])
        self.update_failure_list({"failures": batch.failures})
