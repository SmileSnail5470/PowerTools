from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, Property, QSize
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QPen


class ToggleSwitch(QWidget):
    toggled = Signal(bool)

    def __init__(self, parent=None, width=44, height=24, on_color="#4f46e5"):
        super().__init__(parent)
        self._active = False
        self._anim_pos = 0.0  # 初始在左边
        self._animation = QPropertyAnimation(self, b"animPos", self)
        self._animation.setDuration(150)
        self._animation.setEasingCurve(QEasingCurve.InOutQuad)

        self._on_color = QColor(on_color)
        self._off_color = QColor("#d1d5db")
        self._knob_color = QColor("#ffffff")

        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFixedSize(width, height)

    def getAnimPos(self):
        return self._anim_pos

    def setAnimPos(self, v):
        self._anim_pos = v
        self.update()

    animPos = Property(float, getAnimPos, setAnimPos)

    def isActive(self) -> bool:
        return self._active

    def setActive(self, active: bool, animated: bool = True):
        if self._active == bool(active):
            return
        self._active = bool(active)
        start = self._anim_pos
        end = 1.0 if self._active else 0.0
        self._animation.stop()
        if animated:
            self._animation.setStartValue(start)
            self._animation.setEndValue(end)
            self._animation.start()
        else:
            self._anim_pos = end
            self.update()
        self.toggled.emit(self._active)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setActive(not self._active, animated=True)
            self.clearFocus()
            event.accept()
        else:
            super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self.setActive(not self._active, animated=True)
            event.accept()
        else:
            super().keyPressEvent(event)

    def sizeHint(self) -> QSize:
        return QSize(44, 24)

    def paintEvent(self, event):
        w = self.width()
        h = self.height()
        radius = h / 2.0
        margin = max(2.0, h * 0.08)  # 边距随高度缩放
        knob_d = h - 2 * margin
        x_min = margin
        x_max = w - margin - knob_d
        knob_x = x_min + (x_max - x_min) * self._anim_pos

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        on_rgba = self._on_color
        off_rgba = self._off_color

        def lerp(a, b, t): 
            return a + (b - a) * t
        
        t = self._anim_pos
        bg_color = QColor(
            int(lerp(off_rgba.red(), on_rgba.red(), t)),
            int(lerp(off_rgba.green(), on_rgba.green(), t)),
            int(lerp(off_rgba.blue(), on_rgba.blue(), t)),
            int(lerp(off_rgba.alpha(), on_rgba.alpha(), t))
        )

        painter.setPen(Qt.NoPen)
        painter.setBrush(bg_color)
        painter.drawRoundedRect(0, 0, w, h, radius, radius)

        if self.hasFocus():
            pen = QPen(QColor(0, 0, 0, 30))
            pen.setWidthF(max(1.0, h * 0.06))
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(0.5, 0.5, w - 1, h - 1, radius, radius)
            painter.setPen(Qt.NoPen)

        painter.setBrush(self._knob_color)
        painter.drawEllipse(int(knob_x), int(margin), int(knob_d), int(knob_d))

    def toggle(self):
        self.setActive(not self._active)