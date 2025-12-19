from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QBrush, QPainter, QLinearGradient, QColor


class GradientHeader(QWidget):
    """渐变标题栏"""
    def __init__(self, parent=None, start: QColor = QColor(102, 126, 234), stop: QColor = QColor(118, 75, 162), fixed_height=80):
        super().__init__(parent=parent)
        self.setFixedHeight(fixed_height)
        self.gradient = QLinearGradient(0, 0, self.width(), self.height())
        self.gradient.setColorAt(0, start)  # #667eea
        self.gradient.setColorAt(1, stop)   # #764ba2
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.gradient.setStart(0, 0)
        self.gradient.setFinalStop(self.width(), self.height())
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QBrush(self.gradient))