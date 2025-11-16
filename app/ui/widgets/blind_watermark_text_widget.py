from PySide6.QtCore import Qt, QEasingCurve, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QLabel, QPushButton
)
from app.ui.library.qfluentwidgets import setFont, FlowLayout


class FluentTagButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)

        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(36, 36)
        self.setStyleSheet("""
            FluentTagButton {
                background-color: #F3F3F3;
                border-radius: 12px;
                padding: 4px 12px;
                color: #333;
                border: 1px solid #E0E0E0;
            }
            FluentTagButton:hover {
                background-color: #FFFFFF;
                border: 1px solid #D0D0D0;
            }
            FluentTagButton:pressed {
                background-color: #E9E9E9;
            }
        """)


class BlindWatermarkInputPanel(QWidget):
    textUpdate = Signal(str)

    def __init__(self, allowed_chars=None, parent=None):
        super().__init__(parent)

        if allowed_chars is None:
            allowed_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ,1234")

        self.allowed_chars = allowed_chars

        self.setObjectName("BlindWatermarkInputPanel")
        self.setup_ui()
        self.apply_style()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        label = QLabel(self.tr("水印文本"))
        setFont(label, 13)
        label.setStyleSheet("color: #888888;")  # 设置为浅灰色
        layout.addWidget(label)

        self.input = QLineEdit()
        self.input.setPlaceholderText(self.tr("输入水印文字"))
        self.input.setMinimumHeight(36)
        setFont(self.input, 13)
        self.input.textChanged.connect(lambda s: self.textUpdate.emit(s))
        layout.addWidget(self.input)

        char_label = QLabel(self.tr("可选字符集（最大字符长度 33）"))
        setFont(char_label, 10)
        char_label.setStyleSheet("color: #888888;")  # 设置为浅灰色
        layout.addWidget(char_label)

        flow_layout = FlowLayout(None, needAni=True)
        flow_layout.setAnimation(250, QEasingCurve.OutQuad)
        flow_layout.setContentsMargins(10, 10, 10, 10)
        flow_layout.setVerticalSpacing(8)
        flow_layout.setHorizontalSpacing(4)

        char_container = QWidget()
        char_container.setLayout(flow_layout)

        for ch in self.allowed_chars:
            btn = FluentTagButton(ch)
            btn.clicked.connect(lambda checked, c=ch: self.append_char(c))
            flow_layout.addWidget(btn)
        layout.addWidget(char_container)

    def apply_style(self):
        self.setStyleSheet(
            """
            #BlindWatermarkInputPanel QLineEdit {
                border: 1px solid #cccccc;
                border-radius: 6px;
                padding: 6px 10px;
                color: #333;
            }

            #BlindWatermarkInputPanel QLineEdit:focus {
                border: 1px solid #4A90E2;
            }
            """
        )

    def append_char(self, ch):
        current = self.input.text()
        self.input.setText(current + ch)
