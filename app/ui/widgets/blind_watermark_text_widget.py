from PySide6.QtCore import Qt, QEasingCurve, Signal, QSize, QRegularExpression
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QLabel, QPushButton, QGraphicsDropShadowEffect, QHBoxLayout
)
from PySide6.QtGui import QColor, QRegularExpressionValidator
from app.ui.library.qfluentwidgets import setFont, FlowLayout, TeachingTip, InfoBarIcon, TeachingTipTailPosition, FluentIcon


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


class DeleteButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(36, 36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setText("")

        self._setup_style()
        
        self._setup_shadow()
        
        self.setIcon(FluentIcon.DELETE.icon())
        self.setIconSize(QSize(20, 20))
        
    def _setup_style(self):
        self.setStyleSheet("""
            DeleteButton {
                background-color: transparent;
                border: none;
                border-radius: 16px;
                padding: 0px;
            }
            DeleteButton:hover {
                background-color: rgba(244, 67, 54, 0.1);
            }
            DeleteButton:pressed {
                background-color: rgba(244, 67, 54, 0.2);
            }
            DeleteButton QAbstractButton {
                background: transparent;
            }
        """)
        
    def _setup_shadow(self):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(8)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 20))
        self.setGraphicsEffect(shadow)


class BlindWatermarkInputPanel(QWidget):
    textUpdate = Signal(str)

    def __init__(self, allowed_chars=None, max_length = 15, parent=None):
        super().__init__(parent)

        if allowed_chars is None:
            allowed_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ,1234")

        self.allowed_chars = allowed_chars
        self.max_length = max_length

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

        h_layout = QHBoxLayout()
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(8)

        self.input = QLineEdit()
        self.input.setReadOnly(False)
        self.input.setMaxLength(self.max_length)
        self.input.setValidator(QRegularExpressionValidator(QRegularExpression("[A-Za-z1-4]*")))
        self.input.setPlaceholderText(self.tr("输入水印文字"))
        self.input.setMinimumHeight(36)
        setFont(self.input, 13)
        self.input.textChanged.connect(lambda s: self.textUpdate.emit(s.upper()))
        h_layout.addWidget(self.input)

        remove_btn = DeleteButton(self)
        remove_btn.clicked.connect(lambda: self.input.clear())
        h_layout.addWidget(remove_btn)

        layout.addLayout(h_layout)

        char_label = QLabel(self.tr(f"可选字符集（最大字符长度 {self.max_length}）"))
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
        if len(current) >= self.max_length:
            TeachingTip.create(
                target=self.input,
                icon=InfoBarIcon.WARNING,
                title=self.tr("警告"),
                content=self.tr(f"水印文字超过 {self.max_length} 字符!"),
                isClosable=True,
                tailPosition=TeachingTipTailPosition.BOTTOM,
                duration=2000,
                parent=self
            )
            return
        self.input.setText(current + ch)
