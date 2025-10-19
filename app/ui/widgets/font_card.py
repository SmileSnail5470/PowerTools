from PySide6.QtCore import Qt, QEasingCurve
from PySide6.QtGui import QFontDatabase, QFont
from PySide6.QtWidgets import QVBoxLayout, QLabel, QWidget
from app.ui.library.qfluentwidgets import CardWidget, setFont, FlowLayout


common_fonts_zh = [
    "微软雅黑", "宋体", "黑体", "仿宋", "楷体", "苹方", "思源黑体", "思源宋体"
]
common_fonts_en = [
    "Arial", "Calibri", "Times New Roman", "Courier New", "Segoe UI", "Verdana", "Tahoma", "Helvetica"
]

class FontCard(CardWidget):
    def __init__(self, font_name: str, text: str, parent=None):
        super().__init__(parent)
        self.font_name = font_name

        self.setFixedHeight(60)
        self.setCursor(Qt.ArrowCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(6)

        self.preview = QLabel(text)
        font = QFont(font_name)
        font.setPointSize(11)
        self.preview.setFont(font)
        self.preview.setWordWrap(True)
        self.preview.setStyleSheet("color: rgba(0, 0, 0, 0.6);")
        layout.addWidget(self.preview)

        # 字体名称
        self.name_label = QLabel(font_name)
        setFont(self.name_label, 12)
        self.name_label.setStyleSheet("color: rgba(0, 0, 0, 0.6);")
        layout.addWidget(self.name_label)

        self.setStyleSheet("""
            CardWidget {
                border-radius: 2px;
                background-color: #f8f8f8;
                border: 1px solid rgba(0, 0, 0, 0.05);
            }
        """)
        self.setVisible(True)

    def update_font(self, font_name: str, text: str):
        self.font_name = font_name
        self.name_label.setText(font_name)
        font = QFont(font_name)
        font.setPointSize(10)
        self.preview.setFont(font)
        self.preview.setText(text)


class FontSelectorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.flow_layout = FlowLayout(self, needAni=True)  # 启用动画
        self.flow_layout.setAnimation(250, QEasingCurve.OutQuad)
        self.flow_layout.setContentsMargins(10, 10, 10, 10)
        self.flow_layout.setVerticalSpacing(10)
        self.flow_layout.setHorizontalSpacing(5)

        # 加载字体卡片
        self.load_fonts()


    def load_fonts(self):
        """加载系统字体"""
        font_db = QFontDatabase()
        families = font_db.families()
        for name in common_fonts_zh:
            if name in families:
                card = FontCard(name, "你好，世界")
                self.flow_layout.addWidget(card)
        for name in common_fonts_en:
            if name in families:
                card = FontCard(name, "Hello, world")
                self.flow_layout.addWidget(card)