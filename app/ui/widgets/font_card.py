from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase, QFont
from PySide6.QtWidgets import QVBoxLayout, QLabel
from app.ui.library.qfluentwidgets import CardWidget, setFont


FONT_ALIAS_MAP = {
    "微软雅黑": ["Microsoft YaHei"],
    "宋体": ["SimSun"],
    "黑体": ["SimHei"],
    "仿宋": ["FangSong"],
    "楷体": ["KaiTi"],
    "苹方": ["PingFang SC", "PingFang"],
    "思源黑体": ["Source Han Sans CN", "Source Han Sans", "Noto Sans CJK SC"],
    "思源宋体": ["Source Han Serif CN", "Source Han Serif", "Noto Serif CJK SC"],

    # 英文字体
    "Arial": ["Arial"],
    "Calibri": ["Calibri"],
    "Times New Roman": ["Times New Roman"],
    "Courier New": ["Courier New"],
    "Segoe UI": ["Segoe UI"],
    "Verdana": ["Verdana"],
    "Tahoma": ["Tahoma"],
    "Helvetica": ["Helvetica"]
}

def get_available_fonts():
    families = list(QFontDatabase.families())
    lower_families = [f.lower() for f in families]

    fonts_zh = {}
    fonts_en = {}

    for display_name, aliases in FONT_ALIAS_MAP.items():
        for alias in aliases:
            if alias.lower() in lower_families:
                real_name = families[lower_families.index(alias.lower())]
                entry = {"display": display_name, "system": real_name}
                if display_name in [
                    "微软雅黑", "宋体", "黑体", "仿宋", "楷体", "苹方", "思源黑体", "思源宋体"
                ]:
                    fonts_zh[display_name] = real_name
                else:
                    fonts_en[display_name] = real_name
                break
    return fonts_zh, fonts_en

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
        font.setPointSize(12)
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