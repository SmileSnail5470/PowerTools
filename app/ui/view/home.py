from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QFrame

from app.ui.library.qfluentwidgets import SubtitleLabel, setFont


class Home(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        text = "Home"
        self.label = SubtitleLabel(text, self)
        self.hBoxLayout = QHBoxLayout(self)

        setFont(self.label, 24)
        self.label.setAlignment(Qt.AlignCenter)
        self.hBoxLayout.addWidget(self.label, 1, Qt.AlignCenter)
        self.setObjectName(text.replace(' ', '-'))