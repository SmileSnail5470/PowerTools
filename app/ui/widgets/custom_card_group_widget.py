from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFrame, QLabel
from PySide6.QtGui import QPainter, QColor, QFont
from app.ui.library.qfluentwidgets import isDarkTheme, BodyLabel, CaptionLabel, setFont


class CardSeparator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedHeight(3)

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)

        if isDarkTheme():
            painter.setPen(QColor(255, 255, 255, 46))
        else:
            painter.setPen(QColor(0, 0, 0, 12))

        painter.drawLine(2, 1, self.width() - 2, 1)


class CustomCardGroupWidget(QWidget):
    def __init__(self, title: str, content: str, parent=None, text_layout_contents_margins=(0, 0, 0, 0), label_v_space=0):
        super().__init__(parent=parent)
        self.text_layout_contents_margins = text_layout_contents_margins
        self.label_v_space = label_v_space
        self.vBoxLayout = QVBoxLayout(self)
        self.hBoxLayout = QHBoxLayout()

        self.titleLabel = BodyLabel(title)
        self.contentLabel = CaptionLabel(content)
        self.textLayout = QVBoxLayout()

        self.separator = CardSeparator()

        self.__initWidget()

    def __initWidget(self):
        self.separator.hide()
        self.contentLabel.setTextColor(QColor(96, 96, 96), QColor(206, 206, 206))

        self.vBoxLayout.setSpacing(0)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.addLayout(self.hBoxLayout)
        self.vBoxLayout.addWidget(self.separator)

        self.textLayout.addWidget(self.titleLabel)
        self.textLayout.addWidget(self.contentLabel)
        self.hBoxLayout.addLayout(self.textLayout)
        self.hBoxLayout.addStretch(1)

        self.hBoxLayout.setSpacing(15)
        self.hBoxLayout.setContentsMargins(0, 10, 24, 10)
        left, top, right, bottom = self.text_layout_contents_margins
        self.textLayout.setContentsMargins(left, top, right, bottom)
        self.textLayout.setSpacing(self.label_v_space)
        self.hBoxLayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.textLayout.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def title(self):
        return self.titleLabel.text()

    def setTitle(self, text: str):
        self.titleLabel.setText(text)

    def content(self):
        return self.contentLabel.text()

    def setContent(self, text: str):
        self.contentLabel.setText(text)

    def setSeparatorVisible(self, isVisible: bool):
        self.separator.setVisible(isVisible)

    def isSeparatorVisible(self):
        return self.separator.isVisible()

    def addWidget(self, widget: QWidget, stretch=0):
        self.hBoxLayout.addWidget(widget, stretch=stretch)


class CustomGroupBox(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent=parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.group = QGroupBox(self.tr(title))
        setFont(self.group, 18, QFont.Bold)

        self.group.setStyleSheet("""
            QGroupBox { 
                background: white; 
                border: none; 
                border-radius: 16px; 
                padding-top: 16px;
                color:#1a1a1a;
            }
            QGroupBox::title {
                subcontrol-origin: padding;
                padding: 0 10px;
                left: 16px;
            }
        """)

        self.main_layout = QVBoxLayout(self.group)
        self.main_layout.setContentsMargins(24, 24, 24, 24)
        self.main_layout.setSpacing(12)

        outer.addWidget(self.group)

    def addCard(self, card: QWidget, stretch: int = 0):
        self.main_layout.addWidget(card, stretch)


class StyleCard(QFrame):
    def __init__(self, icon_color, title, subtitle, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            StyleCard {
                background-color: white;
                border: 2px solid transparent;
                border-radius: 12px;
                margin: 2px;
            }
            StyleCard:hover {
                background-color: #f8f9fa;
            }
        """)
        
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 8, 12, 8)
        main_layout.setSpacing(12)
        
        self.icon_label = QLabel(self)
        self.icon_label.setFixedSize(40, 40)
        if icon_color == "#84fab0":
            self.icon_label.setText("🍃")
        elif icon_color == "#fa709a":
            self.icon_label.setText("✏️")
        else:
            self.icon_label.setText("🪣")
            
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setStyleSheet(f"""
            QLabel {{
                background-color: {icon_color};
                border-radius: 8px;
                font-size: 20px;
            }}
        """)
        
        text_layout = QVBoxLayout()
        text_layout.setSpacing(0)
        text_layout.setContentsMargins(0, 0, 0, 0)
        
        self.title_label = QLabel(title)
        setFont(self.title_label, 14, QFont.Bold)
        self.title_label.setStyleSheet("color: #2c3e50;")
        
        self.subtitle_label = QLabel(subtitle)
        setFont(self.subtitle_label, 12)
        self.subtitle_label.setStyleSheet("color: #7f8c8d;")
        self.subtitle_label.setWordWrap(True)
        
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.subtitle_label)
        
        main_layout.addWidget(self.icon_label)
        main_layout.addLayout(text_layout)
        
        self.is_selected = False
        
    def set_selected(self, selected):
        self.is_selected = selected
        if selected:
            self.setStyleSheet("""
                StyleCard {
                    background-color: white;
                    border: 2px solid #667eea;
                    border-radius: 12px;
                    margin: 2px;
                }
                StyleCard:hover {
                    background-color: #f8f9fa;
                }
            """)
        else:
            self.setStyleSheet("""
                StyleCard {
                    background-color: white;
                    border: 2px solid transparent;
                    border-radius: 12px;
                    margin: 2px;
                }
                StyleCard:hover {
                    background-color: #f8f9fa;
                }
            """)