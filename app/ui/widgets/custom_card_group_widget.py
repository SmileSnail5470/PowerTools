from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFrame, QLabel, QSizePolicy, QGraphicsOpacityEffect
from PySide6.QtGui import QPainter, QColor, QFont
from app.ui.library.qfluentwidgets import isDarkTheme, BodyLabel, CaptionLabel, setFont
from app.license.globals import license_manager, feature_gate
from app.ui.common.event_bus import global_event_bus


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
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.contentLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.separator.hide()
        self.contentLabel.setTextColor(QColor(96, 96, 96), QColor(206, 206, 206))

        self.vBoxLayout.setSizeConstraint(QVBoxLayout.SetMinimumSize)
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

    def setContentWordWrap(self, flag: bool):
        self.contentLabel.setWordWrap(flag)

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
    UpdateLicenseInfo = Signal()

    ICON_MAP = {
        "#4facfe": "✏️",   # PatchWiper：细节增强 / 手工修补感
        "#f093fb": "🧠",   # EMDF：智能修补 / 自适应
        "#a18cd1": "⚖️",   # GRIG：平衡修复
        "#84fab0": "🍃",   # LaMa：自然、保守、平滑
        "#fbc2eb": "🪣",   # CoordFill：快速填充
    }

    MODEL_FREE = "free"
    MODEL_PRO = "pro"

    def __init__(self, icon_color, title, subtitle, parent=None):
        super().__init__(parent)
        self.effect = QGraphicsOpacityEffect(self)
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
        icon = self.ICON_MAP.get(icon_color, "🧩")
        self.icon_label.setText(icon)
            
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

        status_row_layout = QHBoxLayout()
        status_row_layout.setSpacing(8)
        status_row_layout.setContentsMargins(0, 2, 0, 0)

        self.auth_label = QLabel(self)
        self.remaining_label = QLabel(self)

        status_row_layout.addWidget(self.auth_label)
        status_row_layout.addWidget(self.remaining_label)
        status_row_layout.addStretch(1)

        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.subtitle_label)
        text_layout.addLayout(status_row_layout)
        
        main_layout.addWidget(self.icon_label)
        main_layout.addLayout(text_layout)
        
        self.is_selected = False
        self.name = ""
        self._model_type = "pro"
        self._is_authorized = False
        self._remaining_uses = 0
        self._is_interactive = True
        global_event_bus.License_update.connect(self.update_license_info)
        self.UpdateLicenseInfo.connect(self.update_license_info)

    def update_license_info(self):
        if license_manager.is_licensed and license_manager.tier == "free":
            feature_name = None
            for one_feature in license_manager.license_data.features:
                feature_name = one_feature[0]
                if self.get_name() and self.get_name() not in feature_name:
                    continue
                self._model_type = one_feature[1]
                self._remaining_uses = one_feature[3]
                feature_name = feature_name
                break
            self._remaining_uses = feature_gate.get_remaining_uses(feature_name=feature_name)
        self.refresh_license_state()

    def refresh_license_state(self):
        self._update_license_state()
        if self.is_selected and not self._is_interactive:
            self.set_selected(False)

    def _update_license_state(self):
        if not license_manager.is_licensed:
            self._is_authorized = False
            self._remaining_uses = 0
            self._is_interactive = False
        elif license_manager.tier == "pro":
            self._is_authorized = True
            self._remaining_uses = -1
            self._is_interactive = True
        else:
            if self._model_type == self.MODEL_PRO:
                self._is_authorized = False
                self._remaining_uses = 0
                self._is_interactive = False
            else:
                self._is_interactive = self._remaining_uses != 0
                self._is_authorized = True

        self._update_auth_label()
        self._update_remaining_label()
        self._update_interactive_style()

    def _update_interactive_style(self):
        if not self._is_interactive:
            self.setStyleSheet("""
                StyleCard {
                    background-color: #f5f5f5;
                    border: 2px solid transparent;
                    border-radius: 12px;
                    margin: 2px;
                }
            """)
            self.effect.setOpacity(0.45)
            self.setGraphicsEffect(self.effect)
            self.setCursor(Qt.ForbiddenCursor)
        else:
            if self.is_selected:
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
            self.effect.setOpacity(1)
            self.setGraphicsEffect(self.effect)
            self.setCursor(Qt.PointingHandCursor)

    def _update_auth_label(self):
        if self._is_authorized:
            self.auth_label.setText("● 已授权")
            self.auth_label.setStyleSheet("""
                QLabel {
                    color: #27ae60;
                    font-size: 11px;
                }
            """)
        else:
            self.auth_label.setText("● 未授权")
            self.auth_label.setStyleSheet("""
                QLabel {
                    color: #e74c3c;
                    font-size: 11px;
                }
            """)

    def _update_remaining_label(self):
        if not self._is_authorized:
            self.remaining_label.hide()
            return
        
        self.remaining_label.show()
        if self._remaining_uses < 0:
            self.remaining_label.setText("剩余次数: 无限制")
            self.remaining_label.setStyleSheet("""
                QLabel {
                    color: #7f8c8d;
                    font-size: 11px;
                }
            """)
        elif self._remaining_uses == 0:
            self.remaining_label.setText("剩余: 0 次")
            self.remaining_label.setStyleSheet("""
                QLabel {
                    color: #e74c3c;
                    font-size: 11px;
                    font-weight: bold;
                }
            """)
        else:
            self.remaining_label.setText(f"剩余: {self._remaining_uses} 次")
            self.remaining_label.setStyleSheet("""
                QLabel {
                    color: #7f8c8d;
                    font-size: 11px;
                }
            """)

    def is_interactive(self) -> bool:
        return self._is_interactive

    def get_remaining_uses(self) -> int:
        return self._remaining_uses

    def set_name(self, name):
        self.name = name
        self.update_license_info()

    def get_name(self):
        return self.name

    def set_selected(self, selected):
        if selected and not self._is_interactive:
            return
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
            if self._is_interactive:
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
            else:
                self._update_interactive_style()