import os
import platform
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, Property, QSize
from PySide6.QtWidgets import(
    QHBoxLayout, QWidget, QVBoxLayout, QLabel, QFrame, QLineEdit, QPushButton, QFileDialog, QGroupBox, 
    QSizePolicy
)
from PySide6.QtGui import QFont, QColor, QPainter, QPen

from app.ui.library.qfluentwidgets import(
    setFont, ScrollArea, TeachingTip, InfoBarIcon, TeachingTipTailPosition, FluentIcon, isDarkTheme,
    BodyLabel, CaptionLabel, ComboBox
)

from app.ui.widgets.gradient_header_widget import GradientHeader


class StatusBadge(QWidget):
    def __init__(self, text: str, color: str, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.dot = QWidget()
        self.dot.setFixedSize(12, 12)
        self.dot.setStyleSheet(f"""
            background: {color};
            border-radius: 6px;
        """)

        self.label = QLabel(text)
        self.label.setStyleSheet("color: #374151; padding: 0; margin: 0;")
        setFont(self.label, 11)

        layout.addWidget(self.dot)
        layout.addWidget(self.label)

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

    def setLable(self, text: str, color: str):
        self.label.setText(text)
        self.dot.setStyleSheet(f"""
            background: {color};
            border-radius: 6px;
        """)


class SoftwareCard(QFrame):
    def __init__(self, name: str, icon: dict, description: str, status: str, parent=None):
        super().__init__(parent)
        self.setObjectName("softwareCard")
        self.name = name
        self.status = status
        self._setup_ui(icon, description)

    def _setup_ui(self, icon: dict, description: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        name_layout = QHBoxLayout()
        icon_label = QLabel(icon['symbol'])
        icon_label.setFixedSize(40, 40)
        icon_label.setAlignment(Qt.AlignCenter)
        setFont(icon_label, 20)
        icon_label.setStyleSheet(f"""
            QLabel {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 {icon['gradient'][0]}, stop:1 {icon['gradient'][1]});
                border-radius: 10px;
                color: white;
            }}
        """)
        name_info = QVBoxLayout()
        name_info.setSpacing(2)
        name_label = QLabel(self.name)
        setFont(name_label, 16, QFont.Bold)
        name_label.setStyleSheet("color: #1f2937;")
        desc_label = QLabel(description)
        setFont(desc_label, 12, QFont.Bold)
        desc_label.setStyleSheet("color: #6b7280;")
        name_info.addWidget(name_label)
        name_info.addWidget(desc_label)
        name_layout.addWidget(icon_label)
        name_layout.addLayout(name_info)

        self.status_label = self._build_status_badge()

        header_layout = QHBoxLayout()
        header_layout.addLayout(name_layout)
        header_layout.addStretch()
        header_layout.addWidget(self.status_label, alignment=Qt.AlignVCenter)

        path_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText(self.tr(f"请配置 {self.name} 软件路径"))
        setFont(self.path_input, 14)
        self.path_input.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                background: white;
                color: #333;
            }
            QLineEdit:focus { border: 1px solid #4f46e5; }
        """)

        browse_btn = QPushButton()
        browse_btn.setIcon(FluentIcon.FOLDER_ADD.qicon())
        browse_btn.setStyleSheet(self._btn_style(bg="#f3f4f6", hover="#d1d5db"))
        setFont(browse_btn, 12, QFont.Bold)
        browse_btn.clicked.connect(lambda: self._select_path(select_file=False))

        test_btn = QPushButton(self.tr("验证"))
        test_btn.setStyleSheet(self._btn_style(bg="#4f46e5", hover="#4338ca", color="white"))
        setFont(test_btn, 12, QFont.Bold)
        test_btn.clicked.connect(self._check_software)

        path_layout.addWidget(self.path_input)
        path_layout.addWidget(browse_btn)
        path_layout.addWidget(test_btn)

        layout.addLayout(header_layout)
        layout.addLayout(path_layout)

        self.setStyleSheet("""
            QFrame#softwareCard {
                background: #fafafa;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
            }
            QFrame#softwareCard:hover {
                background: #f9fafb;
                border: 1px solid #4f46e5;
            }
        """)

    def _build_status_badge(self):
        if self.status == "ok":
            return StatusBadge("OK", self._get_status_badge_color())
        elif self.status == "failed":
            return StatusBadge("Failed", self._get_status_badge_color())
        else:
            return StatusBadge("未验证", self._get_status_badge_color())
        
    def _get_status_badge_color(self):
        if self.status == "ok":
            return "#16a34a"
        elif self.status == "failed":
            return "#dc2626"
        else:
            return "#eab308"

    def _btn_style(self, bg, hover, color="#374151"):
        return f"""
            QPushButton {{
                padding: 8px 16px;
                background: {bg};
                color: {color};
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background: {hover};
            }}
            QPushButton:pressed {{
                background: {hover};
                padding: 9px 16px;
                margin-top: 1px;
            }}
        """

    def _check_software(self):
        error_msg = ""
        if self.name.lower() == "ffmpeg":
            ffmpeg_exe = os.path.join(self.path_input.text(), "ffmpeg" if platform.system().lower() != "windows" else "ffmpeg.exe")
            ffprobe_exe = os.path.join(self.path_input.text(), "ffprobe.exe" if platform.system().lower() == "windows" else "ffprobe")
            files_missing_list = [p for p in [ffmpeg_exe, ffprobe_exe] if not os.path.exists(p)]
            if files_missing_list:
                self.status = "failed"
                text = "Failed"
                color = self._get_status_badge_color()
                error_msg = "\n".join(f"- 文件 {p} 不存在" for p in files_missing_list)
            else:
                self.status = "ok"
                text = "OK"
                color = self._get_status_badge_color()
            self.status_label.setLable(text=text, color=color)

        if error_msg:
            TeachingTip.create(
                target=self.status_label,
                icon=InfoBarIcon.ERROR,
                title=self.tr("警告"),
                content=self.tr(error_msg),
                isClosable=True,
                tailPosition=TeachingTipTailPosition.BOTTOM,
                duration=2000,
                parent=self
            )

    def _select_path(self, select_file=True):
        if not select_file:
            directory = QFileDialog.getExistingDirectory(
                self,
                "选择文件夹",
                "",
                QFileDialog.Option.ShowDirsOnly
            )
            if directory:
                self.path_input.setText(directory)
        else:
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "选择文件",
                "", 
                "所有文件 (*)"
            )
            if files:
                self.path_input.setText(files)

class ToggleSwitch(QWidget):
    toggled = Signal(bool)

    def __init__(self, parent=None, width=44, height=24):
        super().__init__(parent)
        self._active = False
        self._anim_pos = 0.0  # 初始在左边
        self._animation = QPropertyAnimation(self, b"animPos", self)
        self._animation.setDuration(150)
        self._animation.setEasingCurve(QEasingCurve.InOutQuad)

        self._on_color = QColor("#4f46e5")
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
    def __init__(self, title: str, content: str, parent=None):
        super().__init__(parent=parent)
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
        self.textLayout.setContentsMargins(0, 0, 0, 0)
        self.textLayout.setSpacing(0)
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

class Settings(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("Settings")
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(12)
        
        self._setup_header(main_layout)
        
        self._setup_content(main_layout)
    
    def _setup_header(self, main_layout: QVBoxLayout):
        header = GradientHeader(parent=self)
        
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 20, 30, 20)
        header_layout.setSpacing(10)
        
        title_label = QLabel(self.tr("⚙️ 常规设置"))
        setFont(title_label, fontSize=24, weight=QFont.Bold)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
            }
        """)
        header_layout.addWidget(title_label)  
        header_layout.addStretch()
        
        main_layout.addWidget(header)
    
    def _setup_content(self, main_layout: QVBoxLayout):
        scroll = ScrollArea()
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)
        content_layout.setAlignment(Qt.AlignTop)

        general_settings = self._create_general_settings()
        content_layout.addWidget(general_settings)
        
        software_settings = self._create_software_settings()
        content_layout.addWidget(software_settings)

        performance_settings = self._create_performance_settings()
        content_layout.addWidget(performance_settings)
        
        scroll.setWidget(content)

        scroll.setWidgetResizable(True)
        scroll.enableTransparentBackground()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        main_layout.addWidget(scroll)

    def _create_software_settings(self):
        group = CustomGroupBox(title=self.tr("🔌 软件配置"))

        ffmpeg_card = SoftwareCard(
            name="FFmpeg", 
            icon={"gradient": ["#667eea","#764ba2"], "symbol":"🎬"}, 
            description=self.tr("视频处理引擎"), 
            status=""
        )

        self.software_cards = [ffmpeg_card]

        for card in self.software_cards:
            group.addCard(card=card)

        return group
    
    def _create_general_settings(self):
        settings_cards = []
        settings = CustomGroupBox(title=self.tr("⚙️ 通用设置"))
        
        auto_start_switch = ToggleSwitch()
        auto_start_card = CustomCardGroupWidget(title=self.tr("开机自启动"), content=self.tr("系统启动时自动运行程序"), parent=self)
        auto_start_card.addWidget(auto_start_switch, stretch=0)
        auto_start_card.setSeparatorVisible(True)
        settings_cards.append(auto_start_card)

        auto_update_switch = ToggleSwitch()
        auto_update_card = CustomCardGroupWidget(title=self.tr("自动更新"), content=self.tr("自动检查并安装新版本"), parent=self)
        auto_update_card.addWidget(auto_update_switch, stretch=0)
        auto_update_card.setSeparatorVisible(True)
        settings_cards.append(auto_update_card)

        self.cache_line_edit = QLineEdit()
        self.cache_line_edit.setPlaceholderText(self.tr(f"请配置软件缓存文件保存路径"))
        setFont(self.cache_line_edit, 14)
        self.cache_line_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                background: white;
                color: #333;
            }
            QLineEdit:focus { border: 1px solid #4f46e5; }
        """)
        browse_btn = QPushButton()
        browse_btn.setIcon(FluentIcon.FOLDER_ADD.qicon())
        browse_btn.setStyleSheet(self._btn_style(bg="#f3f4f6", hover="#d1d5db"))
        setFont(browse_btn, 12, QFont.Bold)
        browse_btn.clicked.connect(lambda: self._select_path())
        cache_location_card = CustomCardGroupWidget(title=self.tr("缓存保存路径"), content=self.tr("设置缓存文件保存位置"), parent=self)
        cache_location_card.addWidget(self.cache_line_edit, stretch=1)
        cache_location_card.addWidget(browse_btn, stretch=0)
        cache_location_card.setSeparatorVisible(True)
        settings_cards.append(cache_location_card)

        theme_combox = ComboBox()
        setFont(theme_combox, 14)
        theme_combox.addItems(["浅色"])
        theme_card = CustomCardGroupWidget(title=self.tr("界面主题"), content=self.tr("选择您喜欢的界面风格"), parent=self)
        theme_card.addWidget(theme_combox, stretch=0)
        theme_card.setSeparatorVisible(True)
        settings_cards.append(theme_card)

        language_combox = ComboBox()
        setFont(language_combox, 14)
        language_combox.addItems(["简体中文"])
        language_card = CustomCardGroupWidget(title=self.tr("语言设置"), content=self.tr("选择界面显示语言"), parent=self)
        language_card.addWidget(language_combox, stretch=0)
        language_card.setSeparatorVisible(True)
        settings_cards.append(language_card)

        for card in settings_cards:
            settings.addCard(card=card)

        return settings
    
    def _create_performance_settings(self):
        performance_settings_cards = []
        performance_settings = CustomGroupBox(title=self.tr("🌟 高级设置"))

        for card in performance_settings_cards:
            performance_settings.addCard(card=card)

        return performance_settings
    
    def _select_path(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择文件夹",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.cache_line_edit.setText(directory)

    def _btn_style(self, bg, hover, color="#374151"):
        return f"""
            QPushButton {{
                padding: 8px 16px;
                background: {bg};
                color: {color};
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background: {hover};
            }}
            QPushButton:pressed {{
                background: {hover};
                padding: 9px 16px;
                margin-top: 1px;
            }}
        """