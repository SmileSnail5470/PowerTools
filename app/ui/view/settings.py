import os
import platform
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, Property, QSize, QThreadPool, QRunnable, QObject, QThread
from PySide6.QtWidgets import(
    QHBoxLayout, QWidget, QVBoxLayout, QLabel, QFrame, QLineEdit, QPushButton, QFileDialog, QGroupBox,
    QSizePolicy, QDialog, QProgressBar, QTextEdit
)
from PySide6.QtGui import QFont, QColor, QPainter, QPen

from app.ui.library.qfluentwidgets import(
    setFont, ScrollArea, TeachingTip, InfoBarIcon, TeachingTipTailPosition, FluentIcon, isDarkTheme,
    BodyLabel, CaptionLabel, ComboBox, Theme
)

from app.ui.widgets.gradient_header_widget import GradientHeader
from app.ui.common.config import cfg, Language


theme_map = {
    "浅色": Theme.LIGHT,
    "深色": Theme.DARK
}
language_map = {
    "简体中文": Language.CHINESE_SIMPLIFIED,
    "英语": Language.ENGLISH
}

class WorkerSignals(QObject):
    progress = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)


class InitWorker(QRunnable):
    def __init__(self, task_name: str, parent: QObject = None):
        super().__init__()
        self.task_name = task_name
        self.signals = WorkerSignals(parent=parent)
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def _step(self, msg):
        if self.cancelled:
            raise RuntimeError("初始化已被用户取消")
        self.signals.progress.emit(msg)
        QThread.msleep(800)

    def run(self):
        try:
            self._step("正在检查环境依赖…")
            self._step("正在加载模型文件…")
            self._step("正在初始化执行引擎…")

            res = False

            if res:
                self.signals.progress.emit("环境初始化成功")
                self.signals.finished.emit(True, "")
            else:
                raise RuntimeError("模型文件加载失败：缺失 core.bin")

        except Exception as e:
            self.signals.finished.emit(False, str(e))


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

    def setLabel(self, text: str, color: str):
        self.label.setText(text)
        self.dot.setStyleSheet(f"""
            background: {color};
            border-radius: 6px;
        """)


class InitProgressDialog(QDialog):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(660)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)

        self.bg_widget = QDialog()
        self.bg_widget.setStyleSheet("""
            QDialog {
                background-color: rgba(245, 246, 250, 0.95);
                border-radius: 14px;
            }
        """)
        bg_layout = QVBoxLayout(self.bg_widget)
        bg_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(self.bg_widget)

        header_layout = QHBoxLayout()
        title_label = QLabel(self.tr("🔧 环境初始化中，请稍候…"))
        setFont(title_label, 14, QFont.Bold)
        title_label.setStyleSheet("color: #1f2937;")
        header_layout.addWidget(title_label)

        self.close_btn = QPushButton("x")
        self.close_btn.setFixedSize(32, 32)
        setFont(self.close_btn, 14)
        self.close_btn.setStyleSheet("""
            QPushButton {
                border: none;
                background: transparent;
                color: #444;
                border-radius: 16px;
            }
            QPushButton:hover {
                color: rgba(232, 17, 35, 0.2);
                color: #e81123;
            }
            QPushButton:pressed {
                background-color: rgba(232, 17, 35, 0.4);
                color: white;
            }
        """)
        self.close_btn.clicked.connect(self.close)
        header_layout.addWidget(self.close_btn)
        header_layout.setAlignment(self.close_btn, Qt.AlignRight)
        bg_layout.addLayout(header_layout)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                height: 14px;
                border-radius: 7px;
                background: #e5e7eb;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #60a5fa, stop:1 #3b82f6
                );
                border-radius: 7px;
            }
        """)
        bg_layout.addWidget(self.progress)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        setFont(self.log_box, 12)
        self.log_box.setStyleSheet("""
            QTextEdit {
                background: #fefefe;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
                padding: 10px;
                color: #374151;
            }
        """)
        self.log_box.setMinimumHeight(180)
        bg_layout.addWidget(self.log_box)

    def append_log(self, text: str):
        self.log_box.append(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def enableCloseBtn(self):
        self.close_btn.show()

    def disableCloseBtn(self):
        self.close_btn.hide()


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
        self.path_input.textChanged.connect(self._update_global_config)
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
            self.status_label.setLabel(text=text, color=color)

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

    def _update_global_config(self, path: str):
        if self.name.lower() == "ffmpeg":
            cfg.ffmpeg_path.value = path

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
    thread_pool = QThreadPool.globalInstance()
    
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

        local_ai_settings = self._create_local_ai_settings()
        content_layout.addWidget(local_ai_settings)

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
        auto_start_switch.toggled.connect(lambda flag: setattr(cfg.autoStartup, "value", flag))
        auto_start_card = CustomCardGroupWidget(title=self.tr("开机自启动"), content=self.tr("系统启动时自动运行程序"), parent=self)
        auto_start_card.addWidget(auto_start_switch, stretch=0)
        auto_start_card.setSeparatorVisible(True)
        settings_cards.append(auto_start_card)

        auto_update_switch = ToggleSwitch()
        auto_update_switch.toggled.connect(lambda flag: setattr(cfg.autoUpdate, "value", flag))
        auto_update_card = CustomCardGroupWidget(title=self.tr("自动更新"), content=self.tr("自动检查并安装新版本"), parent=self)
        auto_update_card.addWidget(auto_update_switch, stretch=0)
        auto_update_card.setSeparatorVisible(True)
        settings_cards.append(auto_update_card)

        self.cache_line_edit = QLineEdit()
        self.cache_line_edit.textChanged.connect(lambda path: setattr(cfg.cachePath, "value", path))
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
        browse_btn.clicked.connect(lambda: self._select_path(self.cache_line_edit))
        cache_location_card = CustomCardGroupWidget(title=self.tr("缓存保存路径"), content=self.tr("设置缓存文件保存位置"), parent=self)
        cache_location_card.addWidget(self.cache_line_edit, stretch=1)
        cache_location_card.addWidget(browse_btn, stretch=0)
        cache_location_card.setSeparatorVisible(True)
        settings_cards.append(cache_location_card)

        theme_combox = ComboBox()
        theme_combox.currentTextChanged.connect(lambda text: setattr(cfg.uiTheme, "value", theme_map[text]))
        setFont(theme_combox, 14)
        theme_combox.addItems(["浅色"])
        theme_card = CustomCardGroupWidget(title=self.tr("界面主题"), content=self.tr("选择您喜欢的界面风格"), parent=self)
        theme_card.addWidget(theme_combox, stretch=0)
        theme_card.setSeparatorVisible(True)
        settings_cards.append(theme_card)

        language_combox = ComboBox()
        language_combox.currentTextChanged.connect(lambda text: setattr(cfg.language, "value", language_map[text]))
        setFont(language_combox, 14)
        language_combox.addItems(["简体中文"])
        language_card = CustomCardGroupWidget(title=self.tr("语言设置"), content=self.tr("选择界面显示语言"), parent=self)
        language_card.addWidget(language_combox, stretch=0)
        language_card.setSeparatorVisible(True)
        settings_cards.append(language_card)

        for card in settings_cards:
            settings.addCard(card=card)

        return settings
    
    def _create_local_ai_settings(self):
        ai_settings_cards = []
        ai_settings = CustomGroupBox(title=self.tr("🤖 本地AI设置"))

        localAIModelDeps_line_edit = QLineEdit()
        localAIModelDeps_line_edit.textChanged.connect(lambda path: setattr(cfg.localAIModelDeps, "value", path))
        localAIModelDeps_line_edit.setPlaceholderText(self.tr(f"请配置本地AI模型依赖保存路径"))
        setFont(localAIModelDeps_line_edit, 14)
        localAIModelDeps_line_edit.setStyleSheet("""
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
        browse_btn.clicked.connect(lambda: self._select_path(localAIModelDeps_line_edit))
        model_deps_location_card = CustomCardGroupWidget(title=self.tr("AI模型依赖路径"), content=self.tr("设置本地AI模型依赖文件保存位置"), parent=self)
        model_deps_location_card.addWidget(localAIModelDeps_line_edit, stretch=1)
        model_deps_location_card.addWidget(browse_btn, stretch=0)
        model_deps_location_card.setSeparatorVisible(True)
        ai_settings_cards.append(model_deps_location_card)
        
        blind_watermark_switch = ToggleSwitch()
        blind_watermark_switch.toggled.connect(lambda flag: setattr(cfg.localBlindWatermarkEnabled, "value", flag))
        blind_watermark_status = StatusBadge(text=self.tr("未启用"), color="#eab308")
        self._bind_ai_toggle(
            switch=blind_watermark_switch,
            badge=blind_watermark_status,
            config_attr="localBlindWatermarkEnabled"
        )
        blind_watermark_card = CustomCardGroupWidget(title=self.tr("盲水印AI能力"), content=self.tr("为图像添加不可见的数字水印，保护版权"), parent=self)
        blind_watermark_card.addWidget(blind_watermark_status, stretch=0)
        blind_watermark_card.addWidget(blind_watermark_switch, stretch=0)
        blind_watermark_card.setSeparatorVisible(True)
        ai_settings_cards.append(blind_watermark_card)

        watermark_removal_switch = ToggleSwitch()
        watermark_removal_switch.toggled.connect(lambda flag: setattr(cfg.localWatermarkRemovalEnabled, "value", flag))
        watermark_removal_status = StatusBadge(text=self.tr("未启用"), color="#eab308")
        self._bind_ai_toggle(
            switch=watermark_removal_switch,
            badge=watermark_removal_status,
            config_attr="localWatermarkRemovalEnabled"
        )
        watermark_removal_card = CustomCardGroupWidget(title=self.tr("水印去除AI能力"), content=self.tr("智能去除图像中的水印和标志"), parent=self)
        watermark_removal_card.addWidget(watermark_removal_status, stretch=0)
        watermark_removal_card.addWidget(watermark_removal_switch, stretch=0)
        watermark_removal_card.setSeparatorVisible(True)
        ai_settings_cards.append(watermark_removal_card)

        for card in ai_settings_cards:
            ai_settings.addCard(card=card)

        return ai_settings
    
    def _create_performance_settings(self):
        performance_settings_cards = []
        performance_settings = CustomGroupBox(title=self.tr("🌟 高级设置"))

        log_level_combox = ComboBox()
        log_level_combox.currentTextChanged.connect(lambda text: setattr(cfg.logLevel, "value", text.upper()))
        setFont(log_level_combox, 14)
        log_level_combox.addItems(["error", "warning", "info", "debug"])
        log_level_card = CustomCardGroupWidget(title=self.tr("日志级别"), content=self.tr("设置日志记录详细程度"), parent=self)
        log_level_card.addWidget(log_level_combox, stretch=0)
        log_level_card.setSeparatorVisible(True)
        performance_settings_cards.append(log_level_card)

        for card in performance_settings_cards:
            performance_settings.addCard(card=card)

        return performance_settings
    
    def _select_path(self, widget):
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择文件夹",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            widget.setText(directory)

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
    
    def _bind_ai_toggle(self, switch: ToggleSwitch, badge: StatusBadge, config_attr):
        switch.toggled.connect(
            lambda flag, switch=switch, badge=badge, config_attr=config_attr: 
            self._ai_switch_on_toggle(flag=flag, switch=switch, badge=badge, config_attr=config_attr)
        )

    def _ai_switch_on_toggle(self, flag: bool, switch: ToggleSwitch, badge: StatusBadge, config_attr: str):
        if flag:
            badge.setLabel(text=self.tr("环境初始化中…"), color="#60a5fa")

            progress_dialog = InitProgressDialog(title=self.tr("正在初始化环境..."), parent=self)
            progress_dialog.disableCloseBtn()
            progress_dialog.show()

            worker = InitWorker(task_name=config_attr, parent=progress_dialog)
            worker.signals.progress.connect(progress_dialog.append_log)
            worker.signals.finished.connect(
                lambda ok, msg, switch=switch, badge=badge, progress_dialog=progress_dialog: 
                self._on_init_finished(ok, msg, switch, badge, progress_dialog)
            )
            self.thread_pool.start(worker)
        else:
            badge.setLabel(text=self.tr("未启用"), color="#eab308")

    def _on_init_finished(
            self, ok: bool,
            error: str,
            switch: ToggleSwitch, 
            badge: StatusBadge, 
            progress_dialog: InitProgressDialog
        ):
        if ok:
            badge.setLabel(text=self.tr("已启用"), color="#22c55e")
            progress_dialog.accept()
        else:
            switch.setActive(False)
            badge.setLabel(text=self.tr("启用失败"), color="#ef4444")
            progress_dialog.enableCloseBtn()
            progress_dialog.append_log(f"\n❌ 错误信息：{error}")
            progress_dialog.progress.setRange(0, 1)

