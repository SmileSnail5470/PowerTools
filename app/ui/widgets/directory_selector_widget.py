import os
from PySide6.QtCore import Signal, Qt, QSize, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import QVBoxLayout, QLabel, QFileDialog, QPushButton, QGraphicsDropShadowEffect, QFrame, QWidget, QHBoxLayout, QSizePolicy
from app.ui.library.qfluentwidgets import setFont, SimpleCardWidget, BodyLabel, CaptionLabel, FluentIcon


class DirectoryUploadWidget(SimpleCardWidget):
    directory_selected = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        upload_icon = QLabel("📁")
        upload_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        setFont(upload_icon, 48)
        upload_icon.setStyleSheet("""
            QLabel {
                color: #667eea;
            }
        """)
        main_layout.addWidget(upload_icon)

        upload_text = QLabel("点击或拖拽文件夹到此处")
        upload_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        setFont(upload_text, 12)
        upload_text.setStyleSheet("""
            QLabel {
                color: #333;
            }
        """)
        main_layout.addWidget(upload_text)
        
        format_text = QLabel("支持单个或多个文件夹")
        format_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        setFont(format_text, 10)
        format_text.setStyleSheet("""
            QLabel {
                color: #999;
            }
        """)
        main_layout.addSpacing(5)
        main_layout.addWidget(format_text)

        self.setup_style()

    def setup_style(self):
        self.setFixedHeight(140)
        self.setStyleSheet("""
            DirectoryUploadWidget {
                border: 2px dashed #667eea;
                background-color: white;
                border-radius: 12px;
            }
            DirectoryUploadWidget:hover {
                background-color: #f0f4ff;
                border-color: #764ba2;
            }
        """)

    def mousePressEvent(self, event):
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择文件夹",
            "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog
        )
        if directory:
            self.directory_selected.emit([directory])
    
    def dragEnterEvent(self, event):
        if not event.mimeData().hasUrls():
            return
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            if not os.path.isdir(path):
                continue
            event.acceptProposedAction()
            self.setStyleSheet("""
                DirectoryUploadWidget {
                    border: 2px dashed #667eea;
                    background-color: #f0f4ff;
                    border-radius: 12px;
                }
            """)
    
    def dragLeaveEvent(self, event):
        self.setup_style()
    
    def dropEvent(self, event):
        self.setup_style()
        dirs = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            import os
            if os.path.isdir(path):
                dirs.append(path)
        if dirs:
            self.directory_selected.emit(dirs)


class DeleteButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(32, 32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setText("")

        self._setup_style()
        
        self._setup_shadow()
        
        self.setIcon(FluentIcon.DELETE.icon())
        self.setIconSize(QSize(16, 16))
        
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

class DirectoryInfoWidget(SimpleCardWidget):
    def __init__(self, parent=None, dir_path: str = ""):
        super().__init__(parent)
        self.setFixedHeight(63)
        self.setStyleSheet("""
            FileSelectorWidget {
                border: 2px solid rgba(0, 0, 0, 0.08);
                background-color: #f9f9f9;
                border-radius: 12px;
            }
            FileSelectorWidget:hover {
                background-color: #f5f5f5;
                border-color: rgba(0, 0, 0, 0.12);
            }
        """)

        if not dir_path:
            dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "images")
        
        self._create_ui_components(dir_path)
        
        self._setup_layout()
        
    def _create_ui_components(self, dir_path: str):
        self.iconWidget = QLabel("📁")
        setFont(self.iconWidget, 30)
        
        base_dir = os.path.basename(dir_path)
        if os.name == "nt":
            base_dir = "...\\{0}".format(base_dir if len(base_dir) <= 16 else "..."+base_dir[-16:])
        else:
            base_dir = ".../{0}".format(base_dir if len(base_dir) <= 16 else "..."+base_dir[-16:])
        self.titleLabel = BodyLabel(base_dir, self)
        setFont(self.titleLabel, 12, QFont.DemiBold)
        self.titleLabel.setToolTip(dir_path)
        self.titleLabel.setWordWrap(False)
        self.titleLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.titleLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.titleLabel.setStyleSheet("color: #323130;")
        
        content = "文件个数：{0}".format(len(os.listdir(dir_path)))
        self.contentLabel = CaptionLabel(content, self)
        setFont(self.contentLabel, 10)
        self.contentLabel.setTextColor("#606060", "#d2d2d2")
        
        self.removeButton = DeleteButton(self)
        
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.Shape.VLine)
        self.separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.separator.setStyleSheet("""
            QFrame {
                color: rgba(0, 0, 0, 0.08);
                margin: 0px 10px;
            }
        """)
        
    def _setup_layout(self):
        self.hBoxLayout = QHBoxLayout(self)
        self.vBoxLayout = QVBoxLayout()

        # 水平布局设置
        self.hBoxLayout.setContentsMargins(20, 11, 11, 11)
        self.hBoxLayout.setSpacing(15)
        self.hBoxLayout.addWidget(self.iconWidget)

        # 垂直布局设置
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.setSpacing(2)
        self.vBoxLayout.addWidget(self.titleLabel, 0, Qt.AlignVCenter)
        self.vBoxLayout.addWidget(self.contentLabel, 0, Qt.AlignVCenter)
        self.vBoxLayout.setAlignment(Qt.AlignVCenter)
        self.hBoxLayout.addLayout(self.vBoxLayout)

        # 添加弹性空间和分隔线
        self.hBoxLayout.addStretch(1)
        self.hBoxLayout.addWidget(self.separator)
        self.hBoxLayout.addWidget(self.removeButton, 0, Qt.AlignVCenter)
    

class DirectorySelectorWidget(QWidget):
    item_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(10)

        self.dirSelector = DirectoryUploadWidget(self)
        self.main_layout.addWidget(self.dirSelector)

        self.dirInfoWidget = None

        self.dirSelector.directory_selected.connect(self.on_dir_selected)

    def on_dir_selected(self, dirs: list[str]):
        if not dirs:
            return
        dir_path = dirs[0]

        if self.dirInfoWidget:
            self.dirInfoWidget.deleteLater()
            self.dirInfoWidget = None

        self.dirInfoWidget = DirectoryInfoWidget(self, dir_path=dir_path)
        self.main_layout.addWidget(self.dirInfoWidget)

        self.default_height = self.dirSelector.height()

        self.add_height = self.dirInfoWidget.height()

        self.dirInfoWidget.removeButton.clicked.connect(self.on_dir_removed)

        self.animate_height_change(expand=True)

        self.item_selected.emit(dir_path)

    def on_dir_removed(self):
        if self.dirInfoWidget:
            self.dirInfoWidget.deleteLater()
            self.dirInfoWidget = None
        self.animate_height_change(expand=False)
        self.item_selected.emit("")

    def animate_height_change(self, expand: bool):
        start_height = self.height()
        target_height = (self.default_height + self.add_height + 10 if expand else self.default_height)

        animation = QPropertyAnimation(self, b"maximumHeight", self)
        animation.setDuration(200)
        animation.setStartValue(start_height)
        animation.setEndValue(target_height)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.start()