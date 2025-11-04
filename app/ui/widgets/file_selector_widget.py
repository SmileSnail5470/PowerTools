import os
from PySide6.QtCore import Signal, Qt, QSize, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import QVBoxLayout, QLabel, QFileDialog, QHBoxLayout, QPushButton, QGraphicsDropShadowEffect, QFrame, QWidget, QSizePolicy
from app.ui.library.qfluentwidgets import setFont, SimpleCardWidget, BodyLabel, CaptionLabel, FluentIcon


class FileUploadWidget(SimpleCardWidget):
    file_selected = Signal(list)
    format_text_value = "支持 JPG, PNG, AVIF, MP4, AVI 格式"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        upload_icon = QLabel("📤")
        upload_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        setFont(upload_icon, 48)
        upload_icon.setStyleSheet("""
            QLabel {
                color: #667eea;
            }
        """)
        main_layout.addWidget(upload_icon)

        upload_text = QLabel("点击或拖拽文件到此处")
        upload_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        setFont(upload_text, 12)
        upload_text.setStyleSheet("""
            QLabel {
                color: #333;
            }
        """)
        main_layout.addWidget(upload_text)
        
        format_text = QLabel(self.format_text_value)
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
            FileUploadWidget {
                border: 2px dashed #667eea;
                background-color: white;
                border-radius: 12px;
            }
            FileUploadWidget:hover {
                background-color: #f0f4ff;
                border-color: #764ba2;
            }
        """)

    def mousePressEvent(self, event):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择文件",
            "", 
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif);;视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        if files:
            self.file_selected.emit(files)
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                FileUploadWidget {
                    border: 2px dashed #667eea;
                    background-color: #f0f4ff;
                    border-radius: 12px;
                }
            """)
    
    def dragLeaveEvent(self, event):
        self.setup_style()
    
    def dropEvent(self, event):
        self.setup_style()
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        if files:
            self.file_selected.emit(files)

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


class FileInfoWidget(SimpleCardWidget):
    def __init__(self, parent=None, image_path: str = ""):
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

        if not image_path:
            image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "images", "logo.png")

        image_file_name = os.path.basename(image_path)
        file_size = os.path.getsize(image_path)
        
        self._create_ui_components(image_file_name, file_size, image_path)
        
        self._setup_layout()
        
    def _create_ui_components(self, file_name: str, file_size: int, image_path: str):
        self.iconWidget = QLabel("🖼️")
        setFont(self.iconWidget, 30)
        
        self.titleLabel = BodyLabel(file_name, self)
        setFont(self.titleLabel, 12, QFont.DemiBold)
        self.titleLabel.setToolTip(image_path)
        self.titleLabel.setWordWrap(False)
        self.titleLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.titleLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.titleLabel.setStyleSheet("color: #323130;")
        
        content = self.human_readable_size(file_size)
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

    def human_readable_size(self, size_bytes: int) -> str:
        if size_bytes == 0:
            return "0 B"
        units = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(units) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.2f} {units[i]}"
    

class FileSelectorWidget(QWidget):
    item_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(10)

        self.fileSelector = FileUploadWidget(self)
        self.main_layout.addWidget(self.fileSelector)

        self.fileInfoWidget = None

        self.fileSelector.file_selected.connect(self.on_file_selected)

    def on_file_selected(self, files: list[str]):
        if not files:
            return
        file_path = files[0]

        if self.fileInfoWidget:
            self.fileInfoWidget.deleteLater()
            self.fileInfoWidget = None

        self.fileInfoWidget = FileInfoWidget(self, image_path=file_path)
        self.main_layout.addWidget(self.fileInfoWidget)

        self.default_height = self.fileSelector.height()

        self.fileInfoWidget.removeButton.clicked.connect(self.on_file_removed)

        self.animate_height_change(expand=True)

        self.item_selected.emit(file_path)

    def on_file_removed(self):
        if self.fileInfoWidget:
            self.fileInfoWidget.deleteLater()
            self.fileInfoWidget = None
        self.animate_height_change(expand=False)
        self.item_selected.emit("")

    def animate_height_change(self, expand: bool):
        start_height = self.height()
        target_height = (self.default_height + 63 + 10 if expand else self.default_height)

        animation = QPropertyAnimation(self, b"maximumHeight", self)
        animation.setDuration(200)
        animation.setStartValue(start_height)
        animation.setEndValue(target_height)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.start()