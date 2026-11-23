import os
import sys
from PySide6.QtCore import Signal, Qt, QSize, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import QVBoxLayout, QLabel, QFileDialog, QHBoxLayout, QPushButton, QGraphicsDropShadowEffect, QWidget, QSizePolicy
from app.ui.library.qfluentwidgets import setFont, SimpleCardWidget, BodyLabel, CaptionLabel, FluentIcon
from app.ui.widgets.media_split_dialog import MediaSplitDialog


class FileUploadWidget(SimpleCardWidget):
    file_selected = Signal(list)

    def __init__(self, parent=None, format_text_value: str = "支持 JPG, PNG, AVIF, MP4, AVI 格式"):
        super().__init__(parent)
        self.format_text_value = format_text_value
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
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.avif *.webp);;视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)",
            options=QFileDialog.Option.DontUseNativeDialog if sys.platform == "darwin" else QFileDialog.Option(0)
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

class SplitButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(32, 32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setIcon(FluentIcon.CUT.icon())
        self.setIconSize(QSize(16, 16))
        self.setText("")
        self.setToolTip("裁剪文件")
        self._active = False
        self._setup_shadow()
        self._refresh_style()

    def _setup_shadow(self):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(8)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 20))
        self.setGraphicsEffect(shadow)

    def set_active(self, active: bool):
        self._active = active
        self.setToolTip("已裁剪（点击重新设置）" if active else "裁剪文件")
        self._refresh_style()

    def _refresh_style(self):
        if self._active:
            self.setStyleSheet("""
                SplitButton {
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #667eea, stop:1 #764ba2);
                    border: none;
                    border-radius: 16px;
                    padding: 0px;
                }
                SplitButton:hover {
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #5a6fd6, stop:1 #6a3fa0);
                }
                SplitButton:pressed {
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #4f62c4, stop:1 #5e3690);
                }
            """)
        else:
            self.setStyleSheet("""
                SplitButton {
                    background-color: transparent;
                    border: none;
                    border-radius: 16px;
                    padding: 0px;
                }
                SplitButton:hover {
                    background-color: rgba(102, 126, 234, 0.1);
                }
                SplitButton:pressed {
                    background-color: rgba(102, 126, 234, 0.2);
                }
            """)


class FileInfoWidget(SimpleCardWidget):
    split_result_changed = Signal(object)

    def __init__(self, parent=None, image_path: str = ""):
        super().__init__(parent)
        self.setFixedHeight(63)
        self._image_path = image_path
        self._split_result = None

        if not image_path:
            image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "images", "logo.png")

        image_file_name = os.path.basename(image_path)
        file_size = os.path.getsize(image_path)

        self._create_ui_components(image_file_name, file_size, image_path)
        self._setup_layout()

    def paintEvent(self, event):
        pass

    @property
    def split_result(self):
        return self._split_result

    def _create_ui_components(self, file_name: str, file_size: int, image_path: str):
        ext = os.path.splitext(image_path)[-1].lower()
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
        icon_text = "🎬" if ext in video_exts else "🖼️"
        self.iconWidget = QLabel(icon_text)
        setFont(self.iconWidget, 30)

        if len(file_name) > 16:
            self.titleLabel = BodyLabel("..." + file_name[-16:], self)
        else:
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

        self.splitButton = SplitButton(self)
        self.splitButton.clicked.connect(self._on_split_clicked)

        self.removeButton = DeleteButton(self)

    def _setup_layout(self):
        self.hBoxLayout = QHBoxLayout(self)
        self.vBoxLayout = QVBoxLayout()

        self.hBoxLayout.setContentsMargins(20, 11, 11, 11)
        self.hBoxLayout.setSpacing(10)
        self.hBoxLayout.addWidget(self.iconWidget)

        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.setSpacing(2)
        self.vBoxLayout.addWidget(self.titleLabel, 0, Qt.AlignVCenter)
        self.vBoxLayout.addWidget(self.contentLabel, 0, Qt.AlignVCenter)
        self.vBoxLayout.setAlignment(Qt.AlignVCenter)
        self.hBoxLayout.addLayout(self.vBoxLayout)

        self.hBoxLayout.addStretch(1)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(4)
        btn_row.addWidget(self.splitButton)
        btn_row.addWidget(self.removeButton)
        self.hBoxLayout.addLayout(btn_row)

    def _on_split_clicked(self):
        dialog = MediaSplitDialog(self._image_path, parent=self.window())
        dialog.confirmed.connect(self._on_split_confirmed)
        dialog.exec()

    def _on_split_confirmed(self, result):
        self._split_result = result
        self.split_result_changed.emit(result)

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
    layout_add_height = Signal(int)
    split_result_changed = Signal(object)

    def __init__(self, parent=None, format_text_value: str = "支持 JPG, PNG, AVIF, MP4, AVI 格式"):
        super().__init__(parent)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(10)

        self.fileSelector = FileUploadWidget(self, format_text_value=format_text_value)
        self.main_layout.addWidget(self.fileSelector)

        self.fileInfoWidget: FileInfoWidget | None = None

        self.fileSelector.file_selected.connect(self.on_file_selected)

    @property
    def split_result(self):
        if self.fileInfoWidget:
            return self.fileInfoWidget.split_result
        return None

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
        self.fileInfoWidget.split_result_changed.connect(self.split_result_changed)

        self.layout_add_height.emit(self.fileInfoWidget.height() + 10)
        self.add_height = self.fileInfoWidget.height()

        self.animate_height_change(expand=True)

        self.item_selected.emit(file_path)

    def on_file_removed(self):
        if self.fileInfoWidget:
            self.fileInfoWidget.deleteLater()
            self.fileInfoWidget = None
            self.layout_add_height.emit(-(self.add_height + 10))
            self.split_result_changed.emit(None)
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