from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QVBoxLayout, QLabel, QFileDialog
from app.ui.library.qfluentwidgets import setFont, SimpleCardWidget


class FileSelectorWidget(SimpleCardWidget):
    file_selected = Signal(list)

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
        
        format_text = QLabel("支持 JPG, PNG, AVIF, MP4, AVI 格式")
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
            FileSelectorWidget {
                border: 2px dashed #667eea;
                background-color: white;
                border-radius: 12px;
            }
            FileSelectorWidget:hover {
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
                FileSelectorCard {
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