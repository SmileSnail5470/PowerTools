import os
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QVBoxLayout, QLabel, QFileDialog
from app.ui.library.qfluentwidgets import setFont, SimpleCardWidget


class DirectorySelectorWidget(SimpleCardWidget):
    directory_selected = Signal(list)

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
            DirectorySelectorWidget {
                border: 2px dashed #667eea;
                background-color: white;
                border-radius: 12px;
            }
            DirectorySelectorWidget:hover {
                background-color: #f0f4ff;
                border-color: #764ba2;
            }
        """)

    def mousePressEvent(self, event):
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择文件夹",
            "",
            QFileDialog.Option.ShowDirsOnly
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
                DirectorySelectorWidget {
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