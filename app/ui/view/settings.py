from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QWidget, QVBoxLayout, QLabel, QFrame, QLineEdit, QPushButton, QFileDialog, QGroupBox
from PySide6.QtGui import QFont, QIcon

from app.ui.library.qfluentwidgets import setFont, ScrollArea

from app.ui.widgets.gradient_header_widget import GradientHeader


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
        icon_label.setFixedSize(48, 48)
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
        name_label = QLabel(self.name)
        setFont(name_label, 15, QFont.Bold)
        name_label.setStyleSheet("color: #1f2937;")
        desc_label = QLabel(description)
        setFont(desc_label, 13, QFont.Bold)
        desc_label.setStyleSheet("color: #6b7280;")
        name_info.addWidget(name_label)
        name_info.addWidget(desc_label)
        name_layout.addWidget(icon_label)
        name_layout.addLayout(name_info)

        self.status_label = QLabel()
        self.status_label.setContentsMargins(8, 4, 8, 4)
        self._update_status()

        header_layout = QHBoxLayout()
        header_layout.addLayout(name_layout)
        header_layout.addStretch()
        header_layout.addWidget(self.status_label)

        path_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText(self.tr(f"{self.name} 可执行文件路径"))
        setFont(self.path_input, 14)
        self.path_input.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                background: white;
            }
            QLineEdit:focus { border: 1px solid #4f46e5; }
        """)

        browse_btn = QPushButton()
        browse_btn.setIcon(QIcon.fromTheme("document-open"))
        browse_btn.setStyleSheet(self._btn_style("#f3f4f6", "#e5e7eb"))
        setFont(browse_btn, 12, QFont.Bold)
        browse_btn.clicked.connect(lambda: self._select_path())

        test_btn = QPushButton("测试")
        test_btn.setStyleSheet(self._btn_style("#4f46e5", "#4338ca", color="white"))
        setFont(test_btn, 10, QFont.Bold)
        test_btn.clicked.connect(self._test_ok)

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
        """

    def _update_status(self):
        if self.status == "connected":
            self.status_label.setText("已连接")
            self.status_label.setStyleSheet("""
                QLabel { background: #dcfce7; color: #16a34a; border-radius: 20px;
                        font-size: 12px; font-weight: 500; padding: 4px 12px; }
            """)
        else:
            self.status_label.setText("未连接")
            self.status_label.setStyleSheet("""
                QLabel { background: #fee2e2; color: #dc2626; border-radius: 20px;
                        font-size: 12px; font-weight: 500; padding: 4px 12px; }
            """)

    def _test_ok(self):
        self.status_label.setText("测试中...")
        setFont(self.status_label, 12, QFont.DemiBold)
        self.status_label.setStyleSheet("""
            QLabel { 
                background: #fef3c7; 
                color: #d97706; 
                border-radius: 20px;
                padding: 4px 12px; 
            }""")
        pass

    def _select_path(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择文件夹",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.path_input.setText(directory)

class Settings(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("Settings")
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
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
        
        software_settings = self._create_software_settings()
        content_layout.addWidget(software_settings)
        
        scroll.setWidget(content)

        scroll.setWidgetResizable(True)
        scroll.enableTransparentBackground()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        main_layout.addWidget(scroll)

    def _create_software_settings(self):
        group = QGroupBox(self.tr("🔌 软件配置"))
        group.setStyleSheet(self._group_style())
        setFont(group, 18, QFont.Bold)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        self.software_cards = []
        ffmpeg_card = SoftwareCard("FFmpeg", {"gradient": ["#667eea","#764ba2"], "symbol":"🎬"}, "视频处理引擎", "ok")
        ffmpeg_card.path_input.setPlaceholderText(self.tr("请配置 ffmpeg 软件包所在目录路径"))
        paddleOCR_card = SoftwareCard("PaddleOCR", {"gradient": ["#f093fb","#f5576c"], "symbol":"🔤"}, "文字识别引擎", "failed")

        self.software_cards.extend([ffmpeg_card, paddleOCR_card])

        for card in self.software_cards:
            layout.addWidget(card)

        return group
    
    def _group_style(self):
        return """
            QGroupBox { background:white; border:none; border-radius:16px; padding-top:24px; color:#1a1a1a; }
            QGroupBox::title { subcontrol-origin:margin; left:24px; padding:0 10px 0 10px; }
        """