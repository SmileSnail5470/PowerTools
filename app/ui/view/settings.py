from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QWidget, QVBoxLayout, QLabel, QFrame, QLineEdit, QPushButton, QFileDialog, QGroupBox, QSizePolicy
from PySide6.QtGui import QFont, QIcon

from app.ui.library.qfluentwidgets import setFont, ScrollArea 

from app.ui.widgets.gradient_header_widget import GradientHeader


class StatusBadge(QWidget):
    def __init__(self, text: str, color: str, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        dot = QWidget()
        dot.setFixedSize(12, 12)
        dot.setStyleSheet(f"""
            background: {color};
            border-radius: 6px;
        """)

        label = QLabel(text)
        label.setStyleSheet("color: #374151; padding: 0; margin: 0;")
        setFont(label, 11)

        layout.addWidget(dot)
        layout.addWidget(label)

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)


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
            }
            QLineEdit:focus { border: 1px solid #4f46e5; }
        """)

        browse_btn = QPushButton()
        browse_btn.setIcon(QIcon.fromTheme("document-open"))
        browse_btn.setStyleSheet(self._btn_style(bg="#f3f4f6", hover="#d1d5db"))
        setFont(browse_btn, 12, QFont.Bold)
        browse_btn.clicked.connect(lambda: self._select_path(select_file=False))

        test_btn = QPushButton(self.tr("验证"))
        test_btn.setStyleSheet(self._btn_style(bg="#4f46e5", hover="#4338ca", color="white"))
        setFont(test_btn, 12, QFont.Bold)
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

    def _build_status_badge(self):
        if self.status == "ok":
            return StatusBadge("OK", "#16a34a")  # 绿色
        elif self.status == "failed":
            return StatusBadge("Failed", "#dc2626")  # 红色
        else:
            return StatusBadge("未验证", "#eab308")  # 黄色

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
        setFont(self.status_label, 11)
        if self.status == "ok":
            self.status_label.setText("OK")
            self.status_label.setStyleSheet("""
                QLabel { 
                    background: #dcfce7;
                    color: #16a34a; 
                    border-radius: 10px;
                    padding: 1px 6px; 
                    margin: 0px;
                }
            """)
        else:
            self.status_label.setText("Failed" if self.status=="failed" else self.tr("未验证"))
            self.status_label.setStyleSheet("""
                QLabel { 
                    background: #fee2e2;
                    color: #dc2626; 
                    border-radius: 10px;
                    padding: 1px 6px; 
                    margin: 0px;
                }
            """)

    def _test_ok(self):
        setFont(self.status_label, 12, QFont.DemiBold)
        self.status_label.setStyleSheet("""
            QLabel { 
                background: #fef3c7; 
                color: #d97706; 
                border-radius: 20px;
                padding: 4px 12px; 
            }""")
        pass

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
        group.setStyleSheet("""
            QGroupBox { 
                background: white; 
                border: none; 
                border-radius: 16px; 
                padding-top: 24px; 
                color:#1a1a1a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left:24px; 
                padding:0 10px 0 10px; 
            }
        """)
        setFont(group, 18, QFont.Bold)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        ffmpeg_card = SoftwareCard(
            name="FFmpeg", 
            icon={"gradient": ["#667eea","#764ba2"], "symbol":"🎬"}, 
            description=self.tr("视频处理引擎"), 
            status=""
        )

        self.software_cards = [ffmpeg_card]

        for card in self.software_cards:
            layout.addWidget(card)

        return group