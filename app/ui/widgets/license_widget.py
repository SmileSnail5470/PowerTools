from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QFrame, QApplication
from app.ui.library.qfluentwidgets import setFont, TeachingTip, InfoBarIcon, TeachingTipTailPosition, MessageBox
from app.license.machine_id import get_machine_id, get_machine_id_display
from app.license.license_manager import LicenseManager
from app.license.exceptions import LicenseError


class LicenseStatusCard(QFrame):
    def __init__(self, license_manager: LicenseManager, parent=None):
        super().__init__(parent)
        self._license_manager = license_manager
        self.setObjectName("licenseStatusCard")
        self._setup_ui()
        self._update_display()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # Header
        header_layout = QHBoxLayout()
        self.status_icon = QLabel()
        self.status_icon.setFixedSize(36, 36)
        self.status_icon.setAlignment(Qt.AlignCenter)
        setFont(self.status_icon, 20)

        self.status_title = QLabel()
        setFont(self.status_title, 16, QFont.Bold)

        header_layout.addWidget(self.status_icon)
        header_layout.addWidget(self.status_title)
        header_layout.addStretch()

        self.tier_badge = QLabel()
        setFont(self.tier_badge, 11, QFont.Bold)
        self.tier_badge.setAlignment(Qt.AlignCenter)
        self.tier_badge.setFixedHeight(24)
        header_layout.addWidget(self.tier_badge)

        layout.addLayout(header_layout)

        # Details
        self.details_label = QLabel()
        setFont(self.details_label, 12)
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)

        # Expiration info
        self.expiry_label = QLabel()
        setFont(self.expiry_label, 11)
        layout.addWidget(self.expiry_label)

        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet("""
            QFrame#licenseStatusCard {
                background: #fafafa;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
            }
        """)

    def _update_display(self):
        if self._license_manager.is_licensed:
            data = self._license_manager.license_data
            self.status_icon.setText("✅")
            self.status_title.setText(self.tr("已激活"))
            self.status_title.setStyleSheet("color: #059669;")

            self.tier_badge.setText(f" {data.tier.upper()} ")
            if data.tier == "pro":
                self.tier_badge.setStyleSheet("""
                    QLabel {
                        background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7c3aed, stop:1 #4f46e5);
                        color: white;
                        border-radius: 4px;
                        padding: 2px 10px;
                    }
                """)
                features_str = "、".join(data.features)
            else:
                self.tier_badge.setStyleSheet("""
                    QLabel {
                        background: #d1fae5;
                        color: #065f46;
                        border-radius: 4px;
                        padding: 2px 10px;
                    }
                """)
                features_str = "、".join([item[2] for item in data.features if item[1] == "free"])
            self.details_label.setText(self.tr(f"已授权功能：{features_str}"))
            self.details_label.setStyleSheet("color: #374151;")
            days = data.days_remaining
            if days <= 7:
                self.expiry_label.setText(self.tr(f"⚠️ 许可证将在 {days} 天后过期，请及时续费"))
                self.expiry_label.setStyleSheet("color: #dc2626;")
            else:
                self.expiry_label.setText(self.tr(f"有效期至：{data.expires_at[:10]}（剩余 {days} 天）"))
                self.expiry_label.setStyleSheet("color: #6b7280;")
        else:
            self.status_icon.setText("🔒")
            self.status_title.setText(self.tr("未激活"))
            self.status_title.setStyleSheet("color: #6b7280;")
            self.tier_badge.setText(" FREE ")
            self.tier_badge.setStyleSheet("""
                QLabel {
                    background: #f3f4f6;
                    color: #6b7280;
                    border-radius: 4px;
                    padding: 2px 10px;
                }
            """)

            error_msg = self._license_manager.error_message
            if error_msg:
                self.details_label.setText(self.tr(f"状态：{error_msg}"))
                self.details_label.setStyleSheet("color: #dc2626;")
            else:
                self.details_label.setText(self.tr("当前使用免费版，部分高级功能不可用"))
                self.details_label.setStyleSheet("color: #6b7280;")
            self.expiry_label.setText(self.tr("导入许可证文件以激活 Pro 版功能"))
            self.expiry_label.setStyleSheet("color: #9ca3af;")

    def refresh(self):
        self._update_display()


class MachineIdCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("machineIdCard")
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        title_label = QLabel(self.tr("设备标识码"))
        setFont(title_label, 12, QFont.Bold)
        title_label.setStyleSheet("color: #374151;")
        info_layout.addWidget(title_label)

        self.machine_id_label = QLabel(get_machine_id_display())
        setFont(self.machine_id_label, 11)
        self.machine_id_label.setStyleSheet("color: #6b7280; font-family: 'Consolas', 'Courier New', monospace;")
        self.machine_id_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        info_layout.addWidget(self.machine_id_label)

        layout.addLayout(info_layout)
        layout.addStretch()

        copy_btn = QPushButton("📋 " + self.tr("复制"))
        setFont(copy_btn, 12)
        copy_btn.setStyleSheet("""
            QPushButton {
                padding: 6px 16px;
                background: #4f46e5;
                color: white;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background: #4338ca;
            }
            QPushButton:pressed {
                background: #3730a3;
            }
        """)
        copy_btn.clicked.connect(self._copy_machine_id)
        layout.addWidget(copy_btn)

        self.setStyleSheet("""
            QFrame#machineIdCard {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
            }
        """)

    def _copy_machine_id(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(get_machine_id())
        TeachingTip.create(
            target=self,
            icon=InfoBarIcon.SUCCESS,
            title=self.tr("已复制"),
            content=self.tr("设备标识码已复制到剪贴板，请发送给开发者以获取许可证"),
            isClosable=True,
            tailPosition=TeachingTipTailPosition.BOTTOM,
            duration=2000,
            parent=self.window()
        )


class LicenseDropZone(QFrame):
    file_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("licenseDropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(100)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(8)

        icon_label = QLabel("📄")
        setFont(icon_label, 28)
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)

        text_label = QLabel(self.tr("拖拽 .lic 许可证文件到此处"))
        setFont(text_label, 12)
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setStyleSheet("color: #6b7280;")
        layout.addWidget(text_label)

        sub_label = QLabel(self.tr("或点击下方按钮选择文件"))
        setFont(sub_label, 10)
        sub_label.setAlignment(Qt.AlignCenter)
        sub_label.setStyleSheet("color: #9ca3af;")
        layout.addWidget(sub_label)

        self._apply_normal_style()

    def _apply_normal_style(self):
        self.setStyleSheet("""
            QFrame#licenseDropZone {
                background: #fafbfc;
                border: 2px dashed #d1d5db;
                border-radius: 12px;
            }
        """)

    def _apply_hover_style(self):
        self.setStyleSheet("""
            QFrame#licenseDropZone {
                background: #eff6ff;
                border: 2px dashed #4f46e5;
                border-radius: 12px;
            }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(url.toLocalFile().endswith(".lic") for url in urls):
                event.acceptProposedAction()
                self._apply_hover_style()
                return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._apply_normal_style()

    def dropEvent(self, event: QDropEvent):
        self._apply_normal_style()
        urls = event.mimeData().urls()
        for url in urls:
            path = url.toLocalFile()
            if path.endswith(".lic"):
                self.file_dropped.emit(path)
                return


class LicenseWidget(QWidget):
    license_changed = Signal()

    def __init__(self, license_manager: LicenseManager, parent=None):
        super().__init__(parent)
        self._license_manager = license_manager
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Status Card
        self.status_card = LicenseStatusCard(self._license_manager, self)
        layout.addWidget(self.status_card)

        # Machine ID Card
        self.machine_id_card = MachineIdCard(self)
        layout.addWidget(self.machine_id_card)

        # Drop Zone
        self.drop_zone = LicenseDropZone(self)
        self.drop_zone.file_dropped.connect(self._on_file_dropped)
        layout.addWidget(self.drop_zone)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self.browse_btn = QPushButton("📁 " + self.tr("选择许可证文件"))
        setFont(self.browse_btn, 12, QFont.Bold)
        self.browse_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                background: #4f46e5;
                color: white;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover { background: #4338ca; }
            QPushButton:pressed { background: #3730a3; }
        """)
        self.browse_btn.clicked.connect(self._browse_license_file)
        btn_layout.addWidget(self.browse_btn)

        self.deactivate_btn = QPushButton("🗑️ " + self.tr("取消激活"))
        setFont(self.deactivate_btn, 12)
        self.deactivate_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                background: #fee2e2;
                color: #dc2626;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover { background: #fecaca; }
            QPushButton:pressed { background: #fca5a5; }
        """)
        self.deactivate_btn.clicked.connect(self._deactivate_license)
        self.deactivate_btn.setVisible(self._license_manager.is_licensed)
        btn_layout.addWidget(self.deactivate_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _browse_license_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("选择许可证文件"),
            "",
            self.tr("许可证文件 (*.lic);;所有文件 (*)"),
        )
        if file_path:
            self._activate_license(file_path)

    def _on_file_dropped(self, file_path: str):
        self._activate_license(file_path)

    def _activate_license(self, file_path: str):
        try:
            self._license_manager.activate(file_path)
            self._refresh_ui()
            TeachingTip.create(
                target=self.status_card,
                icon=InfoBarIcon.SUCCESS,
                title=self.tr("激活成功"),
                content=self.tr(f"许可证已激活，授权等级：{self._license_manager.tier.upper()}"),
                isClosable=True,
                tailPosition=TeachingTipTailPosition.BOTTOM,
                duration=3000,
                parent=self.window()
            )
            self.license_changed.emit()
        except LicenseError as e:
            TeachingTip.create(
                target=self.status_card,
                icon=InfoBarIcon.ERROR,
                title=self.tr("激活失败"),
                content=str(e),
                isClosable=True,
                tailPosition=TeachingTipTailPosition.BOTTOM,
                duration=5000,
                parent=self.window()
            )

    def _deactivate_license(self):
        msg_box = MessageBox(
            title=self.tr("确认取消激活"),
            content=self.tr("取消激活后将恢复为免费版，Pro 功能将不再可用。\n如需重新激活，请再次导入许可证文件。"),
            parent=self.window()
        )
        if msg_box.exec():
            self._license_manager.deactivate()
            self._refresh_ui()
            self.license_changed.emit()

    def _refresh_ui(self):
        self.status_card.refresh()
        self.deactivate_btn.setVisible(self._license_manager.is_licensed)
