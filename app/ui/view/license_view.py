from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from app.ui.library.qfluentwidgets import setFont, ScrollArea
from app.ui.widgets.gradient_header_widget import GradientHeader
from app.ui.widgets.license_widget import LicenseWidget
from app.ui.widgets.custom_card_group_widget import CustomGroupBox
from app.license.license_manager import LicenseManager
from app.ui.common.event_bus import global_event_bus


class LicenseView(QWidget):
    def __init__(self, license_manager: LicenseManager, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("LicenseView")
        self._license_manager = license_manager
        self._setup_ui()

    def _setup_ui(self):
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

        title_label = QLabel(self.tr("🔑 授权管理"))
        setFont(title_label, fontSize=24, weight=QFont.Bold)
        title_label.setStyleSheet("QLabel { color: white; }")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        main_layout.addWidget(header)

    def _setup_content(self, main_layout: QVBoxLayout):
        scroll = ScrollArea()

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(16)
        content_layout.setAlignment(Qt.AlignTop)

        # License management group
        license_group = CustomGroupBox(title=self.tr("📋 许可证信息"))
        license_widget = LicenseWidget(self._license_manager, self)
        license_widget.license_changed.connect(lambda: global_event_bus.License_update.emit())
        license_group.addCard(card=license_widget)
        content_layout.addWidget(license_group)

        # Instructions group
        instructions_group = CustomGroupBox(title=self.tr("📖 激活说明"))
        instructions_widget = self._create_instructions()
        instructions_group.addCard(card=instructions_widget)
        content_layout.addWidget(instructions_group)

        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.enableTransparentBackground()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        main_layout.addWidget(scroll)

    def _create_instructions(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        steps = [
            "1️⃣  复制上方的「设备标识码」",
            "2️⃣  将设备标识码发送给开发者",
            "3️⃣  收到 .lic 许可证文件后，拖拽到上方区域或点击「选择许可证文件」",
            "4️⃣  激活成功后即可使用全部 Pro 功能",
            "",
            "💡 提示：",
            "• 许可证与设备绑定，换机器需要重新申请",
            "• 许可证到期前 7 天会收到续费提醒",
            "• 如遇问题请联系开发者获取帮助",
        ]

        for step in steps:
            label = QLabel(step)
            setFont(label, 12)
            label.setStyleSheet("color: #374151;")
            label.setWordWrap(True)
            layout.addWidget(label)

        return widget
