from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame, QPushButton, QApplication
)
from app.ui.library.qfluentwidgets import (
    setFont, ScrollArea, TeachingTip, InfoBarIcon, TeachingTipTailPosition
)
from app.ui.widgets.gradient_header_widget import GradientHeader
from app.ui.widgets.license_widget import LicenseWidget, LicenseDropZone
from app.ui.widgets.auto_auth_widget import AutoAuthWidget
from app.ui.widgets.auth_records_widget import AuthRecordsWidget
from app.ui.widgets.custom_card_group_widget import CustomGroupBox
from app.license.license_manager import LicenseManager
from app.license.auto_auth import AutoAuthService
from app.ui.common.event_bus import global_event_bus
import app.library._machine_id as machine_id


PRIMARY = "#6366f1"
PRIMARY_HOVER = "#4f46e5"
PRIMARY_PRESSED = "#4338ca"
DANGER = "#ef4444"
CARD_BG = "#ffffff"
SUBTLE_BG = "#f8fafc"
HOVER_BG = "#f1f5f9"
TEXT_PRIMARY = "#1e293b"
TEXT_SECONDARY = "#64748b"
ACCENT_TEXT = "#b45309"
ACCENT_TEXT_HOVER = "#92400e"
BORDER = "#e2e8f0"
RADIUS = 12

SPACING_SM = 12
SPACING_MD = 20
SPACING_LG = 32


class ReferenceDropZone(LicenseDropZone):
    NORMAL_STYLE = (
        f"QFrame#licenseDropZone {{ background: {SUBTLE_BG};"
        f" border: 2px dashed {BORDER}; border-radius: {RADIUS}px; }}"
    )
    HOVER_STYLE = (
        f"QFrame#licenseDropZone {{ background: {HOVER_BG};"
        f" border: 2px dashed {PRIMARY}; border-radius: {RADIUS}px; }}"
    )

    @classmethod
    def promote(cls, drop_zone: LicenseDropZone) -> "ReferenceDropZone":
        drop_zone.__class__ = cls
        drop_zone._apply_normal_style()
        return drop_zone

    def _apply_normal_style(self):
        self.setStyleSheet(self.NORMAL_STYLE)

    def _apply_hover_style(self):
        self.setStyleSheet(self.HOVER_STYLE)


class MultiUnitAutoAuthWidget(AutoAuthWidget):
    price_updated = Signal()

    UNIT_DAY = 0
    UNIT_MONTH = 1
    UNIT_YEAR = 2
    UNIT_LABELS = ("天", "月", "年")
    DAYS_PER_MONTH = 30
    DAYS_PER_YEAR = 365
    UNIT_RANGES = {
        UNIT_DAY: (1, 3650, 30),
        UNIT_MONTH: (1, 120, 1),
        UNIT_YEAR: (1, 10, 1),
    }

    def __init__(self, service: AutoAuthService, parent=None):
        super().__init__(service, parent)
        self._install_units()

    def _install_units(self):
        combo = self.unit_combo
        combo.clear()
        combo.addItems([self.tr(label) for label in self.UNIT_LABELS])
        combo.setCurrentIndex(self.UNIT_DAY)

    def current_days(self) -> int:
        value = self.duration_spin.value()
        index = self.unit_combo.currentIndex()
        if index == self.UNIT_MONTH:
            value *= self.DAYS_PER_MONTH
        elif index == self.UNIT_YEAR:
            value *= self.DAYS_PER_YEAR
        return self._service.quote(value).days

    def _on_unit_changed(self, index: int):
        minimum, maximum, default = self.UNIT_RANGES.get(index, self.UNIT_RANGES[self.UNIT_DAY])
        self.duration_spin.setRange(minimum, maximum)
        self.duration_spin.setValue(default)
        self._update_price()

    def _update_price(self, *args):
        super()._update_price(*args)
        self.price_updated.emit()


class LicenseView(QWidget):
    DURATION_PRESETS = (
        ("1天", MultiUnitAutoAuthWidget.UNIT_DAY, 1),
        ("1月", MultiUnitAutoAuthWidget.UNIT_MONTH, 1),
        ("3个月", MultiUnitAutoAuthWidget.UNIT_MONTH, 3),
        ("1年", MultiUnitAutoAuthWidget.UNIT_YEAR, 1),
    )

    def __init__(self, license_manager: LicenseManager, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("LicenseView")
        self._license_manager = license_manager
        self._auto_auth_service = AutoAuthService(license_manager=license_manager)
        self._deal_breakpoints = self._collect_deal_breakpoints()
        self._better_deal_days = 0
        self._setup_ui()

    def _collect_deal_breakpoints(self) -> tuple:
        max_days = MultiUnitAutoAuthWidget.UNIT_RANGES[MultiUnitAutoAuthWidget.UNIT_DAY][1]
        breakpoints = []
        previous = None
        for days in range(1, max_days + 1):
            amount = self._auto_auth_service.quote(days).amount
            if previous is not None and amount < previous:
                breakpoints.append(days)
            previous = amount
        return tuple(breakpoints)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(SPACING_SM)

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
        content_layout.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD)
        content_layout.setSpacing(SPACING_LG)
        content_layout.setAlignment(Qt.AlignTop)

        license_group = CustomGroupBox(title=self.tr("📄 许可证信息"))
        license_group.addCard(card=self._create_license_section())
        content_layout.addWidget(license_group)

        auto_auth_group = CustomGroupBox(title=self.tr("⚡ 自动授权服务"))
        auto_auth_group.addCard(card=self._create_auto_auth_section())
        content_layout.addWidget(auto_auth_group)

        records_group = CustomGroupBox(title=self.tr("🧾 授权记录"))
        records_group.addCard(card=self._create_records_section())
        content_layout.addWidget(records_group)

        instructions_group = CustomGroupBox(title=self.tr("📖 激活说明"))
        instructions_group.addCard(card=self._create_instructions())
        content_layout.addWidget(instructions_group)

        content_layout.addWidget(self._create_tips_card())

        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.enableTransparentBackground()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        main_layout.addWidget(scroll)

    def _create_license_section(self) -> QWidget:
        self.license_widget = LicenseWidget(self._license_manager, self)
        self.license_widget.license_changed.connect(lambda: global_event_bus.License_update.emit())
        self.license_widget.hide()

        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING_MD)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(SPACING_MD)

        status_card = self.license_widget.status_card
        status_card.setStyleSheet(
            f"QFrame#licenseStatusCard {{ background: {CARD_BG};"
            f" border: 1px solid {BORDER}; border-radius: {RADIUS}px; }}"
        )
        top_row.addWidget(status_card, 1)
        top_row.addWidget(self._create_device_id_box(), 1)
        layout.addLayout(top_row)

        self.license_widget.machine_id_card.hide()

        layout.addWidget(self._create_upload_area())
        return section

    def _create_device_id_box(self) -> QWidget:
        box = QFrame()
        box.setObjectName("deviceIdBox")
        box.setStyleSheet(
            f"QFrame#deviceIdBox {{ background: {CARD_BG};"
            f" border: 1px solid {BORDER}; border-radius: {RADIUS}px; }}"
        )
        self._device_id_box = box

        layout = QVBoxLayout(box)
        layout.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        title_label = QLabel(self.tr("设备标识码"))
        setFont(title_label, 13, QFont.Bold)
        title_label.setStyleSheet(f"color: {TEXT_PRIMARY};")
        header.addWidget(title_label)
        header.addStretch()

        copy_btn = QPushButton("📋 " + self.tr("复制"))
        setFont(copy_btn, 11)
        copy_btn.setCursor(Qt.PointingHandCursor)
        copy_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 4px 12px;
                background: {PRIMARY};
                color: white;
                border: none;
                border-radius: 6px;
            }}
            QPushButton:hover {{ background: {PRIMARY_HOVER}; }}
            QPushButton:pressed {{ background: {PRIMARY_PRESSED}; }}
        """)
        copy_btn.clicked.connect(self._copy_machine_id)
        header.addWidget(copy_btn)
        layout.addLayout(header)

        value_label = QLabel(machine_id.get_machine_id_display())
        setFont(value_label, 12)
        value_label.setWordWrap(True)
        value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        value_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        layout.addWidget(value_label)
        layout.addStretch()
        return box

    def _copy_machine_id(self):
        QApplication.clipboard().setText(machine_id.get_machine_id())
        TeachingTip.create(
            target=self._device_id_box,
            icon=InfoBarIcon.SUCCESS,
            title=self.tr("已复制"),
            content=self.tr("设备标识码已复制到剪贴板，请发送给开发者以获取许可证"),
            isClosable=True,
            tailPosition=TeachingTipTailPosition.BOTTOM,
            duration=2000,
            parent=self.window()
        )

    def _create_upload_area(self) -> QWidget:
        drop_zone = ReferenceDropZone.promote(self.license_widget.drop_zone)

        zone_layout = drop_zone.layout()
        # zone_layout.setContentsMargins(SPACING_MD, SPACING_LG, SPACING_MD, SPACING_LG)
        zone_layout.setSpacing(SPACING_SM)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(SPACING_SM)
        btn_row.addStretch()
        btn_row.addWidget(self.license_widget.browse_btn)
        btn_row.addWidget(self.license_widget.deactivate_btn)
        btn_row.addStretch()
        zone_layout.addLayout(btn_row)
        return drop_zone

    def _create_auto_auth_section(self) -> QWidget:
        self.auto_auth_widget = MultiUnitAutoAuthWidget(self._auto_auth_service, self)
        self.auto_auth_widget.license_activated.connect(self._on_license_activated)
        self.auto_auth_widget.hide()

        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING_MD)

        layout.addLayout(self._create_auto_auth_header())
        layout.addLayout(self._create_pricing_grid())
        layout.addWidget(self._create_separator())
        layout.addLayout(self._create_subscription_controls())

        help_label = QLabel(self.tr(
            "支付完成后自动校验到账并下发与本机绑定的授权文件；每一笔订单均生成可追溯记录，"
            "如遇异常可凭订单号申诉。"
        ))
        setFont(help_label, 11)
        help_label.setWordWrap(True)
        help_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        layout.addWidget(help_label)

        self.auto_auth_widget.price_updated.connect(self._refresh_better_deal)
        self._refresh_better_deal()
        return section

    def _create_auto_auth_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        subtitle = QLabel(self.tr("阶梯计价，按需订阅，时长越长单价越低"))
        setFont(subtitle, 11)
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY};")
        row.addWidget(subtitle)
        row.addStretch()

        badge = QLabel("AUTOMATED")
        setFont(badge, 10, QFont.Bold)
        badge.setStyleSheet(
            f"QLabel {{ background: {PRIMARY}; color: white;"
            " border-radius: 4px; padding: 2px 8px; }"
        )
        row.addWidget(badge)
        return row

    def _create_pricing_grid(self) -> QHBoxLayout:
        grid = QHBoxLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(SPACING_MD)
        for card in self.auto_auth_widget._tier_cards.values():
            grid.addWidget(card, 1)
        return grid

    def _create_separator(self) -> QWidget:
        line = QFrame()
        line.setObjectName("controlsSeparator")
        line.setFixedHeight(1)
        line.setStyleSheet(f"QFrame#controlsSeparator {{ background: {BORDER}; border: none; }}")
        return line

    def _create_subscription_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING_MD)
        row.setAlignment(Qt.AlignBottom)

        row.addLayout(self._create_duration_input())
        row.addStretch()
        row.addLayout(self._create_checkout_summary())
        return row

    def _create_duration_input(self) -> QVBoxLayout:
        group = QVBoxLayout()
        group.setContentsMargins(0, 0, 0, 0)
        group.setSpacing(8)

        label = QLabel(self.tr("订阅时长"))
        setFont(label, 11)
        label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        group.addWidget(label)

        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(8)
        input_row.addWidget(self.auto_auth_widget.duration_spin)
        input_row.addWidget(self.auto_auth_widget.unit_combo)
        input_row.addStretch()
        group.addLayout(input_row)

        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.setSpacing(8)
        for text, unit, value in self.DURATION_PRESETS:
            preset_row.addWidget(self._create_preset_button(text, unit, value))
        preset_row.addStretch()
        group.addLayout(preset_row)
        return group

    def _create_preset_button(self, text: str, unit: int, value: int) -> QPushButton:
        btn = QPushButton(self.tr(text))
        setFont(btn, 10)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                padding: 4px 10px;
                background: {HOVER_BG};
                color: {TEXT_SECONDARY};
                border: 1px solid {BORDER};
                border-radius: 4px;
            }}
            QPushButton:hover {{ background: {BORDER}; }}
            QPushButton:pressed {{ background: #cbd5e1; }}
        """)
        btn.clicked.connect(
            lambda _=False, u=unit, v=value: self._apply_duration_preset(u, v)
        )
        return btn

    def _apply_duration_preset(self, unit: int, value: int):
        widget = self.auto_auth_widget
        if widget.unit_combo.currentIndex() != unit:
            widget.unit_combo.setCurrentIndex(unit)
        widget.duration_spin.setValue(value)

    def _create_checkout_summary(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING_MD)
        row.setAlignment(Qt.AlignBottom)

        details = QVBoxLayout()
        details.setContentsMargins(0, 0, 0, 0)
        details.setSpacing(4)

        hint = QLabel(self.tr("预计支付金额"))
        setFont(hint, 10)
        hint.setAlignment(Qt.AlignRight)
        hint.setStyleSheet(f"color: {TEXT_SECONDARY};")
        details.addWidget(hint)
        details.addWidget(self.auto_auth_widget.price_label)
        details.addWidget(self.auto_auth_widget.days_hint)
        details.addWidget(self._create_better_deal_button())
        row.addLayout(details)

        row.addWidget(self.auto_auth_widget.checkout_btn, 0, Qt.AlignBottom)
        return row

    def _create_better_deal_button(self) -> QPushButton:
        btn = QPushButton()
        setFont(btn, 10)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                padding: 0;
                text-align: right;
                color: {ACCENT_TEXT};
            }}
            QPushButton:hover {{ color: {ACCENT_TEXT_HOVER}; }}
        """)
        btn.clicked.connect(self._apply_better_deal)
        btn.hide()
        self._better_deal_btn = btn
        return btn

    def _refresh_better_deal(self):
        quote = self._find_better_deal(self.auto_auth_widget.current_days())
        if quote is None:
            self._better_deal_days = 0
            self._better_deal_btn.hide()
            return
        self._better_deal_days = quote.days
        self._better_deal_btn.setText(
            self.tr(f"买 {quote.days} 天更便宜：¥{quote.amount_text}，点此套用")
        )
        self._better_deal_btn.show()

    def _find_better_deal(self, days: int):
        current_amount = self._auto_auth_service.quote(days).amount
        best = None
        for candidate in self._deal_breakpoints:
            if candidate <= days:
                continue
            quote = self._auto_auth_service.quote(candidate)
            if quote.amount < current_amount and (best is None or quote.amount < best.amount):
                best = quote
        return best

    def _apply_better_deal(self):
        days = self._better_deal_days
        if days <= 0:
            return
        widget = self.auto_auth_widget
        if days % widget.DAYS_PER_YEAR == 0:
            self._apply_duration_preset(widget.UNIT_YEAR, days // widget.DAYS_PER_YEAR)
        else:
            self._apply_duration_preset(widget.UNIT_DAY, days)

    def _on_license_activated(self, license_path: str):
        if license_path:
            self.license_widget.activate_license_file(license_path)
        else:
            self.license_widget.refresh()
            global_event_bus.License_update.emit()
        self.auth_records_widget.refresh()

    def _create_records_section(self) -> QWidget:
        self.auth_records_widget = AuthRecordsWidget(self._auto_auth_service, self)
        global_event_bus.License_update.connect(self.auth_records_widget.refresh)

        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING_SM)

        hint = QLabel(self.tr(
            "每笔订单的下单时间、授权时长、状态与金额均落盘留痕；可按状态、时间范围或关键字筛选，"
            "勾选行首复选框后可批量删除，双击行复制订单摘要；删除仅移除看板记录，审计流水不可篡改。"
        ))
        setFont(hint, 11)
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {TEXT_SECONDARY};")
        layout.addWidget(hint)
        layout.addWidget(self.auth_records_widget)
        return section

    def _create_instructions(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING_MD)

        grid = QHBoxLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(SPACING_MD)

        grid.addWidget(self._create_instruction_card(
            badge="A",
            title=self.tr("自动授权 (推荐)"),
            paragraph=self.tr(
                "在「自动授权服务」中输入时长 → 点击「开始授权」→ 扫码付款，"
                "到账后自动下发并激活与本机绑定的许可证。"
            ),
        ), 1)

        grid.addWidget(self._create_instruction_card(
            badge="B",
            title=self.tr("人工授权 (支持免费试用一天)"),
            steps=[
                self.tr("复制上方的「设备标识码」"),
                self.tr("将设备标识码发送给开发者 (QQ群: 1080076113)"),
                self.tr("收到 .lic 许可证文件后，拖拽到上方区域或点击「选择许可证文件」"),
                self.tr("激活成功后即可使用全部 Pro 功能"),
            ],
        ), 1)
        layout.addLayout(grid)
        return widget

    def _create_instruction_card(self, badge: str, title: str,
                                 paragraph: str = "", steps: list = None) -> QWidget:
        card = QFrame()
        card.setObjectName("instructionCard")
        card.setStyleSheet(
            f"QFrame#instructionCard {{ background: {SUBTLE_BG};"
            f" border: 1px solid {BORDER}; border-radius: {RADIUS}px; }}"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD)
        card_layout.setSpacing(SPACING_SM)
        card_layout.setAlignment(Qt.AlignTop)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        badge_label = QLabel(badge)
        setFont(badge_label, 11, QFont.Bold)
        badge_label.setFixedSize(20, 20)
        badge_label.setAlignment(Qt.AlignCenter)
        badge_label.setStyleSheet(
            f"QLabel {{ background: {DANGER}; color: white; border-radius: 4px; }}"
        )
        header.addWidget(badge_label)

        title_label = QLabel(title)
        setFont(title_label, 13, QFont.Bold)
        title_label.setStyleSheet(f"color: {TEXT_PRIMARY};")
        header.addWidget(title_label)
        header.addStretch()
        card_layout.addLayout(header)

        if paragraph:
            para_label = QLabel(paragraph)
            setFont(para_label, 12)
            para_label.setWordWrap(True)
            para_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
            card_layout.addWidget(para_label)

        if steps:
            for index, step in enumerate(steps, start=1):
                step_label = QLabel(f"{index}.  {step}")
                setFont(step_label, 12)
                step_label.setWordWrap(True)
                step_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
                card_layout.addWidget(step_label)

        return card

    def _create_tips_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("tipsCard")
        card.setStyleSheet(
            f"QFrame#tipsCard {{ background: #fffbeb;"
            f" border: 1px solid #fde68a; border-radius: {RADIUS}px; }}"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD)
        card_layout.setSpacing(6)

        title_label = QLabel("💡 " + self.tr("提示"))
        setFont(title_label, 13, QFont.Bold)
        title_label.setStyleSheet("color: #b45309;")
        card_layout.addWidget(title_label)

        tips = [
            self.tr("许可证与设备绑定，换机器需要重新申请"),
            self.tr("许可证到期前 7 天会收到续费提醒"),
            self.tr("每笔订单都有可追溯记录，付款异常可凭订单号申诉"),
            self.tr("如遇问题请联系开发者获取帮助 (QQ群: 1080076113)"),
        ]
        for tip in tips:
            tip_label = QLabel(f"•  {tip}")
            setFont(tip_label, 12)
            tip_label.setWordWrap(True)
            tip_label.setStyleSheet("color: #92400e;")
            card_layout.addWidget(tip_label)

        return card
