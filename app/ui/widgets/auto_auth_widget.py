from decimal import Decimal
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QFrame, QApplication, QSizePolicy,
    QPushButton
)
from app.ui.library.qfluentwidgets import (
    setFont, SpinBox, ComboBox, MessageBoxBase, IndeterminateProgressRing,
    TeachingTip, InfoBarIcon, TeachingTipTailPosition
)
from app.ui.resources import resource
from app.license.auto_auth import AutoAuthService, AuthOrder, AuthStage, PricingTier, money_text


PRIMARY = "#6366f1"
PRIMARY_HOVER = "#4f46e5"
PRIMARY_LIGHT = "#eff6ff"
TEXT_MAIN = "#0f172a"
TEXT_SUB = "#64748b"
BORDER = "#e2e8f0"
SUCCESS = "#10b981"
WARNING = "#f59e0b"
DANGER = "#dc2626"

PAYMENT_CHANNELS = (
    ("支付宝", ":/powertools/images/alipay.jpg"),
    ("微信", ":/powertools/images/wechat.jpg"),
)


class TierCard(QFrame):
    def __init__(self, tier: PricingTier, parent=None):
        super().__init__(parent)
        self.setObjectName("tierCard")
        self._tier = tier
        self._active = False
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumWidth(160)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 0, 0, 12)
        layout.setSpacing(4)

        badge_container = QWidget()
        badge_container.setFixedHeight(20)
        badge_row = QHBoxLayout(badge_container)
        badge_row.setContentsMargins(0, 0, 0, 0)
        badge_row.setSpacing(0)
        if tier.popular:
            badge_row.addStretch()
            badge = QLabel(self.tr("超值推荐"))
            setFont(badge, 10, QFont.Bold)
            badge.setStyleSheet(f"QLabel {{ background: {WARNING}; color: white; border-radius: 4px; padding: 1px 6px; }}")
            badge_row.addWidget(badge)
        layout.addWidget(badge_container)
        self.name_label = QLabel(self.tr(tier.name))
        setFont(self.name_label, 12)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet(f"color: {TEXT_SUB};")
        layout.addWidget(self.name_label)

        price_row = QHBoxLayout()
        price_row.setContentsMargins(0, 0, 0, 0)
        price_row.setSpacing(2)
        price_row.addStretch()
        price_label = QLabel(tier.price_text)
        setFont(price_label, 16, QFont.Bold)
        price_label.setStyleSheet(f"color: {TEXT_MAIN};")
        unit_label = QLabel(tier.unit_text)
        setFont(unit_label, 11)
        unit_label.setStyleSheet(f"color: {TEXT_SUB};")
        price_row.addWidget(price_label)
        price_row.addWidget(unit_label, 0, Qt.AlignBottom)
        price_row.addStretch()
        layout.addLayout(price_row)
        self._apply_style()

    @property
    def tier_key(self) -> str:
        return self._tier.key

    def set_active(self, active: bool):
        if active == self._active:
            return
        self._active = active
        self._apply_style()

    def _apply_style(self):
        if self._active:
            self.setStyleSheet(
                f"QFrame#tierCard {{ background: {PRIMARY_LIGHT}; border: 1px solid {PRIMARY};"
                f" border-radius: 10px; }}"
            )
        else:
            self.setStyleSheet(
                "QFrame#tierCard { background: #f8fafc; border: 1px solid #f1f5f9;"
                " border-radius: 10px; }"
            )


class PaymentDialog(MessageBoxBase):
    POLL_INTERVAL_MS = 3000
    license_activated = Signal(str)

    def __init__(self, service: AutoAuthService, order: AuthOrder, parent=None):
        super().__init__(parent)
        self._service = service
        self._order = order
        self._finished = False

        self._setup_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(self.POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        QTimer.singleShot(100, self._poll)

    def _setup_ui(self):
        self.widget.setMinimumWidth(512)

        title = QLabel(self.tr("扫码完成支付"))
        setFont(title, 16, QFont.Bold)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_MAIN};")
        self.viewLayout.addWidget(title)

        sub = QLabel(self.tr("订单号: ") + self._order.order_id)
        setFont(sub, 12)
        sub.setAlignment(Qt.AlignCenter)
        sub.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sub.setStyleSheet(f"color: {TEXT_SUB};")
        self.viewLayout.addWidget(sub)

        amount = QLabel(f"¥{self._order.amount_text}")
        setFont(amount, 24, QFont.Bold)
        amount.setAlignment(Qt.AlignCenter)
        amount.setStyleSheet(f"color: {TEXT_MAIN};")
        self.viewLayout.addWidget(amount)

        qr_row = QHBoxLayout()
        qr_row.setSpacing(16)
        qr_row.addStretch()
        for name, path in PAYMENT_CHANNELS:
            qr_row.addWidget(self._create_qr_card(name, path))
        qr_row.addStretch()
        self.viewLayout.addLayout(qr_row)

        tip = QLabel(self.tr("请按上方金额付款，并在备注中填写订单号，便于自动核对与对账"))
        setFont(tip, 11)
        tip.setWordWrap(True)
        tip.setAlignment(Qt.AlignCenter)
        tip.setStyleSheet(f"color: {TEXT_SUB};")
        self.viewLayout.addWidget(tip)

        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        status_row.addStretch()
        self.spinner = IndeterminateProgressRing(self)
        self.spinner.setFixedSize(16, 16)
        self.spinner.setStrokeWidth(3)
        status_row.addWidget(self.spinner)
        self.status_label = QLabel(self.tr("等待扫码支付，完成后自动下发文件..."))
        setFont(self.status_label, 12)
        # self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {TEXT_SUB};")
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        self.viewLayout.addLayout(status_row)

        self.yesButton.setText(self.tr("我已完成支付"))
        self.cancelButton.setText(self.tr("取消并返回"))

    def _create_qr_card(self, name: str, resource_path: str) -> QWidget:
        card = QFrame(self)
        card.setObjectName("qrCard")
        card.setStyleSheet(f"QFrame#qrCard {{ background: white; border: 1px solid {BORDER}; border-radius: 12px; }}")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 10)
        layout.setSpacing(6)

        image = QLabel(card)
        image.setFixedSize(240, 240)
        image.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(resource_path)
        if pixmap.isNull():
            image.setText(self.tr("（收款码加载失败）"))
            setFont(image, 11)
            image.setStyleSheet(f"color: {DANGER};")
        else:
            image.setPixmap(pixmap.scaled(240, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(image, 0, Qt.AlignCenter)

        caption = QLabel(name, card)
        setFont(caption, 11, QFont.Bold)
        caption.setAlignment(Qt.AlignCenter)
        caption.setStyleSheet(f"color: {TEXT_SUB};")
        layout.addWidget(caption)
        return card

    def _poll(self):
        if self._finished:
            return
        progress = self._service.poll(self._order)
        if progress.order is not None:
            self._order = progress.order

        if progress.stage == AuthStage.ISSUED:
            self._set_status(progress.message or self.tr("支付成功！授权文件已生成"), SUCCESS, spinning=False)
            self._finished = True
            self._timer.stop()
            self.license_activated.emit(progress.license_path)
            QTimer.singleShot(1200, self.accept)
        elif progress.stage == AuthStage.PAID:
            self._set_status(progress.message, PRIMARY)
        elif progress.stage == AuthStage.WAITING:
            self._set_status(progress.message, TEXT_SUB)
        elif progress.stage == AuthStage.UNAVAILABLE:
            self._timer.stop()
            self._set_status(
                self.tr("付款后请点击「我已完成支付」，订单号与设备码会自动复制并发送给作者，等待作者签发。"),
                WARNING,
                spinning=False,
            )
        elif progress.stage == AuthStage.TIMEOUT:
            self._timer.stop()
            self._set_status(progress.message or self.tr("订单已超时关闭，请重新下单"), DANGER, spinning=False)
        else:
            self._timer.stop()
            self._set_status(progress.message or self.tr("支付未完成"), DANGER, spinning=False)

    def _set_status(self, text: str, color: str, spinning: bool = True):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 600;" if color != TEXT_SUB else f"color: {TEXT_SUB};")
        self.spinner.setVisible(spinning)

    def validate(self) -> bool:
        if self._finished:
            return True
        self._service.mark_user_claimed_paid(self._order, note="user clicked paid button")
        QApplication.clipboard().setText(self._service.support_summary(self._order))
        self._set_status(
            self.tr("已记录支付声明，对账信息已复制到剪贴板。核对通过后将自动下发授权文件。"),
            PRIMARY,
            spinning=self._timer.isActive(),
        )
        return False

    def reject(self):
        self._timer.stop()
        if not self._finished:
            self._service.cancel(self._order)
        super().reject()


class AutoAuthWidget(QWidget):
    license_activated = Signal(str)
    UNIT_DAY = 0
    UNIT_YEAR = 1

    def __init__(self, service: AutoAuthService, parent=None):
        super().__init__(parent)
        self._service = service
        self._tier_cards = {}
        self._setup_ui()
        self._update_price()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addLayout(self._create_subtitle_row())
        layout.addLayout(self._create_tier_grid())
        layout.addWidget(self._create_form())

        footnote = QLabel(self.tr(
            "支付完成后自动校验到账并下发与本机绑定的授权文件；每一笔订单均生成可追溯记录，"
            "如遇异常可凭订单号申诉。"
        ))
        setFont(footnote, 11)
        footnote.setWordWrap(True)
        footnote.setStyleSheet(f"color: {TEXT_SUB};")
        layout.addWidget(footnote)

    def _create_subtitle_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        subtitle = QLabel(self.tr("阶梯计价，按需订阅，时长越长单价越低"))
        setFont(subtitle, 12)
        subtitle.setStyleSheet(f"color: {TEXT_SUB};")
        row.addWidget(subtitle)
        row.addStretch()

        badge = QLabel("AUTOMATED")
        setFont(badge, 10, QFont.Bold)
        badge.setStyleSheet(
            "QLabel { background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            " stop:0 #8b5cf6, stop:1 #6366f1); color: white; border-radius: 9px;"
            " padding: 2px 10px; }"
        )
        row.addWidget(badge)
        return row

    def _create_tier_grid(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        for column, tier in enumerate(self._service.tiers):
            card = TierCard(tier, self)
            self._tier_cards[tier.key] = card
            grid.addWidget(card, 0, column)
            grid.setColumnStretch(column, 1)
        return grid

    def _create_form(self) -> QWidget:
        frame = QFrame(self)
        frame.setObjectName("authFormCard")
        frame.setStyleSheet(
            "QFrame#authFormCard { background: #fafafa; border: 1px solid #f1f5f9;"
            " border-radius: 12px; }"
        )
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        input_group = QVBoxLayout()
        input_group.setSpacing(6)
        input_label = QLabel(self.tr("订阅时长"))
        setFont(input_label, 11, QFont.Bold)
        input_label.setStyleSheet(f"color: {TEXT_SUB};")
        input_group.addWidget(input_label)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self.duration_spin = SpinBox(frame)
        self.duration_spin.setRange(1, 3650)
        self.duration_spin.setValue(30)
        self.duration_spin.setFixedWidth(140)
        self.duration_spin.valueChanged.connect(self._update_price)
        input_row.addWidget(self.duration_spin)

        self.unit_combo = ComboBox(frame)
        self.unit_combo.addItems([self.tr("天"), self.tr("年")])
        self.unit_combo.setCurrentIndex(self.UNIT_DAY)
        self.unit_combo.setFixedWidth(80)
        self.unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        input_row.addWidget(self.unit_combo)
        input_group.addLayout(input_row)
        layout.addLayout(input_group)

        layout.addStretch()

        price_group = QVBoxLayout()
        price_group.setSpacing(2)
        price_hint = QLabel(self.tr("预计支付金额"))
        setFont(price_hint, 11)
        price_hint.setAlignment(Qt.AlignRight)
        price_hint.setStyleSheet(f"color: {TEXT_SUB};")
        price_group.addWidget(price_hint)

        self.price_label = QLabel("¥0.00")
        setFont(self.price_label, 26, QFont.Bold)
        self.price_label.setAlignment(Qt.AlignRight)
        self.price_label.setStyleSheet(f"color: {PRIMARY};")
        price_group.addWidget(self.price_label)

        self.days_hint = QLabel("")
        setFont(self.days_hint, 12, QFont.Bold)
        self.days_hint.setAlignment(Qt.AlignRight)
        self.days_hint.setStyleSheet(f"color: {TEXT_SUB};")
        price_group.addWidget(self.days_hint)
        layout.addLayout(price_group)

        self.checkout_btn = QPushButton("⚡ " + self.tr("开始授权"), frame)
        setFont(self.checkout_btn, 13, QFont.Bold)
        self.checkout_btn.setCursor(Qt.PointingHandCursor)
        self.checkout_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 12px 28px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 {PRIMARY}, stop:1 {PRIMARY_HOVER});
                color: white;
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{ background: {PRIMARY_HOVER}; }}
            QPushButton:pressed {{ background: #3730a3; }}
        """)
        self.checkout_btn.clicked.connect(self._on_checkout)
        layout.addWidget(self.checkout_btn)
        return frame

    def _on_unit_changed(self, index: int):
        if index == self.UNIT_YEAR:
            self.duration_spin.setRange(1, 10)
            self.duration_spin.setValue(1)
        else:
            self.duration_spin.setRange(1, 3650)
            self.duration_spin.setValue(30)
        self._update_price()

    def current_days(self) -> int:
        value = self.duration_spin.value()
        if self.unit_combo.currentIndex() == self.UNIT_YEAR:
            value *= 365
        return self._service.quote(value).days

    def _update_price(self, *_):
        days = self.current_days()
        quote = self._service.quote(days)
        self.price_label.setText(f"¥{quote.amount_text}")
        unit_price = money_text(quote.amount / Decimal(days)) if days else "0.00"
        self.days_hint.setText(self.tr(f"共 {days} 天 · 折合 ¥{unit_price}/天"))
        for key, card in self._tier_cards.items():
            card.set_active(key == quote.tier_key)

    def _on_checkout(self):
        try:
            order = self._service.create_order(self.current_days())
            self._service.mark_waiting_payment(order)
        except Exception as e:
            TeachingTip.create(
                target=self.checkout_btn,
                icon=InfoBarIcon.ERROR,
                title=self.tr("下单失败"),
                content=str(e),
                isClosable=True,
                tailPosition=TeachingTipTailPosition.BOTTOM,
                duration=4000,
                parent=self.window(),
            )
            return

        dialog = PaymentDialog(self._service, order, self.window())
        dialog.license_activated.connect(self.license_activated.emit)
        dialog.exec()
