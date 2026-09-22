from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QHeaderView, QTableWidgetItem, QAbstractItemView, QApplication
)
from app.ui.library.qfluentwidgets import (
    TableWidget, ComboBox, CheckBox, SearchLineEdit, PushButton, PrimaryPushButton, FluentIcon,
    MessageBox, TeachingTip, InfoBarIcon, TeachingTipTailPosition
)
from app.license.auto_auth import AutoAuthService, AuthOrder, OrderStatus


PRIMARY = "#6366f1"
TEXT_MAIN = "#0f172a"
TEXT_SUB = "#64748b"
SUCCESS = "#10b981"
WARNING = "#f59e0b"
DANGER = "#dc2626"

STATUS_TEXTS = {
    OrderStatus.CREATED.value: "已创建",
    OrderStatus.WAITING_PAYMENT.value: "待支付",
    OrderStatus.USER_CLAIMED_PAID.value: "待核对",
    OrderStatus.PAID.value: "已到账",
    OrderStatus.ISSUING.value: "签发中",
    OrderStatus.ISSUED.value: "已签发",
    OrderStatus.ACTIVATED.value: "已激活",
    OrderStatus.CANCELLED.value: "已取消",
    OrderStatus.TIMEOUT.value: "已超时",
    OrderStatus.FAILED.value: "失败",
}

STATUS_COLORS = {
    OrderStatus.ACTIVATED.value: SUCCESS,
    OrderStatus.ISSUED.value: SUCCESS,
    OrderStatus.PAID.value: PRIMARY,
    OrderStatus.ISSUING.value: PRIMARY,
    OrderStatus.CREATED.value: WARNING,
    OrderStatus.WAITING_PAYMENT.value: WARNING,
    OrderStatus.USER_CLAIMED_PAID.value: WARNING,
    OrderStatus.CANCELLED.value: TEXT_SUB,
    OrderStatus.TIMEOUT.value: DANGER,
    OrderStatus.FAILED.value: DANGER,
}

TIME_RANGES = (
    ("全部时间", 0),
    ("近 7 天", 7),
    ("近 30 天", 30),
    ("近 90 天", 90),
    ("近一年", 365),
)


def status_text(status: str) -> str:
    return STATUS_TEXTS.get(status, status or "-")


def duration_text(days: int) -> str:
    days = int(days or 0)
    if days <= 0:
        return "-"
    if days % 365 == 0:
        return f"{days // 365} 年"
    if days % 30 == 0:
        return f"{days // 30} 个月"
    return f"{days} 天"


def local_time_text(iso_text: str) -> str:
    moment = parse_moment(iso_text)
    if moment is None:
        return str(iso_text) if iso_text else "-"
    return moment.astimezone().strftime("%Y-%m-%d %H:%M")


def parse_moment(iso_text: str):
    if not iso_text:
        return None
    try:
        moment = datetime.fromisoformat(str(iso_text).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment


def amount_value(order: AuthOrder) -> Decimal:
    try:
        return Decimal(order.payable_amount or order.amount or "0")
    except (InvalidOperation, TypeError):
        return Decimal("0")


class AuthRecordsWidget(QWidget):
    """授权记录看板：过滤条件 + 记录行，行首复选框勾选后可删除"""

    records_changed = Signal()

    COLUMNS = ("", "订单号", "下单时间", "激活时间", "时长", "订单状态", "金额")
    CHECK_COLUMN = 0

    def __init__(self, service: AutoAuthService, parent=None):
        super().__init__(parent)
        self._service = service
        self._orders: List[AuthOrder] = []
        self._visible_orders: List[AuthOrder] = []
        self._filling = False
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addLayout(self._create_filter_row())
        layout.addWidget(self._create_table())

    def _create_filter_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.select_all_box = CheckBox(self.tr("全选"), self)
        self.select_all_box.setTristate(True)
        self.select_all_box.clicked.connect(self._on_select_all_clicked)
        row.addWidget(self.select_all_box, 0)

        self.search_edit = SearchLineEdit(self)
        self.search_edit.setPlaceholderText(self.tr("搜索订单号 / 交易号 / 设备码"))
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setMinimumWidth(220)
        self.search_edit.textChanged.connect(self._apply_filters)
        self.search_edit.searchSignal.connect(lambda _: self._apply_filters())
        row.addWidget(self.search_edit, 2)

        self.status_combox = ComboBox(self)
        self.status_combox.addItem(self.tr("全部状态"), userData="")
        for value, text in STATUS_TEXTS.items():
            self.status_combox.addItem(self.tr(text), userData=value)
        self.status_combox.setMinimumWidth(120)
        self.status_combox.currentIndexChanged.connect(self._apply_filters)
        row.addWidget(self.status_combox, 0)

        self.range_combox = ComboBox(self)
        for text, days in TIME_RANGES:
            self.range_combox.addItem(self.tr(text), userData=days)
        self.range_combox.setMinimumWidth(110)
        self.range_combox.currentIndexChanged.connect(self._apply_filters)
        row.addWidget(self.range_combox, 0)

        self.refresh_btn = PushButton(self.tr("刷新"), self, FluentIcon.SYNC)
        self.refresh_btn.clicked.connect(self.refresh)
        row.addWidget(self.refresh_btn, 0)

        row.addStretch(1)

        self.delete_btn = PrimaryPushButton(self.tr("删除所选"), self, FluentIcon.DELETE)
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self._delete_checked)
        row.addWidget(self.delete_btn, 0)
        return row

    def _create_table(self) -> QWidget:
        self.table = TableWidget(self)
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels([self.tr(name) for name in self.COLUMNS])
        self.table.setBorderVisible(True)
        self.table.setBorderRadius(8)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setMinimumHeight(260)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemDoubleClicked.connect(self._copy_row)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(self.CHECK_COLUMN, QHeaderView.Fixed)
        self.table.setColumnWidth(self.CHECK_COLUMN, 44)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        return self.table

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    def refresh(self):
        try:
            self._orders = self._service.list_orders()
        except Exception as e:
            self._orders = []
            self._tip(self.tr("读取授权记录失败：{0}").format(e), InfoBarIcon.ERROR)
        self._apply_filters()

    def _apply_filters(self, *_):
        keyword = self.search_edit.text().strip().lower()
        status = self.status_combox.currentData() or ""
        days = int(self.range_combox.currentData() or 0)
        earliest = datetime.now(timezone.utc) - timedelta(days=days) if days else None

        visible = []
        for order in self._orders:
            if status and order.status != status:
                continue
            if earliest is not None:
                created = parse_moment(order.created_at)
                if created is None or created < earliest:
                    continue
            if keyword:
                haystack = " ".join([
                    order.order_id, order.transaction_id, order.machine_id,
                    order.license_id, status_text(order.status),
                ]).lower()
                if keyword not in haystack:
                    continue
            visible.append(order)

        self._visible_orders = visible
        self._fill_table(visible)
        self._refresh_selection_state()

    def _fill_table(self, orders: List[AuthOrder]):
        self._filling = True
        self.table.clearContents()
        self.table.setRowCount(len(orders))
        for row, order in enumerate(orders):
            check_item = QTableWidgetItem()
            check_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            check_item.setCheckState(Qt.Unchecked)
            check_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, self.CHECK_COLUMN, check_item)

            authorized_at = order.activated_at
            values = (
                order.order_id,
                local_time_text(order.created_at),
                local_time_text(authorized_at),
                duration_text(order.days),
                status_text(order.status),
                f"¥{amount_value(order):.2f}",
            )
            for offset, value in enumerate(values):
                column = offset + 1
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                if column in (2, 3, 5, 4, 6):
                    item.setTextAlignment(Qt.AlignCenter)
                if column == 5:
                    item.setForeground(QColor(STATUS_COLORS.get(order.status, TEXT_MAIN)))
                self.table.setItem(row, column, item)
        self._filling = False

    def _checked_rows(self) -> List[int]:
        rows = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.CHECK_COLUMN)
            if item is not None and item.checkState() == Qt.Checked:
                rows.append(row)
        return rows

    def _checked_orders(self) -> List[AuthOrder]:
        return [self._visible_orders[row] for row in self._checked_rows() if row < len(self._visible_orders)]

    def _on_item_changed(self, item: QTableWidgetItem):
        if self._filling or item.column() != self.CHECK_COLUMN:
            return
        self._refresh_selection_state()

    def _on_select_all_clicked(self, *_):
        checked = self.select_all_box.checkState() != Qt.Unchecked
        self._set_all_checked(checked)

    def _set_all_checked(self, checked: bool):
        self._filling = True
        state = Qt.Checked if checked else Qt.Unchecked
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.CHECK_COLUMN)
            if item is not None:
                item.setCheckState(state)
        self._filling = False
        self._refresh_selection_state()

    def _refresh_selection_state(self):
        total = self.table.rowCount()
        checked = len(self._checked_rows())
        self.delete_btn.setEnabled(checked > 0)
        self.delete_btn.setText(self.tr("删除所选") + (f" ({checked})" if checked else ""))

        self.select_all_box.blockSignals(True)
        if total == 0 or checked == 0:
            self.select_all_box.setCheckState(Qt.Unchecked)
        elif checked == total:
            self.select_all_box.setCheckState(Qt.Checked)
        else:
            self.select_all_box.setCheckState(Qt.PartiallyChecked)
        self.select_all_box.setEnabled(total > 0)
        self.select_all_box.blockSignals(False)

    def _copy_row(self, item: QTableWidgetItem):
        row = item.row()
        if not (0 <= row < len(self._visible_orders)):
            return
        QApplication.clipboard().setText(self._service.support_summary(self._visible_orders[row]))
        self._tip(self.tr("订单摘要已复制到剪贴板"))

    def _delete_checked(self):
        orders = self._checked_orders()
        if not orders:
            return
        activated = [o for o in orders if o.status == OrderStatus.ACTIVATED.value]
        content = self.tr("将删除 {0} 条授权记录，删除后无法在看板中查看（审计流水仍保留）。").format(len(orders))
        if activated:
            content += self.tr("\n其中 {0} 条为已激活订单，删除记录不会影响当前许可证的有效性。").format(len(activated))
        box = MessageBox(title=self.tr("删除授权记录"), content=content, parent=self.window())
        box.yesButton.setText(self.tr("删除"))
        box.cancelButton.setText(self.tr("取消"))
        if not box.exec():
            return
        deleted = self._service.delete_orders([o.order_id for o in orders])
        self.refresh()
        self.records_changed.emit()
        self._tip(self.tr("已删除 {0} 条授权记录").format(deleted))

    def _tip(self, content: str, icon=InfoBarIcon.SUCCESS):
        TeachingTip.create(
            target=self.refresh_btn,
            icon=icon,
            title=self.tr("授权记录"),
            content=content,
            isClosable=True,
            tailPosition=TeachingTipTailPosition.BOTTOM,
            duration=2500,
            parent=self.window(),
        )
