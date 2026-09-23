from datetime import datetime
from PySide6.QtCore import Qt, Signal, QTimer, QThread, QObject
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton, QStackedWidget, QLineEdit
from app.ui.library.qfluentwidgets import setFont, IndeterminateProgressRing
from app.license.auto_auth import AutoAuthService, AuthOrder, AuthStage, AuthProgress, OrderStatus
import app.library._machine_id as machine_id


PRIMARY = "#6366f1"
PRIMARY_HOVER = "#4f46e5"
PRIMARY_PRESSED = "#4338ca"
SUCCESS = "#10b981"
WARNING = "#f59e0b"
DANGER = "#ef4444"
CARD_BG = "#ffffff"
SUBTLE_BG = "#f8fafc"
HOVER_BG = "#f1f5f9"
TEXT_PRIMARY = "#1e293b"
TEXT_SECONDARY = "#64748b"
BORDER = "#e2e8f0"
RADIUS = 12
POLL_INTERVAL_MS = 20_000
AUTO_QUERY_TIMEOUT_MS = 10 * 60 * 1000  # 10 分钟后自动停止轮询
_BADGE_POLLING = (
    "QLabel { background: #fef3c7; color: #b45309;"
    " border-radius: 4px; padding: 2px 7px; }"
)
_BADGE_PAUSED = (
    "QLabel { background: #f1f5f9; color: #64748b;"
    " border-radius: 4px; padding: 2px 7px; }"
)
_BADGE_READY = (
    "QLabel { background: #d1fae5; color: #065f46;"
    " border-radius: 4px; padding: 2px 7px; }"
)
_TAB_ACTIVE = (
    f"QPushButton {{ background: {CARD_BG}; color: {PRIMARY};"
    f" border: none; border-radius: 8px; padding: 6px 10px; font-weight: 600; }}"
)
_TAB_INACTIVE = (
    f"QPushButton {{ background: transparent; color: {TEXT_SECONDARY};"
    f" border: none; border-radius: 8px; padding: 6px 10px; }}"
    f"QPushButton:hover {{ color: {TEXT_PRIMARY}; }}"
)


class _PollWorker(QObject):
    finished = Signal(object)

    def __init__(self, service: AutoAuthService, order: AuthOrder):
        super().__init__()
        self._service = service
        self._order = order

    def run(self):
        progress = self._service.poll(self._order)
        self.finished.emit(progress)


class FetchLicenseWidget(QWidget):
    license_activated = Signal(str)
    _STATE_QUERYING = "QUERYING"
    _STATE_STOPPED = "STOPPED"
    _STATE_FOUND = "FOUND"
    _PANEL_AUTO = 0
    _PANEL_MANUAL = 1
    _AUTO_QUERYING = 0
    _AUTO_STOPPED = 1
    _AUTO_FOUND = 2
    _AUTO_IDLE = 3

    def __init__(self, service: AutoAuthService, parent: QWidget | None = None):
        super().__init__(parent)
        self._service = service
        self._state = self._STATE_STOPPED
        self._current_order: AuthOrder | None = None
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._do_poll)
        self._auto_timeout_timer = QTimer(self)
        self._auto_timeout_timer.setSingleShot(True)
        self._auto_timeout_timer.setInterval(AUTO_QUERY_TIMEOUT_MS)
        self._auto_timeout_timer.timeout.connect(self._on_auto_query_timeout)
        self._poll_thread: QThread | None = None   # 当前正在跑的轮询线程
        self._poll_worker = None                    # 持有 worker 引用，防止被 GC
        self._manual_thread: QThread | None = None  # 手动查询子线程
        self._manual_worker = None                  # 持有 worker 引用，防止被 GC

        self._setup_ui()
        self._init_state()

    def start_polling(self, order: AuthOrder | None = None):
        if order is not None:
            self._current_order = order
        self._switch_to_auto_panel()
        self._restart_auto_query()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        card = QFrame(self)
        card.setObjectName("fetchLicenseCard")
        card.setStyleSheet(
            f"QFrame#fetchLicenseCard {{ background: {CARD_BG};"
            f" border: 1px solid {BORDER}; border-radius: {RADIUS}px; }}"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 22, 24, 18)
        card_layout.setSpacing(14)

        card_layout.addLayout(self._build_header())
        card_layout.addWidget(self._build_tab_bar())
        self._panel_stack = QStackedWidget(card)
        self._panel_stack.addWidget(self._build_auto_panel())
        self._panel_stack.addWidget(self._build_manual_panel())
        card_layout.addWidget(self._panel_stack)
        self._notify_warning = self._build_notify_warning()
        card_layout.addWidget(self._notify_warning)
        self._toast = self._build_toast()
        card_layout.addWidget(self._toast)
        card_layout.addLayout(self._build_footer())
        root.addWidget(card)

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        icon_label = QLabel("📄")
        setFont(icon_label, 18)
        icon_label.setFixedSize(32, 32)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet(f"QLabel {{ background: #eef2ff; border-radius: 8px; }}")
        row.addWidget(icon_label)

        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(2)

        title = QLabel(self.tr("获取授权文件"))
        setFont(title, 14, QFont.Bold)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        title_col.addWidget(title)

        sub = QLabel(self.tr("查询并下载关联本机的 .lic 授权凭证"))
        setFont(sub, 10)
        sub.setStyleSheet(f"color: {TEXT_SECONDARY};")
        title_col.addWidget(sub)

        row.addLayout(title_col)
        row.addStretch()

        self._badge = QLabel("POLLING")
        setFont(self._badge, 10, QFont.Bold)
        self._badge.setStyleSheet(_BADGE_PAUSED)
        row.addWidget(self._badge)
        self._set_badge_paused()
        return row

    def _build_tab_bar(self) -> QFrame:
        bar = QFrame(self)
        bar.setObjectName("fetchTabBar")
        bar.setStyleSheet(
            f"QFrame#fetchTabBar {{ background: {SUBTLE_BG};"
            f" border: 1px solid {BORDER}; border-radius: 10px; }}"
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._tab_auto = QPushButton("🔄 " + self.tr("自动查询模式"))
        setFont(self._tab_auto, 11)
        self._tab_auto.setCursor(Qt.PointingHandCursor)
        self._tab_auto.clicked.connect(self._switch_to_auto_panel)
        layout.addWidget(self._tab_auto)

        self._tab_manual = QPushButton("🔍 " + self.tr("手动凭证查询"))
        setFont(self._tab_manual, 11)
        self._tab_manual.setCursor(Qt.PointingHandCursor)
        self._tab_manual.clicked.connect(self._switch_to_manual_panel)
        layout.addWidget(self._tab_manual)

        self._set_tab_auto_active()
        return bar

    def _build_auto_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self._auto_stack = QStackedWidget(panel)
        self._auto_stack.addWidget(self._build_querying_state())
        self._auto_stack.addWidget(self._build_stopped_state())
        self._auto_stack.addWidget(self._build_found_state())
        self._auto_stack.addWidget(self._build_idle_state())
        layout.addWidget(self._auto_stack)

        self._btn_download = QPushButton("⬇️  " + self.tr("激活授权文件 (.lic)"))
        setFont(self._btn_download, 12, QFont.Bold)
        self._btn_download.setCursor(Qt.PointingHandCursor)
        self._btn_download.setMinimumHeight(42)
        self._btn_download.clicked.connect(self._activate_license)
        layout.addWidget(self._btn_download)
        self._set_download_disabled()

        note = QFrame(panel)
        note.setObjectName("fetchNote")
        note.setStyleSheet(
            "QFrame#fetchNote { background: #fffbeb; border: 1px solid #fde68a;"
            " border-radius: 8px; }"
        )
        note_layout = QHBoxLayout(note)
        note_layout.setContentsMargins(10, 8, 10, 8)
        note_layout.setSpacing(6)

        note_icon = QLabel("⚠️")
        setFont(note_icon, 12)
        note_layout.addWidget(note_icon, 0, Qt.AlignTop)

        note_text = QLabel(self.tr("提示：若超过 10 分钟仍未生成授权文件，可联系作者处理 (QQ群: 1080076113)，工作时间 9:00-23:30。"))
        setFont(note_text, 10)
        note_text.setWordWrap(True)
        note_text.setStyleSheet("color: #b45309;")
        note_layout.addWidget(note_text, 1)
        layout.addWidget(note)

        return panel

    def _build_querying_state(self) -> QWidget:
        w = QFrame(self)
        w.setObjectName("stateQuerying")
        w.setStyleSheet(
            "QFrame#stateQuerying { background: #eef2ff; border: 1px solid #c7d2fe;"
            " border-radius: 10px; }"
        )
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignCenter)

        spinner_row = QHBoxLayout()
        spinner_row.setAlignment(Qt.AlignCenter)
        self._spinner = IndeterminateProgressRing(w)
        self._spinner.setFixedSize(32, 32)
        self._spinner.setStrokeWidth(3)
        spinner_row.addWidget(self._spinner)
        layout.addLayout(spinner_row)

        title = QLabel(self.tr("正在自动循环查询授权文件..."))
        setFont(title, 12, QFont.Bold)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {PRIMARY_PRESSED};")
        layout.addWidget(title)

        sub = QLabel(self.tr("轮询服务端状态中..."))
        setFont(sub, 10)
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color: {PRIMARY};")
        layout.addWidget(sub)

        stop_btn = QPushButton("⏹  " + self.tr("停止自动查询"))
        setFont(stop_btn, 11)
        stop_btn.setCursor(Qt.PointingHandCursor)
        stop_btn.setFixedHeight(30)
        stop_btn.setStyleSheet(f"""
            QPushButton {{
                background: {CARD_BG}; color: {DANGER};
                border: 1px solid {BORDER}; border-radius: 6px;
                padding: 0 14px;
            }}
            QPushButton:hover {{ background: #fff1f2; border-color: {DANGER}; }}
            QPushButton:pressed {{ background: #fee2e2; }}
        """)
        stop_btn.clicked.connect(self._stop_auto_query)

        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignCenter)
        btn_row.addWidget(stop_btn)
        layout.addLayout(btn_row)
        return w

    def _build_stopped_state(self) -> QWidget:
        w = QFrame(self)
        w.setObjectName("stateStopped")
        w.setStyleSheet(
            f"QFrame#stateStopped {{ background: {SUBTLE_BG}; border: 1px solid {BORDER};"
            f" border-radius: 10px; }}"
        )
        layout = QHBoxLayout(w)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignVCenter)

        icon = QLabel("⏸️")
        setFont(icon, 20)
        icon.setFixedSize(36, 36)
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet("QLabel { background: #fef3c7; border-radius: 8px; }")
        layout.addWidget(icon, 0, Qt.AlignVCenter)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)
        title = QLabel(self.tr("已停止自动查询"))
        setFont(title, 12, QFont.Bold)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        text_col.addWidget(title)
        sub = QLabel(self.tr("可切换到「手动凭证查询」进行手动检索"))
        setFont(sub, 10)
        sub.setStyleSheet(f"color: {TEXT_SECONDARY};")
        text_col.addWidget(sub)

        text_wrapper = QWidget()
        text_wrapper.setLayout(text_col)
        layout.addWidget(text_wrapper, 1, Qt.AlignVCenter)

        restart_btn = QPushButton(self.tr("重新开启"))
        setFont(restart_btn, 11)
        restart_btn.setCursor(Qt.PointingHandCursor)
        restart_btn.setFixedHeight(28)
        restart_btn.setStyleSheet(f"""
            QPushButton {{
                background: #eef2ff; color: {PRIMARY};
                border: 1px solid #c7d2fe; border-radius: 6px;
                padding: 0 12px;
            }}
            QPushButton:hover {{ background: #e0e7ff; }}
            QPushButton:pressed {{ background: #c7d2fe; }}
        """)
        restart_btn.clicked.connect(self._restart_auto_query)
        layout.addWidget(restart_btn, 0, Qt.AlignVCenter)
        return w

    def _build_idle_state(self) -> QWidget:
        w = QFrame(self)
        w.setObjectName("stateIdle")
        w.setStyleSheet(
            f"QFrame#stateIdle {{ background: {SUBTLE_BG}; border: 1px solid {BORDER};"
            f" border-radius: 10px; }}"
        )
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignCenter)

        icon = QLabel("🕐")
        setFont(icon, 28)
        icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon)

        title = QLabel(self.tr("等待授权申请完成"))
        setFont(title, 12, QFont.Bold)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        layout.addWidget(title)

        sub = QLabel(self.tr("请先在左侧「自动授权服务」中发起订单并完成支付，\n授权文件将自动出现在这里。"))
        setFont(sub, 10)
        sub.setWordWrap(True)
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color: {TEXT_SECONDARY};")
        layout.addWidget(sub)
        return w

    def _build_found_state(self) -> QWidget:
        w = QFrame(self)
        w.setObjectName("stateFound")
        w.setStyleSheet(
            "QFrame#stateFound { background: #f0fdf4; border: 1px solid #bbf7d0;"
            " border-radius: 10px; }"
        )
        layout = QVBoxLayout(w)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        header_row.setSpacing(6)
        check_icon = QLabel("✅")
        setFont(check_icon, 12)
        header_row.addWidget(check_icon)
        header_title = QLabel(self.tr("已匹配待绑定订单信息"))
        setFont(header_title, 12, QFont.Bold)
        header_title.setStyleSheet("color: #14532d;")
        header_row.addWidget(header_title)
        header_row.addStretch()
        ready_badge = QLabel(self.tr("已就绪"))
        setFont(ready_badge, 10, QFont.Bold)
        ready_badge.setStyleSheet(
            "QLabel { background: #bbf7d0; color: #14532d;"
            " border-radius: 4px; padding: 1px 6px; }"
        )
        header_row.addWidget(ready_badge)
        layout.addLayout(header_row)

        sep = QFrame(w)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #bbf7d0; border: none;")
        layout.addWidget(sep)

        self._found_order_id_label = self._make_detail_row(layout, self.tr("订单单号："), "—")
        self._found_hwid_label = self._make_detail_row(
            layout,
            self.tr("绑定设备号："),
            machine_id.get_machine_id_display()[:20] + "…"
            if len(machine_id.get_machine_id_display()) > 20
            else machine_id.get_machine_id_display(),
        )
        self._found_spec_label = self._make_detail_row(layout, self.tr("授权规格："), "—")
        self._found_time_label = self._make_detail_row(layout, self.tr("生成时间："), self.tr("—"))
        return w

    @staticmethod
    def _make_detail_row(parent_layout: QVBoxLayout, key: str, value: str) -> QLabel:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        key_label = QLabel(key)
        setFont(key_label, 10)
        key_label.setStyleSheet("color: #64748b;")
        row.addWidget(key_label)
        row.addStretch()
        val_label = QLabel(value)
        setFont(val_label, 10, QFont.Bold)
        val_label.setStyleSheet("color: #1e293b; font-family: monospace;")
        val_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(val_label)
        parent_layout.addLayout(row)
        return val_label

    def _build_manual_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        label = QLabel(self.tr("手动输入订单号 / 交易单号"))
        setFont(label, 11, QFont.Bold)
        label.setStyleSheet(f"color: {TEXT_PRIMARY};")
        layout.addWidget(label)

        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(6)

        self._manual_input = QLineEdit(panel)
        self._manual_input.setPlaceholderText(self.tr("例如: PT260921-XXXX-YYYY"))
        self._manual_input.setFixedHeight(36)
        self._manual_input.setStyleSheet(f"""
            QLineEdit {{
                background: {SUBTLE_BG}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER}; border-radius: 8px;
                padding: 0 10px; font-size: 12px;
            }}
            QLineEdit:focus {{
                border-color: {PRIMARY};
                background: {CARD_BG};
            }}
        """)
        self._manual_input.returnPressed.connect(self._execute_manual_search)
        input_row.addWidget(self._manual_input, 1)

        self._manual_search_btn = QPushButton(self.tr("查询"))
        setFont(self._manual_search_btn, 11, QFont.Bold)
        self._manual_search_btn.setFixedHeight(36)
        self._manual_search_btn.setCursor(Qt.PointingHandCursor)
        self._manual_search_btn.setStyleSheet(f"""
            QPushButton {{
                background: {PRIMARY}; color: white;
                border: none; border-radius: 8px;
                padding: 0 18px;
            }}
            QPushButton:hover {{ background: {PRIMARY_HOVER}; }}
            QPushButton:pressed {{ background: {PRIMARY_PRESSED}; }}
        """)
        self._manual_search_btn.clicked.connect(self._execute_manual_search)
        input_row.addWidget(self._manual_search_btn)
        layout.addLayout(input_row)

        self._manual_result_stack = QStackedWidget(panel)
        self._manual_result_stack.addWidget(self._build_manual_idle())
        self._manual_result_stack.addWidget(self._build_manual_searching())
        self._manual_result_stack.addWidget(self._build_manual_found())
        self._manual_result_stack.addWidget(self._build_manual_not_found())
        layout.addWidget(self._manual_result_stack)

        self._manual_btn_download = QPushButton("⬇️  " + self.tr("激活授权文件 (.lic)"))
        setFont(self._manual_btn_download, 12, QFont.Bold)
        self._manual_btn_download.setCursor(Qt.PointingHandCursor)
        self._manual_btn_download.setMinimumHeight(42)
        self._manual_btn_download.clicked.connect(self._activate_license)
        layout.addWidget(self._manual_btn_download)
        self._set_manual_download_disabled()

        layout.addStretch()
        return panel

    def _build_manual_idle(self) -> QFrame:
        w = QFrame(self)
        w.setObjectName("manualIdle")
        w.setStyleSheet(
            f"QFrame#manualIdle {{ background: {SUBTLE_BG};"
            f" border: 1px dashed {BORDER}; border-radius: 10px; }}"
        )
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 18, 14, 18)
        lay.setAlignment(Qt.AlignCenter)

        icon = QLabel("🔍")
        setFont(icon, 20)
        icon.setAlignment(Qt.AlignCenter)
        lay.addWidget(icon)

        hint = QLabel(self.tr("输入订单号并点击「查询」即可直接绑定获取文件"))
        setFont(hint, 11)
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(f"color: {TEXT_SECONDARY};")
        lay.addWidget(hint)
        return w

    def _build_manual_searching(self) -> QFrame:
        w = QFrame(self)
        w.setObjectName("manualSearching")
        w.setStyleSheet(
            "QFrame#manualSearching { background: #eef2ff; border: 1px solid #c7d2fe;"
            " border-radius: 10px; }"
        )
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 18, 14, 18)
        lay.setSpacing(10)
        lay.setAlignment(Qt.AlignCenter)

        spinner_row = QHBoxLayout()
        spinner_row.setAlignment(Qt.AlignCenter)
        self._manual_spinner = IndeterminateProgressRing(w)
        self._manual_spinner.setFixedSize(32, 32)
        self._manual_spinner.setStrokeWidth(3)
        spinner_row.addWidget(self._manual_spinner)
        lay.addLayout(spinner_row)

        title = QLabel(self.tr("正在查询授权文件…"))
        setFont(title, 12, QFont.Bold)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {PRIMARY_PRESSED};")
        lay.addWidget(title)

        sub = QLabel(self.tr("正在向服务端核验订单状态，请稍候"))
        setFont(sub, 10)
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color: {PRIMARY};")
        lay.addWidget(sub)
        return w

    def _build_manual_found(self) -> QFrame:
        w = QFrame(self)
        w.setObjectName("manualFound")
        w.setStyleSheet(
            "QFrame#manualFound { background: #f0fdf4; border: 1px solid #bbf7d0;"
            " border-radius: 10px; }"
        )
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(6)
        chk = QLabel("✅")
        setFont(chk, 12)
        header.addWidget(chk)
        hdr_title = QLabel(self.tr("已匹配待绑定订单信息"))
        setFont(hdr_title, 12, QFont.Bold)
        hdr_title.setStyleSheet("color: #14532d;")
        header.addWidget(hdr_title)
        header.addStretch()
        ready_badge = QLabel(self.tr("已就绪"))
        setFont(ready_badge, 10, QFont.Bold)
        ready_badge.setStyleSheet(
            "QLabel { background: #bbf7d0; color: #14532d;"
            " border-radius: 4px; padding: 1px 6px; }"
        )
        header.addWidget(ready_badge)
        lay.addLayout(header)

        sep = QFrame(w)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #bbf7d0; border: none;")
        lay.addWidget(sep)

        self._mfound_order_id = self._make_detail_row(lay, self.tr("订单单号："), "—")
        self._mfound_hwid = self._make_detail_row(
            lay,
            self.tr("绑定设备号："),
            machine_id.get_machine_id_display()[:20] + "…"
            if len(machine_id.get_machine_id_display()) > 20
            else machine_id.get_machine_id_display(),
        )
        self._mfound_spec = self._make_detail_row(lay, self.tr("授权规格："), "—")
        self._mfound_time = self._make_detail_row(lay, self.tr("生成时间："), "—")
        return w

    def _build_manual_not_found(self) -> QFrame:
        w = QFrame(self)
        w.setObjectName("manualNotFound")
        w.setStyleSheet(
            "QFrame#manualNotFound { background: #fefce8; border: 1px solid #fde68a;"
            " border-radius: 10px; }"
        )
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 16, 14, 16)
        lay.setSpacing(8)
        lay.setAlignment(Qt.AlignCenter)

        icon = QLabel("⏳")
        setFont(icon, 20)
        icon.setAlignment(Qt.AlignCenter)
        lay.addWidget(icon)

        self._manual_not_found_title = QLabel(self.tr("暂未查询到授权文件"))
        setFont(self._manual_not_found_title, 12, QFont.Bold)
        self._manual_not_found_title.setAlignment(Qt.AlignCenter)
        self._manual_not_found_title.setStyleSheet("color: #92400e;")
        lay.addWidget(self._manual_not_found_title)

        self._manual_not_found_detail = QLabel("")
        setFont(self._manual_not_found_detail, 10)
        self._manual_not_found_detail.setWordWrap(False)
        self._manual_not_found_detail.setAlignment(Qt.AlignCenter)
        self._manual_not_found_detail.setStyleSheet("color: #b45309;")
        lay.addWidget(self._manual_not_found_detail)
        return w

    def _build_toast(self) -> QLabel:
        toast = QLabel("")
        setFont(toast, 11)
        toast.setAlignment(Qt.AlignCenter)
        toast.setWordWrap(True)
        toast.setStyleSheet(
            "QLabel { background: #1e293b; color: white;"
            " border-radius: 8px; padding: 8px 12px; }"
        )
        toast.hide()
        return toast

    def _build_notify_warning(self) -> QFrame:
        banner = QFrame(self)
        banner.setObjectName("notifyWarning")
        banner.setStyleSheet(
            "QFrame#notifyWarning { background: #fff1f2;"
            " border: 1px solid #fecaca; border-radius: 8px; }"
        )
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        icon = QLabel("❗")
        setFont(icon, 13)
        icon.setAlignment(Qt.AlignTop)
        layout.addWidget(icon, 0, Qt.AlignTop)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(3)

        self._notify_warning_title = QLabel(self.tr("⚠️ 通知发送失败，作者可能未收到付款信息"))
        setFont(self._notify_warning_title, 11, QFont.Bold)
        self._notify_warning_title.setStyleSheet("color: #b91c1c;")
        self._notify_warning_title.setWordWrap(True)
        text_col.addWidget(self._notify_warning_title)

        self._notify_warning_detail = QLabel()
        setFont(self._notify_warning_detail, 10)
        self._notify_warning_detail.setStyleSheet("color: #dc2626;")
        self._notify_warning_detail.setWordWrap(True)
        text_col.addWidget(self._notify_warning_detail)

        hint = QLabel(self.tr("请联系作者并发送订单号（已复制到剪贴板） QQ群: 1080076113。"))
        setFont(hint, 10)
        hint.setStyleSheet("color: #7f1d1d;")
        hint.setWordWrap(True)
        text_col.addWidget(hint)

        layout.addLayout(text_col, 1)

        dismiss_btn = QPushButton(self.tr("知道了"))
        setFont(dismiss_btn, 10)
        dismiss_btn.setFixedHeight(26)
        dismiss_btn.setCursor(Qt.PointingHandCursor)
        dismiss_btn.setStyleSheet(f"""
            QPushButton {{
                background: #fee2e2; color: #b91c1c;
                border: 1px solid #fca5a5; border-radius: 5px;
                padding: 0 10px;
            }}
            QPushButton:hover {{ background: #fecaca; }}
            QPushButton:pressed {{ background: #fca5a5; }}
        """)
        dismiss_btn.clicked.connect(banner.hide)
        layout.addWidget(dismiss_btn, 0, Qt.AlignTop)

        banner.hide()
        return banner

    def show_notify_warning(self, error: str = ""):
        if error:
            self._notify_warning_detail.setText(self.tr(f"失败原因：{error}"))
            self._notify_warning_detail.show()
        else:
            self._notify_warning_detail.hide()
        self._notify_warning.show()

    def _build_footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        lock_label = QLabel("🔒 " + self.tr("绑定本机硬件 ID"))
        setFont(lock_label, 10)
        lock_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        row.addWidget(lock_label)
        row.addStretch()
        return row

    def _init_state(self):
        last_issued = self._find_last_issued_order()
        if last_issued is not None:
            self._current_order = last_issued
            self._fill_found_labels(last_issued)
            self._state = self._STATE_FOUND
            self._auto_stack.setCurrentIndex(self._AUTO_FOUND)
            self._set_badge_ready()
            self._set_download_enabled()
        else:
            self._state = self._STATE_STOPPED
            self._auto_stack.setCurrentIndex(self._AUTO_IDLE)
            self._set_badge_paused()
            self._set_download_disabled()

    def _find_last_issued_order(self) -> "AuthOrder | None":
        for order in self._service.store.list_orders():
            if order.status in (OrderStatus.ISSUED.value, OrderStatus.ACTIVATED.value):
                return order
        return None

    def _fill_found_labels(self, order: "AuthOrder"):
        self._found_order_id_label.setText(order.order_id)
        tier_name = order.tier_key
        for tier in self._service.tiers:
            if tier.key == order.tier_key:
                tier_name = tier.name
                break
        self._found_spec_label.setText(f"{order.days} 天  ·  {tier_name}")
        ts_raw = order.issued_at or order.activated_at or order.created_at
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            ts_str = ts.strftime("%Y-%m-%d %H:%M")
        except Exception:
            ts_str = ts_raw
        self._found_time_label.setText(ts_str)

    def _restart_auto_query(self):
        self._state = self._STATE_QUERYING
        self._poll_timer.start()
        self._auto_timeout_timer.start()
        self._auto_stack.setCurrentIndex(self._AUTO_QUERYING)
        self._set_badge_polling()
        self._set_download_disabled()
        QTimer.singleShot(200, self._do_poll)  # 立即触发一次

    def _stop_auto_query(self):
        self._state = self._STATE_STOPPED
        self._poll_timer.stop()
        self._auto_timeout_timer.stop()
        self._auto_stack.setCurrentIndex(self._AUTO_STOPPED)
        self._set_badge_paused()
        self._set_download_disabled()
        self._show_toast(self.tr("已停止自动查询。可切换到「手动凭证查询」进行检索。"))

    def _on_auto_query_timeout(self):
        if self._state != self._STATE_QUERYING:
            return
        self._state = self._STATE_STOPPED
        self._poll_timer.stop()
        self._auto_stack.setCurrentIndex(self._AUTO_STOPPED)
        self._set_badge_paused()
        self._set_download_disabled()
        self._show_toast(self.tr("自动查询已超时，请切换到「手动凭证查询」进行检索。"), duration_ms=6000)

    def _mark_found(self, order: AuthOrder):
        if self._state == self._STATE_FOUND:
            return
        self._state = self._STATE_FOUND
        self._current_order = order
        self._poll_timer.stop()
        self._auto_timeout_timer.stop()

        self._fill_found_labels(order)

        self._auto_stack.setCurrentIndex(self._AUTO_FOUND)
        self._set_badge_ready()
        self._set_download_enabled()
        self._show_toast(self.tr("✨ 成功获取授权文件！可以点击按钮进行激活。"))

    def _do_poll(self):
        if self._state != self._STATE_QUERYING:
            return
        if self._poll_thread is not None and self._poll_thread.isRunning():
            return

        order = self._current_order
        if order is None:
            order = self._service.store.find_open_order()
        if order is None:
            return

        worker = _PollWorker(self._service, order)
        thread = QThread(self)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_poll_done)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_poll_thread)

        self._poll_worker = worker   # 持有引用，防止子线程启动前被 GC
        self._poll_thread = thread
        thread.start()

    def _clear_poll_thread(self):
        self._poll_worker = None
        self._poll_thread = None

    def _on_poll_done(self, progress: AuthProgress):
        if self._state != self._STATE_QUERYING:
            return
        order = self._current_order
        if progress.order is not None:
            self._current_order = progress.order
            order = progress.order

        if progress.stage == AuthStage.ISSUED:
            self._mark_found(order)
        elif progress.stage in (AuthStage.TIMEOUT, AuthStage.FAILED):
            self._poll_timer.stop()
            self._state = self._STATE_STOPPED
            self._auto_stack.setCurrentIndex(self._AUTO_STOPPED)
            self._set_badge_paused()
            self._show_toast(progress.message or self.tr("查询失败，请联系作者处理 (QQ群: 1080076113))"))

    def _execute_manual_search(self):
        order_id = self._manual_input.text().strip()
        if not order_id:
            self._show_toast(self.tr("请先输入有效的订单号！"))
            return
        if self._manual_thread is not None and self._manual_thread.isRunning():
            return
        order = self._service.store.get(order_id)
        if order is None:
            self._manual_result_stack.setCurrentIndex(3)
            self._manual_not_found_title.setText(self.tr("未找到该订单"))
            self._manual_not_found_detail.setText(self.tr(f"订单号 {order_id} 不存在，请检查是否输入有误。"))
            return
        self._set_manual_searching(True)
        self._manual_result_stack.setCurrentIndex(1)

        worker = _PollWorker(self._service, order)
        thread = QThread(self)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_manual_poll_done)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_manual_thread)

        self._manual_worker = worker  # 持有引用，防止子线程启动前被 GC
        self._manual_thread = thread
        thread.start()

    def _clear_manual_thread(self):
        self._manual_worker = None
        self._manual_thread = None

    def _set_manual_searching(self, searching: bool):
        self._manual_search_btn.setEnabled(not searching)
        self._manual_input.setEnabled(not searching)
        if searching:
            self._manual_search_btn.setText(self.tr("查询中…"))
            self._manual_search_btn.setStyleSheet(f"""
                QPushButton {{
                    background: #cbd5e1; color: #94a3b8;
                    border: none; border-radius: 8px;
                    padding: 0 18px;
                }}
            """)
        else:
            self._manual_search_btn.setText(self.tr("查询"))
            self._manual_search_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {PRIMARY}; color: white;
                    border: none; border-radius: 8px;
                    padding: 0 18px;
                }}
                QPushButton:hover {{ background: {PRIMARY_HOVER}; }}
                QPushButton:pressed {{ background: {PRIMARY_PRESSED}; }}
            """)

    def _on_manual_poll_done(self, progress: AuthProgress):
        in_manual_panel = self._panel_stack.currentIndex() == self._PANEL_MANUAL
        if in_manual_panel:
            self._set_manual_searching(False)
        order = progress.order or self._current_order
        if progress.stage == AuthStage.ISSUED:
            if in_manual_panel:
                if order:
                    self._fill_manual_found_labels(order)
                self._manual_result_stack.setCurrentIndex(2)
                self._set_manual_download_enabled()
                if self._state != self._STATE_QUERYING:
                    self._mark_found(order)
            else:
                if order:
                    self._current_order = order
        else:
            if in_manual_panel:
                self._manual_result_stack.setCurrentIndex(3)
                self._set_manual_download_disabled()
                self._manual_not_found_title.setText(self.tr("暂未查询到授权文件"))
                detail = progress.message or self.tr("服务端尚未生成授权文件")
                self._manual_not_found_detail.setText(detail)

    def _fill_manual_found_labels(self, order: AuthOrder):
        self._mfound_order_id.setText(order.order_id)
        tier_name = order.tier_key
        for tier in self._service.tiers:
            if tier.key == order.tier_key:
                tier_name = tier.name
                break
        self._mfound_spec.setText(f"{order.days} 天  ·  {tier_name}")
        ts_raw = order.issued_at or order.activated_at or order.created_at
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            ts_str = ts.strftime("%Y-%m-%d %H:%M")
        except Exception:
            ts_str = ts_raw
        self._mfound_time.setText(ts_str)

    def _activate_license(self):
        if self._state != self._STATE_FOUND:
            return
        order = self._current_order
        if order is None:
            return
        license_path = getattr(order, "license_path", "")
        if license_path:
            self.license_activated.emit(license_path)
            self._service.update_license_activated(order=order)
            self._show_toast(self.tr("许可证已激活！"))
        else:
            self._show_toast(self.tr("未找到授权文件路径，请联系开发者。"))

    def _set_download_enabled(self):
        self._btn_download.setEnabled(True)
        self._btn_download.setText("⬇️  " + self.tr("激活授权文件 (.lic)"))
        self._btn_download.setStyleSheet(f"""
            QPushButton {{
                background: {PRIMARY}; color: white;
                border: none; border-radius: 10px;
            }}
            QPushButton:hover {{ background: {PRIMARY_HOVER}; }}
            QPushButton:pressed {{ background: {PRIMARY_PRESSED}; }}
        """)

    def _set_download_disabled(self):
        self._btn_download.setEnabled(False)
        self._btn_download.setText("⬇️  " + self.tr("激活授权文件 (尚未就绪)"))
        self._btn_download.setStyleSheet(f"""
            QPushButton {{
                background: #cbd5e1; color: #94a3b8;
                border: none; border-radius: 10px;
            }}
        """)

    def _set_manual_download_enabled(self):
        self._manual_btn_download.setEnabled(True)
        self._manual_btn_download.setText("⬇️  " + self.tr("激活授权文件 (.lic)"))
        self._manual_btn_download.setStyleSheet(f"""
            QPushButton {{
                background: {PRIMARY}; color: white;
                border: none; border-radius: 10px;
            }}
            QPushButton:hover {{ background: {PRIMARY_HOVER}; }}
            QPushButton:pressed {{ background: {PRIMARY_PRESSED}; }}
        """)

    def _set_manual_download_disabled(self):
        self._manual_btn_download.setEnabled(False)
        self._manual_btn_download.setText("⬇️  " + self.tr("激活授权文件 (尚未就绪)"))
        self._manual_btn_download.setStyleSheet(f"""
            QPushButton {{
                background: #cbd5e1; color: #94a3b8;
                border: none; border-radius: 10px;
            }}
        """)

    def _set_badge_polling(self):
        self._badge.setText("POLLING")
        self._badge.setStyleSheet(_BADGE_POLLING)

    def _set_badge_paused(self):
        self._badge.setText("PAUSED")
        self._badge.setStyleSheet(_BADGE_PAUSED)

    def _set_badge_ready(self):
        self._badge.setText("READY")
        self._badge.setStyleSheet(_BADGE_READY)

    def _set_tab_auto_active(self):
        self._tab_auto.setStyleSheet(_TAB_ACTIVE)
        self._tab_manual.setStyleSheet(_TAB_INACTIVE)

    def _set_tab_manual_active(self):
        self._tab_manual.setStyleSheet(_TAB_ACTIVE)
        self._tab_auto.setStyleSheet(_TAB_INACTIVE)

    def _switch_to_auto_panel(self):
        self._panel_stack.setCurrentIndex(self._PANEL_AUTO)
        self._set_tab_auto_active()

    def _switch_to_manual_panel(self):
        if self._state == self._STATE_QUERYING:
            self._stop_auto_query()
        self._panel_stack.setCurrentIndex(self._PANEL_MANUAL)
        self._set_tab_manual_active()

    def _show_toast(self, msg: str, duration_ms: int = 3500):
        self._toast.setText(msg)
        self._toast.show()
        QTimer.singleShot(duration_ms, self._toast.hide)
