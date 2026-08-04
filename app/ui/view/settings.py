import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from PySide6.QtCore import Qt, Signal, QRunnable, QObject, QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import(
    QHBoxLayout, QWidget, QVBoxLayout, QLabel, QFrame, QLineEdit, QPushButton, QFileDialog,
    QSizePolicy, QDialog, QProgressBar, QTextEdit, QButtonGroup
)
from PySide6.QtGui import QFont, QPainter, QPen, QColor
from huggingface_hub import snapshot_download

from app.ui.library.qfluentwidgets import(
    setFont, ScrollArea, TeachingTip, InfoBarIcon, TeachingTipTailPosition, FluentIcon,
    ComboBox, Theme, MessageBox
)
from app.ui.widgets.gradient_header_widget import GradientHeader
from app.ui.widgets.custom_card_group_widget import CustomCardGroupWidget, CustomGroupBox
from app.ui.widgets.toggle_switch_widget import ToggleSwitch
from app.ui.library.qframelesswindow.titlebar import CloseButton
from app.ui.common.config import cfg, Language
from app.controllers.task_manager import InternalTaskManager
from app.ui.common.utils import global_backend_info_cache
from app.utils.logger import get_log_manager
from app.utils.logger.decorators import log_exception, log_function_call


def detect_gpu_available() -> bool:
    status, _ = global_backend_info_cache.get()
    return "GPU" in status


HF_REPO_ID = "SmailSnail/PowerToolsEnc"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

models_deps_urls = {
    "visible_watermark_removal": {
        "cpu": {"path": "CPU/visible_watermark_removal", "sha256": None},
        "gpu": {"path": "GPU/visible_watermark_removal", "sha256": None},
    },
    "blind_watermark_addition": {
        "cpu": {"path": "CPU/blind_watermark_addition", "sha256": None},
        "gpu": {"path": "GPU/blind_watermark_addition", "sha256": None},
    },
    "ocr": {
        "cpu": {"path": "CPU/ocr", "sha256": None},
        "gpu": {"path": "GPU/ocr", "sha256": None},
    },
    "segment": {
        "cpu": {"path": "CPU/segment", "sha256": None},
        "gpu": {"path": "GPU/segment", "sha256": None},
    },
    "video_inpainting": {
        "cpu": {"path": "CPU/video_inpainting", "sha256": None},
        "gpu": {"path": "GPU/video_inpainting", "sha256": None},
    },
    "tracker": {
        "cpu": {"path": "CPU/tracker", "sha256": None},
        "gpu": {"path": "GPU/tracker", "sha256": None},
    },
    "image_edit": {
        "cpu": {"path": "CPU/image_edit", "sha256": None},
        "gpu": {"path": "GPU/image_edit", "sha256": None},
    }
}

model_estimated_sizes = {
    "gpu": {
        "blind_watermark_addition": "1.08 GB",
        "visible_watermark_removal": "2.48 GB",
        "segment": "1.69 GB",
        "ocr": "217 MB",
        "video_inpainting": "183 MB",
        "tracker": "124 MB",
        "image_edit": "3.20 GB"
    },
    "cpu": {
        "blind_watermark_addition": "1.08 GB",
        "visible_watermark_removal": "2.58 GB",
        "segment": "899 MB",
        "ocr": "217 MB",
        "video_inpainting": "261 MB",
        "tracker": "124 MB",
        "image_edit": "3.20 GB"
    }
}


theme_map = {
    "浅色": Theme.LIGHT.value,
    "深色": Theme.DARK.value,
    Theme.LIGHT.value: "浅色",
    Theme.DARK.value: "深色"
}
language_map = {
    "简体中文": Language.CHINESE_SIMPLIFIED,
    "英语": Language.ENGLISH,
    Language.CHINESE_SIMPLIFIED.value.name(): "简体中文",
    Language.ENGLISH.value.name(): "英语"
}
logger = logging.getLogger("UI")

class WorkerSignals(QObject):
    progress = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)


class InitWorker(QRunnable):
    def __init__(self, task_name: str, variant: str = "cpu", use_mirror: bool = False, parent: QObject = None):
        super().__init__()
        self.task_name = task_name
        self.variant = variant
        self.use_mirror = use_mirror
        self.signals = WorkerSignals(parent=parent)
        self.cancelled = False
        self.deps_path = cfg.get(cfg.localAIModelDeps)

    def cancel(self):
        self.cancelled = True

    def _clear_sys_path(self):
        local_settings = cfg.get_local_settings()
        if not local_settings:
            local_deps_path = self.deps_path
        else:
            local_deps_path = local_settings["LocalAISettings"]["LocalAIModelDeps"]
        if local_deps_path != self.deps_path:
            if local_deps_path in sys.path:
                sys.path.remove(local_deps_path)
                logger.info(f"Remove old deps path {local_deps_path} from sys.path success")
                return local_deps_path
        return ""

    def _sha256_of_file(self, path: str):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(blk)
        return h.hexdigest()
    
    def _extract_if_needed(self, file_path: str, output_dir: str):
        lower = file_path.lower()
        if lower.endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as zf:
                zf.extractall(output_dir)
        elif lower.endswith(".tar.gz") or lower.endswith(".tgz"):
            with tarfile.open(file_path, "r:gz") as tf:
                tf.extractall(output_dir)
        return output_dir
    
    @log_function_call(logger=logging.getLogger("UI"), level=logging.INFO)
    def _download_module(self):
        logger.info(f"start download {self.task_name} ({self.variant}) module, mirror={self.use_mirror}.")
        model_info = models_deps_urls[self.task_name][self.variant]
        dir_path = model_info["path"]
        if not dir_path:
            raise Exception(f"{self.task_name} ({self.variant}) download path not exist.")

        endpoint = HF_MIRROR_ENDPOINT if self.use_mirror else None
        variant_deps_path = os.path.join(self.deps_path, self.variant)
        os.makedirs(variant_deps_path, exist_ok=True)

        self.signals.progress.emit(f"正在下载: {self.task_name} ({self.variant})...")
        try:
            snapshot_download(
                repo_id=HF_REPO_ID,
                allow_patterns=f"{dir_path}/**",
                local_dir=variant_deps_path,
                endpoint=endpoint,
            )
        except Exception:
            cache_dir = os.path.join(variant_deps_path, ".cache")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            CPU_dir = os.path.join(variant_deps_path, "CPU")
            if os.path.exists(CPU_dir):
                shutil.rmtree(CPU_dir)
            GPU_dir = os.path.join(variant_deps_path, "GPU")
            if os.path.exists(GPU_dir):
                shutil.rmtree(GPU_dir)
            raise Exception(f"Download {dir_path} failed")

        for item in os.listdir(variant_deps_path):
            path = os.path.join(variant_deps_path, item)
            if item not in ["CPU", "GPU"]:
                continue
            for tmp_item in os.listdir(path):
                src_path = os.path.join(path, tmp_item)
                dst_path = variant_deps_path
                shutil.move(src_path, dst_path)
            shutil.rmtree(path)
        cache_dir = os.path.join(variant_deps_path, ".cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        logger.info(f"download {self.task_name} ({self.variant}) module success.")

    def _init_model(self):
        need_download = False
        try:
            self._valid_model()
        except Exception:
            need_download = True

        if not need_download:
            return
        self._download_module()

    def _valid_model(self):
        current_task_deps_path = os.path.join(self.deps_path, self.variant, self.task_name)
        if not os.path.exists(current_task_deps_path):
            raise Exception(f"{self.task_name} ({self.variant}) deps valid failed for {current_task_deps_path} not exist.")
        if not os.listdir(current_task_deps_path):
            raise Exception(f"{self.task_name} ({self.variant}) deps valid failed for {current_task_deps_path} is empty.")

    @log_exception(logger=logging.getLogger("UI"), reraise=True, log_args=True)
    def _step(self, task_step: str, msg: str):
        if self.cancelled:
            raise RuntimeError("初始化已被用户取消")
        self.signals.progress.emit(msg)
        if task_step == "init-model":
            try:
                self._init_model()
            except Exception:
                raise RuntimeError("初始化本地模型失败")
        elif task_step == "valid-model":
            try:
                self._valid_model()
            except Exception:
                raise RuntimeError("验证算法环境失败")
        else:
            raise Exception(f"不支持的任务流程 {task_step}")

    def run(self):
        try:
            old_deps_path = self._clear_sys_path()

            self._step(task_step="init-model", msg="正在初始化本地模型…")
            logger.info(f"Init local module {self.task_name} success.")

            self._step(task_step="valid-model", msg="正在验证算法环境…")
            logger.info(f"Vaild {self.task_name} module success.")

            # 删除旧路径下的环境依赖
            if old_deps_path and os.path.exists(old_deps_path):
                shutil.rmtree(old_deps_path)
                logger.info(f"Clear {old_deps_path} success.")

            self.signals.progress.emit("环境初始化成功")
            self.signals.finished.emit(True, "")

        except Exception as e:
            self.signals.finished.emit(False, str(e))

class ChevronButton(QWidget):
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(28, 28)
        self.setCursor(Qt.PointingHandCursor)
        self._rotated = False
        self._color = QColor("#9ca3af")
        self._hover = False

    def set_rotated(self, rotated: bool):
        self._rotated = rotated
        self.update()

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()

    def mousePressEvent(self, event):
        self.clicked.emit()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        color = QColor("#4b5563") if self._hover else QColor("#9ca3af")
        pen = QPen(color, 2.0)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        # Draw chevron path centered in widget
        cx, cy = self.width() / 2, self.height() / 2
        if self._rotated:
            # Up chevron: ∧
            p.drawLine(int(cx - 5), int(cy + 2), int(cx), int(cy - 3))
            p.drawLine(int(cx), int(cy - 3), int(cx + 5), int(cy + 2))
        else:
            # Down chevron: ∨
            p.drawLine(int(cx - 5), int(cy - 2), int(cx), int(cy + 3))
            p.drawLine(int(cx), int(cy + 3), int(cx + 5), int(cy - 2))
        p.end()


class StatusBadge(QWidget):
    def __init__(self, text: str, color: str, parent=None, name=""):
        super().__init__(parent)
        self.name = name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.dot = QWidget()
        self.dot.setFixedSize(12, 12)
        self.dot.setStyleSheet(f"""
            background: {color};
            border-radius: 6px;
        """)

        self.label = QLabel(text)
        self.label.setStyleSheet("color: #374151; padding: 0; margin: 0;")
        setFont(self.label, 11)

        layout.addWidget(self.dot)
        layout.addWidget(self.label)

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

    def setLabel(self, text: str, color: str):
        self.label.setText(text)
        self.dot.setStyleSheet(f"""
            background: {color};
            border-radius: 6px;
        """)


class ModelVariantPanel(QWidget):
    variantChanged = Signal(str)

    def __init__(self, config_key: str, parent=None):
        super().__init__(parent)
        self.config_key = config_key
        self._expanded = False
        self._panel_height = 140

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Collapsible panel content
        self.panel = QWidget()
        self.panel.setMaximumHeight(0)
        self.panel.setStyleSheet("background: #fcfcfc; border-top: 1px dashed #e5e5e5;")

        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(24, 12, 24, 16)
        panel_layout.setSpacing(12)

        # Row 1: hardware optimization segmented control
        hw_row = QHBoxLayout()
        hw_row.setSpacing(12)
        label = QLabel(self.tr("硬件优化:"))
        setFont(label, 13, QFont.DemiBold)
        label.setStyleSheet("color: #1a1a1a; border: none;")
        hw_row.addWidget(label)

        seg_container = QWidget()
        seg_container.setStyleSheet("background: #f0f0f0; border-radius: 8px; border: none;")
        seg_layout = QHBoxLayout(seg_container)
        seg_layout.setContentsMargins(3, 3, 3, 3)
        seg_layout.setSpacing(0)

        self.cpu_btn = QPushButton("CPU 优化")
        self.gpu_btn = QPushButton("GPU 加速")
        for btn in (self.cpu_btn, self.gpu_btn):
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            setFont(btn, 13, QFont.Medium)
        self.cpu_btn.setChecked(True)

        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        self._btn_group.addButton(self.cpu_btn, 0)
        self._btn_group.addButton(self.gpu_btn, 1)
        self._btn_group.idClicked.connect(self._on_segment_clicked)

        seg_layout.addWidget(self.cpu_btn)
        seg_layout.addWidget(self.gpu_btn)
        hw_row.addWidget(seg_container)
        hw_row.addStretch()
        panel_layout.addLayout(hw_row)

        # Row 2: mirror acceleration toggle
        mirror_row = QHBoxLayout()
        mirror_row.setSpacing(12)
        mirror_label = QLabel(self.tr("国内加速:"))
        setFont(mirror_label, 13, QFont.DemiBold)
        mirror_label.setStyleSheet("color: #1a1a1a; border: none;")
        mirror_row.addWidget(mirror_label)

        self.mirror_switch = ToggleSwitch()
        self.mirror_switch.setActive(self._load_mirror_preference())
        self.mirror_switch.toggled.connect(self._on_mirror_toggled)
        mirror_row.addWidget(self.mirror_switch)

        mirror_tip = QLabel(self.tr("开启后使用 hf-mirror.com 镜像加速下载"))
        mirror_tip.setStyleSheet("color: #9ca3af; border: none;")
        setFont(mirror_tip, 11)
        mirror_row.addWidget(mirror_tip)
        mirror_row.addStretch()
        panel_layout.addLayout(mirror_row)

        # Row 3: model status, estimated size, download button
        info_row = QHBoxLayout()
        info_row.setSpacing(8)

        self.status_label = QLabel(self.tr("模型状态: ") + f"<b style='color:#e65100;'>{self.tr('未下载')}</b>")
        self.status_label.setStyleSheet("color: #666; border: none;")
        setFont(self.status_label, 13)
        info_row.addWidget(self.status_label)

        sep_label = QLabel("|")
        sep_label.setStyleSheet("color: #ccc; border: none;")
        info_row.addWidget(sep_label)

        estimated_size = model_estimated_sizes.get("cpu", {}).get(config_key, "-- MB")
        self.size_label = QLabel(self.tr("预计大小: ") + estimated_size)
        self.size_label.setStyleSheet("color: #666; border: none;")
        setFont(self.size_label, 13)
        info_row.addWidget(self.size_label)
        info_row.addStretch()
        panel_layout.addLayout(info_row)

        self._update_segment_styles()

        main_layout.addWidget(self.panel)

        # Animation
        self._anim = QPropertyAnimation(self.panel, b"maximumHeight")
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QEasingCurve.InOutCubic)

        self._load_preference()
        self._check_model_status()

    def _check_model_status(self):
        variant = self.get_variant()
        deps_path = cfg.get(cfg.localAIModelDeps)
        downloaded = False
        if deps_path:
            model_path = os.path.join(deps_path, variant, self.config_key)
            if os.path.exists(model_path) and os.listdir(model_path):
                downloaded = True
        if downloaded:
            self.status_label.setText(self.tr("模型状态: ") + f"<b style='color:#2da44e;'>{self.tr('已就绪')}</b>")
        else:
            self.status_label.setText(self.tr("模型状态: ") + f"<b style='color:#e65100;'>{self.tr('未下载')}</b>")

    def _seg_active_style(self):
        return """
            QPushButton { background: #ffffff; color: #1a1a1a; border: none;
                border-radius: 6px; padding: 6px 16px; }
        """

    def _seg_inactive_style(self):
        return """
            QPushButton { background: transparent; color: #555; border: none;
                border-radius: 6px; padding: 6px 16px; }
            QPushButton:hover { background: #e8e8e8; }
        """

    def _update_segment_styles(self):
        self.cpu_btn.setStyleSheet(self._seg_active_style() if self.cpu_btn.isChecked() else self._seg_inactive_style())
        self.gpu_btn.setStyleSheet(self._seg_active_style() if self.gpu_btn.isChecked() else self._seg_inactive_style())

    def _on_segment_clicked(self, id):
        self._update_segment_styles()
        variant = "cpu" if id == 0 else "gpu"
        self._save_preference(variant)
        self._update_size_label(variant)
        self._check_model_status()
        self.variantChanged.emit(variant)

    def _update_size_label(self, variant: str):
        size = model_estimated_sizes.get(variant, {}).get(self.config_key, "-- MB")
        self.size_label.setText(self.tr("预计大小: ") + size)

    def get_variant(self) -> str:
        return "gpu" if self.gpu_btn.isChecked() else "cpu"

    def _load_preference(self):
        try:
            variant = cfg.get(cfg.additionalParams).get("ModelVariants", {}).get(self.config_key, None)
        except Exception:
            variant = None
        if variant is None:
            hw_type = cfg.get(cfg.hardwareOptimizationType)
            if hw_type == "CPU":
                variant = "cpu"
            elif hw_type == "GPU":
                variant = "gpu"
            else:
                variant = "gpu" if detect_gpu_available() else "cpu"
            self._save_preference(variant)
        if variant == "gpu":
            self.gpu_btn.setChecked(True)
        else:
            self.cpu_btn.setChecked(True)
        self._update_segment_styles()
        self._update_size_label(variant)

    def _save_preference(self, variant: str):
        params = cfg.get(cfg.additionalParams)
        variants = params.get("ModelVariants", {})
        variants[self.config_key] = variant
        params["ModelVariants"] = variants
        cfg.additionalParams.value = params

    def _load_mirror_preference(self) -> bool:
        try:
            return cfg.get(cfg.additionalParams).get("use_hf_mirror", True)
        except Exception:
            return False

    def _on_mirror_toggled(self, flag: bool):
        params = cfg.get(cfg.additionalParams)
        params["use_hf_mirror"] = flag
        cfg.additionalParams.value = params

    def use_mirror(self) -> bool:
        return self.mirror_switch.isActive()

    def toggle_expand(self):
        self._expanded = not self._expanded
        self._anim.stop()
        self._anim.setStartValue(self.panel.maximumHeight())
        self._anim.setEndValue(self._panel_height if self._expanded else 0)
        self._anim.start()

    def is_expanded(self) -> bool:
        return self._expanded
    
    def check_model_status(self):
        self._check_model_status()


class InitProgressDialog(QDialog):
    def __init__(self, title="", variant: str = "cpu", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(660)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)

        self.bg_widget = QDialog()
        self.bg_widget.setStyleSheet("""
            QDialog {
                background-color: rgba(245, 246, 250, 0.95);
                border-radius: 14px;
            }
        """)
        bg_layout = QVBoxLayout(self.bg_widget)
        bg_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(self.bg_widget)

        header_layout = QHBoxLayout()
        variant_label = "GPU 加速" if variant == "gpu" else "CPU 优化"
        title_label = QLabel(self.tr(f"🔧 正在初始化 {variant_label} 环境，请稍候…"))
        setFont(title_label, 14, QFont.Bold)
        title_label.setStyleSheet("color: #1f2937;")
        header_layout.addWidget(title_label)

        self.close_btn = CloseButton()
        self.close_btn.clicked.connect(self.close)
        header_layout.addWidget(self.close_btn)
        header_layout.setAlignment(self.close_btn, Qt.AlignRight)
        bg_layout.addLayout(header_layout)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                height: 14px;
                border-radius: 7px;
                background: #e5e7eb;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #60a5fa, stop:1 #3b82f6
                );
                border-radius: 7px;
            }
        """)
        bg_layout.addWidget(self.progress)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        setFont(self.log_box, 12)
        self.log_box.setStyleSheet("""
            QTextEdit {
                background: #fefefe;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
                padding: 10px;
                color: #374151;
            }
        """)
        self.log_box.setMinimumHeight(180)
        bg_layout.addWidget(self.log_box)

    def append_log(self, text: str):
        self.log_box.append(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def enableCloseBtn(self):
        self.close_btn.show()

    def disableCloseBtn(self):
        self.close_btn.hide()

class MultiConfigSoftwareCard(QFrame):
    def __init__(self, name: str, icon: dict, description: str, fields: list, parent=None):
        super().__init__(parent)
        self.setObjectName("softwareCard")
        self._name = name
        self._icon = icon
        self._description = description
        self._fields = fields
        self._widgets: dict = {}
        self._badges: dict = {}
        self._setup_ui()

    def get_widget(self, key: str):
        return self._widgets.get(key)

    def get_value(self, key: str):
        w = self._widgets.get(key)
        if w is None:
            return None
        if isinstance(w, QLineEdit):
            return w.text()
        return w.currentText()

    def get_badge(self, key: str):
        return self._badges.get(key)

    def set_badge(self, key: str, text: str, color: str):
        badge = self._badges.get(key)
        if badge:
            badge.setLabel(text=text, color=color)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        icon_label = QLabel(self._icon["symbol"])
        icon_label.setFixedSize(40, 40)
        icon_label.setAlignment(Qt.AlignCenter)
        setFont(icon_label, 20)
        g0, g1 = self._icon["gradient"]
        icon_label.setStyleSheet(f"""
            QLabel {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 {g0}, stop:1 {g1});
                border-radius: 10px;
                color: white;
            }}
        """)
        name_info = QVBoxLayout()
        name_info.setSpacing(2)
        name_label = QLabel(self.tr(self._name))
        setFont(name_label, 16, QFont.Bold)
        name_label.setStyleSheet("color: #1f2937;")
        desc_label = QLabel(self.tr(self._description))
        setFont(desc_label, 12, QFont.Bold)
        desc_label.setStyleSheet("color: #6b7280;")
        name_info.addWidget(name_label)
        name_info.addWidget(desc_label)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)
        header_layout.addWidget(icon_label)
        header_layout.addLayout(name_info)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        for field in self._fields:
            row = self._build_field_row(field)
            layout.addLayout(row)

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

    def _build_field_row(self, field: dict) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        label = QLabel(self.tr(field["label"]))
        setFont(label, 13, QFont.DemiBold)
        label.setStyleSheet("color: #374151;")
        row.addWidget(label)
        field_type = field.get("type", "path")
        if field_type == "combo":
            row = self._build_combo_row(row, field)
        else:
            row = self._build_path_row(row, field)
        return row

    def _build_path_row(self, row: QHBoxLayout, field: dict) -> QHBoxLayout:
        cfg_item = field["cfg_item"]
        edit = QLineEdit()
        if cfg.get(cfg_item) != cfg.softwareInvalidPath:
            edit.setText(cfg.get(cfg_item))
        else:
            if "placeholder" in field:
                edit.setPlaceholderText(self.tr(field["placeholder"]))
        edit.textChanged.connect(lambda val, item=cfg_item, field=field: self._update_global_config(item, val, field))
        setFont(edit, 13)
        edit.setStyleSheet(self._input_style())

        browse_btn = QPushButton()
        browse_btn.setIcon(FluentIcon.FOLDER_ADD.qicon())
        browse_btn.setStyleSheet(self._btn_style(bg="#f3f4f6", hover="#d1d5db"))
        browse_btn.clicked.connect(lambda _=False, e=edit: self._select_folder(e))

        row.addWidget(edit)
        row.addWidget(browse_btn)
        self._append_verify(row, field)
        self._widgets[field["key"]] = edit
        return row

    def _build_combo_row(self, row: QHBoxLayout, field: dict) -> QHBoxLayout:
        cfg_item = field["cfg_item"]
        value_map: dict = field.get("value_map", {})
        save_map: dict = field.get("save_map", {})
        combo = ComboBox()
        setFont(combo, 13)
        combo.addItems(field.get("options", []))
        current_stored = str(cfg.get(cfg_item))
        display_text = value_map.get(current_stored, current_stored)
        combo.setText(display_text)

        def _on_changed(text: str, item=cfg_item, smap=save_map):
            cfg.set(item, smap.get(text, text))

        combo.currentTextChanged.connect(_on_changed)
        row.addWidget(combo)
        if "hint" in field:
            hint = QLabel(self.tr(field["hint"]))
            setFont(hint, 11)
            hint.setStyleSheet("color: #9ca3af;")
            row.addWidget(hint)
        self._append_verify(row, field)
        row.addStretch()
        self._widgets[field["key"]] = combo
        return row

    def _append_verify(self, row: QHBoxLayout, field: dict):
        verify_cfg = field.get("verify")
        if not verify_cfg:
            return
        badge = StatusBadge(text=self.tr("未验证"), color="#eab308")
        self._badges[field["key"]] = badge
        try:
            text = cfg.get(cfg.additionalParams)["SoftwareSettings"][f"{self._name}_{field["key"]}_status_info"]["text"]
            color = cfg.get(cfg.additionalParams)["SoftwareSettings"][f"{self._name}_{field["key"]}_status_info"]["color"]
            badge.setLabel(text=text, color=color)
        except Exception:
            pass
        row.addWidget(badge, alignment=Qt.AlignVCenter)
        btn_label = self.tr(verify_cfg.get("label", "验证"))
        verify_btn = QPushButton(btn_label)
        setFont(verify_btn, 12, QFont.Bold)
        verify_btn.setStyleSheet(self._btn_style(bg="#4f46e5", hover="#4338ca", color="white"))
        on_verify = verify_cfg.get("on_verify")
        if callable(on_verify):
            verify_btn.clicked.connect(lambda _=False, fn=on_verify, b=badge, field=field: self._run_verify(fn, b, field))
        row.addWidget(verify_btn)

    def _run_verify(self, on_verify, badge: StatusBadge, field: dict):
        try:
            result, err_msg = on_verify(self)
            if result:
                text = "OK"
                color = "#16a34a"
            else:
                text = "Failed"
                color = "#dc2626"
                TeachingTip.create(
                    target=badge,
                    icon=InfoBarIcon.ERROR,
                    title=self.tr("警告"),
                    content=self.tr(err_msg),
                    isClosable=True,
                    tailPosition=TeachingTipTailPosition.BOTTOM,
                    duration=3000,
                    parent=self
                )
        except Exception:
            text = "Failed"
            color = "#dc2626"
        finally:
            badge.setLabel(text=self.tr(text), color=color)
            tmp = cfg.get(cfg.additionalParams).get("SoftwareSettings", {})
            tmp.update({f"{self._name}_{field["key"]}_status_info": {"text": text, "color": color}})
            cfg.additionalParams.value.update({"SoftwareSettings": tmp})

    @staticmethod
    def _input_style() -> str:
        return """
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                background: white;
                color: #333;
            }
            QLineEdit:focus { border: 1px solid #4f46e5; }
        """

    @staticmethod
    def _btn_style(bg: str, hover: str, color: str = "#374151") -> str:
        return f"""
            QPushButton {{
                padding: 8px 12px;
                background: {bg};
                color: {color};
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{ background: {hover}; }}
            QPushButton:pressed {{ background: {hover}; padding: 9px 12px; margin-top: 1px; }}
        """

    def _select_folder(self, target: QLineEdit):
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择文件夹",
            "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog if sys.platform == "darwin" else QFileDialog.Option(0),
        )
        if directory:
            if not directory.isascii():
                MessageBox(title=self.tr("提醒"), content="不支持非英文路径", parent=self.window()).exec()
                return
            target.setText(directory)

    def _update_global_config(self, item, path: str, field: dict):
        cfg.set(item, path)
        # 状态提示信息恢复默认
        badge = self.get_badge(field["key"])
        text = "未验证"
        color = "#eab308"
        badge.setLabel(text=text, color=color)
        tmp = cfg.get(cfg.additionalParams).get("SoftwareSettings", {})
        tmp.update({f"{self._name}_{field["key"]}_status_info": {"text": text, "color": color}})
        cfg.additionalParams.value.update({"SoftwareSettings": tmp})

class SoftwareCard(QFrame):
    global_config_params_name_map = {}

    def __init__(self, name: str, icon: dict, description: str, status: str, parent=None):
        super().__init__(parent)
        self.setObjectName("softwareCard")
        self.name = name
        self.status = status
        self._register_cfg()
        self._setup_ui(icon, description)

    def _register_cfg(self):
        """注册和全局参数配置中心绑定关系.
        
        后续新增软件配置，这里需要适配
        """
        if self.name.lower() == "ffmpeg":
            self.global_config_params_name_map[self.name.lower()] = cfg.ffmpeg_path

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

        path_layout = QHBoxLayout()
        path_layout.setSpacing(8)
        self.path_input = QLineEdit()
        default_path = cfg.get(self.global_config_params_name_map[self.name.lower()])
        if default_path and default_path != cfg.softwareInvalidPath:
            self.path_input.setText(default_path)
            try:
                text = cfg.get(cfg.additionalParams)["SoftwareSettings"][f"{self.name}_status_info"]["text"]
                color = cfg.get(cfg.additionalParams)["SoftwareSettings"][f"{self.name}_status_info"]["color"]
                self.status_label.setLabel(text=text, color=color)
            except Exception:
                pass
        else:
            self.path_input.setPlaceholderText(self.tr(f"请配置 {self.name} 软件路径"))
        self.path_input.textChanged.connect(self._update_global_config)
        setFont(self.path_input, 14)
        self.path_input.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                background: white;
                color: #333;
            }
            QLineEdit:focus { border: 1px solid #4f46e5; }
        """)

        browse_btn = QPushButton()
        browse_btn.setIcon(FluentIcon.FOLDER_ADD.qicon())
        browse_btn.setStyleSheet(self._btn_style(bg="#f3f4f6", hover="#d1d5db"))
        setFont(browse_btn, 12, QFont.Bold)
        browse_btn.clicked.connect(lambda: self._select_path(select_file=False))

        test_btn = QPushButton(self.tr("验证"))
        test_btn.setStyleSheet(self._btn_style(bg="#4f46e5", hover="#4338ca", color="white"))
        setFont(test_btn, 12, QFont.Bold)
        test_btn.clicked.connect(self._check_software)

        path_layout.addWidget(self.path_input)
        path_layout.addWidget(browse_btn)
        path_layout.addWidget(self.status_label)
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
            return StatusBadge("OK", self._get_status_badge_color())
        elif self.status == "failed":
            return StatusBadge("Failed", self._get_status_badge_color())
        else:
            return StatusBadge("未验证", self._get_status_badge_color())
        
    def _get_status_badge_color(self):
        if self.status == "ok":
            return "#16a34a"
        elif self.status == "failed":
            return "#dc2626"
        else:
            return "#eab308"

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
            QPushButton:pressed {{
                background: {hover};
                padding: 9px 16px;
                margin-top: 1px;
            }}
        """

    def _check_software(self):
        error_msg = ""
        if self.name.lower() == "ffmpeg":
            ffmpeg_exe = os.path.join(self.path_input.text(), "ffmpeg" if platform.system().lower() != "windows" else "ffmpeg.exe")
            ffprobe_exe = os.path.join(self.path_input.text(), "ffprobe.exe" if platform.system().lower() == "windows" else "ffprobe")
            exe_list = [ffmpeg_exe, ffprobe_exe]
            files_missing_list = [p for p in [ffmpeg_exe, ffprobe_exe] if not os.path.exists(p)]
            if files_missing_list:
                self.status = "failed"
                text = "Failed"
                color = self._get_status_badge_color()
                error_msg = "\n".join(f"- 文件 {p} 不存在" for p in files_missing_list)
            else:
                for exe in exe_list:
                    if not os.access(exe, os.X_OK):
                        error_msg += f"- 无执行权限: {exe}\n"
                    try:
                        proc = subprocess.run(
                            [exe, "-version"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=1,
                            creationflags=subprocess.CREATE_NO_WINDOW if platform.system().lower() == "windows" else 0
                        )
                        if proc.returncode != 0:
                            error_msg += f"- {os.path.basename(exe)} 执行失败，可能损坏: {exe}\n"
                    except subprocess.TimeoutExpired:
                        error_msg += f"- {os.path.basename(exe)} 执行超时，文件可能不是有效的可执行程序: {exe}\n"
                    except Exception as e:
                        error_msg += f"- {os.path.basename(exe)} 无法运行: {exe}\n  错误: {e}\n"
                if not error_msg:
                    self.status = "ok"
                    text = "OK"
                    color = self._get_status_badge_color()
                else:
                    self.status = "failed"
                    text = "Failed"
                    color = self._get_status_badge_color()
            self.status_label.setLabel(text=text, color=color)
            tmp = cfg.get(cfg.additionalParams).get("SoftwareSettings", {})
            tmp.update({f"{self.name}_status_info": {"text": text, "color": color}})
            cfg.additionalParams.value.update({"SoftwareSettings": tmp})
        if error_msg:
            TeachingTip.create(
                target=self.status_label,
                icon=InfoBarIcon.ERROR,
                title=self.tr("警告"),
                content=self.tr(error_msg),
                isClosable=True,
                tailPosition=TeachingTipTailPosition.BOTTOM,
                duration=3000,
                parent=self
            )

    def _select_path(self, select_file=True):
        if not select_file:
            directory = QFileDialog.getExistingDirectory(
                self,
                "选择文件夹",
                "",
                QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog if sys.platform == "darwin" else QFileDialog.Option(0)
            )
            if directory:
                if isinstance(directory, str) and not directory.isascii():
                    MessageBox(title=self.tr("提醒"), content="不支持非英文路径", parent=self.window()).exec()
                    return
                self.path_input.setText(directory)
        else:
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "选择文件",
                "", 
                "所有文件 (*)",
                options=QFileDialog.Option.DontUseNativeDialog if sys.platform == "darwin" else QFileDialog.Option(0)
            )
            if files:
                if isinstance(files, str) and not files.isascii():
                    MessageBox(title=self.tr("提醒"), content="不支持非英文路径", parent=self.window()).exec()
                    return
                self.path_input.setText(files)

    def _update_global_config(self, path: str):
        self.global_config_params_name_map[self.name.lower()].value = path
        # 状态提示信息恢复默认
        text = "未验证"
        self.status = ""
        color = self._get_status_badge_color()
        self.status_label.setLabel(text=text, color=color)
        tmp = cfg.get(cfg.additionalParams).get("SoftwareSettings", {})
        tmp.update({f"{self.name}_status_info": {"text": text, "color": color}})
        cfg.additionalParams.value.update({"SoftwareSettings": tmp})


class Settings(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("Settings")
        
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

        general_settings = self._create_general_settings()
        content_layout.addWidget(general_settings)
        
        software_settings = self._create_software_settings()
        content_layout.addWidget(software_settings)

        local_ai_settings = self._create_local_ai_settings()
        content_layout.addWidget(local_ai_settings)

        performance_settings = self._create_performance_settings()
        content_layout.addWidget(performance_settings)
        
        scroll.setWidget(content)

        scroll.setWidgetResizable(True)
        scroll.enableTransparentBackground()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        main_layout.addWidget(scroll)

    def _create_software_settings(self):
        group = CustomGroupBox(title=self.tr("🔌 软件配置"))

        ffmpeg_card = SoftwareCard(
            name="FFmpeg", 
            icon={"gradient": ["#667eea","#764ba2"], "symbol":"🎬"}, 
            description=self.tr("视频处理引擎"), 
            status=""
        )

        def _check_cuda_env(card: MultiConfigSoftwareCard):
            cuda_path = card.get_value("cuda")
            if platform.system().lower() == "windows":
                cuda_dll_paths = [
                    "cublasLt64_12.dll",
                    "cublas64_12.dll",
                    "cufft64_11.dll",
                    "cudart64_12.dll",
                ]
            else:
                cuda_dll_paths = [
                    "libcublasLt.so.12",
                    "libcublas.so.12",
                    "libnvrtc.so.12",
                    "libcurand.so.10",
                    "libcufft.so.11",
                    "libcudart.so.12",
                ]
            for item in cuda_dll_paths:
                item_path = os.path.join(cuda_path, item)
                if not os.path.exists(item_path):
                    return False, f"{item_path} 不存在"
            current_path = os.environ.get("PATH", "")
            path_list = [os.path.normpath(p) for p in current_path.split(os.pathsep) if p]
            normalized_cuda = os.path.normpath(cuda_path)
            if normalized_cuda not in path_list:
                os.environ["PATH"] = normalized_cuda + os.pathsep + current_path
            cuda_parent_dir = os.path.dirname(os.path.abspath(cuda_path))
            if "CUDA_PATH" not in os.environ or os.environ["CUDA_PATH"] != cuda_parent_dir:
                os.environ["CUDA_PATH"] = cuda_parent_dir
            os.makedirs(cfg.softwareInvalidPath, exist_ok=True)
            gpu_config_path = os.path.join(cfg.softwareInvalidPath, "gpu_env_config.json")
            with open(gpu_config_path, "r") as fp:
                data = json.loads(fp.read())
            data["cuda_path"] = cuda_path
            with open(gpu_config_path, "w") as fp:
                fp.write(json.dumps(data))
            return True, ""

        def _check_cudnn_env(card: MultiConfigSoftwareCard):
            cudnn_path = card.get_value("cudnn")
            if platform.system().lower() == "windows":
                cudnn_dll_paths = [
                    "cudnn_engines_runtime_compiled64_9.dll",
                    "cudnn_engines_precompiled64_9.dll",
                    "cudnn_heuristic64_9.dll",
                    "cudnn_ops64_9.dll",
                    "cudnn_adv64_9.dll",
                    "cudnn_graph64_9.dll",
                    "cudnn64_9.dll",
                ]
            else:
                cudnn_dll_paths = ["libcudnn.so.9"]
            for item in cudnn_dll_paths:
                item_path = os.path.join(cudnn_path, item)
                if not os.path.exists(item_path):
                    return False, f"{item_path} 不存在"
            current_path = os.environ.get("PATH", "")
            path_list = [os.path.normpath(p) for p in current_path.split(os.pathsep) if p]
            normalized_cudnn = os.path.normpath(cudnn_path)
            if normalized_cudnn not in path_list:
                os.environ["PATH"] = normalized_cudnn + os.pathsep + current_path
            os.makedirs(cfg.softwareInvalidPath, exist_ok=True)
            gpu_config_path = os.path.join(cfg.softwareInvalidPath, "gpu_env_config.json")
            with open(gpu_config_path, "r") as fp:
                data = json.loads(fp.read())
            data["cudnn_path"] = cudnn_path
            with open(gpu_config_path, "w") as fp:
                fp.write(json.dumps(data))
            return True, ""

        gpu_env_card = MultiConfigSoftwareCard(
            name="GPU",
            icon={"symbol": "🎮", "gradient": ["#06b6d4", "#0284c7"]},
            description="CUDA / cuDNN 路径与显存设置",
            fields=[
                {
                    "key":       "vram",
                    "label":     "显存上限",
                    "type":      "combo",
                    "cfg_item":  cfg.gpuMemoryLimit,
                    "options":   ["6 GB", "8 GB", "12 GB", "16 GB", "24 GB"],
                    "value_map": {"6": "6 GB", "8": "8 GB", "12": "12 GB", "16": "16 GB", "24": "24 GB"},
                    "save_map": {"6 GB": "6", "8 GB": "8", "12 GB": "12", "16 GB": "16", "24 GB": "24"},
                    "hint": "最大显存用于指导算法成功运行"
                },
                {
                    "key":         "cuda",
                    "label":       "CUDA",
                    "type":        "path",
                    "cfg_item":    cfg.cudaPath,
                    "placeholder": "设置 cuda 安装路径，指定到驱动文件一级目录",
                    "verify": {
                        "label":     "验证",
                        "on_verify": _check_cuda_env,
                    },
                },
                {
                    "key":         "cudnn",
                    "label":       "cuDNN",
                    "type":        "path",
                    "cfg_item":    cfg.cudnnPath,
                    "placeholder": "设置 cudnn 安装路径，指定到驱动文件一级目录",
                    "verify": {
                        "label":     "验证",
                        "on_verify": _check_cudnn_env,
                    },
                },
            ],
            parent=self,
        )
        self.software_cards = [ffmpeg_card, gpu_env_card]
        for card in self.software_cards:
            group.addCard(card=card)

        return group
    
    def _create_general_settings(self):
        settings_cards = []
        settings = CustomGroupBox(title=self.tr("⚙️ 通用设置"))
        
        auto_start_switch = ToggleSwitch()
        auto_start_switch.setActive(cfg.get(cfg.autoStartup))
        auto_start_switch.toggled.connect(lambda flag: cfg.set(cfg.autoStartup, flag))
        auto_start_card = CustomCardGroupWidget(title=self.tr("开机自启动"), content=self.tr("系统启动时自动运行程序（开发中）"), parent=self)
        auto_start_card.addWidget(auto_start_switch, stretch=0)
        auto_start_card.setSeparatorVisible(True)
        settings_cards.append(auto_start_card)

        auto_update_switch = ToggleSwitch()
        auto_update_switch.setActive(cfg.get(cfg.autoUpdate))
        auto_update_switch.toggled.connect(lambda flag: cfg.set(cfg.autoUpdate, flag))
        auto_update_card = CustomCardGroupWidget(title=self.tr("自动更新"), content=self.tr("自动检查并安装新版本（开发中）"), parent=self)
        auto_update_card.addWidget(auto_update_switch, stretch=0)
        auto_update_card.setSeparatorVisible(True)
        settings_cards.append(auto_update_card)

        self.cache_line_edit = QLineEdit()
        self.cache_line_edit.setText(cfg.get(cfg.cachePath))
        self.cache_line_edit.textChanged.connect(lambda path: cfg.set(cfg.cachePath, path))
        self.cache_line_edit.textChanged.connect(lambda path: get_log_manager().update_log_dir(os.path.join(path, "logs")))
        setFont(self.cache_line_edit, 14)
        self.cache_line_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                background: white;
                color: #333;
            }
            QLineEdit:focus { border: 1px solid #4f46e5; }
        """)
        browse_btn = QPushButton()
        browse_btn.setIcon(FluentIcon.FOLDER_ADD.qicon())
        browse_btn.setStyleSheet(self._btn_style(bg="#f3f4f6", hover="#d1d5db"))
        setFont(browse_btn, 12, QFont.Bold)
        browse_btn.clicked.connect(lambda: self._select_path(self.cache_line_edit))
        cache_location_card = CustomCardGroupWidget(title=self.tr("缓存保存路径"), content=self.tr("设置缓存文件保存位置"), parent=self)
        cache_location_card.addWidget(self.cache_line_edit, stretch=1)
        cache_location_card.addWidget(browse_btn, stretch=0)
        cache_location_card.setSeparatorVisible(True)
        settings_cards.append(cache_location_card)

        theme_combox = ComboBox()
        theme_combox.setText(theme_map[cfg.get(cfg.uiTheme)])
        theme_combox.currentTextChanged.connect(lambda text: cfg.set(cfg.uiTheme, theme_map[text]))
        setFont(theme_combox, 14)
        theme_combox.addItems(["浅色"])
        theme_card = CustomCardGroupWidget(title=self.tr("界面主题"), content=self.tr("选择您喜欢的界面风格"), parent=self)
        theme_card.addWidget(theme_combox, stretch=0)
        theme_card.setSeparatorVisible(True)
        settings_cards.append(theme_card)

        language_combox = ComboBox()
        language_combox.setText(language_map[cfg.language.serialize()])
        language_combox.currentTextChanged.connect(lambda text: cfg.set(cfg.language, language_map[text]))
        setFont(language_combox, 14)
        language_combox.addItems(["简体中文"])
        language_card = CustomCardGroupWidget(title=self.tr("语言设置"), content=self.tr("选择界面显示语言"), parent=self)
        language_card.addWidget(language_combox, stretch=0)
        language_card.setSeparatorVisible(True)
        settings_cards.append(language_card)

        for card in settings_cards:
            settings.addCard(card=card)

        return settings
    
    def _create_chevron_btn(self, panel: ModelVariantPanel):
        btn = ChevronButton()
        btn.clicked.connect(lambda: self._toggle_chevron(btn, panel))
        return btn

    def _toggle_chevron(self, btn: ChevronButton, panel: ModelVariantPanel):
        panel.toggle_expand()
        btn.set_rotated(panel.is_expanded())

    def _create_local_ai_settings(self):
        self.ai_toggle_switchs: list[ToggleSwitch] = []
        ai_settings_cards = []
        ai_settings = CustomGroupBox(title=self.tr("🤖 本地AI设置"))

        localAIModelDeps_line_edit = QLineEdit()
        localAIModelDeps_line_edit.setText(cfg.get(cfg.localAIModelDeps))
        localAIModelDeps_line_edit.textChanged.connect(lambda path: cfg.set(cfg.localAIModelDeps, path))
        localAIModelDeps_line_edit.textChanged.connect(self._update_toggle_switch_off)
        setFont(localAIModelDeps_line_edit, 14)
        localAIModelDeps_line_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                background: white;
                color: #333;
            }
            QLineEdit:focus { border: 1px solid #4f46e5; }
        """)
        browse_btn = QPushButton()
        browse_btn.setIcon(FluentIcon.FOLDER_ADD.qicon())
        browse_btn.setStyleSheet(self._btn_style(bg="#f3f4f6", hover="#d1d5db"))
        setFont(browse_btn, 12, QFont.Bold)
        browse_btn.clicked.connect(lambda: self._select_path(localAIModelDeps_line_edit))
        model_deps_location_card = CustomCardGroupWidget(title=self.tr("AI模型依赖路径"), content=self.tr("设置本地AI模型依赖文件保存位置"), parent=self)
        model_deps_location_card.addWidget(localAIModelDeps_line_edit, stretch=1)
        model_deps_location_card.addWidget(browse_btn, stretch=0)
        model_deps_location_card.setSeparatorVisible(True)
        ai_settings_cards.append(model_deps_location_card)
        
        blind_watermark_switch = ToggleSwitch()
        self.ai_toggle_switchs.append(blind_watermark_switch)
        blind_watermark_switch.setActive(cfg.get(cfg.localBlindWatermarkEnabled))
        blind_watermark_switch.toggled.connect(lambda flag: cfg.set(cfg.localBlindWatermarkEnabled, flag))
        blind_watermark_status = StatusBadge(text=self.tr("未启用"), color="#eab308", name="blind_watermark_addition")
        try:
            text = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{blind_watermark_status.name}_status_info"]["text"]
            color = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{blind_watermark_status.name}_status_info"]["color"]
            blind_watermark_status.setLabel(text=text, color=color)
        except Exception:
            pass
        blind_watermark_panel = ModelVariantPanel(config_key="blind_watermark_addition", parent=self)
        self._bind_ai_toggle(
            switch=blind_watermark_switch,
            badge=blind_watermark_status,
            local_ai_type="blind_watermark_addition",
            panel=blind_watermark_panel
        )
        blind_watermark_chevron = self._create_chevron_btn(blind_watermark_panel)
        blind_watermark_card = CustomCardGroupWidget(title=self.tr("盲水印AI能力"), content=self.tr("为图像添加不可见的数字水印，保护版权"), parent=self)
        blind_watermark_card.addWidget(blind_watermark_status, stretch=0)
        blind_watermark_card.addWidget(blind_watermark_switch, stretch=0)
        blind_watermark_card.addWidget(blind_watermark_chevron, stretch=0)
        blind_watermark_card.vBoxLayout.addWidget(blind_watermark_panel)
        blind_watermark_card.setSeparatorVisible(True)
        ai_settings_cards.append(blind_watermark_card)

        watermark_removal_switch = ToggleSwitch()
        self.ai_toggle_switchs.append(watermark_removal_switch)
        watermark_removal_switch.setActive(cfg.get(cfg.localWatermarkRemovalEnabled))
        watermark_removal_switch.toggled.connect(lambda flag: cfg.set(cfg.localWatermarkRemovalEnabled, flag))
        watermark_removal_status = StatusBadge(text=self.tr("未启用"), color="#eab308", name="watermark_removal")
        try:
            text = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{watermark_removal_status.name}_status_info"]["text"]
            color = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{watermark_removal_status.name}_status_info"]["color"]
            watermark_removal_status.setLabel(text=text, color=color)
        except Exception:
            pass
        watermark_removal_panel = ModelVariantPanel(config_key="visible_watermark_removal", parent=self)
        self._bind_ai_toggle(
            switch=watermark_removal_switch,
            badge=watermark_removal_status,
            local_ai_type="visible_watermark_removal",
            panel=watermark_removal_panel
        )
        watermark_removal_chevron = self._create_chevron_btn(watermark_removal_panel)
        watermark_removal_card = CustomCardGroupWidget(title=self.tr("水印去除AI能力"), content=self.tr("智能去除图像中的水印和标志"), parent=self)
        watermark_removal_card.addWidget(watermark_removal_status, stretch=0)
        watermark_removal_card.addWidget(watermark_removal_switch, stretch=0)
        watermark_removal_card.addWidget(watermark_removal_chevron, stretch=0)
        watermark_removal_card.vBoxLayout.addWidget(watermark_removal_panel)
        watermark_removal_card.setSeparatorVisible(True)
        ai_settings_cards.append(watermark_removal_card)

        object_segmentation_switch = ToggleSwitch()
        self.ai_toggle_switchs.append(object_segmentation_switch)
        object_segmentation_switch.setActive(cfg.get(cfg.localObjectSegmentationEnabled))
        object_segmentation_switch.toggled.connect(lambda flag: cfg.set(cfg.localObjectSegmentationEnabled, flag))
        object_segmentation_status = StatusBadge(text=self.tr("未启用"), color="#eab308", name="object_segmentation")
        try:
            text = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{object_segmentation_status.name}_status_info"]["text"]
            color = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{object_segmentation_status.name}_status_info"]["color"]
            object_segmentation_status.setLabel(text=text, color=color)
        except Exception:
            pass
        object_segmentation_panel = ModelVariantPanel(config_key="segment", parent=self)
        self._bind_ai_toggle(
            switch=object_segmentation_switch,
            badge=object_segmentation_status,
            local_ai_type="segment",
            panel=object_segmentation_panel
        )
        object_segmentation_chevron = self._create_chevron_btn(object_segmentation_panel)
        object_segmentation_card = CustomCardGroupWidget(title=self.tr("物体分割AI能力"), content=self.tr("智能分割图像中的物体"), parent=self)
        object_segmentation_card.addWidget(object_segmentation_status, stretch=0)
        object_segmentation_card.addWidget(object_segmentation_switch, stretch=0)
        object_segmentation_card.addWidget(object_segmentation_chevron, stretch=0)
        object_segmentation_card.vBoxLayout.addWidget(object_segmentation_panel)
        object_segmentation_card.setSeparatorVisible(True)
        ai_settings_cards.append(object_segmentation_card)

        ocr_switch = ToggleSwitch()
        self.ai_toggle_switchs.append(ocr_switch)
        ocr_switch.setActive(cfg.get(cfg.localOCREnabled))
        ocr_switch.toggled.connect(lambda flag: cfg.set(cfg.localOCREnabled, flag))
        ocr_status = StatusBadge(text=self.tr("未启用"), color="#eab308", name="ocr")
        try:
            text = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{ocr_status.name}_status_info"]["text"]
            color = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{ocr_status.name}_status_info"]["color"]
            ocr_status.setLabel(text=text, color=color)
        except Exception:
            pass
        ocr_panel = ModelVariantPanel(config_key="ocr", parent=self)
        self._bind_ai_toggle(
            switch=ocr_switch,
            badge=ocr_status,
            local_ai_type="ocr",
            panel=ocr_panel
        )
        ocr_chevron = self._create_chevron_btn(ocr_panel)
        ocr_card = CustomCardGroupWidget(title=self.tr("OCR 能力"), content=self.tr("智能识别提取图片中的文字"), parent=self)
        ocr_card.addWidget(ocr_status, stretch=0)
        ocr_card.addWidget(ocr_switch, stretch=0)
        ocr_card.addWidget(ocr_chevron, stretch=0)
        ocr_card.vBoxLayout.addWidget(ocr_panel)
        ocr_card.setSeparatorVisible(True)
        ai_settings_cards.append(ocr_card)

        video_inpainting_switch = ToggleSwitch()
        self.ai_toggle_switchs.append(video_inpainting_switch)
        video_inpainting_switch.setActive(cfg.get(cfg.localVideoInpaintingEnabled))
        video_inpainting_switch.toggled.connect(lambda flag: cfg.set(cfg.localVideoInpaintingEnabled, flag))
        video_inpainting_status = StatusBadge(text=self.tr("未启用"), color="#eab308", name="video_inpainting")
        try:
            text = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{video_inpainting_status.name}_status_info"]["text"]
            color = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{video_inpainting_status.name}_status_info"]["color"]
            video_inpainting_status.setLabel(text=text, color=color)
        except Exception:
            pass
        video_inpainting_panel = ModelVariantPanel(config_key="video_inpainting", parent=self)
        self._bind_ai_toggle(
            switch=video_inpainting_switch,
            badge=video_inpainting_status,
            local_ai_type="video_inpainting",
            panel=video_inpainting_panel
        )
        video_inpainting_chevron = self._create_chevron_btn(video_inpainting_panel)
        video_inpainting_card = CustomCardGroupWidget(title=self.tr("视频修复AI能力"), content=self.tr("视频物体移除、水印去除等"), parent=self)
        video_inpainting_card.addWidget(video_inpainting_status, stretch=0)
        video_inpainting_card.addWidget(video_inpainting_switch, stretch=0)
        video_inpainting_card.addWidget(video_inpainting_chevron, stretch=0)
        video_inpainting_card.vBoxLayout.addWidget(video_inpainting_panel)
        video_inpainting_card.setSeparatorVisible(True)
        ai_settings_cards.append(video_inpainting_card)

        object_tracking_switch = ToggleSwitch()
        self.ai_toggle_switchs.append(object_tracking_switch)
        object_tracking_switch.setActive(cfg.get(cfg.localObjectTrackingEnabled))
        object_tracking_switch.toggled.connect(lambda flag: cfg.set(cfg.localObjectTrackingEnabled, flag))
        object_tracking_status = StatusBadge(text=self.tr("未启用"), color="#eab308", name="object_tracking")
        try:
            text = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{object_tracking_status.name}_status_info"]["text"]
            color = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{object_tracking_status.name}_status_info"]["color"]
            object_tracking_status.setLabel(text=text, color=color)
        except Exception:
            pass
        object_tracking_panel = ModelVariantPanel(config_key="tracker", parent=self)
        self._bind_ai_toggle(
            switch=object_tracking_switch,
            badge=object_tracking_status,
            local_ai_type="tracker",
            panel=object_tracking_panel
        )
        object_tracking_chevron = self._create_chevron_btn(object_tracking_panel)
        object_tracking_card = CustomCardGroupWidget(title=self.tr("对象跟踪AI能力"), content=self.tr("智能跟踪视频中的目标对象"), parent=self)
        object_tracking_card.addWidget(object_tracking_status, stretch=0)
        object_tracking_card.addWidget(object_tracking_switch, stretch=0)
        object_tracking_card.addWidget(object_tracking_chevron, stretch=0)
        object_tracking_card.vBoxLayout.addWidget(object_tracking_panel)
        object_tracking_card.setSeparatorVisible(True)
        ai_settings_cards.append(object_tracking_card)

        image_edit_switch = ToggleSwitch()
        self.ai_toggle_switchs.append(image_edit_switch)
        image_edit_switch.setActive(cfg.get(cfg.localImageEditEnabled))
        image_edit_switch.toggled.connect(lambda flag: cfg.set(cfg.localImageEditEnabled, flag))
        image_edit_status = StatusBadge(text=self.tr("未启用"), color="#eab308", name="image_edit")
        try:
            text = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{image_edit_status.name}_status_info"]["text"]
            color = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{image_edit_status.name}_status_info"]["color"]
            image_edit_status.setLabel(text=text, color=color)
        except Exception:
            pass
        image_edit_panel = ModelVariantPanel(config_key="image_edit", parent=self)
        self._bind_ai_toggle(
            switch=image_edit_switch,
            badge=image_edit_status,
            local_ai_type="image_edit",
            panel=image_edit_panel
        )
        image_edit_chevron = self._create_chevron_btn(image_edit_panel)
        image_edit_card = CustomCardGroupWidget(title=self.tr("图像编辑AI能力"), content=self.tr("智能修复、增强与编辑图像内容"), parent=self)
        image_edit_card.addWidget(image_edit_status, stretch=0)
        image_edit_card.addWidget(image_edit_switch, stretch=0)
        image_edit_card.addWidget(image_edit_chevron, stretch=0)
        image_edit_card.vBoxLayout.addWidget(image_edit_panel)
        image_edit_card.setSeparatorVisible(True)
        ai_settings_cards.append(image_edit_card)

        for card in ai_settings_cards:
            ai_settings.addCard(card=card)

        return ai_settings
    
    def _create_performance_settings(self):
        performance_settings_cards = []
        performance_settings = CustomGroupBox(title=self.tr("🌟 高级设置"))

        log_level_combox = ComboBox()
        current_level = cfg.get(cfg.logLevel)
        setFont(log_level_combox, 14)
        log_level_combox.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        log_level_combox.currentTextChanged.connect(lambda text: cfg.set(cfg.logLevel, text.upper()))
        index = log_level_combox.findText(current_level.upper() if isinstance(current_level, str) else current_level)
        if index >= 0:
            log_level_combox.setCurrentIndex(index)
        log_level_card = CustomCardGroupWidget(title=self.tr("日志级别"), content=self.tr("设置日志记录详细程度"), parent=self)
        log_level_card.addWidget(log_level_combox, stretch=0)
        log_level_card.setSeparatorVisible(True)
        performance_settings_cards.append(log_level_card)

        task_parallel_number_combox = ComboBox()
        current_number = int(cfg.get(cfg.taskParallelNumber))
        setFont(task_parallel_number_combox, 14)
        task_parallel_number_combox.addItems([str(n) for n in [1, 2, 4, 8, 16]])
        task_parallel_number_combox.currentTextChanged.connect(lambda number: cfg.set(cfg.taskParallelNumber, int(number)))
        index = task_parallel_number_combox.findText(str(current_number))
        if index >= 0:
            task_parallel_number_combox.setCurrentIndex(index)
        task_parallel_number_card = CustomCardGroupWidget(title=self.tr("任务并行数"), content=self.tr("设置任务并行执行数量（重启软件生效）"), parent=self)
        task_parallel_number_card.addWidget(task_parallel_number_combox, stretch=0)
        task_parallel_number_card.setSeparatorVisible(True)
        performance_settings_cards.append(task_parallel_number_card)

        hardware_optimization_combox = ComboBox()
        current_hardware = cfg.get(cfg.hardwareOptimizationType)
        setFont(hardware_optimization_combox, 14)
        hardware_optimization_combox.addItems(["Auto", "CPU", "GPU"])
        index = hardware_optimization_combox.findText(current_hardware)
        if index >= 0:
            hardware_optimization_combox.setCurrentIndex(index)
        hardware_optimization_combox.currentTextChanged.connect(self._on_hardware_type_changed)
        hardware_optimization_card = CustomCardGroupWidget(title=self.tr("硬件加速"), content=self.tr("设置硬件加速类型"), parent=self)
        hardware_optimization_card.addWidget(hardware_optimization_combox, stretch=0)
        hardware_optimization_card.setSeparatorVisible(True)
        performance_settings_cards.append(hardware_optimization_card)

        for card in performance_settings_cards:
            performance_settings.addCard(card=card)

        return performance_settings
    
    def _select_path(self, widget):
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择文件夹",
            "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog if sys.platform == "darwin" else QFileDialog.Option(0)
        )
        if directory:
            if isinstance(directory, str) and not directory.isascii():
                    MessageBox(title=self.tr("提醒"), content="不支持非英文路径", parent=self.window()).exec()
                    return
            widget.setText(directory)

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
            QPushButton:pressed {{
                background: {hover};
                padding: 9px 16px;
                margin-top: 1px;
            }}
        """
    
    def _update_toggle_switch_off(self, value):
        for switch in self.ai_toggle_switchs:
            old_active = switch.isActive()
            switch.setActive(False)
            if old_active is False:
                switch.toggled.emit(False)
    
    def _bind_ai_toggle(self, switch: ToggleSwitch, badge: StatusBadge, local_ai_type: str, panel: ModelVariantPanel):
        switch.toggled.connect(
            lambda flag, switch=switch, badge=badge, local_ai_type=local_ai_type, panel=panel: 
            self._ai_switch_on_toggle(flag=flag, switch=switch, badge=badge, local_ai_type=local_ai_type, panel=panel)
        )

    def _ai_switch_on_toggle(self, flag: bool, switch: ToggleSwitch, badge: StatusBadge, local_ai_type: str, panel: ModelVariantPanel):
        if flag:
            variant = panel.get_variant()
            badge.setLabel(text=self.tr("环境初始化中…"), color="#60a5fa")

            progress_dialog = InitProgressDialog(title=self.tr("正在初始化环境..."), variant=variant, parent=self)
            progress_dialog.disableCloseBtn()
            progress_dialog.show()

            worker = InitWorker(task_name=local_ai_type, variant=variant, use_mirror=panel.use_mirror(), parent=progress_dialog)
            worker.signals.progress.connect(progress_dialog.append_log)
            worker.signals.finished.connect(
                lambda ok, msg, switch=switch, badge=badge, progress_dialog=progress_dialog: 
                self._on_init_finished(ok, msg, switch, badge, progress_dialog, panel)
            )
            InternalTaskManager.get_pool().start(worker)
        else:
            badge.setLabel(text=self.tr("未启用"), color="#eab308")
            tmp = cfg.get(cfg.additionalParams).get("LocalAISettings", {})
            tmp.update({f"{badge.name}_status_info": {"text": self.tr("未启用"), "color": "#eab308"}})
            cfg.additionalParams.value.update({"LocalAISettings": tmp})
            panel.check_model_status()

    def _on_init_finished(
            self, ok: bool,
            error: str,
            switch: ToggleSwitch, 
            badge: StatusBadge, 
            progress_dialog: InitProgressDialog,
            panel: ModelVariantPanel
        ):
        if ok:
            panel.check_model_status()
            badge.setLabel(text=self.tr("已启用"), color="#22c55e")
            tmp = cfg.get(cfg.additionalParams).get("LocalAISettings", {})
            tmp.update({f"{badge.name}_status_info": {"text": self.tr("已启用"), "color": "#22c55e"}})
            cfg.additionalParams.value.update({"LocalAISettings": tmp})
            progress_dialog.accept()
        else:
            switch.setActive(False)
            badge.setLabel(text=self.tr("启用失败"), color="#ef4444")
            tmp = cfg.get(cfg.additionalParams).get("LocalAISettings", {})
            tmp.update({f"{badge.name}_status_info": {"text": self.tr("启用失败"), "color": "#ef4444"}})
            cfg.additionalParams.value.update({"LocalAISettings": tmp})
            progress_dialog.enableCloseBtn()
            progress_dialog.append_log(f"\n❌ 错误信息：{error}")
            progress_dialog.progress.setRange(0, 1)

    def _on_hardware_type_changed(self, value: str):
        cfg.set(cfg.hardwareOptimizationType, value)
        if value == "CPU":
            variant = "cpu"
        elif value == "GPU":
            variant = "gpu"
        else:
            status, _ = global_backend_info_cache.get()
            variant = "gpu" if "GPU" in status else "cpu"
        enabled_map = {
            "blind_watermark_addition": ("盲水印AI能力", cfg.get(cfg.localBlindWatermarkEnabled)),
            "visible_watermark_removal": ("水印去除AI能力", cfg.get(cfg.localWatermarkRemovalEnabled)),
            "segment": ("物体分割AI能力", cfg.get(cfg.localObjectSegmentationEnabled)),
            "ocr": ("OCR 能力", cfg.get(cfg.localOCREnabled)),
            "video_inpainting": ("视频修复AI能力", cfg.get(cfg.localVideoInpaintingEnabled)),
            "tracker": ("对象跟踪AI能力", cfg.get(cfg.localObjectTrackingEnabled)),
            "image_edit": ("图像编辑AI能力", cfg.get(cfg.localImageEditEnabled)),
        }
        deps_path = cfg.get(cfg.localAIModelDeps)
        missing = []
        for key, (name, enabled) in enabled_map.items():
            if not enabled:
                continue
            model_path = os.path.join(deps_path, variant, key) if deps_path else ""
            if not model_path or not os.path.exists(model_path) or not os.listdir(model_path):
                missing.append(name)
        if missing:
            content = self.tr("以下已激活的AI能力尚未下载对应的 {variant} 模型：\n\n").format(variant=variant.upper())
            content += "\n".join(f"  • {name}" for name in missing)
            content += self.tr("\n\n请在各模型设置中下载对应硬件版本的模型，否则相关功能将无法正常使用。")
            MessageBox(title=self.tr("模型缺失提醒"), content=content, parent=self.window()).exec()

