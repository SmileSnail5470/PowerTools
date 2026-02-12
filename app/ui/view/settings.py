import hashlib
import logging
import os
import platform
import shutil
import ssl
import subprocess
import sys
import tarfile
import zipfile
from PySide6.QtCore import Qt, Signal, QThreadPool, QRunnable, QObject, QThread
from PySide6.QtWidgets import(
    QHBoxLayout, QWidget, QVBoxLayout, QLabel, QFrame, QLineEdit, QPushButton, QFileDialog,
    QSizePolicy, QDialog, QProgressBar, QTextEdit
)
from PySide6.QtGui import QFont
import urllib.request

from app.ui.library.qfluentwidgets import(
    setFont, ScrollArea, TeachingTip, InfoBarIcon, TeachingTipTailPosition, FluentIcon,
    ComboBox, Theme
)
from app.ui.widgets.gradient_header_widget import GradientHeader
from app.ui.widgets.custom_card_group_widget import CustomCardGroupWidget, CustomGroupBox
from app.ui.widgets.toggle_switch_widget import ToggleSwitch
from app.ui.library.qframelesswindow.titlebar import CloseButton
from app.ui.common.config import cfg, Language
from app.utils.logger import get_log_manager
from app.utils.logger.decorators import log_exception, log_function_call


models_deps_urls = {
    "visible_watermark_removal": {
        "url": "",
        "sha256": ""
    },
    "blind_watermark_addition": {
        "url": "",
        "sha256": ""
    },
    "ocr": {
        "url": "",
        "sha256": ""
    },
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
    def __init__(self, task_name: str, parent: QObject = None):
        super().__init__()
        self.task_name = task_name
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
    
    def _download(self, url, output_path, expected_size=None, expected_sha256="", extract_to=None):
        max_retries = 3
        backoff = 0.5
        chunk_size = 1024 * 512
        temp_path = output_path + ".part"
        downloaded = 0

        if os.path.exists(temp_path):
            downloaded = os.path.getsize(temp_path)

        context = ssl._create_unverified_context()
        for attempt in range(1, max_retries + 1):
            try:
                headers = {}
                if downloaded > 0:
                    headers["Range"] = f"bytes={downloaded}-"

                req = urllib.request.Request(url, headers=headers)

                with urllib.request.urlopen(req, context=context) as resp:
                    with open(temp_path, "ab") as f:
                        while True:
                            chunk = resp.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                break
            except Exception as e:
                if attempt >= max_retries:
                    raise Exception(f"download failed: {e}")
                wait = backoff * (2 ** (attempt - 1))
                QThread.msleep(wait * 1000)

        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_path, output_path)

        if expected_size is not None:
            actual_size = os.path.getsize(output_path)
            if actual_size != expected_size:
                raise Exception(f"download file size check failed: expected {expected_size}, actual {actual_size}")

        if expected_sha256 is not None:
            actual_sha256 = self._sha256_of_file(output_path)
            if actual_sha256.lower() != expected_sha256.lower():
                raise Exception(f"SHA256 check failed: \n Expected: {expected_sha256}\n Actual: {actual_sha256}")

        if extract_to:
            os.makedirs(extract_to, exist_ok=True)
            self._extract_if_needed(output_path, extract_to)
            os.remove(output_path)

    @log_function_call(logger=logging.getLogger("UI"), level=logging.INFO)
    def _download_module(self):
        logger.info(f"start download {self.task_name} module.")
        resources_url = models_deps_urls[self.task_name]["url"]
        if not resources_url:
            raise Exception(f"{self.task_name} download_url not exist.")
        self._download(
            url=resources_url,
            output_path=os.path.join(self.deps_path, "modules_deps.zip"),
            expected_sha256=models_deps_urls[self.task_name]["sha256"],
            extract_to=self.deps_path
        )
        logger.info(f"download {self.task_name} module success.")

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
        current_task_deps_path = os.path.join(self.deps_path, self.task_name)
        if not os.path.exists(current_task_deps_path):
            raise Exception(f"{self.task_name} deps valid failed for {current_task_deps_path} not exist.")
        if not os.listdir(current_task_deps_path):
            raise Exception(f"{self.task_name} deps valid failed for {current_task_deps_path} is empty.")

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


class InitProgressDialog(QDialog):
    def __init__(self, title="", parent=None):
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
        title_label = QLabel(self.tr("🔧 环境初始化中，请稍候…"))
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
        header_layout.addWidget(self.status_label, alignment=Qt.AlignVCenter)

        path_layout = QHBoxLayout()
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
            cfg.additionalParams.value.update({"SoftwareSettings": {f"{self.name}_status_info": {"text": text, "color": color}}})
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

    def _update_global_config(self, path: str):
        self.global_config_params_name_map[self.name.lower()].value = path
        # 状态提示信息恢复默认
        text = "未验证"
        self.status = ""
        color = self._get_status_badge_color()
        self.status_label.setLabel(text=text, color=color)
        cfg.additionalParams.value.update({"SoftwareSettings": {f"{self.name}_status_info": {"text": text, "color": color}}})

class Settings(QWidget):
    thread_pool = QThreadPool.globalInstance()
    
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

        self.software_cards = [ffmpeg_card]

        for card in self.software_cards:
            group.addCard(card=card)

        return group
    
    def _create_general_settings(self):
        settings_cards = []
        settings = CustomGroupBox(title=self.tr("⚙️ 通用设置"))
        
        auto_start_switch = ToggleSwitch()
        auto_start_switch.setActive(cfg.get(cfg.autoStartup))
        auto_start_switch.toggled.connect(lambda flag: setattr(cfg.autoStartup, "value", flag))
        auto_start_card = CustomCardGroupWidget(title=self.tr("开机自启动"), content=self.tr("系统启动时自动运行程序（开发中）"), parent=self)
        auto_start_card.addWidget(auto_start_switch, stretch=0)
        auto_start_card.setSeparatorVisible(True)
        settings_cards.append(auto_start_card)

        auto_update_switch = ToggleSwitch()
        auto_update_switch.setActive(cfg.get(cfg.autoUpdate))
        auto_update_switch.toggled.connect(lambda flag: setattr(cfg.autoUpdate, "value", flag))
        auto_update_card = CustomCardGroupWidget(title=self.tr("自动更新"), content=self.tr("自动检查并安装新版本（开发中）"), parent=self)
        auto_update_card.addWidget(auto_update_switch, stretch=0)
        auto_update_card.setSeparatorVisible(True)
        settings_cards.append(auto_update_card)

        self.cache_line_edit = QLineEdit()
        self.cache_line_edit.setText(cfg.get(cfg.cachePath))
        self.cache_line_edit.textChanged.connect(lambda path: setattr(cfg.cachePath, "value", path))
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
        theme_combox.currentTextChanged.connect(lambda text: setattr(cfg.uiTheme, "value", theme_map[text]))
        setFont(theme_combox, 14)
        theme_combox.addItems(["浅色"])
        theme_card = CustomCardGroupWidget(title=self.tr("界面主题"), content=self.tr("选择您喜欢的界面风格"), parent=self)
        theme_card.addWidget(theme_combox, stretch=0)
        theme_card.setSeparatorVisible(True)
        settings_cards.append(theme_card)

        language_combox = ComboBox()
        language_combox.setText(language_map[cfg.language.serialize()])
        language_combox.currentTextChanged.connect(lambda text: setattr(cfg.language, "value", language_map[text]))
        setFont(language_combox, 14)
        language_combox.addItems(["简体中文"])
        language_card = CustomCardGroupWidget(title=self.tr("语言设置"), content=self.tr("选择界面显示语言"), parent=self)
        language_card.addWidget(language_combox, stretch=0)
        language_card.setSeparatorVisible(True)
        settings_cards.append(language_card)

        for card in settings_cards:
            settings.addCard(card=card)

        return settings
    
    def _create_local_ai_settings(self):
        self.ai_toggle_switchs: list[ToggleSwitch] = []
        ai_settings_cards = []
        ai_settings = CustomGroupBox(title=self.tr("🤖 本地AI设置"))

        localAIModelDeps_line_edit = QLineEdit()
        localAIModelDeps_line_edit.setText(cfg.get(cfg.localAIModelDeps))
        localAIModelDeps_line_edit.textChanged.connect(lambda path: setattr(cfg.localAIModelDeps, "value", path))
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
        blind_watermark_switch.toggled.connect(lambda flag: setattr(cfg.localBlindWatermarkEnabled, "value", flag))
        blind_watermark_status = StatusBadge(text=self.tr("未启用"), color="#eab308", name="blind_watermark_addition")
        try:
            text = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{blind_watermark_status.name}_status_info"]["text"]
            color = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{blind_watermark_status.name}_status_info"]["color"]
            blind_watermark_status.setLabel(text=text, color=color)
        except Exception:
            pass
        self._bind_ai_toggle(
            switch=blind_watermark_switch,
            badge=blind_watermark_status,
            local_ai_type="blind_watermark_addition"
        )
        blind_watermark_card = CustomCardGroupWidget(title=self.tr("盲水印AI能力"), content=self.tr("为图像添加不可见的数字水印，保护版权"), parent=self)
        blind_watermark_card.addWidget(blind_watermark_status, stretch=0)
        blind_watermark_card.addWidget(blind_watermark_switch, stretch=0)
        blind_watermark_card.setSeparatorVisible(True)
        ai_settings_cards.append(blind_watermark_card)

        watermark_removal_switch = ToggleSwitch()
        self.ai_toggle_switchs.append(watermark_removal_switch)
        watermark_removal_switch.setActive(cfg.get(cfg.localWatermarkRemovalEnabled))
        watermark_removal_switch.toggled.connect(lambda flag: setattr(cfg.localWatermarkRemovalEnabled, "value", flag))
        watermark_removal_status = StatusBadge(text=self.tr("未启用"), color="#eab308", name="watermark_removal")
        try:
            text = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{watermark_removal_status.name}_status_info"]["text"]
            color = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{watermark_removal_status.name}_status_info"]["color"]
            watermark_removal_status.setLabel(text=text, color=color)
        except Exception:
            pass
        self._bind_ai_toggle(
            switch=watermark_removal_switch,
            badge=watermark_removal_status,
            local_ai_type="visible_watermark_removal"
        )
        watermark_removal_card = CustomCardGroupWidget(title=self.tr("水印去除AI能力"), content=self.tr("智能去除图像中的水印和标志"), parent=self)
        watermark_removal_card.addWidget(watermark_removal_status, stretch=0)
        watermark_removal_card.addWidget(watermark_removal_switch, stretch=0)
        watermark_removal_card.setSeparatorVisible(True)
        ai_settings_cards.append(watermark_removal_card)

        ocr_switch = ToggleSwitch()
        self.ai_toggle_switchs.append(ocr_switch)
        ocr_switch.setActive(cfg.get(cfg.localOCREnabled))
        ocr_switch.toggled.connect(lambda flag: setattr(cfg.localOCREnabled, "value", flag))
        ocr_status = StatusBadge(text=self.tr("未启用"), color="#eab308", name="ocr")
        try:
            text = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{ocr_status.name}_status_info"]["text"]
            color = cfg.get(cfg.additionalParams)["LocalAISettings"][f"{ocr_status.name}_status_info"]["color"]
            ocr_status.setLabel(text=text, color=color)
        except Exception:
            pass
        self._bind_ai_toggle(
            switch=ocr_switch,
            badge=ocr_status,
            local_ai_type="ocr"
        )
        ocr_card = CustomCardGroupWidget(title=self.tr("OCR 能力"), content=self.tr("智能识别提取图片中的文字"), parent=self)
        ocr_card.addWidget(ocr_status, stretch=0)
        ocr_card.addWidget(ocr_switch, stretch=0)
        ocr_card.setSeparatorVisible(True)
        ai_settings_cards.append(ocr_card)

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
        log_level_combox.currentTextChanged.connect(lambda text: setattr(cfg.logLevel, "value", text.upper()))
        index = log_level_combox.findText(current_level.upper() if isinstance(current_level, str) else current_level)
        if index >= 0:
            log_level_combox.setCurrentIndex(index)
        log_level_card = CustomCardGroupWidget(title=self.tr("日志级别"), content=self.tr("设置日志记录详细程度"), parent=self)
        log_level_card.addWidget(log_level_combox, stretch=0)
        log_level_card.setSeparatorVisible(True)
        performance_settings_cards.append(log_level_card)

        for card in performance_settings_cards:
            performance_settings.addCard(card=card)

        return performance_settings
    
    def _select_path(self, widget):
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择文件夹",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
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
    
    def _bind_ai_toggle(self, switch: ToggleSwitch, badge: StatusBadge, local_ai_type: str):
        switch.toggled.connect(
            lambda flag, switch=switch, badge=badge, local_ai_type=local_ai_type: 
            self._ai_switch_on_toggle(flag=flag, switch=switch, badge=badge, local_ai_type=local_ai_type)
        )

    def _ai_switch_on_toggle(self, flag: bool, switch: ToggleSwitch, badge: StatusBadge, local_ai_type: str):
        if flag:
            badge.setLabel(text=self.tr("环境初始化中…"), color="#60a5fa")

            progress_dialog = InitProgressDialog(title=self.tr("正在初始化环境..."), parent=self)
            progress_dialog.disableCloseBtn()
            progress_dialog.show()

            worker = InitWorker(task_name=local_ai_type, parent=progress_dialog)
            worker.signals.progress.connect(progress_dialog.append_log)
            worker.signals.finished.connect(
                lambda ok, msg, switch=switch, badge=badge, progress_dialog=progress_dialog: 
                self._on_init_finished(ok, msg, switch, badge, progress_dialog)
            )
            self.thread_pool.start(worker)
        else:
            badge.setLabel(text=self.tr("未启用"), color="#eab308")

    def _on_init_finished(
            self, ok: bool,
            error: str,
            switch: ToggleSwitch, 
            badge: StatusBadge, 
            progress_dialog: InitProgressDialog
        ):
        if ok:
            badge.setLabel(text=self.tr("已启用"), color="#22c55e")
            cfg.additionalParams.value.update({"LocalAISettings": {f"{badge.name}_status_info": {"text": self.tr("已启用"), "color": "#22c55e"}}})
            progress_dialog.accept()
        else:
            switch.setActive(False)
            badge.setLabel(text=self.tr("启用失败"), color="#ef4444")
            cfg.additionalParams.value.update({"LocalAISettings": {f"{badge.name}_status_info": {"text": self.tr("启用失败"), "color": "#ef4444"}}})
            progress_dialog.enableCloseBtn()
            progress_dialog.append_log(f"\n❌ 错误信息：{error}")
            progress_dialog.progress.setRange(0, 1)

