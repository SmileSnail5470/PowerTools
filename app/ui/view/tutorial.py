from collections.abc import Callable
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from app.ui.library.qfluentwidgets import (
    CaptionLabel,
    FlowLayout,
    FluentIcon as FIF,
    IconWidget,
    InfoBar,
    InfoBarPosition,
    PushButton,
    ScrollArea,
    SimpleCardWidget,
    StrongBodyLabel,
    SubtitleLabel,
    setFont,
)
from app.ui.widgets.gradient_header_widget import GradientHeader


class GuideCard(SimpleCardWidget):
    def __init__(
        self,
        icon: object,
        title: str,
        content: str,
        tags: tuple[str, ...] = (),
        actions: tuple[tuple[str, object, Callable[[], None]], ...] = (),
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("guideCard")
        self.setBorderRadius(12)
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(10)

        icon_container = QFrame(self)
        icon_container.setObjectName("guideIconContainer")
        icon_container.setFixedSize(36, 36)
        icon_container.setStyleSheet(
            """
            QFrame#guideIconContainer {
                background: #f0efff;
                border: 1px solid #e2dfff;
                border-radius: 10px;
            }
            """
        )
        icon_layout = QVBoxLayout(icon_container)
        icon_layout.setContentsMargins(9, 9, 9, 9)
        icon_widget = IconWidget(icon, icon_container)
        icon_widget.setFixedSize(18, 18)
        icon_layout.addWidget(icon_widget)

        title_label = StrongBodyLabel(title, self)
        title_label.setWordWrap(True)
        header_layout.addWidget(icon_container, 0, Qt.AlignTop)
        header_layout.addWidget(title_label, 1, Qt.AlignVCenter)
        layout.addLayout(header_layout)

        content_label = CaptionLabel(content, self)
        content_label.setWordWrap(True)
        content_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        content_label.setTextColor("#606060", "#d0d0d0")
        layout.addWidget(content_label)
        layout.addStretch(1)

        if tags:
            tag_layout = FlowLayout(needAni=False)
            tag_layout.setContentsMargins(0, 0, 0, 0)
            tag_layout.setHorizontalSpacing(6)
            tag_layout.setVerticalSpacing(6)
            for tag in tags:
                tag_layout.addWidget(self._create_tag(tag))
            layout.addLayout(tag_layout)

        if actions:
            action_layout = FlowLayout(needAni=False)
            action_layout.setContentsMargins(0, 2, 0, 0)
            action_layout.setHorizontalSpacing(8)
            action_layout.setVerticalSpacing(8)
            for text, action_icon, callback in actions:
                button = PushButton(text, self, action_icon)
                button.setFixedHeight(30)
                button.setCursor(Qt.PointingHandCursor)
                button.setAccessibleName(text)
                button.clicked.connect(callback)
                action_layout.addWidget(button)
            layout.addLayout(action_layout)

    def _create_tag(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        setFont(label, fontSize=11, weight=QFont.DemiBold)
        label.setStyleSheet(
            """
            QLabel {
                color: #5146b8;
                background: #f0efff;
                border: 1px solid #dedbff;
                border-radius: 9px;
                padding: 3px 8px;
            }
            """
        )
        return label


class Tutorial(QWidget):
    settingsRequested = Signal()

    RELEASES_URL = "https://pan.quark.cn/list#/list/all/0036d165fa7d4b288c1efd4f35d98082-PowerTools%20%E8%BD%AF%E4%BB%B6"
    FFMPEG_URL = "https://pan.quark.cn/list#/list/all/0036d165fa7d4b288c1efd4f35d98082-PowerTools%20%E8%BD%AF%E4%BB%B6/8a2962595bfe4906be261f67260c622c-%E8%A7%86%E9%A2%91%E5%A4%84%E7%90%86%E8%BD%AF%E4%BB%B6*101windows"
    CUDA_URL = "https://pan.quark.cn/list#/list/all/0036d165fa7d4b288c1efd4f35d98082-PowerTools%20%E8%BD%AF%E4%BB%B6/f9b0116b503f47e8a9d2d469a0ea3dde-%E6%98%BE%E5%8D%A1%E4%BE%9D%E8%B5%96%E9%A9%B1%E5%8A%A8"
    CUDNN_URL = "https://pan.quark.cn/list#/list/all/0036d165fa7d4b288c1efd4f35d98082-PowerTools%20%E8%BD%AF%E4%BB%B6/f9b0116b503f47e8a9d2d469a0ea3dde-%E6%98%BE%E5%8D%A1%E4%BE%9D%E8%B5%96%E9%A9%B1%E5%8A%A8"
    MODEL_URL = "https://pan.quark.cn/list#/list/all/410981f94d9843e680b09e971778d7f7-PowerTools%20%E6%A8%A1%E5%9E%8B"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("Tutorial")
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
        header_layout.setContentsMargins(30, 10, 30, 10)
        header_layout.setSpacing(16)

        title_container = QWidget(header)
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(1)

        title_label = QLabel(self.tr("📖 使用指南"), title_container)
        setFont(title_label, fontSize=24, weight=QFont.Bold)
        title_label.setStyleSheet("color: white;")

        title_layout.addWidget(title_label)
        header_layout.addWidget(title_container)
        header_layout.addStretch(1)

        download_button = PushButton(self.tr("📥 下载最新版"), header)
        download_button.setCursor(Qt.PointingHandCursor)
        download_button.setAccessibleName(self.tr("打开 PowerTools Releases 页面"))
        download_button.setStyleSheet(
            """
            PushButton {
                color: white;
                background-color: rgba(255, 255, 255, 0.20);
                border: 1px solid rgba(255, 255, 255, 0.38);
                border-radius: 8px;
                padding: 7px 14px;
            }
            PushButton:hover {
                background-color: rgba(255, 255, 255, 0.30);
            }
            PushButton:pressed {
                background-color: rgba(255, 255, 255, 0.16);
            }
            """
        )
        download_button.clicked.connect(
            lambda: self._open_url(self.RELEASES_URL)
        )
        header_layout.addWidget(download_button, 0, Qt.AlignVCenter)

        main_layout.addWidget(header)

    def _setup_content(self, main_layout: QVBoxLayout):
        scroll = ScrollArea(self)
        scroll.setObjectName("tutorialScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.enableTransparentBackground()

        page = QWidget(scroll)
        page_layout = QHBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)

        content = QWidget(page)
        content.setObjectName("tutorialContent")
        content.setMaximumWidth(1080)
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 20, 24, 48)
        content_layout.setSpacing(28)
        content_layout.setAlignment(Qt.AlignTop)

        content_layout.addWidget(self._create_environment_section())
        content_layout.addWidget(self._create_feature_section())
        content_layout.addWidget(self._create_quick_start_section())
        content_layout.addWidget(self._create_tips_section())

        page_layout.addWidget(content, 1, Qt.AlignTop)
        scroll.setWidget(page)
        main_layout.addWidget(scroll, 1)

    def _create_quick_start_section(self) -> QWidget:
        cards = [
            GuideCard(
                FIF.TILES,
                self.tr("1 · 选择工具"),
                self.tr("从左侧导航进入需要的功能。"),
                parent=self,
            ),
            GuideCard(
                FIF.FOLDER,
                self.tr("2 · 添加素材"),
                self.tr("选择图片或视频，再按页面提示设置参数。"),
                parent=self,
            ),
            GuideCard(
                FIF.PLAY,
                self.tr("3 · 开始处理"),
                self.tr("确认参数后提交任务，并在完成后检查结果。"),
                parent=self,
            ),
        ]
        return self._create_section(
            self.tr("快速开始"),
            self.tr("只需三步即可完成一次任务。"),
            cards,
            columns=3,
        )

    def _create_environment_section(self) -> QWidget:
        cards = [
            GuideCard(
                FIF.SPEED_HIGH,
                self.tr("GPU 加速"),
                self.tr("AI 加速需 CUDA 12.x 与 cuDNN 9.x，安装后在设置中验证。"),
                tags=(self.tr("可选"), self.tr("NVIDIA")),
                actions=(
                    (
                        self.tr("CUDA"),
                        FIF.DOWNLOAD,
                        lambda: self._open_url(self.CUDA_URL),
                    ),
                    (
                        self.tr("cuDNN"),
                        FIF.DOWNLOAD,
                        lambda: self._open_url(self.CUDNN_URL),
                    ),
                    (
                        self.tr("设置"),
                        FIF.SETTING,
                        self.settingsRequested.emit,
                    ),
                ),
                parent=self,
            ),
            GuideCard(
                FIF.ROBOT,
                self.tr("本地 AI 模型"),
                self.tr("在「本地AI设置」选择模型目录，再启用所需能力。若手动下载，将 CPU/GPU 目录放到模型目录下，并改为小写 cpu/gpu。"),
                tags=(self.tr("按需下载"),),
                actions=(
                    (
                        self.tr("手动下载(可选)"),
                        FIF.DOWNLOAD,
                        lambda: self._open_url(self.MODEL_URL),
                    ),
                    (
                        self.tr("打开设置"),
                        FIF.SETTING,
                        self.settingsRequested.emit,
                    ),
                ),
                parent=self,
            ),
            GuideCard(
                FIF.VIDEO,
                self.tr("FFmpeg"),
                self.tr("视频读写需要 FFmpeg，请选择其 bin 目录并完成验证。"),
                tags=(self.tr("视频任务"),),
                actions=(
                    (
                        self.tr("下载"),
                        FIF.DOWNLOAD,
                        lambda: self._open_url(self.FFMPEG_URL),
                    ),
                    (
                        self.tr("设置"),
                        FIF.SETTING,
                        self.settingsRequested.emit,
                    ),
                ),
                parent=self,
            ),
        ]
        return self._create_section(
            self.tr("环境准备"),
            self.tr("普通图片任务可直接使用，视频和 AI 功能需额外配置。"),
            cards,
            columns=3,
        )

    def _create_feature_section(self) -> QWidget:
        cards = [
            GuideCard(
                FIF.ADD_TO,
                self.tr("水印添加"),
                self.tr("设置内容、位置与透明度，支持可见和盲水印。"),
                tags=(self.tr("盲水印模型"),),
                parent=self,
            ),
            GuideCard(
                FIF.ERASE_TOOL,
                self.tr("水印移除"),
                self.tr("自动检测或手动选区，选择模型后完成修复。"),
                tags=(self.tr("水印去除"), self.tr("物体分割"), self.tr("图像编辑"), self.tr("视频修复"), self.tr("对象跟踪(视频)")),
                parent=self,
            ),
            GuideCard(
                FIF.BROOM,
                self.tr("暗印去除"),
                self.tr("匹配暗印类型，可按需修复颜色与视频时序。"),
                tags=(self.tr("图像编辑"),),
                parent=self,
            ),
            GuideCard(
                FIF.DOCUMENT,
                self.tr("文字提取"),
                self.tr("设置语言与置信度，识别后校对低清晰度文字。"),
                tags=(self.tr("OCR"),),
                parent=self,
            ),
            GuideCard(
                FIF.BRUSH,
                self.tr("图像编辑"),
                self.tr("使用提示词和选区描述目标效果，建议小范围迭代。"),
                tags=(self.tr("图像编辑"),),
                parent=self,
            ),
        ]
        return self._create_section(
            self.tr("功能概览"),
            self.tr("AI 功能首次使用前，请先在设置中下载对应模型。"),
            cards,
            columns=3,
        )

    def _create_tips_section(self) -> QWidget:
        cards = [
            GuideCard(
                FIF.SPEED_MEDIUM,
                self.tr("合理设置并行数"),
                self.tr("根据机器内存或显存设置任务并行数，建议 1。"),
                actions=(
                    (
                        self.tr("调整设置"),
                        FIF.SETTING,
                        self.settingsRequested.emit,
                    ),
                ),
                parent=self,
            ),
            GuideCard(
                FIF.SAVE_COPY,
                self.tr("先备份，再试跑"),
                self.tr("不要覆盖唯一原件。先用 3–5 个副本确认参数和输出质量。"),
                tags=(self.tr("安全提示"),),
                parent=self,
            ),
        ]
        return self._create_section(
            self.tr("使用建议"),
            self.tr("正式批处理前做好资源评估和素材备份。"),
            cards,
            columns=2,
        )

    def _create_section(
        self,
        title: str,
        description: str,
        cards: list[GuideCard],
        columns: int,
    ) -> QWidget:
        section = QWidget(self)
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(10)

        title_label = SubtitleLabel(title, section)
        description_label = CaptionLabel(description, section)
        description_label.setWordWrap(True)
        description_label.setTextColor("#606060", "#d0d0d0")
        section_layout.addWidget(title_label)
        section_layout.addWidget(description_label)

        grid = QGridLayout()
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        for index, card in enumerate(cards):
            grid.addWidget(card, index // columns, index % columns)
        for column in range(columns):
            grid.setColumnStretch(column, 1)
        section_layout.addLayout(grid)
        return section

    def _open_url(self, url: str):
        if QDesktopServices.openUrl(QUrl(url)):
            return
        InfoBar.error(
            title=self.tr("无法打开链接"),
            content=self.tr(
                "请检查系统默认浏览器设置，或手动复制链接：{url}"
            ).format(url=url),
            duration=5000,
            parent=self,
            position=InfoBarPosition.TOP,
        )
