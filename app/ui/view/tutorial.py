from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.ui.library.qfluentwidgets import ScrollArea, setFont
from app.ui.widgets.custom_card_group_widget import CustomGroupBox
from app.ui.widgets.gradient_header_widget import GradientHeader


class Tutorial(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("Tutorial")
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(12)

        header = GradientHeader(parent=self)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 20, 30, 20)
        title = QLabel(self.tr("📖 软件教程"))
        setFont(title, fontSize=24, weight=QFont.Bold)
        title.setStyleSheet("QLabel { color: white; }")
        header_layout.addWidget(title)
        header_layout.addStretch()
        main_layout.addWidget(header)

        scroll = ScrollArea()
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(16)
        content_layout.setAlignment(Qt.AlignTop)

        sections = [
            (
                self.tr("🧰 环境配置"),
                [
                    self.tr("1. 基础环境：Windows 或 macOS 可直接使用 Release 安装包；视频处理必须先准备 FFmpeg。"),
                    self.tr("2. FFmpeg：下载完整构建版，在“常规 → 软件配置”中选择 ffmpeg/ffmpeg.exe 所在的 bin 目录，并完成验证。"),
                    self.tr("3. AI 模型：可在“常规 → 本地AI设置”中启用所需能力自动下载，或把“AI模型依赖路径”指向包含 cpu/GPU 子目录的模型根目录。"),
                    self.tr("4. GPU（可选）：NVIDIA GPU 推荐 CUDA 12.x 与 cuDNN 9.x；CUDA、cuDNN 配置项应填写 DLL 文件实际所在的 bin 目录。"),
                    self.tr("5. 缓存与输出：请确保目录存在足够可用空间并具有读写权限。所有配置路径、输入输出路径及文件名均支持中文和空格。"),
                ],
            ),
            (
                self.tr("🚀 软件使用说明"),
                [
                    self.tr("1. 在左侧选择水印添加、水印移除、文字提取、暗印去除或图像编辑。"),
                    self.tr("2. 点击选择器或拖放文件；支持批处理的页面也可选择目录。图片支持 PNG/JPG/JPEG/BMP/AVIF/WEBP，视频支持 MP4/AVI/MOV/MKV。"),
                    self.tr("3. 选择算法或模型，按任务需要设置检测方式、内容、强度、阈值、输出格式和保存目录。"),
                    self.tr("4. 点击处理按钮，确认任务信息后提交。右侧区域会显示准备、检测/识别、处理和导出进度。"),
                    self.tr("5. 完成后可在结果预览中查看输出；批量任务会逐个处理目录中的受支持文件。"),
                ],
            ),
            (
                self.tr("🎛 参数含义"),
                [
                    self.tr("水印添加：位置决定水印锚点；不透明度越高越明显；旋转控制角度；缩放控制水印相对尺寸；输出格式可保持原格式或转换。"),
                    self.tr("水印移除：检测方式可选自动、交互或手动 Mask；Mask 扩张系数越大，覆盖边缘越宽；检测置信度越高，结果越严格；修复模型决定速度、显存占用和画面效果。"),
                    self.tr("文字提取：识别语言应与原图文字匹配；识别置信度是保留文字结果的最低分数，提高可减少误识别，也可能漏掉模糊文字。"),
                    self.tr("暗印去除：模型决定对应的暗水印类型；颜色修复用于减轻色偏；高质量输出通常更慢且占用更多内存；保留区域可避免指定范围被修改。"),
                    self.tr("图像编辑：提示词描述目标效果；Mask/框选区域限定编辑范围；不同模型的生成质量、速度和硬件需求不同。"),
                    self.tr("通用参数：输出目录决定结果位置；保持原格式会沿用输入扩展名；硬件模式自动选择 CPU/GPU 对应模型。"),
                ],
            ),
            (
                self.tr("⚠️ 软件限制"),
                [
                    self.tr("1. AI 功能必须先启用并具备完整模型文件；GPU 模型需要兼容的 NVIDIA 驱动、CUDA 和 cuDNN，环境不可用时将回退或无法启动。"),
                    self.tr("2. 视频解析、合成和部分预览依赖 FFmpeg；未正确配置时视频任务不可用。长视频和高分辨率素材需要较多缓存空间、内存和处理时间；兼容预览可能在系统临时目录额外使用接近一份源视频大小的空间。"),
                    self.tr("3. 复杂背景、透明/动态水印、严重压缩、低清晰度文字或超大遮挡可能降低检测、修复和 OCR 效果，需要调整阈值、Mask 或更换模型。"),
                    self.tr("4. 高分辨率视频、满屏水印和高质量模型可能需要 8–16 GB 显存；显存不足时应降低分辨率、改用 CPU 或选择轻量模型。"),
                    self.tr("5. 批处理只处理软件支持的媒体格式。请勿覆盖唯一原件，重要素材应先备份，并确认输出目录有足够空间。"),
                ],
            ),
        ]

        for heading, paragraphs in sections:
            group = CustomGroupBox(title=heading)
            group.addCard(card=self._create_text_card(paragraphs))
            content_layout.addWidget(group)

        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.enableTransparentBackground()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        main_layout.addWidget(scroll)

    def _create_text_card(self, paragraphs: list[str]) -> QWidget:
        card = QFrame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(9)
        for paragraph in paragraphs:
            label = QLabel(paragraph)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            setFont(label, 12)
            label.setStyleSheet("QLabel { color: #374151; line-height: 1.5; }")
            layout.addWidget(label)
        return card
