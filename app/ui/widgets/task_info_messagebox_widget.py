from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout, QScrollArea
)
from app.ui.library.qfluentwidgets import setFont, MessageBoxBase, TeachingTip, InfoBarIcon, TeachingTipTailPosition, SubtitleLabel


param_name_map = {
    "input_path": "文件路径",
    "watermark_type": "水印类型",
    "blind_watermark_task_type": "盲水印任务类型",
    "blind_watermark_model_name": "盲水印算法",
    "watermark_detect_type": "水印检测方式",
    "manual_watermark_mask_path": "人工标注 Mask 路径",
    "watermark_ai_interactive_type": "AI 交互检测水印方式",
    "watermark_detect_prompt": "AI 水印检测提示词",
    "watermark_boxes": "AI 框选检测 Box",
    "watermark_confidence": "水印置信度",
    "mask_dilate": "水印 Mask 扩张系数",
    "model_name": "模型类型",
    "watermark_text": "水印文本",
    "custom_characters": "盲水印字符集",
    "font": "字体",
    "font_size": "字体大小",
    "font_color": "字体颜色",
    "watermark_image": "图片水印位置",
    "watermark_opacity": "水印透明度（%）",
    "watermark_content": "水印样式",
    "watermark_format": "水印形式",
    "watermark_location": "水印位置",
    "watermark_rotation": "水印旋转角度（{0}）".format("\u00B0"),
    "watermark_zoom": "水印缩放比例（%）",
    "output_path": "保存位置",
    "output_format": "保存格式",
    "ocr_rec_language": "识别目标语言",
    "drop_score": "OCR 识别置信度",
    "use_textline_ori": "文本行方向矫正"
}

class TaskInfoMessageBox(MessageBoxBase):
    def __init__(self, params, task_type, parent=None):
        super().__init__(parent)
        self.task_params = params
        self.task_type = task_type
        self.titleLabel = SubtitleLabel(self.tr("任务信息总结"))
        self.viewLayout.addWidget(self.titleLabel)
        self.setModal(True)
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        body = QFrame()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(24, 10, 24, 20)
        body_layout.setSpacing(10)
        
        if self.task_type == "watermark-add":
            value_map = {
                "visible": "可见水印",
                "blind": "盲水印",
                "ImageSettings": "图片水印",
                "TextSettings": "文本水印",
                "videoseal": "稳定可靠",
                "pixelseal": "追求质量"
            }
            input_section = self.create_section(self.tr("📁 输入文件路径"))
            input_path = self.create_path_label(self.task_params["input_path"])
            input_section.layout().addWidget(input_path)
            
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll_area.setStyleSheet("""
                QScrollArea {
                    background-color: #f5f5f5;
                    border: none;
                    border-radius: 12px;
                }
            """)
            params_section = self.create_section(self.tr("⚙️ 水印参数"))
            params_grid = QGridLayout()
            params_grid.setSpacing(10)
            
            row, col = 0, 0
            is_image_watermark = True if self.task_params.get("watermark_content", "TextSettings") == "ImageSettings" else False
            for key, value in self.task_params.items():
                if key in ["input_path", "output_path", "output_format"]:
                    continue
                if is_image_watermark and key in ["font", "font_size", "font_color", "watermark_text"]:
                    continue
                if not is_image_watermark and key in ["watermark_image"]:
                    continue
                param_widget = self.create_param_widget(param_name_map[key], value if value not in value_map else value_map[value])
                params_grid.addWidget(param_widget, row, col)
                col += 1
                if col >= 1:
                    col = 0
                    row += 1
            params_section.layout().addLayout(params_grid)
            scroll_area.setWidget(params_section)
            
            output_section = self.create_section(self.tr("💾 输出保存设置"))
            output_path = self.create_path_label(self.task_params["output_path"], header=self.tr("输出位置："))
            output_format = self.create_path_label(self.task_params["output_format"], header=self.tr("输出格式："))
            output_section.layout().addWidget(output_path)
            output_section.layout().addWidget(output_format)
            
            body_layout.addWidget(input_section)
            body_layout.addWidget(scroll_area)
            body_layout.addWidget(output_section)
        if self.task_type == "watermark-extract":
            value_map = {
                "visible": "可见水印",
                "blind": "盲水印",
                "extract_blind_watermark": "提取",
                "add_blind_watermark": "添加",
                "videoseal": "稳定可靠",
                "pixelseal": "追求质量"
            }
            input_section = self.create_section(self.tr("📁 输入文件路径"))
            input_path = self.create_path_label(self.task_params["input_path"])
            input_section.layout().addWidget(input_path)
            
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll_area.setStyleSheet("""
                QScrollArea {
                    background-color: #f5f5f5;
                    border: none;
                    border-radius: 12px;
                }
            """)
            params_section = self.create_section(self.tr("⚙️ 水印参数"))
            params_grid = QGridLayout()
            params_grid.setSpacing(10)
            
            row, col = 0, 0
            for key, value in self.task_params.items():
                if key in ["input_path"]:
                    continue
                param_widget = self.create_param_widget(param_name_map[key], value if value not in value_map else value_map[value])
                params_grid.addWidget(param_widget, row, col)
                col += 1
                if col >= 1:
                    col = 0
                    row += 1
            params_section.layout().addLayout(params_grid)
            scroll_area.setWidget(params_section)

            body_layout.addWidget(input_section)
            body_layout.addWidget(scroll_area)
        if self.task_type == "watermark-remove":
            value_map = {
                "ai_auto_detect": "AI 全自动检测",
                "ai_interactive_detect": "AI 交互检测",
                "manual_detect": "手工标注",
                "general_watermark": "通用水印",
                "text_watermark": "文本水印",
                "static_watermark": "静态",
                "dynamic_watermark": "动态",
                "semantic_detect": "语义检测",
                "space_detect": "空间位置检测",
                "patchwiper": "细节增强",
                "emdf": "智能修补",
                "grig": "平衡修复",
                "lama": "自然保守",
                "coordfill": "快速填充"
            }
            input_section = self.create_section(self.tr("📁 输入文件路径"))
            input_path = self.create_path_label(self.task_params["input_path"])
            input_section.layout().addWidget(input_path)
            
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll_area.setStyleSheet("""
                QScrollArea {
                    background-color: #f5f5f5;
                    border: none;
                    border-radius: 12px;
                }
            """)
            params_section = self.create_section(self.tr("⚙️ 水印参数"))
            params_grid = QGridLayout()
            params_grid.setSpacing(10)
            
            row, col = 0, 0
            for key, value in self.task_params.items():
                if key in ["input_path", "output_path", "output_format"]:
                    continue
                param_widget = self.create_param_widget(param_name_map[key], str(value) if str(value) not in value_map else value_map[value])
                params_grid.addWidget(param_widget, row, col)
                col += 1
                if col >= 1:
                    col = 0
                    row += 1
            params_section.layout().addLayout(params_grid)
            scroll_area.setWidget(params_section)

            output_section = self.create_section(self.tr("💾 输出保存设置"))
            output_path = self.create_path_label(self.task_params["output_path"], header=self.tr("输出位置："))
            output_format = self.create_path_label(self.task_params["output_format"], header=self.tr("输出格式："))
            output_section.layout().addWidget(output_path)
            output_section.layout().addWidget(output_format)
            
            body_layout.addWidget(input_section)
            body_layout.addWidget(scroll_area)
            body_layout.addWidget(output_section)

        if self.task_type == "ocr-rec":
            value_map = {
                "zh-jp-en": "中日英",
                "pp_ocr": "通用识别"
            }
            input_section = self.create_section(self.tr("📁 输入文件路径"))
            input_path = self.create_path_label(self.task_params["input_path"])
            input_section.layout().addWidget(input_path)
            
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll_area.setStyleSheet("""
                QScrollArea {
                    background-color: #f5f5f5;
                    border: none;
                    border-radius: 12px;
                }
            """)
            params_section = self.create_section(self.tr("⚙️ OCR 参数"))
            params_grid = QGridLayout()
            params_grid.setSpacing(10)
            
            row, col = 0, 0
            for key, value in self.task_params.items():
                if key in ["input_path"]:
                    continue
                param_widget = self.create_param_widget(param_name_map[key], value if value not in value_map else value_map[value])
                params_grid.addWidget(param_widget, row, col)
                col += 1
                if col >= 1:
                    col = 0
                    row += 1
            params_section.layout().addLayout(params_grid)
            scroll_area.setWidget(params_section)
            
            body_layout.addWidget(input_section)
            body_layout.addWidget(scroll_area)
        
        # 底部区域
        footer = QFrame()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(24, 5, 24, 5)
        
        tip_label = QLabel(self.tr("💡 提示：点击路径或参数可复制内容"))
        tip_label.setStyleSheet("color: #6c757d;")
        setFont(tip_label, 13)
        
        footer_layout.addWidget(tip_label)
        
        main_layout.addWidget(body)
        main_layout.addWidget(footer)

        self.viewLayout.addLayout(main_layout)
        
    def create_section(self, title):
        section = QFrame()
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(20, 16, 20, 20)

        section.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border: none;
                border-radius: 12px;
            }
        """)
        
        title_label = QLabel(title)
        setFont(title_label, 14, QFont.Bold)
        title_label.setStyleSheet("""
            QLabel {
                background: transparent;
                color: #212529;
                padding-bottom: 8px;
                border-bottom: 2px solid #7c3aed;
                border-radius: 0px;
            }
        """)
        
        section_layout.addWidget(title_label)
        return section
    
    def create_path_label(self, path, header=""):
        label = QLabel(header + path)
        label.setCursor(Qt.PointingHandCursor)
        label.setStyleSheet("""
            QLabel {
                color: #0d6efd;
                padding: 12px 16px;
                background-color: #f1f3f5;
                border-radius: 12px;
                border: 1px solid #dee2e6;
            }
            QLabel:hover {
                background-color: #e9ecef;
                border: 1px solid #7c3aed;
            }
        """)
        setFont(label, 13)
        label.mousePressEvent = lambda e: self.copy_to_clipboard(path, label)
        return label

    def create_param_widget(self, key, value):
        widget = QFrame()
        widget.setCursor(Qt.PointingHandCursor)
        widget.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 12px;
            }
            QFrame:hover {
                border: 1px solid #7c3aed;
            }
        """)
        
        layout = QVBoxLayout(widget)
        
        key_label = QLabel(key.replace("_", " ").title())
        key_label.setStyleSheet("""
            QLabel {
                color: #6c757d;
                letter-spacing: 0.5px;
                border: none;
            }
        """)
        setFont(key_label, 12, QFont.Bold)
        key_label.setAlignment(Qt.AlignLeft)
        key_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        value_label = QLabel(str(value))
        value_label.setStyleSheet("""
            QLabel {
                color: #212529;
                background-color: #f1f3f5;
                padding: 6px 6px;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }
        """)
        setFont(value_label, 14, QFont.Bold)
        value_label.setAlignment(Qt.AlignLeft)
        value_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        layout.addWidget(key_label)
        layout.addWidget(value_label)
        
        widget.mousePressEvent = lambda e: self.copy_to_clipboard(str(value), widget)
        return widget
    
    def copy_to_clipboard(self, text, widget):
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        self.show_copy_toast(widget)
    
    def show_copy_toast(self, widget):
        TeachingTip.create(
            target=widget,
            icon=InfoBarIcon.SUCCESS,
            title="Tips",
            content=self.tr("已复制到剪贴板"),
            isClosable=True,
            tailPosition=TeachingTipTailPosition.BOTTOM,
            duration=2000,
            parent=self
        )