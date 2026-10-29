from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout, QScrollArea
from app.ui.library.qfluentwidgets import setFont, MessageBoxBase, TeachingTip, InfoBarIcon, TeachingTipTailPosition, SubtitleLabel

from app.workers.watermark_remove import WatermarkRemoveUIParams
from app.workers.watermark_extract_work import WatermarkExtractUIParams
from app.workers.watermark_add_work import WatermarkAddUIParams
from app.workers.ocr_work import OCRUIParams
from app.workers.image_restoration_work import ImageRestorationUIParams
from app.workers.image_edit_work import ImageEditUIParams
from app.workers.blind_watermark_remove_work import BlindWatermarkRemoveUIParams


class TaskInfoMessageBox(MessageBoxBase):
    def __init__(self, layered_params, task_type: str, parent=None, file_type=None):
        super().__init__(parent)
        self._lp = layered_params          # LayeredTaskParams
        self._p = layered_params.params    # 具体的 XxxUIParams dataclass
        self._file_type = file_type
        self.task_type = task_type
        self.titleLabel = SubtitleLabel(self.tr("任务信息总结"))
        self.viewLayout.addWidget(self.titleLabel)
        self.setModal(True)
        self._build()

    def _build(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        body = QFrame()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(24, 10, 24, 20)
        body_layout.setSpacing(10)

        builder = {
            "watermark-remove":       self._build_watermark_remove,
            "watermark-add":          self._build_watermark_add,
            "watermark-extract":      self._build_watermark_extract,
            "ocr-rec":                self._build_ocr,
            "blind-watermark-remove": self._build_blind_watermark_remove,
            "image-edit":             self._build_image_edit,
            "image-restoration":      self._build_image_restoration,
        }.get(self.task_type)

        if builder:
            builder(body_layout)

        footer = QFrame()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(24, 5, 24, 5)
        tip = QLabel(self.tr("💡 提示：点击路径或参数可复制内容"))
        tip.setStyleSheet("color: #6c757d;")
        setFont(tip, 13)
        footer_layout.addWidget(tip)

        main_layout.addWidget(body)
        main_layout.addWidget(footer)
        self.viewLayout.addLayout(main_layout)

    def _build_watermark_remove(self, layout: QVBoxLayout):
        p: WatermarkRemoveUIParams = self._p
        value_map = {
            "ai_detect":        "AI检测",
            "manual_detect":    "手工标注",
            "semantic_detect":  "语义检测",
            "space_detect":     "框选区域",
            "subtitle":         "字幕",
            "text_watermark":   "文字水印",
            "general_watermark":"通用检测",
            "static_watermark": "静态水印",
            "dynamic_watermark":"动态水印",
            "tracking":         "跟踪检测",
            "frame_by_frame":   "逐帧检测",
            "general_edit":     "智能重绘",
            "object_removal":   "智能消除",
            "patchwiper":       "细节增强",
            "emdf":             "自适应修补",
            "image_restoration":"智能修补",
            "grig":             "平衡修复",
            "lama":             "自然保守",
            "coordfill":        "快速填充",
            "video_engine":     "关键帧擦除",
            "ppt":              "时序擦除"
        }
        layout.addWidget(self._section_input(p.io.input_path))
        params = [(self.tr("水印检测方式"), value_map.get(p.detect.watermark_detect_type, p.detect.watermark_detect_type))]
        if p.detect.watermark_detect_type == "manual_detect":
            if p.detect.manual_watermark_mask_path:
                params.append((self.tr("手工 Mask"), p.detect.manual_watermark_mask_path))
        else:
            need_image_detect_params = True
            if self._file_type == "video":
                params.append((self.tr("水印类型"), value_map.get(p.detect.video_watermark_format, p.detect.video_watermark_format)))
                if p.detect.video_watermark_format == "dynamic_watermark":
                    params.append((self.tr("水印检测策略"), value_map.get(p.detect.video_watermark_ai_detect_strategy, p.detect.video_watermark_ai_detect_strategy)))
                    if p.detect.video_watermark_ai_detect_strategy == "tracking":
                        params.append((self.tr("水印根据检测参数"), value_map.get(p.detect.video_watermark_tracking_data, p.detect.video_watermark_tracking_data)))
                        need_image_detect_params = False
                    else:
                        params.append((self.tr("检测帧范围"), value_map.get(p.detect.video_watermark_ai_detect_frame_range, p.detect.video_watermark_ai_detect_frame_range)))
                else:
                    params.append((self.tr("检测关键帧"), value_map.get(p.detect.video_watermark_ai_detect_anchor_frame, p.detect.video_watermark_ai_detect_anchor_frame)))
                    params.append((self.tr("检测帧范围"), value_map.get(p.detect.video_watermark_ai_detect_frame_range, p.detect.video_watermark_ai_detect_frame_range)))
            if need_image_detect_params:
                params.append((self.tr("水印检测方法"), value_map.get(p.detect.watermark_ai_detect_method, p.detect.watermark_ai_detect_method)))
                if p.detect.watermark_ai_detect_method == "semantic_detect":
                    params.append((self.tr("检测提示词"), p.detect.watermark_ai_detect_prompt))
                    params.append((self.tr("检测灵敏度"), p.detect.watermark_ai_detect_confidence))
                    if p.detect.watermark_ai_detect_neg_boxes:
                        params.append((self.tr("负检测框 (保留)"), p.detect.watermark_ai_detect_neg_boxes))
                    if p.detect.watermark_ai_detect_pos_boxes:
                        params.append((self.tr("正检测框 (去除)"), p.detect.watermark_ai_detect_pos_boxes))
                elif p.detect.watermark_ai_detect_method == "space_detect":
                    params.append((self.tr("检测灵敏度"), p.detect.watermark_ai_detect_confidence))
                    if p.detect.watermark_ai_detect_neg_boxes:
                        params.append((self.tr("负检测框 (保留)"), p.detect.watermark_ai_detect_neg_boxes))
                    if p.detect.watermark_ai_detect_pos_boxes:
                        params.append((self.tr("正检测框 (去除)"), p.detect.watermark_ai_detect_pos_boxes))
                else:
                    if p.detect.watermark_ai_detect_pos_boxes:
                        params.append((self.tr("正检测框 (去除)"), p.detect.watermark_ai_detect_pos_boxes))
        params.append((self.tr("模型"), value_map.get(p.model.model_name, p.model.model_name)))
        if p.model.model_name == "image_restoration":
            params.append((self.tr("水印类型"), p.model.restoration_watermark_type))
            if p.model.restoration_prompt:
                params.append((self.tr("自定义提示词"), p.model.restoration_prompt))
        else:
            params.append((self.tr("Mask 扩张系数"),   str(p.model.mask_dilate)))

        layout.addWidget(self._section_params(self.tr("⚙️ 去除参数"), params))
        layout.addWidget(self._section_output(p.io.output_path, p.io.output_format))

    def _build_watermark_add(self, layout: QVBoxLayout):
        p: WatermarkAddUIParams = self._p
        value_map = {
            "visible":   "可见水印",
            "blind":     "盲水印",
            "TextSettings":  "文字水印",
            "ImageSettings": "图片水印",
            "videoseal": "稳定可靠",
            "pixelseal": "追求质量",
            "top-left": "左上", "top-center": "上中", "top-right": "右上",
            "center-left": "左中", "center": "居中", "center-right": "右中",
            "bottom-left": "左下", "bottom-center": "下中", "bottom-right": "右下",
        }
        layout.addWidget(self._section_input(p.io.input_path))
        params = [(self.tr("水印类型"), value_map.get(p.type.watermark_type, p.type.watermark_type))]
        if p.type.watermark_type == "visible":
            params.append((self.tr("水印样式"), value_map.get(p.type.watermark_content, p.type.watermark_content)))
            if p.type.watermark_content == "TextSettings":
                params += [
                    (self.tr("水印文字"),       p.visible.watermark_text),
                    (self.tr("字体"),           p.visible.font),
                    (self.tr("字体大小"),        str(p.visible.font_size)),
                    (self.tr("字体颜色"),        p.visible.font_color),
                ]
            else:
                params.append((self.tr("水印图片"), p.visible.watermark_image))
            params += [
                (self.tr("位置"),   value_map.get(p.visible.watermark_location, p.visible.watermark_location)),
                (self.tr("透明度"), f"{p.visible.watermark_opacity}%"),
                (self.tr("旋转"),   f"{p.visible.watermark_rotation}°"),
                (self.tr("缩放"),   f"{p.visible.watermark_zoom}%"),
            ]
        else:
            params += [
                (self.tr("盲水印算法"), value_map.get(p.blind.blind_watermark_model_name, p.blind.blind_watermark_model_name)),
                (self.tr("水印文字"),   p.blind.watermark_text),
            ]

        layout.addWidget(self._section_params(self.tr("⚙️ 水印参数"), params))
        layout.addWidget(self._section_output(p.io.output_path, p.io.output_format))

    def _build_watermark_extract(self, layout: QVBoxLayout):
        p: WatermarkExtractUIParams = self._p
        value_map = {
            "videoseal": "稳定可靠",
            "pixelseal": "追求质量",
        }
        layout.addWidget(self._section_input(p.io.input_path))
        params = [(self.tr("盲水印算法"), value_map.get(p.blind.blind_watermark_model_name, p.blind.blind_watermark_model_name))]
        layout.addWidget(self._section_params(self.tr("⚙️ 提取参数"), params))

    def _build_ocr(self, layout: QVBoxLayout):
        p: OCRUIParams = self._p
        model_map = {"pp_ocr": "通用识别"}
        lang_map  = {"zh-jp-en": "中日英"}
        layout.addWidget(self._section_input(p.io.input_path))
        params = [
            (self.tr("模型"),         model_map.get(p.model.model_name, p.model.model_name)),
            (self.tr("识别语言"),      lang_map.get(p.settings.ocr_rec_language, p.settings.ocr_rec_language)),
            (self.tr("识别置信度"),    str(p.settings.drop_score)),
            (self.tr("文本行方向矫正"), self.tr("开") if p.settings.use_textline_ori else self.tr("关")),
        ]
        layout.addWidget(self._section_params(self.tr("⚙️ OCR 参数"), params))

    def _build_blind_watermark_remove(self, layout: QVBoxLayout):
        p: BlindWatermarkRemoveUIParams = self._p
        model_map = {"reverse_edit": "智能消除"}
        layout.addWidget(self._section_input(p.io.input_path))
        params = [
            (self.tr("去除模型"),     model_map.get(p.model.model_name, p.model.model_name)),
            (self.tr("颜色修复"),     self.tr("开") if p.settings.use_color_fix else self.tr("关")),
            (self.tr("高质量输出"),   self.tr("开") if p.settings.high_quality_output else self.tr("关")),
        ]
        if p.settings.reserve_region:
            r = p.settings.reserve_region
            params.append((self.tr("保留区域"), f"X={r.get('x',0)} Y={r.get('y',0)} W={r.get('w',0)} H={r.get('h',0)}"))
        layout.addWidget(self._section_params(self.tr("⚙️ 暗水印去除参数"), params))
        layout.addWidget(self._section_output(p.io.output_path))

    def _build_image_edit(self, layout: QVBoxLayout):
        p: ImageEditUIParams = self._p
        model_map = {"general_edit": "智能重绘"}
        if p.io.input_path:
            layout.addWidget(self._section_input(p.io.input_path))
        params = [
            (self.tr("编辑模型"),  model_map.get(p.model.model_name, p.model.model_name)),
            (self.tr("提示词"),    p.task.prompt),
        ]
        if p.task.mask_boxes:
            params.append((self.tr("编辑区域"), self.tr("{0} 个区域").format(len(p.task.mask_boxes))))
        layout.addWidget(self._section_params(self.tr("⚙️ 图像编辑参数"), params))
        layout.addWidget(self._section_output(p.io.output_path))

    def _build_image_restoration(self, layout: QVBoxLayout):
        p: ImageRestorationUIParams = self._p
        model_map = {
            "image_restoration": "智能修复",
            "image_sr":          "图像高清",
        }
        task_map = {
            "restoration":     "综合修复",
            "dehaze":          "去雾",
            "derain":          "去雨",
            "snow":            "去雪",
            "denoise":         "去噪",
            "deblur":          "去模糊",
            "shadow":          "去阴影",
            "super_resolution":"超分放大",
            "lowlight":        "低光增强",
            "compression":     "去压缩伪影",
            "watermark_remove":"去水印",
        }
        layout.addWidget(self._section_input(p.io.input_path))
        params = [
            (self.tr("修复任务"),  task_map.get(p.task.task_type, p.task.task_type)),
            (self.tr("修复模型"),  model_map.get(p.model.model_name, p.model.model_name)),
            (self.tr("低显存模式"), self.tr("开") if p.task.low_memory else self.tr("关")),
        ]
        if p.task.task_type == "super_resolution":
            params.append((self.tr("放大倍数"), f"{p.task.upscale}x"))
        if p.task.prompt:
            params.append((self.tr("自定义提示词"), p.task.prompt))
        layout.addWidget(self._section_params(self.tr("⚙️ 修复参数"), params))
        layout.addWidget(self._section_output(p.io.output_path))

    def _section_input(self, path: str) -> QFrame:
        sec = self._make_section(self.tr("📁 输入文件路径"))
        sec.layout().addWidget(self._path_label(path))
        return sec

    def _section_output(self, path: str, fmt: str = "") -> QFrame:
        sec = self._make_section(self.tr("💾 输出保存设置"))
        sec.layout().addWidget(self._path_label(path, header=self.tr("输出位置：")))
        if fmt:
            sec.layout().addWidget(self._path_label(fmt, header=self.tr("输出格式：")))
        return sec

    def _section_params(self, title: str, params: list[tuple[str, str]]) -> QFrame:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""QScrollArea { background-color: #f5f5f5; border: none; border-radius: 12px; }""")

        sec = self._make_section(title)
        grid = QGridLayout()
        grid.setSpacing(10)

        for row, (name, value) in enumerate(params):
            grid.addWidget(self._param_widget(name, str(value)), row, 0)

        sec.layout().addLayout(grid)
        scroll.setWidget(sec)
        return scroll

    def _make_section(self, title: str) -> QFrame:
        sec = QFrame()
        sec_layout = QVBoxLayout(sec)
        sec_layout.setContentsMargins(20, 16, 20, 20)
        sec.setStyleSheet("""QFrame { background-color: #f5f5f5; border: none; border-radius: 12px; }""")
        lbl = QLabel(title)
        setFont(lbl, 14, QFont.Bold)
        lbl.setStyleSheet("""
            QLabel {
                background: transparent; color: #212529;
                padding-bottom: 8px;
                border-bottom: 2px solid #7c3aed; border-radius: 0px;
            }
        """)
        sec_layout.addWidget(lbl)
        return sec

    def _path_label(self, path: str, header: str = "") -> QLabel:
        lbl = QLabel(header + path)
        lbl.setCursor(Qt.PointingHandCursor)
        lbl.setStyleSheet("""
            QLabel {
                color: #0d6efd; padding: 12px 16px;
                background-color: #f1f3f5;
                border-radius: 12px; border: 1px solid #dee2e6;
            }
            QLabel:hover { background-color: #e9ecef; border: 1px solid #7c3aed; }
        """)
        setFont(lbl, 13)
        lbl.mousePressEvent = lambda e: self._copy(path, lbl)
        return lbl

    def _param_widget(self, name: str, value: str) -> QFrame:
        widget = QFrame()
        widget.setCursor(Qt.PointingHandCursor)
        widget.setStyleSheet("""
            QFrame { background-color: white; border: 1px solid #dee2e6; border-radius: 12px; }
            QFrame:hover { border: 1px solid #7c3aed; }
        """)
        lay = QVBoxLayout(widget)

        key_lbl = QLabel(name)
        key_lbl.setStyleSheet("QLabel { color: #6c757d; letter-spacing: 0.5px; border: none; }")
        setFont(key_lbl, 12, QFont.Bold)
        key_lbl.setAlignment(Qt.AlignLeft)
        key_lbl.setAttribute(Qt.WA_TransparentForMouseEvents)

        val_lbl = QLabel(value)
        val_lbl.setStyleSheet("""
            QLabel {
                color: #212529; background-color: #f1f3f5;
                padding: 6px 6px; border-radius: 8px; border: 1px solid #e9ecef;
            }
        """)
        setFont(val_lbl, 14, QFont.Bold)
        val_lbl.setAlignment(Qt.AlignLeft)
        val_lbl.setAttribute(Qt.WA_TransparentForMouseEvents)

        lay.addWidget(key_lbl)
        lay.addWidget(val_lbl)

        widget.mousePressEvent = lambda e: self._copy(value, widget)
        return widget

    def _copy(self, text: str, target):
        QApplication.clipboard().setText(text)
        TeachingTip.create(
            target=target,
            icon=InfoBarIcon.SUCCESS,
            title="Tips",
            content=self.tr("已复制到剪贴板"),
            isClosable=True,
            tailPosition=TeachingTipTailPosition.BOTTOM,
            duration=2000,
            parent=self
        )