import os
import sys
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QStackedWidget, QHBoxLayout, QLabel, QLineEdit, QFileDialog, QStackedLayout
)
from PySide6.QtGui import QFont, QAction

from app.ui.library.qfluentwidgets import (
    ScrollArea, HeaderCardWidget, SegmentedWidget, setFont, FluentIcon, MessageBox,
    PushButton, CaptionLabel, TextEdit, SpinBox, ComboBox, Slider, LineEdit,
    TeachingTip, InfoBarIcon, TeachingTipTailPosition
)

from app.ui.widgets.font_card import FontCard, get_available_fonts
from app.ui.widgets.file_selector_widget import FileSelectorWidget, FileUploadWidget
from app.ui.widgets.directory_selector_widget import DirectorySelectorWidget
from app.ui.widgets.color_picker_widget import ColorPicker
from app.ui.widgets.image_preview_widget import SyncImageViewer, ImageNavigationWidget
from app.ui.widgets.video_preview_widget import SyncVideoViewer
from app.ui.widgets.status_bar_widget import StatusInfoWidget
from app.ui.widgets.task_info_messagebox_widget import TaskInfoMessageBox
from app.ui.widgets.blind_watermark_text_widget import BlindWatermarkInputPanel
from app.ui.widgets.gradient_header_widget import GradientHeader
from app.ui.widgets.custom_card_group_widget import CustomCardGroupWidget
from app.ui.widgets.toggle_switch_widget import ToggleSwitch
from app.ui.widgets.custom_card_group_widget import CustomCardGroupWidget, StyleCard, CardSeparator

from app.ui.common.task_params import bind_widget_to_param, TaskParams
from app.ui.common.event_bus import global_event_bus
from app.ui.common.task_status import TaskStatusModel
from app.ui.common.utils import get_file_type, global_backend_info_cache
from app.ui.common.config import cfg

from app.controllers.task_manager import global_task_manager
from app.workers.watermark_add_work import WatermarkAddWork
from app.workers.watermark_extract_work import WatermarkExtractWork

from app.license.globals import feature_gate


watermark_add_params = TaskParams()
task_status_model = TaskStatusModel()

class FileSelectorCard(HeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("📁 文件选择"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        self.pivot = SegmentedWidget(self)
        self.stackedWidget = QStackedWidget(self)
        main_layout.addWidget(self.pivot, 0, Qt.AlignTop)
        main_layout.addWidget(self.stackedWidget)

        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        singleFileSelector = FileSelectorWidget(self)
        singleFileSelector.item_selected.connect(lambda file_path: global_event_bus.watermarkAdd_InputFileUpdate.emit(file_path))
        bind_widget_to_param(singleFileSelector, "item_selected", watermark_add_params, "input_path", transform=None)
        batchFilesSelector = DirectorySelectorWidget(self)
        batchFilesSelector.item_selected.connect(lambda file_path: global_event_bus.watermarkAdd_InputFileUpdate.emit(file_path))
        bind_widget_to_param(batchFilesSelector, "item_selected", watermark_add_params, "input_path", transform=None)

        self.addSubInterface(singleFileSelector, 'FileSelectorWidget', self.tr("文件"))
        self.addSubInterface(batchFilesSelector, 'DirectorySelectorWidget', self.tr("目录"))

        self.stackedWidget.setCurrentWidget(singleFileSelector)
        self.pivot.setCurrentItem(singleFileSelector.objectName())
        self.pivot.currentItemChanged.connect(
            lambda k:  self.stackedWidget.setCurrentWidget(self.findChild(QWidget, k)))

    def addSubInterface(self, widget: QWidget, objectName, text):
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.pivot.addItem(routeKey=objectName, text=text)


class WatermarkTypeSelectorCard(HeaderCardWidget):
    watermark_type = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_type = "visible"  # 表示被选择的水印类型

        self.setTitle(self.tr("💧 水印类型"))
        self.setBorderRadius(8)
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        self.base_style = """
            QWidget[cls="selector_card"] {
                padding: 10px;
                border: 2px solid #e0e0e0;
                background-color: white;
                border-radius: 8px;
            }
            QWidget[cls="selector_card"]:hover {
                border-color: #667eea;
            }
        """
        self.active_style = """
            QWidget[cls="selector_card"] {
                padding: 10px;
                border: 2px solid #667eea;
                background-color: #f0f4ff;
                border-radius: 8px;
            }
        """

        # 可见水印
        self.visible_btn = QWidget()
        self.visible_btn.setFixedHeight(60)
        self.visible_btn.setProperty("cls", "selector_card")
        self.visible_btn.setFocusPolicy(Qt.ClickFocus)
        self.visible_btn.setAttribute(Qt.WA_Hover, True)
        self.visible_btn.setEnabled(True)
        self.visible_btn.setCursor(Qt.PointingHandCursor) # 鼠标变手型
        visible_layout = QVBoxLayout(self.visible_btn)
        visible_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        visible_layout.setSpacing(5)
        
        visible_icon = QLabel("👁️")
        setFont(visible_icon, 20)
        visible_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        visible_text = QLabel("可见水印")
        setFont(visible_text, 12)
        visible_text.setStyleSheet("color: #666666;")  # 黑灰色字体
        visible_text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        visible_icon.setCursor(Qt.PointingHandCursor)
        visible_text.setCursor(Qt.PointingHandCursor) 
        visible_layout.addWidget(visible_icon)
        visible_layout.addWidget(visible_text)

        # 盲水印
        self.blind_btn = QWidget()
        self.blind_btn.setFixedHeight(60)
        self.blind_btn.setProperty("cls", "selector_card")
        self.blind_btn.setFocusPolicy(Qt.ClickFocus)
        self.blind_btn.setAttribute(Qt.WA_Hover, True)
        self.blind_btn.setEnabled(True)
        self.blind_btn.setCursor(Qt.PointingHandCursor) # 鼠标变手型
        blind_layout = QVBoxLayout(self.blind_btn)
        blind_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        blind_layout.setSpacing(5)
        
        blind_icon = QLabel("🔐")
        setFont(blind_icon, 20)
        blind_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        blind_text = QLabel("盲水印")
        setFont(blind_text, 12)
        blind_text.setStyleSheet("color: #666666;")  # 黑灰色字体
        blind_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        visible_icon.setCursor(Qt.PointingHandCursor)
        visible_text.setCursor(Qt.PointingHandCursor)
        blind_layout.addWidget(blind_icon)
        blind_layout.addWidget(blind_text)
        
        main_layout.addWidget(self.visible_btn, 1)
        main_layout.addWidget(self.blind_btn, 1)

        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        # 设置默认选中
        self.update_styles()

         # 连接点击事件
        self.visible_btn.mousePressEvent = lambda e: self.select_type("visible")
        self.blind_btn.mousePressEvent = lambda e: self.select_type("blind")

        bind_widget_to_param(self, "watermark_type", watermark_add_params, "watermark_type", transform=None)
        self.watermark_type.emit("visible")

    def select_type(self, type_name: str):
        self.watermark_type.emit(type_name)
        if type_name == self.selected_type:
            return
        self.selected_type = type_name
        self.update_styles()

    def update_styles(self):
        if self.selected_type == "visible":
            self.visible_btn.setStyleSheet(self.active_style)
            self.blind_btn.setStyleSheet(self.base_style)
        else:
            self.visible_btn.setStyleSheet(self.base_style)
            self.blind_btn.setStyleSheet(self.active_style)


class BlindWatermarkOperateCard(HeaderCardWidget):
    blind_watermark_task_type = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("🔍 任务类型"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        self.toggle_switch_btns: list[ToggleSwitch] = []

        watermark_add_btn= ToggleSwitch(on_color="#4FACFE")
        watermark_add_btn.setActive(True)
        self.toggle_switch_btns.append(watermark_add_btn)
        watermark_add_btn.toggled.connect(lambda state, bt=watermark_add_btn: self._on_state_changed(changed_btn=bt, state=state, task_type="add_blind_watermark"))
        watermark_add_card = CustomCardGroupWidget(
            title=self.tr("盲水印添加"), 
            content=self.tr("在视频或图片中添加不可见水印"), 
            parent=self,
            text_layout_contents_margins=(12, 0, 0, 0),
            label_v_space=2
        )
        watermark_add_card.setSeparatorVisible(True)
        watermark_add_card.addWidget(watermark_add_btn)
        main_layout.addWidget(watermark_add_card)

        watermark_extract_btn = ToggleSwitch(on_color="#4FACFE")
        self.toggle_switch_btns.append(watermark_extract_btn)
        watermark_extract_btn.toggled.connect(lambda state, bt=watermark_extract_btn: self._on_state_changed(changed_btn=bt, state=state, task_type="extract_blind_watermark"))
        watermark_extract_card = CustomCardGroupWidget(
            title=self.tr("盲水印提取"), 
            content=self.tr("提取在视频或图片中添加的盲水印"), 
            parent=self,
            text_layout_contents_margins=(12, 0, 0, 0),
            label_v_space=2
        )
        watermark_extract_card.setSeparatorVisible(True)
        watermark_extract_card.addWidget(watermark_extract_btn)
        main_layout.addWidget(watermark_extract_card)
        bind_widget_to_param(self, "blind_watermark_task_type", watermark_add_params, "blind_watermark_task_type", transform=None)
        self.blind_watermark_task_type.emit("add_blind_watermark")
        self.hide()

    def _on_state_changed(self, changed_btn, state, task_type):
        if not state:
            if len(self.toggle_switch_btns) == 2:
                for btn in self.toggle_switch_btns:
                    if not btn.isActive() and btn is not changed_btn:
                        self.blind_watermark_task_type.emit(task_type)
                        btn.setActive(True)
            return
        for btn in self.toggle_switch_btns:
            if btn is changed_btn:
                self.blind_watermark_task_type.emit(task_type)
                continue
            btn.setActive(False)

    def set_watermark_type(self, type_name: str):
        if type_name == "blind":
            self.show()
        else:
            self.hide()


class BlindWatermarkModelCard(HeaderCardWidget):
    blind_watermark_model_name = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("🎨 模型类型"))
        self.setBorderRadius(8)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        self.cards: list[(StyleCard, str)] = []
        stable_card = StyleCard("#fa709a", self.tr("稳定可靠"), self.tr("添加盲水印后，建议提取验证，验证失败尝试更换算法"))
        stable_card.set_name("videoseal")
        self.cards.append((stable_card, "videoseal"))
        main_layout.addWidget(stable_card)
        main_layout.addWidget(CardSeparator(self))

        high_quality_card = StyleCard("#84fab0", self.tr("追求质量"), self.tr("添加盲水印后，建议提取验证，验证失败尝试更换算法"))
        high_quality_card.set_name("pixelseal")
        self.cards.append((high_quality_card, "pixelseal"))
        main_layout.addWidget(high_quality_card)
        main_layout.addWidget(CardSeparator(self))

        for i, card in enumerate(self.cards):
            card_instance, _ = card
            card_instance.mousePressEvent = lambda event, c=card, idx=i: self.on_card_clicked(c, idx)

        main_layout.addStretch()
        bind_widget_to_param(self, "blind_watermark_model_name", watermark_add_params, "blind_watermark_model_name", transform=None)
        self.select_first_interactive()
        global_event_bus.License_update.connect(self.select_first_interactive)
        global_event_bus.watermarkAdd_TaskFinishedByModel.connect(self.update_model_card_info)
        self.hide()

    def update_model_card_info(self, model_name):
        for one_card, _ in self.cards:
            if one_card.get_name() != model_name:
                continue
            one_card.UpdateLicenseInfo.emit()
            if not one_card.is_interactive():
                self.blind_watermark_model_name.emit("")
                self.select_first_interactive()
            break

    def on_card_clicked(self, card: tuple, index):
        card_instance, model_name = card
        # 如果卡片不可交互，则不响应点击
        if not card_instance.is_interactive():
            return
        # 取消所有卡片的选中状态
        for c, _ in self.cards:
            c.set_selected(False)
        
        # 设置当前卡片为选中状态
        card_instance.set_selected(True)
        self.blind_watermark_model_name.emit(model_name)

    def select_first_interactive(self):
        if any(c.is_selected for c, _ in self.cards):
            return
        for card_instance, model_name in self.cards:
            if card_instance.is_interactive():
                card_instance.set_selected(True)
                self.blind_watermark_model_name.emit(model_name)
                return

    def set_watermark_type(self, type_name: str):
        if type_name == "blind":
            self.show()
        else:
            self.hide()


class WatermarkContentCard(HeaderCardWidget):
    watermark_text_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("✏️ 水印内容"))
        self.setBorderRadius(8)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        self.pivot = SegmentedWidget(self)
        self.stackedWidget = QStackedWidget(self)
        main_layout.addWidget(self.pivot, 0, Qt.AlignTop)
        main_layout.addWidget(self.stackedWidget)

        self.viewLayout.setContentsMargins(10, 10, 10, 10)
        self.viewLayout.addLayout(main_layout)

        # 盲水印文本界面
        self.blind_watermark_input = BlindWatermarkInputPanel()
        self.blind_watermark_input.hide()
        bind_widget_to_param(self.blind_watermark_input, "textUpdate", watermark_add_params, "watermark_text", transform=None)
        bind_widget_to_param(self.blind_watermark_input, "charsChange", watermark_add_params, "custom_characters", transform=None)
        self.blind_watermark_input.textUpdate.emit("POWERTOOLS")
        self.blind_watermark_input.charsChange.emit("ABCDEFGHIJKLMNOPQRSTUVWXYZ,1234")
        main_layout.addWidget(self.blind_watermark_input)
        
        # 文字水印设置界面
        textSettings = QWidget()
        text_settings_layout = QVBoxLayout(textSettings)
        text_settings_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        text_settings_layout.setContentsMargins(0, 0, 0, 0)
        text_settings_layout.setSpacing(8)
        text_label_1 = CaptionLabel(text=self.tr("水印文字"))
        setFont(text_label_1, 13)
        text_label_1.setStyleSheet("color: #888888;")  # 设置为浅灰色
        text_settings_layout.addWidget(text_label_1)
        self.text_edit = TextEdit()
        self.text_edit.setPlaceholderText(self.tr("输入水印文字"))
        self.text_edit.setText("POWERTOOLS")
        self.text_edit.setFixedHeight(50)
        setFont(self.text_edit, 13)
        bind_widget_to_param(self, "watermark_text_changed", watermark_add_params, "watermark_text", transform=None)
        self.text_edit.textChanged.connect(self.watermark_text_update)
        self.text_edit.textChanged.emit()
        text_settings_layout.addWidget(self.text_edit)
        text_settings_layout.addSpacing(10)

        text_label_2 = CaptionLabel(text=self.tr("字体"))
        setFont(text_label_2, 13)
        text_label_2.setStyleSheet("color: #888888;")  # 设置为浅灰色
        text_settings_layout.addWidget(text_label_2)
        font_combo = ComboBox()
        bind_widget_to_param(font_combo, "currentTextChanged", watermark_add_params, "font", transform=lambda name: self.font_real_name(name))
        self.common_fonts_zh, self.common_fonts_en = get_available_fonts()
        font_combo.addItems(list(self.common_fonts_zh.keys()) + list(self.common_fonts_en.keys()))
        font_combo.currentTextChanged.connect(self.font_changed)
        text_settings_layout.addWidget(font_combo)
        if font_combo.currentText() in self.common_fonts_zh.keys():
            self.font_card = FontCard(self.common_fonts_zh[font_combo.currentText()], "你好，世界", parent=self)
        else:
            self.font_card = FontCard(self.common_fonts_en[font_combo.currentText()], "hello, world", parent=self)
        text_settings_layout.addWidget(self.font_card)
        text_settings_layout.addSpacing(10)

        text_label_3 = CaptionLabel(text=self.tr("字体大小"))
        setFont(text_label_3, 13)
        text_label_3.setStyleSheet("color: #888888;")  # 设置为浅灰色
        text_settings_layout.addWidget(text_label_3)
        spinBox = SpinBox()
        setFont(spinBox, 13)
        spinBox.setRange(8, 50)
        spinBox.setValue(15)
        bind_widget_to_param(spinBox, "valueChanged", watermark_add_params, "font_size", transform=None)
        spinBox.valueChanged.emit(spinBox.value())
        text_settings_layout.addWidget(spinBox)
        text_settings_layout.addSpacing(10)

        text_label_4 = CaptionLabel(text=self.tr("颜色"))
        setFont(text_label_4, 13)
        text_label_4.setStyleSheet("color: #888888;")  # 设置为浅灰色
        text_settings_layout.addWidget(text_label_4)
        select_color = ColorPicker()
        bind_widget_to_param(select_color, "color_changed", watermark_add_params, "font_color", transform=None)
        select_color.color_changed.emit(select_color.selected_color)
        text_settings_layout.addWidget(select_color)

        # 图片水印设置界面
        imageSettings = QWidget()
        image_settings_layout = QVBoxLayout(imageSettings)
        image_settings_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        image_settings_layout.setContentsMargins(0, 0, 0, 0)
        image_settings_layout.setSpacing(8)
        text_label_1 = CaptionLabel(text=self.tr("选择水印图片"))
        setFont(text_label_1, 13)
        text_label_1.setStyleSheet("color: #888888;")  # 设置为浅灰色
        image_settings_layout.addWidget(text_label_1)
        FileUploadWidget.format_text_value = self.tr("支持 JPG, PNG 格式")
        upload_file_selector = FileSelectorWidget()
        bind_widget_to_param(upload_file_selector, "item_selected", watermark_add_params, "watermark_image", transform=None)
        upload_file_selector.layout_add_height.connect(lambda x: self.adjust_stacked_height(x))
        image_settings_layout.addWidget(upload_file_selector)

        self.addSubInterface(textSettings, 'TextSettings', self.tr("文字"))
        self.addSubInterface(imageSettings, 'ImageSettings', self.tr("图片"))

        self.stackedWidget.setCurrentWidget(textSettings)
        bind_widget_to_param(self.pivot, "currentItemChanged", watermark_add_params, "watermark_content", transform=None)
        self.pivot.setCurrentItem(textSettings.objectName())
        self.pivot.currentItemChanged.connect(lambda k: self.on_pivot_changed(k))
        
    def on_pivot_changed(self, object_name):
        widget = self.findChild(QWidget, object_name)
        if not widget:
            return
        self.stackedWidget.setCurrentWidget(widget)
        widget.adjustSize()
        hint = widget.sizeHint()
        self.stackedWidget.setFixedHeight(hint.height())
        parent = self.parentWidget()
        if parent:
            parent.adjustSize()

    def adjust_stacked_height(self, add_height):
        current_widget = self.stackedWidget.currentWidget()
        if not current_widget:
            return
        current_widget.adjustSize()
        self.stackedWidget.setFixedHeight(current_widget.sizeHint().height() + add_height)
        if self.parentWidget():
            self.parentWidget().adjustSize()

    def addSubInterface(self, widget: QWidget, objectName, text):
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.pivot.addItem(routeKey=objectName, text=text)

    def font_changed(self, font_name):
        if font_name in self.common_fonts_zh.keys():
            text = "你好，世界"
            self.font_card.update_font(self.common_fonts_zh[font_name], text)
        else:
            text = "hello, world"
            self.font_card.update_font(self.common_fonts_en[font_name], text)

    def font_real_name(self, font_name):
        if font_name in self.common_fonts_zh.keys():
            return self.common_fonts_zh[font_name]
        else:
            return self.common_fonts_en[font_name]

    def watermark_text_update(self):
        self.watermark_text_changed.emit(self.text_edit.toPlainText())

    def set_watermark_type(self, type_name: str):
        if type_name == "blind":
            self.blind_watermark_input.show()
            self.stackedWidget.hide()
            self.pivot.hide()
        else:
            self.blind_watermark_input.hide()
            self.pivot.show()
            self.stackedWidget.show()

class WatermarkSettingsCard(HeaderCardWidget):
    degree = "\u00B0"
    watermark_location_map = {
        "左上": "top-left", "上中": "top-center", "右上": "top-right",
        "左中": "center-left", "居中": "center", "右中": "center-right",
        "左下": "bottom-left", "下中": "bottom-center", "右下": "bottom-right"
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("⚙️ 水印设置"))
        self.setBorderRadius(8)
        self.viewLayout.setContentsMargins(10, 10, 10, 10)

        watermark_location = QWidget()
        watermark_location_layout = QVBoxLayout(watermark_location)
        watermark_location_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        watermark_location_layout.setContentsMargins(0, 0, 0, 0)
        watermark_location_layout.setSpacing(8)

        watermark_location_label = CaptionLabel(text=self.tr("位置"))
        setFont(watermark_location_label, 13)
        watermark_location_label.setStyleSheet("color: #888888;")  # 设置为浅灰色
        watermark_location_layout.addWidget(watermark_location_label)
        watermark_location_combo = ComboBox()
        watermark_location_combo.addItems([
            self.tr("左上"), self.tr("上中"), self.tr("右上"),
            self.tr("左中"), self.tr("居中"), self.tr("右中"),
            self.tr("左下"), self.tr("下中"), self.tr("右下"),
        ])
        bind_widget_to_param(
            watermark_location_combo, "currentTextChanged", watermark_add_params, 
            "watermark_location", transform=lambda x: self.watermark_location_map[x] if x in self.watermark_location_map else None
        )
        watermark_location_combo.currentTextChanged.emit(watermark_location_combo.currentText())
        watermark_location_layout.addWidget(watermark_location_combo)
        watermark_location_layout.addSpacing(10)

        rotation_slider_top_layout = QHBoxLayout()
        rotation_slider_top_layout.setContentsMargins(0, 0, 0, 0)
        watermark_rotation_label = CaptionLabel(text=self.tr("旋转角度"))
        setFont(watermark_rotation_label, 13)
        watermark_rotation_label.setStyleSheet("color: #888888;")  # 设置为浅灰色
        rotation_slider_top_layout.addWidget(watermark_rotation_label)
        self.slider_rotation_value_label = QLabel("0{degree}".format(degree=self.degree))
        setFont(self.slider_rotation_value_label, 13)
        self.slider_rotation_value_label.setStyleSheet("color: #888888;")  # 设置为浅灰色
        rotation_slider_top_layout.addStretch(1)
        rotation_slider_top_layout.addWidget(self.slider_rotation_value_label)
        slider = Slider(Qt.Horizontal)
        slider.setRange(-180, 180)
        slider.setValue(0)
        slider.valueChanged.connect(self.update_rotation_value)
        bind_widget_to_param(slider, "valueChanged", watermark_add_params, "watermark_rotation", transform=None)
        slider.valueChanged.emit(slider.value())
        watermark_location_layout.addLayout(rotation_slider_top_layout)
        watermark_location_layout.addWidget(slider)
        watermark_location_layout.addSpacing(10)

        zoom_slider_top_layout = QHBoxLayout()
        zoom_slider_top_layout.setContentsMargins(0, 0, 0, 0)
        watermark_zoom_label = CaptionLabel(text=self.tr("缩放比例"))
        setFont(watermark_zoom_label, 13)
        watermark_zoom_label.setStyleSheet("color: #888888;")  # 设置为浅灰色
        zoom_slider_top_layout.addWidget(watermark_zoom_label)
        self.slider_zoom_value_label = QLabel("100%")
        setFont(self.slider_zoom_value_label, 13)
        self.slider_zoom_value_label.setStyleSheet("color: #888888;")  # 设置为浅灰色
        zoom_slider_top_layout.addStretch(1)
        zoom_slider_top_layout.addWidget(self.slider_zoom_value_label)
        zoom_slider = Slider(Qt.Horizontal)
        zoom_slider.setRange(10, 200)
        zoom_slider.setValue(100)
        zoom_slider.valueChanged.connect(self.update_zoom_value)
        bind_widget_to_param(zoom_slider, "valueChanged", watermark_add_params, "watermark_zoom", transform=None)
        zoom_slider.valueChanged.emit(zoom_slider.value())
        watermark_location_layout.addLayout(zoom_slider_top_layout)
        watermark_location_layout.addWidget(zoom_slider)
        watermark_location_layout.addSpacing(10)

        opacity_slider_top_layout = QHBoxLayout()
        opacity_slider_top_layout.setContentsMargins(0, 0, 0, 0)
        text_label = CaptionLabel(text=self.tr("透明度"))
        setFont(text_label, 13)
        text_label.setStyleSheet("color: #888888;")  # 设置为浅灰色
        opacity_slider_top_layout.addWidget(text_label)
        self.slider_value_label = QLabel("70%")
        setFont(self.slider_value_label, 13)
        self.slider_value_label.setStyleSheet("color: #888888;")  # 设置为浅灰色
        opacity_slider_top_layout.addStretch(1)
        opacity_slider_top_layout.addWidget(self.slider_value_label)
        watermark_location_layout.addLayout(opacity_slider_top_layout)
        slider = Slider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(70)
        slider.valueChanged.connect(self.update_value)
        bind_widget_to_param(slider, "valueChanged", watermark_add_params, "watermark_opacity", transform=None)
        slider.valueChanged.emit(slider.value())
        watermark_location_layout.addWidget(slider)

        self.viewLayout.addWidget(watermark_location)

    def update_value(self, val):
        self.slider_value_label.setText(str(val)+"%")

    def update_rotation_value(self, val):
        self.slider_rotation_value_label.setText(str(val)+"{degree}".format(degree=self.degree))

    def update_zoom_value(self, val):
        self.slider_zoom_value_label.setText(str(val)+"%")

    def set_visible(self, type_name: str):
        self.setVisible(type_name == "visible")

class OutputSettingsCard(HeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("💾 输出设置"))
        self.setBorderRadius(8)
        self.viewLayout.setContentsMargins(10, 10, 10, 10)

        output_settings = QWidget()
        output_settings_layout = QVBoxLayout(output_settings)
        output_settings_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        output_settings_layout.setContentsMargins(0, 0, 0, 0)
        output_settings_layout.setSpacing(8)

        save_location_label = CaptionLabel(text=self.tr("保存位置"))
        setFont(save_location_label, 13)
        save_location_label.setStyleSheet("color: #888888;")  # 设置为浅灰色
        output_settings_layout.addWidget(save_location_label)
        self.save_location_line_edit = LineEdit()
        self.save_location_line_edit.setPlaceholderText(self.tr("选择保存位置"))
        save_location_action = QAction(FluentIcon.FOLDER_ADD.qicon(), "", triggered=self.save_location_browse)
        self.save_location_line_edit.addAction(save_location_action, QLineEdit.TrailingPosition)
        bind_widget_to_param(self.save_location_line_edit, "textChanged", watermark_add_params, "output_path", transform=None)
        output_settings_layout.addWidget(self.save_location_line_edit)
        output_settings_layout.addSpacing(10)

        output_format_label = CaptionLabel(text=self.tr("输出格式"))
        setFont(output_format_label, 13)
        output_format_label.setStyleSheet("color: #888888;")  # 设置为浅灰色
        output_settings_layout.addWidget(output_format_label)
        self.output_format_combo = ComboBox()
        self.output_format_combo.addItems([self.tr("保持原格式")])
        bind_widget_to_param(self.output_format_combo, "currentTextChanged", watermark_add_params, "output_format", transform=None)
        output_settings_layout.addWidget(self.output_format_combo)

        self.viewLayout.addWidget(output_settings)

        global_event_bus.watermarkAdd_InputFileUpdate.connect(self.update_output_format_combox)

    def update_output_format_combox(self, file_path):
        self.output_format_combo.clear()
        if not file_path:
            self.output_format_combo.addItems([self.tr("保持原格式")])
            return
        if os.path.isdir(file_path):
            file_type = get_file_type(os.path.join(file_path, os.listdir(file_path)[0]))
        else:
            file_type = get_file_type(file_path)
        if file_type == "image":
            self.output_format_combo.addItems([self.tr("保持原格式"), "JPG", "PNG"])
        else:
            self.output_format_combo.addItems([self.tr("保持原格式")])

    def save_location_browse(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择文件夹",
            "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog if sys.platform == "darwin" else QFileDialog.Option(0)
        )
        if directory:
            self.save_location_line_edit.setText(directory)


class ControlPanelWidget(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        view = QWidget(self)
        view.setObjectName('controlPanel')
        main_layout = QVBoxLayout(view)
        main_layout.setContentsMargins(0, 0, 12, 0)
        main_layout.setSpacing(10)
        main_layout.setAlignment(Qt.AlignTop)

        fileSelectorCard = FileSelectorCard(self)
        main_layout.addWidget(fileSelectorCard)

        watermarkTypeSelectorCard = WatermarkTypeSelectorCard(self)
        main_layout.addWidget(watermarkTypeSelectorCard)

        blindWatermarkOperateCard = BlindWatermarkOperateCard()
        main_layout.addWidget(blindWatermarkOperateCard)

        blindWatermarkModelCard = BlindWatermarkModelCard()
        main_layout.addWidget(blindWatermarkModelCard)

        watermarkContentCard = WatermarkContentCard(self)
        main_layout.addWidget(watermarkContentCard)

        watermarkSettingsCard = WatermarkSettingsCard(self)
        main_layout.addWidget(watermarkSettingsCard)

        outputSettingsCard = OutputSettingsCard(self)
        main_layout.addWidget(outputSettingsCard)

        main_layout.addStretch(1)

        self.setWidget(view)
        self.setViewportMargins(0, 0, 0, 0)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        visible_or_hide_widgets = [watermarkContentCard, outputSettingsCard]
        self.blind_watermark_task_type = "add_blind_watermark"

        watermarkTypeSelectorCard.watermark_type.connect(
            lambda t: watermarkContentCard.set_watermark_type(t)
        )
        watermarkTypeSelectorCard.watermark_type.connect(
            lambda t: watermarkSettingsCard.set_visible(t)
        )
        watermarkTypeSelectorCard.watermark_type.connect(
            lambda t: blindWatermarkOperateCard.set_watermark_type(t)
        )
        watermarkTypeSelectorCard.watermark_type.connect(
            lambda t: blindWatermarkModelCard.set_watermark_type(t)
        )
        watermarkTypeSelectorCard.watermark_type.connect(
            lambda t: self.set_widgets_visible(visible_or_hide_widgets, watermark_type=t)
        )
        blindWatermarkOperateCard.blind_watermark_task_type.connect(
            lambda t: self.set_widgets_visible(visible_or_hide_widgets, blind_watermark_task_type=t)
        )

    def set_widgets_visible(self, widgets: list[HeaderCardWidget], watermark_type=None, blind_watermark_task_type=None):
        for widget in widgets:
            if blind_watermark_task_type and blind_watermark_task_type == "extract_blind_watermark":
                widget.hide()
            elif blind_watermark_task_type and blind_watermark_task_type == "add_blind_watermark":
                widget.show()
            elif watermark_type and watermark_type == "visible":
                widget.show()
            elif watermark_type and watermark_type == "blind" and self.blind_watermark_task_type == "extract_blind_watermark":
                widget.hide()
            else:
                widget.show()
        if blind_watermark_task_type:
            self.blind_watermark_task_type = blind_watermark_task_type


class HeaderWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        header = GradientHeader(parent=self)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 20, 30, 20)
        header_layout.setSpacing(10)

        title_label = QLabel(self.tr("💧 水印添加工具"))
        setFont(title_label, fontSize=24, weight=QFont.DemiBold)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
            }
        """)
        header_layout.addWidget(title_label)  
        header_layout.addStretch(1)

        self.process_btn = PushButton(text=self.tr("▶️ 开始处理"))
        self.process_btn.setStyleSheet("""
            PushButton {
                background-color: rgba(255, 255, 255, 0.2);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.3);
                padding: 8px 16px;
                border-radius: 8px;
                font-size: 14px;
            }
            PushButton:hover {
                background-color: rgba(255, 255, 255, 0.3);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.15);
            }
        """)
        header_layout.addWidget(self.process_btn)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(header)

        self.process_btn.clicked.connect(self.add_watermark_process)

    def add_watermark_process(self):
        init_params = watermark_add_params.to_dict()
        error_msg, task_params = self._params_check(params=init_params)
        is_blind_watermark_extract_task = True if "blind_watermark_task_type" in task_params and task_params["blind_watermark_task_type"] == "extract_blind_watermark" else False
        if error_msg:
            MessageBox(title=self.tr("提醒"), content=error_msg, parent=self.window()).exec()
            return
        w = TaskInfoMessageBox(task_params, "watermark-add" if not is_blind_watermark_extract_task else "watermark-extract", self.window())
        if not w.exec():
            return
        
        allowed_use, error_msg = feature_gate.can_use(feature_name=feature_gate.get_feature_name(watermark_add_params.to_dict()["blind_watermark_model_name"]), return_errmsg=True)
        if not allowed_use:
            MessageBox(title=self.tr("提醒"), content=error_msg, parent=self.window()).exec()
            return
        task_params["_feature_name_"] = feature_gate.get_feature_name(watermark_add_params.to_dict()["blind_watermark_model_name"])
        
        total_tasks = []
        input_path = task_params["input_path"]
        global_event_bus.watermarkAdd_ImageNavigationInit.emit()
        if os.path.isdir(input_path):
            self.is_batch_task = True
            for one_file in os.listdir(input_path):
                task_params["input_path"] = os.path.join(input_path, one_file)
                task_instance = WatermarkAddWork(**task_params) if not is_blind_watermark_extract_task else WatermarkExtractWork(**task_params)
                func, args, kwargs = task_instance.to_worker()
                total_tasks.append((func, args, kwargs))
        else:
            self.is_batch_task = False
            task_instance = WatermarkAddWork(**task_params) if not is_blind_watermark_extract_task else WatermarkExtractWork(**task_params)
            func, args, kwargs = task_instance.to_worker()
            total_tasks.append((func, args, kwargs))

        # backend_type, gpu_name = global_backend_info_cache.get(key="backend_info")
        backend_type, gpu_name = "CPU 运行", ""
        task_status_model.start_batch(total=len(total_tasks), backend_type=backend_type, gpu_name=gpu_name)
        if not self.is_batch_task:
            task_status_model.start_step(name=self.tr("准备任务"))

        for func, args, kwargs in total_tasks:
            input_path = kwargs["input_path"]
            future = global_task_manager.submit(func, *args, **kwargs)
            
            future.finished.connect(
                lambda result, path=input_path: self._task_finished(path, result)
            )
            future.failed.connect(
                lambda e, path=input_path: task_status_model.report_failure(path, e)
            )
            future.cancelled.connect(
                lambda path=input_path: task_status_model.report_failure(path, "任务被取消")
            )
            future.progress.connect(
                lambda value, msg, path=input_path: self._task_progress(path, value, msg)
            )

        TeachingTip.create(
            target=self.process_btn,
            icon=InfoBarIcon.SUCCESS,
            title=self.tr("通知"),
            content=self.tr("任务提交成功"),
            isClosable=True,
            tailPosition=TeachingTipTailPosition.BOTTOM,
            duration=2000,
            parent=self
        )

    def _task_progress(self, input_path, value, msg):
        if value == "VisibleWatermarkAddStart":
            if not self.is_batch_task:
                task_status_model.finish_step(name=self.tr("准备任务"))
                task_status_model.start_step(name=self.tr('添加/提取水印'))
        elif value == "VisibleWatermarkAddCompleted":
            if not self.is_batch_task:
                task_status_model.finish_step(name=self.tr("添加/提取水印"))
                task_status_model.start_step(name=self.tr('导出文件'))
        elif value == "BlindWatermarkExtractStart":
            if not self.is_batch_task:
                task_status_model.finish_step(name=self.tr("准备任务"))
                task_status_model.start_step(name=self.tr('添加/提取水印'))
        elif value == "BlindWatermarkExtractCompleted":
            if not self.is_batch_task:
                task_status_model.finish_step(name=self.tr("添加/提取水印"))
                task_status_model.start_step(name=self.tr('导出文件'))

    def _task_finished(self, input_path, output_path):
        if not feature_gate.is_pro:
            try:
                feature_gate.use_feature(feature_name=feature_gate.get_feature_name(watermark_add_params.to_dict()["blind_watermark_model_name"]))
            except Exception:
                pass
            global_event_bus.watermarkAdd_TaskFinishedByModel.emit(watermark_add_params.to_dict()["blind_watermark_model_name"])
        task_status_model.report_success()
        global_event_bus.watermarkAdd_TaskFinished.emit(input_path, output_path)
        if not self.is_batch_task:
            task_status_model.finish_step(name=self.tr("导出文件"))

    def _params_check(self, params):
        error_msg = ""
        task_params = {}
        if not params:
            error_msg = self.tr("请设置水印参数")
            return error_msg, task_params
        if not cfg.get(cfg.localBlindWatermarkEnabled):
            error_msg = self.tr("请在设置页面打开 '盲水印AI能力' 开关")
            return error_msg, task_params
        
        if "input_path" not in params or not params["input_path"] or " " in params["input_path"]:
            error_msg = self.tr("请选择要处理的文件或目录且路径不能有空格")
            return error_msg, task_params
        else:
            if isinstance(params["input_path"], str) and not params["input_path"].isascii():
                return self.tr("输入路径: 不支持非英文路径"), task_params
            task_params["input_path"] = params["input_path"]
        
        if get_file_type(params["input_path"]) == "video" and cfg.get(cfg.additionalParams)["SoftwareSettings"]["FFmpeg_status_info"]["text"] != "OK":
            error_msg = self.tr("请在设置页面配置软件 FFmpeg 的正确路径并通过验证")
            return error_msg, task_params

        if "watermark_type" not in params:
            error_msg = self.tr("请选择水印类型")
            return error_msg, task_params
        else:
            task_params["watermark_type"] = params["watermark_type"]
        
        if "blind_watermark_task_type" in params and params["blind_watermark_task_type"] == "extract_blind_watermark":
            task_params["blind_watermark_task_type"] = params["blind_watermark_task_type"]
            if "blind_watermark_model_name" not in params:
                error_msg = self.tr("请选择盲水印算法")
                return error_msg, task_params
            task_params["blind_watermark_model_name"] = params["blind_watermark_model_name"]
            return error_msg, task_params

        if "output_path" not in params or not params["output_path"]:
            error_msg = self.tr("请设置文件保存位置")
            return error_msg, task_params
        else:
            if isinstance(params["output_path"], str) and not params["output_path"].isascii():
                return self.tr("输出路径: 不支持非英文路径"), task_params
            task_params["output_path"] = params["output_path"]
            task_params["output_format"] = params["output_format"]

        if params["watermark_type"] == "visible":
            task_params["watermark_content"] = params["watermark_content"]
            if params["watermark_content"] == "ImageSettings":
                if "watermark_image" not in params or not params["watermark_image"]:
                    error_msg = self.tr("请选择水印图片")
                    return error_msg, task_params
                task_params["watermark_image"] = params["watermark_image"]
            else:
                if not params["watermark_text"]:
                    error_msg = self.tr("请设置水印文本")
                    return error_msg, task_params
                task_params["watermark_text"] = params["watermark_text"]
                task_params["font"] = params["font"]
                task_params["font_size"] = params["font_size"]
                task_params["font_color"] = params["font_color"]
            task_params["watermark_rotation"] = params["watermark_rotation"]
            task_params["watermark_zoom"] = params["watermark_zoom"]
            task_params["watermark_opacity"] = params["watermark_opacity"]
            task_params["watermark_location"] = params["watermark_location"]
        else:
            if not params["watermark_text"]:
                error_msg = self.tr("请设置水印文本")
                return error_msg, task_params
            if "blind_watermark_model_name" not in params:
                error_msg = self.tr("请选择盲水印算法")
                return error_msg, task_params
            task_params["watermark_text"] = params["watermark_text"]
            task_params["blind_watermark_model_name"] = params["blind_watermark_model_name"]
            task_params["custom_characters"] = params["custom_characters"]
        return error_msg, task_params


class PreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PreviewWidget")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        self.stack = QStackedLayout()
        self.stack.setContentsMargins(0, 0, 0, 0)

        self.placeholder_widget = QLabel("请选择图片或视频文件进行预览", parent=self)
        self.placeholder_widget.setStyleSheet("color: #888888;")  # 设置为浅灰色
        setFont(self.placeholder_widget, 20)
        self.placeholder_widget.setAlignment(Qt.AlignCenter)

        self.image_viewer = SyncImageViewer(img1="", img2="", parent=self)
        self.video_viewer = SyncVideoViewer(self)

        self.stack.addWidget(self.placeholder_widget)
        self.stack.addWidget(self.image_viewer)
        self.stack.addWidget(self.video_viewer) 

        main_layout.addLayout(self.stack, 1)

        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)

        self.image_navigation_widget = ImageNavigationWidget(parent=self)
        bottom_layout.addWidget(self.image_navigation_widget, 3)

        # 底部状态栏
        self.status_info_widget = StatusInfoWidget(task_status_model, self)
        self.status_info_widget.model.set_pipeline_steps(names=[self.tr('准备任务'), self.tr('添加/提取水印'), self.tr('导出文件')])
        bottom_layout.addWidget(self.status_info_widget, 2)

        main_layout.addLayout(bottom_layout)

        self.files_preview_info = {}
        self.media_type = "image"

        global_event_bus.watermarkAdd_InputFileUpdate.connect(self.update_init_preview)
        global_event_bus.watermarkAdd_TaskFinished.connect(self.update_preview)
        global_event_bus.watermarkAdd_PreviewFile.connect(self._on_preview_file)
        global_event_bus.watermarkAdd_ImageNavigationInit.connect(lambda: self.image_navigation_widget.clear_images())

    def update_init_preview(self, file_path):
        self.image_navigation_widget.clear_images()
        self.image_viewer.init_scene()
        self.video_viewer.init_scene()
        self.files_preview_info = {}
        self.media_type = "image"
        if not file_path:
            self.stack.setCurrentIndex(0)
            task_status_model.reset()
            return
        if os.path.isdir(file_path):
            tmp_file_path = os.path.join(file_path, os.listdir(file_path)[0])
            self.status_info_widget.show_batch_pipeline_widget()
        else:
            tmp_file_path = file_path
            self.status_info_widget.show_pipeline_widget()
        file_type = get_file_type(tmp_file_path)
        ext = tmp_file_path.lower().split(".")[-1]
        if file_type == "image":
            self.stack.setCurrentIndex(1)
            self.media_type = "image"
        elif file_type == "video":
            self.stack.setCurrentIndex(2)
            self.media_type = "video"
        else:
            self.placeholder_widget.setText(f"不支持的文件类型: {ext}")
            self.stack.setCurrentIndex(0)
            task_status_model.reset()

    def update_preview(self, input_path, output_path):
        self.files_preview_info[input_path] = output_path
        self.image_navigation_widget.load_images([input_path], self.media_type)

    def _on_preview_file(self, path):
        out = self.files_preview_info.get(path)
        widget = self.stack.currentWidget()

        if self.media_type=="image" and out and widget and hasattr(widget, "set_images"):
            widget.set_images(img1=path, img2=out)
        elif self.media_type=="video" and out and widget and hasattr(widget, "setVideos"):
            widget.setVideos(main_path=path, sub_path=out)

    
class WatermarkAdd(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("WatermarkAdd")

        main_Layout = QVBoxLayout(self)
        main_Layout.setContentsMargins(0, 0, 0, 0)
        main_Layout.setSpacing(0)

        header = HeaderWidget(self)
        main_Layout.addWidget(header, 0, Qt.AlignTop)

        view_layout = QHBoxLayout()
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)

        # 左侧控制面板
        control_panel_widget = ControlPanelWidget(self)
        view_layout.addWidget(control_panel_widget, 3)

        # 右侧预览
        right_content = PreviewWidget(self)
        view_layout.addWidget(right_content, 7)

        main_Layout.addLayout(view_layout)
