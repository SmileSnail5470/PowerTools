from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QStackedWidget, QHBoxLayout, QLabel, QLineEdit, QFileDialog
)
from PySide6.QtGui import QPainter, QBrush, QLinearGradient, QColor, QFont, QAction

from app.ui.library.qfluentwidgets import (
    ScrollArea, HeaderCardWidget, SegmentedWidget, setFont, FluentIcon,
    PushButton, CaptionLabel, TextEdit, SpinBox, ComboBox, Slider, LineEdit
)

from app.ui.widgets.font_card import FontCard, get_available_fonts
from app.ui.widgets.file_selector_widget import FileSelectorWidget, FileUploadWidget
from app.ui.widgets.directory_selector_widget import DirectorySelectorWidget
from app.ui.widgets.color_picker_widget import ColorPicker
from app.ui.widgets.image_preview_widget import SyncImageViewer, ImageNavigationWidget
from app.ui.widgets.video_preview_widget import SyncVideoViewer
from app.ui.widgets.status_bar_widget import StatusInfoWidget

from app.ui.common.task_params import bind_widget_to_param, TaskParams


watermark_add_params = TaskParams()

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
        bind_widget_to_param(singleFileSelector, "item_selected", watermark_add_params, "input_path", transform=None)
        batchFilesSelector = DirectorySelectorWidget(self)
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
        self.text_edit.setText("@ PowerTools")
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
        bind_widget_to_param(font_combo, "currentTextChanged", watermark_add_params, "font", transform=None)
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
        image_settings_layout.addSpacing(10)

        slider_top_layout = QHBoxLayout()
        slider_top_layout.setContentsMargins(0, 0, 0, 0)
        text_label_2 = CaptionLabel(text=self.tr("透明度"))
        setFont(text_label_2, 13)
        text_label_2.setStyleSheet("color: #888888;")  # 设置为浅灰色
        slider_top_layout.addWidget(text_label_2)
        self.slider_value_label = QLabel("20%")
        setFont(self.slider_value_label, 13)
        self.slider_value_label.setStyleSheet("color: #888888;")  # 设置为浅灰色
        slider_top_layout.addStretch(1)
        slider_top_layout.addWidget(self.slider_value_label)
        image_settings_layout.addLayout(slider_top_layout)
        slider = Slider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(20)
        slider.valueChanged.connect(self.update_value)
        bind_widget_to_param(slider, "valueChanged", watermark_add_params, "watermark_opacity", transform=None)
        slider.valueChanged.emit(slider.value())
        image_settings_layout.addWidget(slider)

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

    def update_value(self, val):
        self.slider_value_label.setText(str(val)+"%")

    def font_changed(self, font_name):
        if font_name in self.common_fonts_zh.keys():
            text = "你好，世界"
            self.font_card.update_font(self.common_fonts_zh[font_name], text)
        else:
            text = "hello, world"
            self.font_card.update_font(self.common_fonts_en[font_name], text)

    def watermark_text_update(self):
        self.watermark_text_changed.emit(self.text_edit.toPlainText())

class WatermarkSettingsCard(HeaderCardWidget):
    degree = "\u00B0"

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
        bind_widget_to_param(watermark_location_combo, "currentTextChanged", watermark_add_params, "watermark_location", transform=None)
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

        self.viewLayout.addWidget(watermark_location)

    def update_rotation_value(self, val):
        self.slider_rotation_value_label.setText(str(val)+"{degree}".format(degree=self.degree))

    def update_zoom_value(self, val):
        self.slider_zoom_value_label.setText(str(val)+"%")

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
        output_format_combo = ComboBox()
        bind_widget_to_param(output_format_combo, "currentTextChanged", watermark_add_params, "output_format", transform=None)
        output_format_combo.addItems([
            self.tr("保持原格式"), "JPG", "PNG"
        ])
        output_settings_layout.addWidget(output_format_combo)

        self.viewLayout.addWidget(output_settings)

    def save_location_browse(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择文件夹",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.save_location_line_edit.setText(directory)


class GradientHeader(QWidget):
    """渐变标题栏"""
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedHeight(80)
        self.gradient = QLinearGradient(0, 0, self.width(), self.height())
        self.gradient.setColorAt(0, QColor(102, 126, 234))  # #667eea
        self.gradient.setColorAt(1, QColor(118, 75, 162))   # #764ba2
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.gradient.setStart(0, 0)
        self.gradient.setFinalStop(self.width(), self.height())
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QBrush(self.gradient))


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

        watermarkContentCard = WatermarkContentCard(self)
        main_layout.addWidget(watermarkContentCard)

        watermarkSettingsCard = WatermarkSettingsCard(self)
        main_layout.addWidget(watermarkSettingsCard)

        outputSettingsCard = OutputSettingsCard(self)
        main_layout.addWidget(outputSettingsCard)

        self.setWidget(view)
        self.setViewportMargins(0, 0, 0, 0)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


class HeaderWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        header = GradientHeader(parent=self)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 20, 30, 20)
        header_layout.setSpacing(10)

        title_label = QLabel("🎨 水印添加工具")
        setFont(title_label, fontSize=24, weight=QFont.DemiBold)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
            }
        """)
        header_layout.addWidget(title_label)  
        header_layout.addStretch(1)

        extract_btn = PushButton(text="🔍 提取水印")
        extract_btn.setStyleSheet("""
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
        header_layout.addWidget(extract_btn)

        process_btn = PushButton(text="▶️ 开始处理")
        process_btn.setStyleSheet("""
            PushButton {
                background-color: white;
                color: #667eea;
                padding: 8px 16px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
            }
            PushButton:hover {
                background-color: #f8f9fa;
            }
            PushButton:pressed {
                background-color: #5a67d8;
            }
        """)
        header_layout.addWidget(process_btn)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(header)

        process_btn.clicked.connect(self.add_watermark_process)
        extract_btn.clicked.connect(self.extract_process)

    def add_watermark_process(self):
        print(watermark_add_params.to_dict())

    def extract_process(self):
        pass


class PreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PreviewWidget")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        preview_widget = SyncImageViewer(img1="", img2="", parent=self)
        # preview_widget = SyncVideoViewer(self)
        main_layout.addWidget(preview_widget, 1)

        image_navigation_widget = ImageNavigationWidget(parent=self)
        main_layout.addWidget(image_navigation_widget)

        # 底部状态栏
        status_info_widget = StatusInfoWidget(self)
        main_layout.addWidget(status_info_widget)

    
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
