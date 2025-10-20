from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QStackedWidget, QHBoxLayout, QLabel, QFontComboBox
)
from PySide6.QtGui import QPainter, QBrush, QLinearGradient, QColor, QFont

from app.ui.library.qfluentwidgets import (
    ScrollArea, HeaderCardWidget, GroupHeaderCardWidget, SegmentedWidget, setFont,
    PushButton, CaptionLabel, TextEdit, SpinBox, ComboBox, ColorPickerButton
)

from app.ui.widgets.font_card import FontCard, get_available_fonts
from app.ui.widgets.file_selector_widget import FileSelectorWidget
from app.ui.widgets.directory_selector_widget import DirectorySelectorWidget
from app.ui.widgets.color_picker_widget import ColorPicker


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
        batchFilesSelector = DirectorySelectorWidget(self)

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

    def select_type(self, type_name: str):
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

        textSettings = QWidget()
        text_settings_layout = QVBoxLayout(textSettings)
        text_settings_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        text_settings_layout.setContentsMargins(0, 0, 0, 0)
        text_settings_layout.setSpacing(5)
        text_label_1 = CaptionLabel(text="水印文字")
        setFont(text_label_1, 13)
        text_label_1.setStyleSheet("color: #888888;")  # 设置为浅灰色
        text_settings_layout.addWidget(text_label_1)
        text_edit = TextEdit()
        text_edit.setPlaceholderText("输入水印文字")
        text_edit.setText("@ PowerTools")
        text_edit.setFixedHeight(50)
        setFont(text_edit, 13)
        text_settings_layout.addWidget(text_edit)

        text_settings_layout.addSpacing(10)

        text_label_2 = CaptionLabel(text="字体")
        setFont(text_label_2, 13)
        text_label_2.setStyleSheet("color: #888888;")  # 设置为浅灰色
        text_settings_layout.addWidget(text_label_2)
        font_combo = ComboBox()
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

        text_label_3 = CaptionLabel(text="字体大小")
        setFont(text_label_3, 13)
        text_label_3.setStyleSheet("color: #888888;")  # 设置为浅灰色
        text_settings_layout.addWidget(text_label_3)
        spinBox = SpinBox()
        setFont(spinBox, 13)
        spinBox.setRange(8, 50)
        spinBox.setValue(15)
        # 监听数值改变信号
        # spinBox.valueChanged.connect(lambda value: print("当前值：", value))
        text_settings_layout.addWidget(spinBox)

        text_settings_layout.addSpacing(10)

        text_label_4 = CaptionLabel(text="颜色")
        setFont(text_label_4, 13)
        text_label_4.setStyleSheet("color: #888888;")  # 设置为浅灰色
        text_settings_layout.addWidget(text_label_4)
        select_color = ColorPicker()
        text_settings_layout.addWidget(select_color)


        imageSettings = DirectorySelectorWidget(self)

        self.addSubInterface(textSettings, 'TextSettings', self.tr("文字"))
        self.addSubInterface(imageSettings, 'ImageSettings', self.tr("图片"))

        self.stackedWidget.setCurrentWidget(textSettings)
        self.pivot.setCurrentItem(textSettings.objectName())
        self.pivot.currentItemChanged.connect(
            lambda k:  self.stackedWidget.setCurrentWidget(self.findChild(QWidget, k)))

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

class WatermarkSettingsCard(GroupHeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)


class OutputSettingsCard(GroupHeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)


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


class PreviewWidget():
    pass


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
        right_content = QLabel("右侧预览区")
        right_content.setAlignment(Qt.AlignCenter)
        view_layout.addWidget(right_content, 7)

        main_Layout.addLayout(view_layout)
