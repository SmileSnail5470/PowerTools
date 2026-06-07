import os
import sys
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSlider, QFileDialog,
                             QLabel, QPushButton, QLineEdit, QStackedWidget, 
                             QFrame, QGraphicsDropShadowEffect)
from PySide6.QtCore import Qt, QPropertyAnimation, QPoint, QEasingCurve, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from app.ui.library.qfluentwidgets import setFont
from app.ui.common.utils import get_file_type
from app.ui.widgets.watermark_interactive_widget import AreaSelectorDialog
from app.ui.widgets.watermark_manual_select_widget import WatermarkMaskTool


class OptionCard(QFrame):
    clicked = Signal()

    def __init__(self, title, desc, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.selected = False
        
        layout = QVBoxLayout(self)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("optTitle")
        setFont(self.title_label, 13, QFont.DemiBold)
        self.desc_label = QLabel(desc)
        self.desc_label.setObjectName("optDesc")
        setFont(self.desc_label, 11)
        
        layout.addWidget(self.title_label)
        if desc:
            layout.addWidget(self.desc_label)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(2)
        
        self.update_style()

    def update_style(self):
        self.setProperty("selected", self.selected)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event):
        if self.parent():
            for child in self.parent().findChildren(OptionCard):
                child.selected = False
                child.update_style()
        self.selected = True
        self.update_style()
        self.clicked.emit()

class WatermarkDetectSettings(QWidget):
    watermarkDetectType = Signal(str)         # AI交互/AI自动/人工检测
    watermarkFormat = Signal(str)             # 静态水印/动态水印
    manualWatermarktMaskPath = Signal(str)    # 手动标注 Mask 路径
    maskDirectoryChanged = Signal(str)        # 人工指定 Mask 路径
    # AI 交互检测信号
    watermarkAIInteractiveType = Signal(str)  # 语义检测/空间范围检测
    watermarkDetectPrompt = Signal(str)       # 语义检测提示词
    watermarkBoxes = Signal(list)             # 水印空间范围 boxes 坐标
    watermarkConfidence = Signal(float)       # 水印置信度
    # AI 自动检测
    watermarkContent = Signal(str)            # 通用水印/文字水印


    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.current_tab_index = 0
        self.file_path = None
        self.mask_dir_path = ""

        self.setObjectName("WatermarkDetectSettings")
        self.init_ui()
        self.apply_styles()
        QTimer.singleShot(50, lambda: self.switch_tab(0, animated=False))

    def init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignTop)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.card = QFrame()
        self.card.setObjectName("panelCard")
        self.card_layout = QVBoxLayout(self.card)
        self.card_layout.setContentsMargins(0, 0, 0, 0)
        self.card_layout.setSpacing(0)

        # Tabs 导航栏
        self.tabs_container = QFrame()
        self.tabs_container.setObjectName("tabsContainer")
        self.tabs_container.setFixedHeight(50)
        self.tabs_layout = QHBoxLayout(self.tabs_container)
        self.tabs_layout.setContentsMargins(5, 5, 5, 5)
        self.tabs_layout.setSpacing(0)

        # 动画滑块背景
        self.slider = QFrame(self.tabs_container)
        self.slider.setObjectName("slider")
        self.slider.setAttribute(Qt.WA_TransparentForMouseEvents) 
        
        self.tab_btns = []
        for i, items in enumerate([(self.tr("AI 交互"), "ai_interactive_detect"), (self.tr("AI 全自动"), "ai_auto_detect"), (self.tr("手工标注"), "manual_detect")]):
            text, signal_value = items
            btn = QPushButton(text)
            btn.setObjectName("tabItem")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            setFont(btn, 13, QFont.DemiBold)
            btn.clicked.connect(lambda _, idx=i: self.switch_tab(idx))
            btn.clicked.connect(lambda _, signal_value=signal_value: self.watermarkDetectType.emit(signal_value))
            self.tabs_layout.addWidget(btn)
            self.tab_btns.append(btn)

        self.card_layout.addWidget(self.tabs_container)
        self.card_layout.addSpacing(20)

        self.stack = QStackedWidget()
        
        # AI 全自动 ---
        p0 = QWidget()
        p0_l = QVBoxLayout(p0)
        p0_l.setContentsMargins(0, 0, 0, 0)
        p0_l.addWidget(self.create_label_group(self.tr("水印类型识别"), self.tr("系统将自动分析画面中的元素")))
        grid0 = QWidget()
        grid0_l = QHBoxLayout(grid0)
        grid0_l.setContentsMargins(0,0,0,0)
        c1 = OptionCard(self.tr("通用水印"), self.tr("Logo、图案、复合元素"))
        c1.clicked.connect(lambda: self.watermarkContent.emit("general_watermark"))
        c1.selected = True
        c1.update_style()
        grid0_l.addWidget(c1)
        c1_1 = OptionCard(self.tr("文字水印"), self.tr("纯文本、日期、字幕"))
        c1_1.clicked.connect(lambda: self.watermarkContent.emit("text_watermark"))
        grid0_l.addWidget(c1_1)
        p0_l.addWidget(grid0)
        p0_l.addSpacing(24)
        
        p0_l.addWidget(self.create_label_group(self.tr("水印形式"), self.tr("选择水印在时间轴上的状态")))
        grid1 = QWidget()
        grid1_l = QHBoxLayout(grid1)
        grid1_l.setContentsMargins(0,0,0,0)
        c2 = OptionCard(self.tr("静态水印"), self.tr("固定位置不移动"))
        c2.clicked.connect(lambda: self.watermarkFormat.emit("static_watermark"))
        c2.selected = True
        c2.update_style()
        grid1_l.addWidget(c2)
        c2_1 = OptionCard(self.tr("动态水印"), self.tr("移动、缩放、淡入淡出"))
        c2_1.clicked.connect(lambda: self.watermarkFormat.emit("dynamic_watermark"))
        grid1_l.addWidget(c2_1)
        p0_l.addWidget(grid1)

        # AI 交互 ---
        p1 = QWidget()
        p1_l = QVBoxLayout(p1)
        p1_l.setContentsMargins(0,0,0,0)
        self.mode_switch = QFrame()
        self.mode_switch.setObjectName("modeSwitch")
        ms_l = QHBoxLayout(self.mode_switch)
        ms_l.setContentsMargins(4,4,4,4)
        self.btn_sem = QPushButton(self.tr("语义识别"))
        self.btn_sem.clicked.connect(lambda _: self.watermarkAIInteractiveType.emit("semantic_detect"))
        self.btn_spa = QPushButton(self.tr("空间范围识别"))
        self.btn_spa.clicked.connect(lambda _: self.watermarkAIInteractiveType.emit("space_detect"))
        for b in [self.btn_sem, self.btn_spa]:
            b.setObjectName("modeBtn")
            b.setCheckable(True)
            ms_l.addWidget(b)
        self.btn_sem.setChecked(True)
        p1_l.addWidget(self.mode_switch)

        self.sub_stack = QStackedWidget()
        sem_w = QWidget()
        sem_l = QVBoxLayout(sem_w)
        sem_l.setContentsMargins(0,0,0,0)
        sem_l.addWidget(self.create_label_group(self.tr("文本语义描述"), self.tr("输入想识别的特定元素描述")))
        self.input_box = QLineEdit()
        self.input_box.setObjectName("inputBox")
        setFont(self.input_box, 13)
        self.input_box.setPlaceholderText(self.tr("文本描述（请输入英文）..."))
        self.input_box.textChanged.connect(lambda text: self.watermarkDetectPrompt.emit(text))
        sem_l.addWidget(self.input_box)
        chip_l = QHBoxLayout()
        chip_l.setSpacing(8)
        text_map = {
            "水印": "watermark",
            "字幕": "subtitle",
            "水印和字幕": "watermark,subtitle"
        }
        for t in [self.tr("水印"), self.tr("字幕"), self.tr("水印和字幕")]:
            cp = QPushButton(t)
            cp.setObjectName("chip")
            setFont(cp, 12)
            cp.clicked.connect(lambda _, text=t: self.input_box.setText(text_map[text]))
            chip_l.addWidget(cp)
        sem_l.addLayout(chip_l)
        self.sub_stack.addWidget(sem_w)
        
        spa_w = QWidget()
        spa_l = QVBoxLayout(spa_w)
        spa_l.setContentsMargins(0,10,0,10)
        spa_l.addWidget(self.create_label_group(self.tr("精准框选"), self.tr("在图片区手动框选位置")))
        spa_l.addStretch()
        sec_btn = QPushButton(self.tr("🎯 开始水印框选"))
        sec_btn.setObjectName("secondaryBtn")
        sec_btn.clicked.connect(lambda _: self._watermark_area_selector())
        spa_l.addWidget(sec_btn)
        self.sub_stack.addWidget(spa_w)
        p1_l.addWidget(self.sub_stack)
        p1_l.addSpacing(20)

        conf_w = QWidget()
        conf_l = QVBoxLayout(conf_w)
        conf_l.setContentsMargins(0, 0, 0, 0)
        conf_l.setSpacing(10)

        conf_header = QHBoxLayout()
        conf_info_vbox = QVBoxLayout()
        conf_info_vbox.setSpacing(2)
        ct = QLabel(self.tr("水印置信度"))
        ct.setObjectName("labelTitle")
        setFont(ct, 13, QFont.DemiBold)
        cd = QLabel(self.tr("设置检测算法的灵敏度阈值"))
        cd.setObjectName("labelDesc")
        setFont(cd, 11)
        conf_info_vbox.addWidget(ct)
        conf_info_vbox.addWidget(cd)
        
        self.conf_val_label = QLabel("0.50")
        self.conf_val_label.setObjectName("confValBadge")
        setFont(self.conf_val_label, 12, QFont.Bold)
        
        conf_header.addLayout(conf_info_vbox)
        conf_header.addStretch()
        conf_header.addWidget(self.conf_val_label)
        conf_l.addLayout(conf_header)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setObjectName("confSlider")
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        self.conf_slider.setFixedHeight(24)
        self.conf_slider.setCursor(Qt.PointingHandCursor)
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        conf_l.addWidget(self.conf_slider)
        p1_l.addWidget(conf_w)
        p1_l.addSpacing(20)

        p1_l.addWidget(self.create_label_group(self.tr("水印形式"), self.tr("选择水印在时间轴上的状态")))
        grid2 = QWidget()
        grid2_l = QHBoxLayout(grid2)
        grid2_l.setContentsMargins(0,0,0,0)
        c3 = OptionCard(self.tr("静态水印"), self.tr("固定位置不移动"))
        c3.clicked.connect(lambda: self.watermarkFormat.emit("static_watermark"))
        c3.selected = True
        c3.update_style()
        grid2_l.addWidget(c3)
        c3_1 = OptionCard(self.tr("动态水印"), self.tr("移动、缩放、淡入淡出"))
        c3_1.clicked.connect(lambda: self.watermarkFormat.emit("dynamic_watermark"))
        grid2_l.addWidget(c3_1)
        p1_l.addWidget(grid2)

        # 手工标注 ---
        p2 = QWidget()
        p2_l = QVBoxLayout(p2)
        p2_l.setContentsMargins(0,0,0,0)

        p2_l.addWidget(self.create_label_group(self.tr("导入已有水印标注"), self.tr("若选择 Mask 目录，将跳过手动标注")))
        dir_layout = QHBoxLayout()
        dir_layout.setSpacing(8)
        self.dir_input = QLineEdit()
        self.dir_input.setObjectName("inputBox")
        self.dir_input.setReadOnly(True)
        self.dir_input.setPlaceholderText(self.tr("未选择目录（选填）..."))
        setFont(self.dir_input, 12)
        self.dir_select_btn = QPushButton(self.tr("打开"))
        self.dir_select_btn.setObjectName("secondaryBtn")
        self.dir_select_btn.setFixedHeight(40)
        setFont(self.dir_select_btn, 12, QFont.DemiBold)
        self.dir_select_btn.clicked.connect(self._on_select_mask_dir)
        self.dir_clear_btn = QPushButton("清除")
        self.dir_clear_btn.setObjectName("secondaryBtn")
        self.dir_clear_btn.setFixedHeight(40)
        self.dir_clear_btn.setVisible(False)
        self.dir_clear_btn.clicked.connect(self._on_clear_mask_dir)
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(self.dir_clear_btn)
        dir_layout.addWidget(self.dir_select_btn)
        p2_l.addLayout(dir_layout)
        p2_l.addSpacing(10)

        icon = QLabel("🎨")
        icon.setStyleSheet("font-size: 36px;")
        icon.setAlignment(Qt.AlignCenter)
        t2 = QLabel(self.tr("画笔编辑模式"))
        setFont(t2, 14, QFont.Bold)
        t2.setObjectName("h2")
        t2.setAlignment(Qt.AlignCenter)
        d2 = QLabel(self.tr("请在工作区直接涂抹水印区域。"))
        d2.setObjectName("labelDesc")
        setFont(d2, 11)
        d2.setAlignment(Qt.AlignCenter)
        p2_l.addWidget(icon)
        p2_l.addWidget(t2)
        p2_l.addWidget(d2) 
        
        self.primary_btn = QPushButton(self.tr("开始标注水印"))
        self.primary_btn.setObjectName("primaryBtn")
        self.primary_btn.clicked.connect(lambda _: self._process_manual_detect())
        setFont(self.primary_btn, 13, QFont.DemiBold)
        p2_l.addWidget(self.primary_btn)
        
        self.stack.addWidget(p1)
        self.stack.addWidget(p0)
        self.stack.addWidget(p2)

        self.card_layout.addWidget(self.stack)
        self.main_layout.addWidget(self.card)

        self.btn_sem.clicked.connect(lambda: [self.btn_spa.setChecked(False), self.sub_stack.setCurrentIndex(0)])
        self.btn_spa.clicked.connect(lambda: [self.btn_sem.setChecked(False), self.sub_stack.setCurrentIndex(1)])

    def create_label_group(self, title, desc):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 12)
        l.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("labelTitle")
        setFont(t, 13, QFont.DemiBold)
        l.addWidget(t)
        if desc:
            d = QLabel(desc)
            d.setObjectName("labelDesc")
            setFont(d, 12)
            l.addWidget(d)
        return w
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        btn_width = self.tabs_container.width() // 3
        self.slider.setFixedSize(btn_width - 10, self.tabs_container.height() - 10)
        target_x = 5 + self.current_tab_index * btn_width
        self.slider.move(target_x, 5)

    def get_current_page_height(self, index: int):
        widget = self.stack.widget(index)
        widget.adjustSize()
        return widget.sizeHint().height()
    
    def animate_height_change(self, target_height: int):
        start_height = self.card.height()
        self.height_anim = QPropertyAnimation(self.card, b"maximumHeight")
        self.height_anim.setDuration(250)
        self.height_anim.setStartValue(start_height)
        self.height_anim.setEndValue(target_height)
        self.height_anim.setEasingCurve(QEasingCurve.OutCubic)
        self.height_anim.start()

    def switch_tab(self, index, animated=True):
        self.current_tab_index = index
        btn_width = self.tabs_container.width() // 3
        self.slider.setFixedSize(btn_width - 10, self.tabs_container.height() - 10)
        target_pos = QPoint(5 + index * btn_width, 5)
        
        if animated:
            self.anim = QPropertyAnimation(self.slider, b"pos")
            self.anim.setDuration(250)
            self.anim.setEndValue(target_pos)
            self.anim.setEasingCurve(QEasingCurve.OutCubic)
            self.anim.start()
        else:
            self.slider.move(target_pos)

        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.tab_btns):
            btn.setChecked(i == index)
        content_height = self.get_current_page_height(index)
        extra_height = 90  # tabs + spacing + margins
        target_height = content_height + extra_height
        self.animate_height_change(target_height)

    def apply_styles(self):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 8)
        self.card.setGraphicsEffect(shadow)

        self.setStyleSheet("""
            #mainWin { background-color: #F0F2F5; }
            #panelCard { background: white; border-radius: 24px; }
            #titleIndicator { background: #2563EB; border-radius: 2px; }
            #h2 { color: #0F172A; }
            
            #tabsContainer { background: #F1F5F9; border-radius: 14px; }
            #slider { background: white; border-radius: 10px; }
            #tabItem { 
                border: none; background: transparent; color: #64748B; 
                padding: 12px 0;
            }
            #tabItem:checked { color: #2563EB; }
            
            #labelTitle { color: #0F172A; }
            #labelDesc { color: #94A3B8; }
            
            OptionCard { background: #FAFBFC; border: 1.5px solid #E2E8F0; border-radius: 12px; }
            OptionCard[selected="true"] { background: #EFF6FF; border-color: #2563EB; }
            #optTitle { color: #0F172A; }
            OptionCard[selected="true"] #optTitle { color: #2563EB; }
            #optDesc { color: #94A3B8; }
            
            #inputBox { 
                padding: 12px; border: 1.5px solid #E2E8F0; border-radius: 12px; 
                background: white; color: #0F172A;
            }
            #inputBox:focus { border: 1.5px solid #2563EB; }
            
            #chip { 
                background: #F1F5F9; border: none; border-radius: 15px; 
                padding: 6px 12px; color: #64748B; 
            }
            #chip:hover { background: #2563EB; color: white; }
            
            #modeSwitch { background: #F1F5F9; border-radius: 10px; }
            #modeBtn { border: none; background: transparent; padding: 8px; border-radius: 8px; font-weight: 600; color: #64748B; }
            #modeBtn:checked { background: white; color: #2563EB; }
            
            #secondaryBtn { 
                background: #F1F5F9; 
                border: 1px solid #E2E8F0; 
                border-radius: 12px; 
                padding: 12px; 
                font-weight: 600; 
                color: #0F172A;
            }
            #secondaryBtn:hover { 
                background: #E2E8F0; 
                border-color: #CBD5E1;
            }
            #secondaryBtn:pressed { 
                background: #CBD5E1; 
                padding-top: 14px;
                padding-bottom: 10px;
            }
            #primaryBtn { 
                background: #2563EB; color: white; border: none; border-radius: 14px; 
                padding: 16px; margin-top: 10px;
            }
            #primaryBtn:hover { background: #1D4ED8; }
            #primaryBtn:disabled { background: #CBD5E1; color: #94A3B8; }
            #confValBadge {
                color: #2563EB;
                background-color: #EFF6FF;
                padding: 4px 10px;
                border-radius: 6px;
                min-width: 40px;
            }
            #confSlider::groove:horizontal {
                border-radius: 3px;
                height: 6px;
                background: #E2E8F0;
            }
            #confSlider::handle:horizontal {
                background: #2563EB;
                border: none;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            #confSlider::sub-page:horizontal {
                background: #2563EB;
                border-radius: 3px;
            }
        """)

    def _on_select_mask_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            self.tr("选择 Mask 目录"), 
            "", 
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontUseNativeDialog if sys.platform == "darwin" else QFileDialog.Option(0)
        )
        if dir_path:
            self.mask_dir_path = dir_path
            self.dir_input.setText(dir_path)
            self.dir_clear_btn.setVisible(True)
            self.dir_select_btn.setVisible(False)
            self.primary_btn.setEnabled(False)
            self.maskDirectoryChanged.emit(dir_path) if len(os.listdir(dir_path)) > 1 else self.maskDirectoryChanged.emit(os.path.join(dir_path, os.listdir(dir_path)[0]))

    def _on_clear_mask_dir(self):
        self.mask_dir_path = ""
        self.dir_input.clear()
        self.dir_clear_btn.setVisible(False)
        self.dir_select_btn.setVisible(True)
        self.primary_btn.setEnabled(True)
        self.maskDirectoryChanged.emit("")

    def _on_conf_changed(self, value):
        float_val = value / 100.0
        self.conf_val_label.setText(f"{float_val:.2f}")
        self.watermarkConfidence.emit(float_val)

    def set_file_path(self, file_path):
        if not file_path:
            return
        if os.path.isfile(file_path):
            self.file_path = file_path
        else:
            self.file_path = os.path.join(file_path, os.listdir(file_path)[0])

    def _watermark_area_selector(self):
        if not self.file_path:
            return
        area_selector = AreaSelectorDialog(file_path=self.file_path, parent=self)
        area_selector.exec()
        boxes = []
        for item in area_selector.get_results():
            boxes.append((item["x"], item["y"], item["x"] + item["w"], item["y"] + item["h"]))
        self.watermarkBoxes.emit(boxes)

    def _process_manual_detect(self):
        if not self.file_path:
            return
        is_video = True if get_file_type(self.file_path)=="video" else False
        watermarkMaskTool = WatermarkMaskTool(file_path=self.file_path, is_video=is_video, parent=self.window())
        watermarkMaskTool.exec()
        mask_path = watermarkMaskTool.get_mask_path()
        self.manualWatermarktMaskPath.emit(mask_path)