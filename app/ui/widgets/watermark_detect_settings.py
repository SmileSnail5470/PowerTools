from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QComboBox, QPushButton, QStackedWidget, 
                             QFrame, QGraphicsDropShadowEffect)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QPoint
from PySide6.QtGui import QColor, QFont
from app.ui.library.qfluentwidgets import setFont



class OptionCard(QFrame):
    def __init__(self, title, desc, group_name="default", parent=None):
        super().__init__(parent)
        self.group_name = group_name
        self.setCursor(Qt.PointingHandCursor)
        self.selected = False
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(2)
        
        self.title_label = QLabel(title)
        self.desc_label = QLabel(desc)
        layout.addWidget(self.title_label)
        layout.addWidget(self.desc_label)
        self.update_style()

    def update_style(self):
        color = "#2563EB" if self.selected else "#0F172A"
        desc_color = "rgba(37, 99, 235, 0.7)" if self.selected else "#94A3B8"
        bg = "#EFF6FF" if self.selected else "#FAFBFC"
        border = "#2563EB" if self.selected else "#E2E8F0"
        check_mark = "✓" if self.selected else ""
        
        self.setStyleSheet(f"OptionCard {{ background-color: {bg}; border: 1.5px solid {border}; border-radius: 12px; }}")
        self.title_label.setStyleSheet(f"color: {color}; font-weight:600; background:transparent; font-size:13px;")
        self.desc_label.setStyleSheet(f"color: {desc_color}; background:transparent; font-size:11px;")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.set_selected()

    def set_selected(self):
        siblings = self.parent().findChildren(OptionCard)
        for sibling in siblings:
            if sibling.group_name == self.group_name:
                sibling.selected = False
                sibling.update_style()
        
        self.selected = True
        self.update_style()

class WatermarkDetectSettings(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("WatermarkDetectSettings")
        self.init_ui()
        self.setStyleSheet(
            """
            QWidget#WatermarkDetectSettings { background-color: #F0F2F5; }
            QFrame#PanelCard { background-color: #FFFFFF; border-radius: 24px; }
            QFrame#TabsContainer { background-color: #F1F5F9; border-radius: 14px; }
            QPushButton.TabItem {
                border: none; background: transparent; color: #64748B;
                padding: 10px 0;
            }
            QPushButton.TabItem:checked { color: #2563EB; }
            QComboBox {
                padding: 12px; border: 2px solid #E2E8F0; border-radius: 12px;
                background: white;
            }
            QComboBox:focus { border: 2px solid #2563EB; }
            QPushButton#PrimaryBtn {
                background-color: #2563EB; color: white; border-radius: 14px;
                padding: 16px;
            }
            """
        )

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.panel_card = QFrame()
        self.panel_card.setObjectName("PanelCard")
        
        # 卡片阴影
        shadow = QGraphicsDropShadowEffect(blurRadius=30, xOffset=0, yOffset=10)
        shadow.setColor(QColor(0, 0, 0, 30))
        self.panel_card.setGraphicsEffect(shadow)
        
        card_layout = QVBoxLayout(self.panel_card)
        card_layout.setContentsMargins(28, 28, 28, 28)
        card_layout.setSpacing(0)

        # Tab 导航
        self.tab_widget = QFrame()
        self.tab_widget.setObjectName("TabsContainer")
        self.tab_widget.setFixedHeight(50)
        tab_layout = QHBoxLayout(self.tab_widget)
        tab_layout.setContentsMargins(5, 5, 5, 5)
        
        # 滑块背景
        self.slider_bg = QFrame(self.tab_widget)
        self.slider_bg.setFixedSize(138, 40)
        self.slider_bg.setStyleSheet("background: white; border-radius: 10px;")
        self.slider_bg.move(5, 5)
        
        self.tab_buttons = []
        for i, text in enumerate([self.tr("AI 全自动"), self.tr("AI 交互"), self.tr("手工标注")]):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setFixedSize(143, 40)
            btn.setProperty("class", "TabItem")
            setFont(btn, 13, QFont.Bold)
            btn.clicked.connect(lambda _, idx=i: self.switch_tab(idx))
            if i == 0: btn.setChecked(True)
            self.tab_buttons.append(btn)
            tab_layout.addWidget(btn)
        
        card_layout.addWidget(self.tab_widget)
        card_layout.addSpacing(28)

        # Stacked 内容区
        self.stack = QStackedWidget()
        
        # --- Page 0: AI 全自动 ---
        p0 = QWidget()
        l0 = QVBoxLayout(p0)
        l0.setContentsMargins(0,0,0,0)
        
        l0.addLayout(self.create_label_group(self.tr("水印类型"), self.tr("系统将自动分析画面中的元素")))
        g1_container = QWidget()
        g1 = QHBoxLayout(g1_container)
        g1.setContentsMargins(0,0,0,0)
        card_common = OptionCard(self.tr("通用水印"), "Logo、图案、复合元素", "type", g1_container)
        card_text = OptionCard(self.tr("文字水印"), "纯文本、日期、字幕", "type", g1_container)
        g1.addWidget(card_common)
        g1.addWidget(card_text)
        l0.addWidget(g1_container)
        
        l0.addSpacing(24)
        
        l0.addLayout(self.create_label_group(self.tr("水印形式"), self.tr("选择水印在时间轴上的状态(视频文件)")))
        g2_container = QWidget()
        g2 = QHBoxLayout(g2_container)
        g2.setContentsMargins(0,0,0,0)
        card_static = OptionCard(self.tr("静态水印"), self.tr("固定位置不移动"), "dim", g2_container)
        card_dynamic = OptionCard(self.tr("动态水印"), self.tr("移动、缩放、淡入淡出"), "dim", g2_container)
        g2.addWidget(card_static)
        g2.addWidget(card_dynamic)
        l0.addWidget(g2_container)
        
        # 设置默认选中
        card_common.set_selected()
        card_static.set_selected()
        l0.addStretch()

        # --- Page 1: AI 交互 ---
        p1 = QWidget()
        l1 = QVBoxLayout(p1)
        l1.setContentsMargins(0,0,0,0)
        l1.addLayout(self.create_label_group(self.tr("语义识别引导"), self.tr("通过自然语言描述水印内容")))
        # l1_text_promt = QLineEdit(placeholderText="例如: watermark,subtitle")
        l1_text_prompt = QComboBox()
        l1_text_prompt.setEditable(True)
        l1_text_prompt.addItems([
            self.tr("水印"),
            self.tr("字幕"),
            self.tr("水印加字幕")
        ])
        l1_text_prompt.setCurrentText(self.tr("水印"))
        setFont(l1_text_prompt, 13)
        l1.addWidget(l1_text_prompt)
        l1.addSpacing(24)
        l1.addLayout(self.create_label_group(self.tr("空间范围引导"), self.tr("在预览窗口精准框选")))
        hint = QFrame()
        hint.setStyleSheet("border: 1px dashed #E2E8F0; border-radius: 12px; background: #F8FAFC;")
        hv = QVBoxLayout(hint)
        hv.setAlignment(Qt.AlignCenter)
        hv.addWidget(QLabel("🎯", styleSheet="font-size: 18px;"), alignment=Qt.AlignCenter)
        hv.addWidget(QLabel("开启框选模式", styleSheet="font-size: 11px; color: #64748B;"), alignment=Qt.AlignCenter)
        l1.addWidget(hint)
        l1.addStretch()

        # --- Page 2: 手工标注 ---
        p2 = QWidget()
        l2 = QVBoxLayout(p2)
        l2.setContentsMargins(0,0,0,0)
        l2.setAlignment(Qt.AlignCenter)
        l2.addSpacing(20)
        l2.addWidget(QLabel("🎨", styleSheet="font-size: 36px;"), alignment=Qt.AlignCenter)
        l2.addWidget(QLabel(self.tr("进入画笔编辑模式"), styleSheet="font-weight: 600; font-size: 15px;"), alignment=Qt.AlignCenter)
        l2.addWidget(QLabel(self.tr("请在工作区通过涂抹选择水印。"), 
                           styleSheet="color: #94A3B8; font-size: 11px; margin-top: 8px;"), alignment=Qt.AlignCenter)
        l2.addSpacing(20)
        submit_btn = QPushButton("开始手动标注")
        submit_btn.setObjectName("PrimaryBtn")
        setFont(submit_btn, 16, QFont.Bold)
        l2.addWidget(submit_btn)
        l2.addSpacing(20)

        self.stack.addWidget(p0)
        self.stack.addWidget(p1)
        self.stack.addWidget(p2)
        card_layout.addWidget(self.stack)

        layout.addWidget(self.panel_card, alignment=Qt.AlignCenter)

        # 动画对象
        self.anim = QPropertyAnimation(self.slider_bg, b"pos")
        self.anim.setDuration(250)
        self.anim.setEasingCurve(QEasingCurve.OutQuad)

    def create_label_group(self, title, desc):
        v = QVBoxLayout()
        v.setSpacing(2)
        t = QLabel(title)
        setFont(t, 14, QFont.Bold)
        t.setStyleSheet("color: #0F172A;")
        d = QLabel(desc)
        setFont(d, 11)
        d.setStyleSheet("color: #94A3B8;")
        v.addWidget(t); v.addWidget(d); v.addSpacing(8)
        return v

    def switch_tab(self, idx):
        target_x = 5 + (idx * 143)
        self.anim.setEndValue(QPoint(target_x, 5))
        self.anim.start()
        self.stack.setCurrentIndex(idx)