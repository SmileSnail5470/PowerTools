from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QLineEdit, QStackedWidget, 
                             QFrame, QGraphicsDropShadowEffect)
from PySide6.QtCore import Qt, QPropertyAnimation, QPoint, QEasingCurve, QTimer
from PySide6.QtGui import QColor


class OptionCard(QFrame):
    def __init__(self, title, desc, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.selected = False
        
        layout = QVBoxLayout(self)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("optTitle")
        self.desc_label = QLabel(desc)
        self.desc_label.setObjectName("optDesc")
        
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

class WatermarkDetectSettings(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.current_tab_index = 0
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
        for i, text in enumerate([self.tr("AI 交互"), self.tr("AI 全自动"), self.tr("手工标注")]):
            btn = QPushButton(text)
            btn.setObjectName("tabItem")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _, idx=i: self.switch_tab(idx))
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
        c1.selected = True
        c1.update_style()
        grid0_l.addWidget(c1)
        grid0_l.addWidget(OptionCard(self.tr("文字水印"), self.tr("纯文本、日期、字幕")))
        p0_l.addWidget(grid0)
        p0_l.addSpacing(24)
        
        p0_l.addWidget(self.create_label_group(self.tr("水印形式"), self.tr("选择水印在时间轴上的状态")))
        grid1 = QWidget()
        grid1_l = QHBoxLayout(grid1)
        grid1_l.setContentsMargins(0,0,0,0)
        c2 = OptionCard(self.tr("静态水印"), self.tr("固定位置不移动"))
        c2.selected = True
        c2.update_style()
        grid1_l.addWidget(c2)
        grid1_l.addWidget(OptionCard(self.tr("动态水印"), self.tr("移动、缩放、淡入淡出")))
        p0_l.addWidget(grid1)
        p0_l.addStretch()

        # AI 交互 ---
        p1 = QWidget()
        p1_l = QVBoxLayout(p1)
        p1_l.setContentsMargins(0,0,0,0)
        self.mode_switch = QFrame()
        self.mode_switch.setObjectName("modeSwitch")
        ms_l = QHBoxLayout(self.mode_switch)
        ms_l.setContentsMargins(4,4,4,4)
        self.btn_sem = QPushButton(self.tr("语义识别"))
        self.btn_spa = QPushButton(self.tr("空间范围识别"))
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
        self.input_box.setPlaceholderText(self.tr("文本描述（请输入英文）..."))
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
            cp.clicked.connect(lambda _, text=t: self.input_box.setText(text_map[text]))
            chip_l.addWidget(cp)
        chip_l.addStretch()
        sem_l.addLayout(chip_l)
        self.sub_stack.addWidget(sem_w)
        
        spa_w = QWidget()
        spa_l = QVBoxLayout(spa_w)
        spa_l.setContentsMargins(0,10,0,10)
        spa_l.addWidget(self.create_label_group(self.tr("精准框选"), self.tr("在图片区手动框选位置")))
        sec_btn = QPushButton(self.tr("🎯 开始水印框选"))
        sec_btn.setObjectName("secondaryBtn")
        spa_l.addWidget(sec_btn); self.sub_stack.addWidget(spa_w)
        
        p1_l.addWidget(self.sub_stack)
        p1_l.addSpacing(20)
        p1_l.addWidget(self.create_label_group(self.tr("水印形式"), self.tr("选择水印在时间轴上的状态")))
        grid2 = QWidget()
        grid2_l = QHBoxLayout(grid2)
        grid2_l.setContentsMargins(0,0,0,0)
        c3 = OptionCard(self.tr("静态水印"), self.tr("固定位置不移动"))
        c3.selected = True
        grid2_l.addWidget(c3)
        grid2_l.addWidget(OptionCard(self.tr("动态水印"), self.tr("移动、缩放、淡入淡出")))
        p1_l.addWidget(grid2)
        p1_l.addStretch()

        # 手工标注 ---
        p2 = QWidget()
        p2_l = QVBoxLayout(p2)
        p2_l.setContentsMargins(0,40,0,0)
        icon = QLabel("🎨")
        icon.setStyleSheet("font-size: 36px;")
        icon.setAlignment(Qt.AlignCenter)
        t2 = QLabel(self.tr("画笔编辑模式"))
        t2.setObjectName("h2")
        t2.setAlignment(Qt.AlignCenter)
        d2 = QLabel(self.tr("请在工作区直接涂抹水印区域。"))
        d2.setObjectName("labelDesc")
        d2.setAlignment(Qt.AlignCenter)
        p2_l.addWidget(icon)
        p2_l.addWidget(t2)
        p2_l.addWidget(d2) 
        p2_l.addStretch()
        
        self.primary_btn = QPushButton(self.tr("开始标注水印"))
        self.primary_btn.setObjectName("primaryBtn")
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
        l.addWidget(t)
        if desc:
            d = QLabel(desc)
            d.setObjectName("labelDesc")
            l.addWidget(d)
        return w
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        btn_width = self.tabs_container.width() // 3
        self.slider.setFixedSize(btn_width - 10, self.tabs_container.height() - 10)
        target_x = 5 + self.current_tab_index * btn_width
        self.slider.move(target_x, 5)

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
            #h2 { font-size: 20px; font-weight: 700; color: #0F172A; }
            
            #tabsContainer { background: #F1F5F9; border-radius: 14px; }
            #slider { background: white; border-radius: 10px; }
            #tabItem { 
                border: none; background: transparent; color: #64748B; 
                font-weight: 600; font-size: 14px; padding: 12px 0;
            }
            #tabItem:checked { color: #2563EB; }
            
            #labelTitle { font-size: 15px; font-weight: 600; color: #0F172A; }
            #labelDesc { font-size: 12px; color: #94A3B8; }
            
            OptionCard { background: #FAFBFC; border: 1.5px solid #E2E8F0; border-radius: 12px; }
            OptionCard[selected="true"] { background: #EFF6FF; border-color: #2563EB; }
            #optTitle { font-size: 14px; font-weight: 600; color: #0F172A; }
            OptionCard[selected="true"] #optTitle { color: #2563EB; }
            #optDesc { font-size: 11px; color: #94A3B8; }
            
            #inputBox { 
                padding: 12px; border: 1.5px solid #E2E8F0; border-radius: 12px; 
                font-size: 14px; background: white;
            }
            #inputBox:focus { border: 1.5px solid #2563EB; }
            
            #chip { 
                background: #F1F5F9; border: none; border-radius: 15px; 
                padding: 6px 12px; color: #64748B; font-size: 12px; 
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
                padding: 16px; font-size: 16px; font-weight: 600; margin-top: 10px;
            }
            #primaryBtn:hover { background: #1D4ED8; }
        """)