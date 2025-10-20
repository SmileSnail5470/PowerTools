from PySide6.QtCore import Signal, QEasingCurve, Qt
from PySide6.QtWidgets import QVBoxLayout, QLabel, QWidget, QHBoxLayout
from app.ui.library.qfluentwidgets import PushButton, FlowLayout, SingleDirectionScrollArea


class ColorButton(PushButton):
    color_selected = Signal(str)
    
    def __init__(self, color_name, color_code, size=20):
        super().__init__()
        self.color_name = color_name
        self.color_code = color_code
        self.setFixedSize(size, size)
        self.setStyleSheet(f"""
            PushButton {{
                background-color: {color_code};
                border: 2px solid #ccc;
                border-radius: 4px;
            }}
            PushButton:hover {{
                border: 2px solid #333;
            }}
            PushButton:pressed {{
                border: 2px solid #000;
            }}
        """)
        self.clicked.connect(self.on_clicked)
        self.setToolTip(color_name)
    
    def on_clicked(self):
        self.color_selected.emit(self.color_code)


class ColorPicker(QWidget):
    """颜色选择器面板"""
    color_changed = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.selected_color = "#000000"  # 默认黑色
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        # 预定义颜色列表
        colors = [
            ("黑色", "#000000"), ("深灰", "#333333"), ("中灰", "#666666"), 
            ("浅灰", "#999999"), ("白色", "#FFFFFF"), ("红色", "#FF0000"),
            ("深红", "#800000"), ("橙色", "#FFA500"), ("黄色", "#FFFF00"),
            ("金黄", "#FFD700"), ("绿色", "#00FF00"), ("深绿", "#008000"),
            ("青色", "#00FFFF"), ("深青", "#008080"), ("蓝色", "#0000FF"),
            ("深蓝", "#000080"), ("紫色", "#800080"), ("粉色", "#FFC0CB"),
            ("棕色", "#A52A2A"), ("橄榄", "#808000"), ("海军", "#000080"),
            ("青绿", "#008B8B"), ("深橙", "#FF8C00"), ("番茄", "#FF6347"),
            ("紫罗兰", "#EE82EE"), ("天蓝", "#87CEEB"), ("巧克力", "#D2691E"),
            ("珊瑚", "#FF7F50"), ("深紫", "#4B0082"), ("森林绿", "#228B22"),
            ("金", "#FFD700"), ("靛蓝", "#4B0082"), ("薰衣草", "#E6E6FA"),
            ("酸橙", "#00FF00"), ("栗色", "#800000"), ("海军蓝", "#000080"),
            ("橄榄绿", "#808000"), ("橙红", "#FF4500"), ("兰花", "#DA70D6"),
            ("秘鲁", "#CD853F"), ("粉红", "#FFC0CB"), ("李子", "#DDA0DD"),
            ("鞍棕", "#8B4513"), ("蓝宝石", "#0F52BA"), ("银", "#C0C0C0"),
            ("青柠", "#32CD32"), ("蓝绿", "#008B8B"), ("紫红", "#FF00FF"),
            ("火砖", "#B22222"), ("钢蓝", "#4682B4"), ("黄绿", "#9ACD32")
        ]
        
        flow_layout = FlowLayout(None, needAni=True)
        flow_layout.setAnimation(250, QEasingCurve.OutQuad)
        flow_layout.setContentsMargins(10, 10, 10, 10)
        flow_layout.setVerticalSpacing(5)
        flow_layout.setHorizontalSpacing(5)

        for color_name, color_code in colors:
            color_btn = ColorButton(color_name, color_code)
            color_btn.color_selected.connect(self.on_color_selected)
            flow_layout.addWidget(color_btn)
        
        # 创建滚动区域
        scroll_area = SingleDirectionScrollArea()
        scroll_widget = QWidget()
        scroll_widget.setLayout(flow_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedHeight(80)
        scroll_area.enableTransparentBackground()
        
        # 当前颜色显示
        self.current_color_label = QLabel("当前颜色:")
        self.color_display = QLabel()
        self.color_display.setFixedSize(50, 30)
        self.color_display.setStyleSheet(f"background-color: {self.selected_color}; border: 1px solid black;")
        
        color_info_layout = QHBoxLayout()
        color_info_layout.addWidget(self.current_color_label)
        color_info_layout.addWidget(self.color_display)
        color_info_layout.addStretch()
        
        # 添加到主布局
        layout.addWidget(QLabel("选择字体颜色:"))
        layout.addWidget(scroll_area)
        layout.addLayout(color_info_layout)
        
    def on_color_selected(self, color_code):
        """处理颜色选择"""
        self.selected_color = color_code
        self.color_display.setStyleSheet(f"background-color: {color_code}; border: 1px solid black;")
        self.color_changed.emit(color_code)