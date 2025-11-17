from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QGridLayout
from PySide6.QtGui import QFont, QPainter, QLinearGradient, QColor, QCursor, QRadialGradient

from app.ui.library.qfluentwidgets import ScrollArea, setFont

from app.ui.widgets.gradient_header_widget import GradientHeader


class FeatureCard(QFrame):
    clicked = Signal(str)
    
    def __init__(self, icon: str, title: str, desc: str, badge=None, card_type="default", features=[], parent=None):
        super().__init__(parent)
        self.card_type = card_type
        self.icon = icon
        self.title = title
        self.desc = desc
        self.badge = badge
        self.features = features

        self.setObjectName("FeatureCard")
        self.setCursor(QCursor(Qt.PointingHandCursor))
        
        self._setup_ui()
        self._setup_style_sheet()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        self.gradient_bar = QFrame()
        self.gradient_bar.setFixedHeight(4)
        self.gradient_bar.setStyleSheet(f"""
            QFrame {{
                background: {self._get_gradient(self.card_type)};
                border-radius: 2px;
            }}
        """)
        
        header = QHBoxLayout()
        
        icon_label = QLabel(self.icon)
        icon_label.setFixedSize(62, 62)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet(f"""
            QLabel {{
                background: {self._get_gradient(self.card_type)};
                border-radius: 12px;
                color: white;
            }}
        """)
        setFont(icon_label, 32, QFont.Bold)
        
        title_block = QVBoxLayout()
        title_label = QLabel(self.title)
        setFont(title_label, 16, QFont.Bold)
        title_label.setStyleSheet("color: #1a1a1a;")
        title_block.addWidget(title_label)
        
        if self.badge:
            badge_label = QLabel(self.badge)
            badge_label.setStyleSheet(f"""
                QLabel {{
                    background: {self._get_gradient(self.card_type)};
                    color: white;
                    border-radius: 20px;
                    padding: 3px 12px;
                }}
            """)
            setFont(badge_label, 12, QFont.DemiBold)
            title_block.addWidget(badge_label)
        
        header.addWidget(icon_label)
        header.addLayout(title_block)
        header.addStretch()
        
        desc_label = QLabel(self.desc)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("""
            QLabel {
                color: #666666;
                line-height: 1.5;
            }
        """)
        setFont(desc_label, 13)
        
        features_layout = QVBoxLayout()
        for feature in self.features:
            feature_item = QHBoxLayout()
            check_label = QLabel("✓")
            check_label.setStyleSheet("""
                QLabel {
                    color: #4CAF50;
                    min-width: 20px;
                    font-size: 13;
                }
            """)
            feature_label = QLabel(self.tr(feature))
            feature_label.setStyleSheet("""
                QLabel {
                    color: #666666;
                }
            """)
            setFont(feature_label, 12)
            feature_item.addWidget(check_label)
            feature_item.addWidget(feature_label)
            feature_item.addStretch()
            features_layout.addLayout(feature_item)
        
        action_btn = QPushButton(self.tr("立即使用 →"))
        action_btn.setCursor(QCursor(Qt.PointingHandCursor))
        setFont(action_btn, 14, QFont.Bold)
        action_btn.setStyleSheet(f"""
            QPushButton {{
                background: #f8f9fa;
                border: 2px solid transparent;
                border-radius: 10px;
                padding: 12px 20px;
                color: #1a1a1a;
            }}
            QPushButton:hover {{
                background: {self._get_gradient(self.card_type)};
                color: white;
                border: 2px solid transparent;
            }}
        """)
        action_btn.clicked.connect(lambda: self.clicked.emit(self.card_type))
        
        layout.addWidget(self.gradient_bar)
        layout.addLayout(header)
        layout.addWidget(desc_label)
        layout.addLayout(features_layout)
        layout.addStretch()
        layout.addWidget(action_btn)

    def _setup_style_sheet(self):
        self.setStyleSheet("""
            QFrame#FeatureCard {
                background: rgba(255, 255, 255, 0.98);
                border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            QFrame#FeatureCard:hover {
                border: 1px solid rgba(255, 255, 255, 0.4);
            }
        """)
    
    def _get_gradient(self, card_type):
        gradients = {
            'watermark-add': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #667eea, stop:1 #764ba2)',
            'watermark-remove': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4facfe, stop:1 #00f2fe)',
            'screenshot': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #fa709a, stop:1 #fee140)',
            'scroll-screenshot': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #30cfd0, stop:1 #330867)',
            'text-extract': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f093fb, stop:1 #f5576c)',
            'image-edit': 'qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #a8edea, stop:1 #fed6e3)'
        }
        return gradients.get(card_type, gradients['watermark-add'])
    

class HeroSection(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setAttribute(Qt.WA_StyledBackground, True)
        
        self._create_ui()
        self._apply_style()

    def _create_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(180, 30, 180, 20)
        
        self.title = QLabel("PowerTools")
        self.title.setObjectName("hero_title")
        self.title.setAlignment(Qt.AlignCenter)
        setFont(self.title, 52, QFont.Bold)
        
        self.subtitle = QLabel(
            self.tr("图像处理工具集，为您的创意工作提供支持。从水印管理到截图，从文字识别到图像编辑，一站式解决您的需求。")
        )
        self.subtitle.setObjectName("hero_subtitle")
        self.subtitle.setAlignment(Qt.AlignCenter)
        self.subtitle.setWordWrap(True)
        setFont(self.subtitle, 18)
        
        layout.addWidget(self.title)
        layout.addSpacing(10)
        layout.addWidget(self.subtitle)

    def _apply_style(self):
        self.setStyleSheet("""
            QLabel#hero_title {
                color: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #667eea, stop:0.5 #764ba2, stop:1 #f093fb);
                letter-spacing: -1px;
            }
            QLabel#hero_subtitle {
                color: rgba(26, 26, 26, 0.8);
                line-height: 1.6;
            }
        """)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        base_grad = QLinearGradient(0, 0, self.width(), self.height())
        base_grad.setColorAt(0.0, QColor("#fafbff"))
        base_grad.setColorAt(0.5, QColor("#f5f7ff"))
        base_grad.setColorAt(1.0, QColor("#f0f3ff"))
        painter.fillRect(self.rect(), base_grad)
        
        left_grad = QLinearGradient(0, 0, self.width() * 0.6, self.height())
        left_grad.setColorAt(0.0, QColor(102, 126, 234, 25))
        left_grad.setColorAt(0.7, QColor(118, 75, 162, 15))
        left_grad.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.fillRect(self.rect(), left_grad)
        
        right_grad = QLinearGradient(self.width() * 0.4, 0, self.width(), self.height())
        right_grad.setColorAt(0.0, QColor(255, 255, 255, 0))
        right_grad.setColorAt(0.3, QColor(240, 147, 251, 15))
        right_grad.setColorAt(1.0, QColor(245, 87, 108, 25))
        painter.fillRect(self.rect(), right_grad)
        
        top_glow = QRadialGradient(
            self.width() // 2, 
            self.height() * 0.1, 
            self.width() * 0.8
        )
        top_glow.setColorAt(0.0, QColor(255, 255, 255, 180))
        top_glow.setColorAt(0.3, QColor(230, 230, 255, 100))
        top_glow.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.fillRect(self.rect(), top_glow)
        
        spots = [
            (self.width() * 0.15, self.height() * 0.2, 150, QColor(102, 126, 234, 30)),
            (self.width() * 0.85, self.height() * 0.3, 120, QColor(240, 147, 251, 25)),
            (self.width() * 0.25, self.height() * 0.8, 100, QColor(118, 75, 162, 20)),
            (self.width() * 0.75, self.height() * 0.7, 130, QColor(245, 87, 108, 22)),
        ]
        
        for x, y, radius, color in spots:
            spot = QRadialGradient(x, y, radius)
            spot.setColorAt(0.0, color)
            spot.setColorAt(0.6, QColor(color.red(), color.green(), color.blue(), color.alpha() // 2))
            spot.setColorAt(1.0, QColor(255, 255, 255, 0))
            painter.fillRect(self.rect(), spot)
        
        super().paintEvent(event)
    

class Home(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("Home")
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self._setup_header(main_layout)
        
        self._setup_content(main_layout)
    
    def _setup_header(self, main_layout: QVBoxLayout):
        header = GradientHeader(parent=self)
        
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(30, 20, 30, 20)
        header_layout.setSpacing(10)
        
        title_label = QLabel(self.tr("🏠 主页"))
        setFont(title_label, fontSize=24, weight=QFont.Bold)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
            }
        """)
        header_layout.addWidget(title_label)  
        header_layout.addStretch()
        
        main_layout.addWidget(header)
    
    def _setup_content(self, main_layout: QVBoxLayout):
        scroll = ScrollArea()
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)
        content_layout.setAlignment(Qt.AlignTop)
        
        hero = self._create_hero_section()
        content_layout.addWidget(hero)
        
        features = self._create_features_section()
        content_layout.addWidget(features)
        
        scroll.setWidget(content)

        scroll.setWidgetResizable(True)
        scroll.enableTransparentBackground()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        main_layout.addWidget(scroll)
    
    def _create_hero_section(self):
        hero = HeroSection()
        return hero
    
    def _create_features_section(self):
        features = QWidget()
        features_layout = QVBoxLayout(features)
        features_layout.setContentsMargins(10, 0, 16, 10)
        features_layout.setAlignment(Qt.AlignCenter)
        
        cards_widget = QWidget()
        cards_layout = QGridLayout(cards_widget)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(12)
        cards_layout.setAlignment(Qt.AlignCenter)
        
        cards_data = [
            ("💧", "水印添加", "为您的图片添加个性化水印，保护版权，提升品牌识别度。支持多种水印样式和自定义设置。", None, "watermark-add"),
            ("🧹", "水印移除", "智能识别并移除图片中的水印，还原图片原始状态。采用先进的AI算法，确保移除效果自然。", "AI驱动", "watermark-remove"),
            ("📸", "屏幕截图", "快速截取屏幕内容，支持多种截图模式和编辑功能。内置强大的编辑工具，让截图更加专业。", None, "screenshot"),
            ("📜", "滚动截图", "智能滚动并截取长页面内容，完美保存完整信息。自动识别滚动区域，无需手动操作。", None, "scroll-screenshot"),
            ("📝", "文字提取", "OCR智能识别图片中的文字，支持多语言高精度识别。先进的识别引擎，确保文字提取准确率。", None, "text-extract"),
            ("🎨", "图像编辑", "专业的图像编辑工具，满足您的各种创意需求。丰富的滤镜效果和编辑功能，让图片处理更加简单。", None, "image-edit")
        ]

        features_map = {
            'watermark-add': ['支持文字和图片水印', '自定义位置和透明度', '批量处理功能', '丰富的水印模板'],
            'watermark-remove': ['AI智能识别技术', '保持图片质量', '支持多种水印类型', '一键批量处理'],
            'screenshot': ['区域截图和窗口截图', '内置编辑工具', '快捷键支持', '云同步功能'],
            'scroll-screenshot': ['自动滚动检测', '智能拼接算法', '支持网页和文档', '高质量输出'],
            'text-extract': ['多语言支持', '高精度识别', '可编辑和导出', '表格识别功能'],
            'image-edit': ['丰富的滤镜效果', '图层编辑功能', '批量处理支持', 'AI智能优化']
        }
        
        self.cards = []
        for i, (icon, title, desc, badge, card_type) in enumerate(cards_data):
            card = FeatureCard(icon, title, desc, badge, card_type, features_map[card_type])
            card.clicked.connect(self.handle_card_click)
            self.cards.append(card)
            row = i // 3
            col = i % 3
            cards_layout.addWidget(card, row, col)
        
        features_layout.addWidget(cards_widget)
        
        return features
    
    def handle_card_click(self, card_type):
        card_index = {
            'watermark-add': 2,
            'watermark-remove': 3,
            'screenshot': 4,
            'scroll-screenshot': 5,
            'text-extract': 6,
            'image-edit': 7
        }
        index = card_index.get(card_type, 0)
        self.window().stackedWidget.setCurrentIndex(index)
    
    def paintEvent(self, event):
        # p = QPainter(self)
        # grad = QLinearGradient(0, 0, self.width(), self.height())
        # grad.setColorAt(0, QColor(255, 255, 255, 80))   # 上半透明白色
        # grad.setColorAt(1, QColor(245, 245, 245, 50))   # 下半透明米白色，透明度更低
        # p.fillRect(self.rect(), QBrush(grad))
        super().paintEvent(event)