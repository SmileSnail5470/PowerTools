from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame
from PySide6.QtGui import QFont
from app.ui.library.qfluentwidgets import setFont


class CustomNavigation(QWidget):
    currentIndexChanged = Signal(int)
    currentTextChanged = Signal(str)
    itemClicked = Signal(int, str)

    def __init__(
            self,
            navigation_item_texts=[
                QWidget.tr("结果对比预览"),
                QWidget.tr("Mask对比预览")
            ],
            parent=None
    ):
        super().__init__(parent=parent)
        self.current_index = 0
        self.init_ui(navigation_item_texts)

    def init_ui(self, navigation_item_texts):
        self.setObjectName("CustomNavigation")
        self.setStyleSheet("""
            #CustomNavigation {
                background-color: #f8fafc;
            }
        """)

        self.nav_container = QFrame(self)
        self.nav_container.setContentsMargins(0, 0, 0, 0)
        self.nav_container.setObjectName("navContainer")
        self.nav_container.setStyleSheet("""
            #navContainer {
                background-color: #ffffff;
                border-radius: 14px;
                border: 1px solid rgba(226, 232, 240, 1);
            }
        """)

        self.slider = QFrame(self.nav_container)
        self.slider.setStyleSheet("""
            background-color: #eff6ff;
            border-radius: 10px;
            border: 1px solid rgba(59, 130, 246, 25);
        """)

        self.layout = QHBoxLayout(self.nav_container)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(4)

        self.items = []
        for index, text in enumerate(navigation_item_texts):
            item = NavItem(text, index, self)
            self.items.append(item)
            self.layout.addWidget(item)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 5, 0, 0)
        main_layout.addWidget(self.nav_container, alignment=Qt.AlignCenter)

        self.animation = QPropertyAnimation(self.slider, b"geometry")
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)

        self.current_active = self.items[0]
        self.items[0].set_active(True)

    def showEvent(self, event):
        super().showEvent(event)
        self.update_slider(self.items[0], animate=False)

    def update_slider(self, target, animate=True):
        if animate:
            self.animation.stop()
            self.animation.setStartValue(self.slider.geometry())
            self.animation.setEndValue(target.geometry())
            self.animation.start()
        else:
            self.slider.setGeometry(target.geometry())

    def set_current_index(self, index: int, emit_signal=True):
        if index < 0 or index >= len(self.items):
            return
        target = self.items[index]
        for item in self.items:
            item.set_active(False)
        target.set_active(True)
        self.current_active = target
        self.current_index = index
        self.update_slider(target)
        if emit_signal:
            self.currentIndexChanged.emit(index)
            self.currentTextChanged.emit(target.text())
            self.itemClicked.emit(index, target.text())


class NavItem(QLabel):
    def __init__(self, text, index, parent_nav):
        super().__init__(text)
        self.index = index
        self.parent_nav = parent_nav
        self.is_active = False

        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(30)
        self.setContentsMargins(14, 0, 14, 0)

        setFont(self, 11, weight=QFont.Bold)

        self.update_style()

    def update_style(self):
        color = "#3b82f6" if self.is_active else "#64748b"
        self.setStyleSheet(f"""
            color: {color};
            background: transparent;
            border: none;
        """)

    def set_active(self, active):
        self.is_active = active
        self.update_style()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.parent_nav.set_current_index(self.index)