from PySide6.QtCore import QSize, QTimer, QUrl, Qt
from PySide6.QtGui import QIcon, QDesktopServices, QPixmap, QPainter, QColor
from PySide6.QtWidgets import QApplication, QLabel

from app.ui.library.qfluentwidgets import (
    NavigationItemPosition, FluentWindow, SplashScreen, SystemThemeListener, isDarkTheme, MessageBoxBase,
    SubtitleLabel, CaptionLabel, setFont
)
from app.ui.library.qfluentwidgets import FluentIcon as FIF

from app.ui.view.home import Home
from app.ui.view.settings import Settings
from app.ui.view.image_edit import ImageEdit
from app.ui.view.ocr import OCR
from app.ui.view.screenshot import Screenshot
from app.ui.view.scroll_screenshot import ScrollScreenshot
from app.ui.view.watermark_add import WatermarkAdd
from app.ui.view.watermark_remove import WatermarkRemove
from app.ui.widgets.resources_monitor_widget import ResourcesMonitorWidget

from app.ui.common.config import cfg
from app.ui.common.icon import Icon
from app.ui.resources import resource


class MainWindow(FluentWindow):

    def __init__(self):
        super().__init__()
        self.navigationInterface.setExpandWidth(222)

        self.initWindow()

        # create system theme listener
        self.themeListener = SystemThemeListener(self)

        # create sub interface
        self.homeInterface = Home(self)
        self.settingInterface = Settings(self)
        self.watermarkRemoveInterface = WatermarkRemove(self)
        self.watermarkAddInterface = WatermarkAdd(self)
        self.screenshotInterface = Screenshot(self)
        self.scrollScreenshotInterface = ScrollScreenshot(self)
        self.OCRInterface = OCR(self)
        self.imageEditInterface = ImageEdit(self)

        # enable acrylic effect
        self.navigationInterface.setAcrylicEnabled(True)
        # disable collapsible
        self.navigationInterface.setCollapsible(False)
        # hide menu button
        self.navigationInterface.setMenuButtonVisible(False)

        # add items to navigation interface
        self.initNavigation()
        self.splashScreen.finish()

        # start theme listener
        self.themeListener.start()

    def initNavigation(self):
        # add navigation items
        self.addSubInterface(self.homeInterface, FIF.HOME, self.tr('主页'))
        self.addSubInterface(self.settingInterface, FIF.SETTING, self.tr("常规"))
        self.navigationInterface.addSeparator()

        pos = NavigationItemPosition.SCROLL
        self.addSubInterface(self.watermarkAddInterface, Icon.WATERMARK_ADD, self.tr("水印添加"), pos, parent=None)
        self.addSubInterface(self.watermarkRemoveInterface, Icon.WATERMARK_REMOVE, self.tr("水印移除"), pos, parent=None)

        # self.addSubInterface(self.screenshotInterface, Icon.SCREENSHOT, self.tr("屏幕截图"), pos, parent=None)
        # self.addSubInterface(self.scrollScreenshotInterface, Icon.LONG_SCREENSHOT, self.tr("滚动截图"), pos, parent=None)

        self.addSubInterface(self.OCRInterface, Icon.OCR, self.tr("文字提取"), pos, parent=None)

        self.addSubInterface(self.imageEditInterface, Icon.IMAGE_EDIT, self.tr("图像编辑"), pos, parent=None)

        # add custom widget to bottom
        self.navigationInterface.addItem(
            routeKey='use-powertools',
            icon=Icon.Price,
            text=self.tr("欢迎使用 PowerTools"),
            onClick=self.welcome,
            selectable=False,
            tooltip=self.tr("Price"),
            position=NavigationItemPosition.BOTTOM
        )

        self.navigationInterface.addItem(
            routeKey='feedback',
            icon=FIF.FEEDBACK,
            text=self.tr("反馈"),
            onClick=self.onSupport,
            selectable=False,
            tooltip=self.tr("Price"),
            position=NavigationItemPosition.BOTTOM
        )

    def initWindow(self):
        self.resize(1300, 900)
        self.setMinimumWidth(1300)
        self.setWindowIcon(QIcon(':/powertools/images/logo.png'))
        self.setWindowTitle(self.tr("PowerTools"))
        
        self.resource_monitor_widget = ResourcesMonitorWidget(self)
        # 资源监控组件居中显示
        self.titleBar.hBoxLayout.insertStretch(2, stretch=2)
        self.titleBar.hBoxLayout.insertWidget(3, self.resource_monitor_widget, stretch=0)
        self.titleBar.hBoxLayout.insertStretch(4, stretch=1)

        # only win11 enable mica effect
        self.setMicaEffectEnabled(cfg.get(cfg.micaEnabled))

        # create splash screen
        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(128, 128))
        self.splashScreen.raise_()    # 保证启动画面浮在顶层，防止被遮住

        desktop = QApplication.screens()[0].availableGeometry()  # 获取屏幕的可用区域，不包括任务栏
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)
        self.show()
        QApplication.processEvents()  # 立即处理所有挂起的事件，而不是等待事件循环自然处理（鼠标点击、窗口重绘、定时器事件都被放入事件队列）

    def onSupport(self):
        url = QUrl("https://github.com/SmileSnail5470/PowerTools/issues")
        QDesktopServices.openUrl(url)

    def _combine_qr_codes(self, ali_path, wx_path):
        ali_pix = QPixmap(ali_path)
        wx_pix = QPixmap(wx_path)
    
        size = 256
        ali_pix = ali_pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        wx_pix = wx_pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        padding = 0
        text_height = 0
        combined_width = size * 2 + padding * 3
        combined_height = size + text_height + padding
        
        combined_pixmap = QPixmap(combined_width, combined_height)
        combined_pixmap.fill(Qt.transparent)
        
        painter = QPainter(combined_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.drawPixmap(padding, padding, ali_pix)
        painter.drawPixmap(size + padding * 2, padding, wx_pix)
        
        painter.end()
        return combined_pixmap

    def welcome(self):
        class WelcomeMessageBox(MessageBoxBase):
            def __init__(self, parent=None, combined_pixmap=None):
                super().__init__(parent)
                self.setWindowTitle(self.tr("欢迎使用 PowerTools"))
                self.titleLabel = SubtitleLabel(self.tr("感谢支持开源项目"), self)
                self.contentLabel = CaptionLabel(
                    self.tr("如果您觉得 PowerTools 提升了您的效率，欢迎扫码支持作者，让项目持续更新！"), self
                )
                self.contentLabel.setWordWrap(True)  
                self.qrLabel = QLabel(self)
                if combined_pixmap.isNull():
                    self.qrLabel.setText(self.tr("（赞助码图片加载失败）"))
                else:
                    self.qrLabel.setPixmap(combined_pixmap.scaled(512, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.qrLabel.setAlignment(Qt.AlignCenter)

                # 将组件添加到对话框布局中
                self.viewLayout.addWidget(self.titleLabel)
                self.viewLayout.addWidget(self.contentLabel)
                self.viewLayout.addWidget(self.qrLabel)

                # 修改按钮文本
                self.yesButton.setText(self.tr("给个好评"))
                self.cancelButton.setText(self.tr("下次一定"))

                self.widget.setMinimumWidth(512)

        combined_pixmap = self._combine_qr_codes(":/powertools/images/alipay.jpg", ":/powertools/images/wechat.jpg")
        w = WelcomeMessageBox(self, combined_pixmap)
        if w.exec():
            url = QUrl("https://github.com/SmileSnail5470/PowerTools")
            QDesktopServices.openUrl(url)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, 'splashScreen'):
            self.splashScreen.resize(self.size())

    def closeEvent(self, e):
        cfg.save_config()
        self.themeListener.terminate()
        self.themeListener.deleteLater()
        from app.controllers.task_manager import global_task_manager
        global_task_manager.close()

        self.resource_monitor_widget.clear()
        super().closeEvent(e)

    def _onThemeChangedFinished(self):
        super()._onThemeChangedFinished()

        # retry
        if self.isMicaEffectEnabled():
            QTimer.singleShot(100, lambda: self.windowEffect.setMicaEffect(self.winId(), isDarkTheme()))