from PySide6.QtCore import QUrl, QSize, QTimer
from PySide6.QtGui import QIcon, QDesktopServices
from PySide6.QtWidgets import QApplication

from app.ui.library.qfluentwidgets import (NavigationItemPosition, FluentWindow, SplashScreen, SystemThemeListener, isDarkTheme)
from app.ui.library.qfluentwidgets import FluentIcon as FIF

from app.ui.view.home import Home
from app.ui.view.settings import Settings
from app.ui.view.image_edit import ImageEdit
from app.ui.view.ocr import OCR
from app.ui.view.screenshot import Screenshot
from app.ui.view.scroll_screenshot import ScrollScreenshot
from app.ui.view.watermark_add import WatermarkAdd
from app.ui.view.watermark_remove import WatermarkRemove

from app.ui.common.config import cfg
from app.ui.common.icon import Icon
from app.ui.common.signal_bus import signalBus
from app.ui.resources import resource


class MainWindow(FluentWindow):

    def __init__(self):
        super().__init__()
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

        self.connectSignalToSlot()

        # add items to navigation interface
        self.initNavigation()
        self.splashScreen.finish()

        # start theme listener
        self.themeListener.start()

    def connectSignalToSlot(self):
        signalBus.micaEnableChanged.connect(self.setMicaEffectEnabled)
        signalBus.switchToSampleCard.connect(self.switchToSample)
        signalBus.supportSignal.connect(self.onSupport)

    def initNavigation(self):
        # add navigation items
        self.addSubInterface(self.homeInterface, FIF.HOME, self.tr('主页'))
        self.addSubInterface(self.settingInterface, FIF.SETTING, self.tr("常规"))
        self.navigationInterface.addSeparator()

        pos = NavigationItemPosition.SCROLL
        self.addSubInterface(self.watermarkAddInterface, Icon.WATERMARK_ADD, self.tr("水印添加"), pos, parent=None)
        self.addSubInterface(self.watermarkRemoveInterface, Icon.WATERMARK_REMOVE, self.tr("水印移除"), pos, parent=None)

        self.addSubInterface(self.screenshotInterface, Icon.SCREENSHOT, self.tr("屏幕截图"), pos, parent=None)
        self.addSubInterface(self.scrollScreenshotInterface, Icon.LONG_SCREENSHOT, self.tr("滚动截图"), pos, parent=None)

        self.addSubInterface(self.OCRInterface, Icon.OCR, self.tr("文字提取"), pos, parent=None)

        self.addSubInterface(self.imageEditInterface, Icon.IMAGE_EDIT, self.tr("图像编辑"), pos, parent=None)

        # add custom widget to bottom
        self.navigationInterface.addItem(
            routeKey='use-powertools',
            icon=Icon.Price,
            text=self.tr("欢迎使用 PowerTools"),
            onClick=self.onSupport,
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
        self.resize(1220, 780)
        self.setMinimumWidth(960)
        self.setWindowIcon(QIcon(':/powertools/images/logo.png'))
        self.setWindowTitle(self.tr("PowerTools"))

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
        language = cfg.get(cfg.language).value
        if language.name() == "zh_CN":
            QDesktopServices.openUrl(QUrl(ZH_SUPPORT_URL))
        else:
            QDesktopServices.openUrl(QUrl(EN_SUPPORT_URL))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, 'splashScreen'):
            self.splashScreen.resize(self.size())

    def closeEvent(self, e):
        self.themeListener.terminate()
        self.themeListener.deleteLater()
        super().closeEvent(e)

    def _onThemeChangedFinished(self):
        super()._onThemeChangedFinished()

        # retry
        if self.isMicaEffectEnabled():
            QTimer.singleShot(100, lambda: self.windowEffect.setMicaEffect(self.winId(), isDarkTheme()))

    def switchToSample(self, routeKey, index):
        """ switch to sample """
        interfaces = self.findChildren(GalleryInterface)
        for w in interfaces:
            if w.objectName() == routeKey:
                self.stackedWidget.setCurrentWidget(w, False)
                w.scrollToCard(index)