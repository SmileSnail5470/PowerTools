import os
import sys
workdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(workdir)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTranslator

from app.ui.common.config import cfg
from window import MainWindow
from app.ui.library.qfluentwidgets import FluentTranslator

from core.config import UpdateFontFamilies


def main():
    # Update font families before starting the application
    UpdateFontFamilies(cfg=cfg).run()

    # Start the application
    if cfg.get(cfg.dpiScale) != "Auto":
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"               # 禁止 qt UI 自动缩放
        os.environ["QT_SCALE_FACTOR"] = str(cfg.get(cfg.dpiScale))  # 手动设置缩放因子

    # create application
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)          # 禁止 Qt 在某些情况下自动创建原生控件的兄弟窗口


    # internationalization
    locale = cfg.get(cfg.language).value
    translator = FluentTranslator(locale)
    powertoolsTranslator = QTranslator()
    powertoolsTranslator.load(locale, "powertools", ".", ":/powertools/i18n")

    app.installTranslator(translator)            # 安装主翻译器。安装后，应用中所有使用 tr() 或 QObject.tr() 的文本都会被翻译器自动替换成对应语言
    app.installTranslator(powertoolsTranslator)  # 安装 PowerTools 程序翻译器

    # create main window
    w = MainWindow()
    w.show()

    app.exec()


if __name__ == '__main__':
    main()