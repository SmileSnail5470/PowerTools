import os
import sys
workdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(workdir)

os.environ["QT_LOGGING_RULES"] = "qt.multimedia.*=false"

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTranslator

from app.ui.common.config import cfg
from app.window import MainWindow
from app.ui.library.qfluentwidgets import FluentTranslator

from app.configs.config import UpdateFontFamilies
from app.utils.logger import init_logging


def main():
    log_level = cfg.get(cfg.logLevel)
    log_dir = os.path.join(cfg.get(cfg.cachePath), "logs")
    log_manager = init_logging(log_dir=log_dir, level=log_level)
    logger = log_manager.get_logger(name="UI")
    logger.info("Start PowerTools...")
    
    # Update font families before starting the application
    try:
        UpdateFontFamilies(cfg=cfg).run()
        logger.debug("Font families config update success")
    except Exception as e:
        logger.error(f"Update font families config failed: {e}", exc_info=True)

    # Start the application
    if cfg.get(cfg.dpiScale) != "Auto":
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"               # 禁止 qt UI 自动缩放
        os.environ["QT_SCALE_FACTOR"] = str(cfg.get(cfg.dpiScale))  # 手动设置缩放因子

    # create application
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)          # 禁止 Qt 在某些情况下自动创建原生控件的兄弟窗口
    
    def on_log_level_changed(value):
        log_manager.update_level(value)
        logger.info(f"Log level is set: {value}")

    cfg.logLevel.valueChanged.connect(on_log_level_changed)


    # internationalization
    locale = cfg.get(cfg.language).value
    translator = FluentTranslator(locale)
    powertoolsTranslator = QTranslator()
    powertoolsTranslator.load(locale, "powertools", ".", ":/powertools/i18n")

    app.installTranslator(translator)            # 安装主翻译器。安装后，应用中所有使用 tr() 或 QObject.tr() 的文本都会被翻译器自动替换成对应语言
    app.installTranslator(powertoolsTranslator)  # 安装 PowerTools 程序翻译器

    # create main window
    try:
        w = MainWindow()
        w.show()
        logger.info("Create main windows and show")
    except Exception as e:
        logger.critical(f"Create main windows failed: {e}", exc_info=True)
        sys.exit(1)

    try:
        app.exec()
    except Exception as e:
        logger.critical(f"PowerTools start failed: {e}", exc_info=True)
        raise
    finally:
        logger.info("PowerTools exit")


if __name__ == '__main__':
    main()