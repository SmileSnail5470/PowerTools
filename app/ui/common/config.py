import os
import sys
import pathlib
from enum import Enum

from PySide6.QtCore import QLocale
from app.ui.library.qfluentwidgets import (
    qconfig, QConfig, ConfigItem, OptionsConfigItem, BoolValidator, OptionsValidator, RangeConfigItem, 
    RangeValidator, Theme, ConfigSerializer, FolderValidator
)


class Language(Enum):
    """ Language enumeration """

    CHINESE_SIMPLIFIED = QLocale(QLocale.Chinese, QLocale.China)
    ENGLISH = QLocale(QLocale.English)
    AUTO = QLocale()


class LanguageSerializer(ConfigSerializer):
    """ Language serializer """

    def serialize(self, language):
        return language.value.name() if language != Language.AUTO else "Auto"

    def deserialize(self, value: str):
        return Language(QLocale(value)) if value != "Auto" else Language.AUTO


def isWin11():
    return sys.platform == 'win32' and sys.getwindowsversion().build >= 22000


def update_ffmpeg_path(path: str):
    if not path:
        return
    os.environ["POWERTOOLS_FFMPEG_BIN"] = path


class Config(QConfig):
    """ Config of application """
    # main window
    micaEnabled = ConfigItem("MainWindow", "MicaEnabled", isWin11(), BoolValidator())
    dpiScale = OptionsConfigItem("MainWindow", "DpiScale", "Auto", OptionsValidator([1, 1.25, 1.5, 1.75, 2, "Auto"]), restart=True)

    # Material
    blurRadius  = RangeConfigItem("Material", "AcrylicBlurRadius", 15, RangeValidator(0, 40))

    # 通用设置
    autoStartup = ConfigItem("GeneralSettings", "AutoStartup", False, BoolValidator())
    autoUpdate = ConfigItem("GeneralSettings", "AutoUpdate", False, BoolValidator())
    cachePath = ConfigItem("GeneralSettings", "CachePath", os.path.join(pathlib.Path.home(), ".powertools"), FolderValidator())
    uiTheme = OptionsConfigItem("GeneralSettings", "UiTheme",  Theme.LIGHT, OptionsValidator([ Theme.LIGHT, Theme.DARK]))
    language = OptionsConfigItem("GeneralSettings", "Language", Language.AUTO, OptionsValidator(Language), LanguageSerializer(), restart=True)

    # 软件设置
    ffmpeg_path = ConfigItem("SoftwareSettings", "FFmpegPath", "", FolderValidator())
    ffmpeg_path.valueChanged.connect(update_ffmpeg_path)

    # 本地AI设置
    localAIModelDeps = ConfigItem("LocalAISettings", "LocalAIModelDeps", os.path.join(pathlib.Path.home(), ".powertools", "local_ai_models"), FolderValidator())
    localBlindWatermarkEnabled = ConfigItem("LocalAISettings", "LocalBlindWatermarkEnabled", False, BoolValidator())
    localWatermarkRemovalEnabled = ConfigItem("LocalAISettings", "LocalWatermarkRemovalEnabled", False, BoolValidator())

    # 高级设置
    logLevel = OptionsConfigItem("AdvancedSettings", "LogLevel", "WARNING", OptionsValidator(["DEBUG", "INFO", "WARNING", "ERROR"]))


cfg = Config()
cfg.themeMode.value = Theme.LIGHT
qconfig.load('app/config/config.json', cfg)