import json
import os
import sys
import pathlib
from enum import Enum

from PySide6.QtCore import QLocale
from app.ui.library.qfluentwidgets import (
    QConfig, ConfigItem, OptionsConfigItem, BoolValidator, OptionsValidator, RangeConfigItem, 
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
    softwareInvalidPath = os.path.join(pathlib.Path.home(), ".PowerTools")
    additionalParams = ConfigItem("AdditionalSettings", "additionalParams", {})

    # main window
    micaEnabled = ConfigItem("MainWindow", "MicaEnabled", isWin11(), BoolValidator())
    dpiScale = OptionsConfigItem("MainWindow", "DpiScale", "Auto", OptionsValidator([1, 1.25, 1.5, 1.75, 2, "Auto"]))

    # Material
    blurRadius  = RangeConfigItem("Material", "AcrylicBlurRadius", 15, RangeValidator(0, 40))

    # 通用设置
    autoStartup = ConfigItem("GeneralSettings", "AutoStartup", False, BoolValidator())
    autoUpdate = ConfigItem("GeneralSettings", "AutoUpdate", False, BoolValidator())
    cachePath = ConfigItem("GeneralSettings", "CachePath", os.path.join(pathlib.Path.home(), "PowerToolsCache"), FolderValidator())
    uiTheme = OptionsConfigItem("GeneralSettings", "UiTheme",  Theme.LIGHT.value, OptionsValidator([Theme.LIGHT.value, Theme.DARK.value]))
    language = OptionsConfigItem("GeneralSettings", "Language", Language.CHINESE_SIMPLIFIED, OptionsValidator(Language), LanguageSerializer())

    # 软件设置
    ffmpeg_path = ConfigItem("SoftwareSettings", "FFmpegPath", softwareInvalidPath, FolderValidator())

    # 本地AI设置
    default_deps_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "resources", "deps")
    localAIModelDeps = ConfigItem("LocalAISettings", "LocalAIModelDeps", default_deps_path, FolderValidator())
    localBlindWatermarkEnabled = ConfigItem("LocalAISettings", "LocalBlindWatermarkEnabled", False, BoolValidator())
    localWatermarkRemovalEnabled = ConfigItem("LocalAISettings", "LocalWatermarkRemovalEnabled", False, BoolValidator())
    localObjectSegmentationEnabled = ConfigItem("LocalAISettings", "LocalObjectSegmentationEnabled", False, BoolValidator())
    localOCREnabled = ConfigItem("LocalAISettings", "localOCREnabled", False, BoolValidator())

    # 高级设置
    logLevel = OptionsConfigItem("AdvancedSettings", "LogLevel", "INFO", OptionsValidator(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]))
    taskParallelNumber = OptionsConfigItem("AdvancedSettings", "TaskParallelNumber", 8, OptionsValidator([1, 2, 4, 8, 16]), restart=True)

    def __init__(self):
        super().__init__()
        self.params = self.toDict()
        self._config_file = os.path.join(pathlib.Path.home(), ".PowerTools", "settings.json")
        self._load_config()
        self._init_connect()
        
        os.environ["POWERTOOLS_FFMPEG_BIN"] = self.get(self.ffmpeg_path)
        self.ffmpeg_path.valueChanged.connect(update_ffmpeg_path)
        os.environ["POWERTOOLS_LOCAL_AI_MODEL_DEPS"] = self.get(self.localAIModelDeps)
        self.localAIModelDeps.valueChanged.connect(lambda path: os.environ.update({"POWERTOOLS_LOCAL_AI_MODEL_DEPS": path}))

    def _init_connect(self):
        for name in dir(self.__class__):
            item = getattr(self.__class__, name)
            if isinstance(item, ConfigItem):
                item.valueChanged.connect(self._update_config)

    def _load_config(self):
        if os.path.exists(self._config_file):
            self.load(file=self._config_file)
            self.params = self.toDict()

    def _update_config(self, value):
        self.params.update(self.toDict())

    def save_config(self):
        os.makedirs(os.path.dirname(self._config_file), exist_ok=True)
        with open(self._config_file, "w") as fp:
            fp.write(json.dumps(self.params))

    def get_local_settings(self):
        if not os.path.exists(self._config_file):
            return {}
        with open(self._config_file, "r") as fp:
            data = fp.read()
        return json.loads(data)


cfg = Config()
cfg.themeMode.value = Theme.LIGHT