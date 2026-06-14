import base64
import json
import logging
import os
import pathlib
import shutil
import time
import app.library._license_checker as license_checker
from datetime import datetime
from typing import Optional
from app.license.exceptions import (
    LicenseNotFoundError,
    LicenseInvalidSignatureError,
    LicenseExpiredError,
    MachineMismatchError,
    LicenseCorruptedError,
    TimeRollbackDetectedError,
)
from app.license.machine_id import get_machine_id
from app.license.time_verify import verify_system_time


logger = logging.getLogger("License")


PUBLIC_KEY_B64 = "AXgCTTXpc0PPNs2pPdaliBV8fnDzhtuoMs+cnZNIiUo="


class LicenseData:
    def __init__(self, data: dict):
        self.license_id: str = data.get("license_id", "")
        self.machine_id: str = data.get("machine_id", "")
        self.issued_at: str = data.get("issued_at", "")
        self.expires_at: str = data.get("expires_at", "")
        self.features: list = data.get("features", [])
        self.tier: str = data.get("tier", "free")
        self.max_activations: int = data.get("max_activations", 1)
        self._raw = data

    @property
    def is_pro(self) -> bool:
        return self.tier == "pro"

    @property
    def expires_timestamp(self) -> float:
        try:
            dt = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, AttributeError):
            return 0.0

    @property
    def issued_timestamp(self) -> float:
        try:
            dt = datetime.fromisoformat(self.issued_at.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, AttributeError):
            return 0.0

    @property
    def days_remaining(self) -> int:
        remaining = self.expires_timestamp - time.time()
        return max(0, int(remaining / 86400))

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_timestamp

    def has_feature(self, feature: str) -> bool:
        return feature in self.features or "all" in self.features

    def to_dict(self) -> dict:
        return self._raw.copy()


class LicenseManager:
    def __init__(self, license_dir: Optional[str] = None):
        if license_dir is None:
            license_dir = os.path.join(pathlib.Path.home(), ".PowerTools", "license")
        self._license_dir = license_dir
        self._license_file = os.path.join(self._license_dir, "license.lic")
        self._state_file = os.path.join(self._license_dir, ".state")
        self._license_data: Optional[LicenseData] = None
        self._is_valid = False
        self._error_message = ""

        os.makedirs(self._license_dir, exist_ok=True)
        self._try_load_license()

    @property
    def is_licensed(self) -> bool:
        return self._is_valid

    @property
    def license_data(self) -> Optional[LicenseData]:
        return self._license_data if self._is_valid else None

    @property
    def error_message(self) -> str:
        return self._error_message

    @property
    def tier(self) -> str:
        if self._is_valid and self._license_data:
            return self._license_data.tier
        return "free"

    @property
    def days_remaining(self) -> int:
        if self._is_valid and self._license_data:
            return self._license_data.days_remaining
        return 0

    def activate(self, license_file_path: str) -> bool:
        try:
            license_data = self._load_and_verify(license_file_path)
            shutil.copy2(license_file_path, self._license_file)
            self._license_data = license_data
            self._is_valid = True
            self._error_message = ""
            self._update_state()
            logger.info(f"License activated: tier={license_data.tier}, "
                       f"expires={license_data.expires_at}, "
                       f"features={license_data.features}")
            return True
        except Exception as e:
            self._is_valid = False
            self._error_message = str(e)
            logger.warning(f"License activation failed: {e}")
            raise

    def deactivate(self):
        self._is_valid = False
        self._license_data = None
        self._error_message = ""
        if os.path.exists(self._license_file):
            os.remove(self._license_file)
        if os.path.exists(self._state_file):
            os.remove(self._state_file)
        logger.info("License deactivated")

    def has_feature(self, feature: str) -> bool:
        if not self._is_valid or not self._license_data:
            return False
        return self._license_data.has_feature(feature)

    def _try_load_license(self):
        if not os.path.exists(self._license_file):
            self._is_valid = False
            self._error_message = "未找到许可证文件"
            return
        try:
            license_data = self._load_and_verify(self._license_file)
            self._license_data = license_data
            self._is_valid = True
            self._error_message = ""
            self._check_time_integrity()
            self._update_state()
        except LicenseExpiredError:
            self._is_valid = False
            self._error_message = "许可证已过期"
        except MachineMismatchError:
            self._is_valid = False
            self._error_message = "许可证与当前设备不匹配"
        except TimeRollbackDetectedError:
            self._is_valid = False
            self._error_message = "检测到系统时间异常"
        except Exception as e:
            self._is_valid = False
            self._error_message = f"许可证验证失败: {e}"

    def _load_and_verify(self, lic_path: str) -> LicenseData:
        if not os.path.exists(lic_path):
            raise LicenseNotFoundError(f"许可证文件不存在: {lic_path}")
        try:
            with open(lic_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise LicenseCorruptedError(f"许可证文件格式错误: {e}")
        current_machine_id = get_machine_id()
        validate_result = license_checker.validate_license(license_json=json.dumps(raw_data), machine_id=current_machine_id)
        if not validate_result["valid"]:
            if validate_result["error"] == "runtime_environment_invalid":
                msg = "运行环境有风险，软件被尝试破解"
            elif validate_result["error"] in ["missing_signature", "invalid_signature", "invalid_expiry_format"]:
                msg = "许可证签名无效，请联系作者"
            elif validate_result["error"] == "license_expired":
                msg = "许可证过期，联系作者续费"
            elif validate_result["error"] == "machine_mismatch":
                msg = "机器环境不匹配，软件只支持一机一许可证"
            else:
                msg = "许可证验证失败，请联系作者"
            raise LicenseInvalidSignatureError(msg)
        license_data = LicenseData(raw_data)
        if license_data.is_expired:
            raise LicenseExpiredError(f"许可证已于 {license_data.expires_at} 过期")
        if license_data.machine_id != current_machine_id:
            raise MachineMismatchError(f"许可证绑定的设备ID与当前设备不匹配")
        license_data = LicenseData(raw_data)
        return license_data

    def _check_time_integrity(self):
        is_valid, check_source = verify_system_time()
        if not is_valid:
            raise TimeRollbackDetectedError(f"系统时间异常，请检查系统时间设置。检测来源: {check_source}")
        state = self._load_state()
        if not state:
            return
        last_seen = state.get("last_seen", 0)
        current_time = time.time()
        if current_time < last_seen - 300:
            logger.warning(f"Time rollback detected: current={current_time}, last_seen={last_seen}")
            raise TimeRollbackDetectedError(f"系统时间异常：当前时间早于上次记录时间，请检查系统时间设置")

    def _load_state(self) -> dict:
        if not os.path.exists(self._state_file):
            return {}
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception:
            return {}

    def _update_state(self):
        state = self._load_state()
        current_time = time.time()
        state["last_seen"] = current_time
        state["launch_count"] = state.get("launch_count", 0) + 1
        if "session_start" in state:
            elapsed = current_time - state["session_start"]
            if 0 < elapsed < 86400:
                state["total_runtime_seconds"] = state.get("total_runtime_seconds", 0) + elapsed
        state["session_start"] = current_time
        try:
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to update license state: {e}")