import datetime
from enum import Enum
from typing import Optional
from app.license.license_manager import LicenseManager
from app.license.exceptions import FeatureNotLicensedError


class FeatureTier(Enum):
    FREE = "free"
    PRO = "pro"


class Feature:
    def __init__(self, name: str, required_tier: str, description: str, free_limit: int):
        self.name = name
        self.required_tier = required_tier 
        self.description = description
        self.free_limit = free_limit


class FeatureGate:
    def __init__(self, license_manager: LicenseManager):
        self._license_manager = license_manager
        self._daily_usage: dict = {}  # {feature_name: count}
        self._last_reset_day: Optional[str] = None
        self._features: dict[str, Feature] = {}
        if self._license_manager.is_licensed and self._license_manager.tier == "free":
            for one_feature in self._license_manager.license_data.features:
                feature_name = one_feature[0]
                required_tier = one_feature[1]
                description = one_feature[2]
                free_limit = one_feature[3]
                self._features[feature_name] = Feature(feature_name.rsplit("_")[-1], required_tier, description, free_limit)

    @property
    def current_tier(self) -> FeatureTier:
        if self._license_manager.is_licensed:
            tier_str = self._license_manager.tier
            try:
                return FeatureTier(tier_str)
            except ValueError:
                return FeatureTier.FREE
        return FeatureTier.FREE

    @property
    def is_pro(self) -> bool:
        return self.current_tier == FeatureTier.PRO

    def can_use(self, feature_name: str) -> bool:
        if self.is_pro:
            return True
        
        feature = self._features[feature_name]
        if feature.required_tier == "pro":
            return False
        if feature.free_limit == -1:
            return True
        if feature.free_limit == 0:
            return False
        self._maybe_reset_daily_usage()
        current_usage = self._daily_usage.get(feature.name, 0)
        return current_usage < feature.free_limit

    def use_feature(self, feature_name: str) -> bool:
        feature = self._features[feature_name]
        if not self.can_use(feature_name):
            raise FeatureNotLicensedError(f"功能「{feature.description}」需要 Pro 授权才能使用")
        if not self.is_pro and feature.free_limit > 0:
            self._maybe_reset_daily_usage()
            self._daily_usage[feature.name] = self._daily_usage.get(feature.name, 0) + 1

    def get_remaining_uses(self, feature_name: str) -> int:
        if self.is_pro:
            return -1

        feature = self._features[feature_name]
        if feature.required_tier == "pro":
            return 0

        if feature.free_limit == -1:
            return -1
        self._maybe_reset_daily_usage()
        used = self._daily_usage.get(feature.name, 0)
        return max(0, feature.free_limit - used)

    def get_feature_status(self, feature_name: str) -> dict:
        feature = self._features[feature_name]
        return {
            "available": self.can_use(feature_name),
            "tier_required": feature.required_tier,
            "remaining": self.get_remaining_uses(feature_name),
            "description": feature.description,
            "is_pro_feature": feature.required_tier == "pro",
        }

    def _maybe_reset_daily_usage(self):
        """每天重置一次使用次数."""
        today = datetime.date.today().isoformat()
        if self._last_reset_day != today:
            self._daily_usage = {}
            self._last_reset_day = today
