from app.license.license_manager import LicenseManager
from app.license.feature_gate import FeatureGate


license_manager = LicenseManager()
feature_gate = FeatureGate(license_manager)
