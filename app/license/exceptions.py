"""License system custom exceptions."""


class LicenseError(Exception):
    """Base exception for license errors."""
    pass


class LicenseNotFoundError(LicenseError):
    """License file not found."""
    pass


class LicenseInvalidSignatureError(LicenseError):
    """License signature verification failed."""
    pass


class LicenseExpiredError(LicenseError):
    """License has expired."""
    pass


class MachineMismatchError(LicenseError):
    """Machine ID does not match the license."""
    pass


class LicenseCorruptedError(LicenseError):
    """License file is corrupted or malformed."""
    pass


class FeatureNotLicensedError(LicenseError):
    """The requested feature is not included in the current license tier."""
    pass


class TimeRollbackDetectedError(LicenseError):
    """System time rollback detected, possible tampering."""
    pass


class LicenseStateCorruptedError(LicenseError):
    """License state file is corrupted."""
    pass
