import hashlib
import platform
import subprocess
import uuid
if platform.system().lower() == "windows":
    import winreg


def _get_windows_machine_guid() -> str:
    try:
        reg_path = r"SOFTWARE\Microsoft\Cryptography"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
            value, _ = winreg.QueryValueEx(key, "MachineGuid")
            return value
    except Exception:
        return ""


def _get_mac_serial_number() -> str:
    try:
        result = subprocess.run(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "IOPlatformSerialNumber" in line:
                return line.split("=")[-1].strip().strip('"')
    except Exception:
        pass
    return ""


def _get_linux_machine_id() -> str:
    for path in ["/etc/machine-id", "/var/lib/dbus/machine-id"]:
        try:
            with open(path, "r") as f:
                return f.read().strip()
        except Exception:
            continue
    return ""


def _get_platform_specific_id() -> str:
    system = platform.system().lower()
    if system == "windows":
        return _get_windows_machine_guid()
    elif system == "darwin":
        return _get_mac_serial_number()
    elif system == "linux":
        return _get_linux_machine_id()
    return ""


def get_machine_id() -> str:
    components = []
    platform_id = _get_platform_specific_id()
    if platform_id:
        components.append(platform_id)
    mac = str(uuid.getnode())
    components.append(mac)
    components.append(platform.machine())
    components.append(platform.node())
    components.append(platform.processor())
    raw = "-".join(components)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def get_machine_id_display() -> str:
    """
    Get machine ID formatted for user display (grouped for readability).
    
    Returns:
        Machine ID in format: XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX
    """
    mid = get_machine_id()
    return "-".join(mid[i:i+4] for i in range(0, 32, 4))
