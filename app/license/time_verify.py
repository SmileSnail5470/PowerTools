import json
import logging
import os
import platform
import ssl
import time
import urllib.request
from typing import Optional, Tuple


logger = logging.getLogger("License")

TIME_VERIFY_URL = ""     # e.g., "https://your-worker.workers.dev/time"
TIME_VERIFY_TIMEOUT = 5  # seconds


def get_verified_time() -> Tuple[Optional[float], str]:
    if not TIME_VERIFY_URL:
        return None, "time_verify_not_configured"
    try:
        context = ssl.create_default_context()
        req = urllib.request.Request(TIME_VERIFY_URL, headers={"User-Agent": "PowerTools/1.0"})
        with urllib.request.urlopen(req, timeout=TIME_VERIFY_TIMEOUT, context=context) as resp:
            if resp.status != 200:
                return None, f"http_error_{resp.status}"
            data = json.loads(resp.read().decode("utf-8"))
            ts = data.get("ts")
            if ts and isinstance(ts, (int, float)):
                return float(ts), "online"
        return None, "invalid_response_format"
    except Exception as e:
        return None, f"error: {e}"


def verify_system_time(tolerance_seconds: int = 300) -> Tuple[bool, str]:
    verified_time, source = get_verified_time()
    if verified_time is not None:
        drift = abs(time.time() - verified_time)
        if drift > tolerance_seconds:
            logger.warning(f"Time drift detected: {drift:.0f}s (source: {source})")
            return False, f"online_drift_{drift:.0f}s"
        return True, "online_verified"
    return _filesystem_time_check(), "filesystem_check"


def _filesystem_time_check() -> bool:
    current_time = time.time()
    check_paths = []
    system = platform.system()
    if system == "Windows":
        temp = os.environ.get("TEMP", "")
        if temp:
            check_paths.append(temp)
        system_root = os.environ.get("SYSTEMROOT", r"C:\Windows")
        check_paths.append(os.path.join(system_root, "System32", "config", "SYSTEM"))
    elif system == "Darwin":
        check_paths.extend([
            "/var/log/system.log",
            "/tmp",
            "/private/var/db/.AppleSetupDone",
        ])
    else:
        check_paths.extend([
            "/var/log/syslog",
            "/var/log/auth.log",
            "/tmp",
        ])
    future_count = 0
    checked_count = 0
    for path in check_paths:
        if not path or not os.path.exists(path):
            continue
        try:
            mtime = os.path.getmtime(path)
            checked_count += 1
            if mtime > current_time + 86400:
                future_count += 1
        except OSError:
            continue
    if checked_count == 0:
        return True
    return future_count < (checked_count / 2)