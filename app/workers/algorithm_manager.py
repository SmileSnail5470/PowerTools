import importlib
import json
import os
import subprocess
import sys
import yaml
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.algorithms import ALGORITHMS_ROOT


@dataclass(frozen=True)
class AlgorithmDescriptor:
    name: str
    version: str
    base_path: Path
    manifest: Dict[str, Any]

    def supports(self, capability: str) -> bool:
        capabilities: Iterable[str] = self.manifest.get("capabilities", [])
        return capability in capabilities

    def get_python_method_metadata(self, capability: str) -> Dict[str, Any]:
        entry_points = self.manifest.get("entry_points", {})
        if capability not in entry_points:
            raise ValueError(f"Algorithm '{self.name}' does not support capability '{capability}'")
        entry_point = entry_points[capability]
        return entry_point

    def create_instance(self, capability: str) -> Any:
        entry_points = self.manifest.get("entry_points", {})
        if capability not in entry_points:
            raise ValueError(f"Algorithm '{self.name}' does not support capability '{capability}'")
        entry_point = entry_points[capability]
        runtime = entry_point["runtime"]
        if runtime == "python":
            return self._create_python_instance(entry_point=entry_point)
        if runtime == "ctypes":
            return self._create_ctypes_handle(entry_point=entry_point)
        if runtime == "subprocess":
            return self._create_subprocess_adapter(entry_point=entry_point)
        raise ValueError(f"Unsupported runtime '{runtime}' for algorithm '{self.name}'")

    def _create_python_instance(self, entry_point: dict) -> Any:
        module_path, attr = entry_point["module"], entry_point["attr"]
        module = importlib.import_module(module_path)
        target = getattr(module, attr)
        init_kwargs = entry_point.get("init_kwargs", {})
        return target(**init_kwargs) if callable(target) else target

    def _create_ctypes_handle(self, entry_point: dict) -> Any:
        import ctypes

        binary_rel = entry_point["binary"][sys.platform.lower()]
        binary_path = (self.base_path / binary_rel).resolve()
        if not binary_path.exists():
            raise FileNotFoundError(f"Binary for '{self.name}' not found: {binary_path}")
        return ctypes.CDLL(str(binary_path))

    def _create_subprocess_adapter(self, entry_point: dict) -> "SubprocessAlgorithmAdapter":
        command = entry_point.get("command")
        if not command:
            raise ValueError(f"Algorithm '{self.name}' must declare 'command' for subprocess runtime")
        cwd = str((self.base_path / entry_point.get("working_dir", ".")).resolve())
        env = os.environ.copy()
        env.update(entry_point.get("env", {}))
        return SubprocessAlgorithmAdapter(command=command, cwd=cwd, env=env)


class SubprocessAlgorithmAdapter:
    def __init__(self, command: Any, cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None):
        self.command = command
        self.cwd = cwd
        self.env = env

    def __call__(self, payload: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        if sys.platform != "win32":
            completed = subprocess.run(  # noqa: S603
                self.command,
                input=data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                env=self.env,
                timeout=timeout,
                check=True,
            )
        else:
            completed = subprocess.run(  # noqa: S603
                self.command,
                input=data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                env=self.env,
                timeout=timeout,
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        stdout = completed.stdout.decode("utf-8").strip()
        return json.loads(stdout) if stdout else {}


class AlgorithmManager:
    MANIFEST_FILES = ("manifest.json", "manifest.yml")

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._lock = threading.RLock()
        self._registry: Dict[str, Dict[str, AlgorithmDescriptor]] = {}
        self.reload()

    def reload(self) -> None:
        with self._lock:
            self._registry.clear()
            for child in self.base_dir.iterdir():
                manifest_path = self._resolve_manifest(child)
                if not manifest_path:
                    continue
                manifest = self._load_manifest(manifest_path)
                if not manifest.get("enabled", True):
                    continue
                descriptor = self._build_descriptor(manifest, manifest_path)
                versions = self._registry.setdefault(descriptor.name, {})
                versions[descriptor.version] = descriptor

    def list_algorithms(self) -> Dict[str, List[str]]:
        with self._lock:
            return {name: sorted(descs.keys()) for name, descs in self._registry.items()}

    def get_descriptor(self, name: str, version: Optional[str] = None) -> AlgorithmDescriptor:
        with self._lock:
            if name not in self._registry:
                raise KeyError(f"Algorithm '{name}' is not registered")
            versions = self._registry[name]
            if version:
                try:
                    return versions[version]
                except KeyError as exc:
                    raise KeyError(f"Algorithm '{name}' version '{version}' not found") from exc
            latest_version = sorted(versions.keys())[-1]
            return versions[latest_version]

    def create_instance(self, name: str, capability: str, version: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        descriptor = self.get_descriptor(name, version)
        return descriptor.create_instance(capability), descriptor.get_python_method_metadata(capability)

    def _resolve_manifest(self, directory: Path) -> Optional[Path]:
        if not directory.is_dir():
            return None
        for manifest_name in self.MANIFEST_FILES:
            manifest_path = directory / manifest_name
            if manifest_path.exists():
                return manifest_path
        return None

    def _load_manifest(self, manifest_path: Path) -> Dict[str, Any]:
        suffix = manifest_path.suffix.lower()
        text = manifest_path.read_text(encoding="utf-8")
        if suffix == ".json":
            return json.loads(text)
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to parse YAML manifests")
            return yaml.safe_load(text)
        raise ValueError(f"Unsupported manifest type: {manifest_path}")

    def _build_descriptor(self, manifest: Dict[str, Any], manifest_path: Path) -> AlgorithmDescriptor:
        name = manifest.get("name")
        version = manifest.get("version", "0.0.0")
        entry_points = manifest.get("entry_points")
        if not name or not entry_points:
            raise ValueError(f"Invalid manifest at {manifest_path}: missing 'name' or 'entry_points'")
        self._ensure_python_path(manifest)
        base_path = manifest_path.parent
        return AlgorithmDescriptor(
            name=name,
            version=str(version),
            base_path=base_path,
            manifest=manifest,
        )
    
    def _ensure_python_path(self, manifest: Dict[str, Any]) -> None:
        config_paths = manifest.get("python_paths", [])
        for p in config_paths:
            # 展开环境变量，例如 ${HOME}、${APP_PATH}
            p = os.path.expandvars(p)

            # 转为 Path 对象
            path_obj = Path(p)

            if not path_obj.is_absolute():
                path_obj = (Path(__file__).parent / path_obj).resolve()

            if os.path.exists(path_obj) and path_obj.is_dir():
                if path_obj not in sys.path:
                    sys.path.append(path_obj)

global_algorithm_manager = AlgorithmManager(ALGORITHMS_ROOT)