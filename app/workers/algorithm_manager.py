import importlib
import json
import os
import subprocess
import sys
import yaml
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.algorithms import ALGORITHMS_ROOT


@dataclass(frozen=True)
class AlgorithmDescriptor:
    name: str
    version: str
    runtime: str
    entry_point: str
    base_path: Path
    manifest: Dict[str, Any]

    def supports(self, capability: str) -> bool:
        capabilities: Iterable[str] = self.manifest.get("capabilities", [])
        return capability in capabilities

    def create_instance(self) -> Any:
        runtime = (self.runtime or "python").lower().strip()
        if runtime == "python":
            return self._create_python_instance()
        if runtime == "ctypes":
            return self._create_ctypes_handle()
        if runtime == "subprocess":
            return self._create_subprocess_adapter()
        raise ValueError(f"Unsupported runtime '{self.runtime}' for algorithm '{self.name}'")

    def _create_python_instance(self) -> Any:
        module_path, attr = self._split_entry_point()
        self._ensure_python_path()
        module = importlib.import_module(module_path)
        target = getattr(module, attr)
        init_kwargs = self.manifest.get("init_kwargs", {})
        return target(**init_kwargs) if callable(target) else target

    def _create_ctypes_handle(self) -> Any:
        import ctypes

        binary_rel = self.manifest.get("binary") or self.manifest.get("artifacts", {}).get("binary")
        if not binary_rel:
            raise ValueError(f"Algorithm '{self.name}' misses 'binary' definition for ctypes runtime")
        binary_path = (self.base_path / binary_rel).resolve()
        if not binary_path.exists():
            raise FileNotFoundError(f"Binary for '{self.name}' not found: {binary_path}")
        return ctypes.CDLL(str(binary_path))

    def _create_subprocess_adapter(self) -> "SubprocessAlgorithmAdapter":
        command = self.manifest.get("command")
        if not command:
            raise ValueError(f"Algorithm '{self.name}' must declare 'command' for subprocess runtime")
        cwd = str((self.base_path / self.manifest.get("working_dir", ".")).resolve())
        env = os.environ.copy()
        env.update(self.manifest.get("env", {}))
        return SubprocessAlgorithmAdapter(command=command, cwd=cwd, env=env)

    def _split_entry_point(self) -> List[str]:
        if ":" not in self.entry_point:
            raise ValueError(
                f"Entry point '{self.entry_point}' for algorithm '{self.name}' must be 'module:attr' format"
            )
        return self.entry_point.split(":", 1)

    def _ensure_python_path(self) -> None:
        extra_paths = self.manifest.get("python_paths") or []
        for rel_path in extra_paths:
            abs_path = str((self.base_path / rel_path).resolve())
            if abs_path not in sys.path:
                sys.path.append(abs_path)


class SubprocessAlgorithmAdapter:
    def __init__(self, command: Any, cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None):
        self.command = command
        self.cwd = cwd
        self.env = env

    def __call__(self, payload: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
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
        stdout = completed.stdout.decode("utf-8").strip()
        return json.loads(stdout) if stdout else {}


class AlgorithmManager:
    MANIFEST_FILES = ("manifest.json", "manifest.yaml", "manifest.yml")

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

    def create_instance(self, name: str, version: Optional[str] = None) -> Any:
        descriptor = self.get_descriptor(name, version)
        return descriptor.create_instance()

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
        runtime = manifest.get("runtime", "python")
        entry_point = manifest.get("entry_point")
        if not name or not entry_point:
            raise ValueError(f"Invalid manifest at {manifest_path}: missing 'name' or 'entry_point'")
        base_path = manifest_path.parent
        return AlgorithmDescriptor(
            name=name,
            version=str(version),
            runtime=str(runtime),
            entry_point=str(entry_point),
            base_path=base_path,
            manifest=manifest,
        )

global_algorithm_manager = AlgorithmManager(ALGORITHMS_ROOT)