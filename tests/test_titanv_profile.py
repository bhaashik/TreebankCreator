"""CPU-only regression coverage for the explicit Volta/Titan V install profile."""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "scripts" / "setup_titanv.sh"
CHECKER = ROOT / "scripts" / "check_titanv.py"


def _load_checker():
    spec = spec_from_file_location("titanv_profile_checker", CHECKER)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MockGPU:
    def __init__(self, name="NVIDIA TITAN V", arch=(7, 0), cuda_available=True, arch_list=None):
        self.name, self.arch, self.available = name, arch, cuda_available
        self.arch_list = arch_list or ["sm_70", "sm_75"]
        self.synchronized = False

    def is_available(self): return self.available
    def device_count(self): return 1
    def get_device_name(self, i): return self.name
    def get_device_capability(self, i): return self.arch
    def get_arch_list(self): return self.arch_list
    def synchronize(self, i): self.synchronized = True


def _torch(cuda=None, version="2.14.1+cu126", cuda_ver="12.6"):
    gpu = cuda or MockGPU()
    return SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda=cuda_ver),
        cuda=gpu,
        ones=lambda shape, device: SimpleNamespace(__matmul__=lambda other: None),
    )


def test_titanv_checker_accepts_pinned_build_and_gpu():
    check = _load_checker()
    torch = _torch()
    report = check.inspect_titanv(torch, run_kernel=False)
    assert report["ok"]
    assert report["capability"] == [7, 0]
    assert report["gpu"] == "NVIDIA TITAN V"


def test_titanv_checker_rejects_cuda13_wheel():
    check = _load_checker()
    try:
        check.inspect_titanv(_torch(cuda_ver="13.0"), run_kernel=False)
    except RuntimeError as exc:
        assert "12.6" in str(exc)
    else:
        raise AssertionError("Invalid CUDA build unexpectedly accepted")


def test_titanv_checker_rejects_wrong_gpu_and_missing_sm70():
    check = _load_checker()
    for gpu in (MockGPU(name="NVIDIA RTX 3090", arch=(8, 6)),
                MockGPU(arch_list=["sm_75", "sm_80"])):
        try:
            check.inspect_titanv(_torch(cuda=gpu), run_kernel=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Incompatible GPU/build unexpectedly accepted")


def test_titanv_dry_run_is_read_only_and_displays_cuda126(tmp_path):
    venv = tmp_path / "titanv-profile"
    run = subprocess.run(["bash", str(PROFILE), "--dry-run", "--venv", str(venv),
                          "--with-awesome", "--with-tests"], capture_output=True,
                         text=True, timeout=15)
    assert run.returncode == 0, run.stderr
    assert "torch==2.14.1" in run.stdout
    assert "cu126" in run.stdout
    assert "constraints" in run.stdout
    assert "AWESOME-align" in run.stdout
    assert not venv.exists()


def test_titanv_check_does_not_create_venv(tmp_path):
    venv = tmp_path / "does-not-exist"
    run = subprocess.run(["bash", str(PROFILE), "--check", "--venv", str(venv)],
                         capture_output=True, text=True, timeout=15)
    assert run.returncode != 0
    assert "Missing environment" in run.stderr
    assert not venv.exists()


def test_titanv_profile_bash_syntax_and_activation_guard():
    for filename in ("setup_titanv.sh", "activate_titanv.sh"):
        path = ROOT / "scripts" / filename
        assert subprocess.run(["bash", "-n", str(path)], check=False).returncode == 0
    # Activation must be sourced, not executed.
    process = subprocess.run(["bash", str(ROOT / "scripts" / "activate_titanv.sh")],
                             capture_output=True, text=True, timeout=15)
    assert process.returncode == 2
    assert "Source" in process.stderr


def test_titanv_constraint_pins_public_torch_version():
    content = (ROOT / "constraints" / "titanv.txt").read_text()
    assert "torch==2.14.1" in content
    assert "torch>=2" not in content
