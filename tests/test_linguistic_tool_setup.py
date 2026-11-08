"""Installer is explicit; check/dry-run must never install anything."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install_linguistic_tools.sh"


def run(*args: str):
    return subprocess.run(["bash", str(SCRIPT), *args], check=False,
                          text=True, capture_output=True, timeout=15)


def test_setup_help_and_no_argument_run_are_safe():
    help_ = run("--help")
    assert help_.returncode == 0
    assert "--apertium-package" in help_.stdout
    none = run()
    assert none.returncode == 0
    assert "aligner: none" in none.stdout


def test_dry_run_does_not_modify_python_or_apt_environment(tmp_path):
    # No analyzer exists and SimAlign may or may not be present in the CI runner.
    missing = tmp_path / "never-created.automorf.bin"
    result = run("--aligner", "simalign", "--apertium", "--analyzer-bin",
                 str(missing), "--dry-run", "--python", sys.executable)
    assert result.returncode == 0, result.stderr
    assert "MISSING: compiled analyzer" in result.stdout
    assert not missing.exists()


def test_check_missing_analyzer_and_existing_analyzer(tmp_path):
    analyzer = tmp_path / "test.automorf.bin"
    missing = run("--check", "--analyzer-bin", str(analyzer))
    assert missing.returncode != 0
    analyzer.write_bytes(b"compiled binary placeholder")
    present = run("--check", "--analyzer-bin", str(analyzer))
    assert present.returncode == 0
    assert "analyzer available" in present.stdout


def test_installer_rejects_untrusted_apertium_package_name():
    out = run("--apertium-package", "../wrong")
    assert out.returncode != 0
    assert "invalid package" in out.stderr


def test_setup_rejects_conflicting_modes():
    out = run("--check", "--dry-run")
    assert out.returncode != 0
