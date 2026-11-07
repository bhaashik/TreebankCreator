"""Test cloud transfers without using any real Google Drive / Dropbox credentials."""
from pathlib import Path
import os
import shutil
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/cloud_io.sh"
INSTALLER = Path(__file__).resolve().parents[1] / "scripts/setup_lightning.sh"


def _mock_project(tmp_path):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, project / "scripts/cloud_io.sh")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_rclone = fake_bin / "rclone"
    fake_rclone.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$MOCK_RCLONE_ARGS\"\n")
    fake_rclone.chmod(0o755)
    env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
               MOCK_RCLONE_ARGS=str(tmp_path / "rclone_args.txt"))
    return project, env


def test_cloud_pull_and_push_are_non_destructive(tmp_path):
    project, env = _mock_project(tmp_path)
    pull = subprocess.run(["bash", str(project / "scripts/cloud_io.sh"), "pull-data",
                           "drive:UDTreebankLab/data"], capture_output=True, text=True,
                          env=env, check=True)
    assert "Copying cloud" in pull.stdout
    args = (tmp_path / "rclone_args.txt").read_text().splitlines()
    assert args[:3] == ["copy", "drive:UDTreebankLab/data", str(project / "data")]
    assert "sync" not in args

    (project / "runs").mkdir()
    (project / "runs" / "keep.json").write_text("{}")
    env["UDTL_DRY_RUN"] = "1"
    subprocess.run(["bash", str(project / "scripts/cloud_io.sh"), "push-runs",
                    "dropbox:UDTreebankLab/backups/run_1"], env=env, check=True,
                   capture_output=True, text=True)
    args = (tmp_path / "rclone_args.txt").read_text().splitlines()
    assert args[:3] == ["copy", str(project / "runs"), "dropbox:UDTreebankLab/backups/run_1"]
    assert "--dry-run" in args
    assert "sync" not in args


def test_cloud_rejects_remote_root(tmp_path):
    project, env = _mock_project(tmp_path)
    result = subprocess.run(["bash", str(project / "scripts/cloud_io.sh"), "push-runs", "drive:"],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 2
    assert "SUBFOLDER" in result.stderr


def test_scripts_have_help_and_valid_shell_syntax():
    for script in [SCRIPT, INSTALLER]:
        subprocess.run(["bash", "-n", str(script)], check=True)
        result = subprocess.run(["bash", str(script), "--help"],
                                capture_output=True, text=True, check=True)
        assert "Usage:" in result.stdout
