#!/usr/bin/env python3
"""Runtime check for the isolated Titan V / PyTorch CUDA 12.6 environment.

No installation occurs here. Invoke after setting up the venv, or independently
before a GPU experiment. The small CUDA matrix multiplication detects binaries
that import successfully but cannot actually execute an sm_70 kernel.
"""
from __future__ import annotations

import argparse
import json
import sys

TORCH_VERSION = "2.14.1"
CUDA_VERSION = "12.6"


def inspect_titanv(torch, *, run_kernel: bool = True) -> dict:
    """Validate the required runtime against a torch-like module (also unit-testable)."""
    installed = str(torch.__version__).split("+", 1)[0]
    if installed != TORCH_VERSION:
        raise RuntimeError(
            f"Expected torch {TORCH_VERSION}, found {torch.__version__}. "
            "Use scripts/setup_titanv.sh to install the pinned CUDA 12.6 wheel."
        )
    cuda_version = str(torch.version.cuda)
    if cuda_version != CUDA_VERSION:
        raise RuntimeError(
            f"Expected PyTorch CUDA runtime {CUDA_VERSION}, found {cuda_version}. "
            "The ordinary pip torch wheel may not support sm_70; use the cu126 index."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the NVIDIA driver with 'nvidia-smi' "
            "and ensure the GPU is visible in CUDA_VISIBLE_DEVICES."
        )
    found = []
    for idx in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(idx)
        cap = tuple(torch.cuda.get_device_capability(idx))
        found.append({"index": idx, "name": name, "capability": list(cap)})
        if "TITAN V" not in name.upper() or cap != (7, 0):
            continue
        supported = torch.cuda.get_arch_list()
        if "sm_70" not in supported:
            raise RuntimeError(f"PyTorch binary does not include sm_70 kernels: {supported}")
        if run_kernel:
            x = torch.ones((128, 128), device=f"cuda:{idx}")
            y = x @ x
            torch.cuda.synchronize(idx)
            if y.shape != (128, 128):
                raise RuntimeError("GPU matrix multiplication produced an unexpected shape")
        return {
            "ok": True,
            "torch": str(torch.__version__),
            "cuda_runtime": cuda_version,
            "gpu": name,
            "gpu_index": idx,
            "capability": [7, 0],
            "kernel_test": "passed" if run_kernel else "skipped",
        }
    raise RuntimeError(
        "No visible NVIDIA TITAN V (sm_70) was found. Visible devices: "
        + json.dumps(found)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-kernel", action="store_true", help="Skip the small GPU compute test")
    args = parser.parse_args(argv)
    try:
        import torch  # optional dependency, intentional lazy import
        report = inspect_titanv(torch, run_kernel=not args.skip_kernel)
    except (ImportError, RuntimeError, OSError) as exc:
        print(f"Titan V GPU check failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
