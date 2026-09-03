"""Fail fast unless this Python environment has a CUDA-capable PyTorch build."""

import sys

import torch


def main() -> None:
    """Print the selected environment and GPU, or reject a CPU-only setup."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch in this Conda environment."
        )

    probe = torch.ones(1, device="cuda").square()
    if probe.item() != 1:
        raise RuntimeError("The CUDA computation probe returned an invalid result.")

    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")


if __name__ == "__main__":
    main()
