"""Select a single CUDA device before importing CUDA-dependent libraries."""
import argparse
import os
import re


def single_cuda_device(value):
    value = value.strip()
    # Empty/-1 retain the standard CUDA way to disable GPUs for CPU runs.
    if value in ("", "-1") or re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+", value):
        return value
    raise argparse.ArgumentTypeError(
        "Choose one GPU index or GPU/MIG UUID (e.g. 0 or 2), not a comma-separated list; "
        "use -1 to disable CUDA."
    )


def add_cuda_device_argument(parser):
    # Preserve externally selected GPUs, but reduce multi-GPU masks to one GPU.
    default = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0]
    parser.add_argument(
        "--cuda-visible-devices", type=single_cuda_device, default=default,
        help="One CUDA GPU index or UUID; overrides CUDA_VISIBLE_DEVICES before CUDA imports. "
             "Default: first inherited visible GPU, otherwise 0. Use -1 for CPU only.",
    )


def configure_cuda_device(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    selection = args.cuda_visible_devices
    detail = "CUDA disabled" if selection in ("", "-1") else "single GPU; local CUDA device 0"
    print(f"CUDA_VISIBLE_DEVICES={selection!r} ({detail})", flush=True)


def model_device_map(device):
    """Place the entire model explicitly; never ask Accelerate to infer a map.

    Call only after configure_cuda_device. A selected physical GPU is local 0.
    CPU is an explicit testing/inference option, never an OOM fallback.
    """
    import torch
    target = torch.device(device)
    if target.type == 'cuda':
        return {'': target.index if target.index is not None else 0}
    if target.type == 'cpu':
        return {'': 'cpu'}
    raise ValueError(f'Unsupported model device: {device}')
