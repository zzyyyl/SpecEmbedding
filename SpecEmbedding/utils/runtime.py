import logging
import os

import torch

from SpecEmbedding.config import config


def configure_runtime_cache():
    """Use writable cache directories in sandboxed or restricted environments."""
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    requested_device = torch.device(device or config.general.device)
    if requested_device.type != "cuda":
        return requested_device

    if not torch.cuda.is_available():
        logging.warning("CUDA device %s was requested, but CUDA is unavailable. Falling back to CPU.", requested_device)
        return torch.device("cpu")

    if requested_device.index is not None and requested_device.index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested device {requested_device}, but only {torch.cuda.device_count()} CUDA device(s) are available."
        )

    return requested_device


def setup_logging(log_file=None):
    handlers = [
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ] if log_file else [logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def startup_logging(args, message: str = "Start"):
    logging.info("=" * 50)
    logging.info(message)
    logging.info("=" * 50)
    logging.info("Parsed Arguments:")
    for key, value in vars(args).items():
        logging.info("  %s: %s", key, value)

    device = resolve_device(getattr(args, "device", None))
    if device.type == "cuda":
        logging.info("Using device: %s (%s)", device, torch.cuda.get_device_name(device))
    else:
        logging.info("Using device: %s", device)
