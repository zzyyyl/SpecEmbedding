"""Synchronous epoch measurements; no polling process, CUDA fallback or model changes."""

import json
import math
import os
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from SpecEmbedding.utils.fulltrain import sha256_file


class EpochResources:
    def __init__(self, model, device):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            raise ValueError("Resource measurements require an explicit CUDA device index")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Resource measurements support CPU smoke or explicit CUDA only")
        self.parameters = sum(parameter.numel() for parameter in model.parameters())
        self.trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        self._sync()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started = self.previous = time.monotonic()
        self.phases = {}

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def phase(self, name):
        if name in self.phases:
            raise ValueError(f"Resource phase was already recorded: {name}")
        self._sync()
        current = time.monotonic()
        self.phases[name] = current - self.previous
        self.previous = current

    def finish(self, observed_counts=None):
        self.phase("checkpoint_and_selection")
        expected = {"train", "contrastive_validation", "retrieval_validation", "checkpoint_and_selection"}
        if set(self.phases) != expected:
            raise ValueError("Incomplete epoch resource phases")
        counts = None if observed_counts is None else {key: observed_counts[key] for key in ("train", "val")}
        train_seconds = self.phases["train"]
        result = {"schema_version": 1, "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                  "device": str(self.device), "pid": os.getpid(), "parameter_count": self.parameters,
                  "trainable_parameter_count": self.trainable_parameters, "phase_seconds": dict(self.phases),
                  "epoch_seconds": self.previous - self.started, "observed_queries": counts,
                  "train_queries_per_second": counts["train"] / train_seconds if counts and train_seconds > 0 else None,
                  "cpu_main_process_lifetime_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                  * (1 if sys.platform == "darwin" else 1024),
                  "cpu_scope": "Main process lifetime peak RSS, including earlier preparation; excludes DataLoader workers",
                  "cuda_scope": "This process/device PyTorch allocator peak during this epoch, including selection/checkpoint work",
                  "timing_scope": "Synchronized wall time; includes data loading, validation and checkpoint IO; not deployment latency"}
        if self.device.type == "cuda":
            result.update(cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                          cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                          cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        else:
            result.update(cuda_visible_devices=None, cuda_peak_allocated_bytes=None, cuda_peak_reserved_bytes=None)
        return result


def audit_resource_profiles(directory, stage, expected_counts, device):
    """Verify recorded resources; older immutable runs explicitly remain unmeasured."""
    profiles = stage.get("resource_profiles")
    if profiles is None:
        return {"state": "not_recorded", "reason": "The fixed training source did not save epoch resource profiles"}, {}
    if not isinstance(profiles, list) or len(profiles) != stage["stop_epoch"] or not profiles:
        raise ValueError("Incomplete resource profile trajectory")
    hashes = {}
    phases = {"train", "contrastive_validation", "retrieval_validation", "checkpoint_and_selection"}
    for epoch, profile in enumerate(profiles, 1):
        seconds = profile["phase_seconds"]
        if (profile["schema_version"] != 1 or profile["stage"] != "stage2" or profile["epoch"] != epoch
                or profile["device"] != str(device) or profile["observed_queries"] != expected_counts
                or set(seconds) != phases or any(not math.isfinite(value) or value < 0 for value in seconds.values())
                or not math.isclose(profile["epoch_seconds"], sum(seconds.values()), rel_tol=1e-9, abs_tol=1e-6)
                or seconds["train"] <= 0 or not math.isclose(profile["train_queries_per_second"],
                                                            expected_counts["train"] / seconds["train"], rel_tol=1e-9)
                or not 0 <= profile["trainable_parameter_count"] <= profile["parameter_count"]
                or profile["parameter_count"] != profiles[0]["parameter_count"]
                or not isinstance(profile["cpu_main_process_lifetime_peak_rss_bytes"], int)
                or profile["cpu_main_process_lifetime_peak_rss_bytes"] <= 0):
            raise ValueError("Resource profile timing/count/device mismatch")
        allocated, reserved = profile["cuda_peak_allocated_bytes"], profile["cuda_peak_reserved_bytes"]
        if torch.device(device).type == "cuda":
            if not isinstance(allocated, int) or not isinstance(reserved, int) or not 0 < allocated <= reserved:
                raise ValueError("Invalid CUDA resource peaks")
        elif allocated is not None or reserved is not None:
            raise ValueError("CPU resource profile claims CUDA peaks")
        path = Path(directory) / "resources" / f"stage2_epoch{epoch:03d}.json"
        if json.loads(path.read_text()) != profile:
            raise ValueError("Resource file differs from the selection record")
        hashes[str(path)] = sha256_file(path)
    return {"state": "verified", "epochs": len(profiles), "first_epoch": profiles[0],
            "sum_epoch_seconds": sum(profile["epoch_seconds"] for profile in profiles),
            "maximum_epoch_cuda_peak_allocated_bytes": max(profile["cuda_peak_allocated_bytes"] for profile in profiles)
            if torch.device(device).type == "cuda" else None}, hashes
