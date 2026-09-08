"""Bounded GPU-pool queries and per-child UUID binding; no CUDA initialization."""

import logging
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from SpecEmbedding.utils.gpu import gpu_inventory, gpu_snapshot, parse_cuda_device


def add_gpu_arguments(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gpu", type=int, help="One physical nvidia-smi GPU index")
    group.add_argument("--gpus", type=int, nargs="+", help="Physical GPU pool, in tie-break priority order; requires --device cuda:0")


def validate_gpu_arguments(args, *, matching_single=False):
    logical = parse_cuda_device(args.device)
    pool = getattr(args, "gpus", None)
    if pool is not None:
        validate_pool(pool)
        if args.gpu is not None or logical != 0:
            raise ValueError("--gpus requires --device cuda:0 and cannot be combined with --gpu")
    elif args.gpu is None or args.gpu < 0 or (matching_single and logical != args.gpu):
        raise ValueError("Requires a nonnegative physical GPU and matching explicit cuda:N")


def validate_pool(gpus):
    if not gpus or any(type(gpu) is not int or gpu < 0 for gpu in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("GPU pool must contain distinct nonnegative physical indices")


def pin_pool(gpus):
    validate_pool(gpus)
    inventory = gpu_inventory()
    if any(gpu not in inventory for gpu in gpus):
        raise ValueError("GPU pool contains a nonexistent physical GPU")
    # String keys round-trip through preflight JSON without changing equality.
    return {str(gpu): inventory[gpu] for gpu in gpus}


def pool_environment(state, base=None):
    """Bind only the selected UUID; logical cuda:0 stays stable across stages."""
    environment = dict(os.environ if base is None else base)
    environment.update(CUDA_VISIBLE_DEVICES=state["uuid"], SPECEMBEDDING_REQUIRE_CUDA="1",
                       SPECEMBEDDING_EXPECTED_CUDA_DEVICE="cuda:0")
    return environment


def wait_for_any_gpu(gpus, settings, pinned, *, clock=time.monotonic, sleep=time.sleep, snapshot=gpu_snapshot, before_select=None):
    """Independently time every GPU; choose one eligible card, then recheck it.

    A failed query resets only that card. UUID replacement fails closed.
    Polls run concurrently with bounded nvidia-smi timeouts. Ties use CLI order.
    This observes headroom; it does not reserve hardware or stop other jobs.
    """
    validate_pool(gpus)
    if set(pinned) != {str(gpu) for gpu in gpus} or len(set(pinned.values())) != len(gpus):
        raise ValueError("Pinned GPU pool does not match the requested indices")
    for name in ("poll_seconds", "hold_seconds", "min_free_mib", "max_utilization"):
        value = getattr(settings, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid GPU setting: {name}")
    if settings.poll_seconds == 0 or settings.max_utilization > 100:
        raise ValueError("Invalid GPU polling interval/utilization")
    since = dict.fromkeys(gpus)

    def sample(gpu):
        state, _ = snapshot(f"cuda:{gpu}", timeout=10, include_table=False)
        if state["index"] != gpu or state["uuid"] != pinned[str(gpu)]:
            raise RuntimeError(f"Physical GPU {gpu} identity changed during pool wait")
        return state

    def available(state):
        return state["utilization_gpu_pct"] <= settings.max_utilization and state["memory_free_mib"] >= settings.min_free_mib

    with ThreadPoolExecutor(max_workers=min(8, len(gpus))) as queries:
        while True:
            pending = {gpu: queries.submit(sample, gpu) for gpu in gpus}
            ready = []
            for gpu, future in pending.items():
                try:
                    state = future.result()
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    logging.warning("Pool GPU %s query failed; its timer reset: %s", gpu, error)
                    since[gpu] = None
                    continue
                logging.info("Pool GPU %s: util=%s%% free=%s MiB", gpu, state["utilization_gpu_pct"], state["memory_free_mib"])
                if not available(state):
                    since[gpu] = None
                    continue
                now = clock()
                since[gpu] = now if since[gpu] is None else since[gpu]
                if now - since[gpu] >= settings.hold_seconds:
                    ready.append(gpu)
            for gpu in ready:
                # Callers may recheck large input fingerprints here. Query the
                # selected card AFTER that work, immediately before binding it.
                if before_select is not None:
                    before_select()
                try:
                    state = sample(gpu)
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    logging.warning("Pool GPU %s final query failed: %s", gpu, error)
                    since[gpu] = None
                    continue
                if available(state):
                    logging.info("Selected physical GPU %s (%s) -> cuda:0", gpu, state["uuid"])
                    return state
                since[gpu] = None
            sleep(settings.poll_seconds)
