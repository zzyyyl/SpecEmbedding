"""Wait for GPU headroom, then send one command to an existing idle tmux pane."""

import argparse
import logging
import math
import subprocess
from pathlib import Path

from SpecEmbedding.utils.gpu_tmux import Tmux, WatchOptions, cuda_mapping, training_argv, watch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, help="Physical nvidia-smi GPU index")
    parser.add_argument("--target", required=True, help="Exact session name/$ID, session:window.pane, or preferably %%pane_id")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--command", help="One executable and its arguments, parsed with shlex (no shell operators)")
    source.add_argument("--script", type=Path, help="Existing .py/.sh or executable script")
    parser.add_argument("--script-args", default="", help="Quoted argument string for --script; include --device cuda:N")
    parser.add_argument("--python", default="python", help="Python executable in the target pane for .py scripts")
    parser.add_argument("--cuda-visible-devices", help="Explicit physical index mapping, e.g. 1 or 1,0; default: all in physical-index order")
    parser.add_argument("--max-utilization", type=int, default=10, help="Maximum GPU utilization, percent (inclusive)")
    parser.add_argument("--min-free-mib", type=int, default=20000, help="Minimum free GPU memory, MiB (inclusive)")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--hold-seconds", type=float, default=120, help="Required continuous qualifying sampled duration")
    parser.add_argument("--max-wait-seconds", type=float)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Observe and validate, but never send any tmux input")
    parser.add_argument("--tmux-socket", help="Optional explicit tmux server socket path (-S), useful for isolated tests")
    args = parser.parse_args(argv)
    if args.gpu < 0 or args.min_free_mib < 0 or not 0 <= args.max_utilization <= 100:
        parser.error("GPU index/memory must be nonnegative and utilization must be in [0, 100]")
    for name in ("poll_seconds", "hold_seconds", "max_wait_seconds"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0 or (name != "hold_seconds" and value == 0)):
            parser.error(f"--{name.replace('_', '-')} must be finite and {'nonnegative' if name == 'hold_seconds' else 'positive'}")
    try:
        args.training_argv = training_argv(args.command, args.script, args.script_args, args.python)
        args.indices, args.device = cuda_mapping(args.gpu, args.cuda_visible_devices, args.training_argv)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    handlers = [logging.StreamHandler()]
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)
    options = WatchOptions(**{name: getattr(args, name) for name in WatchOptions.__dataclass_fields__})
    try:
        watch(options, Tmux(args.tmux_socket), args.training_argv, args.indices, args.device)
    except KeyboardInterrupt:
        logging.info("Interrupted by Ctrl+C; monitor stopped, no training/tmux processes terminated")
        return 130
    except TimeoutError as error:
        logging.warning("%s", error)
        return 124
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        logging.error("Monitor stopped without automatic retry: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
