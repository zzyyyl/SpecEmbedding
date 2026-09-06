"""Linux-only, conservative one-shot tmux dispatch. No training/ML dependencies."""

import fcntl
import hashlib
import json
import logging
import os
import re
import shlex
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from SpecEmbedding.utils.gpu import gpu_inventory, gpu_snapshot, parse_cuda_device

LOGGER = logging.getLogger(__name__)
SHELLS = {"bash", "zsh", "sh", "dash", "ksh"}
PANE_FIELDS = (
    "session_id", "session_name", "window_index", "window_name", "pane_index",
    "pane_id", "pane_pid", "pane_tty", "pane_current_command", "pane_dead",
    "pane_in_mode", "socket_path", "pid", "synchronize-panes",
)


def single_line(value: str) -> str:
    if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("Commands, arguments and targets must be nonempty and contain no control characters")
    return value


def training_argv(command: str | None, script: Path | None, script_args: str, python: str) -> list[str]:
    if command is not None:
        if script_args:
            raise ValueError("--script-args requires --script")
        argv = shlex.split(single_line(command))
    else:
        if script is None:
            raise ValueError("Supply --command or --script")
        path = script.expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("--script must be a regular file")
        if path.suffix == ".py":
            argv = [python, str(path)]
        elif path.suffix == ".sh":
            argv = ["bash", str(path)]
        elif os.access(path, os.X_OK):
            argv = [str(path)]
        else:
            raise ValueError("Script must be .py, .sh, or executable with a valid shebang")
        argv += shlex.split(single_line(script_args) if script_args else "")
    if not argv or "=" in argv[0]:
        raise ValueError("Supply an executable command, not shell environment assignments")
    if Path(argv[0]).name in {"env", "sudo", "su", "ssh", "tmux", "nohup", "xargs", "watch"}:
        raise ValueError("Environment/remote/background launch wrappers are unsupported; supply the training executable/script")
    if Path(argv[0]).name in SHELLS and any(token in {"-c", "-lc", "-ic"} for token in argv[1:]):
        raise ValueError("Inline shell programs are unsupported; use --script")
    for argument in argv:
        if argument:  # Empty quoted argv entries are legitimate.
            single_line(argument)
        if argument in {";", "&", "&&", "||", "|", ">", ">>", "<", "2>"}:
            raise ValueError("Shell operators are not supported; use --script for compound commands")
    return argv


def cuda_mapping(gpu: int, visible: str | None, argv: list[str]) -> tuple[list[int] | None, str]:
    if visible is None and "CUDA_VISIBLE_DEVICES" in os.environ:
        raise ValueError("Inherited CUDA_VISIBLE_DEVICES is ambiguous; specify --cuda-visible-devices explicitly")
    indices = None
    if visible not in (None, "all"):
        if not re.fullmatch(r"\d+(,\d+)*", visible):
            raise ValueError("--cuda-visible-devices must be 'all' or comma-separated physical GPU indices")
        indices = [int(index) for index in visible.split(",")]
        if len(indices) != len(set(indices)) or gpu not in indices:
            raise ValueError("CUDA mapping must contain the monitored GPU exactly once, without duplicates")
    logical = gpu if indices is None else indices.index(gpu)
    expected = f"cuda:{logical}"
    devices = []
    for index, token in enumerate(argv):
        if token == "--device":
            devices.append(argv[index + 1] if index + 1 < len(argv) else "")
        elif token.startswith("--device="):
            devices.append(token.partition("=")[2])
        elif token == "--devices" or token.startswith("--devices="):
            raise ValueError("Multi-device commands are not supported by this single-GPU watcher")
    if devices != [expected]:
        raise ValueError(f"Training command must contain exactly one explicit --device {expected}; got {devices}")
    parse_cuda_device(expected)
    return indices, expected


@dataclass(frozen=True)
class Pane:
    fields: dict[str, str]

    @property
    def id(self) -> str:
        return self.fields["pane_id"]

    @property
    def identity(self) -> tuple[str, ...]:
        return tuple(self.fields[key] for key in ("socket_path", "pid", "pane_id", "pane_pid", "pane_tty"))


class Tmux:
    def __init__(self, socket: str | None = None, timeout: float = 10):
        self.prefix = ["tmux"] + (["-S", socket] if socket else [])
        self.timeout = timeout

    def run(self, *args: str) -> str:
        return subprocess.run(
            [*self.prefix, *args], capture_output=True, text=True, check=True, timeout=self.timeout,
        ).stdout

    def resolve(self, target: str) -> Pane:
        """Use exact matching, never tmux's prefix/fuzzy or active-pane fallback."""
        single_line(target)
        output = self.run("list-panes", "-a", "-F", "\t".join(f"#{{{key}}}" for key in PANE_FIELDS))
        matches = {}
        for line in output.splitlines():
            values = line.split("\t")
            if len(values) != len(PANE_FIELDS):
                raise ValueError("Cannot safely parse tmux pane metadata (tabs/newlines in names?)")
            data = dict(zip(PANE_FIELDS, values, strict=True))
            session_matches = target in (data["session_id"], data["session_name"])
            exact_targets = {
                f"{session}:{window}.{data['pane_index']}"
                for session in (data["session_id"], data["session_name"])
                for window in (data["window_index"], data["window_name"])
            }
            if target == data["pane_id"] or session_matches or target in exact_targets:
                matches[data["pane_id"]] = Pane(data)
        if len(matches) != 1:
            raise ValueError(f"tmux target {target!r} is missing or ambiguous ({len(matches)} panes); use %pane_id")
        return next(iter(matches.values()))

    def send_once(self, pane: Pane, command: str) -> None:
        # Literal text and Enter in one tmux command queue; never shell=True/eval.
        self.run("send-keys", "-t", pane.id, "-l", "--", command, ";", "send-keys", "-t", pane.id, "Enter")


def process_state(pid: str) -> dict:
    proc = Path("/proc") / str(int(pid))
    raw = (proc / "stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    return {
        "state": fields[0], "pgrp": int(fields[2]), "tpgid": int(fields[5]),
        "starttime": fields[19], "exe": (proc / "exe").resolve(strict=True).name,
        "children": (proc / "task" / pid / "children").read_text().split(),
    }


def idle_shell(pane: Pane) -> tuple[bool, str]:
    data = pane.fields
    if data["pane_dead"] != "0" or data["pane_in_mode"] != "0":
        return False, "pane is dead or in copy/view mode"
    if data["synchronize-panes"] not in {"0", "off"}:
        return False, "window has synchronized pane input or its status cannot be verified"
    try:
        state = process_state(data["pane_pid"])
    except (OSError, ValueError, IndexError) as error:
        return False, f"cannot verify Linux /proc shell state: {error}"
    if state["exe"] not in SHELLS or data["pane_current_command"] not in SHELLS:
        return False, "pane is not a supported foreground shell"
    if state["pgrp"] != state["tpgid"] or state["state"] != "S" or state["children"]:
        return False, "shell is active, has foreground/background/stopped children, or lacks terminal foreground"
    return True, "foreground sleeping shell with no child processes (heuristic, not proof of an empty prompt)"


class PaneLock:
    """An advisory lock plus persistent attempt receipt: at-most-once, even after exit/crash."""

    def __init__(self, identity: tuple[str, ...], directory: Path | None = None):
        self.directory = directory or Path(f"/tmp/specembedding-gpu-watch-{os.getuid()}")
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        self.path = self.directory / f"{key}.lock"
        self.fd = None

    def __enter__(self):
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RuntimeError(f"Lock directory must be a private, owned directory: {self.directory}")
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            info = os.fstat(self.fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise RuntimeError("Unsafe pane lock file")
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if os.read(self.fd, 1):
                raise RuntimeError(f"A dispatch was already attempted for this pane; inspect {self.path}")
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise
        return self

    def record(self, status: str) -> None:
        record = json.dumps({"time": time.time(), "monitor_pid": os.getpid(), "status": status}).encode()
        os.lseek(self.fd, 0, os.SEEK_SET)
        os.ftruncate(self.fd, 0)
        os.write(self.fd, record)
        os.fsync(self.fd)

    def __exit__(self, *_):
        os.close(self.fd)  # Do not unlink: another monitor may have opened the same inode.


@dataclass(frozen=True)
class WatchOptions:
    gpu: int
    target: str
    max_utilization: int = 10
    min_free_mib: int = 20000
    poll_seconds: float = 30
    hold_seconds: float = 120
    max_wait_seconds: float | None = None
    dry_run: bool = False


def dispatch_command(
    argv: list[str], gpu: int, indices: list[int] | None, device: str, uuid: str, *, timeout: float = 10
) -> str:
    inventory = gpu_inventory(timeout=timeout)
    if inventory.get(gpu) != uuid:
        raise RuntimeError("Monitored physical GPU identity changed; restart the watcher")
    if indices is None:
        indices = sorted(inventory)
        if indices != list(range(len(indices))):
            raise RuntimeError("Non-contiguous physical indices; specify --cuda-visible-devices explicitly")
    if any(index not in inventory for index in indices):
        raise RuntimeError("CUDA mapping contains a nonexistent GPU")
    visible = ",".join(inventory[index] for index in indices)
    # UUID order removes dependence on CUDA_DEVICE_ORDER and unknown pane environment.
    payload = shlex.join([
        "env", f"CUDA_VISIBLE_DEVICES={visible}", "SPECEMBEDDING_REQUIRE_CUDA=1",
        f"SPECEMBEDDING_EXPECTED_CUDA_DEVICE={device}", *argv,
    ])
    if len(payload.encode()) > 8192:
        raise RuntimeError("Dispatch command exceeds 8192 bytes; use a script")
    return payload


def watch(
    options: WatchOptions, tmux: Tmux, argv: list[str], indices: list[int] | None, device: str,
    *, clock=time.monotonic, sleep=time.sleep, lock_directory: Path | None = None,
) -> str:
    started = clock()
    pane = tmux.resolve(options.target)
    starttime = process_state(pane.fields["pane_pid"])["starttime"]
    identity = (*pane.identity, starttime)
    LOGGER.info("Pinned tmux target %r -> %s; physical GPU %s -> %s", options.target, pane.id, options.gpu, device)
    with PaneLock(identity, lock_directory) as lock:
        LOGGER.info("Pane lock: %s", lock.path)
        ready_since = None
        gpu_uuid = None

        def remaining():
            if options.max_wait_seconds is None:
                return float("inf")
            value = options.max_wait_seconds - (clock() - started)
            if value <= 0:
                raise TimeoutError("GPU/pane waiting deadline reached; no command sent")
            return value

        def sample():
            state, _ = gpu_snapshot(f"cuda:{options.gpu}", timeout=min(10, remaining()), include_table=False)
            LOGGER.info(
                "GPU %s uuid=%s utilization=%s%% free=%s MiB", options.gpu, state["uuid"],
                state["utilization_gpu_pct"], state["memory_free_mib"],
            )
            return state

        def available(state):
            return state["utilization_gpu_pct"] <= options.max_utilization and state["memory_free_mib"] >= options.min_free_mib

        while True:
            remaining()
            try:
                state = sample()
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                LOGGER.warning("GPU query failed; resetting continuous timer: %s", error)
                ready_since = None
            else:
                if gpu_uuid is None:
                    gpu_uuid = state["uuid"]
                if state["uuid"] != gpu_uuid:
                    raise RuntimeError("GPU UUID changed while waiting; refusing dispatch")
                if not available(state):
                    LOGGER.info("Waiting: GPU exceeds utilization/free-memory thresholds; continuous timer reset")
                    ready_since = None
                else:
                    now = clock()
                    ready_since = now if ready_since is None else ready_since
                    held = now - ready_since
                    LOGGER.info("GPU conditions satisfied for %.1f / %.1f seconds", held, options.hold_seconds)
                    if held >= options.hold_seconds:
                        try:
                            command = dispatch_command(
                                argv, options.gpu, indices, device, gpu_uuid, timeout=min(10, remaining()),
                            )
                            final_state = sample()
                        except (OSError, subprocess.SubprocessError, ValueError) as error:
                            LOGGER.warning("Final GPU query failed; timer reset: %s", error)
                            ready_since = None
                        else:
                            if final_state["uuid"] != gpu_uuid or not available(final_state):
                                LOGGER.info("Final GPU check no longer satisfies conditions; timer reset")
                                ready_since = None
                            else:
                                tmux.timeout = min(10, remaining())
                                final_pane = tmux.resolve(options.target)
                                current_start = process_state(final_pane.fields["pane_pid"])["starttime"]
                                if (*final_pane.identity, current_start) != identity:
                                    raise RuntimeError("tmux target changed since startup; refusing dispatch")
                                idle, reason = idle_shell(final_pane)
                                if not idle:
                                    LOGGER.info("Waiting: target %s busy/unverifiable: %s; timer reset", pane.id, reason)
                                    ready_since = None
                                else:
                                    remaining()
                                    LOGGER.info("Final pane check: %s", reason)
                                    LOGGER.info("Explicit environment and argv to dispatch: %s", command)
                                    if options.dry_run:
                                        LOGGER.info("DRY-RUN: would send once to %s; no tmux input sent", pane.id)
                                        return "dry_run"
                                    lock.record("dispatch_attempted")
                                    try:
                                        tmux.send_once(final_pane, command)
                                    except BaseException:
                                        LOGGER.error("Dispatch interrupted/failed: delivery is uncertain; receipt retained, no automatic retry")
                                        raise
                                    # Keep the pre-send receipt on disk; rewriting it risks losing crash safety.
                                    LOGGER.info("SENT: tmux accepted command + Enter for %s; training startup/completion NOT verified", pane.id)
                                    return "sent"
            sleep(min(options.poll_seconds, remaining()))
