import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from SpecEmbedding.utils import gpu, gpu_tmux
from SpecEmbedding.utils.gpu_tmux import Pane, PaneLock, Tmux, WatchOptions
from watch_gpu_tmux import main, parse_args


def pane(**changes):
    values = dict(zip(gpu_tmux.PANE_FIELDS, (
        "$0", "training", "0", "main", "0", "%1", "123", "/dev/pts/3", "bash", "0", "0",
        "/tmp/isolated-test/socket", "999", "0",
    ), strict=True))
    values.update(changes)
    return Pane(values)


IDLE_STATE = {"state": "S", "pgrp": 123, "tpgid": 123, "starttime": "111", "exe": "bash", "children": []}
LOW = {"index": 0, "uuid": "GPU-aaaa", "utilization_gpu_pct": 10, "memory_free_mib": 20000}
HIGH = {**LOW, "utilization_gpu_pct": 99}
COMMAND = ["python", "train.py", "--device", "cuda:0"]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class WatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name) / "locks"
        self.clock = Clock()
        self.tmux = Mock(spec=Tmux)
        self.tmux.resolve.return_value = pane()
        self.options = WatchOptions(gpu=0, target="%1", poll_seconds=10, hold_seconds=20, max_wait_seconds=100)

    def run_watch(self, states, *, idle=None, options=None, clock=None, sleep=None):
        samples = iter(states)

        def snapshot(*args, **kwargs):
            value = next(samples)
            if isinstance(value, Exception):
                raise value
            return value, ""

        with ExitStack() as stack:
            stack.enter_context(patch.object(gpu_tmux, "gpu_snapshot", side_effect=snapshot))
            stack.enter_context(patch.object(gpu_tmux, "gpu_inventory", return_value={0: "GPU-aaaa"}))
            stack.enter_context(patch.object(gpu_tmux, "process_state", return_value=IDLE_STATE))
            stack.enter_context(patch.object(gpu_tmux, "idle_shell", side_effect=idle or [(True, "idle")]))
            return gpu_tmux.watch(
                options or self.options, self.tmux, COMMAND, None, "cuda:0",
                clock=clock or self.clock, sleep=sleep or self.clock.sleep, lock_directory=self.directory,
            )

    def test_continuous_thresholds_and_one_final_recheck(self):
        self.assertEqual(self.run_watch([LOW] * 4), "sent")
        self.assertEqual(self.clock.now, 20)
        self.tmux.send_once.assert_called_once()

    def test_busy_gpu_resets_timer(self):
        self.run_watch([LOW, LOW, HIGH, LOW, LOW, LOW, LOW])
        self.assertEqual(self.clock.now, 50)

    def test_insufficient_memory_resets_timer(self):
        self.run_watch([LOW, {**LOW, "memory_free_mib": 19999}, LOW, LOW, LOW, LOW])
        self.assertEqual(self.clock.now, 40)

    def test_query_error_resets_timer(self):
        for error in (ValueError("N/A"), OSError("missing nvidia-smi"), subprocess.TimeoutExpired("nvidia-smi", 10)):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                self.directory = Path(directory) / "locks"
                self.clock = Clock()
                self.run_watch([LOW, LOW, error, LOW, LOW, LOW, LOW])
                self.assertEqual(self.clock.now, 50)

    def test_final_gpu_race_resets_timer(self):
        self.run_watch([LOW, LOW, LOW, HIGH, LOW, LOW, LOW, LOW])
        self.assertEqual(self.clock.now, 50)

    def test_final_query_error_is_not_idle(self):
        self.run_watch([LOW, LOW, LOW, ValueError("N/A"), LOW, LOW, LOW, LOW])
        self.assertEqual(self.clock.now, 50)

    def test_busy_pane_waits_without_sending_or_interrupting(self):
        self.run_watch([LOW] * 8, idle=[(False, "training"), (True, "idle")])
        self.assertEqual(self.clock.now, 50)
        self.tmux.send_once.assert_called_once()

    def test_missing_target_fails_before_query_or_send(self):
        self.tmux.resolve.side_effect = ValueError("missing target")
        with self.assertRaisesRegex(ValueError, "missing"):
            self.run_watch([])
        self.tmux.send_once.assert_not_called()

    def test_target_disappears_at_final_recheck(self):
        self.tmux.resolve.side_effect = [pane(), ValueError("missing target")]
        with self.assertRaisesRegex(ValueError, "missing"):
            self.run_watch([LOW] * 4)
        self.tmux.send_once.assert_not_called()

    def test_replaced_target_is_not_used(self):
        self.tmux.resolve.side_effect = [pane(), pane(pane_pid="456")]
        with self.assertRaisesRegex(RuntimeError, "target changed"):
            self.run_watch([LOW] * 4)
        self.tmux.send_once.assert_not_called()

    def test_timeout_sends_nothing(self):
        with self.assertRaises(TimeoutError):
            self.run_watch([HIGH] * 4, options=replace(self.options, max_wait_seconds=25))
        self.assertEqual(self.clock.now, 25)
        self.tmux.send_once.assert_not_called()

    def test_dry_run_does_not_consume_dispatch_receipt(self):
        self.assertEqual(self.run_watch([LOW] * 4, options=replace(self.options, dry_run=True)), "dry_run")
        self.tmux.send_once.assert_not_called()
        self.assertEqual(self.run_watch([LOW] * 4), "sent")

    def test_fast_exit_cannot_cause_a_second_dispatch(self):
        self.run_watch([LOW] * 4)
        with self.assertRaisesRegex(RuntimeError, "already attempted"):
            self.run_watch([])
        self.tmux.send_once.assert_called_once()

    def test_send_failure_retains_receipt_and_does_not_retry(self):
        self.tmux.send_once.side_effect = subprocess.TimeoutExpired("tmux", 10)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_watch([LOW] * 4)
        with self.assertRaisesRegex(RuntimeError, "already attempted"):
            self.run_watch([])
        self.tmux.send_once.assert_called_once()

    def test_ctrl_c_releases_unconsumed_lock(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_watch([HIGH], sleep=Mock(side_effect=KeyboardInterrupt))
        self.tmux.send_once.assert_not_called()
        self.run_watch([LOW] * 4)

    def test_two_live_instances_cannot_lock_same_pane(self):
        with PaneLock(("same",), self.directory):
            with self.assertRaises(BlockingIOError):
                with PaneLock(("same",), self.directory):
                    self.fail("acquired a held pane lock")


class TmuxAndArgumentsTest(unittest.TestCase):
    def test_exact_target_forms_and_ambiguity(self):
        tmux = Tmux()
        one = pane()
        with patch.object(tmux, "run", return_value="\t".join(one.fields.values())):
            for target in ("%1", "$0", "training", "$0:0.0", "training:main.0", "training:0.0"):
                self.assertEqual(tmux.resolve(target).id, "%1")
            with self.assertRaises(ValueError):
                tmux.resolve("train")
        two = pane(pane_id="%2", pane_index="1")
        output = "\n".join("\t".join(item.fields.values()) for item in (one, two))
        with patch.object(tmux, "run", return_value=output):
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                tmux.resolve("training")
            self.assertEqual(tmux.resolve("%2").id, "%2")

    def test_linked_windows_are_deduplicated_by_pane_id(self):
        one = pane()
        output = "\n".join(["\t".join(one.fields.values())] * 2)
        with patch.object(Tmux, "run", return_value=output):
            self.assertEqual(Tmux().resolve("%1").id, "%1")

    def test_idle_heuristic_rejects_jobs_modes_and_unknown_shells(self):
        with patch.object(gpu_tmux, "process_state", return_value=IDLE_STATE):
            self.assertTrue(gpu_tmux.idle_shell(pane())[0])
            for changes in ({"pane_dead": "1"}, {"pane_in_mode": "1"}, {"pane_current_command": "python"}, {"synchronize-panes": "1"}):
                self.assertFalse(gpu_tmux.idle_shell(pane(**changes))[0])
        for changes in ({"state": "R"}, {"children": ["456"]}, {"tpgid": 456}, {"exe": "python"}):
            with patch.object(gpu_tmux, "process_state", return_value={**IDLE_STATE, **changes}):
                self.assertFalse(gpu_tmux.idle_shell(pane())[0])
        with patch.object(gpu_tmux, "process_state", side_effect=PermissionError("/proc hidden")):
            self.assertFalse(gpu_tmux.idle_shell(pane())[0])

    def test_safe_argv_and_tmux_literal_transport(self):
        original = [*COMMAND, "--name", "a b 'single' \"double\" $() ; #{pane_id} 中文 \\", ""]
        parsed = gpu_tmux.training_argv(shlex.join(original), None, "", "python")
        self.assertEqual(parsed, original)
        with patch.object(gpu_tmux, "gpu_inventory", return_value={0: "GPU-aaaa"}):
            command = gpu_tmux.dispatch_command(parsed, 0, None, "cuda:0", "GPU-aaaa")
        self.assertEqual(shlex.split(command)[4:], original)
        tmux = Tmux("/tmp/isolated/socket")
        with patch("subprocess.run", return_value=Mock(stdout="")) as run:
            tmux.send_once(pane(), command)
        self.assertEqual(run.call_args.args[0], [
            "tmux", "-S", "/tmp/isolated/socket", "send-keys", "-t", "%1", "-l", "--", command,
            ";", "send-keys", "-t", "%1", "Enter",
        ])
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_mapping_and_device_validation(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(gpu_tmux.cuda_mapping(1, None, ["python", "train.py", "--device=cuda:1"]), (None, "cuda:1"))
            self.assertEqual(gpu_tmux.cuda_mapping(1, "1", COMMAND), ([1], "cuda:0"))
            self.assertEqual(gpu_tmux.cuda_mapping(0, "1,0", ["train", "--device", "cuda:1"]), ([1, 0], "cuda:1"))
            for argv in ([], ["--device", "cpu"], ["--device", "cuda"], ["--device", "cuda:1"], COMMAND + ["--device", "cuda:0"]):
                with self.assertRaises(ValueError):
                    gpu_tmux.cuda_mapping(0, None, argv)
            for visible in ("", "0,0", "1", "GPU-aaaa"):
                with self.assertRaises(ValueError):
                    gpu_tmux.cuda_mapping(0, visible, COMMAND)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            with self.assertRaisesRegex(ValueError, "Inherited"):
                gpu_tmux.cuda_mapping(0, None, COMMAND)
            self.assertEqual(gpu_tmux.cuda_mapping(0, "all", COMMAND), (None, "cuda:0"))

    def test_script_path_spaces_and_quotes(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "train 'space'.py"
            script.touch()
            argv = gpu_tmux.training_argv(None, script, '--device cuda:0 --name "a b"', "python3")
            self.assertEqual(argv, ["python3", str(script), "--device", "cuda:0", "--name", "a b"])

    def test_reject_control_characters_shell_operators_and_wrappers(self):
        for command in ("python train.py\nwhoami", "python train.py && whoami", "CUDA_VISIBLE_DEVICES=0 python x", "env CUDA_VISIBLE_DEVICES=1 python x", "bash -c 'python x'"):
            with self.assertRaises(ValueError):
                gpu_tmux.training_argv(command, None, "", "python")

    def test_invalid_cli_values(self):
        base = ["--gpu", "0", "--target", "%1", "--command", shlex.join(COMMAND), "--cuda-visible-devices", "all"]
        for args in (["--poll-seconds", "0"], ["--hold-seconds", "nan"], ["--max-wait-seconds", "inf"], ["--gpu", "-1"], ["--max-utilization", "101"]):
            with self.subTest(args=args), self.assertRaises(SystemExit), patch("sys.stderr"):
                parse_args(base + args)

    def test_main_exit_codes(self):
        args = ["--gpu", "0", "--target", "%1", "--command", shlex.join(COMMAND), "--cuda-visible-devices", "all"]
        for error, expected in ((KeyboardInterrupt(), 130), (TimeoutError("timeout"), 124), (ValueError("missing pane"), 1)):
            with patch("watch_gpu_tmux.watch", side_effect=error):
                self.assertEqual(main(args), expected)


class GPUQueryTest(unittest.TestCase):
    def test_snapshot_uses_one_bounded_query_when_table_not_needed(self):
        result = Mock(stdout="0, GPU-aaaa, GPU name, 24564, 4564, 20000, 10, 35\n")
        with patch("subprocess.run", return_value=result) as run:
            state, _ = gpu.gpu_snapshot("cuda:0", timeout=2, include_table=False)
        self.assertEqual(state["memory_free_mib"], 20000)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs["timeout"], 2)

    def test_bad_gpu_metrics_are_never_idle(self):
        for text in ("0, GPU-aaaa, GPU, 24564, 0, 24564, N/A, 35", "1, GPU-aaaa, GPU, 24564, 0, 24564, 0, 35", "0, GPU-aaaa, GPU, 24564, 0, -1, 0, 35"):
            with patch("subprocess.run", return_value=Mock(stdout=text)), self.assertRaises(ValueError):
                gpu.gpu_snapshot("cuda:0", include_table=False)

    def test_inventory_and_invalid_entries(self):
        with patch("subprocess.run", return_value=Mock(stdout="0, GPU-aaaa\n1, GPU-bbbb\n")):
            self.assertEqual(gpu.gpu_inventory(), {0: "GPU-aaaa", 1: "GPU-bbbb"})
        for text in ("", "0, N/A", "0, GPU-aaaa\n0, GPU-bbbb"):
            with patch("subprocess.run", return_value=Mock(stdout=text)), self.assertRaises(ValueError):
                gpu.gpu_inventory()

    def test_repository_strict_cuda_guard(self):
        from SpecEmbedding.utils.runtime import resolve_device

        with patch.dict(os.environ, {"SPECEMBEDDING_REQUIRE_CUDA": "1", "SPECEMBEDDING_EXPECTED_CUDA_DEVICE": "cuda:0"}):
            for device in ("cpu", "cuda", "cuda:1"):
                with self.assertRaises(RuntimeError):
                    resolve_device(device)
            with patch("torch.cuda.is_available", return_value=False), self.assertRaisesRegex(RuntimeError, "forbids CPU"):
                resolve_device("cuda:0")
            with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1):
                self.assertEqual(str(resolve_device("cuda:0")), "cuda:0")


@unittest.skipUnless(os.environ.get("RUN_TMUX_INTEGRATION") == "1" and shutil.which("tmux"), "Opt-in isolated tmux test")
class IsolatedTmuxTest(unittest.TestCase):
    def test_literal_dispatch_to_new_private_server_only(self):
        with tempfile.TemporaryDirectory(prefix="specembedding-tmux-test-") as directory:
            root = Path(directory)
            socket = str(root / "tmux.sock")
            prefix = ["tmux", "-S", socket, "-f", "/dev/null"]
            subprocess.run([*prefix, "new-session", "-d", "-s", "isolated", "/bin/bash --noprofile --norc"], check=True, timeout=10)
            try:
                tmux = Tmux(socket)
                target = tmux.resolve("isolated")
                for _ in range(50):
                    if gpu_tmux.idle_shell(tmux.resolve(target.id))[0]:
                        break
                    time.sleep(0.1)
                else:
                    self.fail("Isolated test shell did not become idle")
                result_path = root / "literal result.json"
                # This writes argv to a temporary file; it performs no training or GPU work.
                code = f"import json,sys;from pathlib import Path;Path({str(result_path)!r}).write_text(json.dumps(sys.argv[1:]))"
                sentinel = root / "NEVER_RUN"
                expected = ["--device", "cuda:0", "--name", f"spaces ' \" ; $(touch {sentinel}) #{{pane_id}} 中文 \\", ""]
                argv = [sys.executable, "-c", code, *expected]
                options = WatchOptions(gpu=0, target=target.id, hold_seconds=0, poll_seconds=0.1, max_wait_seconds=5)
                with patch.object(gpu_tmux, "gpu_snapshot", return_value=(LOW, "")), patch.object(gpu_tmux, "gpu_inventory", return_value={0: "GPU-aaaa"}):
                    self.assertEqual(gpu_tmux.watch(options, tmux, argv, None, "cuda:0", lock_directory=root / "locks"), "sent")
                for _ in range(50):
                    if result_path.exists():
                        break
                    time.sleep(0.1)
                self.assertEqual(json.loads(result_path.read_text()), expected)
                self.assertFalse(sentinel.exists())
            finally:
                subprocess.run(["tmux", "-S", socket, "kill-server"], check=False, timeout=10, capture_output=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
