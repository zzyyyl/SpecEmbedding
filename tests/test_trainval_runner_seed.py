import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "run_trainval_overlap_sensitivity.sh"

EXPECTED_HASHES = {
    "train.pkl": "071b6c28ade639261118fdcb0da2510efcea764b2978e2a5d0b61a95a1960c48",
    "val.pkl": "bd13a336e86fb35e77a534e9415cd40cc1a9f15b4a33174c85573b8639c9ca6e",
    "test.pkl": "2309cd689b06a493c66e22f4aa70613b0cfbe2f8fd1d82e605d657f27015d660",
    "tokenset_massspecgym.pkl": "efb9fec12afdd1d66aa6a18492deb83d6f7fc22128d044598fb8cf25619b377c",
    "candidates_mass.pkl": "b4ddf39783ea2dceb32217eab68b71133572f63cc482ba9d17ed9a530c906a8f",
    "candidates_formula.pkl": "0b0d8ffff166eeda9c9dc4060618f823ae4984a71d045119b13b01b5dd5f8194",
}


class TrainValidationRunnerSeedTest(unittest.TestCase):
    def _write_executable(self, path: Path, source: str) -> None:
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)

    def _fake_bin(self, root: Path) -> Path:
        fake_bin = root / "bin"
        fake_bin.mkdir()
        self._write_executable(
            fake_bin / "git",
            """#!/usr/bin/env bash
if [[ "${1:-}" == "rev-parse" ]]; then
  printf '%s\n' seedtest1
fi
exit 0
""",
        )
        hash_cases = "\n".join(
            f'  *"/{filename}") digest="{digest}" ;;' for filename, digest in EXPECTED_HASHES.items()
        )
        self._write_executable(
            fake_bin / "sha256sum",
            f"""#!/usr/bin/env bash
path="${{@: -1}}"
case "$path" in
{hash_cases}
  *) printf 'unexpected path: %s\\n' "$path" >&2; exit 1 ;;
esac
printf '%s  %s\\n' "$digest" "$path"
""",
        )
        for command in ("conda", "nvidia-smi"):
            self._write_executable(fake_bin / command, "#!/usr/bin/env bash\nexit 0\n")
        return fake_bin

    def _run(self, *, alignment_seed: str | None = None, run_prefix: str | None = None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = self._fake_bin(root)
            env = os.environ.copy()
            env.pop("ALIGNMENT_SEED", None)
            env.pop("RUN_PREFIX", None)
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "DATA_PATH": str(root / "data"),
                    "TOKENSET_CACHE": str(root / "tokenset_massspecgym.pkl"),
                    "ALIGN_DEVICE": "cuda:0",
                    "RERANK_DEVICE_LIST": "cuda:0",
                    "MIN_FREE_MIB": "0",
                    "MAX_UTILIZATION": "100",
                }
            )
            if alignment_seed is not None:
                env["ALIGNMENT_SEED"] = alignment_seed
            if run_prefix is not None:
                env["RUN_PREFIX"] = run_prefix
            return subprocess.run(
                ["bash", str(RUNNER), "--dry-run"],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

    def test_default_seed_preserves_existing_run_prefix(self):
        result = self._run()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Alignment seed: 42", result.stdout)
        self.assertIn("Run prefix: seedtest1_massspecgym_nopretrain_valoverlapclean", result.stdout)
        self.assertIn("--seed 42", result.stdout)
        self.assertIn(
            "--run-prefix seedtest1_massspecgym_nopretrain_valoverlapclean",
            result.stdout,
        )
        self.assertIn(
            "Alignment: checkpoints_align/seedtest1_massspecgym_nopretrain_valoverlapclean",
            result.stdout,
        )

    def test_external_prefix_carries_custom_alignment_seed_provenance(self):
        for seed in ("43", "44"):
            with self.subTest(seed=seed):
                run_prefix = f"transfer_massspecgym_valoverlapclean_alignseed{seed}"
                result = self._run(alignment_seed=seed, run_prefix=run_prefix)

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"Alignment seed: {seed}", result.stdout)
                self.assertIn(f"Run prefix: {run_prefix}", result.stdout)
                self.assertIn(f"--seed {seed}", result.stdout)
                self.assertIn(f"--run-prefix {run_prefix}", result.stdout)
                self.assertIn(f"Alignment: checkpoints_align/{run_prefix}", result.stdout)

    def test_nondefault_seed_gets_a_safe_default_prefix(self):
        result = self._run(alignment_seed="43")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "Run prefix: seedtest1_massspecgym_nopretrain_valoverlapclean_alignseed43",
            result.stdout,
        )
        self.assertIn(
            "--run-prefix seedtest1_massspecgym_nopretrain_valoverlapclean_alignseed43",
            result.stdout,
        )

    def test_alignment_seed_must_be_non_negative_integer(self):
        for seed in ("-1", "1.5", "not-a-seed"):
            with self.subTest(seed=seed):
                result = self._run(alignment_seed=seed)

                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    "ALIGNMENT_SEED must be a non-negative integer.",
                    result.stderr,
                )

    def test_completed_alignment_validation_uses_requested_seed(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn(
            'summary_path, checkpoint_path, tokenset_cache, expected_seed, *raw_indices = sys.argv[1:]',
            source,
        )
        self.assertIn('assert summary["seed"] == int(expected_seed)', source)
        self.assertNotIn('assert summary["seed"] == 42', source)
        self.assertIn(
            '"$ALIGN_SELECTION" "$ALIGN_CHECKPOINT" "$TOKENSET_CACHE" "$ALIGNMENT_SEED"',
            source,
        )


if __name__ == "__main__":
    unittest.main()
