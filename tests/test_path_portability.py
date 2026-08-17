import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from SpecEmbedding.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigPathTest(unittest.TestCase):
    def write_config(self, directory: Path) -> Path:
        path = directory / "portable.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "general": {"save_dir": "outputs"},
                    "data": {
                        "data_path": "data/processed",
                        "cache_path": "data/cache",
                    },
                    "rerank": {
                        "prepare": {"checkpoint": None},
                        "train": {"save_dir": "rerank"},
                        "eval": {"checkpoint": None},
                    },
                    "paths": {"pretrained_models": {"legacy": "models/model.ckpt"}},
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_relative_paths_are_resolved_from_selected_yaml(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_config(self.write_config(root))

            self.assertEqual(Path(config.general.save_dir), root / "outputs")
            self.assertEqual(Path(config.data.data_path), root / "data/processed")
            self.assertEqual(Path(config.rerank.train.save_dir), root / "rerank")
            self.assertEqual(
                Path(config.paths.pretrained_models.legacy),
                root / "models/model.ckpt",
            )

    def test_environment_selects_config_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_config(root)
            with mock.patch.dict(os.environ, {"SPECEMBEDDING_CONFIG": str(path)}):
                config = load_config()
            self.assertEqual(Path(config.data.cache_path), root / "data/cache")

    def test_default_config_import_is_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = os.environ.copy()
            environment.pop("SPECEMBEDDING_CONFIG", None)
            environment["PYTHONPATH"] = os.pathsep.join(
                [str(REPO_ROOT), environment.get("PYTHONPATH", "")]
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from SpecEmbedding.config import config; print(config.data.data_path)",
                ],
                cwd=temporary,
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertEqual(Path(result.stdout.strip()), REPO_ROOT / "data/processed")


class ImportPurityTest(unittest.TestCase):
    def test_package_import_does_not_disable_numba_jit(self):
        environment = os.environ.copy()
        environment.pop("NUMBA_DISABLE_JIT", None)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT), environment.get("PYTHONPATH", "")]
        )

        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os; import SpecEmbedding; "
                    "assert 'NUMBA_DISABLE_JIT' not in os.environ"
                ),
            ],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )

    def test_imports_do_not_create_runtime_or_dataset_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_paths = {
                "NUMBA_CACHE_DIR": root / "numba",
                "MPLCONFIGDIR": root / "matplotlib",
                "XDG_CACHE_HOME": root / "xdg",
                "SPECEMBEDDING_LEGACY_DATA_ROOT": root / "legacy",
            }
            environment = os.environ.copy()
            environment.update({key: str(value) for key, value in cache_paths.items()})
            environment["PYTHONPATH"] = os.pathsep.join(
                [str(REPO_ROOT), environment.get("PYTHONPATH", "")]
            )
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        "import SpecEmbedding; "
                        "import SpecEmbedding.const.gnps; "
                        "import SpecEmbedding.const.mona; "
                        "import SpecEmbedding.const.tsne_cluster; "
                        f"assert not any(Path(p).exists() for p in {list(map(str, cache_paths.values()))!r}); "
                        "import SpecEmbedding.utils.runtime as runtime; "
                        "runtime.configure_runtime_cache = lambda: (_ for _ in ()).throw(RuntimeError('import-time cache setup')); "
                        "import prepare_rerank_cache; "
                        "import train_rerank; "
                        "import eval_rerank; "
                        "import numpy as np; "
                        "np.load = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('import-time data load')); "
                        "import unique_seed_train"
                    ),
                ],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertFalse(cache_paths["SPECEMBEDDING_LEGACY_DATA_ROOT"].exists())

    def test_explicit_runtime_setup_creates_caches_without_disabling_jit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = os.environ.copy()
            environment.pop("NUMBA_DISABLE_JIT", None)
            environment.update(
                {
                    "NUMBA_CACHE_DIR": str(root / "numba"),
                    "MPLCONFIGDIR": str(root / "matplotlib"),
                    "PYTHONPATH": os.pathsep.join(
                        [str(REPO_ROOT), environment.get("PYTHONPATH", "")]
                    ),
                }
            )
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os; from pathlib import Path; "
                        "from SpecEmbedding.utils.runtime import configure_runtime_cache; "
                        "configure_runtime_cache(); "
                        "assert 'NUMBA_DISABLE_JIT' not in os.environ; "
                        "assert Path(os.environ['NUMBA_CACHE_DIR']).is_dir(); "
                        "assert Path(os.environ['MPLCONFIGDIR']).is_dir()"
                    ),
                ],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            )


class TrackedPathAuditTest(unittest.TestCase):
    def test_tracked_sources_do_not_contain_private_machine_roots(self):
        private_roots = ("/data1/" + "xp", "/data1/" + "zyl")
        files = subprocess.check_output(
            ["git", "ls-files"], cwd=REPO_ROOT, text=True
        ).splitlines()
        offenders = []
        for relative in files:
            path = REPO_ROOT / relative
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(root in text for root in private_roots):
                offenders.append(relative)
        self.assertEqual(offenders, [])

    def test_tracked_notebooks_remain_valid_json(self):
        notebooks = subprocess.check_output(
            ["git", "ls-files", "*.ipynb"], cwd=REPO_ROOT, text=True
        ).splitlines()
        for relative in notebooks:
            with self.subTest(notebook=relative):
                json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))

    def test_legacy_processing_notebook_prepares_output_directories(self):
        notebook = json.loads(
            (REPO_ROOT / "clean/process_msbert_data.ipynb").read_text(
                encoding="utf-8"
            )
        )
        save_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if "np.save(" in "".join(cell.get("source", []))
        ]
        self.assertTrue(save_cells)
        for source in save_cells:
            self.assertIn("mkdir(", source)

    def test_unique_seed_summary_is_written_inside_seed_loop(self):
        tree = ast.parse(
            (REPO_ROOT / "unique_seed_train.py").read_text(encoding="utf-8")
        )
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        seed_loop = next(
            node
            for node in main.body
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Tuple)
            and any(
                isinstance(target, ast.Name) and target.id == "seed"
                for target in node.target.elts
            )
        )
        summary_writes = [
            node
            for node in ast.walk(seed_loop)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to_csv"
        ]
        self.assertEqual(len(summary_writes), 2)


if __name__ == "__main__":
    unittest.main()
