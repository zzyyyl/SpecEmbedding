import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from SpecEmbedding.config import load_config
from SpecEmbedding.utils.storage import external_storage_root, storage_environment, storage_receipt

ROOT = Path(__file__).resolve().parents[1]


def test_config_external_root_resolves_defaults_and_preserves_historical_config(tmp_path, monkeypatch):
    monkeypatch.delenv("SPECEMBEDDING_STORAGE_ROOT", raising=False)
    path = tmp_path / 'params.yaml'
    path.write_text(yaml.safe_dump({'general': {'save_dir': 'checkpoints'}, 'data': {'cache_path': 'train_cache'}}))
    assert load_config(path).data.cache_path == str(tmp_path / 'train_cache')
    external = tmp_path / 'external'
    monkeypatch.setenv("SPECEMBEDDING_STORAGE_ROOT", str(external))
    loaded = load_config(path)
    assert loaded.data.cache_path == str(external / 'train_cache')
    assert loaded.general.save_dir == str(external / 'checkpoints')
    assert not external.exists()  # Loading configuration is read-only.


@pytest.mark.parametrize('value', ['relative', '/home/example/cache', '/'])
def test_home_or_relative_storage_root_rejected(value):
    with pytest.raises(ValueError):
        external_storage_root(value)


def test_undefined_environment_variable_and_home_symlink_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv('SPECEMBEDDING_UNDEFINED_ROOT', raising=False)
    with pytest.raises(ValueError, match='Unresolved'):
        external_storage_root('${SPECEMBEDDING_UNDEFINED_ROOT}/cache')
    link = tmp_path / 'external'
    link.symlink_to(Path.home(), target_is_directory=True)
    with pytest.raises(ValueError, match='home'):
        external_storage_root(link)


def test_cache_symlink_and_explicit_output_escape_rejected(tmp_path, monkeypatch):
    root = tmp_path / 'external'
    root.mkdir()
    (root / 'cache').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes'):
        storage_environment(root)
    (root / 'cache').unlink()
    path = tmp_path / 'params.yaml'
    path.write_text(yaml.safe_dump({'storage': {'root': str(root)}, 'general': {'save_dir': str(tmp_path / 'outside')}}))
    monkeypatch.delenv('SPECEMBEDDING_STORAGE_ROOT', raising=False)
    with pytest.raises(ValueError, match='escapes'):
        load_config(path)


def test_real_launcher_config_and_child_inherit_external_paths(tmp_path):
    root = tmp_path / 'external'
    receipt = tmp_path / 'receipt.json'
    code = (
        "import json, os, subprocess, sys, tempfile; from pathlib import Path; "
        "from SpecEmbedding.config import config; "
        "from SpecEmbedding.utils.storage import storage_receipt; "
        f"r=storage_receipt(Path({str(root / 'experiments' / 'a05_topk256')!r})); "
        "r['config_cache']=config.data.cache_path; r['temporary']=tempfile.gettempdir(); "
        "r['child']=subprocess.check_output([sys.executable, '-c', 'import os; print(os.environ[\"HF_HOME\"])'],text=True).strip(); "
        f"Path({str(receipt)!r}).write_text(json.dumps(r))"
    )
    environment = os.environ.copy()
    environment.pop('SPECEMBEDDING_CONFIG', None)
    environment.pop('SPECEMBEDDING_STORAGE_ROOT', None)
    environment.update(PYTHONPATH=str(ROOT), HF_HOME=str(Path.home() / '.cache/huggingface'))
    subprocess.run([sys.executable, str(ROOT / 'run_with_storage.py'), '--storage-root', str(root),
                    '--', sys.executable, '-c', code], check=True, env=environment, capture_output=True, text=True)
    result = json.loads(receipt.read_text())
    assert result['config_cache'] == str(root / 'train_cache')
    assert result['temporary'] == str(root / 'tmp')
    assert result['child'] == str(root / 'cache/huggingface')
    assert all(Path(value).is_relative_to(root) for value in result['environment'].values())


def test_storage_receipt_detects_changed_env_and_output(tmp_path, monkeypatch):
    root = tmp_path / 'external'
    for key, value in storage_environment(root).items():
        monkeypatch.setenv(key, value)
    assert storage_receipt(root / 'experiments')['root'] == str(root)
    with pytest.raises(ValueError, match='escapes'):
        storage_receipt(tmp_path / 'wrong')
    monkeypatch.setenv('HF_HOME', str(Path.home() / '.cache/huggingface'))
    with pytest.raises(ValueError, match='environment mismatch'):
        storage_receipt(root / 'experiments')


def test_dry_run_creates_no_directory_or_child(tmp_path):
    root = tmp_path / 'external'
    result = subprocess.run([sys.executable, str(ROOT / 'run_with_storage.py'), '--storage-root', str(root),
                             '--dry-run', '--', 'not-an-executable'], text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)['storage_root'] == str(root)
    assert not root.exists()


def test_actual_queue_child_preserves_external_cache_environment(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import run_massspecgym_v15 as runner

    root = tmp_path / 'external'
    root.mkdir()
    overrides = storage_environment(root)
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    args = SimpleNamespace(output_root=root, source_dir=root / 'raw', legacy_tsv=root / 'old.tsv',
                           gpu=1, device='cuda:1')
    manifest = {'runtime_config': runner.config.to_dict(), 'stages': runner.commands(args)}
    monkeypatch.setattr(runner, 'gpu_inventory', lambda: {0: 'GPU-a', 1: 'GPU-b'})
    monkeypatch.setattr(runner, 'preflight', lambda _: manifest)
    captured = {}
    def child(*args, **kwargs):
        captured.update(kwargs['env'])
        raise RuntimeError('Synthetic child stopped before any data preparation')
    monkeypatch.setattr(runner.subprocess, 'run', child)
    with pytest.raises(RuntimeError, match='Synthetic child'):
        runner.execute(args, manifest)
    assert {key: captured[key] for key in overrides} == overrides
