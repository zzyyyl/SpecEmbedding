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


@pytest.mark.parametrize('bad', ['..', '.', 'runtime/../../outside'])
def test_runtime_root_cannot_equal_or_escape_storage(tmp_path, bad):
    with pytest.raises(ValueError):
        storage_environment(tmp_path, tmp_path / bad)


def test_runtime_cache_symlink_escape_and_receipt_tampering_rejected(tmp_path, monkeypatch):
    runtime = tmp_path / 'runtime' / 'a12'
    runtime.mkdir(parents=True)
    (runtime / 'cache').symlink_to(tmp_path / 'shared', target_is_directory=True)
    with pytest.raises(ValueError, match='escapes'):
        storage_environment(tmp_path, runtime)
    (runtime / 'cache').unlink()
    for key, value in storage_environment(tmp_path, runtime).items():
        monkeypatch.setenv(key, value)
    assert storage_receipt(tmp_path / 'experiments/a12')['environment']['TMPDIR'] == str(runtime / 'tmp')
    monkeypatch.setenv('TRITON_CACHE_DIR', str(tmp_path / 'runtime/a10/cache/triton'))
    with pytest.raises(ValueError, match='environment mismatch'):
        storage_receipt(tmp_path / 'experiments/a12')


def test_two_actual_launchers_isolate_imports_config_and_runtime_writes(tmp_path):
    root = tmp_path / 'external'
    environment = os.environ.copy()
    environment.pop('SPECEMBEDDING_STORAGE_ROOT', None)
    # A stale inherited namespace must not leak into either explicit invocation.
    environment['SPECEMBEDDING_RUNTIME_ROOT'] = str(root / 'stale')
    children = []
    for name in ('a12', 'a13'):
        source = tmp_path / name
        source.mkdir()
        (source / 'isolated_model.py').write_text(f"VALUE = {name!r}\n")
        params = source / 'params.yaml'
        params.write_text(yaml.safe_dump({'data': {'cache_path': f'runtime/{name}/train_cache'}}))
        output = tmp_path / f'{name}.json'
        code = (
            "import json, tempfile, isolated_model; from pathlib import Path; "
            "from SpecEmbedding.config import config; "
            "from SpecEmbedding.utils.storage import storage_receipt; "
            f"r=storage_receipt(Path({str(root / 'experiments' / name)!r})); "
            "r.update(model=isolated_model.VALUE, module=isolated_model.__file__, "
            "cache=config.data.cache_path, temp=tempfile.gettempdir()); "
            "Path(r['temp'], 'same_filename').write_text(isolated_model.VALUE); "
            f"Path({str(output)!r}).write_text(json.dumps(r))"
        )
        env = {**environment, 'PYTHONPATH': os.pathsep.join((str(source), str(ROOT))),
               'SPECEMBEDDING_CONFIG': str(params)}
        command = [sys.executable, str(ROOT / 'run_with_storage.py'), '--storage-root', str(root),
                   '--runtime-root', str(root / 'runtime' / name), '--', sys.executable, '-c', code]
        children.append(subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    for child in children:
        stdout, stderr = child.communicate(timeout=60)
        assert child.returncode == 0, (stdout, stderr)
    receipts = [json.loads((tmp_path / f'{name}.json').read_text()) for name in ('a12', 'a13')]
    for name, receipt in zip(('a12', 'a13'), receipts, strict=True):
        assert receipt['model'] == name and receipt['module'] == str(tmp_path / name / 'isolated_model.py')
        assert receipt['cache'] == str(root / 'runtime' / name / 'train_cache')
        assert Path(receipt['temp'], 'same_filename').read_text() == name
    shared = {'SPECEMBEDDING_STORAGE_ROOT', 'SPECEMBEDDING_DATA_ROOT'}
    writable = [{v for k, v in r['environment'].items() if k not in shared} for r in receipts]
    assert writable[0].isdisjoint(writable[1])


def test_launcher_without_runtime_option_clears_inherited_namespace(tmp_path):
    root = tmp_path / 'external'
    environment = {**os.environ, 'SPECEMBEDDING_RUNTIME_ROOT': str(root / 'old'), 'PYTHONPATH': str(ROOT)}
    command = [sys.executable, str(ROOT / 'run_with_storage.py'), '--storage-root', str(root), '--',
               sys.executable, '-c',
               "import os; assert 'SPECEMBEDDING_RUNTIME_ROOT' not in os.environ"]
    subprocess.run(command, env=environment, check=True, capture_output=True, timeout=60)
