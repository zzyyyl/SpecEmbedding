"""Prepare complete input fingerprints from already audited candidate inventories."""

import json
from pathlib import Path

from SpecEmbedding.utils.fingerprint_cache import (
    audit_fingerprint_cache,
    build_fingerprint_cache,
    fingerprint_provenance,
    load_fingerprint_cache,
)
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.retrieval_validation import load_validation_index, prepared_validation_input
from SpecEmbedding.utils.training_candidates import load_training_candidates


def fingerprint_source(data_path, index_path, kind, *, expected_counts, exclusions, tokenizer_config, pool_cache_size):
    data_path, index_path = Path(data_path), Path(index_path)
    if kind == 'train':
        index = load_training_candidates(index_path, data_path, expected_counts, exclusions,
                                         pool_cache_size=pool_cache_size)
        return index.metadata['mol_smiles'], {'kind': kind, 'index_path': str(index_path.resolve()),
                                             **index.provenance}
    if kind == 'validation':
        options = (data_path, expected_counts, exclusions, tokenizer_config)
        receipt = prepared_validation_input(index_path, *options)
        index = load_validation_index(index_path, *options)
        return index['mol_smiles'], {'kind': kind, 'index_path': str(index_path.resolve()), **receipt}
    raise ValueError('Fingerprint preparation only supports full train or validation candidate inventories')


def prepare_fingerprint_inputs(data_path, index_path, kind, output, *, source_options, settings):
    output = Path(output).resolve()
    preparation_source_sha = sha256_file(Path(__file__))
    if output.exists():
        raise FileExistsError('Refusing an existing fingerprint preparation output')
    if set(settings) != {'radius', 'bits', 'workers', 'chunk_size'}:
        raise ValueError('Incomplete fingerprint preparation settings')
    smiles, source = fingerprint_source(data_path, index_path, kind, **source_options)
    provenance = fingerprint_provenance(smiles, index_sha256=source['sha256'],
                                        dataset_manifest_sha256=source['dataset_manifest_sha256'],
                                        radius=settings['radius'], bits=settings['bits'])
    worker_options = {key: settings[key] for key in ('workers', 'chunk_size')}
    build_fingerprint_cache(smiles, output, provenance, **worker_options)
    audit = audit_fingerprint_cache(smiles, output, provenance, **worker_options)
    _, cache = load_fingerprint_cache(smiles, output, provenance)
    after_smiles, after_source = fingerprint_source(data_path, index_path, kind, **source_options)
    if after_source != source or after_smiles != smiles:
        raise ValueError('Candidate inventory changed during fingerprint preparation')
    if sha256_file(Path(__file__)) != preparation_source_sha:
        raise ValueError('Fingerprint preparation source changed')
    report = {'state': 'complete_candidate_fingerprint_preparation', 'source': source, 'cache': cache,
              'audit': audit, 'settings': settings, 'preparation_source_sha256': preparation_source_sha,
              'cpu_only': True, 'model_encoding_performed': False, 'test_evaluated': False,
              'positive_labels_created_from_fingerprints': False}
    with (output / 'preparation.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return report
