"""Build and independently audit every fixed validation molecule graph on CPU."""

import argparse
import json
import logging
from pathlib import Path

from SpecEmbedding.config import config
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.molecule_graph_cache import (
    audit_graph_cache,
    build_graph_cache,
    graph_cache_provenance,
    load_graph_cache,
)
from SpecEmbedding.utils.retrieval_validation import load_validation_index, prepared_validation_input


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory; never overwrite or resume partial artifacts')
    args = parser.parse_args(argv)
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error('Refusing an existing graph cache output')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    options = (args.data_path, config.fulltrain.expected_counts.to_dict(),
               config.fulltrain.exclude_val_query_indices, config.data.tokenizer.to_dict())
    source = prepared_validation_input(args.index, *options)
    index = load_validation_index(args.index, *options)
    provenance = graph_cache_provenance(index['mol_smiles'], index_sha256=source['sha256'],
                                        dataset_manifest_sha256=source['dataset_manifest_sha256'])
    settings = config.retrieval_validation.graph_cache_preparation.to_dict()
    build_graph_cache(index['mol_smiles'], args.output, provenance, **settings)
    audit = audit_graph_cache(index['mol_smiles'], args.output, provenance, **settings)
    _, receipt = load_graph_cache(index['mol_smiles'], args.output, provenance)
    if prepared_validation_input(args.index, *options) != source:
        raise ValueError('Validation input changed during graph cache preparation')
    report = {'state': 'complete_validation_graph_cache_preparation', 'prepared_validation': source,
              'graph_cache': receipt, 'audit': audit, 'settings': settings,
              'entry_source_sha256': sha256_file(Path(__file__)), 'cpu_only': True,
              'model_encoding_performed': False, 'test_evaluated': False}
    with (args.output / 'preparation.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'state': report['state'], 'molecules': provenance['molecules'],
                      'output': str(args.output), 'manifest_sha256': receipt['manifest_sha256']}, indent=2))


if __name__ == '__main__':
    main()
