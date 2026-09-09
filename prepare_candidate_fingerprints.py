"""Build and independently audit fixed Morgan inputs for every train or validation candidate."""

import argparse
import json
import logging
from pathlib import Path

from SpecEmbedding.config import config
from SpecEmbedding.utils.fingerprint_preparation import prepare_fingerprint_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--training-candidates', type=Path)
    inputs.add_argument('--validation-index', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='New directory; no overwrite, limits or resuming')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    source_options = {'expected_counts': config.fulltrain.expected_counts.to_dict(),
                      'exclusions': config.fulltrain.exclude_val_query_indices,
                      'tokenizer_config': config.data.tokenizer.to_dict(),
                      'pool_cache_size': config.train.align.candidate_supervision.pool_cache_size}
    kind = 'train' if args.training_candidates else 'validation'
    report = prepare_fingerprint_inputs(args.data_path, args.training_candidates or args.validation_index,
                                        kind, args.output, source_options=source_options,
                                        settings=config.molecule_fingerprints.to_dict())
    print(json.dumps({'state': report['state'], 'kind': kind,
                      'molecules': report['cache']['provenance']['molecules'],
                      'output': str(args.output.resolve()), 'manifest_sha256': report['cache']['manifest_sha256']}, indent=2))


if __name__ == '__main__':
    main()
