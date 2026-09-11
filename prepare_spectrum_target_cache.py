"""Build and independently replay all train-only targets for the spectrum auxiliary task."""

import argparse
import json
import logging
import os
from pathlib import Path

from SpecEmbedding.config import config
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.spectrum_targets import (
    audit_spectrum_target_cache,
    load_spectrum_target_cache,
    prepare_spectrum_target_cache,
)
from SpecEmbedding.utils.storage import storage_receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New external cache directory; no overwrite or resume')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Refusing an existing spectrum target cache output')
    if storage_receipt(args.output) is None or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('Launch CPU target preparation through run_with_storage.py with CUDA_VISIBLE_DEVICES empty')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    options = (config.fulltrain.expected_counts.to_dict(), config.fulltrain.exclude_val_query_indices,
               config.data.tokenizer.to_dict(), config.molecule_spectrum_auxiliary.target.to_dict())
    prepare_spectrum_target_cache(args.data_path, args.output, *options)
    audit_spectrum_target_cache(args.data_path, args.output, *options)
    cache = load_spectrum_target_cache(args.data_path, args.output, *options)
    report = {'state': 'complete_train_spectrum_target_preparation', 'queries': len(cache),
              'cpu_only': True, 'model_encoding_performed': False, 'provenance': cache.provenance,
              'storage': storage_receipt(args.output), 'entry_source_sha256': sha256_file(Path(__file__))}
    with (args.output / 'preparation.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'state': report['state'], 'queries': len(cache), 'cpu_only': True}, indent=2))


if __name__ == '__main__':
    main()
