"""Prepare and independently audit complete natural training-candidate structural similarities on CPU."""

import argparse
import json
import logging
from pathlib import Path

from SpecEmbedding.config import config
from SpecEmbedding.utils.storage import storage_receipt
from SpecEmbedding.utils.training_similarity import prepare_training_similarity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-path', 'training-candidates', 'fingerprint-cache', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args(argv)
    if storage_receipt(args.output) is None:
        parser.error('Launch through run_with_storage.py with an explicit external storage root')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    settings = config.training_candidate_sampling
    receipt = prepare_training_similarity(
        args.data_path, args.training_candidates, args.fingerprint_cache, args.output,
        counts=config.fulltrain.expected_counts.to_dict(), exclusions=config.fulltrain.exclude_val_query_indices,
        pool_cache_size=config.train.align.candidate_supervision.pool_cache_size,
        radius=settings.fingerprint_radius, bits=settings.fingerprint_bits,
    )
    print(json.dumps({'state': 'verified_complete_training_similarity', 'directory': receipt['directory'],
                      'queries': receipt['source']['queries'], 'target_rows': receipt['source']['target_rows'],
                      'manifest_sha256': receipt['manifest_sha256'], 'audit_sha256': receipt['audit_sha256']}, indent=2))


if __name__ == '__main__':
    main()
