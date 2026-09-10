"""Prepare and freshly audit observed adducts for every eligible train/val query on CPU."""

import argparse
import json
import logging
from pathlib import Path

from SpecEmbedding.config import config
from SpecEmbedding.utils.adduct_metadata import audit_adduct_cache, load_adduct_cache, prepare_adduct_cache, write_new
from SpecEmbedding.utils.fulltrain import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New external storage directory; no overwrite/resume')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Refusing an existing adduct cache output')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    options = (config.fulltrain.expected_counts.to_dict(), config.fulltrain.exclude_val_query_indices,
               config.data.tokenizer.to_dict(), config.adduct_conditioning_encoder.to_dict())
    prepare_adduct_cache(args.data_path, args.output, *options)
    audit_adduct_cache(args.data_path, args.output, *options)
    splits = {split: load_adduct_cache(args.data_path, args.output, split, *options) for split in ('train', 'val')}
    report = {'state': 'complete_adduct_cache_preparation', 'cpu_only': True,
              'model_encoding_performed': False, 'test_evaluated': False,
              'splits': {split: {'queries': len(value.rows), 'provenance': value.provenance}
                         for split, value in splits.items()}, 'entry_source_sha256': sha256_file(Path(__file__))}
    write_new(args.output / 'preparation.json', report)
    print(json.dumps({'state': report['state'], 'queries': {split: len(value.rows) for split, value in splits.items()}}, indent=2))


if __name__ == '__main__':
    main()
