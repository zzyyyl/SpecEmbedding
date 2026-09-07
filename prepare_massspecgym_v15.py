"""Prepare and audit the full pinned v1.5 dataset on CPU, without overwriting prior data."""

import argparse
from pathlib import Path

from SpecEmbedding.config import config
from SpecEmbedding.utils.massspecgym_v15 import prepare_dataset
from SpecEmbedding.utils.runtime import setup_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--legacy-tsv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    setup_logging()
    prepare_dataset(args.source_dir.resolve(), args.legacy_tsv.resolve(), args.output_dir.resolve(),
                    config.fulltrain.v15, config.fulltrain.expected_counts.to_dict(), config.fulltrain.exclude_val_query_indices)


if __name__ == "__main__":
    main()
