"""Audit a completed single-seed alignment optimization run without starting GPU work."""

import argparse
import json
import logging
from pathlib import Path

from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.optimization_audit import audit_optimization_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON receipt; existing files are never replaced")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.output.exists():
        parser.error("Refusing an existing audit output")
    result = audit_optimization_run(args.run)
    module = Path(__file__).parent / "SpecEmbedding" / "utils" / "optimization_audit.py"
    result["audit_source_sha256"] = {str(path.resolve()): sha256_file(path) for path in (Path(__file__), module)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("state", "epochs", "selected_epoch", "selected_metrics", "selected_vs_baseline")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
