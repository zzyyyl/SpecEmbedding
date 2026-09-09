"""Launch a future job with data and dependency caches on external storage."""

import argparse
import json
import os
from pathlib import Path

from SpecEmbedding.config import config
from SpecEmbedding.utils.storage import external_storage_root, storage_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", default=getattr(getattr(config, "storage", None), "root", None))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not args.storage_root or not command:
        parser.error("An external storage root and a command after -- are required")
    root = external_storage_root(args.storage_root)
    overrides = storage_environment(root)
    print(json.dumps({"storage_root": str(root), "environment": overrides, "command": command}, indent=2), flush=True)
    if args.dry_run:
        return
    # Set these before starting Python or importing any model/download libraries.
    for key, value in overrides.items():
        if key not in {"SPECEMBEDDING_STORAGE_ROOT", "SPECEMBEDDING_DATA_ROOT"}:
            Path(value).mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(overrides)
    os.chdir(root)
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
