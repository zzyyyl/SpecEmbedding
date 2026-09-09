"""Prepare a full CPU validation index or freshly evaluate an alignment checkpoint on it."""

import argparse
import json
import os
from pathlib import Path

import torch

from SpecEmbedding.config import config
from SpecEmbedding.utils.align import load_align_model
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.massspecgym_v15 import write_json
from SpecEmbedding.utils.optimization_audit import audit_snapshot
from SpecEmbedding.utils.retrieval_validation import (
    AlignmentRetrievalValidator,
    load_validation_index,
    prepare_validation_index,
    validation_index_receipt,
)
from SpecEmbedding.utils.runtime import resolve_device, setup_logging


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--spectrum-control", choices=("permuted", "constant"),
                        help="Full validation input intervention; separate output, never checkpoint selection")
    args = parser.parse_args(argv)
    if args.spectrum_control is not None and args.prepare_only:
        parser.error("Spectrum controls require checkpoint evaluation, not index preparation")
    if args.prepare_only:
        if args.index.exists() or args.index.with_suffix(".json").exists():
            parser.error("Refusing existing validation index artifacts")
        args.index.parent.mkdir(parents=True, exist_ok=True)
        setup_logging(args.index.with_suffix(".log"))
        index = prepare_validation_index(args.data_path, config.fulltrain.expected_counts.to_dict(),
                                         config.fulltrain.exclude_val_query_indices, config.data.tokenizer.to_dict(),
                                         workers=config.fulltrain.v15.audit_workers)
        with args.index.open("xb") as handle:
            torch.save(index, handle)
        write_json(args.index.with_suffix(".json"), validation_index_receipt(index, args.index))
        return
    if args.output is None or args.output.exists():
        parser.error("Fresh checkpoint evaluation requires a new --output directory")
    if not args.device or not args.device.startswith("cuda:") or os.environ.get("SPECEMBEDDING_REQUIRE_CUDA") != "1":
        parser.error("Formal validation encoding requires strict explicit cuda:N after the GPU gate")
    if config.model.mol_encoder.graph_policy != "rdkit_sanitized":
        parser.error("Formal validation requires sanitized molecular graphs")
    device = resolve_device(args.device)
    index = load_validation_index(args.index, args.data_path, config.fulltrain.expected_counts.to_dict(),
                                  config.fulltrain.exclude_val_query_indices, config.data.tokenizer.to_dict())
    selection = json.loads((args.checkpoint.parent / "alignment_selection.json").read_text())
    if (selection["seed"] != 42 or selection["fulltrain_audit"]["dataset_version"] != "1.5"
            or selection["fulltrain_audit"]["input_outputs"] != index["dataset_outputs"]):
        raise ValueError("Baseline checkpoint has different data provenance")
    if args.spectrum_control is not None and (
        selection["checkpoint_sha256"] != sha256_file(args.checkpoint)
        or selection["model_config"] != config.model.to_dict()
        or not selection["fulltrain_audit"]["formal_fulltrain"]
        or selection["exclude_val_query_indices"] != config.fulltrain.exclude_val_query_indices
    ):
        raise ValueError("Spectrum control checkpoint/configuration provenance mismatch")
    validator = AlignmentRetrievalValidator(index, config.retrieval_validation, args.output,
                                            spectrum_control=args.spectrum_control,
                                            control_settings=config.retrieval_validation.spectrum_controls
                                            if args.spectrum_control is not None else None)
    args.output.mkdir(parents=True)
    setup_logging(args.output / "validation.log")
    model = load_align_model(args.checkpoint, device, config.model.mol_encoder.norm_type, config.model.mol_encoder.norm_eps)
    stage = f"control_{args.spectrum_control}" if args.spectrum_control is not None else "baseline"
    metrics = validator(model, device, 0, stage)
    extra = {"spectrum_control": validator.control_metadata, "used_for_checkpoint_selection": False,
             "selection_sha256": sha256_file(args.checkpoint.parent / "alignment_selection.json")}
    if args.spectrum_control is None:
        extra = {}
    else:
        snapshot = args.output / f"{stage}_epoch000.pt"
        audit_snapshot(snapshot, index, expected_spectrum_control=validator.control_metadata)
        extra.update(saved_score_audit="passed", snapshot_sha256=sha256_file(snapshot))
    write_json(args.output / "metrics.json", {"metrics": metrics, "checkpoint_sha256": sha256_file(args.checkpoint),
                                              "index_sha256": sha256_file(args.index), "protocol": index["protocol"],
                                              "device": str(device), "split": "val", "test_evaluated": False, **extra})


if __name__ == "__main__":
    main()
