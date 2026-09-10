"""Prepare a full CPU validation index or freshly evaluate an alignment checkpoint on it."""

import argparse
import json
import os
from pathlib import Path

import torch

from SpecEmbedding.config import config
from SpecEmbedding.utils.adduct_alignment_inputs import adduct_model_settings, load_spectrum_metadata
from SpecEmbedding.utils.align import load_align_model
from SpecEmbedding.utils.fingerprint_alignment_inputs import load_alignment_fingerprints
from SpecEmbedding.utils.fingerprint_validation import FingerprintRetrievalValidator
from SpecEmbedding.utils.formal_alignment import fingerprint_input_bits, formal_model_type, load_formal_alignment
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.graph_fingerprint_validation import GraphFingerprintRetrievalValidator
from SpecEmbedding.utils.massspecgym_v15 import write_json
from SpecEmbedding.utils.optimization_audit import audit_snapshot
from SpecEmbedding.utils.retrieval_validation import (
    AlignmentRetrievalValidator,
    load_validation_graph_cache,
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
    parser.add_argument("--graph-cache", type=Path, help="Audited fixed molecule inputs, never model embeddings")
    parser.add_argument('--fingerprint-cache', type=Path, help='The fingerprint checkpoint\'s own audited validation inputs')
    parser.add_argument('--spectrum-metadata-cache', type=Path,
                        help='The conditioned checkpoint\'s own complete observed-adduct inputs')
    parser.add_argument("--checkpoint-model-config", action="store_true",
                        help="Construct the baseline from its own verified selection metadata; preserve shared data/tokenizer protocol")
    parser.add_argument("--spectrum-control", choices=("permuted", "constant", "precursor_only"),
                        help="Full validation input intervention; separate output, never checkpoint selection")
    args = parser.parse_args(argv)
    if args.graph_cache and args.prepare_only:
        parser.error("--graph-cache requires checkpoint evaluation")
    if args.checkpoint_model_config and args.prepare_only:
        parser.error("--checkpoint-model-config requires checkpoint evaluation")
    if args.fingerprint_cache and (args.prepare_only or not args.checkpoint_model_config):
        parser.error('Fingerprint validation requires independent checkpoint construction')
    if args.spectrum_metadata_cache and (args.prepare_only or not args.checkpoint_model_config):
        parser.error('Adduct validation requires independent checkpoint construction')
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
    index = load_validation_index(args.index, args.data_path, config.fulltrain.expected_counts.to_dict(),
                                  config.fulltrain.exclude_val_query_indices, config.data.tokenizer.to_dict())
    selection = json.loads((args.checkpoint.parent / "alignment_selection.json").read_text())
    kind = formal_model_type(selection['model_config']) if args.checkpoint_model_config else 'gine'
    adduct_settings = adduct_model_settings(selection['model_config'])
    if (adduct_settings is not None) != bool(args.spectrum_metadata_cache):
        raise ValueError('Baseline adduct model and explicit observed metadata disagree')
    if not args.checkpoint_model_config and (
        adduct_settings is not None or adduct_model_settings(config.model.to_dict()) is not None
    ):
        raise ValueError('Adduct validation requires independently bound checkpoint construction')
    metadata = metadata_receipts = None
    if args.spectrum_metadata_cache:
        if args.spectrum_control is not None:
            raise ValueError('Adduct spectrum controls require a separately registered metadata policy')
        metadata, metadata_receipts = load_spectrum_metadata(args.data_path, args.spectrum_metadata_cache,
            counts=config.fulltrain.expected_counts.to_dict(), exclusions=config.fulltrain.exclude_val_query_indices,
            tokenizer_config=config.data.tokenizer.to_dict(), settings=adduct_settings, index=index)
        if metadata_receipts != selection.get('spectrum_metadata'):
            raise ValueError('Baseline adduct inputs differ from its own selected model')
    if (kind in ('fingerprint', 'gine_fingerprint')) != bool(args.fingerprint_cache):
        raise ValueError('Baseline model type and molecular inputs disagree')
    if kind == 'fingerprint' and args.graph_cache:
        raise ValueError('Fingerprint-only baseline does not accept graph-cache input')
    graph_cache = graph_receipt = None
    if args.graph_cache:
        graph_cache, graph_receipt = load_validation_graph_cache(args.index, index, args.graph_cache)
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
    control = {'spectrum_control': args.spectrum_control,
               'control_settings': config.retrieval_validation.spectrum_controls if args.spectrum_control is not None else None}
    if metadata is not None:
        control['spectrum_metadata'] = metadata['val']
    fingerprint_input = None
    if args.fingerprint_cache:
        expected_cache = selection.get('validation_fingerprint_cache')
        if not expected_cache:
            raise ValueError('Fingerprint checkpoint is missing its validation input receipt')
        _, fingerprint_input = load_alignment_fingerprints(
            args.data_path, args.index, 'validation', args.fingerprint_cache,
            counts=config.fulltrain.expected_counts.to_dict(), exclusions=config.fulltrain.exclude_val_query_indices,
            tokenizer_config=config.data.tokenizer.to_dict(), settings=expected_cache['provenance']['options'],
            pool_cache_size=config.train.align.candidate_supervision.pool_cache_size)
        if fingerprint_input['cache'] != expected_cache:
            raise ValueError('Validation inputs differ from the fingerprint checkpoint\'s own inputs')
        validator_class = FingerprintRetrievalValidator if kind == 'fingerprint' else GraphFingerprintRetrievalValidator
        validator = validator_class(
            index, config.retrieval_validation, args.output, fingerprint_root=args.fingerprint_cache,
            fingerprint_provenance=expected_cache['provenance'], index_sha256=sha256_file(args.index), **control,
            **({} if kind == 'fingerprint' else {'graph_cache': graph_cache}))
    else:
        validator = AlignmentRetrievalValidator(index, config.retrieval_validation, args.output, graph_cache=graph_cache, **control)
    args.output.mkdir(parents=True)
    setup_logging(args.output / "validation.log")
    device = resolve_device(args.device)
    model_receipt = None
    if args.checkpoint_model_config:
        model, _, model_receipt = load_formal_alignment(
            args.checkpoint, device, dataset_outputs=index['dataset_outputs'],
            dataset_manifest_sha256=index['dataset_manifest_sha256'], tokenizer_config=config.data.tokenizer.to_dict(),
            expected_counts={'train': config.fulltrain.expected_counts.train,
                             'val': config.fulltrain.expected_counts.val - len(config.fulltrain.exclude_val_query_indices)},
            exclusions=config.fulltrain.exclude_val_query_indices,
        )
        if model_receipt['model_type'] != kind:
            raise ValueError('Baseline model type and molecular inputs disagree')
        if args.fingerprint_cache and (fingerprint_input_bits(model_receipt['model_config'])
                                        != fingerprint_input['cache']['provenance']['options']['bits']):
            raise ValueError('Fingerprint model width differs from its fixed inputs')
    else:
        model = load_align_model(args.checkpoint, device, config.model.mol_encoder.norm_type, config.model.mol_encoder.norm_eps)
    stage = f"control_{args.spectrum_control}" if args.spectrum_control is not None else "baseline"
    metrics = validator(model, device, 0, stage)
    if args.graph_cache:
        _, final_graph_receipt = load_validation_graph_cache(args.index, index, args.graph_cache)
        if final_graph_receipt != graph_receipt:
            raise ValueError("Validation graph cache changed during encoding")
    if fingerprint_input is not None:
        _, after = load_alignment_fingerprints(
            args.data_path, args.index, 'validation', args.fingerprint_cache,
            counts=config.fulltrain.expected_counts.to_dict(), exclusions=config.fulltrain.exclude_val_query_indices,
            tokenizer_config=config.data.tokenizer.to_dict(), settings=expected_cache['provenance']['options'],
            pool_cache_size=config.train.align.candidate_supervision.pool_cache_size)
        if after != fingerprint_input:
            raise ValueError('Fingerprint validation inputs changed during encoding')
    extra = {"spectrum_control": validator.control_metadata, "used_for_checkpoint_selection": False,
             "selection_sha256": sha256_file(args.checkpoint.parent / "alignment_selection.json")}
    if args.spectrum_control is None:
        extra = {}
    else:
        snapshot = args.output / f"{stage}_epoch000.pt"
        audit_snapshot(snapshot, index, expected_spectrum_control=validator.control_metadata,
                       expected_graph_cache=validator.graph_cache_fingerprint,
                       expected_fingerprint_cache=fingerprint_input['cache'] if fingerprint_input else None)
        extra.update(saved_score_audit="passed", snapshot_sha256=sha256_file(snapshot))
    if model_receipt is not None:
        extra['checkpoint_model'] = model_receipt
    if fingerprint_input is not None:
        extra['validation_fingerprint_cache'] = fingerprint_input['cache']
    if metadata is not None:
        _, after = load_spectrum_metadata(args.data_path, args.spectrum_metadata_cache,
            counts=config.fulltrain.expected_counts.to_dict(), exclusions=config.fulltrain.exclude_val_query_indices,
            tokenizer_config=config.data.tokenizer.to_dict(), settings=adduct_settings, index=index)
        if after != metadata_receipts:
            raise ValueError('Adduct validation inputs changed during encoding')
        extra['spectrum_metadata'] = metadata_receipts
    write_json(args.output / "metrics.json", {"metrics": metrics, "checkpoint_sha256": sha256_file(args.checkpoint),
                                              "index_sha256": sha256_file(args.index), "protocol": index["protocol"],
                                              "device": str(device), "split": "val", "test_evaluated": False,
                                              "validation_graph_cache": graph_receipt, **extra})


if __name__ == "__main__":
    main()
