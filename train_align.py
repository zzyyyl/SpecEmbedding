import argparse
import hashlib
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from rdkit import rdBase
from torch.utils.data import DataLoader

from SpecEmbedding.config import config
from SpecEmbedding.data.datasets_adduct import AdductAlignmentDataset, adduct_align_collate_fn, bind_candidate_adducts
from SpecEmbedding.data.datasets_align import AlignGraphDataset, align_collate_fn
from SpecEmbedding.data.datasets_candidates import CandidateAlignDataset, candidate_align_collate_fn
from SpecEmbedding.data.datasets_fingerprint import (
    CandidateFingerprintDataset,
    FingerprintAlignmentDataset,
    candidate_fingerprint_collate_fn,
    fingerprint_align_collate_fn,
)
from SpecEmbedding.data.datasets_graph_fingerprint import (
    CandidateGraphFingerprintDataset,
    GraphFingerprintAlignmentDataset,
)
from SpecEmbedding.data.overlap import filter_classified_validation
from SpecEmbedding.models_align import GINEEncoder, SpecMolAlignModel
from SpecEmbedding.models_precursor_delta import build_spectrum_encoder
from SpecEmbedding.models_spectrum_aux import attach_spectrum_auxiliary, auxiliary_model_settings
from SpecEmbedding.trainer.trainer import set_seed
from SpecEmbedding.trainer.trainer_align import TrainerAlign
from SpecEmbedding.trainer.trainer_candidates import CandidateTrainerAlign
from SpecEmbedding.trainer.trainer_spectrum_aux import SpectrumAuxiliaryTrainer
from SpecEmbedding.utils.adduct_alignment_inputs import (
    adduct_model_settings,
    load_spectrum_metadata,
    spectrum_metadata_files,
)
from SpecEmbedding.utils.adduct_metadata import SpectrumMetadata
from SpecEmbedding.utils.candidate_training import (
    read_candidate_training_input,
    validate_candidate_sampling_binding,
    validate_candidate_settings,
)
from SpecEmbedding.utils.fingerprint_alignment_inputs import load_alignment_fingerprints
from SpecEmbedding.utils.fingerprint_validation import FingerprintRetrievalValidator
from SpecEmbedding.utils.formal_alignment import build_formal_alignment, fingerprint_input_bits, formal_model_type
from SpecEmbedding.utils.fulltrain import sha256_file
from SpecEmbedding.utils.graph_fingerprint_validation import GraphFingerprintRetrievalValidator
from SpecEmbedding.utils.mass_batching import MassBlockBatchSampler, molecular_exact_masses
from SpecEmbedding.utils.massspecgym_v15 import classified_full_spectra
from SpecEmbedding.utils.model import SiameseModel
from SpecEmbedding.utils.providers import get_provider
from SpecEmbedding.utils.retrieval_validation import (
    AlignmentRetrievalValidator,
    load_validation_graph_cache,
    load_validation_index,
)
from SpecEmbedding.utils.runtime import resolve_device, setup_logging, startup_logging
from SpecEmbedding.utils.spectrum_auxiliary_inputs import auxiliary_training_weight
from SpecEmbedding.utils.spectrum_targets import SpectrumTargetCache, load_spectrum_target_cache
from train import add_base_argument, get_classified_data


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_align(
    train_data: dict,
    train_keys: list,
    val_data: dict,
    val_keys: list,
    spec_encoder: SiameseModel,
    batch_size: int = config.train.align.batch_size,
    lr: float = config.train.align.lr,
    save_dir: str = config.general.save_dir,
    graph_cache_size: int = config.train.align.graph_cache_size,
    mol_norm_type: str = getattr(config.model.mol_encoder, "norm_type", "layernorm"),
    mol_norm_eps: float = getattr(config.model.mol_encoder, "norm_eps", 1e-5),
    device: str | torch.device | None = None,
    selection_metadata: dict | None = None,
    seed: int = config.general.seed,
    formal_fulltrain: bool = False,
    retrieval_validator=None,
    training_candidates=None,
    candidate_input_receipt=None,
    fingerprint_inputs=None,
    fingerprint_smiles=None,
    spectrum_metadata=None,
    spectrum_targets=None,
):
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    candidate_settings = config.train.align.candidate_supervision.to_dict()
    validate_candidate_settings(candidate_settings)
    if candidate_settings["enabled"] != (training_candidates is not None):
        raise ValueError("Candidate supervision requires both enabled configuration and verified input")
    if candidate_settings["enabled"] and (
        not formal_fulltrain or retrieval_validator is None or spec_encoder is not None or candidate_input_receipt is None
    ):
        raise ValueError("Candidate supervision requires formal fresh training and full retrieval selection")
    if not candidate_settings["enabled"] and candidate_input_receipt is not None:
        raise ValueError("Unexpected candidate input in an inactive run")
    auxiliary_settings = auxiliary_model_settings(config.model.to_dict())
    auxiliary_weight = auxiliary_training_weight(config.model.to_dict(), config.train.align.to_dict())
    if (auxiliary_settings is not None) != (spectrum_targets is not None):
        raise ValueError('Auxiliary training requires explicit model, weight and full target cache together')
    if auxiliary_settings is not None:
        if (not formal_fulltrain or retrieval_validator is None or spec_encoder is not None
                or not candidate_settings['enabled'] or not isinstance(spectrum_targets, SpectrumTargetCache)):
            raise ValueError('Spectrum auxiliary requires fresh full candidate training and retrieval selection')
        source = spectrum_targets.provenance['source']
        if (source['settings'] != auxiliary_settings['target']
                or source['dataset_manifest_sha256'] != selection_metadata['fulltrain_audit']['dataset_manifest_sha256']
                or source['tokenizer_config'] != config.data.tokenizer.to_dict()
                or source['exclude_val_query_indices'] != selection_metadata['exclude_val_query_indices']
                or source['queries'] != selection_metadata['fulltrain_audit']['expected_epoch_counts']['train']):
            raise ValueError('Spectrum auxiliary targets disagree with the actual full training protocol')
        formal_model_type(config.model.to_dict())
    graph_context = hasattr(config.model, 'graph_global_context')
    if graph_context:
        if (not formal_fulltrain or retrieval_validator is None or spec_encoder is not None
                or not candidate_settings['enabled']):
            raise ValueError('Graph context requires fresh full candidate training and bound validation')
        formal_model_type(config.model.to_dict())
        if mol_norm_type != config.model.mol_encoder.norm_type or mol_norm_eps != config.model.mol_encoder.norm_eps:
            raise ValueError('Graph context normalization must match the pinned molecular configuration')
    if candidate_settings['enabled']:
        validate_candidate_sampling_binding(training_candidates, candidate_settings, candidate_input_receipt)
    adduct_settings = adduct_model_settings(config.model.to_dict())
    if (adduct_settings is not None) != (spectrum_metadata is not None):
        raise ValueError('Adduct conditioning requires both explicit configuration and complete metadata inputs')
    metadata_receipts = None
    if adduct_settings is not None:
        if (not formal_fulltrain or retrieval_validator is None or spec_encoder is not None
                or not candidate_settings['enabled'] or set(spectrum_metadata) != {'train', 'val'}
                or any(not isinstance(item, SpectrumMetadata) for item in spectrum_metadata.values())):
            raise ValueError('Adduct conditioning requires fresh full candidate training and bound validation')
        formal_model_type(config.model.to_dict())
        metadata_receipts = {split: item.provenance for split, item in spectrum_metadata.items()}
        spectrum_metadata_files(metadata_receipts)
        if (metadata_receipts['train']['source']['settings'] != adduct_settings
                or getattr(retrieval_validator, 'spectrum_metadata_fingerprint', None) != metadata_receipts['val']
                or selection_metadata.get('spectrum_metadata', metadata_receipts) != metadata_receipts):
            raise ValueError('Adduct model, selection and validation metadata bindings disagree')
    kind = getattr(config.model, 'type', 'gine')
    fingerprint_model = kind == 'fingerprint'
    uses_fingerprints = kind in ('fingerprint', 'gine_fingerprint')
    if kind not in ('gine', 'fingerprint', 'gine_fingerprint'):
        raise ValueError('Unknown alignment model type')
    if ((uses_fingerprints and (fingerprint_inputs is None or fingerprint_smiles is None))
            or (not uses_fingerprints and (fingerprint_inputs is not None or fingerprint_smiles is not None))):
        raise ValueError('Fingerprint model requires both complete train and validation inputs')
    if uses_fingerprints and (not formal_fulltrain or retrieval_validator is None or spec_encoder is not None):
        raise ValueError('Fingerprint alignment currently requires fresh formal training and full retrieval selection')
    if uses_fingerprints:
        if set(fingerprint_inputs) != {'train', 'validation'} or set(fingerprint_smiles) != {'train', 'validation'}:
            raise ValueError('Fingerprint model requires both complete train and validation inputs')
        if any(item['cache']['provenance']['options']['bits'] != fingerprint_input_bits(config.model.to_dict())
               for item in fingerprint_inputs.values()):
            raise ValueError('Fingerprint model width differs from its fixed inputs')
    if hasattr(config.model.spec_encoder, 'precursor_delta') and (
        not formal_fulltrain or retrieval_validator is None or spec_encoder is not None
    ):
        raise ValueError('Precursor delta training requires fresh formal training and full retrieval selection')
    if hasattr(config.model.spec_encoder, 'attention_pool'):
        if not formal_fulltrain or retrieval_validator is None or spec_encoder is not None:
            raise ValueError('Attention pooling requires fresh formal training and full retrieval selection')
        formal_model_type(config.model.to_dict())
    device = resolve_device(device)
    batching = config.train.align.batching
    if batching not in {"random", "mass_blocks"}:
        raise ValueError(f"Unknown alignment batching: {batching}")
    if batching == "mass_blocks" and not formal_fulltrain:
        raise ValueError("Mass-block batching requires full-spectrum alignment")
    epochs_stage1 = config.train.align.epochs_stage1
    epochs_stage2 = config.train.align.epochs_stage2

    logging.info("1. 初始化数据集与 DataLoader...")
    # 使用自定义的 AlignGraphDataset (继承自 TrainDataset)
    dataset_class = {'gine': AlignGraphDataset, 'fingerprint': FingerprintAlignmentDataset,
                     'gine_fingerprint': GraphFingerprintAlignmentDataset}[kind]
    def dataset_extra(split):
        if not uses_fingerprints:
            return {}
        receipt = fingerprint_inputs[split]['cache']
        return {'fingerprint_root': receipt['directory'], 'fingerprint_smiles': fingerprint_smiles[split],
                'fingerprint_provenance': receipt['provenance'],
                'dataset_manifest_sha256': selection_metadata['fulltrain_audit']['dataset_manifest_sha256']}
    train_dataset = dataset_class(
        data=train_data,
        keys=train_keys,
        n_views=1,
        is_augment=True,
        graph_cache_size=0 if fingerprint_model else graph_cache_size,
        full_spectra=formal_fulltrain,
        graph_policy=config.model.mol_encoder.graph_policy,
        **({"augment_config": config.augmentation.to_dict()} if formal_fulltrain else {}),
        **dataset_extra('train'),
    )
    val_dataset = dataset_class(
        data=val_data,
        keys=val_keys,
        n_views=1,
        is_augment=False,
        graph_cache_size=0 if fingerprint_model else graph_cache_size,
        full_spectra=formal_fulltrain,
        graph_policy=config.model.mol_encoder.graph_policy,
        **({"augment_config": config.augmentation.to_dict()} if formal_fulltrain else {}),
        **dataset_extra('validation'),
    )
    if formal_fulltrain:
        expected = selection_metadata["fulltrain_audit"]["expected_epoch_counts"]
        if len(train_dataset) != expected["train"] or len(val_dataset) != expected["val"]:
            raise ValueError("Formal alignment Dataset does not cover every eligible spectrum")
        # Fixed validation ordering mixes identities without resampling spectra each epoch.
        random.Random(seed).shuffle(val_dataset._spectrum_indices)
        logging.info("Formal alignment spectra: train=%s val=%s; validation permutation seed=%s", len(train_dataset), len(val_dataset), seed)

    # 为了确保可复现性，设置 Generator 和 worker_init_fn
    g = torch.Generator()
    g.manual_seed(seed)

    # 必须使用 align_collate_fn 来组装 PyG 的 Graph Batch
    train_batching = {"batch_size": batch_size, "shuffle": True}
    if batching == "mass_blocks":
        smiles = [train_data[key][offset]["smiles"] for key, offset in train_dataset._spectrum_indices]
        sampler = MassBlockBatchSampler(molecular_exact_masses(smiles), batch_size=batch_size,
                                       block_size=config.train.align.mass_block_size, seed=seed)
        train_batching = {"batch_sampler": sampler}
        logging.info("Alignment mass blocks: queries=%s batch=%s block=%s mass_sha256=%s",
                     len(smiles), batch_size, config.train.align.mass_block_size, sampler.mass_sha256)
    if candidate_settings["enabled"]:
        candidate_dataset_class = {'gine': CandidateAlignDataset, 'fingerprint': CandidateFingerprintDataset,
                                   'gine_fingerprint': CandidateGraphFingerprintDataset}[kind]
        train_dataset = candidate_dataset_class(
            train_dataset, training_candidates,
            dataset_manifest_sha256=selection_metadata["fulltrain_audit"]["dataset_manifest_sha256"],
            negative_count=candidate_settings["negative_count"], seed=seed,
            **({} if fingerprint_model else {'graph_cache_size': candidate_settings['graph_cache_size']}),
        )
        if (train_dataset.provenance["dataset_to_raw_query_sha256"]
                != candidate_input_receipt["dataset_to_raw_query_sha256"]):
            raise ValueError("Actual training query order differs from candidate input preflight")
        logging.info("Formal natural candidate supervision: %s", train_dataset.provenance)
    if spectrum_metadata is not None:
        bind_candidate_adducts(train_dataset, spectrum_metadata['train'])
        val_dataset = AdductAlignmentDataset(val_dataset, spectrum_metadata['val'])
    paired_collate = fingerprint_align_collate_fn if fingerprint_model else align_collate_fn
    validation_collate = adduct_align_collate_fn if spectrum_metadata is not None else paired_collate
    candidate_collate = candidate_fingerprint_collate_fn if fingerprint_model else candidate_align_collate_fn
    train_loader = DataLoader(
        train_dataset,
        **train_batching,
        collate_fn=candidate_collate if candidate_settings["enabled"] else paired_collate,
        num_workers=config.train.align.num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=validation_collate,
        num_workers=config.train.align.num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    logging.info("2. 初始化模型...")

    has_pretrained_spec = spec_encoder is not None
    if uses_fingerprints or graph_context:
        model = build_formal_alignment({key: value for key, value in config.model.to_dict().items()
                                        if key != 'spectrum_auxiliary'})
    else:
        # Preserve the original GINE initialization order and legacy pretrained path.
        mol_encoder = GINEEncoder(
            emb_dim=config.model.mol_encoder.emb_dim,
            n_layers=config.model.mol_encoder.n_layers,
            dropout_rate=config.model.mol_encoder.dropout_rate,
            size_feature_dim=config.model.mol_encoder.size_feature_dim,
            norm_type=mol_norm_type,
            norm_eps=mol_norm_eps,
        )
        if not spec_encoder:
            spec_encoder = build_spectrum_encoder(config.model.spec_encoder.to_dict())
        model = SpecMolAlignModel(
            spec_encoder=spec_encoder,
            mol_encoder=mol_encoder,
            spec_dim=config.model.spec_encoder.dim_target,
            hidden_dim=config.model.align.final_dim,
            final_dim=config.model.align.final_dim,
            dropout_rate=config.model.align.dropout_rate,
            tau=config.model.align.tau
        )
    if auxiliary_settings is not None:
        attach_spectrum_auxiliary(model, auxiliary_settings)
    trainer_class = CandidateTrainerAlign if candidate_settings["enabled"] else TrainerAlign
    if auxiliary_settings is not None:
        trainer_class = SpectrumAuxiliaryTrainer
    trainer = trainer_class(
        model,
        train_loader,
        val_loader,
        device,
        save_dir=save_dir,
        retrieval_validator=retrieval_validator,
        record_resources=formal_fulltrain,
        **({"candidate_loss_weight": candidate_settings["loss_weight"]} if candidate_settings["enabled"] else {}),
        **({'spectrum_targets': spectrum_targets, 'auxiliary_loss_weight': auxiliary_weight}
           if auxiliary_settings is not None else {}),
    )
    if formal_fulltrain:
        trainer.expected_epoch_counts = expected

    # ==========================================
    # 阶段一：冻结质谱特征，仅训练分子编码器与投影头
    # ==========================================
    logging.info("\n" + "="*50)
    logging.info("Stage 1: Frozen MS Encoder, Train Mol Encoder & Projection Heads")
    logging.info("="*50)

    if not has_pretrained_spec:
        logging.info("No pretrained model, skipped.")
    else:
        for param in model.spec_encoder.parameters():
            param.requires_grad = False

        stage1_params = [
            {'params': model.mol_encoder.parameters(), 'lr': lr * 10}, # 1e-3
            {'params': model.spec_proj.parameters(), 'lr': lr * 10},
            {'params': model.mol_proj.parameters(), 'lr': lr * 10},
            {'params': [model.logit_scale], 'lr': lr * 10},
        ]

        optimizer1 = optim.AdamW(stage1_params, weight_decay=config.train.align.weight_decay)
        trainer.fit(epochs=epochs_stage1, optimizer=optimizer1, stage_name="stage1", patience=config.train.align.patience)

    # ==========================================
    # 阶段二：解冻所有参数，端到端微调
    # ==========================================
    logging.info("\n" + "="*50)
    logging.info("Stage 2: Unfreeze All, End-to-End Fine-tuning")
    logging.info("="*50)

    for param in model.spec_encoder.parameters():
        param.requires_grad = True

    spec_encoder_lr = lr / 10 if has_pretrained_spec else lr
    logging.info(f"Using spec_encoder lr={spec_encoder_lr:.2e} in stage2.")

    stage2_params = [
        {'params': model.spec_encoder.parameters(), 'lr': spec_encoder_lr},
        {'params': model.mol_encoder.parameters(), 'lr': lr},
        {'params': model.spec_proj.parameters(), 'lr': lr},
        {'params': model.mol_proj.parameters(), 'lr': lr},
        {'params': [model.logit_scale], 'lr': lr},
    ]
    if auxiliary_settings is not None:
        stage2_params.append({'params': model.spectrum_auxiliary.parameters(), 'lr': lr})

    optimizer2 = optim.AdamW(stage2_params, weight_decay=config.train.align.weight_decay)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs_stage2)

    trainer.fit(epochs=epochs_stage2, optimizer=optimizer2, scheduler=scheduler2, stage_name="stage2", patience=config.train.align.patience)

    best_checkpoint = Path(save_dir) / "best_model_stage2.pth"
    checkpoint_digest = hashlib.sha256()
    with best_checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checkpoint_digest.update(chunk)
    selection_summary = {
        **(selection_metadata or {}),
        "checkpoint": str(best_checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_digest.hexdigest(),
        "stages": trainer.stage_summaries,
        "model_config": config.model.to_dict(),
        "training_config": config.train.align.to_dict(),
        "config_snapshot": config.to_dict(),
        "graph_policy": config.model.mol_encoder.graph_policy,
        "rdkit_version": rdBase.rdkitVersion,
        "device": str(device),
    }
    if metadata_receipts is not None:
        selection_summary['spectrum_metadata'] = metadata_receipts
    if spectrum_targets is not None:
        selection_summary['spectrum_targets'] = spectrum_targets.provenance
    if formal_fulltrain:
        selection_summary["fulltrain_audit"]["epochs"] = trainer.epoch_counts
        selection_summary["fulltrain_audit"]["validation_permutation_seed"] = seed
        selection_summary["fulltrain_audit"]["training_batching"] = batching
    (Path(save_dir) / "alignment_selection.json").write_text(
        json.dumps(selection_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logging.info("\nTwo-stage training completed successfully.")
    return model

def main():
    parser = argparse.ArgumentParser(description="Two-Stage Cross-Modal Alignment Training for SpecEmbedding")
    add_base_argument(parser)
    parser.add_argument("--batch_size", type=int, default=config.train.align.batch_size, help="Batch size for alignment training")
    parser.add_argument("--lr", type=float, default=config.train.align.lr, help="Base learning rate")
    parser.add_argument("--graph_cache_size", type=int, default=config.train.align.graph_cache_size, help="Lazy LRU molecule graph cache size per DataLoader worker. Use 0 to disable and -1 for unlimited.")
    parser.add_argument("--mol_norm_type", type=str, choices=["layernorm", "rmsnorm"], default=getattr(config.model.mol_encoder, "norm_type", "layernorm"), help="Normalization used in the molecule GINE encoder.")
    parser.add_argument("--mol_norm_eps", type=float, default=getattr(config.model.mol_encoder, "norm_eps", 1e-5), help="Epsilon used by molecule encoder normalization.")
    parser.add_argument("--seed", type=int, default=config.general.seed, help="Random seed for alignment training")
    parser.add_argument("--pretrained_spec", type=str, help="Path to your pre-trained SpecEmbedding model weights")
    parser.add_argument("--formal-fulltrain", action="store_true", help="Audited v1.5, fresh all-spectrum tokenization, strict CUDA, no old weights or caches")
    parser.add_argument("--validation-index", type=Path, help="Audited full Mass validation candidate/identity index for retrieval checkpoint selection")
    parser.add_argument("--validation-graph-cache", type=Path, help="Fully audited fixed input graphs; embeddings remain fresh")
    parser.add_argument("--candidate-training-input", type=Path,
                        help="Pinned full natural training-candidate receipt from the optimization runner")
    parser.add_argument('--fingerprint-training-index', type=Path)
    parser.add_argument('--training-fingerprint-cache', type=Path)
    parser.add_argument('--validation-fingerprint-cache', type=Path)
    parser.add_argument('--spectrum-metadata-cache', type=Path,
                        help='Complete audited observed-adduct cache required by the conditioned spectrum tower')
    parser.add_argument('--spectrum-target-cache', type=Path,
                        help='Audited complete train-only peak targets required by the auxiliary prediction head')
    parser.add_argument(
        "--tokenset_cache",
        "--tokenset-cache",
        dest="tokenset_cache",
        type=str,
        help="Exact classified TokenSet cache file used for alignment training.",
    )
    parser.add_argument(
        "--exclude_val_query_indices",
        "--exclude-val-query-indices",
        dest="exclude_val_query_indices",
        nargs="*",
        type=int,
        default=[],
        help="Zero-based raw validation query indices excluded before alignment model selection.",
    )

    args = parser.parse_args()
    auxiliary_settings = auxiliary_model_settings(config.model.to_dict())
    auxiliary_training_weight(config.model.to_dict(), config.train.align.to_dict())
    if (auxiliary_settings is not None) != bool(args.spectrum_target_cache):
        parser.error('Auxiliary configuration and --spectrum-target-cache must be supplied together')
    if auxiliary_settings is not None and (
        not args.formal_fulltrain or not args.validation_index or args.pretrained_spec or not args.candidate_training_input
    ):
        parser.error('Spectrum auxiliary requires fresh formal candidate training and full retrieval selection')
    if hasattr(config.model, 'graph_global_context'):
        if (not args.formal_fulltrain or not args.validation_index or args.pretrained_spec
                or not args.candidate_training_input or not config.train.align.candidate_supervision.enabled):
            parser.error('Graph context requires fresh formal candidate training and full retrieval selection')
        formal_model_type(config.model.to_dict())
    adduct_settings = adduct_model_settings(config.model.to_dict())
    if (adduct_settings is not None) != bool(args.spectrum_metadata_cache):
        parser.error('Adduct configuration and --spectrum-metadata-cache must be supplied together')
    if adduct_settings is not None and (
        not args.formal_fulltrain or not args.validation_index or args.pretrained_spec or not args.candidate_training_input
    ):
        parser.error('Adduct conditioning requires fresh formal candidate training and full retrieval selection')
    if hasattr(config.model.spec_encoder, 'attention_pool'):
        if not args.formal_fulltrain or not args.validation_index or args.pretrained_spec:
            parser.error('Attention pooling requires fresh formal training and full retrieval selection')
        formal_model_type(config.model.to_dict())
    if hasattr(config.model.spec_encoder, 'precursor_delta'):
        if not args.formal_fulltrain or not args.validation_index or args.pretrained_spec:
            parser.error('Precursor delta training requires fresh formal training and full retrieval selection')
        formal_model_type(config.model.to_dict())
    if config.train.align.metric_for_best not in ("validation_contrastive_loss", "validation_top1_then_mrr"):
        parser.error("Unknown alignment checkpoint-selection metric")
    if (config.train.align.metric_for_best == "validation_top1_then_mrr") != bool(args.validation_index):
        parser.error("Retrieval selection requires --validation-index and the matching pinned metric configuration")
    if args.validation_index and not args.formal_fulltrain:
        parser.error("Retrieval selection requires audited formal full-training data")
    if args.validation_graph_cache and not args.validation_index:
        parser.error("--validation-graph-cache requires --validation-index")
    kind = getattr(config.model, 'type', 'gine')
    fingerprint_model = kind == 'fingerprint'
    uses_fingerprints = kind in ('fingerprint', 'gine_fingerprint')
    fingerprint_paths = (args.fingerprint_training_index, args.training_fingerprint_cache, args.validation_fingerprint_cache)
    if any(fingerprint_paths) != uses_fingerprints or (uses_fingerprints and not all(fingerprint_paths)):
        parser.error('Fingerprint model and all three fixed input paths must be provided together')
    if uses_fingerprints and (not args.formal_fulltrain or not args.validation_index
                              or (fingerprint_model and args.validation_graph_cache)):
        parser.error('Fingerprint training requires formal retrieval selection and fingerprint validation inputs')
    if uses_fingerprints:
        formal_model_type(config.model.to_dict())
        if (fingerprint_input_bits(config.model.to_dict()) != config.molecule_fingerprints.bits
                or (fingerprint_model and (config.augmentation.node_drop_rate != 0 or config.augmentation.edge_mask_rate != 0))):
            parser.error('Fingerprint width must match fixed inputs and graph augmentation must be explicitly disabled')
    candidate_settings = config.train.align.candidate_supervision.to_dict()
    validate_candidate_settings(candidate_settings)
    if candidate_settings["enabled"] != bool(args.candidate_training_input):
        parser.error("Candidate supervision requires matching enabled configuration and --candidate-training-input")
    if args.candidate_training_input and (not args.formal_fulltrain or not args.validation_index):
        parser.error("Candidate supervision requires formal training and full retrieval selection")
    args.exclude_val_query_indices = sorted(set(args.exclude_val_query_indices))
    if any(index < 0 for index in args.exclude_val_query_indices):
        parser.error("--exclude-val-query-indices must contain non-negative integers")
    if args.seed < 0:
        parser.error("--seed must be a non-negative integer")
    if args.formal_fulltrain:
        if args.pretrained_spec or args.tokenset_cache or args.dataset_type != "massspecgym":
            parser.error("Formal v1.5 alignment forbids inherited weights/TokenSet caches and requires MassSpecGym")
        if config.model.mol_encoder.graph_policy != "rdkit_sanitized":
            parser.error("Formal v1.5 alignment requires rdkit_sanitized graphs")
        if (args.batch_size != config.train.align.batch_size or args.lr != config.train.align.lr
                or args.graph_cache_size != config.train.align.graph_cache_size
                or (not fingerprint_model and args.mol_norm_type != config.model.mol_encoder.norm_type)
                or args.mol_norm_eps != config.model.mol_encoder.norm_eps
                or args.seed != config.fulltrain.v15.alignment_seed):
            parser.error("Formal alignment hyperparameters must match the pinned configuration")
        if args.exclude_val_query_indices != config.fulltrain.exclude_val_query_indices:
            parser.error("Formal alignment validation exclusions differ from the audited protocol")
        if not args.device.startswith("cuda:") or os.environ.get("SPECEMBEDDING_REQUIRE_CUDA") != "1":
            parser.error("Formal alignment requires the strict GPU runner with explicit cuda:N")
        if Path(args.save_dir).exists():
            parser.error("Formal alignment requires a new output directory")

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    setup_logging(save_path / "align_train.log")
    startup_logging(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    training_candidates = candidate_input_receipt = candidate_input_fingerprint = None
    if args.candidate_training_input:
        training_candidates, candidate_input_receipt, candidate_input_fingerprint = read_candidate_training_input(
            args.candidate_training_input, args.data_path, candidate_settings,
            config.fulltrain.expected_counts.to_dict(), args.exclude_val_query_indices,
        )
    retrieval_validator = None
    graph_receipt = None
    spectrum_metadata = metadata_receipts = None
    spectrum_targets = None
    if args.spectrum_target_cache:
        spectrum_targets = load_spectrum_target_cache(args.data_path, args.spectrum_target_cache,
            config.fulltrain.expected_counts.to_dict(), args.exclude_val_query_indices,
            config.data.tokenizer.to_dict(), auxiliary_settings['target'])
    if args.spectrum_metadata_cache:
        spectrum_metadata, metadata_receipts = load_spectrum_metadata(
            args.data_path, args.spectrum_metadata_cache, counts=config.fulltrain.expected_counts.to_dict(),
            exclusions=args.exclude_val_query_indices, tokenizer_config=config.data.tokenizer.to_dict(), settings=adduct_settings)
    metadata_extra = {} if spectrum_metadata is None else {'spectrum_metadata': spectrum_metadata['val']}
    fingerprint_inputs = fingerprint_smiles = None
    if uses_fingerprints:
        fingerprint_inputs, fingerprint_smiles = {}, {}
        for split, index_path, cache_path in (('train', args.fingerprint_training_index, args.training_fingerprint_cache),
                                               ('validation', args.validation_index, args.validation_fingerprint_cache)):
            fingerprint_smiles[split], fingerprint_inputs[split] = load_alignment_fingerprints(
                args.data_path, index_path, split, cache_path, counts=config.fulltrain.expected_counts.to_dict(),
                exclusions=args.exclude_val_query_indices, tokenizer_config=config.data.tokenizer.to_dict(),
                settings=config.molecule_fingerprints.to_dict(), pool_cache_size=candidate_settings['pool_cache_size'])
    if args.validation_index:
        index = load_validation_index(args.validation_index, args.data_path, config.fulltrain.expected_counts.to_dict(),
                                      args.exclude_val_query_indices, config.data.tokenizer.to_dict())
        graph_cache = None
        if args.validation_graph_cache:
            graph_cache, graph_receipt = load_validation_graph_cache(args.validation_index, index, args.validation_graph_cache)
        if uses_fingerprints:
            receipt = fingerprint_inputs['validation']['cache']
            validator_class = FingerprintRetrievalValidator if fingerprint_model else GraphFingerprintRetrievalValidator
            retrieval_validator = validator_class(
                index, config.retrieval_validation, save_path / 'validation_retrieval',
                fingerprint_root=receipt['directory'], fingerprint_provenance=receipt['provenance'],
                index_sha256=sha256_file(args.validation_index),
                **metadata_extra,
                **({} if fingerprint_model else {'graph_cache': graph_cache}))
        else:
            retrieval_validator = AlignmentRetrievalValidator(index, config.retrieval_validation,
                                                            save_path / "validation_retrieval", graph_cache=graph_cache,
                                                            **metadata_extra)

    fulltrain_audit = None
    if args.formal_fulltrain:
        from SpecEmbedding.data.tokenizer import Tokenizer
        classified_data, fulltrain_audit = classified_full_spectra(
            args.data_path, config.fulltrain.expected_counts.to_dict(), args.exclude_val_query_indices,
            Tokenizer(**config.data.tokenizer.to_dict()),
        )
    else:
        classified_data = get_classified_data(
            dataset_type=args.dataset_type,
            data_path=args.data_path,
            cache_path=args.cache_path,
            cache_file=args.tokenset_cache,
        )
    exclusion_report = {
        "query_indices": [],
        "smiles": [],
        "keys": [],
    }
    if args.formal_fulltrain:
        exclusion_report = fulltrain_audit["validation_exclusion_report"]
    if args.exclude_val_query_indices and not args.formal_fulltrain:
        provider = get_provider(args.dataset_type, args.data_path)
        val_raw = provider.load_data(mode="val")
        classified_data, exclusion_report = filter_classified_validation(
            classified_data,
            val_raw,
            args.exclude_val_query_indices,
        )
        logging.info("Validation exclusion report: %s", exclusion_report)
    train_data = classified_data['train_data']
    train_keys = classified_data['train_keys']
    val_data = classified_data['val_data']
    val_keys = classified_data['val_keys']

    # 检查预训练权重
    pretrained_spec_path = args.pretrained_spec
    if pretrained_spec_path and not os.path.exists(pretrained_spec_path):
        logging.warning(f"Pretrained SpecEmbedding not found at {pretrained_spec_path}.")
        logging.warning("Training will start from scratch (random initialization) for the MS Encoder.")
        pretrained_spec_path = None
    elif not pretrained_spec_path:
        logging.warning("No --pretrained_spec provided. MS Encoder will train from scratch.")

    if pretrained_spec_path:
        logging.info(f"Loading pretrained SpecEmbedding from {pretrained_spec_path}")
        spec_encoder = SiameseModel(
            embedding_dim=config.model.spec_encoder.embedding_dim,
            n_head=config.model.spec_encoder.n_head,
            n_layer=config.model.spec_encoder.n_layer,
            dim_feedward=config.model.spec_encoder.dim_feedward,
            dim_target=config.model.spec_encoder.dim_target,
            feedward_activation=config.model.spec_encoder.feedward_activation
        )
        spec_encoder.load_state_dict(torch.load(pretrained_spec_path, map_location=device))
    else:
        spec_encoder = None

    # 启动两阶段对齐训练流水线
    logging.info("\nStarting the two-stage cross-modal alignment training pipeline...")

    final_model = train_align(
        train_data=train_data,
        train_keys=train_keys,
        val_data=val_data,
        val_keys=val_keys,
        spec_encoder=spec_encoder,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        graph_cache_size=args.graph_cache_size,
        mol_norm_type=args.mol_norm_type,
        mol_norm_eps=args.mol_norm_eps,
        device=device,
        seed=args.seed,
        formal_fulltrain=args.formal_fulltrain,
        retrieval_validator=retrieval_validator,
        training_candidates=training_candidates,
        candidate_input_receipt=candidate_input_receipt,
        fingerprint_inputs=fingerprint_inputs,
        fingerprint_smiles=fingerprint_smiles,
        spectrum_metadata=spectrum_metadata,
        spectrum_targets=spectrum_targets,
        selection_metadata={
            "dataset_type": args.dataset_type,
            "candidate_training_input": candidate_input_fingerprint,
            "validation_graph_cache": graph_receipt,
            "validation_index": ({"path": str(args.validation_index.resolve()),
                                  "sha256": sha256_file(args.validation_index)}
                                 if args.validation_index else None),
            "data_path": str(Path(args.data_path).resolve()),
            "tokenset_cache": (
                str(Path(args.tokenset_cache).resolve())
                if args.tokenset_cache
                else None
            ),
            "seed": args.seed,
            "exclude_val_query_indices": args.exclude_val_query_indices,
            "validation_exclusion_report": exclusion_report,
            "fulltrain_audit": fulltrain_audit,
            **({'training_fingerprint_cache': fingerprint_inputs['train']['cache'],
                'validation_fingerprint_cache': fingerprint_inputs['validation']['cache']} if uses_fingerprints else {}),
        },
    )

    if args.validation_graph_cache:
        _, final_graph_receipt = load_validation_graph_cache(args.validation_index, index, args.validation_graph_cache)
        if final_graph_receipt != graph_receipt:
            raise ValueError("Validation graph cache changed during training")
    if uses_fingerprints:
        for split, index_path, cache_path in (('train', args.fingerprint_training_index, args.training_fingerprint_cache),
                                               ('validation', args.validation_index, args.validation_fingerprint_cache)):
            _, verified = load_alignment_fingerprints(
                args.data_path, index_path, split, cache_path, counts=config.fulltrain.expected_counts.to_dict(),
                exclusions=args.exclude_val_query_indices, tokenizer_config=config.data.tokenizer.to_dict(),
                settings=config.molecule_fingerprints.to_dict(), pool_cache_size=candidate_settings['pool_cache_size'])
            if verified != fingerprint_inputs[split]:
                raise ValueError('Fingerprint inputs changed during training')
    if spectrum_metadata is not None:
        _, observed = load_spectrum_metadata(args.data_path, args.spectrum_metadata_cache,
            counts=config.fulltrain.expected_counts.to_dict(), exclusions=args.exclude_val_query_indices,
            tokenizer_config=config.data.tokenizer.to_dict(), settings=adduct_settings, index=index)
        if observed != metadata_receipts:
            raise ValueError('Adduct inputs changed during training')
    if spectrum_targets is not None:
        observed = load_spectrum_target_cache(args.data_path, args.spectrum_target_cache,
            config.fulltrain.expected_counts.to_dict(), args.exclude_val_query_indices,
            config.data.tokenizer.to_dict(), auxiliary_settings['target'])
        if observed.provenance != spectrum_targets.provenance:
            raise ValueError('Spectrum auxiliary targets changed during training')

    logging.info("\nTraining complete! The final aligned model is returned and ready for evaluation/inference.")

    # 最终保存
    final_model_path = os.path.join(args.save_dir, "final_aligned_model.pth")
    torch.save(final_model.state_dict(), final_model_path)
    logging.info(f"Saved final end-to-end aligned model to {final_model_path}")

if __name__ == "__main__":
    main()
