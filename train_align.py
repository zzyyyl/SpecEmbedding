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
from SpecEmbedding.data.datasets_align import AlignGraphDataset, align_collate_fn
from SpecEmbedding.data.datasets_candidates import CandidateAlignDataset, candidate_align_collate_fn
from SpecEmbedding.data.overlap import filter_classified_validation
from SpecEmbedding.models_align import GINEEncoder, SpecMolAlignModel
from SpecEmbedding.trainer.trainer import set_seed
from SpecEmbedding.trainer.trainer_align import TrainerAlign
from SpecEmbedding.trainer.trainer_candidates import CandidateTrainerAlign
from SpecEmbedding.utils.candidate_training import read_candidate_training_input, validate_candidate_settings
from SpecEmbedding.utils.fulltrain import sha256_file
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
    train_dataset = AlignGraphDataset(
        data=train_data,
        keys=train_keys,
        n_views=1,
        is_augment=True,
        graph_cache_size=graph_cache_size,
        full_spectra=formal_fulltrain,
        graph_policy=config.model.mol_encoder.graph_policy,
        **({"augment_config": config.augmentation.to_dict()} if formal_fulltrain else {}),
    )
    val_dataset = AlignGraphDataset(
        data=val_data,
        keys=val_keys,
        n_views=1,
        is_augment=False,
        graph_cache_size=graph_cache_size,
        full_spectra=formal_fulltrain,
        graph_policy=config.model.mol_encoder.graph_policy,
        **({"augment_config": config.augmentation.to_dict()} if formal_fulltrain else {}),
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
        train_dataset = CandidateAlignDataset(
            train_dataset, training_candidates,
            dataset_manifest_sha256=selection_metadata["fulltrain_audit"]["dataset_manifest_sha256"],
            negative_count=candidate_settings["negative_count"], seed=seed,
            graph_cache_size=candidate_settings["graph_cache_size"],
        )
        if (train_dataset.provenance["dataset_to_raw_query_sha256"]
                != candidate_input_receipt["dataset_to_raw_query_sha256"]):
            raise ValueError("Actual training query order differs from candidate input preflight")
        logging.info("Formal natural candidate supervision: %s", train_dataset.provenance)
    train_loader = DataLoader(
        train_dataset,
        **train_batching,
        collate_fn=candidate_align_collate_fn if candidate_settings["enabled"] else align_collate_fn,
        num_workers=config.train.align.num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=align_collate_fn, 
        num_workers=config.train.align.num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    logging.info("2. 初始化模型...")

    # 实例化新的分子图编码器
    mol_encoder = GINEEncoder(
        emb_dim=config.model.mol_encoder.emb_dim,
        n_layers=config.model.mol_encoder.n_layers,
        dropout_rate=config.model.mol_encoder.dropout_rate,
        size_feature_dim=config.model.mol_encoder.size_feature_dim,
        norm_type=mol_norm_type,
        norm_eps=mol_norm_eps,
    )

    has_pretrained_spec = spec_encoder is not None
    if not spec_encoder:
        spec_encoder = SiameseModel(
            embedding_dim=config.model.spec_encoder.embedding_dim,
            n_head=config.model.spec_encoder.n_head,
            n_layer=config.model.spec_encoder.n_layer,
            dim_feedward=config.model.spec_encoder.dim_feedward,
            dim_target=config.model.spec_encoder.dim_target,
            feedward_activation=config.model.spec_encoder.feedward_activation
        )

    # 实例化双塔对齐模型
    model = SpecMolAlignModel(
        spec_encoder=spec_encoder,
        mol_encoder=mol_encoder,
        spec_dim=config.model.spec_encoder.dim_target,
        hidden_dim=config.model.align.final_dim,
        final_dim=config.model.align.final_dim,
        dropout_rate=config.model.align.dropout_rate,
        tau=config.model.align.tau
    )
    trainer_class = CandidateTrainerAlign if candidate_settings["enabled"] else TrainerAlign
    trainer = trainer_class(
        model,
        train_loader,
        val_loader,
        device,
        save_dir=save_dir,
        retrieval_validator=retrieval_validator,
        record_resources=formal_fulltrain,
        **({"candidate_loss_weight": candidate_settings["loss_weight"]} if candidate_settings["enabled"] else {}),
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
    if config.train.align.metric_for_best not in ("validation_contrastive_loss", "validation_top1_then_mrr"):
        parser.error("Unknown alignment checkpoint-selection metric")
    if (config.train.align.metric_for_best == "validation_top1_then_mrr") != bool(args.validation_index):
        parser.error("Retrieval selection requires --validation-index and the matching pinned metric configuration")
    if args.validation_index and not args.formal_fulltrain:
        parser.error("Retrieval selection requires audited formal full-training data")
    if args.validation_graph_cache and not args.validation_index:
        parser.error("--validation-graph-cache requires --validation-index")
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
                or args.mol_norm_type != config.model.mol_encoder.norm_type
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
    if args.validation_index:
        index = load_validation_index(args.validation_index, args.data_path, config.fulltrain.expected_counts.to_dict(),
                                      args.exclude_val_query_indices, config.data.tokenizer.to_dict())
        graph_cache = None
        if args.validation_graph_cache:
            graph_cache, graph_receipt = load_validation_graph_cache(args.validation_index, index, args.validation_graph_cache)
        retrieval_validator = AlignmentRetrievalValidator(index, config.retrieval_validation,
                                                        save_path / "validation_retrieval", graph_cache=graph_cache)

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
        },
    )

    if args.validation_graph_cache:
        _, final_graph_receipt = load_validation_graph_cache(args.validation_index, index, args.validation_graph_cache)
        if final_graph_receipt != graph_receipt:
            raise ValueError("Validation graph cache changed during training")

    logging.info("\nTraining complete! The final aligned model is returned and ready for evaluation/inference.")

    # 最终保存
    final_model_path = os.path.join(args.save_dir, "final_aligned_model.pth")
    torch.save(final_model.state_dict(), final_model_path)
    logging.info(f"Saved final end-to-end aligned model to {final_model_path}")

if __name__ == "__main__":
    main()
