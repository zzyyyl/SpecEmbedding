import argparse
import os
import logging
from pathlib import Path

import torch
import numpy as np

# 导入原来项目里的功能模块
from data_provider import MSPProvider, MassSpecGymProvider
from train import dict_to_spectrum, setup_logging
from SpecEmbedding.data.tokenizer import Tokenizer
from SpecEmbedding.utils.clean import get_classified_tokenset
from SpecEmbedding.type import TokenizerConfig

# 导入咱们新的流水线模块
from SpecEmbedding.run_align_pipeline import run_two_stage_training

def main():
    parser = argparse.ArgumentParser(description="Two-Stage Cross-Modal Alignment Training for SpecEmbedding")
    parser.add_argument("--dataset_type", type=str, choices=["local", "massspecgym"], required=True, help="Dataset type")
    parser.add_argument("--data_path", type=str, help="Path to .msp file (required for local dataset)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_align", help="Directory to save model and logs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for alignment training")
    parser.add_argument("--epochs_stage1", type=int, default=20, help="Number of epochs for Stage 1 (Frozen MS Encoder)")
    parser.add_argument("--epochs_stage2", type=int, default=30, help="Number of epochs for Stage 2 (End-to-End Fine-tuning)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--pretrained_spec", type=str, help="Path to your pre-trained SpecEmbedding model weights")
    
    args = parser.parse_args()
    
    # 初始化日志记录
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    setup_logging(save_path / "align_train.log")
    
    device = args.device
    logging.info(f"Using device: {device}")

    # ---------------------------------------------------------
    # 1. 加载数据 (与原始 train.py 完全一致)
    # ---------------------------------------------------------
    if args.dataset_type == "local":
        if not args.data_path:
            raise ValueError("--data_path is required for local dataset")
        provider = MSPProvider(args.data_path)
        train_raw = provider.load_data(mode='train')
        val_raw = provider.load_data(mode='val')
    else:
        provider = MassSpecGymProvider()
        train_raw = provider.load_data(mode='train')
        val_raw = provider.load_data(mode='val')

    if not train_raw:
        logging.error("No training data loaded. Check your data paths or internet connection.")
        return

    logging.info(f"Loaded {len(train_raw)} train records and {len(val_raw)} val records.")

    # ---------------------------------------------------------
    # 2. Tokenize 处理
    # ---------------------------------------------------------
    tokenizer_config = TokenizerConfig(max_len=100, show_progress_bar=True)
    tokenizer = Tokenizer(**tokenizer_config)
    
    train_spectra = dict_to_spectrum(train_raw)
    val_spectra = dict_to_spectrum(val_raw)
    
    train_sequences = tokenizer.tokenize_sequence(train_spectra)
    val_sequences = tokenizer.tokenize_sequence(val_spectra)

    # ---------------------------------------------------------
    # 3. 按 SMILES 分组，并获取训练/验证的 TokenSet 和 Keys
    # ---------------------------------------------------------
    all_smiles = np.unique([s['smiles'] for s in train_raw] + [s['smiles'] for s in val_raw])
    train_data, train_keys = get_classified_tokenset(all_smiles, train_sequences)
    val_data, val_keys = get_classified_tokenset(all_smiles, val_sequences)

    # ---------------------------------------------------------
    # 4. 检查预训练权重
    # ---------------------------------------------------------
    pretrained_spec_path = args.pretrained_spec
    if pretrained_spec_path and not os.path.exists(pretrained_spec_path):
        logging.warning(f"Pretrained SpecEmbedding not found at {pretrained_spec_path}.")
        logging.warning("Training will start from scratch (random initialization) for the MS Encoder.")
        pretrained_spec_path = None
    elif not pretrained_spec_path:
        logging.warning("No --pretrained_spec provided. MS Encoder will train from scratch.")

    # ---------------------------------------------------------
    # 5. 启动两阶段对齐训练流水线
    # ---------------------------------------------------------
    logging.info("\nStarting the two-stage cross-modal alignment training pipeline...")
    
    final_model = run_two_stage_training(
        train_data=train_data,
        train_keys=train_keys,
        val_data=val_data,
        val_keys=val_keys,
        pretrained_spec_path=pretrained_spec_path,
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        lr=args.lr,
        device_name=device,
        save_dir=args.save_dir
    )
    
    logging.info("\nTraining complete! The final aligned model is returned and ready for evaluation/inference.")
    
    # 最终保存
    final_model_path = os.path.join(args.save_dir, "final_aligned_model.pth")
    torch.save(final_model.state_dict(), final_model_path)
    logging.info(f"Saved final end-to-end aligned model to {final_model_path}")

if __name__ == "__main__":
    main()
