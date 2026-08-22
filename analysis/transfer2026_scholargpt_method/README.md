# ScholarGPT 方法重构分析

本目录保存 2026-08-23 外部评审后新方法的配置、训练/评价汇总和形式性质核验。核心模型为 `model_type=relative`：先对完整的最多 256 个候选进行 pointwise coarse scoring，再将 coarse top-40 送入谱图条件的成对相对判别分支。

## 证据边界

- 新训练缓存必须使用 `force_include_positive=false`；训练 query 的正例只在自然进入 256 候选池时参与 reranker loss。
- `base_ranks` 在新模型中只作为缓存兼容字段，不进入任何模型参数路径。
- `p_ij` 由 `g(r_ij)-g(r_ji)` 构造；模型测试覆盖候选置换等变性和有限值。
- 主指标仍需在实际生成的结果 JSON 中标注 local exact-target-SMILES 协议；二维 InChIKey 结果若运行成功，另存为 reference-identity audit，不能直接称 official evaluator。
- 未有真实运行证据的 external baseline、query bootstrap、跨编码器和完整效率对照不得写入论文主结果。

## 推荐运行入口

缓存生成（长任务使用 detached `tmux`）：

```bash
conda run -n specembedding python prepare_rerank_cache.py \
  --checkpoint checkpoints_align/d4c1f70_massspecgym_nopretrain/best_model_stage2.pth \
  --dataset_type massspecgym --data_path /path/to/processed \
  --candidate_type mass --split train --device cuda:0 --topk 256 \
  --no-force_include_positive \
  --save_path rerank_cache/scholargpt_relative_fullpool_mass/massspecgym_mass_train.pt
```

训练和评价命令应把 `--model_type relative`、`--train-k 256`、`--relation-top-k 40`、`--no-rank-embedding`、`--lambda-spec 0.1` 和 seed 显式记录到结果 manifest；初始 smoke/预算受限运行可使用 `--max-train-queries`，但不得冒充全训练集结果。
