# 当前 reranker 实现说明

本文档只说明当前实现，不保留已废弃的 rank-aware 设计参数和历史预期结果。

## 1. 目标与边界

reranker 接收固定候选池、谱图 embedding、候选分子 embedding 和 base score，在候选池内重新排序。它不生成 SMILES，不访问候选池之外的分子，也不能恢复未被 base retrieval 召回的真值。

```text
MS/MS -> aligned spectrum embedding
      -> base candidate retrieval (最多 256)
      -> pointwise coarse scoring
      -> relative refinement (coarse top-40)
```

## 2. 当前模型

当前主模型为 `relative`，使用无 rank 的绝对分支和候选对关系分支。

对候选 $m_i$，绝对特征包含：

```text
[spectrum_embedding,
 molecule_embedding,
 spectrum * molecule,
 abs(spectrum - molecule),
 base_score]
```

不输入 base rank 或 rank embedding。pointwise 分支根据上述特征给出 coarse score；relative 分支进一步对候选对构造谱图条件关系，并用

```text
pair_preference(i, j) = g(i, j) - g(j, i)
```

形成反对称偏好。候选顺序变化时，输出按同样顺序置换。relation branch 只在 coarse top-40 上运行，以限制 pairwise 计算规模。

模型配置中的 `pair_mode` 显式取 `antisymmetric` 或 `directed`。前者是当前默认方法；后者只用于普通 directed pair control。旧 checkpoint 若没有该字段，则由 `use_antisymmetric` 推断，不改变旧模型语义。

`pointwise` 是容量匹配对照；`transformer` 是 legacy 兼容模型，不是当前论文主线。完整超参数以 `params.yaml` 和 checkpoint 配置为准。

## 3. Cache 与正例协议

当前评价使用自然进入候选池的样本：

- train、val、test 默认均为 `force_include_positive=false`；
- 正例未进入候选池时，验证/测试记为 coverage miss；
- 所有 query 都纳入指标，并报告 pre-retrieval recall upper bound；
- `--force-include-positive` 只用于复核 legacy 训练 cache，且只允许作用于 train。

候选类型为 `mass` 或 `formula`。`formula` 是已知分子式条件下的检索，不是完全开放库检索。

## 4. 推荐运行方式

统一入口：

```bash
python run_rerank_pipeline.py massspecgym \
  --align_save_dir checkpoints_align/<run> \
  --candidate_type mass \
  --pre_top_k 256 \
  --model_type relative \
  --device cuda:1
```

分阶段运行：

```bash
python run_rerank_pipeline.py massspecgym --mode prepare \
  --align_save_dir checkpoints_align/<run> --candidate_type mass \
  --pre_top_k 256 --device cuda:1

python run_rerank_pipeline.py massspecgym --mode train \
  --align_save_dir checkpoints_align/<run> --candidate_type mass \
  --pre_top_k 256 --model_type relative --device cuda:1

python run_rerank_pipeline.py massspecgym --mode eval \
  --align_save_dir checkpoints_align/<run> --candidate_type mass \
  --pre_top_k 256 --model_type relative --device cuda:1
```

调试运行使用 `--limit N`，并检查输出目录后缀，避免覆盖正式工件。使用 `--dry-run` 检查实际命令。默认入口会打印并执行 train/val/test 的 no-forcing cache 命令。

训练 checkpoint 的 `data_summary` 记录实际 labeled train query 数、`max_train_queries`、`train_k`、`pair_mode` 和 cache 的 forcing/top-k 摘要，便于区分 pilot 与 full-split 运行。

固定硬件的候选规模 forward benchmark：

```bash
PYTHONPATH=. python analysis/benchmark_reranker_efficiency.py \
  --cache <test-cache.pt> \
  --checkpoint relative=<relative.pth> \
  --checkpoint pointwise=<pointwise.pth> \
  --candidate-counts 40 80 160 256 \
  --warmup-batches 2 --measure-batches 16 \
  --batch-size 64 --device cuda:0 \
  --output analysis/<run>/efficiency.json
```

该工具只测 forward work，不包含数据读取，也不生成 Recall/MRR 或部署保证。指定 query 的 base/rerank top-k 导出使用 `analysis/export_rerank_cases.py`；导出结果需要结合原始谱峰和结构资料人工解释。

## 5. 代码与验证

- 模型：`SpecEmbedding/models_rerank.py`；
- 数据：`SpecEmbedding/data/datasets_rerank.py`；
- 公共逻辑：`SpecEmbedding/utils/rerank.py`；
- 入口：`prepare_rerank_cache.py`、`train_rerank.py`、`eval_rerank.py`、`run_rerank_pipeline.py`；
- 回归测试：`tests/test_rerank_core.py`、`tests/test_rerank_tools.py`、`tests/test_rerank_pipeline_paths.py`。

至少运行：

```bash
python -m pytest -q tests/test_rerank_core.py tests/test_rerank_pipeline_paths.py
ruff check .
```

## 6. 评价边界

论文主张集中于固定跨模态检索器之后的监督 residual learning-to-rank。relative 分支的等变性和反对称性是实现性质，不等于关系模块具有独立实验收益；静态前向审计也不等于端到端部署 latency。官方兼容结果复用保存 embedding，不能写成完整 official-loader 端到端重跑。
