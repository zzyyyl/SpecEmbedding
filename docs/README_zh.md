# SpecEmbedding 文档入口

SpecEmbedding 是一个 MS/MS-to-molecule retrieval 系统：先用谱图--分子双塔召回候选，再在固定候选池内进行非生成式监督重排序。

本文档只记录当前可运行流程。历史实验、旧方案和逐次修改记录不再放在这里；可核验工件见 `analysis/`、`paper/release/` 和 `reproducibility/`。

## 1. 环境与检查

首选环境为 `specembedding`。环境定义在 `environment.yml`，开发工具在 `requirements-dev.txt`。

```bash
conda env create -f environment.yml
conda activate specembedding
python -m pytest -q
ruff check .
python -m compileall -q .
```

当前运行环境和匿名补充包验收记录分别见 `reproducibility/runtime-snapshot.yaml` 与
`reproducibility/anonymous-supplement-validation.yaml`；三组 alignment 的 36 个主运行和 30 个
relative 消融的索引见 `reproducibility/mentor2026_experiment_index.json`。快照是验收证据，不代表所有历史实验逐包使用同一环境。

## 2. 代码分工

| 路径 | 用途 |
| --- | --- |
| `SpecEmbedding/models.py` | 峰序列、m/z embedding 和谱图 Transformer |
| `SpecEmbedding/models_align.py` | GINE 分子编码器与谱图--分子对齐 |
| `train.py` / `eval.py` | 谱图表征学习与谱图检索 |
| `train_align.py` / `eval_align.py` | 跨模态对齐与基础候选检索 |
| `prepare_rerank_cache.py` | 保存谱图、候选分子 embedding 和 base 分数 |
| `train_rerank.py` / `eval_rerank.py` | 二阶段 reranker 训练与评价 |
| `run_rerank_pipeline.py` | cache、训练和评价的统一编排入口 |
| `SpecEmbedding/utils/` | 模型加载、设备、候选、评价和 rerank 公共逻辑 |

## 3. 当前主流程

```text
MS/MS 谱图
  -> 谱图 Transformer
  -> GINE 分子塔与跨模态对齐
  -> base candidate retrieval
  -> 最多 256 个候选的 coarse-to-fine reranking
```

当前 rerank 主线是无 rank 的 `relative` 模型：先做 pointwise coarse scoring，再对 coarse top-40 做谱图条件的候选相对判别。`pointwise` 是容量匹配对照；旧的 rank-aware Transformer 只用于历史结果复核。

推荐使用统一入口：

```bash
python run_rerank_pipeline.py massspecgym \
  --align_save_dir checkpoints_align/<run> \
  --candidate_type formula \
  --pre_top_k 256 \
  --model_type relative \
  --device cuda:1
```

调试时加 `--limit N`，并确认输出目录带有 `_limitN` 后缀。只打印命令不执行时加 `--dry-run`。

统一入口默认不强制插入正例。只有复核 legacy 训练 cache 时才使用
`--force-include-positive`；该选项不会作用于验证或测试 cache。

## 4. 评价协议

- 数据集：MassSpecGym；官方 structure-disjoint split 不表述为 scaffold-disjoint。
- 候选协议：`mass` 或已知分子式的 `formula`；后者称为 formula-conditioned retrieval。
- reranker 是 closed-library 方法，只能重排输入候选，不能恢复未被 base retrieval 召回的真值。
- 当前主评价使用官方 retrieval JSON 的候选顺序、二维 InChIKey 身份规则和保存的 alignment embedding；这是 official-compatible candidate-order/identity evaluation，不是 fresh loader 重编码或 alignment 重训。
- 训练、验证、测试的 no-forcing 语义和候选覆盖上界必须与 cache 元数据一起核对。
- JESTR/GLMR 是 reported-only 外部背景，未在本仓库统一复现，不与本地数字直接排序比较。

## 5. 论文与发布

```bash
bash paper/build_release.sh
```

发布 PDF、工具链、页数和哈希见 `paper/release/`。当前英文 18 页、中文 15 页。论文正文以方法叙事为主，协议、历史对照和复现边界集中在正文后的附录；双语稿和参考文献以 `paper/` 为准，本文不重复实验表格。

论文的核心发现分为两层：监督式第二阶段重排序改善固定候选池内的排序；relative interaction 的结构性质已形式化，但当前结果尚未证明其独立优于容量匹配的 pointwise 对照。

匿名补充包由显式 allowlist 构建，当前发布包为 `dist/specembedding-anonymous-supplement.zip`，排除 Git 历史、数据、checkpoint、cache、日志和内部 artifact manifest。构建与独立验收记录见 `reproducibility/`。

## 6. 文档地图

- `docs/project_memory_zh.md`：当前工程、实验和证据边界的短版记忆。
- `docs/reranker_solution_zh.md`：当前 reranker 的实现说明与运行方式。
- `docs/paper-change-plans/`：计划模板和当前最终计划；已完成的历史计划不在工作树中重复保存。
- `paper/TRANSFER_2026_PLAN.md`、`paper/ADMA2026_TODO.md`：投稿待办与 venue-specific 检查。
