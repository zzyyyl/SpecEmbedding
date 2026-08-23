# SpecEmbedding 项目记忆

最后更新：2026-08-23

本文档是内部协作的当前状态摘要，不是实验原始记录，也不应放入匿名补充材料。数字、路径和结论以代码、`params.yaml`、原始日志及可校验工件为准。

## 1. 系统与代码

项目流程为：

```text
MS/MS 谱图
  -> Transformer 谱图编码器
  -> GINE 分子编码器与跨模态对齐
  -> base candidate retrieval
  -> closed-library supervised reranking
```

主要入口：

- 谱图编码器：`SpecEmbedding/models.py`、`train.py`、`eval.py`；
- 对齐模型：`SpecEmbedding/models_align.py`、`train_align.py`、`eval_align.py`；
- rerank：`prepare_rerank_cache.py`、`train_rerank.py`、`eval_rerank.py`、`run_rerank_pipeline.py`；
- 公共逻辑：`SpecEmbedding/utils/` 和 `SpecEmbedding/trainer/`；
- 全量测试：`tests/`。

## 2. 当前方法

当前主线是无 rank、无正例 forcing 的 `relative` reranker：

1. 从 base retrieval 保留最多 256 个自然候选；
2. 用谱图--候选 pair features 做 pointwise coarse scoring；
3. 在 coarse top-40 上计算谱图条件的候选相对偏好；
4. 通过反对称构造保证候选对关系分数满足 `p_ij = -p_ji`，并保持候选置换等变性。

`pointwise` 是容量匹配对照。旧的 rank-aware Transformer 和训练期 positive forcing 只属于 legacy 复核，不代表当前论文协议。当前默认入口如下：

```bash
python run_rerank_pipeline.py massspecgym \
  --align_save_dir checkpoints_align/<run> \
  --candidate_type formula \
  --pre_top_k 256 \
  --model_type relative \
  --device cuda:1
```

缺少完整配置时应尽早失败；reranker checkpoint 必须保存构建模型所需的完整配置。所有训练、评价和运行参数统一放在 `params.yaml`，不要把文档中的默认值当作实验结果。

## 3. 数据与评价边界

- 数据集为 MassSpecGym，使用官方 structure-disjoint split；不要写成 scaffold-disjoint。
- `mass` 是质量候选协议；`formula` 假设真实分子式已知，称为 formula-conditioned retrieval。
- reranker 只能重排输入候选，不能生成候选库外分子，也不能救回未被 base top-$K$ 召回的真值。
- 当前官方兼容评价复用 official retrieval JSON 顺序、二维 InChIKey 身份规则和保存的 alignment embedding；没有重新运行 official loader、谱图/分子编码或 alignment 重训。
- local exact-target-SMILES 只作为敏感性视图。训练、验证和测试的候选覆盖、正例 forcing 和 miss 处理必须以 cache 元数据为准。
- JESTR/GLMR 只作为 reported-only 机制背景；不做未经协议匹配的性能比较。

当前可辩护的论文主张是：在固定跨模态检索器已召回的候选列表内，监督式、非生成式 residual learning-to-rank 可以改善 Recall/MRR；当前容量匹配的 pointwise 对照在主要 R@1/MRR 汇总中略高于 relative。不能把增益归因于 candidate self-attention 或 relative 模块的独立普遍收益，也不能据此声称 SOTA、统计显著性、端到端稳定性或部署 latency。

论文当前采用两部分科学叙事：监督式第二阶段重排序在固定候选池内有效；relative candidate interaction 的结构性质值得研究，但其相对 pointwise 的独立收益尚未被当前 pilot 建立。

## 4. 运行与验证

```bash
conda activate specembedding
python -m pytest -q
ruff check .
python -m compileall -q .
bash paper/build_release.sh
```

最近一次发布验收的完整结果、匿名扫描、解包测试和 smoke 记录在
`reproducibility/anonymous-supplement-validation.yaml`；双语 PDF 与构建信息在
`paper/release/`；实验原始结果和分析工件在 `analysis/`。不要把这里的摘要替代为日志或工件核验。

匿名补充包为生成工件，默认位于 `dist/` 且不纳入 Git。它只包含显式 allowlist 源文件，不包含 `.git`、数据、checkpoint、cache、日志和内部 manifest。

## 5. 论文与待办

- 英文稿：`paper/main.tex`；中文讨论稿：`paper/main_cn.tex`；参考文献：`paper/references.bib`。
- 当前正文与 BibTeX 引用闭包为 24 个键；未使用的 `kretschmer2025coverage` 已删除。
- 当前发布记录仍需按目标 venue 完成最终页数、作者/COI、预印本政策和联合匿名检查。
- 外部强 baseline、candidate-aware alignment、跨数据集验证和完整 official-loader 端到端重跑不在当前证据范围内。

文档修改计划的规则见 `docs/paper-change-plans/README.md`，当前执行计划见
`docs/paper-change-plans/2026-08-23-ChatGPT导师修改执行计划.md`。
