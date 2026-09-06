# SpecEmbedding 项目记忆

最后更新：2026-09-07

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
- 当前 canonical 官方兼容评价复用 official retrieval JSON 顺序、二维 InChIKey 身份规则和 alignment-42 的保存嵌入；没有重新运行 official loader、谱图/分子编码或 alignment 重训。另有 cache-local 的 alignment-42/43/44 敏感性矩阵，不能与 canonical 表混称为完整 official evaluator 重跑。
- local exact-target-SMILES 只作为敏感性视图。训练、验证和测试的候选覆盖、正例 forcing 和 miss 处理必须以 cache 元数据为准。
- JESTR/GLMR 只作为外部机制背景；本仓库的 JESTR-style cosine 是保存嵌入上的 reimplemented control，不是官方 JESTR encoder/checkpoint 复现，GLMR 仍为 reported-only，不做未经协议匹配的性能比较。

当前可辩护的论文主张是：在固定跨模态检索器已召回的候选列表内，监督式、非生成式 residual learning-to-rank 可以改善 Recall/MRR；在三组 alignment checkpoint 的描述性 cache-local 矩阵中，pointwise 在主要 R@1/MRR 汇总略高于 relative。36 个主矩阵运行使用最多 20,000 条训练 query、alignment seeds 42--44 和 reranker seeds 42--44；另有 30 个 relative 消融运行。不能把增益归因于 candidate self-attention 或 relative 模块的独立普遍收益，也不能据此声称 SOTA、统计显著性、端到端稳定性或部署 latency。

论文当前采用两部分科学叙事：监督式第二阶段重排序在固定候选池内有效；relative candidate interaction 的结构性质得到形式化和消融，但其相对 pointwise 的独立收益尚未被当前多 checkpoint 证据建立。

## 4. 运行与验证

正式实验的训练集规则（用户于 2026-09-07 明确要求）：**不要在正式实验时限制训练集**。
alignment、reranker 以及用于论文结论的对照/消融均须使用完整训练划分中符合既定协议的全部
可训练样本，不得为节省时间或资源而自行设置样本数量上限、截取前 N 条或缩小为抽样子集。
验证集和测试集保持独立，不并入训练集。限量仅可用于明确标注的非正式调试或 smoke，结果
不得替代正式实验；正式运行前后应核验实际训练样本数与完整可训练样本集一致。

正式论文实验的训练设备规则：alignment 与 reranker 训练严格使用显式 `cuda:N`，启动前检查
GPU 空闲显存和利用率；GPU 高负载或显存不足时等待，不切换到 CPU 训练。CPU 只用于 synthetic
测试、数据处理、审计和明确标注为非论文结果的实现 smoke，不能产生正式 checkpoint、模型选择
指标或论文结果。被中断的 CPU 训练工件必须隔离并明确排除。

外部复现工件使用以下固定存储约定，不下载到本仓库工作树：

- 外部代码仓库放在 `/data1/zyl/repos/<repo_name>/`；
- 外部模型的数据、预训练权重及配套推理工件放在 `/data1/zyl/<model_name>/`；
- 例如 GLACIER 的代码放在 `/data1/zyl/repos/ms-pred/`，数据与权重放在
  `/data1/zyl/GLACIER/`。下载前记录来源 URL、版本或 commit、文件校验值和取得日期。

```bash
conda activate specembedding
python -m pytest -q
ruff check .
python -m compileall -q .
bash paper/build_release.sh
```

最近一次发布验收的完整结果、匿名扫描、解包测试和 smoke 记录在
`reproducibility/anonymous-supplement-validation.yaml`；36+30 实验索引在
`reproducibility/mentor2026_experiment_index.json`；双语 PDF 与构建信息在
`paper/release/`；实验原始结果和分析工件在 `analysis/`。不要把这里的摘要替代为日志或工件核验。

匿名补充包为由显式 allowlist 构建的当前发布工件，位于 `dist/specembedding-anonymous-supplement.zip` 并已纳入 Git 以便交接。它不包含 `.git`、数据、checkpoint、cache、日志和内部 artifact manifest；当前包含 43 个 allowlisted 源文件。

## 5. 论文与待办

- 英文稿：`paper/main.tex`；中文讨论稿：`paper/main_cn.tex`；参考文献：`paper/references.bib`。
- 当前正文与 BibTeX 引用闭包为 24 个键；未使用的 `kretschmer2025coverage` 已删除。
- 当前 release 源稿提交为 `6f27a5f`，英文 PDF 18 页、中文 PDF 15 页；当前代码 HEAD 为 `67cf29f`。仍需按目标 venue 完成最终页数、作者/COI、预印本政策和联合匿名检查。
- 外部强 baseline、candidate-aware alignment、跨数据集验证和完整 official-loader 端到端重跑不在当前证据范围内。
- GLACIER 官方仓库、MassSpecGym checkpoint 和开放评价输入已于 2026-09-07 下载到约定的
  `/data1` 路径，但尚未安装环境或运行推理；来源、SHA-256、已知缺口和后续步骤见
  `analysis/glacier_reproduction_manifest.json` 与 `docs/glacier_reproduction_handoff_zh.md`。

文档修改计划的规则见 `docs/paper-change-plans/README.md`，当前执行计划见
`docs/paper-change-plans/2026-08-23-ChatGPT导师修改执行计划.md`。
