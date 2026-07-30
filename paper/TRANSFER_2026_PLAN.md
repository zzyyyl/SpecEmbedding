# SpecEmbedding 转投稿件完成计划

- 最后更新：2026-07-30
- 计划完成日期：2026-08-31
- 转投开发分支：`codex/transfer-2026-0831`

## 1. 目标与完成定义

本轮工作的目标不是清空旧的 `paper/ADMA2026_TODO.md`，而是在 2026-08-31
前形成一份证据闭环、数字可追溯、代码可复演、满足目标 venue 要求的转投稿件。

8 月 31 日的“完成”同时满足：

1. 核心主张由受控实验直接支持，不依赖未复现的外部结果。
2. 主结果覆盖多个 alignment seeds，并正确区分 alignment 与 reranker 两层不确定性。
3. 论文中出现的主要模型组件均有对应消融证据。
4. 所有表格、摘要、正文和结论中的数字可以追溯到冻结实验工件。
5. 目标 venue 的格式、页数、匿名、AI 使用、补充材料和预印本规则均已核对。
6. 匿名代码包可在干净环境中完成安装、测试和最小规模 smoke run。
7. PDF、源文件和补充材料通过身份信息、元数据和本地路径扫描。

## 2. 论文定位与证据边界

### 2.1 默认核心主张

在跨模态基础检索器之上，直接、非生成式的监督残差 learning-to-rank，
能够在 mass-conditioned 和 formula-conditioned 两种候选协议下稳定提高
Top-1 与 MRR。

该主张同时保留以下边界：

- reranker 只能重新排序基础 top-\(K\)，不能恢复候选列表之外的真实分子。
- formula 场景假设分子式已知，不能解释为 unrestricted molecule annotation。
- Pointwise 和 Set Transformer 都是有效的学习式 reranker。
- 当前证据尚未证明 candidate self-attention 具有跨 alignment checkpoint
  的稳健独立收益。
- 未按统一协议复现的 JESTR/GLMR 结果只能标记为 `reported`，不能用于严格
  SOTA 声明。
- 未完成同硬件测量前，不声称相对生成式方法具有实测速度优势。

### 2.2 主张升级门槛

只有在 alignment seeds 42/43/44 下，Set Transformer 相对 Pointwise 的配对
差异方向和量级均保持稳定，才考虑在标题、摘要或贡献中突出
`set-aware reranking`。

如果多个 alignment checkpoints 下差异方向混合，则最终论文继续以
“监督残差重排序有效”为核心结论，并明确候选交互收益尚未确立。

当前标题暂时保持：

> Non-Generative Learning to Rerank for MS/MS-Based Molecule Retrieval

## 3. 冻结基线与新实验 provenance

### 3.1 不可覆盖的既有基线

- canonical 实验代码提交：`a2280d2828ce872da1f69319b49e0ef7f1bed572`
- overlap-clean alignment seed：42
- validation 排除索引：`7686 7687 7688 8464 8465 8466`
- reranker seeds：42、43、44
- candidate types：mass、formula
- rerank top-\(K\)：40
- canonical artifacts：60 项
- manifest：`paper/adma2026_artifact_manifest.json`

2026-07-30 已重新执行冻结校验，60 项 canonical artifacts 全部通过，manifest
与现有工件一致。

旧 checkpoint、cache、日志和 manifest 不原位覆盖。旧结果继续通过：

```bash
python freeze_adma2026_artifacts.py --check
```

进行验证。

### 3.2 新实验规则

所有转投新增实验必须：

- 使用新的 run prefix、输出目录和 manifest。
- 记录完整 Git commit、dirty-worktree 状态、`params.yaml` 哈希、随机种子、
  checkpoint 哈希、candidate 文件哈希和运行命令。
- 明确记录 alignment seed 与 reranker seed，不能只记录一个通用 `seed`。
- validation/test 均不得插入正例；所有官方 test queries 必须进入最终指标。
- 保留失败、重试和恢复状态，不复用 fingerprint 不一致的 attempt。
- 正确性修复后的结果必须使用新的 provenance，不能继续标记为 `a2280d2`。

## 4. 工作优先级

### 4.1 P0：投稿前必须完成

#### A. 多 alignment seeds 的端到端实验

目标是补充 alignment seeds 43、44，并复用现有 seed 42。

每个新增 alignment checkpoint 必须完成：

1. 使用相同 overlap-clean validation 协议训练 alignment。
2. 为 mass/formula 的 train、validation、test 重建 top-40 cache。
3. 运行：

```text
2 candidate types
× 2 reranker variants
× 3 reranker seeds
= 12 experiments per alignment seed
```

新增实验共 24 组。汇总时按 alignment seed 分层报告：

- 每个 alignment 内的 reranker mean/std。
- 每个 alignment 内 Transformer - Pointwise 的配对差异。
- alignment 之间的 base、Pointwise 和 Transformer 变化。
- 不把 9 个 alignment/reranker 组合简单当作完全独立样本。

根据现有日志估算，每个新增 alignment seed：

- alignment 约 38 分钟；
- 两类候选 cache 约 45 分钟；
- 12 个 reranker 在两张 RTX 4090 上约 3.7 小时墙钟时间；
- 总计约 5 小时墙钟时间和约 5 GB 新 cache。

实际排期按两倍缓冲预留。

验收标准：

- alignment seeds 42/43/44 全部有 checkpoint、selection metadata 和 cache hashes。
- 新增 24 组 reranker 全部为 `complete` 且 `errors=[]`。
- 汇总表同时展示 base、Pointwise、Transformer 和分层不确定性。
- 形成是否升级 self-attention 主张的明确结论。

#### B. 核心组件消融

在固定的 overlap-clean alignment/cache 上，对完整 Set Transformer 补充：

- `no_residual`
- `no_base_score`
- `no_rank_embedding`
- `no_interaction_features`
- `listwise_only`

协议：

```text
5 ablations
× 2 candidate types
× 3 reranker seeds
= 30 experiments
```

Pointwise 已作为“移除 candidate self-attention”的受控基线，不要求为每个
feature ablation 再重复 Pointwise 全矩阵。

验收标准：

- 所有消融保持相同 cache、数据划分、模型选择指标和训练预算。
- feature ablation 保持 MLP 宽度和参数量可比。
- 主文至少报告 residual/base pathway、rank embedding、interaction features
  和 pairwise loss 四类结论。
- 每个保留在贡献列表中的组件都有实验支撑；否则从贡献中删除。

#### C. 最低限度效率测量

必须测量：

- Base、Pointwise、Set Transformer 的可训练参数量。
- train/validation/test cache 构建时间与磁盘占用。
- 完整测试集 rerank latency。
- 单查询 latency 或吞吐量。
- 峰值 GPU memory。

效率结果只用于描述本方法自身成本。若没有统一硬件与实现，不与 GLMR 做
实测速度优劣声明。

#### D. 论文结果与口径统一

必须统一：

- Recall 与 MRR 使用百分比还是 raw 值。
- 代表性 checkpoint 与多种子均值的展示方式。
- MCES@1 仅计算单个 checkpoint 时的显式标记。
- alignment-level 与 reranker-level uncertainty。
- Base、Pointwise、Transformer 的数据、cache 和候选协议。
- `reported external` 与 `locally reproduced` 的表格位置和标签。

### 4.2 P1：强烈建议完成

#### A. top-\(K\) / upper-bound 分析

统一比较 \(K=20,40,100\)。每个 \(K\) 同时报告：

- base retrieval upper bound；
- 平均实际候选数；
- Recall@1、Recall@5、Recall@20 和 MRR；
- latency、显存和 cache 体积。

旧 checkpoint 下 formula-only 的 K=50/100/256 pilot 仅作历史参考，不能进入
新的受控主表。K=256 仅在时间允许或目标 venue 明确要求时补充。

#### B. 排名迁移与案例分析

至少形成：

- 真实分子 base rank 到 final rank 的迁移分布。
- 按 base rank 分桶的 Top-1/Top-5 增益。
- 少量成功、失败和排序退化案例。
- mass/formula 两种候选协议下的差异解释。

#### C. 外部 baseline 可行性调查

JESTR/GLMR 统一复现必须满足：

- 相同 MassSpecGym split；
- 相同官方候选文件；
- 相同 canonicalization；
- 所有 test queries 计入指标；
- test 不插入正例；
- 相同指标实现。

止损日期为 2026-08-12。若此前无法获得可靠代码、权重或数据处理细节，
停止追逐严格复现，保留独立的 `reported cross-paper context` 表，并从摘要和
结论中删除“领先外部方法多少”的表述。

### 4.3 P2：时间允许时完成

- 候选顺序 shuffle 开/关。
- product 和 absolute-difference 特征的单独消融。
- 有无 SpecEmbedding 预训练；开始前先审计预训练数据与 MassSpecGym 的重叠风险。
- K=256。
- GLMR 同硬件生成推理成本比较。
- 全面重构旧 SpecEmbedding Trainer、eval 和 checkpoint 格式。

## 5. 论文表图计划

建议最终至少包含：

1. 数据与协议表：split 数量、候选池、平均候选数、upper bound、正例协议。
2. 本地主结果表：Base、Pointwise、Transformer，统一指标和 uncertainty。
3. 核心组件消融表。
4. 外部 baseline 表：单独标记 `reported` 或 `reproduced`。
5. 方法架构图：沿用并按目标 venue 调整现有 TikZ 图。
6. 跨 alignment checkpoint 的 Transformer - Pointwise 配对差异图。
7. base rank 到 final rank 的迁移或分桶增益图。
8. 参数量、延迟、显存和 cache 成本小表。

## 6. 工程与复现计划

### 6.1 实验期间

- 冻结工作树只用于复核旧 60 项 canonical artifacts。
- 新实验在独立分支和新 run prefix 中进行。
- 在启动长实验前先完成 CLI dry-run、单元测试和小规模 smoke。
- 新 runner 必须拒绝 dirty worktree，或在显式允许时记录所有变更。
- 不把大规模 cache/checkpoint 提交进 Git。

### 6.2 投稿包

投稿前完成：

- 锁定 Python、PyTorch、PyG、matchms、RDKit、MCES 等直接依赖版本。
- 修正 README、environment 和实际环境之间的版本不一致。
- Ruff、compileall、unit tests 和最小 CLI/config smoke 全绿。
- 使用脱敏配置模板替代机器专属绝对路径。
- 在全新目录完成一次小规模 cache -> train -> eval cold smoke。
- 使用显式 allowlist 构建匿名代码包。

匿名包必须排除：

- `.git`、`.codegraph`
- checkpoint、rerank cache 和原始日志
- 内部 artifact manifest
- hostname、GPU UUID、Git remote、本机用户名和绝对路径
- 未审查的公开项目、Figshare 或演示链接

匿名包验收：

- 解包后测试和 smoke 均通过。
- 身份字符串扫描零命中。
- 包体积满足目标 venue 限制。
- 冻结工作树再次通过 60/60 canonical artifact 校验。

## 7. 时间表与冻结点

| 日期 | 工作 | 里程碑 |
| --- | --- | --- |
| 7/30–8/2 | 确定 venue、锁定主张与实验协议、建立新 provenance | 最晚 8/3 确定 venue |
| 8/3–8/9 | alignment seeds 43/44、cache、端到端矩阵、核心消融 | 8/9 决定 self-attention 主张 |
| 8/10–8/16 | top-\(K\)、效率、排名迁移、baseline 可行性 | 8/12 baseline 止损；8/16 实验冻结 |
| 8/17–8/23 | 重写摘要、贡献、结果、限制、结论；更新表图 | 8/23 内容冻结 |
| 8/24–8/28 | venue 模板、环境复演、匿名包、独立技术审读 | 8/28 代码与补充材料冻结 |
| 8/29–8/31 | 数字反查、英文终审、PDF/身份/合规检查 | 8/31 完成可投稿版本 |

8 月 16 日之后只允许：

- 修复明确的正确性错误；
- 补齐失败或缺失的既定实验；
- 重新生成由已冻结数据直接导出的表图。

不再新增方法、扩大实验矩阵或改变核心主张。

## 8. 风险与止损规则

| 风险 | 处理 |
| --- | --- |
| Transformer 优势跨 alignment seeds 不稳定 | 回退到 residual reranking 核心主张 |
| 外部 baseline 无法可靠复现 | 8/12 后转为 reported-only |
| top-\(K\) 或扩展消融挤压写作时间 | 优先保留多 alignment seeds 与核心组件消融 |
| 新代码影响旧工件校验 | 使用新 provenance；旧工作树只做冻结验证 |
| 依赖或路径无法在新环境复现 | 8/24 前完成 cold-install/cold-smoke，不留到最终三天 |
| venue 页数不足 | 优先保留主结果、核心消融、限制；扩展结果移入补充材料 |
| 8/16 后仍有实验未完成 | 删除对应主张，而不是推迟内容冻结 |

## 9. 最终验收清单

### 科学结果

- [ ] alignment seeds 42/43/44 均完成并有完整 provenance。
- [ ] 新增 24 组端到端 reranker 实验全部完成。
- [ ] 五类核心消融全部完成。
- [ ] top-\(K\)、效率和排名迁移达到最终保留范围。
- [ ] val/test 无正例插入，所有 test queries 进入指标。
- [ ] uncertainty 作用域准确。
- [ ] 所有论文数字可追溯到冻结 artifact。

### 稿件

- [ ] 目标 venue 已确定，模板和规则已核对。
- [ ] 摘要、正文、表格、限制和结论数字一致。
- [ ] 每项贡献都有实验支撑。
- [ ] external reported 与 local reproduced 明确分开。
- [ ] 引用、DOI、作者、年份经过人工核对。
- [ ] 不包含旧 ADMA 日期、占位文本或待办式表述。
- [ ] 完成人工英文润色和独立技术审读。

### 合规与工件

- [ ] 环境可从零重建。
- [ ] lint、compile、unit tests、cold smoke 全部通过。
- [ ] 匿名包由 allowlist 构建并独立解包复演。
- [ ] PDF/source/archive 身份信息扫描零命中。
- [ ] 作者、COI、基金、既有公开版本和 AI 披露由作者最终确认。
- [ ] 最终代码、论文和工件 checksum 已冻结。

## 10. 下一步执行顺序

本计划提交后，按以下顺序开始实现：

1. 为 alignment 训练增加显式 seed 参数，并在 selection metadata 中记录。
2. 增加多 alignment-seed 端到端 runner、dry-run 和测试。
3. 使用新 run prefix 完成 seed 43 的小规模 smoke。
4. smoke 验收后启动 seeds 43/44 正式实验。
5. GPU 实验运行期间并行完成核心消融编排与 venue 规则矩阵。
