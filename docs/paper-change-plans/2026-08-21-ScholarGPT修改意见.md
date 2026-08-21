总体判断是：**论文已经具备较完整的研究闭环，实验记录和局限性披露尤其认真，但当前最大问题不是“实验做得不够多”，而是“主创新点没有被凝练成一个足够有力、可验证的学术命题”**。论文自己实际上也承认，最稳定的证据支持的是“supervised non-generative second-stage reranking”，而不是 Transformer/self-attention 或某个具体组件的独立贡献。

## 对论文的修改意见

这篇论文围绕 MS/MS 分子结构检索中的候选分子重排序问题，提出了 SpecEmbedding-Rerank 两阶段框架：首先学习质谱—分子的跨模态表示并完成候选检索，然后在 top-40 候选集合上进行监督式重排序。论文分别考察了 pointwise reranker 和 candidate-set Transformer，并进行了多随机种子、跨 alignment checkpoint 以及组件移除实验。整体来看，论文工作量较充实，实验记录较完整，作者对于评价协议差异、训练阶段 positive forcing、数据重叠、参数量不匹配以及随机种子稳定性等问题都有主动披露，这一点是值得肯定的。尤其是没有为了强化结论而回避负面的消融结果，说明目前论文在科研真实性和结果报告方面是比较谨慎的。

但是，从正式投稿的角度，我认为目前版本仍然存在几个比较关键的问题，需要进一步修改。

**第一，论文的核心创新点需要重新定位和收紧。** 当前方法包含 cross-modal alignment、explicit interaction features、base score/rank prior、residual scoring、listwise/pairwise loss 以及 candidate-set Transformer 等多个组成部分，看起来方法较丰富，但真正经过实验稳定支持的创新结论其实比较有限。论文自己的跨 alignment 实验已经表明，Transformer 相对于 pointwise reranker 的优势并不稳定；同时，消融实验中去掉 rank embedding 后反而在两个 candidate pool、三个 reranker seed 上均取得更好的 Recall@1 和 MRR。

因此，建议不要继续把“candidate interaction/self-attention”作为论文最主要的方法创新来强调。更合适的论文主线应该是：

> **在 MS/MS candidate retrieval 中，显式的监督式 second-stage learning-to-rank 本身就能够显著改善固定 embedding retriever 的排序质量，而这一收益并不依赖生成式 molecular decoder，也不主要来自 candidate self-attention。**

实际上，从你们自己的结果来看，这个结论反而更加扎实。Pointwise reranker 已经获得了 Transformer 总体提升中的约 95%，说明真正值得研究的问题可能不是“Transformer 能不能提升排序”，而是**为什么简单的 supervised reranking 可以显著修正 cosine retrieval**。这一点如果讲清楚，论文的学术逻辑会比现在更加集中。

**第二，需要进一步回答“reranker 到底学到了什么”。** 这是目前论文最重要但尚未充分回答的问题。现有结果证明 reranking 有效，但没有充分证明模型利用了哪些新的化学或谱图信息完成 reranking。因为 reranker 输入主要来自已经冻结的 spectrum embedding、molecule embedding、base cosine score 及其派生特征，所以审稿人很容易提出一个问题：模型究竟学习到了新的 spectrum–molecule compatibility function，还是主要在学习 base score distribution 的非线性校正？

尤其值得注意的是，去掉 rank embedding 后性能反而显著提高，这实际上是一个非常有价值的发现，而不应该只作为 limitation 处理。建议把这一现象提升为正文中的重要分析对象。可以进一步比较：

* 仅使用 (z_s,z_m)；
* 仅使用 (b_i)；
* 使用 (z_s,z_m,b_i)；
* 使用 interaction features；
* 使用 rank；
* 使用全部信息。

如果“仅 base score + MLP”就能获得相当比例的提升，那么论文需要坦率地解释这是 score calibration/nonlinear score transformation 的作用；如果 embedding interaction 带来了明显额外提升，则能够更有力地证明 reranker 确实学到了 cosine similarity 之外的匹配关系。

**第三，必须尽可能解决评价协议与官方 MassSpecGym evaluator 不一致的问题。** 目前论文使用 exact-target-SMILES single-positive protocol，而官方参考实现采用二维 InChIKey identity，并可能存在 multiple positives。论文已经非常明确地披露这一差异，这在真实性方面是加分项，但仅仅披露还不够。

这是当前论文对外部可比性影响最大的技术问题之一。建议至少补做一次 official evaluator 或与其完全一致的 identity/canonicalization evaluation。如果计算资源允许，主表最好同时给出：

**Local exact-SMILES evaluation + official-compatible evaluation。**

这样可以直接回答“结果是否只是评价规则导致”的质疑。如果确实无法重跑，也至少应该统计测试集中有多少 query 因二维 InChIKey 合并而产生 multiple positives，以及这种 identity difference 对 R@1/MRR 的可能影响范围。否则论文很难与 JESTR、GLMR 以及未来工作形成可靠的数值比较。

**第四，当前 external baseline 明显不足，这是投稿时比较容易被审稿人抓住的问题。** 现在 JESTR 和 GLMR 只作为 reported-only background，并没有在相同 candidate pool、identity rule 和 evaluator 下复现。作者对此进行了充分披露，因此不存在夸大比较的问题，但从实验完整性角度仍然不足。

建议至少加入一个具有代表性的、能够在统一 protocol 下运行的外部 retrieval baseline。如果 JESTR/GLMR 的完整复现确实困难，那么至少应该选择公开代码中最容易统一候选集合的模型进行 matched evaluation。否则现在实验主要回答的是：

> “reranker 比自己的 base retriever 好。”

这个结论成立，但对于一篇方法论文而言说服力仍然偏弱。更强的问题应该是：

> “supervised reranking 是否能够稳定改善不同类型的 pretrained/base retrievers？”

如果能够把 reranker 接到两个不同 base retriever 上，即使不增加复杂的新模型，论文价值都会明显提高，因为这可以证明方法具有一定的 **retriever-agnostic generality**。

**第五，建议增加 top-K 敏感性实验。** 当前 (K=40) 是固定的 design/compute-budget choice，论文也明确承认没有比较其他 truncation depth。 但对于 second-stage reranking 来说，K 实际上是一个核心变量，因为它同时决定了 first-stage upper bound (U_K)、reranker 难度和 Transformer 计算复杂度。

至少建议测试 (K=20,40,80)，如果计算允许再增加 (K=100) 或 full candidate pool。这样能够回答两个重要问题：第一，reranker 的收益是否依赖 K=40 这一特定设置；第二，当 candidate recall 提升时，reranker 是否仍然能够有效利用额外候选。这个实验的学术价值比再增加几个随机种子更高。

**第六，Transformer 与 pointwise 的比较需要做 capacity control。** 当前两者参数量分别约为 1.48M 和 7.78M，相差超过 5 倍，因此即使 canonical checkpoint 上 Transformer 更好，也不能将差异归因于 candidate interaction。论文已经主动承认这一 confounding，这是正确的，但最好进一步用实验解决，而不仅仅写进 limitation。

建议增加一个 parameter-matched pointwise MLP，例如增加 hidden width/depth，使参数量接近 candidate-set Transformer。如果 parameter-matched pointwise 仍然明显落后，才能更有力地说明 cross-candidate interaction 的价值；如果差距消失，则应该进一步弱化 Transformer contribution，把论文完全转向 supervised reranking framework。

**第七，建议补充统计检验和 query-level paired analysis。** 目前大量结果采用 3 个 seeds 的 mean ± sample SD，这能够描述随机性，但样本量过小，不能支撑稳定性或显著性结论，论文对此也进行了正确说明。

不过，这里其实不一定需要大量增加训练 seed。因为所有模型都在相同的 17,556 个 test queries 上评估，可以采用 query-level paired bootstrap，对 R@1、MRR difference 给出 bootstrap confidence interval。这样尤其适合检验 Transformer 与 pointwise 之间只有 0.x–1.x percentage point 的差异。这个补充成本不高，但能显著提高结果报告的规范程度。

**第八，论文需要减少“防御性说明”的篇幅，把重要问题转化成实验结果。** 当前稿件对于“不等价于 official evaluator”“不是 confidence interval”“不能证明 causal component benefit”“不是 SOTA comparison”等说明非常频繁。这些说明本身都是正确的，也体现了作者对真实性的重视，但是出现次数过多后，会削弱论文的主线。

建议保留一次清晰、集中的 protocol statement，并在 Limitations 中统一总结。正文结果部分更多回答三个问题即可：

1. supervised reranking 是否稳定改善 base retrieval；
2. candidate interaction 是否提供额外收益；
3. 哪些输入和训练组件真正贡献性能。

这样论文会更加简洁、有力量。

## 关于创新性的具体判断

如果按照目前版本直接评价，我认为创新性属于**“问题组合和实验验证具有一定新意，但单个技术组件创新有限”**。

Residual ranking、listwise loss、pairwise ranking、Set Transformer、cross-modal contrastive learning 本身都不是新的；而 MS/MS candidate reranking 也已有 MetFusion、MS2Query、LC-MS2Struct 等相关思路。论文真正有价值的地方，是把**固定的 spectrum–molecule retriever 与直接的、非生成式、监督式 top-K reranking 系统结合起来，并在 MassSpecGym 环境下系统研究其行为**。论文自身也明确指出，“the contribution is neither reranking nor self-attention itself”，这个定位是基本正确的。

因此不要试图通过增加网络结构复杂度来“制造创新”。更建议把创新表述为**研究发现和系统性证据**：cosine similarity 并不是 cross-modal embedding 的最佳最终排序函数；简单 supervised second-stage reranking 就可以恢复相当一部分 top-K 内部的 ranking headroom；candidate-level self-attention 的额外收益则远没有最初假设的稳定。

这个故事实际上比“我们提出了一个 Transformer reranker”更可信。

## 关于真实性和科研规范的评价

这一部分是目前稿件相对较强的地方。论文主动报告了 cross-split spectrum-record overlap、training-only positive forcing 所产生的 label-correlated rank shortcut、local evaluator 与 reference evaluator 的 identity difference、alignment seeds 跨代码 revision、未公开真实数据 artifacts、历史环境不能保证 bitwise reproduction，以及 negative ablation results。

尤其是 **no-rank-embedding 优于 full model** 的结果没有被隐藏，而是明确报告出来，这符合规范的科研报告方式。

但“真实性披露充分”和“可复现性充分”是两个概念。目前论文自己说明，anonymous supplement 并不包含 MassSpecGym 数据、candidate pools、trained checkpoints、reranker caches、logs 和 private analysis artifacts，因此 smoke test 无法复现论文中的 scientific metrics。 如果投稿政策允许，建议最终至少公开训练配置、evaluation scripts、candidate construction scripts、checkpoint hashes，以及能够重新生成 caches 的完整流程。否则实验真实性虽然在文字层面较强，但第三方验证能力仍然有限。

## 建议你下一轮修改优先做什么

不建议现在继续大规模扩展模型。优先级应当是：**先解决评价协议 → 做 parameter-matched pointwise → 做 top-K sensitivity → 做 score/embedding 信息来源消融 → 尽可能增加一个 matched external/base retriever → query-level bootstrap CI。**

完成这些以后，再重新写 Abstract、Introduction 和 Conclusion。届时不要再把论文中心放在 candidate-set Transformer，而应该把标题、贡献和结论统一到 **“direct supervised non-generative reranking is the robust finding; candidate interaction is secondary and conditional”** 这一主线上。

如果这些实验能够得到比较清楚的结果，我认为论文会从现在的“一个结构比较复杂但核心贡献略显分散的方法稿”，变成一篇**研究问题明确、结论克制而且实验可信度较高的 retrieval/learning-to-rank 工作**。这会是更适合投稿的版本。

另外，从近期相关研究的方向来看，MassSpecGym 的评价陷阱、数据泄漏、shortcut 和 metric definition 已经成为需要特别谨慎处理的问题，因此你们现在主动进行 protocol audit 的方向是正确的；下一步关键不是增加更多免责声明，而是尽可能把这些风险通过 matched evaluation 实验消掉。

可重点对照的相关工作包括：*Bushuiev et al. (2024), MassSpecGym, NeurIPS*；*Kalia et al. (2025), JESTR, Bioinformatics*；*Zhang et al. (2026), GLMR, AAAI*；*Xiong et al. (2025), SpecEmbedding, Analytical Chemistry*；*Liu et al. (2026), MassSpecGym in the Wild, arXiv*；*de Jonge et al. (2023), MS2Query, Nature Communications*；*Dührkop et al. (2015), CSI:FingerID, PNAS*；以及 *Goldman et al. (2023), MIST, Nature Machine Intelligence*。这些文献与稿件当前的技术定位和评价问题最直接相关。
