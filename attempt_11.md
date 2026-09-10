# A11：下游谱图塔的可学习注意力汇聚

- 状态：**准备中，未启动**。在A08训练、A09等待、A10准备期间预登记；最终parent须等
  A10完整审计后决定，当前没有本轮训练结果。
- 唯一主要假设：将谱图Transformer输出的等权均值改为可学习的加权均值，可能更好地
  汇聚有判别力的上下文峰表示，从而改善完整验证Top-k/MRR。现有self-attention已经能
  传播峰信息，不能据均值汇聚断言它是瓶颈，也不预设该改动优于原模型。

## 依据与固定定义

当前`SiameseModel.forward`在屏蔽padding后，对有效token取均值，再执行原decoder。
[Set Transformer原文](https://proceedings.mlr.press/v97/lee19d/lee19d.pdf)讨论用学习查询
汇聚集合表示，其[补充材料Lemma 1](https://proceedings.mlr.press/v97/lee19d/lee19d-supp.pdf)
说明零查询的点积softmax注意力可表示均值；[Attention-based MIL](https://proceedings.mlr.press/v80/ilse18a.html)
也研究可学习加权聚合。这里只借鉴集合汇聚思路，不复现完整PMA、多头网络或MIL分类器，
不把其它任务的成绩作为质谱检索证据。D17的输入敏感性结果支持继续研究谱图表示，不能
证明汇聚方式有因果收益；A07未晋升也不证明所有谱图改动无效。

固定Transformer后的有效token表示为h_i，维度D；仅新增一个D维可学习向量u：

`score_i = dot(u, LayerNorm(h_i; affine=False, eps=1e-5)) / sqrt(D)`；
对有效token做softmax得到a_i，`pooled = sum_i(a_i * h_i)`，随后保持原decoder、activation
和投影头。LayerNorm只用于计算权重，汇聚的value仍是原h_i。padding权重严格为零；
保留既有母离子token及角色，不额外增加真值分子式、元数据、位置编码或查询—候选交互。

u初始化为全零，不新增随机初始化操作；数学上的起点为原等权均值。浮点求和顺序可带来
舍入差异，须比较初始权重、RNG与数值容差，不能声称逐位等价。D=512时仅新增512个
参数，不增加token数量、Transformer层数或候选编码预算；实际显存及整轮耗时另行测量。
本轮不扫描多个查询头、池化温度、隐层宽度或归一化参数。

## Parent与继承范围

以A10完整独立审计后唯一保留的incumbent为parent；A10未晋升则继承其已审计parent。
若只有未选中Pareto候选达标，先解决选择与checkpoint绑定歧义，不自动挑一个继续。
从随机初始化训练，parent权重仅用于fresh完整验证比较；继承已晋升的GINE/指纹残差、
Q/K归一化、均匀或结构相似采样等设置，只增加上述下游谱图汇聚组件。

保持seed42、batch128、质量block32、学习率1e-4、CE16、100轮上限/patience5、原增强、
优化器、调度和余弦检索。MassSpecGym v1.5完整train194,119、有效val19,423，既定六条
排除与自然候选/二维正例协议保持；不用test选择设置，不提前扩seed矩阵。
SpecEmbedding预训练共享类及其结构保持原样，下游变体独立定义、显式构造并保存配置；
现存来源不足的旧预训练权重继续不准入。

## 缓存、实现与验收

tokenizer、图、Morgan与候选抽样构造均继承parent；来源指纹有效的固定输入缓存可复用。
若继承A10采样，必须保留其相似度缓存和独立重放绑定；若未晋升则不启用该采样。
学习表示随新权重重编码，不复用旧embedding。数据、缓存、临时文件和模型工件均放在
外部项目存储根；固定构造变化时失效重建对应缓存，不删除现有活跃任务引用的文件。

先核验组件初始均值、mask/峰排列不变性、有效峰与padding的梯度、训练/eval路径、两塔
原权重/RNG及严格保存重载；未知/不完整配置必须被正式入口拒绝。完成正式训练、baseline
重编码及独立完成审计接入后，执行真实全量输入条件预检，再绑定最终parent与新固定源码。
原A08/A09/A10源码、队列与派发锁不修改；正式运行继续显式CUDA、GPU0/1择一和逐阶段等待。

完整训练后重算全部验证排名、Top-1/5/10/20和MRR，核验样本数、选优/早停、权重关联及
资源成本。按原整体晋升标准决定保留，未晋升则在attempts.md加删除线；注意力权重至多
作为诊断，不能解释为独立的化学贡献或模型收益。SOTA同协议审计和最终稳定性仍单独完成。

当前结果：**无**。

## 组件准备记录

独立下游组件`SpecEmbedding/models_attention_pool.py`及未激活preset已实现：包装原谱图
编码器，保留其embedding、Transformer、decoder和activation，仅替换最后汇聚；原
SpecEmbedding共享类不修改。现阶段仅支持原SiameseModel及其已安装的Q/K层归一化，
其它forward覆盖显式拒绝，避免绕过额外分支。包装后原谱图state_dict键迁移到`base`下，
新增`pool.query`；这要求后续正式checkpoint构造显式保存新配置，不能直接加载为旧模型。

24项CPU合成检查通过：GINE及GINE+指纹、Q/K开关下的原权重/RNG保持；数学均值起点
的数值核对；独立标量非均匀权重、padding零梯度；学习后排列/padding与train/eval语义；
对比损失梯度到达汇聚和两塔，优化后严格保存重载。正式配置尚未接通，入口拒绝新开关；
这些是组件检查，不是MassSpecGym训练或收益。论文原稿、补充材料和SHA已归档并读回，
来源见优化索引R09及阶段计划。正式构造、完整训练/评价审计及实际全量预检仍待完成。

真实512维构造核验谱图参数由7,888,384增至7,888,896，增加512个，包装不消耗随机数或
初始化CUDA。全仓首次723通过、1跳过、3失败；其中两项相对路径测试受检查环境的全局
`SPECEMBEDDING_STORAGE_ROOT`覆盖影响，仅移除该覆盖后单独重跑两项均通过，测试临时
数据仍在外部TMPDIR。剩余为原四份历史记录的既有路径审计失败，未改生产或测试代码以
规避失败。Ruff、compileall和diff检查通过；原始检查日志与环境修正记录均保留。
