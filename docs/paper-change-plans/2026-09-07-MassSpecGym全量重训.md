# 论文修改计划：MassSpecGym 全量重训与重新评价

- 状态：`执行中`
- 创建日期：2026-09-07
- 最后更新：2026-09-08
- 负责人：Codex / 作者核验
- 关联论文：`paper/main.tex` / `paper/main_cn.tex`
- 计划约束：用户已授权取消正式训练的 query 数量限制并重新训练、推理；GPU-only，繁忙时等待。
  用户曾要求暂时休息；2026-09-07 明确授权在 GPU 监测工具完成测试后，使用该工具等待并启动本任务。
  2026-09-07 晚间用户进一步授权暂停旧队列、迁移至 v1.5，并在审计证明必要时重训 alignment；
  此授权替代下文历史记录中的“固定旧 alignment、不重训”限制。
  2026-09-07 22 时用户要求停止任务、排查并修复；修复与完整 CPU 验证完成后保持停止。
  2026-09-08 用户明确要求“继续推进目标”，现恢复正式 v1.5 队列，不再等待重复确认。

## 1. 修改目标与动机

以 MassSpecGym v1.5 完整训练划分中符合协议的全部样本重训 alignment 与新版 reranker，
重新评价，建立可核验的新结果。保留官方划分；规范化分子图输入，逐谱遍历正式 alignment 训练集。
旧版模型及 20,000-query 结果只保留历史追溯，不直接代替新版全量结果，不预设性能方向。

## 2. 当前证据与问题定位

- `train_rerank.py` / `run_rerank_multiseed.py` 的限量参数默认 `None`；20,000 是历史运行命令
  显式传入的值，不是硬件自动限制。不删除调试功能，正式入口必须拒绝非空训练数量上限。
- 当前主表对应 `mentor2026_align42_topk40_full_gpu_b32`；已核查 mass/relative/seed42
  checkpoint 和缓存：训练数量 20,000、top-40、训练缓存 forcing=true，测试 forcing=false。
  不以正文的 full-pool/no-forcing 描述替代实际工件证据。
- 新版目标协议为无 rank、全池最多 256 候选、relative coarse top-40，所有 split 不 forcing。
- 已有 alignment-42 checkpoint 位于
  `checkpoints_align/a2280d2_massspecgym_nopretrain_valoverlapclean/best_model_stage2.pth`。
  该 checkpoint 属于 v1 历史协议，仅保留追溯。v1.5 首批重训一个 seed=42、无外部预训练的
  alignment，再固定它运行 2 候选协议 × 2 模型 × 3 reranker seeds。
- 版本审计：当前 processed 文件逐条匹配 v1；GLACIER 独立目录中已有校验过的官方 v1.5 输入。
  v1.5 保留谱图 ID、顺序、划分与峰数据，改写大量 SMILES。旧构图使用 `sanitize=False`，
  芳香性和键类型随输入表示变化，因此本轮不能把旧 alignment 当成经过 v1.5 训练的模型。
- 现有 alignment Dataset 按分子随机抽谱，不保证每轮全谱覆盖；新增正式模式必须逐谱遍历，
  核验训练/验证实际数量，拒绝丢样、非有限损失和不完整 epoch。旧模式保留历史兼容。
- 来源与可校验工件：`reproducibility/mentor2026_experiment_index.json`、原始训练/评价日志及
  `rerank_cache/a2280d2_massspecgym_nopretrain_valoverlapclean_mass_topk40/`。

## 3. 修改范围

### 3.1 涉及文件与章节

- [x] `params.yaml`：独立正式全量运行配置、路径、GPU 等待阈值与 seed 范围。
- [x] 正式编排入口与公共校验：缓存重建、完整性/无 forcing 校验、全量训练、GPU-only 等待、推理和状态记录。
- [x] 相关测试：拒绝限量/旧 forcing 缓存、CUDA 不可用不回退、完整样本数核验、命令和队列状态。
- [ ] 本计划：记录运行路径、来源 commit、验证、进展与结果边界。
- [ ] v1.5 独立预处理与来源/二维身份/候选顺序审计；不覆盖旧 processed 文件。
- [x] 规范分子图构建、alignment 全谱训练与 checkpoint 来源校验及相应测试。
- [x] 将正式队列扩展为新 alignment → 六缓存 → 十二训练/测试，冻结新源码与输入指纹。
- [ ] 新实验全部完成并审计后，才决定如何更新双语稿主结果、协议描述与训练规模说明。

### 3.2 明确不做的事项

- 不变更官方 v1.5 划分或原始候选顺序，不混入验证/测试样本；分子图规范化与全谱遍历是
  本次明确的协议修正，不作为新的模型架构贡献，不扩展为多 alignment seed 矩阵。
- 不把 12 个首批运行说成 36-run 多 alignment 或 30-run 全量消融已完成；其余实验另行安排。
- 不中断其他 GPU 任务，不切换 CPU 训练，不覆盖旧 cache/checkpoint/log，不以减少训练样本解决资源问题。
- 不在结果产生前改写论文数字，不直接复用旧模型高分，不宣称 SOTA 或统计显著性。

## 4. 具体内容设计

1. Mass / Formula 分别新建 train/val/test 缓存：`pre_top_k=256`、`limit=0`、forcing=false，
   使用新训练的 v1.5 alignment-42 重新编码。全量训练集 194,119、验证集 19,429、测试集 17,556 条；
   校验映射/选择/保存数量，缺失与无正例必须显式记录，异常丢样不得静默继续。
2. 新版 relative / pointwise 各训练 seeds 42/43/44，train_k=256，不传 `--max-train-queries`。
   全量指全部符合标签与协议条件的训练样本；checkpoint 记录可训练数、实际数、缓存协议和指纹。
3. reranker 固定验证 MRR 选择模型、最多 30 epochs、patience=5。保留历史约定的 6 条验证记录排除
   `[7686,7687,7688,8464,8465,8466]`，这是预先确定的协议排除，不是训练集限量。
   alignment 采用验证对比损失选择、最多 100 epochs、patience=5，随机初始化两塔；每轮遍历全部
   194,119 条训练谱图和 19,423 条验证谱图，按二维身份定义 multi-positive 标签，验证使用固定
   seed=42 排列。不再沿用旧的“每分子随机抽谱、长度为分子数十倍”训练语义。
4. 对全部测试 query 重新推理，报告 Recall@1/5/10/20、MRR、coverage、三 seed 均值/样本 SD，
   保存每次运行的配置、checkpoint、日志和状态。首批不计算昂贵 MCES，也不把其缺失隐藏。
5. 当前入口标签为 local exact-target-SMILES；新缓存的二维身份/官方候选顺序兼容性必须重新审计
   后才能赋予相应评价名称。本轮不是 fresh official-loader 端到端复现，不继承旧缓存身份审计结论。
6. 用新的独立 detached 源码 worktree 固定运行 commit。v1.5 数据、缓存和工件采用新的版本目录；
   下文旧运行目录及固定 worktree 仅保留历史，禁止原地换版或复用旧派发锁。
7. 后台 `tmux` 队列逐阶段检查 GPU（至少 20,000 MiB 空闲且利用率不高于 10%，间隔 30 秒）；
   GPU 繁忙则等待。CUDA/驱动不可用时报错，不以 CPU 代替。首批单 GPU 顺序执行，避免抢占其他任务。

## 5. 引用文献与真实性核验

本轮暂不修改 BibTeX；版本依据为官方 Hugging Face 数据卡和审计预印本
`https://arxiv.org/html/2606.19624v1`（2026-06-17，不写成正式发表成果）。
数据采用校验过的官方 v1.5 TSV 和 Mass/Formula JSON，运行 manifest
记录候选文件、split 文件及 alignment checkpoint 的实际路径和 SHA-256。后续论文更新不得
将本地 loader 评价写成官方完整复现；若新增文献或改变评价定义，先更新计划并核验一手来源。

## 6. 实验与计算边界

- 是否新增训练实验：是，用户已明确授权；首批一个从头训练的 v1.5 alignment-42 与 12 个全量 reranker 运行。
- 是否新增评价运行：是，每个新 checkpoint 在完整测试集推理。
- 是否进行静态数据统计：是，缓存/样本数量、正例覆盖、forcing 标记、指纹与配置核对。
- 是否只修改文字、公式或引用：否；静态检查不计作实验结果。

## 7. 分步执行清单

- [x] 步骤 1：读取项目要求、检查工作树，建立独立分支和本计划。
- [x] 步骤 2：实现正式入口与约束，完成针对性测试、Ruff、compileall 和 dry-run。
- [x] 步骤 3（历史 v1）：提交/推送实现，固定源码 worktree 与输入指纹，启动 detached GPU 等待队列；现已停止监测。
- [x] 步骤 3a：完成 v1.5 审计与正式 alignment 实现，测试、提交并冻结新的队列；完整数据审计由队列执行。
- [ ] 步骤 3b：完成全部 v1.5 数据/候选身份审计和 alignment 训练，核验实际工件。
- [ ] 步骤 4：新建并逐一核验 6 个缓存，拒绝限量或 forcing 工件。
- [ ] 步骤 5：完成 12 个全量训练与测试推理，核验 checkpoint 数量/类型/seed/样本数。
- [ ] 步骤 6：汇总结果与协议审计；发现改变主张或需要扩大实验的问题时标为 `已偏离待确认`。
- [ ] 步骤 7：结果与证据边界确认后同步双语论文，编译、核对引用/页数及 diff，提交实质修改。
- [ ] 步骤 8：回填论文 commit 和验证，完成后以独立文档提交归档计划。

## 8. 风险、证据边界与待确认事项

- 训练规模增加不保证性能恢复或超越旧模型；不能将高分预设为成功判据。
- GPU 排队是正常状态，不通过限量、CPU fallback 或终止其他任务绕过。
- 输入指纹、完整 split 数量、模型配置或缓存语义不一致时停止对应任务，保留失败日志。
- 第一批只有一个 alignment checkpoint，不能据此替换/宣称完成跨 alignment 全量敏感性。
- 原论文其他限量结果不能混称为全量；完整稿件如何处理旧敏感性/消融，须在新结果审计时明确。

## 9. 验证方案

- [x] GPU 严格模式、等待、全量保护与缓存拒绝条件的单元测试。
- [x] 执行 `python -m pytest -q`（196 通过、1 跳过、1 个既有路径审计失败）；Ruff、compileall、diff 检查通过。
- [x] dry-run 确认 6 个缓存、12 个训练/推理组合且无 query cap。
- [ ] 原始日志与 checkpoint 的实际训练样本数、候选上限、forcing、CUDA 设备和 seed 一致。
- [ ] 全部结果完成后检查模型身份与评价协议、三 seed 汇总、失败/不完整项不进入正式表。
- [ ] 若更新论文，运行 `bash paper/build_release.sh` 并核对双语内容、页数、引用和 release manifest。

## 10. 执行记录

- **当前正式队列，2026-09-08 00:05（北京时间）启动**：r3 固定源码为
  `/data1/zyl/repos/SpecEmbedding-v15-fulltrain-20260908-r3/`，detached `15b1b5d`、工作树干净；
  Python 代码和 `params.yaml` 与已通过完整 CPU 验证的 `aecdfee` 一致，仅更新恢复状态文档。
  运行根为 `/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_fulltrain_20260908_r3_topk256/`，
  独立 socket 为 `/tmp/specembedding-v15-fulltrain-20260908-r3-1010/tmux.sock`，
  session `v15` / pane `%0`，入口 PID 2990241，初始 CPU 预处理 PID 2990250。
  先保存并复核 `inputs_and_commands.json`，再单次启动；实测 `status.json` 为
  `prepare_v15/running`，数据 manifest 已记录 v1.5 与修复后的身份策略，尚无 alignment 日志。
  子进程已核验 UUID 全卡顺序、严格 `cuda:1` 环境及本运行 `runtime_params.yaml`；
  后续物理 GPU 1 门槛仍为利用率 ≤10%、空闲 ≥20,000 MiB、30 秒轮询并连续满足 120 秒，
  无最长等待时间。启动前读数为利用率 96%、空闲 4,019 MiB，不代表后续实时状态。
  本入口自带逐阶段等待，没有新增外部监测器；旧队列、审计目录和固定源码均保留。
  状态入口为运行根的 `status.json`、`runner.log`、`logs/prepare_v15.log` 和
  `data/MassSpecGym/dataset_manifest.json`；此启动记录不代表正式数据审计、训练或评价完成。
- 2026-09-08：用户明确恢复目标，计划设为 `执行中`。重新核验旧队列与独立 CPU 审计均已退出，
  完整审计 manifest 及其输出通过哈希复查；当前没有运行中的正式训练。
  继续使用已验证的 InChI 修复和现有正式入口，在新 worktree、新运行根和独立 socket 中启动。
  入口在新目录重新生成并审计完整数据，再等待物理 GPU 1，以 `cuda:1` 训练；不绕过旧工件保护，
  不把独立 CPU 验证目录伪装成完成了正式阶段的运行根。启动后的实测状态另行回填。
- 2026-09-07 22:53：修复 `aecdfee` 的独立完整 CPU 验证通过；正式队列继续暂停。
  固定源码为 `/data1/zyl/repos/SpecEmbedding-v15-inchi-audit-20260907/`（detached、干净），
  验证根为 `/data1/zyl/SpecEmbedding/audits/massspecgym_v15_inchi_fix_20260907_topk256/`。
  独立 socket `/tmp/specembedding-v15-inchi-audit-20260907/tmux.sock` 中仅运行
  `prepare_massspecgym_v15.py`，CUDA 显式不可见，没有训练或监测后续阶段；该进程现已退出。
  `verification_receipt.json` 保存固定提交与完整命令，`prepare.log` 为原始日志；
  `data/dataset_manifest.json` 与 `independent_verification.json` 均为 `complete`。
  两候选协议所有源列表审计完成；独立读取检查了全部 split 数量、文件哈希、候选内容/顺序、
  图排除与身份转换的源位置和身份结果。数量汇总只维护于[修复记录](../../analysis/massspecgym_v15_inchi_fix.md)。
  这是修复验证，不复用为自动恢复的正式队列，不改变旧运行工件或论文结论。
- 2026-09-07 22 时：用户要求停止任务并排查修复。核验三个指定 server：旧 v1 只剩空闲 shell，
  两个 v1.5 pane 均已退出，没有训练或监测进程；不删除锁、日志、失败工件或固定源码。
  已定位 RDKit 2026.03.1 的默认 InChI 与 sanitization 使用不同 Kekulé 搜索顺序。
  修复仅在身份转换的分子副本上重试搜索，保留标准 InChI 选项及全部候选；新增逐项记录和校验，
  仍失败则带候选上下文停止。详见[故障修复记录](../../analysis/massspecgym_v15_inchi_fix.md)。
  专项 18 项、全仓 214 项通过，1 跳过，1 个相同的既有路径审计失败；静态检查通过。
  后续只执行独立 CPU 验证；正式队列维持停止，修复完成不自动恢复。本计划状态为 `暂停`。
- 2026-09-07 21:54（北京时间，历史核验）：r2 队列已于 20:33:12 失败退出，
  顶层 `status.json` 与数据 manifest 均为 `failed_or_interrupted`；指定 socket 的 `v15 / %0`
  为 `dead=1 exit=1`，PID 2977661 已不存在。未生成 alignment 日志/目录或 rerank 状态。
  原始 traceback 指向 `audit_candidate_list` 的 `Chem.MolToInchiKey(mol)`：候选 SMILES
  已通过解析，生成身份时抛出 `KekulizeException`。这是身份转换失败，不能未经核验套用
  上次的无效图排除，也不能据此判定源分子无效或修改候选协议。
  最后持久化进度为 Mass 4,096/32,010 个列表、1,034,580 个条目（约 12.8% 列表）；
  子审计的 `running` 是失败前进度快照，不代表仍在运行。Formula 审计尚未开始。
  三个 split 文件已生成，数量为 194,119/19,429/17,556；完整数据审计未通过，不可用于正式训练。
  本次只读核验运行现场，保留所有失败工件、锁与固定源码，未重发或创建队列。
  下一步需定位触发候选并核验身份转换原因；若处理需要改变协议，按计划规则确认后执行。
- 2026-09-07 20:29（北京时间，历史启动记录）：修复提交 `d15510b` 已推送；新的固定源码为
  `/data1/zyl/repos/SpecEmbedding-v15-fulltrain-20260907-r2/`，detached HEAD、工作树干净。
  新运行根为 `/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_fulltrain_20260907_r2_topk256/`；
  新 socket 为 `/tmp/specembedding-v15-fulltrain-20260907-r2-1010/tmux.sock`，session `v15` / pane `%0`，
  主进程 PID 2977661。已先保存并复核预检指纹，再启动；`status.json` 实测为 `prepare_v15/running`。
  候选审计通过后自动等待物理 GPU 1，以 UUID 顺序映射的 `cuda:1` 训练；其后仍逐阶段等待。
  不存在附加外部监测器。旧 v1 监测器和首个 v1.5 队列均已退出，没有向任何旧 pane 再派发。
  本目录的 `data/MassSpecGym/dataset_manifest.json`、`logs/prepare_v15.log` 记录 CPU 审计进度，
  `invalid_graph_mass.jsonl` / `invalid_graph_formula.jsonl` 位于数据目录，完整记录源列表的图排除。
  此状态不代表数据审计、alignment、十二组训练/测试、结果身份审计或论文更新已完成。
- 2026-09-07 20:17：首个 v1.5 队列在 CPU 候选审计阶段失败停止，未进入 GPU 阶段。
  三个 split 已生成并核验完整，失败原因是官方候选包含 RDKit 无法解析的过价 Si 分子；
  日志和原始失败目录均保留。现有 `MolSmilesDataset` / `mol_collate_fn` 本来就排除无法构图的分子，
  新审计不应把该已存在的编码资格规则误当成所有源候选均可解析的保证。
  修正审计为逐项记录无法构图条目、源目标和位置，保留原始候选内容/顺序；目标不可编码仍失败。
  缓存记录实际图排除和版本指纹，拒绝丢失 query 或正例。本次没有缩减训练数据，也没有更换官方候选库。
  必须以新 commit、新 worktree、新运行目录再次执行，不修改或自动重试 `6843e12` 失败队列。
- 修复验证：全仓 212 通过、1 跳过、1 个相同的既有路径审计失败；v1.5/正式队列专项 32 项通过，
  Ruff、compileall、diff 检查通过。额外修正旧 anonymous CPU smoke 的合成候选夹具：其入口显式
  no-forcing 却未提供正例，与验证断言矛盾；现在源夹具自然包含正例，未启用 forcing。
  完整 synthetic prepare → train → eval CPU smoke 已通过（6/4/4 条合成 query、1 epoch），
  anonymous supplement 相关测试 13 项通过。没有重建或发布匿名归档，也不是正式数据实验。
- 2026-09-07 20:15（北京时间）：v1.5 实现提交 `6843e12` 已推送，固定 detached 源码位于
  `/data1/zyl/repos/SpecEmbedding-v15-fulltrain-20260907/`，工作树干净；旧 `509fbeb` worktree 未修改。
  新运行根为 `/data1/zyl/SpecEmbedding/experiments/massspecgym_v15_fulltrain_20260907_topk256/`，
  已保存原始输入哈希、依赖版本、运行配置及全部阶段命令至 `inputs_and_commands.json`。
  独立 socket：`/tmp/specembedding-v15-fulltrain-20260907-1010/tmux.sock`，session `v15` / pane `%0`，
  主进程 PID 2964130，初始 CPU 预处理 PID 2964153。该入口自带 GPU 等待，不启动第二个外部监测器。
  `status.json` 已核验为 `prepare_v15/running`；源数据 manifest 已生成，完整候选审计尚未完成，
  未启动 GPU 训练。进度见 `runner.log`、`logs/prepare_v15.log`、`data/MassSpecGym/dataset_manifest.json`；
  后续 GPU 阶段使用物理 GPU 1、UUID 顺序映射的 `cuda:1`，保持既定利用率/显存及持续时间门槛。
  不复用或删除旧运行锁、不向旧训练 pane 派发、不自动重试失败工件。
- 2026-09-07 19:48（北京时间）：用户授权 v1.5 迁移及必要的 alignment 重训。
  核验旧监测器 PID 2853075、指定 socket 与命令后，仅向该监测器发送 SIGINT；
  `monitor.log` 于 19:48:40 记录停止。复查仅剩 `fulltrain` / `%0` 空白 bash，未派发训练，
  无 `runner.log`、`status.json`、缓存或 checkpoint。未删除锁、未改固定 worktree，未终止其他任务。
  本计划保持 `执行中`；旧排队状态不再代表当前 v1.5 进度。
- v1.5 实现：新增独立 CPU 数据/候选身份审计与整队入口；先审计所有源记录，再等待 GPU 训练。
  明确保存源候选的顺序和内容，只统计二维重复、不静默改写候选池。构图策略 opt-in，旧权重加载
  不继承新策略；新 alignment 保存构建配置、RDKit 版本、数据哈希与逐轮样本计数。
  官方源目标的二维 train/val/test 交集未发现；历史六条验证排除继续保留，不能把它们写成
  本次新发现的交集。完整候选审计、训练、测试结果仍待新队列执行。
- v1.5 专项 synthetic CPU 验证包含真实小模型 alignment 训练/保存/重载、全谱计数、丢尾批拒绝、
  非有限损失拒绝、规范图表示一致性、真实 spawn 进程池预处理及 CPU 审计失败阻断 GPU 阶段。
  这些测试不是正式训练或性能实验；临时工件不进入论文。最终全仓为 211 通过、1 跳过、
  1 个交接时已存在的内部路径审计失败（相同三个文件，未修改 GLACIER 记录）；新增 v1.5 专项
  15 项全部通过。Ruff、compileall、`git diff --check` 通过；固定源码 commit 在队列启动后回填。
- 2026-09-07：正式入口 `run_fulltrain_rerank.py` 及公共检查 `SpecEmbedding/utils/fulltrain.py`
  已提交并推送为 `509fbeb`。固定 detached 源码在 `/data1/zyl/repos/SpecEmbedding-fulltrain-20260907/`；
  已核验输入 train/val/test 为 194,119/19,429/17,556，alignment-42 checkpoint 与 selection 记录的
  SHA-256 相符。来源指纹和全部命令只维护于运行根目录的 `inputs_and_commands.json`。
- 2026-09-07：已启动后台自动等待流程，不需要前台会话存活；独立 tmux server socket 为
  `/tmp/specembedding-fulltrain-20260907-1010/tmux.sock`，训练 session `fulltrain` / pane `%0`，
  监测 session `gpu-monitor` / pane `%1`。监测物理 GPU 1，按 UUID 顺序显露所有 GPU，训练
  显式 `cuda:1`；利用率 ≤10%、空闲 ≥20,000 MiB，30 秒轮询、连续 120 秒，不设最长等待时间。
  初次核验利用率 87%、空闲 1,687 MiB，仍在等待，未发送训练命令；训练 pane 为专用空白 bash。
  从未向原有工作会话派发测试或训练命令。各阶段再次等待，失败停止，不自动重试或覆盖工件。
- 运行根目录：`/data1/zyl/SpecEmbedding/experiments/massspecgym_align42_fulltrain_20260907/`。
  `monitor.log` 是 GPU 等待/派发日志；进入正式入口后生成 `runner.log` 和 `status.json`。
  **SENT 仅表示命令派发，不代表训练启动/完成；新实验数字、三 seed 汇总与论文更新尚未完成。**
  查看监测器：`tmux -S /tmp/specembedding-fulltrain-20260907-1010/tmux.sock attach -t gpu-monitor`。
  如需取消等待，只在监测 session 中 Ctrl+C，不向训练 pane 发送中断。
- 独立 tmux 进程能跨终端/当前任务关闭继续运行，但不保证主机重启后恢复，也不是 GPU 资源调度器。
  低占用不证明设备无人使用，最终复查仍无法消除竞争；源码/输入指纹变化或新证据影响协议时停止核对。
- 2026-09-07：用户明确恢复本计划。GPU/tmux 监测器已实现，30 项专项测试（含独立 tmux
  server 的真实传递测试）通过，提交 `67dadfb` 已推送。将先补齐正式全量入口及校验，再固定
  源码、创建专用空白训练 pane，并由监测器等待派发；不向现有工作 pane 发送测试或训练命令。
  全仓测试发现既有路径审计失败（GLACIER 交接和项目记忆中的内部路径），其余 180 项通过、1 项跳过；
  不以修复该无关测试为由改动既有交接信息。
- 2026-09-07：用户要求继续全量重训；创建 `codex/massspecgym-fulltrain` 分支。
  主机环境已核验 PyTorch/PyG/RDKit/matchms 与 CUDA 可用；两张 RTX 4090 正在高负载，未启动新 GPU 计算。
- 2026-09-07：用户要求计划制定后暂时休息；计划标记为 `暂停`。尚未修改训练代码或配置，
  尚未创建运行源码 worktree、候选缓存、后台等待队列，未启动任何新训练或推理。
  只完成计划文档检查与提交；恢复执行需用户明确指示，不因 GPU 自动空闲而自行启动。

## 11. 最终结果

- 完成日期：尚未完成
- 最终状态：执行中；r3 正式队列已启动，当前进行新运行目录的 CPU 准备与审计，之后按门槛等待 GPU；尚无新 GPU 训练或测试结果
- 验证结果：代码与模拟/独立 tmux 测试、dry-run、输入指纹预检完成；正式实验与论文构建未完成
- 论文修改 commit：尚未提交
- 计划归档 commit：无需在本文件中自我引用
- 相对原计划的偏差：用户已批准迁移至 v1.5，并根据分子图表示差异重训 alignment；继续保持全量、GPU-only、无 forcing。
