# GPU 监测与 tmux 单次派发

`watch_gpu_tmux.py` 持续查询一个物理 GPU，满足显存/利用率条件达到指定时长后，将一条命令
发送到**已有、唯一、空闲的 tmux shell pane** 并回车。脚本不会创建训练 pane、终止其他任务、
恢复暂停的论文计划、缩减训练数据或监测训练完成；本工具不是 GPU 资源调度器。

监测器只使用 Python 标准库和仓库 GPU 公共逻辑，无需导入 PyTorch、NVML 或安装新依赖。
运行平台为同一台 Linux 主机，依赖 `nvidia-smi`、tmux、`/proc`、POSIX shell 和 `flock`。

## 查看目标 session 与 pane

以下命令只读取 tmux 信息：

```bash
tmux list-sessions -F '#{session_id} #{session_name} windows=#{session_windows}'
tmux list-panes -a -F '#{session_id} #{session_name}:#{window_index}.#{pane_index} #{pane_id} #{pane_current_command}'
```

例如 session ID 为 `$2`、pane ID 为 `%7`。支持 `--target train_session`、`--target '$2'`、
`--target 'train_session:0.1'`、`--target '$2:main.1'` 和推荐的 `--target '%7'`。
session/window 名称均为精确匹配，不接受前缀猜测。只给 session 时，该 session 必须只有一个
唯一 pane（所有 window 合计）；多 pane 时直接报歧义，不默认选择 active pane。
`$2` 必须加单引号，避免被当前 shell 当成位置参数展开。

建议手动准备专用 pane，确认提示符空白且没有后台作业；监测期间不要在该 pane 输入文字。
命令使用**目标 pane 的工作目录、PATH 和训练环境**，不是监测器的环境。推荐训练解释器、脚本、
cache 和输出目录均用绝对路径，提前激活训练所需 Conda 环境。

## 推荐启动方式：精确 pane ID

下面的 `%7`、输入缓存和新输出目录都是示例，必须替换并核验，不能指向正在工作的 pane：

```bash
training_command='/home/zyl/anaconda3/envs/specembedding/bin/python /home/zyl/MYPROS/SpecEmbedding/train_rerank.py --train_cache /path/to/train.pt --val_cache /path/to/val.pt --save_dir /path/to/new_run --model_type relative --train-k 256 --device cuda:1'

python /home/zyl/MYPROS/SpecEmbedding/watch_gpu_tmux.py \
  --gpu 1 --target '%7' --cuda-visible-devices all \
  --command "$training_command" \
  --max-utilization 10 --min-free-mib 20000 \
  --poll-seconds 30 --hold-seconds 120 \
  --max-wait-seconds 43200 --log /tmp/gpu-monitor-1.log --dry-run
```

`dry-run` 仍等待条件、读取 GPU/tmux 和验证目标，但**不发送任何文字或回车**，也不消耗单次
派发记录。确认配置后移除 `--dry-run` 才会真正派发。不会自动加/减训练参数，尤其不加入
`--max-train-queries` 或 `--limit`。全量样本、no-forcing、正确 cache 协议仍须由训练入口核验，
本监测器不审计训练数据。

## 放入独立 detached tmux 会话

先在当前 shell 定义上面的 `training_command`，再执行：

```bash
tmux new-session -d -s gpu-monitor-1 \
  /home/zyl/anaconda3/envs/specembedding/bin/python \
  /home/zyl/MYPROS/SpecEmbedding/watch_gpu_tmux.py \
  --gpu 1 --target '%7' --cuda-visible-devices all \
  --command "$training_command" \
  --max-utilization 10 --min-free-mib 20000 \
  --poll-seconds 30 --hold-seconds 120 \
  --max-wait-seconds 43200 --log /tmp/gpu-monitor-1.log

tail -f /tmp/gpu-monitor-1.log
```

监测器必须在与训练目标不同的 pane 中运行，否则目标本身会被检测为繁忙。
不使用嵌套 `eval`；tmux 直接接收解释器及参数列表。`gpu-monitor-1` 必须是尚不存在的新 session
名称。监测进程发送一次后退出，所以其 detached session 可能消失，以日志为准。
需要取消时，只进入**监测器**所在 pane 按 Ctrl+C，不向训练目标发送 Ctrl+C。

## 更换 GPU、会话和脚本以复用

例如改为物理 GPU 0，目标为已有单 pane session `train_formula`，使用一个 `.py` 脚本：

```bash
python /home/zyl/MYPROS/SpecEmbedding/watch_gpu_tmux.py \
  --gpu 0 --target train_formula --cuda-visible-devices all \
  --script '/path/with spaces/train_formula.py' \
  --python /home/zyl/anaconda3/envs/specembedding/bin/python \
  --script-args='--device cuda:0 --output "/path/with spaces/new run"' \
  --max-utilization 5 --min-free-mib 18000 \
  --poll-seconds 15 --hold-seconds 90 --log /tmp/gpu-monitor-0.log
```

`.py` 使用 `--python`（默认目标 pane PATH 中的 `python`）；`.sh` 使用 `bash`；其他脚本必须
可执行且具有正确 shebang。脚本路径在监测器启动时解析为绝对路径。脚本及其参数必须显式包含
`--device cuda:N`，本工具不自动追加。脚本内部必须转发并遵守设备参数和完整训练数据约束。

`--command` 是经 `shlex.split` 解析的单条 executable/argv，不是任意 shell 程序。不展开 `$变量`、
`~`、glob 或命令替换，不支持 `cd ... && ...`、管道、重定向、环境赋值或 `bash -c` 等启动包装；
复杂流程请写成已有脚本并通过 `--script` 使用。参数中的空格、引号、分号、`$()`、反斜杠、中文
会被重新安全引用，作为字面参数发送。换行、控制字符和超过 8192 字节的派发文本被拒绝。

## GPU 编号与 CUDA_VISIBLE_DEVICES

`--gpu` 始终指 **nvidia-smi 的物理编号**。训练参数中的 `cuda:N` 是进程内的逻辑编号。
为避免目标 shell 中未知的 `CUDA_VISIBLE_DEVICES` / CUDA 枚举顺序导致用错 GPU，派发命令会
**明确、仅对这条命令设置** `CUDA_VISIBLE_DEVICES=<按指定顺序排列的 GPU UUID>`，不持久修改 shell。
这一环境前缀及完整命令会写入日志；用户的训练 argv 保持不变。

| 监测器参数 | 训练必须提供的参数 | 映射 |
| --- | --- | --- |
| `--gpu 1 --cuda-visible-devices all` | `--device cuda:1` | 默认按物理编号顺序显露全部 GPU |
| `--gpu 1 --cuda-visible-devices 1` | `--device cuda:0` | 只显露物理 GPU 1，它变为逻辑 0 |
| `--gpu 0 --cuda-visible-devices 1,0` | `--device cuda:1` | 逻辑 1 对应物理 0 |

不接受 GPU UUID/MIG 作为 CLI 编号、多 GPU 训练、重复设备参数、仅 `cuda` 或 `cpu`。
若监测器自身继承了 `CUDA_VISIBLE_DEVICES`，必须显式给出 `--cuda-visible-devices`，否则报错，
避免静默误解映射。没有继承时默认 `all`。不支持跨主机 tmux、容器内外不同 GPU 枚举或 MIG 重映射。

同时设置 `SPECEMBEDDING_REQUIRE_CUDA=1` 和 `SPECEMBEDDING_EXPECTED_CUDA_DEVICE=cuda:N`。
仓库公共 `resolve_device` 在此模式下拒绝 CPU、非显式编号、设备不匹配及 CUDA 不可用，不再
静默回退 CPU；不启用此模式的旧调用行为保持兼容。监测器不修改用户的设备或训练规模参数。
**任意外部脚本的内部行为不能仅靠命令行验证保证**：外部程序必须真正遵守 `--device`，不得
在内部重写可见设备、丢弃设备参数或自行回退 CPU；不使用仓库公共 resolver 的脚本需实现同等检查。

## 触发规则、锁与安全边界

- 使用 `time.monotonic()` 计时。利用率 `<= max` 且空闲显存 `>= min` 的连续成功采样达到
  `--hold-seconds` 后才候选触发；任一指标不合格或查询失败（含超时、N/A、非法输出）重置计时。
  最终再次查询 GPU，并重新解析原始 tmux 目标、核验固定 pane/server/shell 身份与空闲状态。
- 目标不存在、歧义或身份变更则退出；目标繁忙/无法证明空闲则等待并重置计时。不按回车、
  不发送 Ctrl+C/Ctrl+U、不清空输入、不切换 active pane，不中断任务。
- 默认只接受 bash/zsh/sh/dash/ksh：tmux 显示 shell、pane 存活且不处于 copy mode，
  `synchronize-panes` 关闭；Linux `/proc` 证明 pane PID 为受支持 shell、shell 处于休眠状态、
  位于终端前台进程组，且无任何直接子进程。前台训练、后台/stopped 子作业均保守拒绝。
- 这是启发式检查：不能证明 readline 缓冲为空、shell builtin 未在工作，也不能排除 shell
  提示符 hook 或刚脱离父 shell 的任务。`/proc` 不可访问、嵌套 shell 等可能导致保守拒绝。
  **使用专用、空白提示符的 pane，并避免监测期间手动输入。**
- `flock` 按服务器 socket/PID、pane ID、shell PID/starttime 加锁，统一存于
  `/tmp/specembedding-gpu-watch-<uid>/`。同一用户的并发实例（含不同 GPU/命令/目标别名）
  不能同时占用同一 pane；锁不依赖工作目录或 TMPDIR。
- 发送前 fsync 持久化 `dispatch_attempted` 记录。成功、失败、发送超时、Ctrl+C 或训练快速退出后，
  该 pane 均不自动重发；后续实例拒绝重复派发。记录保留“尝试发送”，不能据此认定 tmux 收到。
  需要下一次训练时，核验已有任务后使用新 pane。**不要删除运行中监测器的锁文件**，否则可能
  破坏互斥。tmux/pane 重建会产生新身份；这不是跨主机或跨用户的调度/锁服务。
- `send-keys -l --` 按字面发送安全引用后的单行命令，随后在同一个 tmux 命令队列中发 Enter。
  发送失败可能已有部分文字到达，甚至命令已经执行；此时状态不确定，保留记录并人工检查，不重试。
- **低利用率不证明 GPU 无人使用**。显存常驻、间歇性任务可能仍占用设备；采样之间的瞬时负载
  无法观测。最后复查与实际执行之间仍存在其他用户抢占 GPU 或输入 pane 的竞态。本脚本既不
  预留 GPU，也不限制其他程序使用 GPU，不能代替集群调度器。

## 参数与结果状态

默认阈值为利用率 10%、空闲 20,000 MiB、轮询 30 秒、连续 120 秒；阈值边界包含等号。
`--hold-seconds 0` 表示一次合格采样后仍执行最终复查。等待时间默认不限，`--max-wait-seconds`
同时限制 GPU 和 pane 等待。外部查询设置超时；超时退出可能有少量系统调用开销。

日志记录时间、GPU 指标、持续合格时间、等待/重置原因、查询错误、固定 pane、映射、派发结果。
日志含完整命令，避免在参数中放密码或令牌，也不要公开带内部路径/UUID 的运行日志。

- `SENT`：tmux 客户端接受了命令及回车，**没有验证训练启动或完成**，需要另外检查训练日志。
- `DRY-RUN`：通过门禁但未发送任何输入。
- 退出码：`0` 已发送或 dry-run 成功；`124` 等待超时；`130` Ctrl+C；`2` 参数错误；`1` 目标、锁或派发等错误。
- Ctrl+C 只退出监测器、释放互斥锁，不终止目标 pane 的任何进程；若中断发生在发送阶段，记录保留且不自动重试。

## 接入已批准的 MassSpecGym 全量重训

### 当前 v1.5 迁移队列

正式入口 `run_massspecgym_v15.py` 与 `run_fulltrain_rerank.py` 现在也支持 GPU 池择一：

```bash
python run_massspecgym_v15.py --gpus 0 1 --device cuda:0 \
  --source-dir "$V15_RAW_DIR" --legacy-tsv "$V1_TSV" \
  --prepared-data "$COMPLETE_V15_DATA" \
  --output-root "$NEW_V15_ROOT_topk256" --write-preflight
```

`--gpus` 可指定两个或更多物理编号，与 `--gpu` 互斥；列表不允许重复、负数或不存在的 GPU。
每张卡独立累计持续合格时间；查询并发且每次有超时，一张卡繁忙或查询失败只重置该卡计时。
满足既定显存/利用率与持续时间门槛后择一；同轮多卡合格按参数顺序优先，选择前再复查。
指纹复查放在最后一次 GPU 查询之前；GPU UUID 变化则停止，不能悄悄换卡。

GPU 池模式必须显式 `--device cuda:0`：每个计算子进程只显露选中 GPU 的 UUID，物理卡 0
或 1 在该子进程中都映射为逻辑 0。父入口不初始化 CUDA 或改写自己的设备环境。
`status.json` 的 `gpu_selection` 保存物理编号、UUID、显存/利用率和逻辑设备。
alignment、缓存、训练、评价各阶段重新择卡，仍顺序执行一个任务；不是多卡并行训练，
不中途移动正在训练的模型。现有 `--gpu N --device cuda:N` 用法保留，通用
`watch_gpu_tmux.py` 仍是独立的单卡/tmux 派发器，不应叠加到这些正式入口上。

可选 `--prepared-data` 指向已经完成审计的 v1.5 CPU 数据目录。预检核对策略、RDKit 版本、
全部文件哈希、原始输入指纹、划分数量与排除项，再将独立副本导入**新的**运行目录并再次校验；
原 manifest 与原始目录保留，来源目录和 manifest 哈希写入预检及导入阶段记录。
不复用失败数据，不覆盖已有运行根，也不继承缓存、checkpoint 或训练完成状态。
未提供该参数时仍重新准备全部数据。此选项适合在确认尚未开始训练后迁移等待队列。
先停止旧等待入口并确认退出，再用新固定源码、新 socket、新运行根派发，不能同时保留两个
会启动同一实验的等待入口。池查询仍不是资源预约，无法消除与其他任务的竞争。

2026-09-07 晚间用户授权迁移至 v1.5。旧 v1 外部监测器已停止且没有派发训练；不重新启动它，
不复用其 pane/锁作为新版派发目标。新的 `run_massspecgym_v15.py` 自带逐阶段 GPU 等待，
整个入口放在一个新的独立 detached tmux server 中，**不再叠加外部监测器**。

```bash
python run_massspecgym_v15.py --gpu 1 --device cuda:1 \
  --source-dir "$V15_RAW_DIR" --legacy-tsv "$V1_TSV" \
  --output-root "$NEW_V15_ROOT_topk256" --dry-run
```

输入目录须包含官方 v1.5 TSV 和两份候选 JSON，SHA-256 必须匹配 `params.yaml` 中的固定值。
`--legacy-tsv` 只用于 CPU 版本比较。`--dry-run` 仅读取；`--write-preflight` 只保存来源、配置和
阶段命令。正式执行不能复用已有 `status.json`、数据、alignment、rerank 或阶段日志目录。

执行顺序：

1. CPU 读取并校验全部谱图及候选，保留官方候选内容和顺序，记录二维身份重复/正例数量；
   无效分子图按既有编码器规则排除，原始列表不修改；全部排除逐项写入 JSONL（目标与源位置），
   统计二维身份时单列这些无法编码的条目。缺失/无法编码的目标、意外跨 split 身份交集、
   有效图无法确定二维身份或输入变化均失败停止。每 256 个候选列表记录进度。
   RDKit 默认 InChI 转换抛出 `KekulizeException` 时，仅在分子副本上以 `canonical=False`
   完成 Kekulé 搜索后再生成标准 InChIKey，保留原始候选与编码器图；成功记录写入
   `identity_retry_mass.jsonl` / `identity_retry_formula.jsonl`，并保存数量、策略版本与文件哈希。
   目标身份转换使用同一策略，记录于 manifest 的 `target_audit.identity_retry_targets`。
   此操作不把身份转换失败归入无效图，也不更改 InChI 选项；再次失败仍停止并输出目标、
   候选源位置与 SMILES。失败时子审计状态也同步标为失败，保留最后完整进度。
2. 审计通过后，满足既定 GPU 门槛才从随机初始化训练 alignment seed=42。新运行配置显式使用
   `rdkit_sanitized` 构图；训练每轮遍历 194,119 条谱图，验证保留既定六条排除，实际 19,423 条。
   同二维身份使用 multi-positive 标签；验证采用一次固定 seed 排列，跨 epoch 不重抽谱。
3. 验证新 checkpoint 的数据、构图、配置、设备及每轮数量后，运行 Mass 三个 top-256/no-forcing
   缓存和一组 relative/seed42 训练及完整测试。2026-09-09 起按用户要求固定单 baseline，
   不再自动展开 Formula、pointwise 或多 seed；后续每个 GPU 阶段仍重新等待。

`model.mol_encoder.graph_policy` 默认 `legacy_raw`，用于旧工件复核；v1.5 入口从 `params.yaml`
生成独立 `runtime_params.yaml`，显式切换构图策略。新 alignment 的 selection 文件保存完整模型配置、
构图策略、RDKit 版本、数据指纹和逐轮数量；加载拒绝不匹配的权重或配置。不得把旧权重配上新图
策略后称为 v1.5 重训。原始来源的 CPU 审计也不替代新模型输出的官方候选顺序/二维身份结果审计。

状态入口为运行根的 `status.json`、`runner.log`、`logs/prepare_v15.log`、
`data/MassSpecGym/dataset_manifest.json`。alignment 日志在 `logs/alignment42.log`，reranker 状态在
`rerank_topk256/status.json`。顶层完成只代表配置的训练/测试阶段完成，结果身份审计和论文状态另记。
缓存另外保存数据版本、数据 manifest 哈希、实际无法编码的分子列表与数量；任何目标编码丢失均
拒绝生成缓存。源候选身份审计与有效图候选池必须区分，不能把图排除后的结果称为未经处理的完整官方评价。

### 历史固定 alignment 入口

`run_fulltrain_rerank.py` 是[全量重训计划](paper-change-plans/2026-09-07-MassSpecGym全量重训.md)
的独立正式入口，不改变通用监测器语义。它要求新输出目录、干净的固定源码 worktree，读取
`params.yaml` 的 `fulltrain` 配置：先重建并核验 Mass train/val/test 三个全量/no-forcing/top-256
缓存，再执行 relative/seed42 一组训练与完整测试评价。入口拒绝扩大候选协议、模型和 seed 范围。
旧 12 组矩阵只保留在原固定源码及历史配置中。每个 GPU 阶段再次等待。

先把下面变量替换为已核验的**绝对路径**，运行只读检查；`MSG_DATA` 直接指向包含 `train.pkl`
等文件的目录，checkpoint 旁须有匹配的 `alignment_selection.json`：

```bash
python run_fulltrain_rerank.py --gpu 1 --device cuda:1 \
  --data-path "$MSG_DATA" --checkpoint "$ALIGN_CHECKPOINT" \
  --output-root "$EXPERIMENT_ROOT" --dry-run
```

固定源码后，用 `--write-preflight` 代替 `--dry-run` 保存 `inputs_and_commands.json`。
然后将相同训练命令（去掉上述两个检查开关）作为监测器的 `--command`，使用前文的独立
detached 会话方式启动监测器。实际入口重新核对输入/config/commit/命令指纹，要求监测器设置的
严格 CUDA 环境和 UUID 映射；不直接在未知 GPU 环境下运行。

正式入口没有训练样本数上限参数。`train_rerank.py --formal-fulltrain` 拒绝任何 query cap、
旧 forcing/不完整缓存，核验全部可训练 query 和每个 epoch 实际训练数量，并保存 cache SHA-256。
无正例 query 不能监督训练，单独统计，不混同人为限量；完整测试仍包含无正例 query。
每阶段记录 `status.json`，失败立即停止、不重试或覆盖旧工件；人工排查后使用新输出目录。
`monitor.log` 的 SENT 只表示派发，训练/评价进度须查看 `status.json`、`runner.log` 和各阶段日志。
单 baseline 优化使用验证集 Top-k（1/5/10/20）整体表现、MRR 为辅作决策，测试集不用于反复调参。
默认入口仍按验证损失/alignment、验证 MRR/reranker 选 checkpoint。新增 alignment 优化分支
显式按完整验证检索 Top-1、MRR 顺序选择，并保存其它 Top-k 的 Pareto 候选 checkpoint：

```bash
python run_massspecgym_v15.py --gpus 0 1 --device cuda:0 \
  --source-dir "$V15_SOURCE" --legacy-tsv "$V1_TSV" \
  --prepared-data "$AUDITED_V15_DATA" --output-root "$NEW_OPTIMIZATION_ROOT" \
  --optimize-alignment --baseline-checkpoint "$V15_BASELINE_CHECKPOINT" --dry-run
```

该分支只执行已审计数据导入、CPU 验证索引准备、固定 checkpoint 的完整验证，以及一轮
seed42 全量 alignment 训练；不派发测试或 reranker。去掉 `--dry-run` 前固定干净源码并使用
独立 detached tmux。GPU 阶段各自等待，不复用旧目录。

可在上述命令追加 `--prepared-validation-index "$VERIFIED_VAL_INDEX"`，导入另一已审计运行
的完整CPU验证索引；同时要求`--optimize-alignment`与`--prepared-data`。预检固定索引和
相邻JSON的SHA-256，检查数据来源、tokenizer、验证排除项及完整候选统计，执行时逐字节
复制到新run并重新核验；已有目标或源文件变化时停止，不能覆盖重试。此选项只省去CPU
索引重建，不缓存模型embedding，baseline及训练各轮仍完整重新编码。

索引位于 `validation/mass_val_topk256.pt`，相邻 JSON 保存 SHA-256 和样本统计。保留官方
候选源顺序及二维多正例，只沿用已审计的无效图排除；每轮重新编码候选，采用 float32 CPU
嵌入存储。主指标匹配 `torchmetrics 1.8.2` 的逐 query CPU `argsort`，稳定排序另列敏感性视图；
不把主指标误称为 `torch.topk`。模型/数据/分母/候选身份的完整最终审计仍是独立验收事项。
`validation_retrieval/` 保存每轮分数和排名，selection JSON 记录选优轨迹及 Pareto 文件名。
训练完成只代表这一个优化候选完成，不代表已接近 SOTA。表现改善后再恢复矩阵。

新源码的正式 alignment 自动逐轮保存 `resources/stage2_epochNNN.json`，并将相同记录写入
selection 的 `resource_profiles`。记录参数量、实际 query 数、训练吞吐、训练/对比验证/
完整检索验证/选优与 checkpoint 的同步墙钟耗时，以及该进程在显式 CUDA 设备上的
PyTorch allocated/reserved 峰值；早停的最后一轮也保留记录。检索验证耗时包含候选重编码，
这些阶段耗时不等于部署延迟。CPU 内存字段仅为主进程生命周期峰值 RSS，包含之前的准备工作，
不包含 DataLoader workers，不能称为全进程树或整机峰值。该记录器不创建后台监测进程，
也不改变训练 RNG、选优或早停规则。运行中的旧固定源码不会新增此记录，不能事后补造峰值。

训练顺序实验可在上述优化命令中显式添加 `--alignment-batching mass_blocks`。
默认仍为 `train.align.batching: random`；`mass_block_size: 32` 来自 `params.yaml`，
必须整除 batch size。该选项只改变训练 batch 的组装：按计算精确质量形成小组，混合小组
并保留尾批，每条训练谱图每轮恰好一次；不改变验证顺序、模型、loss 或候选池。
selection 的逐轮审计记录质量与排列指纹、query 唯一覆盖数，队列完成检查拒绝配置不符或丢样。
按计划一次只评估一个方案；新选项的实现或 dry-run 不表示它已派发，也不改运行中的固定源码。

检验分子图扰动的单项实验可在优化命令中添加 `--no-alignment-mol-augmentation`。
该开关仅把运行配置中的 `augmentation.node_drop_rate` 和 `augmentation.edge_mask_rate` 置零，
保留谱图增强、`prob`、batch、网络和损失。省略开关或使用正向开关时沿用 `params.yaml` 的
扰动率；该选项不能用于默认队列。完成检查与独立审计均核对 selection 的实际
`config_snapshot.augmentation` 是否与运行前指纹配置一致，避免只核对预期命令。

自然候选监督已有输入、损失、正式入口与完成审计，默认不启用，不改变现有运行的loss。
`SpecEmbedding/utils/training_candidates.py` 的加载器要求准备与独立复核回执，重新比较全部
训练query、候选源顺序、二维标签和图排除，并在读取结束时复查输入指纹。
采样按不同负例二维身份均匀、不放回进行，再在该身份的原始条目中均匀选择表示；
排除全部同身份正例，负例不足时保留实际数量，不丢query、不forcing、不改变评价候选池。
负例预算与候选池缓存容量均需显式提供；采样使用 seed、epoch 和原始query索引的独立随机数流，
与调用顺序无关，不消耗训练的全局随机数。实现和CPU预检不代表已经启动候选监督实验。

`CandidateAlignDataset` 把完整谱图 Dataset 的二维身份分组顺序映射回原始query索引，
逐条核对目标、检查全量唯一覆盖，并要求 tokenization 审计与候选元数据具有相同manifest。
正负图均显式绑定 `rdkit_sanitized`，避免spawn worker重新导入全局配置后改变构图；
负图采用与正图相同的增强概率和扰动设置，缓存原图不被增强修改。每轮迭代前必须显式
`set_epoch`，训练组件拒绝persistent workers和丢尾批；完成每轮再核验原始query恰好一次。

`CandidateTrainerAlign` 使用原双向in-batch loss，加显式权重的query内候选CE；当前分子塔
对负例实时编码并反向传播，不缓存随训练变化的embedding。每条query一个原有正例表示，
负例来自已核验自然池并排除全部同二维身份别名；无负例query的候选CE为零，仍保留在均值
分母和原in-batch监督中。验证沿用原路径，模型结构和余弦检索分数不变。
该组件记录逐轮负例数量、query覆盖与抽样顺序指纹；当前完整负图batch保留反向图，
实际显存和时间尚未测量，不能把分子输入缓存称为激活内存优化。
优化入口可显式添加 `--alignment-training-candidates "$VERIFIED_TRAIN_CANDIDATE_METADATA"`，
同时要求 `--optimize-alignment` 和 `--prepared-data`；正式参数来自
`params.yaml` 的 `train.align.candidate_supervision`。省略候选参数时必须保持 `enabled: false`，
提供参数时在独立运行配置中启用；预检重新核验完整来源并固定 `candidate_training_input.json`。
训练入口只接受匹配该回执的配置和完整验证选优，记录每轮 `candidate_training/stage2_epochNNN.npy`
实际原始query顺序及相邻JSON统计；完成检查和独立审计重放全部query的负例采样、批次边界
与来源，输入不符、重复/缺失query或采样指纹异常均失败。模型实验仍须先固定唯一实验卡和parent。

单次优化完成后，使用独立 CPU 审计生成新回执：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python audit_alignment_optimization.py --run "$COMPLETED_OPTIMIZATION_ROOT" \
  --output "$NEW_AUDIT_RECEIPT"
```

该入口拒绝未完成的队列和已有输出文件；重新检查输入/配置指纹、每轮完整计数，
从全部保存分数独立重算排名和 Top-k/MRR，并核对选优、Pareto 候选及最终权重。
若保存了逐轮资源记录，还会检查阶段时长/吞吐/设备/计数/显存值及独立文件与 selection 的一致性，
并保存文件哈希；旧固定源码未记录的资源明确标为 `not_recorded`。
报告比较原 baseline，以及同一已观察轨迹中的最低验证 loss 轮；后者不代表重新执行了
按 loss 早停的反事实训练。容差固定为各 Top-k 0.2 个百分点、MRR 0.002 原值，
报告所有改善与退步项，不自动换 checkpoint、重启训练或修改论文。
它是保存分数的完整 CPU 审计，不是新的模型推理、官方 loader 测试或 SOTA 验收。

冻结候选前的谱图输入敏感性检查使用同一验证索引与选中的 checkpoint，显式添加
`--spectrum-control permuted`、`constant` 或 `precursor_only`。下列是供 GPU 等待入口派发的命令，
需先固定源码、独立配置和新输出目录，并由等待入口选卡及注入严格 CUDA 环境；不直接跳过 GPU 门槛：

```bash
SPECEMBEDDING_CONFIG="$FROZEN_CONTROL_CONFIG" python alignment_validation.py \
  --data-path "$AUDITED_V15_DATA" --index "$FULL_VALIDATION_INDEX" \
  --checkpoint "$FROZEN_ALIGNMENT_CHECKPOINT" --output "$NEW_CONTROL_OUTPUT" \
  --device cuda:0 --spectrum-control permuted
```

每种模式使用独立新输出目录；一次只运行一个已排队阶段。独立配置必须显式包含
`retrieval_validation.spectrum_controls`，并使 `model` 与 checkpoint selection 完全一致，包含
`rdkit_sanitized`。A01/A02 的旧运行配置没有控制参数，不直接用作控制配置，也不修改原文件。
参数集中在新版 `params.yaml`：置换 seed42；常量输入为 m/z `[100, 50]`、强度 `[2, 1]`，
其余位置为 padding。这些设置在观察控制结果前固定，不用 test 选择参数。

- `permuted` 对全部有效验证 query 构造一个随机循环，每条谱图恰好作为一次输入，且不分配给
  原 query；置换包含实测 precursor、强度与 mask。二维身份仅用于事后统计偶然同身份 donor，
  不参与选择排列；保存完整 donor 原始索引、排列哈希和同身份数量。
- `constant` 给所有 query 相同的固定 token 与 mask，不保留各自的 precursor 或峰数。
- `precursor_only` 仅保留每条 query 的实测 precursor m/z 与既有强度标记2，后续位置全部
  置零并设为 padding，不保留碎片或原峰数；构造前逐条检查首 token 有效且标记正确，保存
  按完整 query 顺序排列的 float32 小端母离子质量 SHA-256。它不读取目标标签或候选信息。
- 每个 query 的候选、顺序、正例、原始索引和分母均不变；无正例与空候选 query 仍纳入分母。
  使用新目录与 `control_*_epoch000.pt`，保存分数后独立 CPU 重算全部排名；正常训练完成审计
  拒绝将控制 snapshot 当成普通验证结果，控制不参与 checkpoint 选择。

置换和常量检查改变了整个谱图输入，包括 precursor；`precursor_only` 补充保留母离子时
移除碎片的敏感性视图。后者仍改变输入分布，不是重新训练的消融；成绩差异本身不证明
碎片的独立因果贡献，常量下非零成绩也不自动证明数据泄漏。单个固定置换不是统计显著性或官方 test 结论。
当前正式优化队列不自动派发这些检查，需在候选冻结及独立 GPU 阶段安排后执行。

## 验证（不启动训练）

下一轮优化可显式复用完整验证图，先在独立冻结源码内运行CPU准备：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python prepare_validation_graph_cache.py --data-path "$AUDITED_V15_DATA" \
  --index "$FULL_VALIDATION_INDEX" --output "$NEW_VALIDATION_GRAPH_CACHE"
```

准备使用`params.yaml`中的`retrieval_validation.graph_cache_preparation`，完整构建后重新
构图并逐张核对全部张量，保存`manifest.json`、`audit.json`和`preparation.json`；无limit或
覆盖/续跑选项。只保存固定输入图，不包含模型embedding。完成审计后，在新的优化队列命令
中同时加入`--prepared-validation-index "$FULL_VALIDATION_INDEX"`与
`--validation-graph-cache "$NEW_VALIDATION_GRAPH_CACHE"`。baseline和每轮验证均重新编码，
来源/软件版本/图文件指纹不符立即失败。不要修改或重启正在运行的队列来切换缓存。

```bash
python -m pytest -q tests/test_gpu_tmux_watch.py
python -m pytest -q tests/test_fulltrain_runner.py
RUN_TMUX_INTEGRATION=1 python -m pytest -q tests/test_gpu_tmux_watch.py
ruff check .
python -m compileall -q .
```

普通测试模拟 nvidia-smi、tmux 和单调时钟，覆盖连续阈值、重置、失败、目标保护、锁、单次派发、
dry-run、超时、Ctrl+C、编号映射及参数转义。显式启用集成测试时，测试创建全新临时 socket、
空配置的独立 tmux server/pane，仅发送一个将 argv 写入临时 JSON 的 Python 命令，GPU 查询为
模拟值，不导入训练模型、不做 GPU 计算；只关闭测试自己创建的 server，不读取或操作工作会话。
