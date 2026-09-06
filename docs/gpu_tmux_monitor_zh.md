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

`run_fulltrain_rerank.py` 是[全量重训计划](paper-change-plans/2026-09-07-MassSpecGym全量重训.md)
的独立正式入口，不改变通用监测器语义。它要求新输出目录、干净的固定源码 worktree，读取
`params.yaml` 的 `fulltrain` 配置：先重建并核验六个全量/no-forcing/top-256 缓存，再依次执行
mass/formula × relative/pointwise × seeds 42/43/44 的训练与完整测试评价。每个 GPU 阶段再次等待。

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
三 seed 汇总、二维身份审计及论文更新仍是后续工作，不能将队列派发说成实验完成。

## 验证（不启动训练）

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
