# 后续实验存储

从A05及后续启动起，数据、缓存和训练工件存放在外部项目根；本机启动配置使用
`/data1/${USER}/SpecEmbedding`。A05实际启动环境已核验采用此外部根，代码和Conda环境可以
保留原位置。已完成A04的固定源码、输入和输出路径保留追溯；不重写历史manifest中的路径。

| 根目录内位置 | 用途 |
| --- | --- |
| `raw/MassSpecGym/v1/`、`raw/MassSpecGym/v1.5/` | 经SHA-256核验的原始文件副本 |
| `processed/`、`train_cache/` | 通用处理数据及TokenSet缓存默认位置 |
| `graph_cache/`、`fingerprint_cache/` | 已审计固定输入缓存，保持现有版本目录 |
| `spectrum_metadata_cache/` | 逐原始query绑定的观测元数据及峰token来源，版本化准备与独立审计 |
| `experiments/<run>/` | 每轮独立数据、验证索引、checkpoint、日志和运行配置 |
| `audits/` | 准备、审计、迁移回执及条件启动草案 |
| `cache/`、`tmp/` | 依赖下载/编译缓存、Python字节码及运行临时文件 |
| `runtime/<attempt>/cache/`、`runtime/<attempt>/tmp/` | 新并行批次各自的依赖/编译缓存与临时文件 |

`params.yaml`的`storage.root`可由`SPECEMBEDDING_STORAGE_ROOT`覆盖，必须是绝对外部路径。
未定义的环境变量、home路径和越出根目录的输出/缓存符号链接会报错；不静默回退到仓库目录。
已有显式输入路径不会自动搬迁。未设置storage的历史YAML仍按原有规则解析，保持历史复核能力。

新任务使用以下入口；命令中的脚本使用绝对路径，因为子命令的工作目录设为外部根：

```bash
export SPECEMBEDDING_STORAGE_ROOT="/data1/${USER}/SpecEmbedding"
python run_with_storage.py --dry-run -- python "$PWD/train_align.py" --help
python run_with_storage.py -- python "$PWD/train_align.py" --help
```

第一条只显示环境和命令，不创建目录或运行子进程；第二条示范子进程启动。正式训练继续使用
已有GPU等待队列及显式设备，不能用示例绕过空闲检查。`run_massspecgym_v15.py`将外部存储
环境与输出根绑定到预检指纹，等待阶段复查；启动时继承的HF缓存等旧变量由入口统一覆盖。
入口不修改全局shell或当前任务的环境，不主动下载或重新训练。

用新存储入口复核历史run时，导入配置中的输出/缓存路径也必须满足存储根约束；不能直接
把含旧默认输出路径的历史runtime当作新入口配置。实际审计继续读取并验证原runtime及其SHA，
可另存只调整存储路径、保持模型/训练/增强/tokenizer语义一致的导入配置，不改历史文件或放宽保护。
启动Python子命令须使用脚本绝对路径；`python -c`的模块搜索路径取决于外部工作目录，不能
以其导入失败判定绝对路径入口同样失败。

修改来源、构造参数或版本后，新任务拒绝失效缓存，重新构建并审计新版本；只有确认没有
活跃训练、准备或审计引用后，才清理旧缓存载荷，保留来源、manifest和审计/清理记录。
仅改存储位置且字节与来源均相同，不使固定分子输入失效；迁移后的新启动回执须重新生成。

## 并行批次隔离

新批次显式增加`--runtime-root "$SPECEMBEDDING_STORAGE_ROOT/runtime/<attempt>"`，把HF、
Torch扩展/Inductor、Triton、CUDA、Numba、Python字节码、Matplotlib和TMPDIR统一指向
该批次目录。公共数据根与已有固定输入路径保留。正式预检将运行缓存根及所有环境项
写入存储回执，任何环境不匹配都拒绝；不传该选项时保持历史共享缓存行为，并清除继承的
旧命名空间变量。仅设置这个选项不隔离模型输出或配置，启动方仍必须逐项检查。

每个批次使用新建的detached worktree及固定commit，`PYTHONPATH`、绝对入口脚本与
子进程cwd均指向它；`SPECEMBEDDING_CONFIG`指向该批冻结配置，正式入口再保存独立的
`runtime_params.yaml`。实验数据副本、日志、checkpoint、验证分数/embedding、训练输入
回执和legacy `data.cache_path`均使用该批目录，不指向其他批次的可写目录。禁止运行中
切换源码、改配置、修改共享环境或清理被引用的文件；提交主仓库不会更新已冻结worktree。

当前正式`train_align.py --formal-fulltrain`走`classified_full_spectra`，读取原始谱图后
在进程内tokenize，不调用legacy `get_classified_data`的TokenSet缓存读写。验证固定图
缓存由`MoleculeGraphCache`以`np.memmap(mode='r')`加载，每次返回复制的tensor，允许
来源指纹一致的并行只读复用。构图/构造规则改变时必须另建版本；不共享学习embedding。

启动回执须记录解析后的目录、源码与配置SHA、模块导入位置、共享输入SHA及只读加载
依据。派发后另查实际PID/starttime、cwd、argv、配置和GPU UUID，完成审计再复核来源。
这些约束防止批次读错代码或覆盖工件；CPU、RAM和磁盘带宽仍共享，速度可能互相影响。
