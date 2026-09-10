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
