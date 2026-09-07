# MassSpecGym v1.5 InChI 转换故障修复

2026-09-07；这是 CPU 数据审计与实现修复记录，不是训练或检索性能结果。
本次修复完成时正式训练按用户要求保持停止；后续执行状态见[全量重训计划](../docs/paper-change-plans/2026-09-07-MassSpecGym全量重训.md)。

## 原因与复现

r2 源码 `d15510b` 在生成候选 InChIKey 时直接使用默认 RDKit 调用，20:33:12 失败。
官方 Mass 候选 JSON 的列表索引 **4163**、候选索引 **180**（均从 0 开始）可独立复现。
该分子已收入 `tests/test_massspecgym_v15.py` 的真实回归夹具：64 个原子、80 条键，
标准二维身份为 `XPHPEGPJWUASIV`。原始文件指纹由 `params.yaml` 固定，未修改候选源。

当前 RDKit 2026.03.1 的解析与 InChI 转换使用不同的 Kekulé 搜索顺序：

- [MolOps.cpp](https://github.com/rdkit/rdkit/blob/Release_2026_03_1/Code/GraphMol/MolOps.cpp#L598)
  的 sanitization 调用 `Kekulize(mol, true, false)`，即 `canonical=false`。
- [InChI 实现](https://github.com/rdkit/rdkit/blob/Release_2026_03_1/External/INCHI-API/inchi.cpp#L1747)
  调用 `Kekulize(*m, false)`，使用默认 `canonical=true`；
  [参数声明](https://github.com/rdkit/rdkit/blob/Release_2026_03_1/Code/GraphMol/MolOps.h#L764)
  同时限定最大回溯次数为 100。

在这个真实候选上，默认 SMILES 解析成功，默认 InChI 转换抛出 `KekulizeException`；
对已解析分子的副本使用 `canonical=False`、清除芳香标志，再生成标准 InChIKey 成功。
这证明本次失败与搜索顺序有关，不证明源分子无效。没有调整回溯上限、降级 RDKit 或修改分子结构。

## 修复与保护

`molecule_identity` 优先保持默认标准 InChI 调用；只捕获 `KekulizeException`，在副本上执行
与 sanitization 相同的搜索方式，再用相同默认 InChI 选项转换。原始候选列表、SMILES、顺序、
编码器使用的 sanitized 图及正例定义保持原样。无法构图仍按已有规则单独统计。

成功的候选身份转换重试逐项写入 `identity_retry_mass.jsonl` / `identity_retry_formula.jsonl`，
含源目标、候选位置、SMILES 和二维身份；manifest 保存计数、哈希与策略版本。
目标转换采用同一逻辑，在 `target_audit.identity_retry_targets` 中保留记录。
验证入口拒绝缺少新策略或记录文件被改动的数据。

若转换仍失败，则保留异常链，并在错误中注明目标、源候选位置和 SMILES；不跳过候选、不生成
替代身份。总体失败时同步将正在运行的子审计状态改为失败，保留最后进度，避免误读为仍在运行。
原 r2 失败目录和固定 worktree 不修改，也不复用或自动重试。

## 验证

- 专项 18 项通过：真实故障候选、分子二进制前后相同、原子逆序后二维身份相同、源候选保留，
  异常仍失败停止并保留上下文，以及真实 spawn 预处理与重试记录指纹校验。
- 全仓 214 通过、1 跳过、1 个既有路径审计失败；仍为既有三个文件的内部路径记录，未清洗这些记录。
- Ruff、compileall、`git diff --check` 通过。
- 完整官方源 CPU 审计已于 2026-09-07 22:52 完成，固定源码 `aecdfee`。
  随后的独立核验通过：重新读取全部 split，验证文件哈希，逐项比较候选 pkl 与官方 JSON
  的内容、列表顺序及目标键顺序，并重查所有图排除记录和转换重试记录对应的源位置。

| 候选协议 | 源列表 | 源条目 | 无法构图条目 | 成功的身份转换重试 |
| --- | ---: | ---: | ---: | ---: |
| Mass | 32,010 | 8,093,599 | 69 | 1 |
| Formula | 32,010 | 6,788,068 | 0 | 0 |

全部 231,104 条谱图保留，train/val/test 实际读取数量为 194,119/19,429/17,556；
所有目标直接生成身份，未使用重试。两个协议全部源列表保留 exact target，没有 forcing。
69 个无法构图条目只在审计中单列，原始候选及输出 pkl 均未删除这些条目；编码器仍使用原有
图资格规则。唯一重试候选的输入图前后字节一致，另外用保留首次 sanitization Kekulé 结果的
独立表示生成了相同二维身份。三个 split 的记录哈希也与 r2 一致。

运行目录、固定 worktree、独立 socket 与回执入口集中于计划；`dataset_manifest.json`
及 `independent_verification.json` 均为 `complete`。本次 CPU 审计进程退出时，训练尚未恢复。
本次没有 alignment/reranker checkpoint、新 top-256 模型缓存或测试指标，也未更新论文。
