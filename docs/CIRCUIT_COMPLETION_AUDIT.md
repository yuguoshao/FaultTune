# 首轮电路级验收研究：交付完成审查

本表对应本轮目标：从三量子比特模型出发，为固定参数更新提供独立付费的任务验收，完整比较固定与顺序 A/B 及适用的强方法，交付代码、原始数据和中文报告，并如实报告挑战达标、未达标或证据不足。三倍优势是待检验指标，负面结果不等于研究工作没有完成。

**当前状态：本轮研究交付完成，三倍成本挑战未通过。** 正式运行、统计汇总、全部账本与最终决定证书审计、704 条 seed 重现、交付完整性检查、补充诊断、中文研究报告及最终 98 项文件清单均已完成核验。用户追加的 45 分钟用于收尾计算，截止为 2026-11-03 04:45:04 UTC（北京时间 12:45:04）；没有重置原实验起点，也没有新增正式实例或重复。

## 逐项覆盖

| 目标或必要交付 | 当前状态 | 证据与范围 |
| --- | --- | --- |
| 统一物理模型产生上下文效应 | 已完成 | [模型引理](CIRCUIT_MODEL_LEMMA.md)、`quantum_model.py`、`physics_audit.json`。固定 Hamiltonian、受控相位与 Kraus 通道；不是手工指定上下文失败率，也不是实机波形模型。 |
| 指定校准菜单存在无法确定目标改变量符号的盲区 | 已完成 | 完整八结果校准分布相同，而指定目标中的更新效果异号；有解析有理参考与独立数值交叉核对。只针对明定菜单和目标，不能推广成所有校准无效。 |
| 旧、新失败概率均未知并独立付费 | 实现与正式全账本已核验 | 验证器读取 paid loss-bit 接口，old/new 条件测量噪声独立；配对共享的是公开随机上下文。一次旧或新线路各计一次执行，完整比特串并未作为免费信息交给策略；全部 704,000 条账本核验通过。 |
| 候选选择与验收数据隔离 | 已完成，限本轮固定候选 | 候选按解析规则在验收前固定，本轮没有训练候选优化器或复用选择样本；因此选择数据集为空。验证数据身份由冻结 spec、case、policy、trial 和独立 seed 确定。 |
| 策略不读取评分器真值 | 实现审查与回归检查通过 | 策略只取得公开配置和采样方法；隐藏物理参数和精确概率留在评分器。相同 paid transcript、不同隐藏真值的测试给出相同策略行为。该结论是本实现的数据流审计，不是对敌意 Python 反射的安全沙箱证明。 |
| 固定/顺序 A/B、Track-and-Stop、CombGame 等预定比较 | 运行与正式统计已完成 | 冻结 44 个实例、8 个策略、每格 2,000 次，共 704,000 次。任何不利、失败或弃权格均保留；`calibration_only` 为失配诊断，不参加可靠性认证。 |
| 冻结前测试 | 已完成 | [frozen_tests.txt](../experiments/circuit_validation/results/frozen_tests.txt)记录 **170 tests, OK**。`preconfirmation_tests.txt` 是此前 166 项测试的阶段记录，不能单独作为 170 项的证据。 |
| 全部预定运行结束 | 已完成 | [运行状态](../experiments/circuit_validation/results/confirmation_v1_status.json)为 complete，704,000/704,000 条，运行用时 5,074.916 秒。全记录审计确认 704,000 个唯一 key、704,000 个唯一 seed，0 缺项。 |
| 全部 batch 付费账本、偏移、失败边际、切换、时间、读数对象、event hash | 已完成 | [全记录审计](../experiments/circuit_validation/results/confirmation_v1_records_audit.json)实际覆盖 704,000/704,000 条；0 invalid、0 missing、issue counts 为空，`passed=true`、`confirmation_engineering_eligible=true`。 |
| 全部最终接受/拒绝证书重新计算 | 已完成，614,229 个最终决定 | 同一全记录审计从 paid counts 调用对应证书；固定 McNemar 与顺序/联合方法分别重算，不以统一 evidence/threshold 比较代替。弃权记录保留并核对成本和证书 shots；本工具不宣称重算全部弃权证书，也未审查中间最早 crossing。 |
| 预先固定的 704 个 seed 确定性重现 | 已完成 | [replay 审核](../experiments/circuit_validation/results/confirmation_v1_replay_audit.json)：704 个唯一 key 全部匹配，0 差异、0 缺项。每格固定 `{0,1999}`，严格比较除两个墙钟字段外的完整记录。 |
| 正式统计汇总、同时置信区间及挑战判定 | 已完成；成本挑战未通过 | [正式 summary](../experiments/circuit_validation/results/confirmation_v1_summary.json)及三个 CSV 已生成；alpha 精确账本无超支。两候选安全/功效资格通过，成本点估计及同时上界均小于 1，详见下表。 |
| 独立原始统计汇总复查 | 已完成 | `experiments/verify_raw_summary.py` 与[独立核验结果](../results/confirmation_summary_verification.json)：全读 704,000 条，352 格整数 moments、计数、分位数及主格/分层比较共 8,019 项通过；原始 SHA 与运行记录一致。 |
| 零变化 McNemar 补充诊断 | 已完成 | 非冻结 `experiments/verify_zero_mcnemar.py` 与[诊断结果](../results/zero_mcnemar_diagnostic.json)：再次读取全部 704,000 条，独立重建三零格的 6,000 条 pair counts、固定样本量与 SciPy McNemar 决定，0 差异；原始 SHA 一致。该事后诊断不加入冻结同时置信族，详见下文。 |
| 正式冻结早于所有验证运行 | 已完成 | [delivery_integrity.json](../experiments/circuit_validation/results/delivery_integrity.json)：704,000 个有限 `started_epoch` 全部晚于 spec 创建时间，0 bad records；不是只比较整个 runner 的启动时间。 |
| 31 份冻结源码、源码 ZIP 及 spec 校验 | 已完成 | 同一独立交付检查确认全部 current/archived source hashes、ZIP 容器 SHA256、spec/sidecar/独立指定 SHA256 一致。 |
| 旧两轮研究完整保留 | 已完成 | 两份既有 manifest 的 **13 条与 23 条文件条目**逐项核验字节数和 SHA256，均未变化。这两份清单有重叠，不把它们称为 36 个不同文件。 |
| 中文最终研究报告 | 已完成并交叉核对 | [中文报告](CIRCUIT_VALIDATION_RESEARCH.md)分别报告物理结论、正式性能结果、补充诊断及适用边界；关键计数、成本区间、源文件与正式 Python 3.8.3 环境已与结果文件交叉核对。成本挑战明确未通过。 |
| 最终文件清单与可复现入口 | 已完成 | 非冻结 `experiments/build_circuit_manifest.py` 已生成 `experiments/circuit_validation/results/delivery_manifest.json` 及其 SHA256；98 项文件的字节数与 hash 验证通过。当前审查说明更新后已重建并再次验证清单。可用 `python -B experiments/build_circuit_manifest.py --verify` 重核。 |

## 已核实的冻结与保留信息

冻结 spec SHA256：

```text
ab2c711aacca68e84972ac5f6163a08bbc2f3e64544f90d26c9777904e146235
```

源码 ZIP SHA256：

```text
47dac2f4363d089ab2d7325084445d1282debfb35af7bd970c873f9d3f631dce
```

spec 创建时间为 `1793644548.657013`；全部原始记录中最早、最晚的启动时间分别为 `1793644566.2981691`、`1793649636.7404313`。这建立了本地记录中的冻结先后关系；它不是外部可信时间戳服务的证明。

运行器保存的压缩原始文件 SHA256 为：

```text
55e75e6c6aa88ca5931b6257f67879aa48c437ad23dc7963b6879ac2c7fa98e9
```

原始文件完整摘要还应由最终 manifest 对实际交付文件复核。本表不重复启动 1.19 GB 文件的全量账本、随机重现或汇总任务。

正式 spec 与 replay 一致的环境是 **Conda Python 3.8.3、NumPy 1.18.5、SciPy 1.5.0**；另一个标准库 Python 3.8.5 解释器曾用于部分独立检查，不能把它误写成正式采样环境。replay 的 Python、平台、NumPy、SciPy 四项环境比对均通过。

冻结 `audit_records.py` 的归档选项可核对逐 entry 源码 hash，但它只识别嵌套 `source_archive.sha256`，未识别正式 spec 使用的顶层 `source_archive_sha256`。非冻结、只读的 `delivery_integrity.py` 已补足容器 hash 和全部时间戳核验；这是审计覆盖补强，没有修改冻结策略、采样器或科学设置，也不需要重跑实验。旧开发记录则显式选择各自源码归档，不能静默忽略当前代码漂移。

## 必须分开的三类证据

1. **数学与数值检查。** 引理、鞅论证和明确假设支撑理论保证；`independent_checks.json` 的精确路径枚举、闭式 KL 包围及 `physics_audit.json` 的有限网格提供独立交叉核对。有限检查不单独证明全模型族安全或设备真实性。
2. **原始账本与决定证书审计。** 全部 704,000 条正式记录的已经付费统计量和成本重放通过，614,229 个最终接受/拒绝证书重新计算通过；这不重新抽样 RNG，也不等于独立实现整个算法。本次审计的中间 checkpoint 请求数为零；快速预筛允许延后决定，不能据此宣称每条运行都在最早精确 crossing 停止。当前证书模块与冻结 hash 一致；ZIP 的补充检查由独立交付完整性文件提供。
3. **同实现 seed 重现。** 固定首尾索引的 704 次 replay 验证冻结实现、运行环境与记录之间的确定性一致性；它不是外部实验复现，也不是新增的 704 个统计重复。重播结果不得进入原 2,000 次分母或用于重新选择算法。

## 最终结论的边界

正式成本比较已给出以下结果。这里 \(R\) 是每个主格“预定强基线中最小平均成本 / 候选平均成本”的几何平均；因此 \(R<1\) 表示候选更慢。区间继承冻结的同时覆盖账本，实际成本和满预算惩罚成本的本轮聚合结果相同。

| 候选 | 安全/功效资格 | \(R\) 点估计 | 同时区间 | 三倍成本目标 |
| --- | --- | ---: | --- | --- |
| `neyman_glr` | 安全 44/44、功效 38/38；所有强基线的 36 个主格资格通过 | 0.3389307020 | [0.2384272382, 0.4562802073] | 未通过 |
| `balanced_glr` | 安全 44/44、功效 38/38；所有强基线的 36 个主格资格通过 | 0.2530176557 | [0.1799066945, 0.3364658988] | 未通过 |

两个同时上界都小于 1，支持本预定模型矩阵上的负面成本结论，而不只是“没有证实三倍”。这个比较仍保留全部预定强基线；不外推为所有量子校准问题的性能结论。

诊断格另有经验安全资格未确认：`direct_fixed` 的三个零变化格为 23、15、36 次任何决定（每格 2,000 次）；`direct_mixture` 的三个零变化格为 9、10、13 次任何决定，另一个 near-boundary 格有 6 次误接受。这些格的同时上界超过 1%，使更严格的“全部基线全部诊断格合格”闸口未通过，但不抹掉所有主格已合格的成本比较，也不单独证明安全定理失效。

独立 McNemar 诊断逐条重算固定 A/B 三个零变化格的全部 6,000 个决定，全部一致；最接近 `0.005` 检验阈值的 p 值仍相差约 `1.3491e-5`。用原安全端点相同的严格 α 所作事后单侧 CP 下界分别为 `0.0041490973`、`0.0019960066`、`0.0082274086`，均小于 `0.01`，且经精确整数尾概率检查为向外舍入。`P[Bin(2000,0.01) >= 36] = 0.000753272` 是未经多重选择校正的参考尾概率；应保留这一异常提示，不能将它并入原冻结 95% 同时置信族或写成已经证明模型或理论违规。该检查只复查既有 counts 和条件检验，不重新验证所有 oracle 模型假设。

正式报告应分别给出理论错误保证、观察频率、同时置信结论以及完整工程审计。特别是，某格安全上置信界未降到 1%，只表示经验资格尚未确认；不能仅据此断言理论失效。证明违规需要相应下界超过门槛或定位实际假设/实现错误。

成本必须报告全部运行的实际执行数及弃权/超时/异常按满预算计费的保守口径。相对固定样本方法的优势不能替代相对全部预定强基线的比较；不能删除资格尚未确认的强基线来制造胜出。效率聚合使用预定的 36 个 main 实例，零、near 和 blind 诊断不事后混入效率分母。

本轮只覆盖一个静态三比特模型族。原草案的多机制、多任务、多规模基准是后续目标，不是此次已接受研究范围中的硬性完成条件；同样，本轮也不能据此声称跨设备泛化、真实量子校准墙钟加速或新统计算法的普遍领先。即使正式结果没有三倍优势，只要预定比较、审计和报告完整，仍可完成本轮研究目标。

## 最终 manifest 必须覆盖的材料

| 材料组 | 必须包含的交付 |
| --- | --- |
| 最终说明 | `docs/CIRCUIT_VALIDATION_RESEARCH.md`、本审查表、目录 README、预注册协议、统计方法、物理引理，以及被最终报告引用的文献定位与方差诊断文档。 |
| 冻结输入和源码 | `confirmation_spec_v1.json`、其 `.sha256`、`confirmation_spec_v1_sources.zip`，以及 spec 内 31 份冻结源码的路径/hash；非冻结交付辅助器须单独标注。 |
| 原始与运行记录 | `confirmation_v1_trials.jsonl.gz`、`confirmation_v1_status.json`；保留真实失败/弃权而非用重跑覆盖。 |
| 正式汇总 | `confirmation_v1_summary.json`、`confirmation_v1_cells.csv`、`confirmation_v1_policies.csv`、`confirmation_v1_candidates.csv`；如有图表，也必须指向同一冻结汇总。 |
| 正式审计 | `confirmation_v1_records_audit.json`、`confirmation_v1_replay_audit.json`、`delivery_integrity.json`，以及对应 closeout 日志与任务完成状态。 |
| 独立检查 | `physics_audit.json`、`independent_checks.json`、`combgame_numerical_audit.json`、`frozen_tests.txt`；保留较早测试记录并标明它们的阶段。 |
| 收尾独立补强 | 非冻结 `experiments/verify_raw_summary.py`、根目录 `results/confirmation_raw_aggregate.json`、`results/confirmation_summary_verification.json`；另含 `experiments/verify_zero_mcnemar.py` 与其最终诊断 JSON。这些辅助核验不改变正式试验或统计分母。 |
| 开发过程 | development v1/v2 的 spec、sidecar、源码 ZIP、原始 gzip、状态、汇总及审计；原 v1 当前源码漂移诊断和随后归档认证结果分别保留。 |
| 旧成果 | 两份旧 manifest 及本次 13/23 条核验结果；原 local/aggregate 文件仍在其既有路径。 |
| 时间与收尾记录 | 原截止状态作为历史记录、用户新授权下的 closeout 状态；`freeze_integrity.json` 是此前时间戳 pending 的快照，最终以完整 `delivery_integrity.json` 为准。 |

本次逐项审查的必要交付均有已核验的文件证据，最终 manifest 验证通过，因此研究交付完成。正式成本结果不满足预定三倍门槛，挑战未通过；两者是不同判定。
