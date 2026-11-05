# FaultTune：首轮电路级候选验收

这个目录实现“更新参数前，以实际任务验收”的独立仿真研究。它不修改此前的 `local_validation` 实验，也不是 FaultTune 产品主分支的硬件集成。最终结论以中文研究报告、冻结确认协议及原始记录为准。

## 入口与证据

- [最终研究报告](../../docs/CIRCUIT_VALIDATION_RESEARCH.md)：正式结果、挑战判定和后续研究决定。
- [逐项交付核验](../../docs/CIRCUIT_COMPLETION_AUDIT.md)：原始数据、证书、种子复现及旧成果保留的证据。
- [预登记协议](../../docs/CIRCUIT_PREREGISTRATION.md)：44 个实例、8 种方法、每格 2,000 次，成本及统计口径。
- [物理模型与不可辨识引理](../../docs/CIRCUIT_MODEL_LEMMA.md)：固定哈密顿量、噪声通道、完整读出分布和解析反例。
- [统计方法](../../docs/CIRCUIT_STATISTICAL_METHODS.md)：未知 old/new、顺序有效性、CombGame 对应关系和同时置信区间。
- `confirmation_spec_v1.json`、`.sha256`、`_sources.zip`：正式运行前固定的输入和完整源码快照。
- `results/confirmation_v1_trials.jsonl.gz`：每批动作、全部执行和失败数、种子、成本及最终证书。
- `results/confirmation_v1_summary.json` 与三个 CSV：可从原始记录重建的逐格、逐方法和分层结果。
- `results/confirmation_v1_records_audit.json`：逐条账本和所有接受/拒绝证书的独立重算。
- `results/confirmation_v1_replay_audit.json`：预先固定的每格首尾种子重现；这是同实现的确定性复现，与独立数学检查分列。

## 环境

纯物理模型、统计证书和独立数学检查只需 Python 标准库。采样、数值功效规划、SciPy 优化交叉核验及完整运行需要 NumPy/SciPy。本机实际冻结运行使用 Python 3.8.3、NumPy 1.18.5、SciPy 1.5.0；系统默认的 Python 3.8.5 没有相应依赖。实际版本另外保存在冻结协议中。

以下命令在项目根目录运行。Windows 上该已有 Anaconda 安装需要先把其 DLL 路径加入本次会话：

```powershell
Set-Location 'D:\Engineering Projects\FaultTune'
$env:PATH = 'D:\softwares\anaconda\anaconda;D:\softwares\anaconda\anaconda\Library\bin;D:\softwares\anaconda\anaconda\DLLs;' + $env:PATH
$ftPython = 'D:\softwares\anaconda\anaconda\python.exe'
```

其他机器可将后续 `& $ftPython` 换成已安装 NumPy/SciPy 的 Python。随机计数算法的跨版本逐位一致性不作保证；精确复现使用冻结环境版本。不要把主机上的计数模拟耗时写成真实设备验收耗时。

## 验证与重新汇总

```powershell
& $ftPython -B -m unittest discover -s experiments/circuit_validation -p 'test_*.py' -v
& $ftPython -B experiments/circuit_validation/summarize_results.py --spec experiments/circuit_validation/confirmation_spec_v1.json --trials experiments/circuit_validation/results/confirmation_v1_trials.jsonl.gz --output-prefix experiments/circuit_validation/results/recomputed
& $ftPython -B experiments/circuit_validation/audit_records.py --spec experiments/circuit_validation/confirmation_spec_v1.json --trials experiments/circuit_validation/results/confirmation_v1_trials.jsonl.gz --output experiments/circuit_validation/results/recomputed_records_audit.json --workers 12 --chunk-records 16
& $ftPython -B experiments/circuit_validation/replay_registered.py --spec experiments/circuit_validation/confirmation_spec_v1.json --trials experiments/circuit_validation/results/confirmation_v1_trials.jsonl.gz --output experiments/circuit_validation/results/recomputed_replay_audit.json --workers 12
& $ftPython -B experiments/build_circuit_manifest.py --verify
```

开发阶段允许正常修改代码，原始源码已逐版归档。重新审计旧开发记录时，使用显式 `--source-archive experiments/circuit_validation/development_spec_v1_sources.zip` 或相应 v2 文件；审计器会核验压缩包内每个文件的冻结哈希，且分别报告当前源码是否一致。这个选项不会跳过源码验证。

独立物理与数学核验：

```powershell
& $ftPython -B experiments/circuit_validation/physics_audit.py --output experiments/circuit_validation/results/recomputed_physics_audit.json
& $ftPython -B experiments/circuit_validation/independent_checks.py --output experiments/circuit_validation/results/recomputed_independent_checks.json
& $ftPython -B experiments/circuit_validation/audit_combgame.py --output experiments/circuit_validation/results/recomputed_combgame_audit.json --seed 93291 --cases 200
```

这些命令使用不同输出名，已有原始审计记录和哈希清单可用于核对。SLSQP 对照验证采样优化器的数值结果，不能替代安全证书的证明。

## 完整重新运行

原始实验受本次研究的绝对截止时间约束。事后重现可显式使用 `--reproduce` 和新的运行超时，保持全部科学设置、种子和重复数不变，并写入不同结果前缀：

```powershell
& $ftPython -B experiments/circuit_validation/run_experiments.py run --spec experiments/circuit_validation/confirmation_spec_v1.json --prefix experiments/circuit_validation/results/reproduction --workers 12 --chunk-size 4 --wall-seconds 10800 --reproduce
```

不要覆盖已有原始结果。程序默认拒绝复用已有结果前缀；`--resume` 只补缺失的试验键，已记录的失败和超时不会被重跑成成功。若修改了冻结科学源码，先使用源码快照恢复到独立目录，再复现；不能把变更后的算法输出混入原试验。

## 方法和成本边界

直接 A/B 使用 IID 目标上下文，old/new 噪声独立。联合方法可按公开权重选择四个 old/new × context 臂。所有线路执行均付费；只返回失败指标不产生免费表征数据。选择候选的规则在验收前固定，本轮不声称已实现调参优化器或实机试验。

`combgame_glr` 实现两个 AdaHedge 学习器、最佳响应、乐观奖励和 C-tracking，动作是固定大小的独立线路批次。采样端的紧化与共用 Beta-GLR 阈值是明确修改；不自动沿用原论文全部渐近最优结论。

三倍速度比是相对全部预设强基线中最快者的研究门槛。弃权、超时和错误保留，另按满预算惩罚。零变化时经验上界未降到 1% 表示经验认证不足，不等于安全定理失效。当前范围是一个三比特静态模型族，不支持“普遍量子校准加速”或硬件领先表述。
