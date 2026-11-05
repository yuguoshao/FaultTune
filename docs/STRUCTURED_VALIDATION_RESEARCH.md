# FaultTune：完整块证据与最强已实现对照

固定前瞻验证的 **9,600 条方法轨迹已全部完成并通过独立审计，主优越性门槛未通过**。候选 `closed_rb` 相对最强已实现对照 `closed_hedge` 的成本比为 **0.998296566**，95% 配对 bootstrap 区间为 **[0.994338232, 1.002477868]**：点估计节省约 0.17%，区间跨过 1。相对次对照 `psr_beta` 的成本比为 **0.973521278**，区间 **[0.961423784, 0.985180414]**，节省约 2.65%。因此，本轮没有证明超过最强已实现对照，更不能称为超过全领域 SOTA；也未达到早期三倍加速目标。

本轮原五小时窗口从 2026 年 11 月 3 日 14:07:14 UTC 开始，前瞻实验于 18:37:14 UTC 按预设截止退出，保留 7,937 条记录。用户随后明确授权追加最多 90 分钟，仅补齐既定验证、总审计和报告。[行政时间补充记录](../results/structured_validation/prospective_completion_v1/amendment.json)固定窗口为 2026 年 11 月 4 日 08:22:02–09:52:02 UTC，停止启动新执行组的时点为 09:32:02 UTC。原计划和旧目录保持不变；新目录逐字节保留全部旧记录，仅补原计划剩余 1,663 条。三份续跑及审计包装器通过[独立边界审查](../results/structured_validation/completion_wrapper_review.json)后，于 08:41:56 UTC 放行；采样完成状态于 09:14:48 UTC 写入，原冻结分析于 09:15:05 UTC 完成。

## 要解决的具体问题

上一轮独立验证中，闭式选角相对 PSR Beta 的成本比为 0.992238，95% bootstrap 区间上界为 1.000045，未通过优越性标准。本轮保留这项负结果，并把重点移到证据过程：相同的付费测量，是否能去掉任意配对造成的信息损失，从而更早作出可靠决定？

简洁的研究故事是：**边测边学，把每次测量变成更有效的证据。** 本轮分别检验完整计数、条件化、Rao–Blackwell 化和不等额分配，不把增加算法复杂度当成改进本身。

模型仍为 \(p(x)=A+B\cos x+C\sin x\)，给定过去历史与当前查询后按 Bernoulli 条件分布采样，块内测量独立。一次轨迹内参数固定，查询费用相同；目标是判断 \(C\) 的符号。所有初始化、探索、错误停止和达到上限后的弃权都计入总成本。当前证书不认证目标差值的幅度，也未覆盖硬件漂移或完整优化循环。

## 已完成的排除实验

固定 PSR、24 个旧公开主实例的正方向、每格 32 个独立重复，两种停止规则共享观测路径，共 1,536 条协议轨迹。完整双臂 Jeffreys universal inference 相对 exact Jeffreys 配对 Beta 的等实例几何平均成本为 **1.6598932893**：0 个实例更省、1 个相同、23 个更费。完整计数带来的潜在收益不足以抵消未知公共成功概率的学习代价。两种方法分别出现 0 和 1 次方向错误，均无弃权。

[固定 PSR 结果](../results/structured_validation/fixed_psr_probe/summary.json)及[审计](../results/structured_validation/fixed_psr_probe/audit.json)保留全部记录。全部 768 个联合种子与 1,536 个最终证书均重算通过。这是开发探针，不是前瞻优越性检验。

随后用 33 个固定 odds 专家的条件 Fisher 方法，完成 24 实例 × 16 重复 × 3 种角度策略 × 4 种证据，共 4,608 条轨迹。相对相同先验的配对网格，条件 Fisher 节省约 1.85%–2.12%；但相对更强的连续 Jeffreys Beta，先验离散化的代价抵消了收益。

| CF33 开发方法 | 相对同期 PSR Beta 成本 |
| --- | ---: |
| PSR 条件 Fisher | 1.077962 |
| 闭式选角 Beta | 0.958948 |
| 闭式选角条件 Fisher | 1.046824 |
| KL 网格选角 Beta | 0.951557 |
| KL 网格选角条件 Fisher | 1.046473 |

完整结果见[开发分析](../results/structured_validation/development_v1/analysis.json)。这些数值用于开发，不能从中挑选最优一项直接宣称独立验证成功。该批次有 1 次 grid Beta 错误、无弃权，全部最终证书和历史检查通过。

## 新方法与强对照

均分比较使用三种查询策略：固定 PSR、根据过去付费数据选出的闭式角度、同样数据上的 KL 网格角度。每种策略运行六种证据过程，使用共同轨迹但分别记录各自首次经认证的停止费用。

| 证据 | 公平比较中的作用 |
| --- | --- |
| 连续 Jeffreys 配对 Beta | 原有强基线，无公共成功概率拟合负担 |
| 71 专家配对网格 | 与新方法完全匹配 odds 和先验，隔离信息处理收益 |
| 71 专家条件 Fisher | 在每个块内条件于总成功数，消去公共成功概率 |
| Beta/Fisher 固定等权混合 | 混合资本至少为 Beta 资本的一半，检验混合代价 |
| 固定 odds 的 full-safe 方法 | 基于既有安全两样本因子、由过去数据预测公共成功率的强对照 |
| 71 专家 Rao–Blackwell 方法 | 对任意配对取条件期望，消除辅助配对随机性 |

71 专家的权重来自 Jeffreys CDF 分箱的固定有理近似，**不是连续 Jeffreys 混合**。所有新方法均使用同一个总风险阈值 \(1/\delta\)，\(\delta=0.01\)；full-safe 对照也去掉了旧实现不必要的双方向分拆和额外 fallback 权重。另加三种不等额分配的条件 Fisher 控制，其中 grid 控制求解 65 个有理角度上的信息设计，再实现被支持的反射两角分配。

所有预测、角度和分配均只依赖过去付费结果。查询坐标采用精确有理单位圆表示；模拟器使用 binary64 概率采样，最终停止则使用向外舍入的 Decimal 证据界。浮点筛查只决定何时尝试认证，不能直接授权决定。

71 专家均分开发实验已完成 **6,912 条协议轨迹**（24 实例 × 16 重复 × 18 方法），全部 18 种方法均未记录错误或弃权。全量审计已通过：6,912 个最终证书、52,656 个数据块和 244,084 次筛查重建一致，并完成 864 条协议记录的原运行器种子重放。以下为相对同期 PSR Beta 的开发成本比；这些区间与方法选择不作前瞻确认使用。

| 证据 | PSR | 闭式选角 | KL 网格选角 |
| --- | ---: | ---: | ---: |
| 连续 Beta | 1.000000 | 0.958648 | 0.969803 |
| 匹配先验配对网格 | 1.000000 | 0.958648 | 0.969803 |
| 条件 Fisher | 0.989892 | 0.952863 | 0.963350 |
| Beta/Fisher 混合 | 0.985571 | **0.946868** | 0.958571 |
| full-safe | 1.103719 | 1.063508 | 1.076674 |
| Rao–Blackwell | 0.983708 | **0.947329** | 0.957617 |

RB 在同先验、同查询策略的比较中，成本降低约 1.18%–1.63%。闭式 RB 相对 PSR Beta 节省约 5.27%，但相对闭式混合对照的成本比为约 1.000487，尚无领先点估计。不能只展示相对较弱基线的收益。配对网格与连续 Beta 在这些重复上的停止费用恰好一致，不表示两种证据在数学上恒等。详见[完整均分开发分析](../results/structured_validation/refined_development_v1/analysis.json)。

不等额开发也已完成同 24 个实例、每格 16 重复的 **1,152 条轨迹**，均未记录错误或弃权，grid 未触发支持回退。PSR、闭式和 grid 三种不等额协议相对同期 PSR Beta 的成本比依次为 **0.979729、0.997265、0.998841**，均慢于上述最强均分对照。此处同时改变了分配、重规划节奏和随机采样路径，不能把差异归因于分配比例一项。全量审计已通过，全部 51,972 个历史块和 1,152 个原始完整区间证书重算一致。详见[不等额开发比较](../results/structured_validation/unequal_development_v1/comparison.json)和[审计](../results/structured_validation/unequal_development_v1/audit.json)。

## 可以证明的进展

在固定先验、固定块划分、策略仅依赖过去两侧边际计数，且未来真实概率不泄漏过去辅助配对信息的条件下，Rao–Blackwell 证据满足

\[
M_n^{\mathrm{RB}}=\mathbb E[M_n^{\mathrm{pair}}\mid\text{边际计数历史}],
\qquad
\mathbb E\log M_n^{\mathrm{RB}}\ge\mathbb E\log M_n^{\mathrm{pair}}.
\]

这是同先验、同设计下的**固定时刻期望对数证据改善**。它不推出逐路径支配、平均停止费用更低，也不直接比较连续 Beta。当前固定物理模型和只用边际计数的控制器满足文中条件；一般可预测设计还需额外核对条件独立性。具体证明和反例见 [Rao–Blackwell 说明](RAO_BLACKWELL_BLOCK_EVIDENCE.md)。

本轮还给出适用于共同排序、自适应换角和固定专家混合的双方向风险论证；条件 Fisher 与 full-safe 均可使用 \(1/\delta\)，无需额外拆分两个方向。均分块的条件 Fisher oracle 信息损失至多 \(\log(N+1)+1\)，其中 \(N=2m\) 为该块总 shots；这一界不保证有限专家网格对任意弱信号都有正增长。推导与文献边界见[证据理论](STRUCTURED_EVIDENCE_THEORY.md)。这些是纸面数学证明与独立审查，未由形式证明助手认证。

65 角度的信息设计在 213 个拟合状态和相应符号反射上通过数值检查，最大相对原始—对偶差约 \(1.48\times10^{-8}\)。全部检查状态都得到一个反射两角支持；这只是数值发现，不是全域两点最优性定理。选角优化器给出的数值诊断也不参与最终证据认证。

本轮进一步严格解决了 **\(A=1/2\) 子类**：允许连续角度、任意多点支持及不等额分配时，均分 PSR 恰好达到最优 oracle 信息值

\[
I_*=\frac{1+2C}{2}\log(1+2C)+\frac{1-2C}{2}\log(1-2C).
\]

证明构造了覆盖所有角度的同一个反号零假设证书，并用矩阵对数凹性给出一致 KL 上界；均分 PSR 达到相同下界。详见[完整证明](CONTINUOUS_SYMMETRIC_DESIGN.md)和[独立逐式审查](../results/structured_validation/continuous_symmetric_design_review.json)。这划出了一类无需继续追求角度信息率提升的实例；它不证明有限阈值停止成本最优，尚未解决一般 \(A\) 或文献新颖性。

定向文献核查已确认，该算子论证属于 [Berta–Fawzi–Tomamichel（2017）](https://arxiv.org/abs/1512.02615) Lemma 1 的 measured relative entropy 凹变分框架，不能称为新的变分原理。当前未确认其中是否已有此单频子类的闭式结论；详见[原文核查记录](../results/structured_validation/continuous_symmetric_design_literature_review.json)。

从对称定理出发，再用经典 KL 数据处理不等式，得到覆盖**全部物理参数**的界。先设 \(0<A<1\)、\(B^2+C^2\le\min(A,1-A)^2\)，令 \(m=\min(A,1-A)\)，\(\Phi(v)=[(1+v)\log(1+v)+(1-v)\log(1-v)]/2\)，边界取连续延拓。\(E\) 为标准均分 PSR 的完整 Bernoulli 信息值，\(V\) 为连续全角度最优值，每 shot 以自然对数计量，则

\[
E=A\Phi(C/A)+(1-A)\Phi(C/(1-A)),\qquad
E\le V\le2m\Phi(C/m)<2E\quad(C\ne0).
\]

当 \(A=0\) 或 \(A=1\) 时，物理约束强制 \(B=C=0\)，信息值为零，不使用上面的含零分母公式。

统一因子 **2 是紧的**：已有一个严格物理参数族和明确反射两角设计，其相对 PSR 信息比可任意接近 2。进一步的仅依赖均值的界说明，当真实 \(A\in[0.499,0.501]\) 时，任意选角的额外完整 oracle 信息收益至多 **0.849839%**，无需径向裕量或信号下限；零信号时两种信息均为零。全部证明、余项界和确定性数值表见[全参数信息界](CONTINUOUS_PSR_INFORMATION_BOUND.md)，已通过[独立审查](../results/structured_validation/continuous_psr_bound_sota_review.json)。

这里的 \(E\) 是完整数据的 oracle 值，**当前配对 Beta/RB 不自动继承这个比值保证**。例如 \(A=1/2,B=0,C=1/2\) 时完整信息为每 shot \(\log2\)，配对 Beta 的实际渐近增长率只有 \(\log2/2\)。因此不能将这项定理宣传成“实际测量次数最多还能改善两倍”；它为区分选角损失和证据压缩损失提供了可证明的参考。具体界的文献新颖性仍未确认。

## 连续先验探索：不改变本轮前瞻候选

在前瞻实验运行期间，另外实现了固定连续先验的条件 Fisher 证据，试图消除有限专家网格的离散化损失。这里的 Jeffreys 指赔率经 logistic 变换后的 Bernoulli `Beta(1/2,1/2)` 先验，不宣称是任意超几何块模型的 Jeffreys 先验。算法对 log 被积函数的凹性使用分段指数弦积分，并以向外舍入给出完整积分的严格下界；计算节点可以按已观察数据加密，固定先验保持不变。

新方法的数学论证和确定性检查已通过[独立审查](../results/structured_validation/continuous_fisher_independent_review.json)。检查包含 9 组 exact Beta 参考、9 组独立完整积分包络、17 个精确中央归一化以及 8 组有限时域错误越界枚举。它们检验实现和有限例，不是新的模拟性能实验。

完整连续混合的最强本地逐路径界使用已观察到的条件支持大小。令 `N=2 sum(m_b)` 为 physical shots，`S=sum(min(T_b,2m_b-T_b))`，其中 `T_b` 是每块两侧总成功数。对任意有限 log-odds `lambda`，完整资本 `M` 与相应单专家资本 `L(lambda)` 满足

\[
M\ge\sqrt{\frac{2}{\pi(S+1)}}\frac{L(\lambda)}{\cosh(\lambda/2)}
\ge\frac{2L(\lambda)}{\sqrt{\pi(N+2)}\cosh(\lambda/2)}.
\]

因此，相对任意固定有限赔率专家，额外对数证据代价至多为 `0.5 log(S+1)` 加显式常数；如果赔率随 `N` 改变，不能省略 `log cosh(lambda/2)` 项。关键是两种等价加权子集表示和 Newton 负相关性给出 `Var(K_b) <= min(T_b,2m_b-T_b)/4`，再积分 log 被积函数的全局二次下界。完整证明及退化情形见[曲率界说明](CONTINUOUS_FISHER_CURVATURE_BOUND.md)。另有统一正 gap 下理想完整混合几乎必然有限停止、方向错误至多 `delta` 的[功效充分条件](CONTINUOUS_FISHER_EXPLORATION.md)。有限 shots cap 内的功效和平均停止费用仍需单独验证。

[单点曲率证书](../experiments/structured_validation/continuous_fisher_curvature.py)只需在一个精确有理赔率处计算被积函数的严格下界；该点可以根据数据选择，不改变固定先验。实现通过[独立数值审查](../results/structured_validation/continuous_fisher_curvature_implementation_review.json)。在全部 48 个旧开发末检查点上，[确定性检查](../results/structured_validation/continuous_fisher_curvature_checks.json)记录证书计算本身合计约 2.33 秒墙钟、1.92 秒 CPU，46 个末点能认证越界；它在这些点均弱于 193 点积分下界，不能据此声称证据更强或测量成本更低。这是便宜的初筛/保底计算，可与积分下界取最大值，未用于本轮冻结候选。

该探索只回放已经完成的开发历史，没有新增 Monte Carlo，也没有更换本轮冻结的前瞻候选、对手或样本量。回放选择 24 个旧正方向开发实例、两种策略、每种策略固定取 `rep=0`，共 48 条旧轨迹；同一实例的两策略存在随机流耦合，不能称为 48 个独立验证样本。

[49 点回放](../results/structured_validation/continuous_fisher_replay_development.json)覆盖 48 个原 Fisher 停止点和 46 个前一检查点；另两条轨迹在第一个检查点已经停止。48 个末点均被新下界认证越界，46 个已核前点均未越界。为分辨数值积分损失，另外对同一组 48 个末点作[193 点回放](../results/structured_validation/continuous_fisher_replay_refined.json)：其中 **22 个新下界严格高于旧 71 专家 Fisher 的证据上界**，确定的 log 证据增量约为 `0.000871` 至 `0.005750`，幅度较小。其余 26 个末点仅是未认证出优势，不能从积分下界推断完整连续积分更差。回放没有覆盖全部早期检查点及停止后的数据，因此不能估计新方法的首次停止费用，也未证明测量成本优势。

[回放独立审查](../results/structured_validation/continuous_fisher_replay_independent_review.json)通过了完整选择矩阵、费用、score、来源和汇总核对，并机械抽取首末轨迹重算两种节点密度下的整份证书及原 71 专家边界。

定向[文献核查](../results/structured_validation/continuous_fisher_literature_review.json)确认：Turner–Ly–Grünwald 第 7.2 节已有 `m=1` 条件 odds 因子，*Safe Testing* 已覆盖条件证据乘积、连续先验混合及对数级混合代价的背景。不能把这些框架本身称为 FaultTune 原创；本地大块推导、方向论证和数值认证的文献优先性仍待确认。

## 冻结的前瞻比较

[前瞻计划](../results/structured_validation/prospective_plan.json)在新验证观测产生前冻结，SHA-256 为 `c9baab61fc33594ac21d3231ae51fc40bc3824506145ae9f45c0c71960e81d37`。这是本地前瞻冻结，不是公开预注册。

候选从 3 种 RB 协议中选取；主对手从 18 种非 RB 实现中选取。二者都按完整开发集的等实例几何平均成本自动选择，精确并列时按方法名排序。必须先有完整开发汇总与绑定当前数据哈希的通过审计，才能选择及释放。PSR Beta 固定保留为次对照。

验证使用 48 个新主实例，每例 64 个独立联合重复，另有 8 个零变化实例、每例 16 个重复。主实例 cap 为 \(2^{22}\) shots；零变化专用 cap 为 65,536 shots。不同方法在同一实例、重复编号下共享种子标签，保留方法相关性；重复之间独立。所有错误和弃权计入均值。

唯一主指标先计算每个固定主实例的无条件平均成本比，再取 48 个比值的几何平均。使用固定种子 914207 做 5,000 次实例内配对 bootstrap，以两侧 95% percentile 区间的**未舍入上界严格小于 1**为数值成功条件，并额外要求独立审计通过。次指标只作描述。部分数据不计算主验收结果，也不允许看过验证结果后增样、调参或更换对手。

该 bootstrap 区间近似描述这 48 个固定实例的 Monte Carlo 不确定性，不是任意新实例总体的保证，也不是精确有限样本覆盖定理。每格 64 次主重复及 16 次零变化重复不足以经验认证逐实例 1% 风险；方向错误控制仍依赖已述采样假设、数学论证与经认证的停止规则。

两份完整开发审计通过后，[正式选择](../results/structured_validation/prospective_v1/selection.json)确定候选为 `closed_rb`、主对手为 `closed_hedge`、参考为 `psr_beta`。在检视选择后，于 2026 年 11 月 3 日 16:59:42 UTC 写入[执行放行记录](../results/structured_validation/prospective_v1/release.json)，随后开始新观测。旧目录的[截止快照](../results/structured_validation/prospective_v1/progress.json)仍保留 `incomplete_cutoff`：主实例 0–40 完整，实例 41 部分完成，共 7,937 条，距计划数还差 1,663 条。当时逐例审计覆盖前 525,326,930 字节，旧账本共 531,045,065 字节，未审尾部为 5,718,135 字节、65 条记录；未计算部分数据上的主验收结果。

授权补齐后的[完成状态](../results/structured_validation/prospective_completion_v1/progress.json)为 `complete`：56 个实例全部完成，7,937 条旧记录加 1,663 条新增记录共 **9,600 条方法轨迹**，对应 **3,200 个独立联合重复标签**（48 × 64 个主标签、8 × 16 个零变化标签）。完整账本为 630,379,098 字节，SHA-256 为 `1115fbb8e4a96de8bc244925e61291bb28ed9efa95734ed3d8842259dcf4b320`。同一标签中的三种方法相关，9,600 不是独立重复数。

## 完整前瞻结果

下表采用[原冻结分析的完整输出](../results/structured_validation/prospective_completion_v1/analysis.json)。成本比小于 1 表示候选更省；计算先求每例无条件平均成本，再对 48 个固定主实例等权取几何平均，不等于汇总所有实例总 shots 后的比值。

| 对照 | `closed_rb` / 对照成本比 | 两侧 95% 配对 bootstrap 区间 | 更省 / 相同 / 更费的实例数 |
| --- | ---: | --- | ---: |
| 主对照 `closed_hedge` | 0.998296566 | [0.994338232, 1.002477868] | 25 / 3 / 20 |
| 次对照 `psr_beta` | 0.973521278 | [0.961423784, 0.985180414] | 33 / 1 / 14 |

主区间未舍入上界为 `1.0024778683822761`，不满足严格小于 1，`primary_numerical_threshold_met=false`。主点估计为 `0.9982965658422733`，下界为 `0.994338232299159`。该结果未建立候选优越性，也不是两者等效的检验。最差单实例相对主对照成本比为 **1.096647**（实例 19）；相对次对照最差为 **1.152628**（实例 5）。次对照的较好结果只作描述，不替代主验收。

[总审计](../results/structured_validation/prospective_completion_v1/audit.json)的 `status=passed`，但 `primary_pass=false`；数据与计算通过审计、主优越性验收失败是两个不同结论。[对比图](../results/structured_validation/prospective_completion_v1/comparison.png)和[绘图来源清单](../results/structured_validation/prospective_completion_v1/plot_manifest.json)同时保留主、次比较及各实例差异。

| 方法 | 主实例轨迹 | 主实例错误 / 弃权 | 零变化轨迹 | 零变化错误 / 弃权 |
| --- | ---: | ---: | ---: | ---: |
| `closed_rb` | 3,072 | 0 / 0 | 128 | 0 / 128 |
| `closed_hedge` | 3,072 | 0 / 0 | 128 | 0 / 128 |
| `psr_beta` | 3,072 | 0 / 0 | 128 | 0 / 128 |

主实例共 9,216 条，全部作出正确方向决定；零变化共 384 条，全部在 65,536 shots 上限正常弃权。错误、探索和弃权成本均已计入汇总；零次观测错误不构成逐实例 1% 风险的经验认证。

## 与 SOTA 的关系

目前比较覆盖 exact 配对 Beta、条件 Fisher、Turner–Ly–Grünwald full-safe 因子的固定 odds 实现、65 角度有限网格信息设计及受支持的两角不等额分配；无法支持的分配有显式回退并记录。它比上一轮的对照更强，能区分证据处理、先验和分配造成的收益。本轮完整验证未通过主优越性门槛。即便将来通过同类检验，结论也仅限**固定模拟实例矩阵及按规则选出的命名实现**。

本轮的 `<1` 成本比门槛检验是否存在优势，并不替代[早期验收目标](VALIDATION_ACCEPTANCE_CRITERIA.md)中的三倍加速、同时覆盖和完整任务要求。通过本轮比较也不能称为已经达到那些更强的研究目标。

Degenne–Koolen–Ménard 的结构化纯探索、Liu–Qin–You 的 Frank–Wolfe self-play 及 Zecchin–Simeone–Ramdas 的量子 universal testing 有不同的算法或资源模型。本轮核对了原文，但尚未完整复现这些算法，因此不作全领域 SOTA 优越性或数学新颖性的宣称。文献链接与逐项边界保存在[证据理论第 8 节](STRUCTURED_EVIDENCE_THEORY.md#8-必要文献核对与主张边界)。

## 审计、时间与复现

本轮实验目录与旧研究分开。旧 streaming 证据清单 SHA-256 为 `3af4a45b0a3a7aa9d139f0efb0752651da1286635b54e7166f7e529f982343ed`；该清单 107 个文件与更早 learned 清单 39 个文件合计 127 个不同文件，封存器逐项核验其哈希未变。

审计包括历史块、付费数据拟合、可预测动作、费用、浮点筛查和最终停止。前瞻验证的每个决定都用严格证据下界独立确认；机械选择的首末重复、全部错误及全部 cap 弃权额外重算原始完整区间。首末重复和全部错误另作原运行器重放，其余轨迹也逐块重建 RNG。加速审计只利用正权混合的部分和作为下界，不改变冻结算法或停止时间；独立决策认证与完整区间重算分别计数。审计层独立重建数据流程，但仍复用冻结的 CGF、Decimal 基础例程，其正确性另有精确有限例及数学审查支持。

总审计覆盖 56 个实例、全部 9,600 条轨迹：其中复用并重新核对原始字节及来源哈希的 41 个完整旧实例审计，覆盖 7,872 条；本次重新执行 15 个实例的协议审计，覆盖 1,728 条，即原 65 条未审尾部加新增 1,663 条。原 531,045,065 字节前缀通过直接逐字节比较，旧目录保留原截止状态。

| 总审计项目 | 计数 |
| --- | ---: |
| RNG 历史块重建、付费动作与预测重建 | 各 249,595 |
| 检查点筛查重算 | 318,193 |
| 完整区间最终证书重算 | 3,706 |
| 独立决策证据下界核验 | 5,894 |
| 经严格认证的方向决定 | 9,216 |
| 终态记录核验与逐块种子重建覆盖 | 各 9,600 |
| 额外原运行器重放 | 224 个执行组、336 条方法记录 |

上述总数包含已通过且重新绑定的旧审计；本次新增执行分别为 40,606 个 RNG/动作块、50,245 次筛查、870 条完整区间重算、858 条独立下界核验，以及 60 个原运行器重放组、90 条方法记录。全部 5,000 次冻结 bootstrap 的区间和判定也由审计端独立重算一致。完整区间重算包含 cap 弃权的终态核对；不能把 9,600 条逐块种子重建写成 9,600 条原运行器额外重放。

原窗口的新执行组截止为 2026 年 11 月 3 日 18:37:14 UTC，硬截止为 19:07:14 UTC；按截止保留的数据未被覆盖。用户授权的补充窗口为 11 月 4 日 08:22:02–09:52:02 UTC（北京时间 16:22:02–17:52:02），新执行组截止为 09:32:02 UTC。补采样于 09:14:48 UTC 完成，原冻结分析于 09:15:05 UTC 完成；总审计文件于 09:19:10 UTC 落盘，状态 `passed`，运行器和审计器均正常退出。补采样阶段墙钟约 32 分 51 秒，审计阶段约 36 分 35 秒，二者并行且各自计时，不相加作为总用时。行政延长只改变执行窗口和新结果目录，未改变候选、对手、样本量、种子或统计方法。

本报告、代码、检查记录和最终结果一同封存于[证据清单](../results/structured_validation/evidence_manifest.json)，[源码与说明归档](../results/structured_validation/structured_validation_sources.zip)提供重放所需本地源文件。清单记录实际封存时间，同时保留旧目录的未完成快照和新目录的完整结果，以 `current_prospective_result` 指向后者，最终 `primary_accepted=false`。原始轨迹仍位于对应结果目录，来源归档不代替原始数据。

收尾后的只读完整性检查入口为 `experiments/structured_validation/freeze_evidence.py --verify-only`。另提供 [replay_stored.py](../experiments/structured_validation/replay_stored.py)，在原实验时限结束后也能按既有种子重放指定实例、策略和重复，除规划器墙钟计时外逐字段比较全部历史与证书；该入口不新增种子、不修改结果，也不代替全量审计。例如在项目根目录运行：

~~~powershell
$env:PATH = 'D:\softwares\anaconda\anaconda\Library\bin;' + $env:PATH
& 'D:\softwares\anaconda\anaconda\python.exe' -B experiments/structured_validation/replay_stored.py results/structured_validation/prospective_completion_v1 --case-id 42 --policy closed --rep 0
~~~

该命令已复现新完成目录中实例 42、重复 0 的两条闭式方法记录，种子为 `1100042378`，除规划器墙钟诊断外逐字段一致。此前均分 smoke 的 6 个方法记录及不等额 smoke 的对应 grid 记录也已验证；这些便于用户复现的入口不替代总审计。
