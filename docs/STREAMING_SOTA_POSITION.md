# FaultTune 连续验收的对照方法与主张边界

2026 年 11 月 3 日。当前最直接的强对照是既有的安全两样本 Bernoulli 检验，而不只是旧配对符号混合检验。固定角度的 Track-and-Stop 也可运行，但现有实现没有使用全部角度共享的单频结构，不能称为完整结构化识别 SOTA。

## 已核对的文献与可运行实现

| 方法 | 能比较的内容 | 不能由此声称的内容 |
| --- | --- | --- |
| 标准 PSR 加 Jeffreys 配对符号证据 | 无学习开销的任意时基线；使用所有合格配对中的不一致符号 | 没有保留全部 Bernoulli 计数信息 |
| 可预测 Kelly 与固定下注混合 | 数据驱动下注及证据复用的实际总成本 | 这是已有 e-process 方法，不是新统计框架 |
| Turner 等安全两样本检验 | 使用一致与不一致结果；相同过去数据可用于下一块预测 | 未知参数预测不自动拥有 oracle 效率 |
| 固定 PSR 的 Track-and-Stop 型对照 | 自适应不等两臂分配、全部两臂计数及审计后的联合 GLR | 两臂独立参数模型，不是全部角度结构化算法 |
| 使用相同数据的数值选角 | 区分角度公式、实际信息目标和计算速度带来的收益 | 数值网格本身不代表已完整复现 SOTA |
| Overshifted Parameter-Shift Rules | 已有移位优化及考虑不同测量方差的分配方法 | 移位优化、Neyman 分配本身不能作为 FaultTune 的原创性 |

Turner、Ly 与 Grünwald 的 [Generic E-variables for Exact Sequential k-Sample Tests](https://arxiv.org/abs/2106.02693) 于 2024 年发表于 *Journal of Statistical Planning and Inference*。其可预测估计量形式及增长率最优性质也在 [Turner 与 Grünwald 的 AISTATS 2023 全文](https://safestatistics.com/wp-content/uploads/2023/10/TurnerAIStats23.pdf) Section 2、式 (3)–(6) 中明确给出。该全文已核对：块大小可在块开始前依据过去数据决定，估计量可使用此前数据。

对当前一对独立观测，设过去数据给出的预测满足 \(0<\widehat p_- \le \widehat p_+<1\)，取 \(r=(\widehat p_-+\widehat p_+)/2\)。已有检验的单对因子为
\[
S(Y^-,Y^+)=
\frac{f_{\widehat p_-}(Y^-)f_{\widehat p_+}(Y^+)}
     {f_r(Y^-)f_r(Y^+)},
\]
其中 \(f_p\) 为 Bernoulli 概率质量函数。将各块因子相乘给出安全的证据过程，方向相反时交换两侧预测。两条方向过程各以 \(2/\delta\) 为阈值，其错误风险之和至多 \(\delta\)。

一侧有序零假设的有效性在这里可直接证明。记真实两侧概率为 \(u\ge v\)，并令
\[
k=\frac{\widehat p_+-\widehat p_-}{2r(1-r)},\quad
A(u)=1-k(u-r),\quad B(v)=1+k(v-r).
\]
两次观测独立使 \(\mathbb E S=A(u)B(v)\)。两因子非负，\(A\) 递减、\(B\) 递增，因此
\[
A(u)B(v)\le A(v)B(v)=1-k^2(v-r)^2\le1.
\]
在各块开始前可预测地改变角度和估计量，不影响此条件期望论证。取真实概率作为预测时，每个物理测量的预期对数增长为
\[
\frac{\operatorname{kl}(p_-\Vert r)+\operatorname{kl}(p_+\Vert r)}2,
\]
即均匀两角设计的完整 Bernoulli KL 目标。原配对符号证据只使用不一致结果，在有限差距下不必达到此增长率。这个改进属于已有安全两样本方法；FaultTune 必须将它同样提供给 PSR 与自适应选角对照。

## 移位优化与异方差分配已有先例

Banchi、Branford 与 Waghela 的 [Overshifted Parameter-Shift Rules: Optimizing Complex Quantum Systems with Few Measurements](https://arxiv.org/html/2510.05289v1) 为 2025 年预印本，后发表于 *Quantum Science and Technology* **11** (2026) 035031，[DOI: 10.1088/2058-9565/ae73ad](https://doi.org/10.1088/2058-9565/ae73ad)。本轮独立文献审查核对了全文 Section III 和 Appendix A，记录见 [theory_review.json](../results/streaming_validation/theory_review.json)。Section III 在共同测量方差上界下，以系数的 \(\ell_1\) 范数为目标优化移位网格；Appendix A 式 (59) 给出依赖各测量标准差的分配，形式为 \(n_j\propto |a_j|\sigma_j\)。因此，优化 parameter-shift 的移位，以及根据异方差使用 Neyman 型分配，均已有明确先例。

这些结果主要针对导数估计量的方差。FaultTune 当前证明讨论的是单频 Bernoulli 模型中辨别变化方向的 KL 信息，以及特定闭式双角设计相对于全部角度设计的信息比界；这是需要逐项比较的不同准则。已查阅的上述章节没有给出当前显式的 \(\eta\) 相关界，但这不足以证明首创性。一般异方差 \(c\)-最优设计、Fourier 模型判别设计和 KL 判别设计也已有文献。因此，目前只能称为“本项目已证明的模型特定信息界”，不能称为首次发现该闭式角度、首次优化移位或国际首个此类界。具体闭式公式及 \(\eta\) 界的文献优先性仍待追溯核对。

## 现有 Track and Stop 代码的实际适用范围

冻结模块 [kl_trackstop.py](../experiments/circuit_validation/kl_trackstop.py) 将各臂概率视为独立未知参数，只对目标施加一个固定线性对比约束。新包装器 [sota_baseline.py](../experiments/streaming_validation/sota_baseline.py) 使用两条固定角度臂 \(-\pi/2,+\pi/2\)，设置系数 \((-1,1)\)，原样调用 TrackStopState 的分配及数值证书。

每次建议只依赖已有计数；所有新增测量分别计费，并保存完整增量记录、最终原始计数及证书。两条轨迹的初步计时为主实例约 0.026 秒/条、零变化实例约 0.050 秒/条，是本机运行时间估计，不是测量效率结果。

改变角度后的数据不能继续并入这两条固定概率臂。给每个角度另开一个独立臂虽可保持另一种模型下的有效性，却会丢失单频结构并增加混合阈值的维度开销，不能将其当成充分强的同结构对照。

## 完整结构化识别仍需补齐

[Degenne、Koolen 与 Ménard 2019](https://proceedings.neurips.cc/paper_files/paper/2019/file/8d1de7457fa769ece8d93a13a59c8552-Paper.pdf) 已研究一般指数族结构化纯探索，并给出信息设计博弈及有限置信分析。单频有限角度模型可放入其结构集合。若要声称超过这类 SOTA，需要实现其相应最佳响应、探索分配和有效停止规则，不能仅沿用名字。

另一个有效的完整计数对照可由既有[顺序 universal inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC7382245/)构造：用过去数据给出下一块预测似然作为分子，以物理单频反号模型的最大似然作分母。该比值由每个真实零假设下的似然比鞅支配，允许自适应实验。实际停止还需要对反号模型的最大似然给出经认证的上界；现有两臂 GLR 并未完成这个三参数结构优化。

本轮可形成的明确结论应是：在指定单频模型、同样的数据复用和全部测量费用下，某个控制器是否超过列明的安全检验及数值选角实现。只有整体成本优势在保留实例与种子上复现，并补齐完整结构化比较后，才进一步评估 SOTA 主张。
