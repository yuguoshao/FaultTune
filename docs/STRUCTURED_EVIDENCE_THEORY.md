# 完整计数、条件 Fisher 与固定 odds 安全混合

2026 年 11 月 3 日。本轮先检验“完整计数是否真的减少 shots”，再增加复杂度。固定 PSR 的独立双臂 Jeffreys universal-inference 开发探针已产生负结果：24 个旧公开主实例的正方向、每格 32 个独立种子，两种停止规则共享观测路径，共 1,536 条方法记录；完整计数 UI 的等实例几何平均成本为配对 Beta 的 **1.6598932893 倍**。这是开发结果，不是独立确认。完整计数并不自动抵消 nuisance 和预测遗憾。

当前优先比较两种不丢弃付费学习数据的机制：大块条件 Fisher 专家混合，以及固定 odds、可预测 nuisance 中心的 full-safe 专家混合。二者使用相同有理 odds 和先验权重、同一总体风险阈值；与 exact Jeffreys 配对 Beta 和相同专家网格的配对控制一起比较。本文给出有效性证明，不预先宣称有限预算更快或文献首创。

## 1. 共同模型与费用

真实物理模型为 \(p(x)=A+B\cos x+C\sin x\)。在每个块开始前，依据已完成的数据选择 \(t\in(0,\pi)\)、样本数和预测参数；块内分别在 \(-t,+t\) 独立重复采样。于是所有块共享方向：\(p_+-p_-=2C\sin t\)。默认每侧 \(m\) 次，记两侧成功数为 \(S_-,S_+\)，\(T=S_-+S_+\)，\(U=S_+-T/2\)。所有初始化、探索、利用和错误停止的样本均计费；没有免费先验观测。

不同角度的子块必须分别保存。把不同角度的数据合成一个“同分布 Bernoulli 块”会破坏下面的条件分布论证。公共外层检查点可以晚于子块完成时刻；增加可用证据不等于授权在未登记时刻停止。

## 2. 共享预测分子的 universal inference

设 \(q_b(y_b\mid\mathcal F_{b-1})\) 是块开始前指定、对该块所有可能结果正规化的预测分布，\(Q_k=\prod_{b\le k}q_b(Y_b\mid\mathcal F_{b-1})\)。令 \(L_k(\theta)\) 为同一观测协议下的真实似然。可预测查询和内部随机化的共同条件概率在比值中抵消。

对每个真实 \(\theta\)，
\[
R_k(\theta)=Q_k/L_k(\theta)
\]
是初值 1 的非负鞅；若存在零概率支持边界，则采用相应的超鞅版本。对某个方向的复合零假设 \(\Theta_0^s\)，
\[
E_k^s=Q_k/\sup_{\vartheta\in\Theta_0^s}L_k(\vartheta)
\le R_k(\theta),\qquad\theta\in\Theta_0^s.
\tag{1}
\]
因此不要求 profile 比值本身是鞅；它被一个真参数似然比鞅支配，足以使用 Ville 不等式。

两个方向共用同一个 \(Q_k\) 时可以统一使用 **\(1/\delta\)**，无需 \(2/\delta\)。当 \(C\ne0\) 时只有一个方向是错误方向；当 \(C=0\) 时两侧零假设均包含同一个真参数，所以所有错误越界都被同一个 \(R_k(\theta)\) 越界事件支配。若两方向使用不同分子，则这一共同支配不能直接沿用。

数值认证需要分子的下界与 null 最大似然的上界。未经认证的局部最优值通常只是最大似然的下界，放在分母会夸大证据。`certified_likelihood.py` 以凹似然切平面和物理域支撑函数构造全局上界；优化器只提供切点，不负责证明。

### 可复现的预测分布与遗憾

固定 PSR 下，单臂已有成功/失败数 \(s,f\)，当前块有 \(x,m-x\)，Jeffreys 后验预测的有序似然为
\[
q(x\text{ 的有序结果}\mid s,f)
=\frac{\mathrm B(s+x+1/2,f+m-x+1/2)}{\mathrm B(s+1/2,f+1/2)}.
\tag{2}
\]
若记录无序计数，需同时乘上二项系数；分子分母采用相同约定时该系数抵消。块内联合积分避免把同一 posterior mean 固定使用整块造成的额外预测损失。固定角度的真实历史使这些因子精确望远镜相消为一个全程 Bayes marginal。

改变角度后，旧臂计数不能冒充新臂的同参数 posterior。可以由过去的结构拟合和设计杠杆值选择伪成功/失败数，再用 (2) 作为正规化预测分布。其有效性只依赖可预测性与正规化；这种做法是预测启发式，不是自动正确的结构后验。伪计数不能进入观测分母或算作真实 shots。广义 Bayes 的幂似然本身也不是合法预测密度，必须先构造正规化预测分布。

完整双臂 Bayes numerator 在常规固定内点下付出约 \(\log n\) 的两参数编码遗憾，而配对符号 Beta 只需约 \(\tfrac12\log D\)，\(D\) 为不一致对数。小信号时完整 KL 与配对增长率已很接近，额外 nuisance 代价足以使前者更慢；本轮固定 PSR 探针正显示此问题。该近似只用于解释，不是对全部边界或自适应设计的统一遗憾定理。

不同合法预测器可通过固定初始权重的 Bayes model averaging 合成一个共同分子，或在每块开始前用过去的 log loss 更新正规化预测权重。不能看完当前块后选最大似然预测器，再把它当成事先预测的密度；也不能事后重新分配已有证据资本。

## 3. 条件 Fisher：去掉公共成功概率

对于一个同角、均分块，条件于 \(T\) 后，\(S_+\) 的分布是 Fisher 非中心超几何：
\[
f_\omega(s\mid T,m)=
\frac{{m\choose s}{m\choose T-s}\,\omega^s}
{Z_{m,T}(\omega)},\quad
Z_{m,T}(\omega)=\sum_{j=\max(0,T-m)}^{\min(m,T)}{m\choose j}{m\choose T-j}\omega^j,
\]
\[
\omega=\frac{p_+/(1-p_+)}{p_-/(1-p_-)}.
\tag{3}
\]
这里组合系数必须保留在归一化中。当前 \(T\) 可以是随机的：只要对每个 \(T\) 都正规化，并在条件期望之后再对 \(T\) 积分，就能得到合法的块证据。反复查看普通 Fisher 固定样本 p 值不能替代这个构造。

令 \(\lambda=\log\omega\)，在中央超几何下定义
\[
K_b(\lambda)=\log\mathbb E_0\exp\{\lambda(S_+-T/2)\}.
\]
它凸、非负，且因均分设计的对称性为偶函数。固定专家 \(\lambda_j\) 的累计资本为
\[
L_{j,k}=\exp\left\{\lambda_j U_k-\sum_{b\le k}K_b(\lambda_j)\right\},
\qquad U_k=\sum_{b\le k}(S_{+,b}-T_b/2).
\tag{4}
\]
固定 PSR 时真实 \(\omega\) 不变，固定先验混合 \(Q_{\rm cond}\) 与两侧 odds-profile 可以直接使用 (1)。因为累计 log likelihood 对 \(\lambda\) 凹，且在零处导数为 \(U_k\)，当 \(U_k>0\) 时 null \(\lambda\le0\) 的 profile 恰在零处；反向同理。因此按 \(\sum_j w_jL_{j,k}\ge1/\delta\) 和 \(U_k\) 的符号决策，无需数值求解 profile。

## 4. 自适应角度也安全的带符号专家引理

真实 odds 随角度改变时，不能继续声称它是一个固定参数的条件似然。以下证明只需要共同排序，仍适用于这样的自适应设计。

设固定权重的专家分为正、负和中性三组，权重总和为 1。要求：真方向为正时，所有负专家均是初值 1 的非负超鞅；在错误符号事件 \(U_k<0\) 上，所有正专家资本不超过 1。反方向条件对称；在零变化下全部专家为非负超鞅。令
\[
M_k=w_0+\sum_{j\in J_+}w_jL_{j,k}+\sum_{j\in J_-}w_jL_{j,k}.
\]
则按 \(M_k\ge1/\delta\) 且 \(U_k\ne0\) 决策，总错误风险至多 \(\delta\)，即使允许任意停止。

证明：真方向为正、错误符号为负时，
\[
M_k\le 1-w_-+\sum_{j\in J_-}w_jL_{j,k}.
\tag{5}
\]
右边是初值 1 的非负超鞅，Ville 不等式给出风险界。若 \(w_->0\)，用归一化负专家混合还得到更细的界
\[
\Pr(\text{错误负向越界})\le
\frac{w_-\delta}{1-\delta(1-w_-)}\le\delta.
\tag{6}
\]
真方向为负对称；零变化时直接对整个 \(M_k\) 使用 Ville。

对于 (4)，真 odds 满足 \(\omega_b\ge1\) 时，负 \(\lambda_j\) 的单块因子是 \(S_+\) 的递减函数。超几何族的单调似然比性质给出其在真实条件分布下的期望不超过中央分布下的 1，故负专家乘积为超鞅。同时错误 \(U_k<0\) 时，正专家由 \(K_b\ge0\) 立即满足 \(L_{j,k}\le1\)。这证明相同阈值在可预测换角下仍然有效。

专家的 \(\lambda_j\) 与初始权重必须全程固定；不能因为已看到结果就重新启动先验或改变旧资本权重。相反，正常 Bayes posterior 更新只是同一个固定混合的另一种计算方式，不构成重新启动。网格稀疏时可能没有正增长专家，必须保留满 cap 弃权，不能声称对所有小信号均有功效。

## 5. 固定 odds、可预测 nuisance 的 full-safe 网格

另一种可运行的强对照来自已有 Turner–Ly–Grünwald full-pair safe 因子。为每个固定有理 \(\omega_j>1\)，用过去数据预测一个有理 \(u_b\in(0,1)\)，再精确设置
\[
v_b=\frac{\omega_j u_b}{1-u_b+\omega_j u_b},\qquad r_b=(u_b+v_b)/2.
\]
正专家单对因子为
\[
S_j^+=\frac{f_{u_b}(Y^-)f_{v_b}(Y^+)}{f_{r_b}(Y^-)f_{r_b}(Y^+)},
\]
负专家严格交换同一组 \(u_b,v_b\)。已有有序零假设期望界保证反向专家为超鞅。并且逐结果都有
\[
\log(S_j^+/S_j^-)=\log\omega_j\,(Y^+-Y^-),\qquad S_j^+S_j^-\le1.
\]
所以 \(\log S_j^+\le\tfrac12\log\omega_j(Y^+-Y^-)\)。跨块累计后，错误的 \(U_k<0\) 使全部正专家资本不超过 1，即可应用第 4 节，无须旧实现的方向分拆和额外固定 fallback 权重。

`safe_grid.py` 用稳定二次方程尽量保持过去拟合的中心 \(q\)，把下端点有理化到固定分母 \(2^{40}\)，然后按上述有理公式求上端点。上端点不能独立四舍五入，否则固定 odds 的代数恒等式可能被破坏。两个方向也不能各自重新估计端点。变化的 nuisance 预测只影响效率，不影响上述共同排序证明。

该因子实际上只依赖两侧成功数；任意重新配对得到的四格计数都有同一似然。因此它可公平使用与条件 Fisher 相同的两侧完整计数。它仍可能因 nuisance 预测不准而损失资本；“没有方向 split”不是成本必胜的证明。

## 6. 信息收益、先验和公平比较

对单个固定 \(m/m\) 块，设 \(q=(p_-+p_+)/2\)，\(I=[\operatorname{kl}(p_-\Vert q)+\operatorname{kl}(p_+\Vert q)]/2\)。在知道真实 odds 的 oracle 情况下，KL 链式分解给出
\[
\mathbb E_\theta\log\frac{f_{\omega_\theta}(S_+\mid T)}{f_1(S_+\mid T)}
=2mI-\operatorname{KL}(\mathcal L_\theta(T)\Vert\operatorname{Bin}(2m,q)).
\tag{7}
\]
条件化会丢弃 \(T\) 的信息，并非“完整数据无损”。但损失项有简单上界：二项类型概率下界、\(\operatorname{kl}(x\Vert q)\le(x-q)^2/[q(1-q)]\)，以及 \(\operatorname{Var}(T)\le2mq(1-q)\) 给出
\[
0\le\operatorname{KL}(\mathcal L_\theta(T)\Vert\operatorname{Bin}(2m,q))
\le\log(2m+1)+1.
\tag{8}
\]
因此 oracle 大块条件证据每 shot 的信息损失趋零；\(m=1\) 时则恰为不一致符号的 Bernoulli odds 证据。这个信息空间解释了探索大块条件法的理由，却没有证明有限固定网格、未知 odds 或给定停止规则达到 oracle 界。

一个固定先验混合满足 \(\log M\ge\log L_j+\log w_j\)。先验权重小会付出明确的资本代价；CF33 的均匀 log-odds 网格与 exact Jeffreys Beta 的先验并不相同。因此后续比较同时保留：

| 对照 | 用途 |
| --- | --- |
| exact Jeffreys 配对 Beta | 原有无 nuisance 学习的强基线 |
| 相同 odds/权重的配对网格 | 区分先验离散化收益与完整块计数收益 |
| 条件 Fisher 网格 | 消去每块公共概率，保留条件两臂计数信息 |
| fixed-odds full-safe 网格 | 同样先验、同样总体阈值；用过去数据预测 nuisance |
| 双臂 Jeffreys UI | 明确保留已发现的 nuisance-penalty 负结果 |

CF71 是固定 Jeffreys CDF 分箱的有理近似，不是 exact 连续 Jeffreys 混合；其极弱信号功效仍受有限支持限制。所有机制必须使用相同代价、外层停止时刻、角度学习资源和错误/弃权计费。调整开发先验后必须另作新确认，不能覆盖旧结果。

若两类证据共享同一带符号统计量且都满足第 4 节条件，固定权重组合也安全。例如均分块中配对 Beta 与条件 Fisher 的固定混合保留 \(M\ge wM_{\rm Beta}\)，代价最多是 \(-\log w\) 的证据损失；这不是确定的 shots 回归上界。

## 7. 不等分块的扩展边界

每块两侧样本数为 \(n_-,n_+\) 时，条件超几何的中央均值为 \(Tn_+/(n_-+n_+)\)。定义该中心下的累计 \(U\) 和 CGF，仍有 CGF 非负及单调似然比，故固定带符号专家引理继续成立；CGF 一般不再为偶函数，正负归一化不能直接共用。`conditional_fisher_unequal.py` 单独处理这一情形。

第 5 节 full-safe 的配对几何平均证明针对均分独立配对，不直接推出不等分版本。同样，不等分条件 score 与旧配对 Beta score 不同，不能未经证明就沿用二者的共同符号 hedge。

## 8. 必要文献核对与主张边界

- Wasserman、Ramdas 与 Balakrishnan 的 [Universal Inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC7382245/)（2020）Section 7、Theorem 9 明确用 running likelihood numerator 与 null 最大似然构造被真参数鞅支配的过程；也允许用最大似然的上界。本文的可预测查询证明沿用该原则，共享分子处理双方向的论证已在上面逐项给出。
- Grünwald、de Heide 与 Koolen 的 [Safe Testing](https://arxiv.org/html/1906.07801v5) 已系统研究含 nuisance 的 GRO e-values、条件 e-variable 和可选延续。不能把“先验混合”“安全 hedge”或“处理 nuisance”本身当作新统计框架。
- Turner、Ly 与 Grünwald 的 [Generic E-Variables for Exact Sequential k-Sample Tests](https://arxiv.org/html/2106.02693v3) 给出块证据及全计数 safe 方法，并明确指出反复查看普通 Fisher p 值的问题。其目标 Bayes-factor 讨论不把固定成功数的 hypergeometric sampling scheme 作为本文中的 block design；这里对随机 \(T\) 的处理依赖第 3–4 节的条件期望证明，而非仅借用 Fisher 的名称。
- Degenne、Koolen 与 Ménard 的 [Non-Asymptotic Pure Exploration by Solving Games](https://arxiv.org/abs/1906.10431)（NeurIPS 2019）已给出一般结构化指数族识别的游戏分配与有限置信分析。当前固定网格和双角策略未完整复现该类 structured pure-exploration 算法，不能把它们称作全部结构化 SOTA。
- Liu、Qin 与 You 的 [Pure Exploration via Frank-Wolfe Self-Play](https://arxiv.org/html/2509.19901v1)（2025 预印本）研究结构化分配游戏；其广义讨论包含 Bernoulli 指数族，但主要收敛分析和实例聚焦 linear-bandit BAI。它不是已经运行的本项目 Bernoulli 单频基线，也不能仅依据摘要名称宣称被超过。
- Zecchin、Simeone 与 Ramdas 的 [Quantum Sequential Universal Hypothesis Testing](https://arxiv.org/html/2508.21594v1)（2025 预印本）Theorem 4.1 针对自适应量子测量策略给出 type-I 控制；它使用一般复合量子态与局部/联合测量，资源模型与这里的固定 Bernoulli 查询不同。自适应量子 universal inference 已有直接先例。

上述原文已于本轮通过只读网络访问核对。搜索连接器不可用时改为读取公开原站，没有把摘要检索视为完整算法复现。当前能够主张的是：给定假设下的有效性推导、经认证的新实现，以及后续冻结实验实际支持的比较。具体 signed-mixture 组合的文献优先性尚未证明，有限模拟领先也不等于国际 SOTA 已被超过。

审查记录见 [theory_review.json](../results/structured_validation/theory_review.json)；数值检查见 [safe_grid_checks.json](../results/structured_validation/safe_grid_checks.json) 与条件 Fisher 的相应检查文件。本文不替代实验审计或形式证明助手认证。
