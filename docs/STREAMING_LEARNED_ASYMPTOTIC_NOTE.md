# 付费探索、岭拟合与连续 Beta 验收的渐近信息率

2026 年 11 月 3 日。以下给出 `guarded_closed_beta` 的一个**无限数学延伸版本**的可审查证明：付费的平方根探索足以使自适应岭拟合一致；在明确的投影条件下，实际学习后的配对符号证据达到先前 oracle 角度的渐近信息率。此结论没有证明固定预算内的成本优势，也没有证明期望停止时间最优。

本稿只新增理论文档；没有改变冻结运行源码、计划或实验结果。对应源码为 [run_guarded.py](../experiments/streaming_validation/run_guarded.py)、[designs.py](../experiments/streaming_validation/designs.py) 与 [benchmark.py](../experiments/streaming_validation/benchmark.py)。它们的 SHA256 分别为 `0364781f54f0368af3de4fef0765cfe8d642e5e988a7d73ae5d4d46823f25cbc`、`518139aa874db8533fc9ca4b1417b5025a488745e4c20548d9bc344512ab0a26`、`19fe60408a74b68e18dbea4f40472ee448b608be2accabe39c024a3ede0f486f`。

## 1. 模型及无限延伸的含义

固定真实参数 \(\theta=(A,B,C)\)，令
\[
p_\theta(x)=A+B\cos x+C\sin x,\qquad
0<A<1,\quad B^2+C^2<\min(A^2,(1-A)^2),\quad C\ne0.
\]
每一对观测分别在 \(-t,+t\) 独立采集，角度与子块样本数在该子块开始前确定，只依赖已经完成的付费观测。允许自适应，但观测噪声必须在给定过去与查询角度后独立服从相应 Bernoulli 分布。

冻结实验的总预算上限是 \(2^{22}\) shots，风险参数是 \(0.01\)，因此它本身没有 \(n\to\infty\) 的运行路径。这里定义如下数学延伸：

1. 去掉有限总预算上限，并在背景路径上忽略停止指令，使数据继续产生；之后可在同一条路径上定义不同阈值的停止时间。
2. 延伸公共检查点递推 \(N_{k+1}=2\lceil1.125N_k/2\rceil\)。有限的附加检查点和前 4096 shots 的 PSR 阶段保持不变，均不影响极限。
3. 保持同样的探索规则：检查点 \(N_k\) 的累计探索目标为 \(T_k=\lceil4\sqrt{N_k/2}\rceil\) 对；本块可用对数限制新增探索，新增数量按向上/向下取整均分给 \(\pi/3,2\pi/3\)。完成两个探索子块后，用更新拟合选择余下样本的角度。
4. 保持 ridge=1、mean clip=0.0001、radial fraction=0.98 及 shift cap=0.2。数学论证采用精确实数统计量、无溢出整数及精确 Beta 证据。若讨论数值证书，可改为精度随规模增加、误差为 \(o(n)\) 的认证实现；不能把固定精度、有限整数类型的原代码直接当作无限精度算法。

所有探索、利用和初始化观测均收费，\(n\) 始终指累计物理 shots。以下结论对每个固定真实模型成立，不是对趋近物理边界的参数族给出的统一收敛速度。

## 2. 命题

定义源码中的连续投影映射
\[
a=\operatorname{clip}(A,\ell,1-\ell),\quad \ell=10^{-4},\quad
r=\sqrt{B^2+C^2},\quad
s=\min\!\left(1,\frac{\rho\min(a,1-a)}r\right),\quad \rho=0.98,
\]
\[
P(\theta)=(a,b,c)=(a,sB,sC).
\]
在 \(r=0\) 时按连续延伸定义 \((b,c)=(0,0)\)。令 \(t_0(a,b)\) 为 [oracle 理论笔记](STREAMING_KL_THEORY_NOTE.md) 式 (1) 的闭式角度，并记
\[
t_\infty=\operatorname{clip}(t_0(a,b),0.2,\pi-0.2).
\]
则上述无限延伸满足：

- 所有已完成观测的未投影岭估计 \(\widehat\theta_n\to\theta\) 几乎必然，并有
  \(\|\widehat\theta_n-\theta\|=O(n^{-1/4}\sqrt{\log n})\) 几乎必然。
- 投影估计趋于 \(P(\theta)\)，每个利用子块的角度趋于 \(t_\infty\)。探索角度持续出现，因此**整个逐对角度序列通常不收敛**；但探索占比趋于零。
- 源码中合并所有不一致符号所得的精确 Jeffreys Beta 证据 \(M_n\) 满足
  \[
  \frac{\log M_n}{n}\longrightarrow G_\theta(t_\infty)>0
  \quad\text{几乎必然},
  \tag{A}
  \]
  其中概率始终由原始真实参数 \(\theta\) 计算，不能用投影参数代替真实概率。

一个使极限等于原 oracle \(G_0\) 的充分条件是
\[
10^{-4}\le A\le1-10^{-4},\qquad
\sqrt{B^2+C^2}\le0.98\min(A,1-A).
\tag{B}
\]
此时 \(P(\theta)=\theta\)，而下文证明 0.2 cap 自动不生效，所以 \(t_\infty=t_0(A,B)\)，式 (A) 的极限为 \(G_0\)。条件 (B) 不是必要条件，但不能无条件删除：固定的 0.98 收缩和均值截断在其外一般会留下永久偏差。

## 3. 平方根探索给出最小特征值增长

单次物理查询的设计向量为 \(f(x)=(1,\cos x,\sin x)^\top\)。一对角度 \(\pm t\) 的 Gram 增量是
\[
Q(t)=f(t)f(t)^\top+f(-t)f(-t)^\top
=\begin{pmatrix}
2&2\cos t&0\\
2\cos t&2\cos^2t&0\\
0&0&2\sin^2t
\end{pmatrix}.
\]
若两类探索分别有 \(e_1,e_2\) 对，令 \(E=e_1+e_2\)、\(d=e_1-e_2\)，则
\[
G^{\rm explore}=
\begin{pmatrix}
2E&d&0\\ d&E/2&0\\0&0&3E/2
\end{pmatrix}.
\]
每个外层检查点的向上/向下拆分至多增加一次差额，所以 \(|d|\le k+O(1)=O(\log N_k)\)。也可直接取 \(m_e=\min(e_1,e_2)\)：
\[
G^{\rm explore}\succeq m_e\,[Q(\pi/3)+Q(2\pi/3)]
=m_e\operatorname{diag}(4,1,3),
\]
从而 \(\lambda_{\min}(G^{\rm explore})\ge(E-|d|)/2\)。

外层块的可用对数为 \(\Theta(N_k)\)，目标 \(T_k=\Theta(\sqrt{N_k})\)。因此从某个确定检查点起，单块容量本身已大于 \(T_k\)，可以补齐任意欠额，块末恒有 \(E=T_k\)。于是
\[
\lambda_{\min}(G_{N_k})\ge c\sqrt{N_k}
\quad\text{最终成立},\qquad E/N_k\to0.
\tag{C}
\]
其余观测只增加半正定 Gram。相邻预算之比有界，故对于块内部的已完成 pair 时刻，同样有 \(\lambda_{\min}(G_n)=\Omega(\sqrt n)\)。算法只在完成子块后更新拟合；这些更新时间是上述时刻的子序列，足以应用这个结论。

## 4. 自适应岭拟合一致的直接鞅证明

按实际物理查询逐个排列结果 \(Y_i\)，令
\[
\varepsilon_i=Y_i-f_i^\top\theta,\quad
S_n=\sum_{i\le n}f_i\varepsilon_i,\quad
G_n=\sum_{i\le n}f_if_i^\top,\quad V_n=I+G_n.
\]
角度可预测，且 \(\varepsilon_i\) 条件均值为零、有界。即使每对子块只以四类计数保存，源码的 Gram 与 rhs 仍精确等于这些逐次观测的充分统计量。

这里可以不把普通独立同分布最小二乘定理错误套到自适应数据上。对任意固定 \(v\in\mathbb R^3\)，保守的条件次高斯界给出
\[
\mathbb E\exp\!\left(v^\top S_n-\tfrac12v^\top G_nv\right)\le1.
\]
将 \(v\) 对标准三维 Gaussian 密度积分并交换非负积分，得到
\[
\mathbb E\left[\det(V_n)^{-1/2}
\exp\!\left(\tfrac12 S_n^\top V_n^{-1}S_n\right)\right]\le1.
\]
因此对 \(n\ge2\)，
\[
\Pr\{S_n^\top V_n^{-1}S_n>\log\det V_n+4\log n\}\le n^{-2}.
\]
由 \(\|f_i\|^2=2\)，有 \(\det V_n\le(1+2n/3)^3\)。Borel–Cantelli 给出
\(\|S_n\|_{V_n^{-1}}=O(\sqrt{\log n})\) 几乎必然。

源码的固定 ridge 先验中心是 \(\theta_c=(1/2,0,0)\)，所以
\[
\widehat\theta_n-\theta
=V_n^{-1}\{S_n+(\theta_c-\theta)\}.
\]
结合 (C)，
\[
\|V_n^{-1}S_n\|
\le\frac{\|S_n\|_{V_n^{-1}}}{\sqrt{\lambda_{\min}(V_n)}}
=O(n^{-1/4}\sqrt{\log n}),
\]
而先验偏差项为 \(O(n^{-1/2})\)。这证明未投影估计的一致性和所述速率。关键不只是 \(\lambda_{\min}\to\infty\)，而是这里已验证足够强的 \(\lambda_{\min}\gg\log n\)。

## 5. 投影与闭式选角的连续性；0.2 cap 的确切角色

均值截断及径向收缩映射 \(P\) 连续。投影后的均值属于 \([10^{-4},1-10^{-4}]\)，振幅至多为物理余量的 0.98 倍，因此闭式角度公式中的分母严格为正，角度映射也连续。连续映射定理直接给出利用角收敛至 \(t_\infty\)。

对当前 **closed variance** 公式，0.2 cap 在精确算术下实际上始终不生效。令 \(m=\min(a,1-a)\)、\(s_b=|b|/m\le\rho\)，则
\[
h=\frac{|(1-2a)b|}{a(1-a)-b^2}
=\frac{s_b(1-2m)}{1-m(1+s_b^2)}\le s_b\le\rho.
\]
这个不等式利用 \(1-2m\le1-m(1+s_b^2)\)。闭式解满足
\[
|\cos t_0|=\frac{h}{1+\sqrt{1-h^2}}
\le\frac{\rho}{1+\sqrt{1-\rho^2}}=0.8173495026\ldots,
\]
故
\[
0.6140008579\ldots\le t_0\le\pi-0.6140008579\ldots.
\tag{D}
\]
这个范围严格包含于代码允许的 \([0.2,\pi-0.2]\)。因此在当前投影之后，极限角可直接写作 \(t_0(P(\theta)_A,P(\theta)_B)\)。此结论只针对该闭式方差角，不推广到 Fisher KL 或 paired KL 网格策略。

条件 (B) 使投影极限为真参数，包括其等号边界；若要求投影最终完全关闭，则需要相应严格不等式，但证明极限为 \(G_0\) 不需要这个更强要求。

## 6. 变化角度下合并 Beta 证据的极限

写累计 pair 数为 \(m_n=n/2\)。对第 \(j\) 对，令
\[
a_j=(1-p_\theta(-t_j))p_\theta(t_j),\qquad
b_j=p_\theta(-t_j)(1-p_\theta(t_j)),\qquad d_j=a_j+b_j.
\]
利用子序列 \(t_j\to t_\infty\)，而初始化和探索的占比趋零。概率函数连续且有界，因此 Cesàro 平均满足
\[
\frac1{m_n}\sum_{j\le m_n}(a_j,b_j)
\longrightarrow(a_\infty,b_\infty).
\]
对应 01 和 10 指示量减去条件均值后是有界鞅差。Azuma 不等式加 Borel–Cantelli（或鞅强大数律）给出
\[
N_{01}/m_n\to a_\infty,\qquad N_{10}/m_n\to b_\infty
\quad\text{几乎必然}.
\]
真实模型严格物理，且 \(t_\infty\in(0,\pi)\)，所以 \(d_\infty=a_\infty+b_\infty>0\)，\(\alpha_\infty=a_\infty/d_\infty\in(0,1)\)，并且
\(\alpha_\infty-1/2=C\sin(t_\infty)/d_\infty\ne0\)。

令 \(D_n=N_{01}+N_{10}\)。源码使用的精确 Jeffreys 证据是
\[
M_n=2^{D_n}\frac{\mathrm B(N_{01}+1/2,N_{10}+1/2)}{\mathrm B(1/2,1/2)}.
\]
由于两个不一致类别的极限比例均严格为正，Stirling 公式给出
\[
\log M_n
=D_n\operatorname{kl}\!\left(\frac{N_{01}}{D_n}\middle\Vert\frac12\right)
+O(\log n).
\]
除以收费 shots 数 \(n=2m_n\)，就得到
\[
\frac{\log M_n}{n}\to
\frac{d_\infty}{2}\operatorname{kl}(\alpha_\infty\Vert1/2)
=G_\theta(t_\infty)>0.
\]
这一步没有假设各角度下的条件符号概率相同，也没有把各固定角度增长率简单相加。它通过最终占比集中到一个利用角，证明合并造成的信息损失在每 shot 极限中消失。

## 7. 可以继承的信息率保证，以及不能继承的停止成本主张

在条件 (B) 下，实际付费学习的无限延伸达到 \(G_0\)。因此 [oracle 理论笔记](STREAMING_KL_THEORY_NOTE.md) 的界可用于这个渐近增长率：
\[
\frac{I_*}{G_0}\le2F(\eta)(1+2KC^2)\le2F(\eta)(1+\eta^2).
\]
例如在额外满足 \(\eta\le0.1\) 时，其极限增长率至少为全部角度最优识别信息率的约 **95.65%**。这是补上“真实参数未知、探索必须收费”的一个严格信息率结论；固定投影域外只能先声称收敛到 \(G_\theta(t_\infty)\)，不能直接沿用这个比例。

还可以给出一个有限内容的路径式停止推论。固定真实 \(\theta\)，将风险阈值 \(\delta\downarrow0\)，沿同一条无限背景路径，在外层检查点第一次以**精确对数证据 \(\log M_n\)** 超过 \(\log(1/\delta)\) 时停止。因为符号最终正确且增长率 \(g=G_\theta(t_\infty)>0\)，有
\[
\frac1g\le\liminf_{\delta\downarrow0}\frac{N_\delta}{\log(1/\delta)}
\le\limsup_{\delta\downarrow0}\frac{N_\delta}{\log(1/\delta)}
\le\frac{1.125}{g}
\quad\text{几乎必然}.
\tag{E}
\]
证明只需对充分大的检查点夹住 \((g-\varepsilon)n\le\log M_n\le(g+\varepsilon)n\)，再用相邻预算比趋于 1.125。任意固定前缀的证据有限，所以阈值趋于无穷时首次越界也离开该前缀。式 (E) 同样适用于与精确证据相差 \(o(n)\) 的可靠数值认证；固定浮点筛选器的无限规模行为没有在这里验证。

**式 (E) 不是“1.125 渐近最优”。** 1.125 只来自当前稀疏检查点的确定性超额开销；\(g\) 也未必等于 \(I_*\)。几乎必然界不能直接取期望，故这里没有证明 \(\mathbb E N_\delta/\log(1/\delta)\) 的相应上界。要得到期望停时，需要补充尾部控制或一致可积性证明。

此外，固定 \(\delta\)、令 \(C\to0\) 是另一种极限：估计常数、投影是否生效与 Beta 混合代价可能共同变化，本稿没有证明这种极限下的成本最优性。有限预算实验的预定主指标是否通过，必须继续依据其冻结分析，不能用本渐近定理替代。

## 8. 审查状态

这里的 Gram、自归一化鞅、连续映射及合并计数证明是数学推导，没有新增 Monte Carlo。完整证明链条已由独立代理复核通过，包括 0.98 投影使 0.2 cap 对 closed variance 角不生效的代数，以及路径停止界与期望停时的区别；详见[独立审查记录](../results/streaming_validation/learned_asymptotic_review.json)。本文不是形式证明助手认证，也未完成具体定理的文献首创性判定。
