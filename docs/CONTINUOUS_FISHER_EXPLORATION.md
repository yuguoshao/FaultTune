# 连续 Jeffreys 条件 Fisher：数学对象与可认证积分下界

状态：这是本轮冻结实验之外的探索。下文给出可逐项检查的数学论证，
实现及确定性检查另行记录；不新增 Monte Carlo，不改变已经冻结的前瞻
候选、对手、先验或样本数，不据此声称已验证超过 SOTA。这里的条件化、
Jeffreys 混合、凹性和 Ville 不等式均使用既有数学工具，文献新颖性未确认。

## 1. 固定的连续混合

在块 \(b\) 开始前选择两个设置及整数样本数 \(m_b\)。每侧有 \(m_b\)
次 Bernoulli 测量；给定过去历史和本块设置后，块内测量独立，每侧概率
分别为 \(p_{b,-},p_{b,+}\)。记

\[
T_b=K_{b,+}+K_{b,-},\qquad
U_t=\sum_{b\le t}(K_{b,+}-K_{b,-})/2,\qquad
N_t=2\sum_{b\le t}m_b.
\]

\(N_t\) 是 physical shots；\(U_t\) 是配对方向计数差的一半，不是完整计数差。
中央条件分布为 \(K_{b,+}\mid T_b\sim\mathrm{Hypergeom}(2m_b,T_b,m_b)\)，
其均值是 \(T_b/2\)。定义

\[
K_b(\lambda)=\log\mathbb E_0
 \exp\{\lambda(K_{b,+}-T_b/2)\},\qquad
L_t(\lambda)=\exp\{\lambda U_t-\sum_{b\le t}K_b(\lambda)\}.
\]

这里 CGF 的参数 \(T_b\) 使用本块真实观察到的总成功数。中央 CGF 凸、
非负；均分块下还关于 \(\lambda\) 偶对称。为避免混淆，\(K_b\) 在本式中
表示 CGF，而 \(K_{b,+},K_{b,-}\) 表示计数。

令 \(q=e^\lambda/(1+e^\lambda)\)。将 \(q\) 上固定的
\(\mathrm{Beta}(1/2,1/2)\) 先验换元，得到 log-odds 密度

\[
\rho(\lambda)=\frac{e^{\lambda/2}}{\pi(1+e^\lambda)}
              =\frac{1}{2\pi\cosh(\lambda/2)},\qquad
\int_{\mathbb R}\rho(\lambda)d\lambda=1.
\]

这里“Jeffreys”特指 \(q=\operatorname{logistic}(\lambda)\) 上 Bernoulli
模型的 \(\mathrm{Beta}(1/2,1/2)\) 先验，不宣称它是任意块设计下非中心
超几何模型自身的 Jeffreys 先验。

换元中的 Jacobian 是 \(dq/d\lambda=q(1-q)\)，不能漏掉。研究对象始终是

\[
M_t=\int_{\mathbb R}L_t(\lambda)\rho(\lambda)d\lambda.
\tag{1}
\]

这是固定连续先验的混合，不是离散专家权重随数据更新的另一个模型。
有限数据时，每个条件似然比至多为观察值中央概率的倒数，因此 (1) 有限。

## 2. 凹性给出逐路径积分下界

设

\[
g_t(\lambda)=\lambda U_t-\sum_b K_b(\lambda)+\log\rho(\lambda).
\]

由于

\[
(\log\rho)''(\lambda)
=-\frac{1}{4\cosh^2(\lambda/2)}<0,
\]

\(g_t\) 严格凹。任取严格递增的节点
\(\lambda_i=\log\omega_i\)，其中 \(\omega_i>0\) 为精确有理数。
若经向外舍入计算得到 \(a_i\le g_t(\lambda_i)\)，凹性保证每个区间上
\(g_t\) 不低于连接 \((\lambda_i,a_i)\) 与
\((\lambda_{i+1},a_{i+1})\) 的直线。因此

\[
\int_{\lambda_i}^{\lambda_{i+1}}e^{g_t(\lambda)}d\lambda
\ge (\lambda_{i+1}-\lambda_i)
\begin{cases}
\dfrac{e^{a_{i+1}}-e^{a_i}}{a_{i+1}-a_i},&a_{i+1}\ne a_i,\\[6pt]
e^{a_i},&a_{i+1}=a_i.
\end{cases}
\tag{2}
\]

差值两项同号，所以商始终为正。相加互不重叠的相邻区间并舍弃外侧正质量，
得到完整 (1) 的下界。实现还必须对区间长度、指数、差商和求和进行严格
向下认证；端点接近时不能依赖不稳定的浮点相减。

例如令 \(d=|a_{i+1}-a_i|\)，差商也等于
\(e^{\max(a_i,a_{i+1})}(1-e^{-d})/d\)。这允许先减去公共 log offset，
避免计算很大的正指数；\(d\to0\) 使用可认证级数或极限。

节点可以根据已经观察到的数据自适应选择，因为对每条数据路径，(2)
仍然界住同一个固定的 (1)。这些节点只是积分的计算位置，不能把它们
当成事后重新加权的先验。不得重复累加重叠区间，也不得按数据依赖的
保留先验质量重新归一化。截断积分下界自身不必是超鞅。

## 3. 一个阈值控制方向错误

假设所有块具有共同真实排序。以 \(p_{b,+}\ge p_{b,-}\) 为例，
负方向决定是错误。条件非中心超几何族的单调似然比性质保证，对固定
\(\lambda<0\)，\(L_t(\lambda)\) 是非负超鞅：给定任何 \(T_b\)，其
递减似然比因子在真实条件分布下的期望不超过中央分布下的 1。

在 \(U_t<0\) 上，所有 \(\lambda\ge0\) 满足
\(L_t(\lambda)\le1\)，因为中央 CGF 非负。因此此事件上

\[
M_t\le V_t^-:=\frac12+
       \int_{-\infty}^0L_t(\lambda)\rho(\lambda)d\lambda.
\]

Tonelli 与条件期望保证 \(V_t^-\) 是初值为 1 的非负超鞅。
若任何经过认证的下界 \(B_t\le M_t\) 满足
\(B_t\ge1/\delta\)、\(U_t<0\)，就必有 \(V_t^-\ge1/\delta\)。
Ville 不等式给出任意时刻错误停止概率不超过 \(\delta\)。
反方向交换两侧即可。当每块两率相等时，整个 \(M_t\) 是初值为 1 的
非负鞅，对任一方向越界的合并概率也不超过 \(\delta\)。

因此 \(\delta=0.01\) 时完整资本或其认证下界的阈值是 **100**；
log 资本的阈值是 \(\log100\)。不需要再分拆两方向风险，也不能用
未经认证的浮点积分直接停止。\(U_t=0\) 不作方向决定。

## 4. 每块每侧一次测量时，完整积分恰等于 Beta

若每个非空块都有 \(m_b=1\)，一致对的 CGF 与 score 都为零。
不一致对有 \(T_b=1\)，中央 CGF 是 \(\log\cosh(\lambda/2)\)。
令 \(b\) 为正侧独有成功的对数，\(c\) 为负侧独有成功的对数，则

\[
L_t(\lambda)=(2q)^b[2(1-q)]^c,
\quad
M_t=2^{b+c}\frac{\mathrm B(b+1/2,c+1/2)}{\mathrm B(1/2,1/2)}.
\tag{3}
\]

这正是原 exact Jeffreys 配对 Beta 资本，还可写成精确有理数

\[
M_t=\frac{(2b)!(2c)!}{2^{b+c}b!c!(b+c)!}.
\]

一致对仍消耗两次测量。等式 (3) 指完整连续积分；有限区间、有限弦段
计算出的下界一般严格更小，不能称作与 Beta 数值完全相等。

## 5. 两个独立的单块闭式核查点

单个 \(m=2,T=2\) 块的中央分布是 \((1,4,1)/6\)。完整 (1) 为

\[
M(0,2)=M(2,0)=2\sqrt6-3,
\qquad M(1,1)=3-\sqrt6,
\tag{4}
\]

括号按 \((K_-,K_+)\) 排列。其中央加权平均严格为 1。
这提供不调用 CGF 例程的独立参考。

推导：令 \(x=\sqrt\omega\)，先验测度成为
\(2\,dx/[\pi(1+x^2)]\)。极端块因子是
\(6x^4/(1+4x^2+x^4)\)，中心块因子是
\(6x^2/(1+4x^2+x^4)\)。利用部分分式以及

\[
\int_0^\infty\frac{dx}{x^4+4x^2+1}
=\int_0^\infty\frac{x^2dx}{x^4+4x^2+1}
=\frac{\pi}{2\sqrt6}
\]

得到 (4)。最后一个积分等式可由 \(x\mapsto1/x\) 的对称性和
\(u=x-1/x\) 换元直接证明。这些是确定性恒等式，不是模拟性能证据。

## 6. 理想连续混合的一个精确功效充分条件

仅有共同排序不够保证功效：例如选角可使真实两率之差趋于零，累积信息
也可能不足。以下给出一个较强但明确的充分条件。

假设反事实地持续执行可预测的有限均分块，
\(n_t=\sum_{b\le t}m_b\to\infty\)，且存在固定 \(d>0\)，每块都有
\(p_{b,+}-p_{b,-}\ge d\)。中央超几何是从 \(2m_b\) 个 0/1 数中不放回
抽取 \(m_b\) 个数。Hoeffding 的不放回抽样比较与有界变量引理给出

\[
K_b(\lambda)\le m_b\lambda^2/8.
\tag{5}
\]

具体地，不放回抽样的指数矩不超过同一总体有放回抽样的指数矩，后者
中心化后的 log 矩不超过 \(m_b\lambda^2/8\)。
将每对的 \((Y-X)/2\) 按实际采样顺序展开，给定过去的期望至少为
\(d/2\)，鞅差有界。因此有界鞅强大数律保证

\[
\liminf_t U_t/n_t\ge d/2\quad\text{几乎必然}.
\]

对任何固定 \(0<\lambda<4d\)，由 (5)

\[
\liminf_t\frac{\log L_t(\lambda)}{n_t}
\ge\lambda d/2-\lambda^2/8>0.
\tag{6}
\]

为严格转到积分，选固定 \(0<a<b<4d\)，并令
\(c=\min\{ad/2-a^2/8,bd/2-b^2/8\}>0\)。对足够小的
\(\varepsilon>0\)，几乎必然最终对全部 \(\lambda\in[a,b]\) 同时有
\(\log L_t(\lambda)\ge n_t(c-b\varepsilon)\)。因该区间先验质量严格正，

\[
M_t\ge\left(\int_a^b\rho(\lambda)d\lambda\right)
          e^{n_t(c-b\varepsilon)}\longrightarrow\infty.
\]

式 (6) 的单位是每对，即每 \(n_t\)；换成每 physical shot 必须再除以 2。
例如 \(\lambda=2d\) 给出至少 \(d^2/2\) 每对、\(d^2/4\) 每 shot 的
渐近下界。这是充分下界，不声称最优信息率。

该论证保证理想连续混合的首次方向停止时间有限，几乎必然成立；错误
方向的概率仍至多 \(\delta\)。它**不保证首次停止几乎必然正确**，因为
可能较早发生错误越界。若反事实继续采样，则最终 score 为正且资本增长。
固定 \(C\ne0\) 且查询的正侧 \(\sin x\) 有统一正下界的当前物理模型
满足统一 gap 条件；负方向对称。

此结论没有自动传递给有限节点、固定截断区间、固定数值精度或固定 shot
cap 的计算方案。要保证这些方案也终止，需要另证节点细化、截断与舍入
损失受控。当前探索不作这一额外功效承诺，也不作停止费用优于强基线的承诺。

## 7. 相对任意固定单专家的逐路径对数损失界

连续先验还给出一个不依赖真实参数的定量结论。令
\(n=\sum_bm_b=N/2\)。对任意已经观察到的有限历史，
\(|U|\le n/2\)。每个中央条件变量 \(K_+-T/2\) 的支持都位于
\([-m_b/2,m_b/2]\)，故其任意指数倾斜均值，也就是 \(K'_b(\lambda)\)，
仍在该区间中。因此

\[
| (\log L)'(\lambda)|\le n,\qquad
| (\log\rho)'(\lambda)|\le1/2,
\qquad |g'(\lambda)|\le n+1/2.
\]

任取有限 \(\lambda_*\in\mathbb R\)，全局 Lipschitz 性保证
\(g(\lambda)\ge g(\lambda_*)-(n+1/2)|\lambda-\lambda_*|\)。
对这个双侧指数下包络直接积分得到

\[
\boxed{\quad
M\ge\frac{2e^{g(\lambda_*)}}{n+1/2}
 =\frac{L(\lambda_*)}{\pi(n+1/2)\cosh(\lambda_*/2)}
 =\frac{2L(\lambda_*)}{\pi(N+1)\cosh(\lambda_*/2)}.
\quad}
\tag{7}
\]

等价地，完整连续混合相对该单专家的 log 资本损失满足

\[
\log L(\lambda_*)-\log M
\le\log\!\left(\frac{\pi(N+1)}2\right)
  +\log\cosh(\lambda_*/2).
\tag{8}
\]

对每个固定的有限 \(\lambda_*\)，右侧是 \(O(\log N)\)；一个未包含该
专家的固定离散网格则可能有随 \(N\) 线性增长的近似损失。(7)–(8)
逐路径成立，所以其中的 \(\lambda_*\) 也可在看到数据后选择；此时仍须
保留实际的 \(\log\cosh(\lambda_*/2)\) 项，不能对任意增长的
\(\lambda_{*,N}\) 宣称与位置无关的 \(O(\log N)\) 界。

这不是关于最优专家端点 \(\lambda=\pm\infty\) 的无条件界，也不是平均
停止费用的上界。它针对完整积分 \(M\)，当前仅积分有限弦段的计算值
\(Q\le M\) 不会自动获得同一保证。若未来将 (7) 作为另一条数值认证
路径，也必须另行认证其端点及常数；本节没有修改已冻结的新实现。
