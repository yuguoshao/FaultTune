# FaultTune 相位观测与最优验收设计探索

2026 年 11 月 3 日。本轮获得了单频 Bernoulli 响应的显式最优观测设计和匹配方差下界，证明标准均匀相位移位设计在这一模型内至多损失两倍局部方差。另完成了 6,912 次探索运行：相位移位验收相对端点 A/B 的主格成本几何平均优势为 17.656 倍，但该优势来自已有 parameter-shift 方法，不能作为超过同信息 SOTA 的证据。当前没有新的全流程或有限预算 SOTA 胜出结果。

本轮代码位于 `experiments/phase_validation`，原正式实验的 31 个冻结 Python 文件保持不变。这里的同持续时间 Pauli 模型与原时长 1 和 2 的三比特模型分别讨论；新结果不替换原有负面确认结论。

## 原模型的未知退相干阻碍统一速率改善

原模型使用旧控制 `(0,1)` 和新控制 `(-omega,2)`。只在旧观测菜单之外增加 quadrature 分析，并不能普遍把成本从 \(O(\varepsilon^{-2})\) 降到 \(O(\varepsilon^{-1})\)。原因是改变持续时间引入了一阶未知退相干影响。

令 \(a=1-2r\)、\(q=1-r\)、\(s=\sin^2(\omega/2)\)，在隐藏相位耦合为零时，记 \(z=e^{-\kappa}\)，目标变化精确为

\[
\Delta=(q-as)\left(\frac{1+az}{2}\right)^2
-q\left(\frac{1+az^2}{2}\right)^2.
\]

取 \(r=0.01\)、\(s=2\varepsilon\)、\(0<\varepsilon\le0.01\)。两个世界使用同一组公开控制，只改变未知退相干率：好世界 \(\kappa=0\)，坏世界 \(\kappa=-\log(1-4\varepsilon)\)。于是

\[
\Delta_-=-1.920996\varepsilon,
\]

\[
\Delta_+=1.920996\varepsilon-19.171152\varepsilon^2
+53.321408\varepsilon^3-60.850944\varepsilon^4
\ge1.729223629056\varepsilon.
\]

对每个时长至多 2 的分析线路，两个退相干通道给出的输出分布 L1 差至多 \(4|\delta\kappa|\)。最终独立读出翻转保证八个输出概率均至少为 \(r^3\)，所以单次 KL 不超过 \(16(\delta\kappa)^2/r^3=O(\varepsilon^2)\)。若要求好世界接受概率至少 0.95、坏世界误接受至多 0.01，停止转录换测度给出

\[
\mathbb E_-N\ \ge
\frac{\operatorname{kl}(0.95,0.01)}{\sup_e D(P_-^e\Vert P_+^e)}
=\Omega(\varepsilon^{-2}).
\]

这允许自适应选择有界时长分析线路，不允许免费增加相干作用时长。二项换测度是标准工具；实质是本模型内的明确 nuisance 反例。精确多项式使用 `Fraction` 核验。

24 个预先列出的参数点的数值诊断也保留未知退相干造成的常数方差底。扩展 24 项相位菜单的均匀分配，相对保留全部八结果的原任务观测，局部方差改善仅为 1.0007 至 1.7265 倍。以某一点的 loss-bit 为对照可产生约 4.87 倍的表面改善，但同模型完整输出对照将它降到约 1.10 倍。Fisher 诊断不等于有限预算验收成本。

证据：[原模型相位信息诊断](../results/phase_information_probe.json)、[物理实现](../experiments/phase_validation/phase_model.py)、[精确代数检查](../experiments/phase_validation/theory_check.py)。

## 同一噪声通道下的标准相位移位验收

另取一次 Pauli 旋转 \(R_P(x)=e^{-ixP/2}\)，\(P^2=I\)。前后可以有任意未知但不依赖控制角 \(x\) 的固定 CPTP 通道，最终使用固定二元任务 POVM。响应必为

\[
f(x)=A+B\cos x+C\sin x.
\]

这里的目标是实际含噪任务损失，不是无噪声保真度。本分支固定持续时间，但这一点本身不足以保证共同固定通道；若驱动与同时发生的噪声不对易，不能自动断言整体响应为单频。

以旧、新角度 \(m-h/2,m+h/2\) 比较，令 \(s=\sin(h/2)\)，则

\[
\Delta=f(m+h/2)-f(m-h/2)
=s\,[f(m+\pi/2)-f(m-\pi/2)].
\]

这是已有 parameter-shift 公式的直接推论。共同未知读出、可见度和固定通道只改变 \(A,B,C\)，无需为使用这个恒等式预先估计它们。每次移位线路执行仍支付一个 shot；两侧测量噪声独立。

两个移位点各用 \(n\) 次观测，以 Hoeffding 固定终点检验可达到误接受至多 \(\delta\)、对 \(\Delta\le-\varepsilon\) 的接受概率至少 \(1-\beta\)。一个保守的充分设计为

\[
n=\left\lceil\frac{2s^2}{\varepsilon^2}
\left(\sqrt{\log(1/\delta)}+\sqrt{\log(1/\beta)}\right)^2\right\rceil.
\]

对 \(\delta=.01,\beta=.05\)，全部费用 \(2n\le60.118s^2/\varepsilon^2+2\)。此固定终点阈值不允许任意窥视；下述模拟使用已有任意时有效的 Beta 混合证书。

条件性匹配下界可以取

\[
f_\pm(x)=\tfrac12\pm\frac{\varepsilon}{2s}\sin(x-m).
\]

原任务端点的跨世界概率差为 \(\varepsilon\)，而任意角度的差至多为 \(\varepsilon/|s|\)。固定 \(0<r<1/2\) 并限制 \(\varepsilon/(|s|(1-2r))\le1/2\)，该族可由相反混合 Bloch 态、一次 Rx、Z 测量和共同正读出噪声实现。单拷贝量子相对熵给出任意独立重置探测的下界 \(\Omega(s^2\varepsilon^{-2})\)，端点固定任务测量的下界为 \(\Omega(\varepsilon^{-2})\)。相位移位达到前者的阶数；该下界不授予跨执行的相干量子访问。

固定 \(h\) 时仍是 \(\varepsilon^{-2}\) 速率；只有在 \(s^2=\Theta(\varepsilon)\) 的小更新实例族中才得到 \(\varepsilon^{-1}\)。新的测量权限带来这种分离，同信息的已有 PSR 方法也享有它。

### 探索运行结果

24 个主实例来自 \(h\in\{0.4,0.2,0.1\}\)、四组共同读出和退相干可见度、两个更新方向；另有三个零变化实例。每个方法每个实例重复 128 次，合计 6,912 次。公开检查点和预算上限仅依赖 \(h\)，每次独立配对付费两次，错误和弃权保留。

| 公开更新大小 h | 端点 A/B 成本除以移位验收成本的主格几何平均 |
| --- | ---: |
| 0.4 | 5.0526 |
| 0.2 | 16.7266 |
| 0.1 | 65.1240 |
| 全部 24 主格 | 17.6559 |

两种方法的全部主格均完成决定，各有一次错误决定；计入零变化诊断后，端点法共两次错误，移位法共三次错误。每格 128 次的探索样本不足以重新通过原正式实验的同时安全与功效资格。这些比值是探索均值，没有正式同时性能区间。

标准模型感知 PSR 加同一个停止证书与这里的移位方法完全相同，不能再把它列成一个被本方法击败的对手。物理恒等式另用 1,024 个含非幺正通道及不对称读出的 Bloch 模型交叉核验，最大残差 \(3.75\times10^{-16}\)。

证据：[运行脚本](../experiments/phase_validation/shift_validation.py)、[汇总](../results/phase_validation/shift_pilot_summary.json)、[原始记录](../results/phase_validation/shift_pilot_trials.jsonl)。

## 单频 Bernoulli 的显式最优设计

这一结果回答：在已有 PSR 的基础上，利用真实异方差重新挑角度和分配 shots，局部方差还能改善多少。

将中点平移为零。假设 \(0<A<1\)、\(R^2=B^2+C^2<\min(A^2,(1-A)^2)\)，三个系数都未知。每个独立角度执行只返回一个 Bernoulli 结果，单次费用相同。估计量要求对整个三参数开集线性无偏，目标为 \(2C\)；实际更新乘上已知 \(s\)，下列方差全部乘 \(s^2\)。这里 \(V\) 是总样本量乘估计方差的连续分配系数。

定义

\[
D=A(1-A),\quad L=1-2A,\quad
S=\sqrt{(A^2-R^2)((1-A)^2-R^2)},
\]

\[
u=\frac{D+B^2-C^2+S}{2},\qquad w=u-B^2.
\]

**最优值具有闭式**

\[
\boxed{V_*=4u=2\,[D+B^2-C^2+S].}
\]

最优设计只需两个对称角度 \(x_\pm=\pm\arccos z\)，其中 \(z=-LB/(2w)\)。以 \(\sigma_\pm=\sqrt{f(x_\pm)(1-f(x_\pm))}\) 表示真实标准差，分配比例为 \(\xi_\pm=\sigma_\pm/(\sigma_++\sigma_-)\)。这些 oracle 参数用于刻画最优值，实际算法需要由付费数据学习它们。

### 全角度下界与可达性证明

取对偶向量

\[
d=\left(\frac{LC}{2\sqrt u},-\frac{BC}{\sqrt u},\sqrt u\right).
\]

直接配平得到一个对所有角度成立的平方恒等式：

\[
f(x)(1-f(x))-[d_0+d_1\cos x+d_2\sin x]^2
=\frac{(u+C^2)w}{u}(\cos x-z)^2\ge0.
\]

其常数项由 \(4(u-B^2)(D-C^2-u)=L^2R^2\) 保证。又因为

\[
(D-R^2)^2-L^2R^2=S^2>0,
\]

有 \(w>|L|R/2\)，所以 \(w>0\)、\(|z|<1\)。对于任意分配测度 \(\xi\) 和线性无偏系数 \(a(x)\)，无偏条件为

\[
\int a(x)(1,\cos x,\sin x)\,d\xi=(0,0,2).
\]

将上面的对偶向量代入并用 Cauchy–Schwarz，得

\[
4u\le
\left[\int a(x)^2f(x)(1-f(x))d\xi\right]
\left[\int\frac{(d_0+d_1\cos x+d_2\sin x)^2}{f(x)(1-f(x))}d\xi\right]
\le V.
\]

在两个接触点 \(x_\pm\)，对偶表达式一正一负，且

\[
\sigma_++\sigma_-=2\sqrt u\sqrt{1-z^2}.
\]

按上述比例采样，用 

\[
\widehat{2C}=\frac{\bar Y_+-\bar Y_-}{\sqrt{1-z^2}}
\]

估计，方差系数恰为 \(4u\)，下界可达。证明对连续全部角度成立，不依赖有限网格。

### 两倍界限

均匀采样标准 \(\pm\pi/2\) 两点有 \(V_{\pi/2}=4(D-C^2)\)，因此

\[
\boxed{1\le\frac{V_{\pi/2}}{V_*}=\frac{D-C^2}{u}<2.}
\]

固定 \(A\ne1/2\)，令 \(B=0\)、\(|C|\uparrow\min(A,1-A)\)，比值趋近 2；常数不能统一改小。特别地 \(A=1/2\) 时均匀标准 PSR 已最优。

这个界限排除单频模型中单靠此类方差优化取得三倍收益。它不限制所有有限预算顺序检验，不适用于额外比特串信息、已知参数关系、不等观测费用或相干多次查询。

此前的 147 点、128 角网格仅用来发现和交叉检验闭式：相对标准 PSR 的最优双侧分配，最大改善为 1.9692；在响应全域属于 \([0.01,0.99]\) 的子网格中，最大为 1.3126。最终结论依据上述解析证明。另用 140 个严格内点和 80 位 Decimal 精度核验对偶及可达值，最大差 \(8.02\times10^{-78}\)。

证据：[闭式与独立检查](../experiments/phase_validation/design_bound_check.py)、[高精度核验结果](../results/phase_validation/design_bound_checks.json)、[网格原始结果](../results/heteroscedastic_design.json)。一般 c-optimal 与 Elfving 理论已有文献基础；本闭式的文献优先权尚未确定，不能据此称国际首创。

## 多频量子线路的补充筛查

为了检查单频两倍界限之外是否存在明显机会，另取两比特的公开控制
\(U(x)=\operatorname{diag}(1,e^{ix},e^{2ix},e^{3ix})\)，固定初态和投影给出最高频率为 3 的响应。固定种子生成 16 组态与投影，配合 \(r=.01,.05\) 两种读出，共 32 个参数点，角度菜单仍为 128 个公共点。推断时保留全部七个 Fourier 系数未知；私有态矢只用于产生物理分布。

异方差最优线性设计与“最小系数范数 PSR 加真实方差最优分配”比较，最大方差改善为 1.4726 倍，非零更新效果实例最多 1.2853 倍。还对全部同样最小范数的 PSR 规则进行方差择优，排除了因平局任选弱对照产生优势的可能。全部概率在 \([r,1-r]\) 内；物理响应、Fourier 表达和 LP 原始及对偶检查一致。

这是 32 点有限网格的探索结果，不是多频全域上限，也没有将 oracle 方差优势当作付费学习算法的性能。该网格没有为三倍改进提供继续扩大实验的依据。见 [多频检查](../experiments/phase_validation/multifrequency_design.py) 与 [完整结果](../results/multifrequency_design.json)。

## 模型失配的可认证阈值

设真实响应 \(f=g+e\)，\(g\) 为单频函数，仅知道全角度误差 \(\|e\|_\infty\le\eta\)。令 \(s=|\sin(h/2)|>0\)，则实际变化与移位预测的残差上界为

\[
R=2\eta(1+s).
\]

这是线性算子的标准范数界。要保证 \(\Delta\ge0\) 不被错误接受，接受阈值须额外扣除 \(R\)；要对所有 \(\Delta\le-\varepsilon\) 保有规定接受概率，充分样本公式中的有效间隙是

\[
\varepsilon-2R=\varepsilon-4\eta(1+s).
\]

这个仅移位观测的条件是尖锐的。以下反例取 \(0<h<\pi\)，使 \(s=\sin(h/2)>0\)。令中点为零，取

\[
g_\pm(x)=\tfrac12+(C_0\pm\eta)\sin x,
\qquad C_0=-\eta(1+s)/s.
\]

误差在 old/new/shift-plus/shift-minus 四点分别取

\[
e_\pm=\pm\eta\,(-1,+1,-1,+1).
\]

四点互异时可用不相交的周期光滑 bump 延拓，并保留 \(\|e\|_\infty\le\eta\)。两个世界的移位观测完全相同，但任务变化分别为 \(-4\eta(1+s)\) 和 0。因此在概率合法时，若

\[
\boxed{\eta\ge\eta_*:=\frac{\varepsilon}{4(1+s)},}
\]

仅靠两个移位点，即使无限采样也不能同时达到 95% 接受能力和 1% 误接受。零改善在这里属于不应接受的原假设。

进一步在 \(s^2=\varepsilon\le.01\)、\(\eta=\eta_*\) 的子族中，任意角度的两世界概率差均至多 \(4\eta_*\le\varepsilon\)。用角度依赖的 replacement channel 输出对角混合态可以严格实现这些响应；其 Kraus 算符为 \(K_{ij}=\sqrt{p_i}|i\rangle\langle j|\)。在概率远离 0、1 时，任意单拷贝 POVM 的 KL 仍为 \(O(\varepsilon^2)\)，故所有独立重置探测再次需要 \(\Omega(\varepsilon^{-2})\)。这条下界只适用于包含这些通道、仅限制响应误差的广泛模型类，不表示同样反例属于某个指定的小 diamond-distance 或 Lindblad 噪声族。

阈值以下，例如 \(\eta=c\varepsilon\) 且 \(4c(1+s)\le1-\nu\)，其中 \(\nu>0\) 固定，移位方法仍保留 \(O(s^2\varepsilon^{-2})\) 的充分上界。因此不能笼统地说所有 \(\eta=\Theta(\varepsilon)\) 都摧毁收益，也不能把未知模型误差界当作免费可得。

## 与已有方法的关系

| 工作 | 已覆盖内容 | 本轮定位 |
| --- | --- | --- |
| [Mari 等 2021](https://arxiv.org/abs/2008.06517) | 任意移位的精确导数及异方差统计公式 | 有限更新移位身份和相对端点的巨大收益属于其直接推论 |
| [Theis 2022](https://arxiv.org/abs/2112.14669) | 最小范数移位规则及有界偏差的紧放大界 | 不能将系数范数或失配残差上界作为新一般方法 |
| [Lai 等 2026](https://arxiv.org/abs/2508.08802) | 广义 PSR 与 weighted-shot 最优移位；方差最优性使用常方差假设 | 真实 Bernoulli 异方差须单独处理；本轮给出单频闭式最优设计 |
| [Degenne 等 2019](https://arxiv.org/abs/1906.10431) | 一般结构化指数族纯探索、KL 下界、任意时检验与渐近最优分配 | 有限角 Bernoulli 单频识别已属于其结构集合，不能重新命名为新通用统计理论 |
| [Réda 等 2021](https://arxiv.org/abs/2111.01479) | 已知有界失配下识别；超过阈值后失去结构优势；未知失配幅度的障碍 | 本轮四点阈值是明确的 Bernoulli 量子响应专化，退化机制已有 |
| [Özgüler 2026 预印本](https://arxiv.org/abs/2609.19090) | 观测盲区、补测、更新认证、漂移与读出估计成本 | 安全放行主线已有直接先行工作 |

上述对照同时说明：更快地复现已有移位公式，不等于超过同信息 SOTA；解析证明达到一个明确模型内的最优值，也不自动证明该公式具备文献首创性。

闭式设计的直接理论背景还包括 [Dette 与 Holland-Letz 的异方差 c-optimal 几何表征](https://arxiv.org/abs/0911.3801)、[Papp 的有理函数及三角回归最优设计](https://arxiv.org/abs/1009.1444)。[Das 等的量子测量加权精度界](https://arxiv.org/abs/2512.20091)也允许关注单一参数方向的半正定权重。此次检索尚未定位同一闭式和两倍界限，但这些一般框架必须作为先行工作保留。

## 后续投入的判据

保留闭式最优设计、失配阈值和精确物理反例作为可审查的理论成果。单频方差分配已受两倍界限约束，多频小网格也没有显示三倍空间，因此不继续增加这两组的重复次数。

下一项候选必须先给出超出这些模型限制的具体物理结构，例如额外可获得的输出信息、真实不等实验费用或多控制量的共享信息，并与获得相同信息的结构化识别和最优实验设计比较。只有付费学习、模型误差和停止证书全部计入后仍有优势，才重新设置确认实验；单纯相对端点 A/B 的数十倍改善不足以启动“超过 SOTA”的宣传。

## 审计与复现

运行与理论证据保存在 `results/phase_validation`。6,912 条记录均核对计数、费用、唯一性及汇总；6,147 个决定证书重算一致，独立 SciPy 数值交叉检查一致；六个完整向量种子单元共 768 条运行复现匹配。所有错误和弃权仍在记录中。原 31 个冻结 Python 文件与归档逐一核对未改变。

这些是实现和代数审计，不是 6,912 次运行全部无错误，也不是新的正式安全资格。见 [审计结果](../results/phase_validation/exploration_audit.json)。

在原项目 NumPy/SciPy 环境中，从项目根目录执行：

```powershell
python -B experiments/phase_validation/theory_check.py
python -B experiments/phase_validation/design_bound_check.py
python -B experiments/phase_validation/shift_validation.py --repetitions 128
python -B experiments/phase_validation/audit_exploration.py
```

本机 Python 选择和 DLL 设置沿用 [已有复现说明](../experiments/circuit_validation/README.md)。同一 NumPy 版本和批次形状用于种子复现；主机计数模拟耗时不代表硬件耗时。
