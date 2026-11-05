# 全参数 Bernoulli oracle 信息界：因子二界及其紧性

状态：下文给出一个完整数学证明，使用已独立审查的
[对称子类定理](CONTINUOUS_SYMMETRIC_DESIGN.md)及经典 KL 数据处理不等式。
数值表只是公式的确定性区间计算，没有新增 Monte Carlo。
文献新颖性尚未系统核查，不据此声称超过 SOTA。

这里比较的是**完整 Bernoulli 数据的 oracle 信息率**，不是当前 paired-beta、
RB 或其他具体证据过程的增长率，也不是有限样本停止费用。
第 7 节给出二者确实不同的例子。

## 1. 模型、信息设计值与结论

设
\[
p(x)=A+B\cos x+C\sin x,\qquad
0<A<1,\qquad B^2+C^2\le\min(A,1-A)^2.
\]
令
\[
m=\min(A,1-A),\qquad
\Phi(v)=\frac{1+v}{2}\log(1+v)+\frac{1-v}{2}\log(1-v),
\quad |v|\le1,
\]
其中 \(\Phi\) 为偶函数，\(\Phi(0)=0\)、\(\Phi(\pm1)=\log2\)。
零假设为全角度合法的仿射概率
\[
q_{a,b}(x)=a+b\cos x,\qquad |b|\le\min(a,1-a).
\]
每次查询费用相同，允许任意圆周概率测度 \(w\)。定义
\[
V(A,B,C)=\sup_w\inf_{a,b}
\int\operatorname{kl}(p(x)\Vert q_{a,b}(x))\,w(dx).
\tag{1}
\]
查询分配及下面构造的零假设可以使用真实参数，所以 (1) 是 oracle 对象。

均分 PSR 两角 \(x=\pm\pi/2\) 的完整数据信息值为
\[
E(A,C)=A\Phi(C/A)+(1-A)\Phi(C/(1-A)).
\tag{2}
\]

**定理 1。** 对全部允许的 \(A,B,C\)，
\[
\boxed{\qquad
E(A,C)\le V(A,B,C)\le U(A,C):=2m\Phi(C/m).
\qquad}
\tag{3}
\]
如果 \(C\ne0\)，还得到
\[
\boxed{\qquad V(A,B,C)<2E(A,C).\qquad}
\tag{4}
\]
因此，在这个完整 Bernoulli oracle 信息设计问题中，均分 PSR 的信息率
严格大于最优连续角度设计的一半。第 9 节证明，统一因子二不能改成任何更小
的常数。这个结论不要求最优设计本身只有两个支持点。

当 \(0<|C|<m\) 时，产生 (3) 上界的一个明确物理零假设是
\[
q^*(x)=A+B\lambda(C/m)\cos x,\qquad
\lambda(v)=\frac{\operatorname{atanh}v}{v}.
\tag{5}
\]
它逐角度满足 \(\operatorname{kl}(p(x)\Vert q^*(x))\le U(A,C)\)。
当 \(|C|=m\) 时，物理约束强制 \(B=0\)，可直接取 \(q^*\equiv A\)。
当 \(C=0\) 时，取 \(q^*=p\)，信息值为零。

## 2. PSR 下界是完整数据的精确计算

在 PSR 两角处，真实概率是 \(A+C,A-C\)，零假设的两率都为 \(a\)。
均分设计的目标为
\[
F(a)=\tfrac12\operatorname{kl}(A+C\Vert a)
     +\tfrac12\operatorname{kl}(A-C\Vert a).
\]
其与 \(a\) 有关的项为 \(-A\log a-(1-A)\log(1-a)\)，最小值在
\(a=A\) 取得；\(b=0\) 是合法选择。直接展开得到
\[
F(A)=A\Phi(C/A)+(1-A)\Phi(C/(1-A))=E(A,C).
\]
这给出 \(V\ge E\)，没有使用任何具体 e-process 或停止规则。

## 3. 稀释通道与全角度上界

先记录一个经典数据处理事实。对 \(r,s\in[0,1]\) 和 \(0\le\alpha\le1\)，
\[
\operatorname{kl}(\alpha r\Vert\alpha s)
\le\alpha\,\operatorname{kl}(r\Vert s).
\tag{6}
\]
证明：比较三点分布
\[
(\alpha r,\alpha(1-r),1-\alpha),\qquad
(\alpha s,\alpha(1-s),1-\alpha).
\]
两者 KL 为 \(\alpha\operatorname{kl}(r\Vert s)\)。将后两个类别合并，
得到 Bernoulli 概率 \(\alpha r,\alpha s\)，KL 不增加。这也可以直接由
log-sum 不等式证明，边界按相对熵约定处理。

先设 \(A=m\le1/2\)，并写
\[
r(x)=\frac{p(x)}{2m}
=\frac12+\frac{B}{2m}\cos x+\frac{C}{2m}\sin x.
\]
物理约束保证
\[
\left(\frac{B}{2m}\right)^2+\left(\frac{C}{2m}\right)^2\le\frac14,
\]
所以 \(r\) 正好属于已证明的 \(A=1/2\) 子类。该定理给出
\[
s^*(x)=\frac12+\frac{B}{2m}\lambda(C/m)\cos x,\qquad
\operatorname{kl}(r(x)\Vert s^*(x))\le\Phi(C/m).
\]
对 (6) 取 \(\alpha=2m\)，得到
\[
\operatorname{kl}(p(x)\Vert 2m\,s^*(x))
\le 2m\Phi(C/m).
\]
而 \(2m\,s^*(x)=A+B\lambda(C/m)\cos x\)，正是 (5)。
对称定理保证 \(|B\lambda(C/m)|<m\) 当 \(0<|C|<m\)，因此这个零假设
在所有角度都物理合法。

若 \(A>1/2\)，先将概率互补为 \(1-p(x)\)。其均值为 \(m=1-A\)，
余弦和正弦系数为 \(-B,-C\)。应用刚才的结果，再利用
\(\operatorname{kl}(p\Vert q)=\operatorname{kl}(1-p\Vert1-q)\)，仍得到 (5)
及同样的上界。由于 \(\lambda\) 为偶函数，系数的符号正确恢复。

对任意查询设计 \(w\)，使用同一个明确零假设 \(q^*\)，就有
\[
\inf_{a,b}\int\operatorname{kl}(p\Vert q_{a,b})\,dw
\le\int\operatorname{kl}(p\Vert q^*)\,dw\le U(A,C).
\]
结合第 2 节完成 (3) 的证明。这里不需要交换 supremum 和 infimum。
稀释通道只是证明中的辅助表示，不增加实际实验资源，也不要求实际算法执行稀释。

当 \(C\ne0\) 时，令 \(u=|C|/m>0\)，则
\[
E=m\Phi(u)+(1-m)\Phi\!\left(\frac{m}{1-m}u\right)>m\Phi(u)=U/2.
\]
因而 \(V\le U<2E\)，得到 (4)。

### 对单侧物理零假设同样成立

若 \(C>0\)，把零假设扩展为
\[
q(x)=a+b\cos x+c\sin x,\qquad c\le0,\qquad
b^2+c^2\le\min(a,1-a)^2.
\]
上界仍成立，因为 (5) 的 \(c=0\) 候选仍然可行。
在均分 PSR 两角上，任意此类零假设满足 \(q_+\le q_-\)。
在 \((q_+,q_-)=(A,A)\) 处，均分 KL 目标的梯度为
\[
\left(-\frac{C}{2A(1-A)},\frac{C}{2A(1-A)}\right).
\]
与任何允许位移的内积为
\(C(q_--q_+)/[2A(1-A)]\ge0\)。凸性说明最小值仍为 \(E(A,C)\)，
且常数零假设 \(A\) 可行。因此 (3)–(4) 对更大的单侧零假设也成立。
\(C<0\) 时反转不等号即可。

## 4. 给定信号的比例界与仅依赖均值的一致界

设 \(C\ne0\)，定义
\[
t=\frac{m}{1-m}\in(0,1],\qquad u=\frac{|C|}{m}\in(0,1].
\]
由 (3) 直接得到一个同时使用均值和信号大小的界：
\[
\boxed{\quad
\frac{V}{E}\le R(m,u)
=\frac{2}{1+\Phi(tu)/(t\Phi(u))}.
\quad}
\tag{7}
\]

**引理。** 对固定 \(0<t\le1\)，函数
\(\Phi(tu)/\Phi(u)\) 在 \(0<u\le1\) 上非增。

证明：对 \(0<u<1\)，
\[
\Phi(u)=\sum_{n=1}^{\infty}\frac{u^{2n}}{(2n)(2n-1)}.
\]
把这些正项归一化为概率权重
\(\pi_n(u)=u^{2n}/[(2n)(2n-1)\Phi(u)]\)。则
\(\Phi(tu)/\Phi(u)=\mathbb E_{\pi(u)}t^{2N}\)，并且
\[
u\frac{d}{du}\frac{\Phi(tu)}{\Phi(u)}
=\operatorname{Cov}_{\pi(u)}(2N,t^{2N})\le0.
\]
最后一步来自一个函数递增、另一个非增；也可将协方差写成双重求和，
其中每项 \((2i-2j)(t^{2i}-t^{2j})\le0\)。对 \(u<1\) 可合法逐项求导；
\(u\uparrow1\) 用连续性取极限，不需要边界处的矩有限。

由于 \(\Phi(1)=\log2\)，引理给出
\[
\boxed{\quad
\frac{V}{E}\le R(m,u)\le G(m)
:=\frac{2}{1+\Phi(t)/(t\log2)}<2
\quad(m<1/2).
\quad}
\tag{8}
\]
当 \(m=1/2\) 时，\(G=R=1\)，恢复精确对称定理。
当 \(C\to0\) 且 \(m\) 固定时，(7) 的连续极限为
\[
R(m,0+)=\frac{2}{1+t}=2(1-m).
\tag{9}
\]
这里 \(C=0\) 本身的 \(V/E\) 未定义，但 \(V=E=0\)；不把极限解释成
零信号时存在正信息率。

由连续性，\(G(m)\to1\) 当 \(m\to1/2\)。又因为 \(\Phi(t)/t\) 随 \(t\)
非减，\(G(m)\) 随 \(m\) 非增。所以，只要真实均值满足
\(|A-1/2|\le\varepsilon<1/2\)，就有对所有允许的 \(B,C\) 一致成立的界
\[
V(A,B,C)\le G(1/2-\varepsilon)\,E(A,C).
\tag{10}
\]
它不需要额外的径向裕量或信号下限。最优角度仍可能沿联合边界趋近
\(0\) 或 \(\pi\)：**角度漂移与信息目标值漂移不是一回事**。
式 (10) 约束的是目标值，没有证明最优角度的连续性。

## 5. 对称均值附近的确定性上界表

下面按 \(m=\min(A,1-A)\) 列出 (8) 的一致界，因此同一行同时适用于
\(A=m\) 和 \(A=1-m\)。数值用 outward Decimal 区间的上端点计算，
各列分别向上舍入到六位小数；它们是上界，不是拟合实验中的实际收益。

| \(m\) | \(V/E\) 的一致上界 \(G(m)\) | 最多额外完整信息 \((G-1)\times100\%\) |
|---:|---:|---:|
| 0.5000 | 1.000000 | 0.000000% |
| 0.4999 | 1.001175 | 0.117467% |
| 0.4990 | 1.008499 | 0.849839% |
| 0.4950 | 1.031754 | 3.175374% |
| 0.4900 | 1.054908 | 5.490777% |
| 0.4800 | 1.093769 | 9.376850% |
| 0.4500 | 1.186902 | 18.690190% |
| 0.4000 | 1.311505 | 31.150434% |
| 0.2500 | 1.606281 | 60.628056% |
| 0.1000 | 1.851312 | 85.131110% |
| 0.0100 | 1.985533 | 98.553250% |

例如，真实 \(A\in[0.499,0.501]\) 时，即使其余参数同时逼近物理边界，
连续角度、任意支持和分配在完整 oracle 信息上超过均分 PSR 的空间也不大于
0.849839%。这是信息界，不能直接作为有限样本节省费用百分比。

## 6. 知道信号大小时更紧的界

以下是 (7)，其中 \(u=|C|/m\)。第一列信号极限只是 (9)，不是在 \(C=0\)
计算比值。其余列对应指定的真实信号大小，仍对全部允许的 \(B\) 有效。

| \(m\) | \(u\to0\) | \(u=0.01\) | \(u=0.1\) | \(u=0.5\) | \(u=0.9\) | \(u=1\) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.49 | 1.020000 | 1.020001 | 1.020065 | 1.021900 | 1.032141 | 1.054908 |
| 0.45 | 1.100000 | 1.100003 | 1.100275 | 1.107885 | 1.143189 | 1.186902 |
| 0.25 | 1.500000 | 1.500006 | 1.500558 | 1.515135 | 1.565952 | 1.606281 |
| 0.10 | 1.800000 | 1.800003 | 1.800298 | 1.807944 | 1.832964 | 1.851312 |

计算脚本：
[psr_information_bound_table.py](../experiments/structured_validation/psr_information_bound_table.py)。
保留全部区间端点、参数及源码哈希的结果：
[psr_information_bound_table.json](../results/structured_validation/psr_information_bound_table.json)。
这次计算没有生成观测、运行控制器或查看新的模拟样本。

## 7. 完整 oracle 信息不能移植给当前证据引擎

取 \(A=1/2,B=0,C=1/2\)。在 PSR 两角，正侧总是成功、负侧总是失败。
完整数据相对最优同率零假设的 KL 是每次查询 \(\log2\)，所以
\(E=V=\log2\)。

但若获得 \(n\) 对这样的观测，paired-beta 证据为
\[
M_n^{\rm beta}
=2^n\frac{\mathrm B(n+1/2,1/2)}{\mathrm B(1/2,1/2)},
\]
因而
\[
\frac{\log M_n^{\rm beta}}{2n}\longrightarrow\frac{\log2}{2}.
\]
每一对消耗两次查询，不能把“每对”的增长率当成“每次查询”的增长率。

在这个例子中，给定每块边际后双成功数 \(J=0\) 已确定。
真正的配对 Rao–Blackwell 化没有任何剩余随机配对可以平均，
因此对同一连续 beta 先验得到的证据与 paired-beta 完全相同。
当前固定有限 odds 网格的 RB 过程也不会改善这一点；其每次查询的渐近增长率
至多为 \(\log2/2\)，有限最大 odds 时严格更小。

所以“均分 PSR 在完整数据 oracle 设计中最优”不等于
“当前 PSR paired-beta/RB 已达到最优信息率”。本证明不替代证据引擎效率、
付费学习成本、固定阈值停止成本以及同预算强基线比较。

## 8. 边界、未解决问题与算法使用范围

- \(A=0\) 或 \(A=1\) 时，物理约束强制 \(B=C=0\)，信息值为零；
  不使用含零分母的比例公式。
- \(|C|=m\) 时 \(B=0\)，上界用常数零假设处理，不计算
  \(0\cdot\operatorname{atanh}(1)\)。
- 全参数因子二界及一致均值界没有证明一般 \(A\ne1/2\) 时的最优设计一定
  是反射两角；这个精确结构问题仍未解决。
- 这些界针对真实参数。未来若据此决定是否继续付费探索，必须使用已付费数据
  和适当的不确定性处理；不能把模拟器真值传给控制器。
- 本文没有改变任何已冻结的实验、算法、先验、样本数、验收条件或旧理论文档。

## 9. 统一因子二的常数是紧的

这里证明的是整个物理参数类上的常数紧性，不是说某个非退化固定实例取到
\(V/E=2\)。第 1 节已证明每个 \(0<A<1,C\ne0\) 的固定实例都严格小于二。

取 \(0<A<1/2\)、\(0<\varepsilon<1\)，令
\[
B=A(1-\varepsilon),\qquad C=A\varepsilon,\qquad
z_\varepsilon=\cos x_\varepsilon=-1+\sqrt{\varepsilon}.
\]
此参数族严格物理可行，因为
\[
B^2+C^2=A^2(1-2\varepsilon+2\varepsilon^2)<A^2.
\]
定义
\[
f_\varepsilon=\varepsilon+(1-\varepsilon)\sqrt{\varepsilon},
\qquad
g_\varepsilon=\varepsilon\sqrt{2\sqrt{\varepsilon}-\varepsilon}.
\]
在反射两角 \(\pm x_\varepsilon\)，两率分别为
\[
\mu\pm d,\qquad
\mu=A f_\varepsilon,\qquad d=A g_\varepsilon.
\]
因为模型严格物理，所以 \(0<d<\mu<1\) 且两率都位于 \((0,1)\)。
该两角均分设计的完整信息值，和第 2 节同样按公共均值最小化后，为
\[
J(A,\varepsilon)
=A f_\varepsilon\Phi(g_\varepsilon/f_\varepsilon)
 +(1-A f_\varepsilon)
   \Phi\!\left(\frac{A g_\varepsilon}{1-A f_\varepsilon}\right).
\tag{11}
\]
相应 PSR 信息为
\[
E(A,A\varepsilon)
=A\Phi(\varepsilon)
 +(1-A)\Phi\!\left(\frac{A\varepsilon}{1-A}\right).
\tag{12}
\]
零假设在反射两角仍是公共概率，因此 \(V\ge J\)。扩展为单侧物理零假设时，
同样的有序率凸性论证也给出这个下界。

先固定 \(\varepsilon>0\)，令 \(A\downarrow0\)。式 (11)、(12) 的第二项均为
\(O_\varepsilon(A^2)\)，而第一项恰为常数乘 \(A\)。因此
\[
\lim_{A\downarrow0}\frac{J(A,\varepsilon)}{E(A,A\varepsilon)}
=\frac{f_\varepsilon\Phi(g_\varepsilon/f_\varepsilon)}
       {\Phi(\varepsilon)}.
\tag{13}
\]
这里分母 \(\Phi(\varepsilon)>0\)，尚未让 \(\varepsilon\) 趋零。

再令 \(\varepsilon\downarrow0\)。可以直接从正项级数得到显式余项界
\[
0\le\Phi(v)-\frac{v^2}{2}
\le\frac{v^4}{12(1-v^2)},\qquad |v|<1.
\tag{14}
\]
由于 \(g_\varepsilon/f_\varepsilon\sim\sqrt2\,\varepsilon^{3/4}\to0\)，
(14) 可同时用于 \(\Phi(\varepsilon)\) 和
\(\Phi(g_\varepsilon/f_\varepsilon)\)，从而
\[
\frac{f_\varepsilon\Phi(g_\varepsilon/f_\varepsilon)}
     {\Phi(\varepsilon)}
\sim\frac{g_\varepsilon^2}{f_\varepsilon\varepsilon^2}
=\frac{2-\sqrt{\varepsilon}}{1-\varepsilon+\sqrt{\varepsilon}}
\longrightarrow2.
\tag{15}
\]

因此，对任意 \(\kappa<2\)，先选足够小的 \(\varepsilon\)，再选足够小的
\(A>0\)，就能使严格物理实例满足
\[
\frac{V(A,B,C)}{E(A,C)}
\ge\frac{J(A,\varepsilon)}{E(A,C)}>\kappa.
\]
结合 \(V<2E\)，证明统一常数二是最优的。这个论证使用明确的顺序选参和
可控 Taylor 余项，不要求 \(V/E\) 的某个内层极限预先存在，也没有以数值扫描
替代证明。
