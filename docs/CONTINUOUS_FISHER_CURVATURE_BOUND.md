# 连续条件 Fisher 混合的曲率界与半对数损失

结论：任意有限历史、任意有限实数 \(\lambda_*\) 下，完整连续混合满足

\[
\boxed{
M\ge\sqrt{\frac{2}{\pi(S+1)}}\,
       \frac{L(\lambda_*)}{\cosh(\lambda_*/2)}
 \ge\frac{2L(\lambda_*)}{\sqrt{\pi(N+2)}\cosh(\lambda_*/2)}.
}
\tag{1}
\]

这里 \(S=\sum_b\min(T_b,2m_b-T_b)\le n=\sum_bm_b\)，
\(n\) 是每侧样本数的和，\(N=2n\) 是 physical shots。
因此，相对任意固定有限专家，理想连续混合的 log 资本损失至多为
\(\tfrac12\log(n+1)\) 加一个显式常数。这个有限样本、逐路径界加强了
[先前探索](CONTINUOUS_FISHER_EXPLORATION.md) 第 7 节的 Lipschitz 界。
证明使用固定大小加权子集的负相关性和高斯积分；新颖性未确认。

## 1 对象与条件

每块每侧采样 \(m_b\) 次，记总成功数 \(T_b=K_{b,+}+K_{b,-}\)，并定义

\[
U=\sum_b(K_{b,+}-K_{b,-})/2,\qquad
\psi_b(\lambda)=\log\mathbb E_0e^{\lambda(K_{b,+}-T_b/2)},
\]

\[
L(\lambda)=e^{\lambda U-\sum_b\psi_b(\lambda)},\qquad
\rho(\lambda)=\frac1{2\pi\cosh(\lambda/2)},\qquad
M=\int_{\mathbb R}L(\lambda)\rho(\lambda)\,d\lambda.
\]

\(\mathbb E_0\) 使用给定 \(T_b\) 的中央超几何分布。
所有 \(m_b\) 为有限非负整数，\(0\le T_b\le2m_b\)；无数据或退化块也允许。
\(\rho\) 是 \(q=\operatorname{logistic}(\lambda)\) 上 Bernoulli
Jeffreys 先验的换元，不宣称是任意块模型自身的 Jeffreys 先验。

式 (1) 是关于有限数据的代数结论，不要求真实 odds 固定、也不使用
真实概率。将它用于方向错误控制时，仍须满足原证明的可预测均分块、
块内独立 Bernoulli 测量和共同真实排序条件。

## 2 固定大小加权子集的两两负相关

取 \(d\) 个位置、严格正权重 \(w_1,\ldots,w_d\)，在大小固定为
\(T\in\{0,\ldots,d\}\) 的子集上赋概率

\[
\Pr(S)=\frac{\prod_{i\in S}w_i}{e_T(w_1,\ldots,w_d)},\qquad |S|=T.
\]

这里 \(e_r\) 是 \(r\) 次 elementary symmetric sum，约定
\(e_0=1\)，超出 \(0\le r\le d\) 时为零。令 \(I_i=1_{\{i\in S\}}\)。
固定不同位置 \(i,j\)，用其余 \(d-2\) 个权重定义

\[
A=e_T,\quad B=e_{T-1},\quad C=e_{T-2},\qquad
Z=A+(w_i+w_j)B+w_iw_jC.
\]

直接分类包含哪些位置，得到

\[
\Pr(I_i=1)=\frac{w_i(B+w_jC)}Z,\qquad
\Pr(I_i=I_j=1)=\frac{w_iw_jC}Z,
\]

从而

\[
\operatorname{Cov}(I_i,I_j)
=\frac{w_iw_j(AC-B^2)}{Z^2}\le0.
\tag{2}
\]

最后一步使用 Newton 不等式：正权重的 elementary symmetric sums
满足 \(e_r^2\ge e_{r-1}e_{r+1}\)。具体说，若余下权重数为 \(q\)，
则内点 \(1\le r\le q-1\) 有

\[
\left(\frac{e_r}{\binom qr}\right)^2
\ge\frac{e_{r-1}}{\binom q{r-1}}
    \frac{e_{r+1}}{\binom q{r+1}},
\]

其中恢复组合数后的乘数为
\((r+1)(q-r+1)/(r(q-r))\ge1\)。端点由越界项为零直接成立。
\(T=0,d\) 时全部指标确定，协方差为零；\(d=0,1\) 时没有两位置要比较。

## 3 任意非中心均分超几何的方差界

在 \(2m\) 个位置中固定选择 \(T\) 个，正侧 \(m\) 个位置权重均为
\(\omega=e^\lambda>0\)，负侧 \(m\) 个权重均为 1。
正侧被选位置数 \(K\) 满足

\[
\Pr_\lambda(K=k)
\propto\binom mk\binom m{T-k}\omega^k.
\tag{3}
\]

这正是 \(K_+\mid T\) 的 Fisher 非中心超几何分布。特别地，
\(K\) 是正侧恰好 \(m\) 个指标之和，而不是 \(T\) 个指标之和。
由 (2) 以及 Bernoulli 方差至多 \(1/4\)，

\[
\boxed{\operatorname{Var}_\lambda(K)
=\sum_{i\in+}\operatorname{Var}(I_i)
 +2\sum_{i<j,\ i,j\in+}\operatorname{Cov}(I_i,I_j)
\le\frac m4.}
\tag{4}
\]

式 (4) 对所有有限 \(\lambda\) 和全部 \(T\) 成立，包括退化的
\(T=0,2m\) 与 \(m=0\)。\(\omega\to0,\infty\) 的极限仍满足方差界，
但以下专家比较仅定义于有限 \(\lambda_*\)。

## 4 使用较少成功数或失败数的加强

同一个 (3) 还可以写成

\[
\Pr_\lambda(K=k)\propto
\binom Tk\binom{2m-T}{m-k}\omega^k,
\tag{4a}
\]

因为在整个共同支持上，两种组合系数的比为与 \(k\) 无关的

\[
\frac{\binom Tk\binom{2m-T}{m-k}}
     {\binom mk\binom m{T-k}}
=\frac{T!(2m-T)!}{(m!)^2}.
\]

将 (4a) 解释为在 \(2m\) 个位置固定选择 \(m\) 个，其中 \(T\) 个
位置权重为 \(\omega\)，其余 \(2m-T\) 个为 1。则 \(K\) 是前
\(T\) 个指标之和，\(m-K\) 是后 \(2m-T\) 个指标之和。分别使用
同一个 Newton 负相关结论，得到

\[
\boxed{\operatorname{Var}_\lambda(K)
\le\frac{\min(T,2m-T)}4\le\frac m4.}
\tag{4b}
\]

退化的 \(T=0,2m\) 在右侧给出零，完全吻合。这个加强只需同一加权
子集证明，无须额外假设，也无须 Jacobi 多项式根的位置定理。

## 5 全局曲率界与完整积分

令 \(g(\lambda)=\log L(\lambda)+\log\rho(\lambda)\)。有限支持指数族满足
\(\psi_b''(\lambda)=\operatorname{Var}_\lambda(K_{b,+})\)，而

\[
(\log\rho)''(\lambda)=-\tfrac14\operatorname{sech}^2(\lambda/2).
\]

因此对每个实数 \(\lambda\)，

\[
-\frac{n+1}4\le-\frac{S+1}4\le g''(\lambda)<0.
\tag{5}
\]

取 \(C=(S+1)/4>0\)。函数 \(g(\lambda)+C\lambda^2/2\) 凸，
其切线下界给出对全部 \(h\in\mathbb R\) 同时成立的

\[
g(\lambda_*+h)\ge g(\lambda_*)+g'(\lambda_*)h-\frac{S+1}{8}h^2.
\]

对两侧指数积分并配方，得到更强的界

\[
\boxed{
M\ge e^{g(\lambda_*)}\sqrt{\frac{8\pi}{S+1}}
       \exp\!\left(\frac{2g'(\lambda_*)^2}{S+1}\right).
}
\tag{6}
\]

最后一个因子至少为 1。代入
\(e^{g(\lambda_*)}=L(\lambda_*)/[2\pi\cosh(\lambda_*/2)]\)
即得 (1)。取 \(g'\) 在某点为零不是证明条件；丢掉平方因子后，
无需寻找驻点或计算导数即可使用 (1)。

## 6 能够得到的保证

取对数后，(1) 等价于

\[
\log L(\lambda_*)-\log M
\le\frac12\log\!\left(\frac{\pi(S+1)}2\right)
 +\log\cosh(\lambda_*/2)
\le\frac12\log\!\bigl(\pi(N+2)\bigr)-\log2
 +\log\cosh(\lambda_*/2).
\tag{7}
\]

因此固定有限专家的显式损失至多为
\(\tfrac12\log(S+1)+O_{\lambda_*}(1)\le
\tfrac12\log(n+1)+O_{\lambda_*}(1)\)。
所有实数参数同时满足逐路径不等式，所以 \(\lambda_*\) 可在看到数据后选择；
若其位置随样本数增大，必须保留 \(\log\cosh(\lambda_*/2)\) 项，不能
对任意专家序列宣称统一半对数损失，也不能直接代入无限 odds 端点。

若对 (1) 的右侧做严格向下数值认证，它本身就是完整混合的下界，
可以与已有弦段积分下界取最大值。按同一个 \(1/\delta\) 阈值及
非零 score 方向使用时，原先的方向风险证明继续适用；数据依赖地选择
计算节点并未改变固定先验。已经存在的有限弦段计算值不会自动继承
(1) 或 (7)，必须实际计算或另行证明其下界。

该结论没有给出平均首次停止费用优势，也没有证明有限 shot cap 内必停。
\(\tfrac12\log n\) 混合损失在统计文献中已有背景；此处提供当前模型的
显式有限样本推导，不据此主张首创或已超过 SOTA。
