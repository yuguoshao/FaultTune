# 对称单频 Bernoulli 模型的连续角度 oracle 最优设计

本文给出一个可逐项检查的数学证明。它针对 **\(A=1/2\)** 的子类，不推出一般 \(A\) 的两点最优性，也不推出任何有限样本停止费用优势。新颖性及与既有最优设计、量子测量理论的关系尚未系统核查；不据此声称超过 SOTA。

这份说明独立于此前冻结的理论文档及实验来源，不改变前瞻计划、算法、样本数或验收指标。证明使用已有的矩阵对数凹性；这里只需实对称 \(2\times2\) 矩阵。

## 1. 问题与精确结论

设

\[
p(x)=\frac12+B\cos x+C\sin x,
\qquad B^2+C^2\le\frac14,
\]

并允许零假设模型

\[
q_{a,b}(x)=a+b\cos x,
\qquad 0\le a-|b|\le a+|b|\le1.
\]

令 \(w\) 为圆周上任意概率测度，定义每次等费用查询的 oracle 信息设计值

\[
V(B,C)=\sup_w\inf_{a,b}
\int \operatorname{kl}\!\left(p(x)\Vert q_{a,b}(x)\right)\,w(dx),
\]

其中 \(\operatorname{kl}\) 使用自然对数，边界值按连续延拓及通常的相对熵约定解释。\(w\) 在这里可使用真实 \(B,C\)，所以这是 oracle 设计问题。

**定理。** 写 \(\gamma=2C\)，并定义

\[
\Phi(\gamma)=
\frac{1+\gamma}{2}\log(1+\gamma)
+\frac{1-\gamma}{2}\log(1-\gamma).
\]

则

\[
\boxed{\quad V(B,C)=\Phi(2C),\qquad
w^*=\tfrac12\delta_{\pi/2}+\tfrac12\delta_{-\pi/2}\quad}
\]

对全部允许的 \(B,C\) 成立。特别地，在这个子类中，即便允许连续角度、任意多点支持和不等额分配，均分 PSR 两角已经达到 oracle 最优值。

当 \(0<|\gamma|<1\) 时，一个全角度对偶证书为

\[
q^*(x)=\frac12+b^*\cos x,
\qquad
b^*=B\frac{\operatorname{atanh}(2C)}{2C}
=B\frac{\log((1+2C)/(1-2C))}{4C}.
\]

它满足逐角度不等式

\[
\boxed{\quad
\operatorname{kl}\!\left(p(x)\Vert q^*(x)\right)
\le\Phi(2C)\quad\text{对所有 }x.\quad}
\]

证明直接构造一个原问题下界和一个一致上界，不需要假定 minimax 交换。

## 2. 矩阵表示及合法性

取两个实 Pauli 矩阵

\[
Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix},
\qquad
X=\begin{pmatrix}0&1\\1&0\end{pmatrix}.
\]

令 \(\beta=2B\)、\(\gamma=2C\)。由物理约束，\(\beta^2+\gamma^2\le1\)。定义

\[
\rho=\tfrac12(I+\beta Z+\gamma X),
\quad
\lambda=\frac{\operatorname{atanh}\gamma}{\gamma},
\quad
\sigma=\tfrac12(I+\beta\lambda Z),
\quad
H_*=I+\gamma X.
\]

先处理 \(0<|\gamma|<1\)。\(\rho\) 的特征值为 \((1\pm\sqrt{\beta^2+\gamma^2})/2\)，所以 \(\rho\succeq0\)、\(\operatorname{Tr}\rho=1\)。\(H_*\) 的特征值为 \(1\pm\gamma>0\)。

\(\lambda\) 为偶函数且为正。对于 \(0<t<1\)，

\[
\operatorname{atanh}t<\frac{t}{\sqrt{1-t^2}}.
\]

因为两边之差在 \(t=0\) 为零，而

\[
\frac{d}{dt}\left(\frac{t}{\sqrt{1-t^2}}-\operatorname{atanh}t\right)
=\frac1{(1-t^2)^{3/2}}-\frac1{1-t^2}>0.
\]

因此

\[
|\beta\lambda|
\le\sqrt{1-\gamma^2}\frac{\operatorname{atanh}|\gamma|}{|\gamma|}<1.
\]

所以 \(\sigma\succ0\)、\(\operatorname{Tr}\sigma=1\)，且对应的 \(q^*\) 在所有角度都严格位于 \((0,1)\)。

每个角度对应两个正交投影

\[
P_x=\tfrac12(I+\cos x\,Z+\sin x\,X),
\qquad I-P_x.
\]

直接计算得到

\[
\operatorname{Tr}(\rho P_x)=p(x),
\qquad
\operatorname{Tr}(\sigma P_x)=q^*(x).
\]

这里矩阵只用于证明原有 Bernoulli 不等式，没有改变原问题的测量资源模型。

## 3. 矩阵对数的导数

对正定矩阵 \(H\)，矩阵对数的 Fréchet 导数可写为

\[
\mathcal D\log(H)[E]
=\int_0^\infty(H+tI)^{-1}E(H+tI)^{-1}\,dt.
\]

这个线性映射对迹内积自伴，因此

\[
d\,\operatorname{Tr}(\rho\log H)[E]
=\operatorname{Tr}\bigl(\mathcal D\log(H)[\rho]E\bigr).
\]

在 \(X\) 的特征基中，

\[
H_*=\begin{pmatrix}1+\gamma&0\\0&1-\gamma\end{pmatrix},
\qquad
\rho=\tfrac12\begin{pmatrix}1+\gamma&\beta\\\beta&1-\gamma\end{pmatrix}.
\]

对角元素的导数系数分别为 \(1/(1+\gamma)\)、\(1/(1-\gamma)\)；非对角元素的系数为

\[
\frac{\log(1+\gamma)-\log(1-\gamma)}{(1+\gamma)-(1-\gamma)}
=\lambda.
\]

所以

\[
\mathcal D\log(H_*)[\rho]
=\tfrac12\begin{pmatrix}1&\beta\lambda\\\beta\lambda&1\end{pmatrix}_{X\text{ 基}}
=\sigma.
\]

这也核对了梯度的正规化：\(\operatorname{Tr}\sigma=1\)，并且

\[
\operatorname{Tr}(\sigma H_*)=1.
\]

## 4. 凹性给出统一 KL 上界

矩阵对数具有算子凹性，因此 \(H\mapsto\operatorname{Tr}(\rho\log H)\) 在正定锥上凹。这里 \(\rho\) 只需半正定，不必正定。结合上一节的导数，任意 \(H\succ0\) 满足支持超平面不等式

\[
\operatorname{Tr}(\rho\log H)
\le\operatorname{Tr}(\rho\log H_*)
+\operatorname{Tr}\bigl(\sigma(H-H_*)\bigr).
\]

凹性也可以从上一节的积分表示直接检查：沿 Hermitian 方向 \(E\) 的二阶导数为

\[
-2\int_0^\infty
(H+tI)^{-1}E(H+tI)^{-1}E(H+tI)^{-1}\,dt\preceq0.
\]

积分中的正半定性来自
\(RERER=R^{1/2}(R^{1/2}ER^{1/2})^2R^{1/2}\succeq0\)。

固定角度 \(x\)，先设 \(0<p(x)<1\)，简写 \(p=p(x)\)、\(q=q^*(x)\)。取经典似然比对应的矩阵

\[
H_x=\frac p qP_x+\frac{1-p}{1-q}(I-P_x)\succ0.
\]

由于两个投影正交，

\[
\operatorname{Tr}(\sigma H_x)=p+(1-p)=1,
\]

以及

\[
\operatorname{Tr}(\rho\log H_x)
=p\log\frac pq+(1-p)\log\frac{1-p}{1-q}
=\operatorname{kl}(p\Vert q).
\]

把 \(H_x\) 代入支持超平面不等式，两项迹正规化恰好消去。又因为

\[
\operatorname{Tr}(\rho\log H_*)
=\frac{1+\gamma}{2}\log(1+\gamma)
+\frac{1-\gamma}{2}\log(1-\gamma)=\Phi(\gamma),
\]

便得到所需的逐角度上界。

若 \(p=0\) 或 \(p=1\)，用

\[
H_{x,\varepsilon}=\frac{H_x+\varepsilon I}{1+\varepsilon}\succ0
\]

代替 \(H_x\)。它仍满足 \(\operatorname{Tr}(\sigma H_{x,\varepsilon})=1\)。令 \(\varepsilon\downarrow0\)，按 \(0\log0=0\) 取极限，得到相同上界。因此 \(\beta^2+\gamma^2=1\) 的纯态边界也已覆盖。

## 5. PSR 两角给出匹配下界

在 \(x=\pm\pi/2\) 上，零假设的两个概率都为 \(a\)，真实概率为 \((1\pm\gamma)/2\)。均分设计的目标为

\[
\frac12\operatorname{kl}\!\left(\frac{1+\gamma}{2}\Vert a\right)
+\frac12\operatorname{kl}\!\left(\frac{1-\gamma}{2}\Vert a\right).
\]

其中 \(a\) 相关项是 \(-\tfrac12\log a-\tfrac12\log(1-a)\)，故最小值在 \(a=1/2\) 取得，值为 \(\Phi(\gamma)\)。\(b=0\) 是一个允许的选择，因而这里的最小值确实属于原有物理零假设参数集合。

这证明 \(V\ge\Phi(\gamma)\)。另一方面，对任意设计 \(w\)，第 4 节构造的同一个物理零假设 \(q^*\) 给出

\[
\inf_{a,b}\int\operatorname{kl}(p\Vert q_{a,b})\,dw
\le\int\operatorname{kl}(p\Vert q^*)\,dw
\le\Phi(\gamma).
\]

因此 \(V\le\Phi(\gamma)\)，上下界一致。

### 单侧零假设的推论

若 \(C>0\)，结论也适用于更大的物理零假设集合
\(q(x)=a+b\cos x+c\sin x\)、\(c\le0\)，其中 \(a,b\) 均自由且满足全角度概率合法性。上界仍使用 \(c=0\) 的同一个 \(q^*\)。对均分 PSR 设计，任何此类零假设都满足 \(q_+\le q_-\)，而真实概率满足 \(p_+>p_-\)。

均分 KL 目标对 \((q_+,q_-)\) 凸；在 \((1/2,1/2)\) 处的梯度为 \((-2C,2C)\)。对任意 \(q_+\le q_-\)，支持线性项为

\[
-2C(q_+-\tfrac12)+2C(q_--\tfrac12)
=2C(q_--q_+)\ge0.
\]

所以约束最小值仍在两率均为 \(1/2\) 时取得，值为 \(\Phi(2C)\)。这个常数零假设是物理可行的。\(C<0\) 时相应单侧零假设取 \(c\ge0\)，符号反射给出相同结论。

## 6. 零信号与剩余边界

- 当 \(C=0\) 时，真实模型已在零假设中，故 \(V=0=\Phi(0)\)。\(\lambda\to1\)、\(b^*\to B\)、\(q^*\to p\)，公式连续衔接；即便 \(|B|=1/2\)，相同分布的边界 KL 仍为零。
- 当 \(|C|=1/2\) 时，物理约束强制 \(B=0\)。取 \(q^*\equiv1/2\)，任意 Bernoulli 概率相对此零假设的 KL 至多 \(\log2\)。PSR 两角取到概率 \(0,1\)，均分设计达到 \(\log2=\Phi(\pm1)\)。这一点直接处理，不把 \(0\cdot\operatorname{atanh}(1)\) 当成定义。
- \(C<0\) 不需要另一个证明：\(\lambda\) 和 \(\Phi\) 均为偶函数，以上导数和矩阵计算对带符号的 \(\gamma\) 仍成立。

## 7. 主张边界与意义

这个结论严格解决了 \(A=1/2\) 子类的连续角度 oracle 信息设计：额外角度及不等额分配不能超过均分 PSR 的信息值。它可以解释为什么在这个子类里，仅靠复杂选角不应期待 oracle 信息率优势。

它不说明固定有限专家先验达到该值，不说明付费学习能免费获得 oracle 参数，也不说明有限阈值下任一停止规则的平均费用最小。它没有解决 \(A\ne1/2\) 时是否总存在反射两角的最优设计。上述前瞻实验含非对称实例，其算法比较与验收计划保持不变。

证明已经进行独立逐式审查，但未经形式证明助手认证。数学新颖性、既有测量相对熵结果是否直接包含此子类，仍待专门文献核查。
