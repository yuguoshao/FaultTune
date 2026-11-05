# Rao–Blackwell evidence from balanced Bernoulli blocks

Status: the propositions below are proved under their stated assumptions.
The implementation has exact small-state and outward-arithmetic checks.
No proposition below establishes lower expected stopping cost, superiority
to the continuous Jeffreys beta mixture, or literature priority.

## Setup and the object being averaged

At block \(t\), select two settings and an equal integer sample count \(m_t\)
before observing that block. The block contains independent observations
\(X_{t,i}\sim\mathrm{Bernoulli}(q_{t,-})\) and
\(Y_{t,i}\sim\mathrm{Bernoulli}(q_{t,+})\), \(i=1,\ldots,m_t\).
The probability is constant on each side within a block. For sign testing,
the two sides have a common ordering throughout the run.

Write
\[
K_{t,-}=\sum_iX_{t,i},\quad K_{t,+}=\sum_iY_{t,i},\quad
S_t=K_{t,+}-K_{t,-},\quad T_t=K_{t,+}+K_{t,-}.
\]
An arbitrary pairing of the two lists produces a double-success count
\(J_t=\sum_iX_{t,i}Y_{t,i}\) and discordance count \(D_t=T_t-2J_t\).
The pairing carries no additional information about the two probabilities
once both margins are known:
\[
\Pr(J_t=j\mid K_{t,+},K_{t,-},m_t)
=\frac{\binom{K_{t,+}}j
        \binom{m_t-K_{t,+}}{K_{t,-}-j}}
       {\binom{m_t}{K_{t,-}}}.
\tag{1}
\]
The support is
\(\max(0,K_{t,+}+K_{t,-}-m_t)\le j\le\min(K_{t,+},K_{t,-})\).
This conditional law does not depend on \(q_{t,+}\) or \(q_{t,-}\).
It concerns the ancillary pairing, not a randomization of the measured
settings or a resampling of observations.

Fix a finite collection of positive odds \(\omega_e\), with positive weights
\(w_e\) summing to one. Both odds and weights remain fixed throughout a run.
For one paired block, the expert factor is
\[
L_{e,t}
=\left(\frac{2\omega_e}{1+\omega_e}\right)^{K_{t,+}-J_t}
 \left(\frac{2}{1+\omega_e}\right)^{K_{t,-}-J_t}
=\exp(\eta_e S_t/2)\cosh(\eta_e/2)^{-D_t},
\quad \eta_e=\log\omega_e.
\tag{2}
\]
The paired mixture is \(M_t^{\rm pair}=\sum_e w_e\prod_{s\le t}L_{e,s}\).

Define the averaged block factor
\[
R_{e,t}
=\mathbb E[L_{e,t}\mid m_t,K_{t,+},K_{t,-}]
=\left(\frac{2\omega_e}{1+\omega_e}\right)^{K_{t,+}}
 \left(\frac{2}{1+\omega_e}\right)^{K_{t,-}}
 \mathbb E\!\left[
   \left(\frac{(1+\omega_e)^2}{4\omega_e}\right)^{J_t}
   \,\middle|\,m_t,K_{t,+},K_{t,-}\right].
\tag{3}
\]
The new evidence is \(M_t^{\rm RB}=\sum_e w_e\prod_{s\le t}R_{e,s}\).
For \(m_t=0\), the factor is one. For \(m_t=1\), the margins determine the
pairing, and (3) equals (2) exactly.

## Proposition 1: oriented expert validity

For any predictable block and any fixed \(\omega>0\),
\[
\mathbb E[R_{\omega,t}\mid\mathcal F_{t-1}]
=\left[
1+(q_{t,+}-q_{t,-})\frac{\omega-1}{\omega+1}
\right]^{m_t}.
\tag{4}
\]
Consequently, the product of \(R_{\omega,t}\) is a nonnegative
supermartingale under \(q_{t,+}\le q_{t,-}\) if \(\omega\ge1\), and under
\(q_{t,+}\ge q_{t,-}\) if \(\omega\le1\).

Proof. Given the past, (1) implies that averaging (3) over the margins has
the same expectation as the original paired factor. For one independent
pair, the expectation of (2) is the bracket in (4). Independence across the
\(m_t\) pairs gives its \(m_t\)-th power. The bracket is nonnegative and at
most one under the indicated ordering. Multiplication by the capital from
previous blocks proves the supermartingale statement.

This validity statement requires predictable settings and block sizes.
It permits dependence on previous paired observations. The stronger
Rao–Blackwell identity in Proposition 3 requires the more restrictive
history assumptions stated there.

## Proposition 2: a single threshold controls wrong-direction decisions

Let \(S_{1:t}=\sum_{s\le t}S_s\). Stop with the sign of \(S_{1:t}\) when
\(S_{1:t}\ne0\) and \(M_t^{\rm RB}\ge1/\delta\).
Under a common true ordering, the probability of ever stopping in the
wrong direction is at most \(\delta\). At equality in every block, the
probability of ever making either directional decision is at most
\(\delta\).

Proof. Put
\[
H_{\omega,t}
=\mathbb E[\cosh(\log\omega/2)^{-D_t}\mid m_t,K_{t,+},K_{t,-}]
\le1.
\]
Since each expert keeps the same odds for the whole run,
\[
\prod_{s\le t}R_{\omega,s}
=\exp((\log\omega)S_{1:t}/2)\prod_{s\le t}H_{\omega,s}.
\tag{5}
\]
Assume \(q_{t,+}\ge q_{t,-}\) for all blocks. A negative decision is wrong.
On \(S_{1:t}<0\), every expert with \(\omega\ge1\) has capital at most one
by (5). Hence the mixture on this event is bounded by
\[
V_t^-=\sum_{\omega_e<1}w_e\prod_{s\le t}R_{e,s}
      +\sum_{\omega_e\ge1}w_e.
\]
By Proposition 1, \(V_t^-\) is a nonnegative supermartingale starting at
one. Ville's inequality bounds the probability of its ever crossing
\(1/\delta\) by \(\delta\). The reverse ordering is identical after swapping
the sides. If the two probabilities are equal in every block, all experts,
and therefore their complete mixture, have conditional mean-one factors;
Ville's inequality then controls crossings in either direction together.

This argument does not require symmetric expert weights. It does require
fixed odds and fixed weights: changing an expert's odds using observations
would invalidate the cumulative-score simplification in (5).

## Proposition 3: fixed-time expected log evidence improves

For this proposition, assume:

1. Settings and block sizes use only previous block margins and independent
   algorithmic randomization. They do not depend on the arbitrary pairing.
2. The conditional probabilities within the next block also depend only on
   that margin history, the chosen settings, and any exogenous information
   included in the conditioning. A fixed physical parameter evaluated at
   the adaptively selected settings satisfies this assumption.
3. The paired and RB processes use exactly the same block partition, expert
   odds and weights. Compare a fixed block index or a deterministic shot
   checkpoint at a block boundary. The logarithms are integrable; a finite
   shot cap and finite positive expert odds suffice.

Let \(\mathcal G_t\) contain this margin history, selected actions and
algorithmic randomization through block \(t\). Then
\[
M_t^{\rm RB}
=\mathbb E[M_t^{\rm pair}\mid\mathcal G_t],
\tag{6}
\]
and
\[
\mathbb E[\log M_t^{\rm RB}]
\ge \mathbb E[\log M_t^{\rm pair}].
\tag{7}
\]
Indeed, given \(\mathcal G_t\), the ancillary pairings in distinct blocks
factor into the hypergeometric laws (1). The assumptions ensure that future
margin observations do not reveal additional information about an earlier
pairing. Thus the conditional expectation of each expert's product factors
into the product of (3). Fixed weights allow exchanging the finite sum and
the conditional expectation, proving (6). Conditional Jensen applied to the
concave logarithm proves (7).

The inequality is strict whenever, with positive probability, the
conditional paired-mixture capital is not constant over possible pairings.
This is a classical conditional-expectation argument applied to the present
block experiment; it is not a claim of novelty for Rao–Blackwellization or
Jensen's inequality.

The assumptions about future observations matter. If settings use previous
discordances, or if the next true probability can depend on an earlier
ancillary pairing, conditioning on the full margin history need not leave
the pairings independent. Proposition 1 can still apply to predictable
blocks, while the identity (6) requires a separate argument.

## What the improvement does not establish

Equation (7) is a fixed-time statement. It does not order the evidence
pointwise, the distribution of first crossing times, expected stopping
cost, or capped cost. Those are measured separately using complete
protocol trajectories.

For example, take \(m=2\), \(K_-=K_+=1\) and \(\omega=4\). Conditional on
the margins, the two possible pairings have expert factors \(1\) and
\(16/25\), each with probability \(1/2\). The RB factor is \(41/50\),
which lies strictly between them. Its conditional expected log improvement
over the paired expert is \(\log(41/40)>0\), despite the absence of
pointwise dominance.

Matching the prior is essential for the comparison in (6)–(7). The current
71-expert prior is a fixed rational discretization associated with
Jeffreys cells. It is not the continuous Jeffreys prior. Consequently these
propositions give no direct dominance theorem over the paired beta-mixture
baseline. A finite odds grid also need not have positive asymptotic growth
at every arbitrarily weak alternative.

Conditional Fisher evidence is a different construction: it normalizes
using the total success count \(T\). In the example above, its factor is
\(8/11\), whereas the RB factor is \(41/50\). It should not be described as
the conditional expectation in (6).

## Exact rational implementation and validation

For rational expert odds \(\omega=a/b\), the tilt
\(\kappa=(a+b)^2/(4ab)\) is rational. Let
\(\mu_J=K_+K_-/m\) and let \(\Psi_\kappa\) denote the centered CGF of
\(J\sim\mathrm{Hypergeom}(m,K_+,K_-)\). Then
\[
\log H_\omega
=\Psi_\kappa-\frac{K_++K_--2\mu_J}{2}\log\kappa.
\tag{8}
\]
The implementation reuses the frozen unequal-arm conditional Fisher
normalizer with population counts \(m-K_+,K_+\) and total \(K_-\).
It combines outward Decimal bounds for (8), the fixed rational weights and
the log mixture. Intersecting the upper bound for \(\log H\) with zero is
justified by the exact inequality \(H\le1\). Floating calculations only
screen candidate crossings; they do not authorize a decision.

Implementation: [rb_grid.py](../experiments/structured_validation/rb_grid.py).
Checks: [rb_grid_checks.py](../experiments/structured_validation/rb_grid_checks.py).
Recorded results:
[rb_grid_checks.json](../results/structured_validation/rb_grid_checks.json).

The recorded checks include 840 exact finite-state block-factor enclosures,
48 exact ordered-null expectation identities, complete rational mixture
comparisons for 71, 33 and asymmetric three-expert priors, conditional
Jensen examples, the \(m=1\) matched-prior identity, and large-block
floating-versus-certified comparisons. These validate the tested
implementation cases; the proofs above provide the mathematical
guarantees under the stated sampling assumptions.
