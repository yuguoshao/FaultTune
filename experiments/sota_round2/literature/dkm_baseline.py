"""Executable DKM-style structured Bernoulli sign baseline, with honest scope.

Sampling follows Degenne--Koolen--Menard (NeurIPS 2019), Algorithm 1 and
Section 3.1: per-answer AdaHedge, opposite-answer KL best response, optimism,
and cumulative-ratio tracking. The finite rational query set is explicit.
SLSQP oracles are numerical implementations, not exact-oracle proofs.

``kk21`` stopping uses Kaufmann--Koolen (JMLR 2021), Theorem 7, with an
outward-rounded calibration and a certified global physical null likelihood.
``kt_safe`` is a separately named adaptation using independent per-arm
Jeffreys predictive likelihoods; it is not the threshold used in DKM's plots.
The paper's empirically convenient but unproved threshold is NOT implemented.

One-shot calls follow the original sampling schedule. A requested batch > 1
is an explicitly labelled block adaptation and does not inherit the original
sample-complexity theorem. Both certificates remain valid under predictable
sampling. No true A/B/C is supplied to the sampler or stopping certificate.
"""
from functools import lru_cache
from decimal import Decimal, localcontext
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import gammaln, logsumexp, xlogy, rel_entr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/structured_validation"))
import certified_likelihood as likelihood
from direct_tests import _Bounds, _log_marginal, _float_up


def default_arms():
    """Six fixed arms; callers may register a denser rational grid beforehand."""
    arms = []
    for numerator, denominator in ((1, 2), (1, 1), (2, 1)):
        arm = likelihood.rational_circle(numerator, denominator)
        arms.extend([dict(arm, sin_numerator=-arm["sin_numerator"]), arm])
    return arms


def features(arms):
    parsed = [likelihood._coordinates(arm) for arm in arms]
    if len(set(parsed)) != len(parsed):
        raise ValueError("The finite arm grid must not contain duplicate queries")
    return np.array([[1., co/den, si/den] for co, si, den in parsed])


def bernoulli_kl(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    return np.maximum(0., rel_entr(p,q)+rel_entr(1-p,1-q))


def observations(arms, trials, successes):
    if len(arms) != len(trials) or len(arms) != len(successes):
        raise ValueError("One count pair is required per registered arm")
    rows = []
    for arm, n, k in zip(arms, trials, successes):
        if isinstance(n, bool) or isinstance(k, bool) or int(n) != n or int(k) != k:
            raise ValueError("Observation counts must be integers")
        if not 0 <= k <= n:
            raise ValueError("Invalid observation counts")
        rows.append(dict(arm, trials=int(n), successes=int(k)))
    return rows


def _atan_inverse_bounds(integer, terms=100):
    value = _Bounds(0)
    for j in range(terms):
        term = _Bounds(1) / ((2*j+1) * integer**(2*j+1))
        value += term if j % 2 == 0 else -term
    remainder = _Bounds(1) / ((2*terms+1)*integer**(2*terms+1))
    return value + (_Bounds(0, remainder.hi) if terms % 2 == 0 else _Bounds(-remainder.hi, 0))


@lru_cache(maxsize=1)
def _zeta2_bounds():
    # Machin's identity plus alternating-series remainder, then Basel's identity.
    pi = 16*_atan_inverse_bounds(5) - 4*_atan_inverse_bounds(239)
    return pi*pi/6


def _hinv_upper(target):
    """Verified upper inverse of h(u)=u-log(u) on [1,infinity)."""
    target = _Bounds.as_bounds(target).hi
    if target < 1:
        raise ValueError("h inverse needs an argument at least one")
    if target == 1:
        return Decimal(1)
    lower, upper = Decimal(1), Decimal(max(2, int(target)+2))
    while (_Bounds(upper)-_Bounds(upper).log()).lo < target:
        upper *= 2
    for unused in range(140):
        with localcontext() as context:
            context.prec = 90
            middle = (lower+upper)/2
        value = _Bounds(middle)-_Bounds(middle).log()
        if value.lo >= target:
            upper = middle
        elif value.hi <= target:
            lower = middle
        else:
            break
    return upper


def _htilde_upper(argument):
    point = _Bounds(_Bounds.as_bounds(argument).hi)
    z = _Bounds(3)/2
    inverse_log_z = _Bounds(1)/z.log()
    cutoff = inverse_log_z-inverse_log_z.log()
    linear = z*(point-z.log().log())
    if point.hi < cutoff.lo:
        return linear.hi
    root = _Bounds(_hinv_upper(point))
    curved = root*(_Bounds(1)/root).exp()
    if point.lo >= cutoff.hi:
        return curved.hi
    return max(linear.hi, curved.hi)


@lru_cache(maxsize=32)
def kk_calibration_upper(delta_text, number_of_arms):
    """K*C_exp(log(1/delta)/K), certified from KK21 eqs. (9)-(10)."""
    delta = _Bounds(Decimal(delta_text))
    if not 0 < delta.lo < 1 or number_of_arms < 1:
        raise ValueError("Invalid risk or arm count")
    x = -delta.log()/number_of_arms
    argument = (_Bounds(_hinv_upper(1+x))+(2*_zeta2_bounds()).log())/2
    return (2*number_of_arms*_Bounds(_htilde_upper(argument))).hi


def kk_threshold_upper(trials, delta):
    if any(int(n) != n or n < 1 for n in trials):
        raise ValueError("KK21 certificate requires at least one paid pull of each arm")
    total = _Bounds(kk_calibration_upper(str(delta), len(trials)))
    for n in trials:
        total += 3*(1+_Bounds(int(n)).log()).log()
    return total.hi


def certificate(arms, trials, successes, sign, delta=.01, calibration="kk21", proposal=None):
    """Certify a selected sign against the entire opposite physical cone.

    Either sign may be selected from the data. At C=0 both false directions
    are controlled by the SAME all-arm concentration/LR event, with no
    unexplained factor-of-two omission.
    """
    rows = observations(arms, trials, successes)
    if sign not in (-1, 1) or not 0 < delta < 1:
        raise ValueError("Invalid sign or risk")
    if any(row["trials"] < 1 for row in rows):
        raise ValueError("Initialize all fixed arms with paid observations")
    null = likelihood.null_loglik_upper(rows, sign, candidate=proposal)
    if calibration == "kk21":
        numerator = likelihood._saturated(likelihood._rows(rows, True))
        threshold = kk_threshold_upper(trials, delta)
        description = "DKM GLR with KK21 Theorem 7 two-sided all-arm calibration"
    elif calibration == "kt_safe":
        numerator = sum((_log_marginal(int(k), int(n), .5) for n, k in zip(trials, successes)), _Bounds(0))
        threshold = (-_Bounds(Decimal(str(delta))).log()).hi
        description = "DKM sampling adapted to a product-Jeffreys universal-inference certificate"
    else:
        raise ValueError("Unknown calibration")
    lower = (numerator-_Bounds(Decimal(null["loglik_upper_decimal"]))).lo
    decided = lower > threshold
    return {"decision": ("reject" if sign > 0 else "accept") if decided else "abstain",
            "sign": int(sign), "shots": sum(int(n) for n in trials), "delta": float(delta),
            "calibration": calibration, "description": description,
            "evidence_lower_decimal": str(lower), "threshold_upper_decimal": str(threshold),
            "numerator_bounds": [str(numerator.lo), str(numerator.hi)],
            "null_certificate": null, "anytime_valid_under_stated_model": True,
            "arithmetic": "Outward Decimal likelihood and threshold bounds; optimizer is only a proposal"}


class AdaHedge:
    """de Rooij et al. AdaHedge, exponential weights with mixability-gap eta."""
    def __init__(self, dimension):
        self.loss = np.zeros(dimension)
        self.gap = 0.
        self.dimension = dimension

    def distribution(self):
        if self.gap == 0.:
            minima = self.loss == self.loss.min()
            return minima.astype(float)/minima.sum(), math.inf
        rate = math.log(self.dimension)/self.gap
        logits = -rate*(self.loss-self.loss.min())
        return np.exp(logits-logsumexp(logits)), rate

    def update(self, losses, weights, rate):
        losses = np.asarray(losses, dtype=float)
        average = float(weights.dot(losses))
        if math.isinf(rate):
            mixed = float(losses.min())
        else:
            centered = losses-losses.min()
            previous = self.loss-self.loss.min()
            mixed = float(losses.min()-(logsumexp(-rate*(previous+centered))-logsumexp(-rate*previous))/rate)
        self.gap += max(0., average-mixed)
        self.loss += losses
        self.loss -= self.loss.min()


class DKMSampler:
    """Observed-count-only allocation API; no oracle truth is accepted.

    ``choose`` returns one arm and a paid batch length. For the original
    per-observation schedule always request batch_size=1. Labels have separate
    AdaHedge states, while tracking accumulates all proposed allocations.
    The inner model used for NUMERICAL DESIGN has a disclosed small margin;
    the stopping certificate always uses the full closed physical model.
    """
    def __init__(self, arms=None, design_margin=1e-6, exploration_a=.1, exploration_b=.1):
        self.arms = default_arms() if arms is None else arms
        self.x = features(self.arms)
        if np.linalg.matrix_rank(self.x) < 3:
            raise ValueError("The structured allocation grid must identify A,B,C")
        if not 0 < design_margin < .5:
            raise ValueError("design_margin must be strictly between zero and one half")
        self.margin, self.factor = design_margin, 1-2*design_margin
        self.a, self.b = exploration_a, exploration_b
        if self.a <= 0 or self.b <= 0:
            raise ValueError("The theoretical exploration parameters must be positive")
        self.learners = {sign: AdaHedge(len(self.arms)) for sign in (-1, 1)}
        self.target = np.ones(len(self.arms))
        self.last_total = -1
        self.label = 1
        self.fit_start = np.array([.5, 0., 0.])
        self.null_start = {sign: np.array([.5, 0., 0.]) for sign in (-1, 1)}
        self.ever_batched = False

    def _fit(self, means, weights, sign=None, lower=None, upper=None, initial=None):
        weights = np.asarray(weights, dtype=float)
        weights = weights/weights.sum()
        initial = self.fit_start if initial is None else initial
        def predictions(point):
            return self.margin+self.factor*self.x.dot(point)
        def objective(point):
            raw = predictions(point)
            p = np.clip(raw, self.margin/2, 1-self.margin/2)
            derivative = (p-means)/(p*(1-p))
            # Convex linear extension outside the log domain, with a matching
            # derivative. Clipping the value alone gives SLSQP a false gradient.
            value = float(weights.dot(-means*np.log(p)-(1-means)*np.log1p(-p)+derivative*(raw-p)))
            gradient = self.factor*self.x.T.dot(weights*derivative)
            return value, gradient
        def cone(point):
            a, b, c = point
            radius = math.hypot(b,c)
            return np.array([a-radius, 1-a-radius])
        def cone_jac(point):
            a, b, c = point
            radius = math.hypot(b,c)
            db,dc = (b/radius,c/radius) if radius else (0.,0.)
            return np.array([[1.,-db,-dc],[-1.,-db,-dc]])
        constraints = [{"type":"ineq", "fun":cone, "jac":cone_jac}]
        if sign is not None:
            constraints.append({"type":"ineq", "fun":lambda p: -sign*p[2],
                                "jac":lambda p: np.array([0.,0.,-sign])})
        if lower is not None:
            constraints.extend([
                {"type":"ineq", "fun":lambda p: predictions(p)-lower, "jac":lambda p: self.factor*self.x},
                {"type":"ineq", "fun":lambda p: upper-predictions(p), "jac":lambda p: -self.factor*self.x}])
        def solve(start):
            return minimize(objective, start, jac=True, method="SLSQP",
                            bounds=[(0.,1.),(-.5,.5),(-.5,.5)], constraints=constraints,
                            options={"ftol":1e-13,"maxiter":120})
        def repair(point):
            point = point.copy()
            point[0] = np.clip(point[0],0.,1.)
            if sign is not None:
                point[2] = min(0.,point[2]) if sign > 0 else max(0.,point[2])
            radius, maximum = np.linalg.norm(point[1:]),min(point[0],1-point[0])
            if radius > maximum:
                point[1:] *= maximum/radius
            return point
        def oracle_gap(point):
            unused_value, gradient = objective(point)
            return max(0.,float(gradient.dot(point))+likelihood.support_float(-gradient,sign))
        result = solve(initial)
        point, restarts, corrections = repair(result.x),0,0
        if sign is not None and oracle_gap(point) > 1e-5:
            pooled = np.clip((float(weights.dot(means))-self.margin)/self.factor,0.,1.)
            for start in (np.array([pooled,0.,0.]),np.array([.5,0.,0.])):
                alternative = solve(start)
                other = repair(alternative.x)
                restarts += 1
                if objective(other)[0] < objective(point)[0]:
                    point, result = other,alternative
                if oracle_gap(point) <= 1e-5:
                    break
        if sign is not None:
            # A convex feasible line search escapes false SLSQP convergence at
            # cone vertices. Every correction decreases the actual objective.
            for unused in range(64):
                if oracle_gap(point) <= 1e-5:
                    break
                unused_value, gradient = objective(point)
                direction = -gradient
                b,c = direction[1],min(0.,direction[2]) if sign > 0 else max(0.,direction[2])
                radius = math.hypot(b,c)
                vertices = [np.array([0.,0.,0.]),np.array([1.,0.,0.])]
                if radius:
                    vertices.append(np.array([.5,.5*b/radius,.5*c/radius]))
                vertex = max(vertices,key=lambda v:float(direction.dot(v)))
                displacement = vertex-point
                def slope(alpha):
                    return float(objective(point+alpha*displacement)[1].dot(displacement))
                alpha = 1. if slope(1.) <= 0 else brentq(slope,0.,1.,xtol=1e-14)
                point = repair(point+alpha*displacement)
                corrections += 1
                candidate_result = solve(point)
                candidate_point = repair(candidate_result.x)
                if objective(candidate_point)[0] < objective(point)[0]:
                    point,result = candidate_point,candidate_result
                restarts += 1
            if oracle_gap(point) > 1e-5:
                raise RuntimeError("Numerical best response failed its convex-gap tolerance")
        feasible = min(float(np.min(c["fun"](point))) for c in constraints) >= -1e-8
        p = predictions(point)
        original = [self.margin+self.factor*point[0], self.factor*point[1], self.factor*point[2]]
        gap = None
        if sign is not None:
            unused_value, gradient = objective(point)
            gap = max(0.,float(gradient.dot(point))+likelihood.support_float(-gradient,sign))
        return {"point":point, "theta":original, "means":p,
                "success":bool(feasible and (gap <= 1e-5 if sign is not None else result.success)),
                "optimizer_reported_success":bool(result.success),"iterations":int(result.nit),
                "message":str(result.message), "oracle_gap_float_diagnostic":gap,
                "oracle_restarts":restarts,"feasible_line_corrections":corrections}

    def choose(self, trials, successes, batch_size=1):
        rows = observations(self.arms, trials, successes)
        n = np.array([row["trials"] for row in rows], dtype=np.int64)
        k = np.array([row["successes"] for row in rows], dtype=np.int64)
        total = int(n.sum())
        if total <= self.last_total:
            raise ValueError("Each allocation update requires fresh paid observations")
        self.last_total = total
        if int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if (n == 0).any():
            return {"arm":int(np.flatnonzero(n == 0)[0]), "batch_size":1,
                    "stage":"paid_one_pull_per_arm_initialization", "observed_shots":total}
        self.ever_batched |= batch_size > 1
        means = np.clip(k/n, self.margin, 1-self.margin)
        argument = (1+self.a)*(1+self.b)*math.log(max(2,total))
        bonus = argument+math.log(argument)+.5
        intervals = []
        for mean, count in zip(means,n):
            radius = bonus/count
            def equation(q): return float(bernoulli_kl(mean,q))-radius
            lo = self.margin if equation(self.margin) <= 0 else brentq(equation,self.margin,mean,xtol=1e-13)
            hi = 1-self.margin if equation(1-self.margin) <= 0 else brentq(equation,mean,1-self.margin,xtol=1e-13)
            intervals.append((lo,hi))
        lower, upper = np.array(intervals).T
        fitted = self._fit(means,n,lower=lower,upper=upper,initial=self.fit_start)
        if fitted["success"]:
            self.fit_start = fitted["point"]
            self.label = 1 if fitted["theta"][2] >= 0 else -1
        # DKM permits an arbitrary answer when its confidence/model intersection is empty.
        learner = self.learners[self.label]
        weight, rate = learner.distribution()
        response = self._fit(means,weight,sign=self.label,initial=self.null_start[self.label])
        self.null_start[self.label] = response["point"]
        optimistic = np.maximum(bonus/n, np.maximum(bernoulli_kl(lower,response["means"]),
                                                    bernoulli_kl(upper,response["means"])))
        learner.update(-batch_size*optimistic,weight,rate)
        self.target += batch_size*weight
        arm = int(np.argmin(n/self.target))
        return {"arm":arm,"batch_size":int(batch_size),"stage":"dkm_game",
                "observed_shots":total,"selected_sign":self.label,"allocation":weight.tolist(),
                "optimistic_payoffs":optimistic.tolist(),"exploration_bonus":bonus,
                "confidence_intervals":[list(interval) for interval in intervals],
                "tracking_target":self.target.tolist(),"fitted_theta":fitted["theta"],
                "model_projection_success":fitted["success"],"best_response_theta":response["theta"],
                "best_response_success":response["success"],"best_response_iterations":response["iterations"],
                "best_response_gap_float_diagnostic":response["oracle_gap_float_diagnostic"],
                "best_response_restarts":response["oracle_restarts"],
                "best_response_feasible_line_corrections":response["feasible_line_corrections"],
                "design_margin":self.margin,
                "schedule":"explicit_batch_adaptation" if self.ever_batched else "per_observation_DKM_schedule"}


def run_trial(case, seed, cap=4096, delta=.01, arms=None, calibration="kk21", batch_size=1,
              check_every=1):
    """Small executable driver; simulator truth never enters allocation/certificates."""
    sampler = DKMSampler(arms)
    true_means = sampler.x.dot(np.array([case["A"],case["B"],case["C"]]))
    if np.min(true_means) < 0 or np.max(true_means) > 1:
        raise ValueError("Invalid simulator instance")
    if cap < len(sampler.arms) or check_every < 1:
        raise ValueError("Invalid cap/check schedule")
    rng = np.random.RandomState(seed)
    n, k = np.zeros(len(sampler.arms),dtype=np.int64), np.zeros(len(sampler.arms),dtype=np.int64)
    history, last_check, final = [], 0, None
    while True:
        total = int(n.sum())
        proposed = min(batch_size,max(1,cap-total))
        plan = sampler.choose(n,k,proposed)
        # Current-count projection and current-count test precede the next pull.
        # At a terminal check this allocation computation is unused CPU work.
        if (n > 0).all() and (total-last_check >= check_every or total == cap):
            last_check = total
            rows = observations(sampler.arms,n,k)
            # Projection, selected label, and tested likelihood use the same data.
            sign = sampler.label
            fit = likelihood.null_fit(rows,sign)
            means = k/n
            if calibration == "kk21":
                numerator = float((xlogy(k,means)+xlogy(n-k,1-means)).sum())
                threshold = float(kk_calibration_upper(str(delta),len(n)))+float((3*np.log1p(np.log(n))).sum())
            elif calibration == "kt_safe":
                numerator = float((gammaln(k+.5)+gammaln(n-k+.5)-gammaln(n+1)-2*gammaln(.5)).sum())
                threshold = math.log(1/delta)
            else:
                raise ValueError("Unknown calibration")
            if numerator-fit["loglik"] >= threshold-1e-6 or total == cap:
                final = certificate(sampler.arms,n,k,sign,delta,calibration,fit["candidate"])
                if final["decision"] != "abstain" or total == cap:
                    break
        arm, length = plan["arm"],plan["batch_size"]
        fresh = int(rng.binomial(length,true_means[arm]))
        n[arm] += length
        k[arm] += fresh
        history.append(dict(plan,successes=fresh))
    return {"method":"dkm_"+calibration+("_batch_adapt" if batch_size > 1 else "_one_shot"),
            "seed":int(seed),"cap":int(cap),"shots":int(n.sum()),"decision":final["decision"],
            "trials":n.tolist(),"successes":k.tolist(),"arms":sampler.arms,"history":history,
            "certificate":final,"requested_batch_size":batch_size,"check_every":check_every,
            "terminal_plan":plan,"terminal_plan_executed":False,
            "stopping_schedule":"current-count projection and sign, then current-count test, before next paid draw",
            "claim_boundary":"Finite explicit grid and numerical oracle implementation; no full-field SOTA or exact-oracle sample-complexity claim"}
