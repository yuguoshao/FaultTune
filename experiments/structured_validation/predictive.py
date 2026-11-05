"""Legal block predictive densities; all pseudo-data chosen before outcomes.

The synthetic pseudo-counts only specify a predictive distribution. They are
never observations, never used in a denominator, and never counted as shots.
"""
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import betaln, logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/circuit_validation"))
from direct_tests import _Bounds, _log_marginal, _as_float_bounds


def feature(row):
    return np.array([1., row["cos_numerator"]/row["denominator"],
                     row["sin_numerator"]/row["denominator"]])


def fit_rows(rows):
    gram, rhs = np.eye(3), np.array([.5, 0., 0.])
    total = 0
    for row in rows:
        x, n, k = feature(row), row["trials"], row["successes"]
        gram += n*np.outer(x, x)
        rhs += k*x
        total += n
    raw = np.linalg.solve(gram, rhs)
    a = float(np.clip(raw[0], .0001, .9999))
    radius = float(np.hypot(raw[1], raw[2]))
    scale = min(1., .98*min(a, 1-a)/max(radius, 1e-300))
    projected = np.array([a, raw[1]*scale, raw[2]*scale])
    return {"theta": projected, "raw": raw, "gram": gram, "shots": total}


def arm_key(row):
    return (row["cos_numerator"], row["sin_numerator"], row["denominator"])


def propose_prior(rows, query, policy="structured"):
    """Return integer Jeffreys pseudo-history (successes, trials)."""
    if policy == "per_arm":
        key = arm_key(query)
        for row in rows:
            if arm_key(row) == key:
                return int(row["successes"]), int(row["trials"])
        return 0, 0
    if policy != "structured":
        raise ValueError("unknown predictive policy")
    fit = fit_rows(rows)
    x = feature(query)
    p = float(np.clip(x.dot(fit["theta"]), .0001, .9999))
    leverage = float(x.dot(np.linalg.solve(fit["gram"], x)))
    # A transparent design-leverage heuristic; validity needs only predictability.
    trials = min(fit["shots"], max(0, int(round(1/max(leverage, 1e-300)-1))))
    successes = min(trials, max(0, int(round(p*trials))))
    return successes, trials


def log_predictive(k, n, prior_successes, prior_trials):
    s, f = prior_successes, prior_trials-prior_successes
    return float(betaln(s+k+.5, f+n-k+.5)-betaln(s+.5, f+.5))


def log_predictive_bounds(k, n, prior_successes, prior_trials):
    values = (k, n, prior_successes, prior_trials)
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in values):
        raise ValueError("counts must be integers")
    if not (0 <= k <= n and 0 <= prior_successes <= prior_trials):
        raise ValueError("invalid counts")
    k, n, prior_successes, prior_trials = map(int, values)
    return (_log_marginal(prior_successes+k, prior_trials+n, .5)
            -_log_marginal(prior_successes, prior_trials, .5))


def numerator_bounds(history, engine):
    result = _Bounds(0)
    for block in history:
        s, m = block["priors"][engine]
        result += log_predictive_bounds(block["successes"], block["trials"], s, m)
    return result


def verify_prediction_normalization(prior_successes, prior_trials, n):
    from scipy.special import gammaln
    logs = [gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)
            +log_predictive(k, n, prior_successes, prior_trials) for k in range(n+1)]
    return float(math.exp(logsumexp(logs)))


def checks():
    normalized = []
    for s, m in ((0, 0), (0, 8), (3, 8), (8, 8), (17, 53)):
        for n in (1, 2, 9, 32):
            normalized.append(verify_prediction_normalization(s, m, n))
            for k in (0, n//2, n):
                bounds = log_predictive_bounds(k, n, s, m)
                point = log_predictive(k, n, s, m)
                assert float(bounds.lo)-1e-10 <= point <= float(bounds.hi)+1e-10
    assert max(abs(x-1) for x in normalized) < 1e-12
    for engine in ("per_arm", "structured"):
        assert propose_prior([], {"cos_numerator": 0, "sin_numerator": 1,
                                  "denominator": 1}, engine) == (0, 0)
    return {"status": "passed", "normalized_block_densities": len(normalized),
            "maximum_normalization_error": max(abs(x-1) for x in normalized),
            "outward_marginal_checks": 3*len(normalized),
            "scope": "Predictive normalization and arithmetic, not a power claim."}


if __name__ == "__main__":
    import hashlib
    import json
    result = checks()
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path = ROOT / "results/structured_validation/predictive_checks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
