"""Reproducible sampling-only KKT cross-check against independent SciPy SLSQP.

This numerical comparison is evidence about implementation accuracy, not a
rigorous optimization certificate and not a prerequisite for statistical safety.
Run from this directory with a Python environment providing NumPy and SciPy:
  python audit_combgame.py --output ../../results/combgame_numerical_audit.json
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

from combgame import best_response, bernoulli_kl


def run_audit(seed=93291, cases=200):
    import numpy
    import scipy
    from scipy.optimize import minimize

    rng = random.Random(seed)
    coefficients = [-.3, -.7, .3, .7]
    records = []
    began = time.perf_counter()
    for case in range(cases):
        probabilities = [rng.uniform(.01, .99) for unused in coefficients]
        weights = [rng.uniform(.01, 1.) for unused in coefficients]
        mass = sum(weights)
        weights = [w / mass for w in weights]
        qs = best_response(probabilities, weights, coefficients)

        def objective(vector):
            return sum(w * bernoulli_kl(p, q)
                       for w, p, q in zip(weights, probabilities, vector))

        result = minimize(
            objective, [.5] * 4, method='SLSQP', bounds=[(1e-9, 1 - 1e-9)] * 4,
            constraints=[{'type': 'eq', 'fun': lambda x: sum(a * q for a, q in zip(coefficients, x))}],
            options={'ftol': 1e-12, 'maxiter': 500},
        )
        records.append({
            'case': case, 'probabilities': probabilities, 'weights': weights,
            'combgame_q': qs, 'slsqp_q': result.x.tolist(),
            'slsqp_success': bool(result.success), 'slsqp_message': str(result.message),
            'combgame_objective': objective(qs), 'slsqp_objective': float(result.fun),
            'absolute_objective_difference': abs(objective(qs) - result.fun),
            'absolute_equality_residual': abs(sum(a * q for a, q in zip(coefficients, qs))),
        })
    directory = Path(__file__).resolve().parent
    source_hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                     for name in ('combgame.py', 'kl_trackstop.py', 'audit_combgame.py')}
    max_difference = max(r['absolute_objective_difference'] for r in records)
    max_residual = max(r['absolute_equality_residual'] for r in records)
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'passed' if all(r['slsqp_success'] for r in records)
                  and max_difference < 1e-8 and max_residual < 1e-10 else 'failed',
        'purpose': 'independent numerical cross-check of sampling optimizer only',
        'rigorous_certificate': False, 'seed': seed, 'cases': cases,
        'probability_distribution': 'independent uniform(.01,.99)',
        'weight_distribution': 'independent uniform(.01,1), normalized by sum',
        'coefficients': coefficients, 'constraint': 'a.q=0',
        'bounds': [1e-9, 1 - 1e-9], 'slsqp_initial': [.5] * 4,
        'slsqp_ftol': 1e-12, 'slsqp_maxiter': 500,
        'pass_tolerances': {'objective_difference': 1e-8, 'equality_residual': 1e-10},
        'max_objective_difference': max_difference, 'max_equality_residual': max_residual,
        'elapsed_seconds': time.perf_counter() - began,
        'python_version': sys.version, 'numpy_version': numpy.__version__,
        'scipy_version': scipy.__version__, 'source_sha256': source_hashes,
        'records': records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=93291)
    parser.add_argument('--cases', type=int, default=200)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error('--cases must be positive')
    report = run_audit(args.seed, args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
