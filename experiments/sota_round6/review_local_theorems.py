"""Root review of two local oracle theorems; no finite-cost claim."""
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[2]


def main():
    output = ROOT/'results/sota_round6/theory/local_theorems_root_review.json'
    if output.exists():
        raise FileExistsError('Preserve completed review')
    names = ('reflected_pair_small_background_theorem',
             'reflected_pair_near_symmetric_theorem')
    files, reproduced = [], {}
    for name in names:
        source = ROOT/('results/sota_round6/theory/'+name+'.py')
        result = source.with_suffix('.json')
        spec = importlib.util.spec_from_file_location(name, str(source))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        check = getattr(module, 'run_checks', None) or module.checks
        reproduced[name] = len(check())
        payload = json.loads(result.read_text(encoding='utf-8'))
        for rel, expected in payload['source_sha256'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == expected
        files.extend((source, result))
    t, lam, B = sp.symbols('t lam B', real=True)
    D = 1/(1-t*t)-2*lam+lam*lam
    H = t*t*lam-4*B*B*D
    assert sp.simplify(D-((lam-1)**2+t*t/(1-t*t))) == 0
    assert sp.simplify((1-t*t)*D-t*t*lam-(lam-1)*((1-t*t)*lam-1)) == 0
    assert sp.simplify((4*D+(4*B*D)**2/H)/2-2*D*t*t*lam/H) == 0
    q, c, gp, gm = sp.symbols('q c gp gm', real=True)
    w = sp.symbols('w', real=True)
    KA, KC = w*gp+(1-w)*gm, w*gp-(1-w)*gm
    QA, QC = q*(1-q)*(gp-gm)/(2*c), q*(1-q)*(gp+gm)/(2*c)
    assert sp.simplify(QA-KA*QC/KC+q*(1-q)*gp*gm/(c*KC)) == 0
    payload = {
        'status': 'independent_root_analytic_review_passed',
        'new_simulation_observations': 0,
        'blocking_findings': [],
        'reviewed_claims': [
            'For every fixed 0<A<1 and 0<C<min(A,1-A), a sufficiently small neighborhood of B=0 admits an exact globally optimal reflected pair and a physical affine tangent null.',
            'For every fixed B,C with C>0 and B^2+C^2<1/4, a sufficiently small neighborhood of A=1/2 admits the same exact reflected-pair/global-null conclusion.'
        ],
        'analytic_checks': [
            'At B=0, strict convexity of Bernoulli KL in p gives a unique pair of global contacts at z=0; the two branch second derivatives are strictly negative because the capacity center is strictly between the endpoint rates.',
            'At A=1/2, rho is positive definite. For nonzero Hermitian E, R E R E R = R^(1/2)(R^(1/2) E R^(1/2))^2 R^(1/2) is nonzero positive semidefinite. Thus Tr(rho log H) has strictly negative Hessian, making the previous tangent bound strict unless H=H*. Distinct eigenvalues of H* force PSR contacts.',
            'The factorization (1-t^2)D-t^2 lambda=(lambda-1)((1-t^2)lambda-1) plus 1<lambda<1/(1-t^2) proves strict negative baseline curvature. Strict physicality supplies 4B^2<1-t^2.',
            'Symmetric entropy secants and equal endpoint capacities are even analytic functions of c; consequently they extend analytically in c^2 across z=+/-1 when all rates retain a strict interior margin.',
            'For both theorems, choose a closed contact neighborhood with uniform negative second derivative and its compact exterior with a strict value gap. Local C2 convergence and global C0 convergence preserve both properties. The implicit-function maximum and affine tangent therefore give global domination, not only stationary contacts.',
            'Both null coefficients remain strictly physical by continuity from an interior null. Any design has infimum at most the integrated KL to this single null; the reflected pair with capacity weights supplies the matching lower bound, including physical boundary nulls. No unjustified minimax exchange is used.',
            'The extension to opposite-sign nulls is valid because their two rates have reversed order; the weighted Bernoulli order-cone projection pools at the capacity center, attainable by a physical constant null.',
            'Small-B expansions follow from R_zB=K_A and R_zz=-C K_C; near-symmetry expansions follow from K_AA=4D, R_zalpha=4BD and R_zz=-H. The stated odd/even remainders follow from probability/angle symmetries.'
        ],
        'symbolic_identity_checks': 4,
        'reproduced_110_digit_diagnostic_cases': reproduced,
        'scope_limits': [
            'Neither proof supplies a numerical epsilon; the diagnostic nonzero perturbations are not individually certified by these existence proofs.',
            'These are separate open neighborhoods, not an all-parameter proof or a continuation argument covering the entire physical region.',
            'The checks of numerical contacts reproduce the author functions; analytic review and symbolic simplifications are separate evidence. They do not constitute proof-assistant verification.',
            'No novelty, paid-learning, finite-risk stopping, cost improvement or SOTA claim is accepted.'
        ],
        'source_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in files+[Path(__file__).resolve(), ROOT/'docs/CONTINUOUS_SYMMETRIC_DESIGN.md']}
    }
    with output.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({k: payload[k] for k in ('status', 'symbolic_identity_checks', 'reproduced_110_digit_diagnostic_cases')}))


if __name__ == '__main__':
    main()
