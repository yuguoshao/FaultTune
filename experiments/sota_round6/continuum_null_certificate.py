"""All-angle oracle bounds for exact, strictly interior Bernoulli models.

Two rational charts cover the circle. Directed Decimal interval arithmetic,
including a second-order Taylor enclosure on every accepted subinterval,
certifies a proposed affine null against ALL angles, not a sampled mesh.
An exact reflected design supplies a lower bound. This is a fixed-model
information certificate, not an implementable stopping rule or SOTA result.
"""
import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.circuit_validation.direct_tests import _Bounds as I


def enclosure(value):
    value = Fraction(value)
    return I(value.numerator) / value.denominator


def square(value):
    """Tight interval square, including intervals that cross zero."""
    value = I.as_bounds(value)
    if value.lo <= 0 <= value.hi:
        return I(0, max((I(value.lo) * value.lo).hi,
                        (I(value.hi) * value.hi).hi))
    return value * value


def abs_upper(value):
    return max(value.lo.copy_abs(), value.hi.copy_abs())


def intersect(left, right):
    return I(max(left.lo, right.lo), min(left.hi, right.hi))


def kl(p, q):
    return p * (p / q).log() + (1-p) * ((1-p)/(1-q)).log()


def logit(p):
    return p.log() - (1-p).log()


def chart(t, sign):
    """(cos, sin) and their first two t derivatives on [-1,1]."""
    if sign not in (-1, 1):
        raise ValueError('Chart sign must be -1 or 1')
    t2 = square(t)
    den = 1+t2
    den2 = square(den)
    den3 = den2*den
    c, s = sign*(1-t2)/den, 2*t/den
    cp, sp = -4*sign*t/den2, 2*(1-t2)/den2
    cpp = -4*sign*(1-3*t2)/den3
    spp = -4*t*(3-t2)/den3
    return c, s, cp, sp, cpp, spp


class Model:
    def __init__(self, theta, null):
        self.theta = tuple(Fraction(v) for v in theta)
        self.null = tuple(Fraction(v) for v in null)
        A, B, C = self.theta
        a, b = self.null
        if not 0 < A < 1 or B*B+C*C >= min(A, 1-A)**2:
            raise ValueError('Strict physical interior required')
        if not 0 < a-abs(b) <= a+abs(b) < 1:
            raise ValueError('Strict physical affine null required')
        self.A, self.B, self.C = map(enclosure, self.theta)
        self.a, self.b = map(enclosure, self.null)
        radius = enclosure(B*B+C*C).sqrt()
        self.p_range = I((self.A-radius).lo, (self.A+radius).hi)
        self.q_range = I(enclosure(a-abs(b)).lo,
                         enclosure(a+abs(b)).hi)
        if not 0 < self.p_range.lo <= self.p_range.hi < 1:
            raise ValueError('Physical margin needs greater precision')

    def derivatives(self, t, sign):
        c, s, cp, sp, cpp, spp = chart(t, sign)
        p = intersect(self.A+self.B*c+self.C*s, self.p_range)
        q = intersect(self.a+self.b*c, self.q_range)
        pp, qp = self.B*cp+self.C*sp, self.b*cp
        ppp, qpp = self.B*cpp+self.C*spp, self.b*cpp
        dp = logit(p)-logit(q)
        dq = (q-p)/(q*(1-q))
        dpp = I(1)/(p*(1-p))
        dpq = -I(1)/(q*(1-q))
        dqq = p/square(q)+(1-p)/square(1-q)
        first = dp*pp+dq*qp
        second = (dpp*square(pp)+2*dpq*pp*qp+dqq*square(qp)
                  +dp*ppp+dq*qpp)
        return kl(p, q), first, second

    def interval_upper(self, left, right, sign):
        mid = (left+right)/2
        half = (right-left)/2
        box = I(enclosure(left).lo, enclosure(right).hi)
        direct, _, second = self.derivatives(box, sign)
        value, first, _ = self.derivatives(enclosure(mid), sign)
        h = enclosure(half)
        taylor = value + abs_upper(first)*h + abs_upper(second)*square(h)/2
        return min(direct.hi, taylor.hi)

    def reflected_lower(self, z):
        z = Fraction(z)
        if not -1 < z < 1 or self.theta[2] == 0:
            raise ValueError('Interior cosine and nonzero signal required')
        zi = enclosure(z)
        s = enclosure(1-z*z).sqrt()
        center = self.A+self.B*zi
        spread = enclosure(abs(self.theta[2]))*s
        pm, pp = center-spread, center+spread
        f = lambda p: p*p.log()+(1-p)*(1-p).log()
        secant = (f(pp)-f(pm))/(pp-pm)
        qc = I(1)/(1+(-secant).exp())
        left, right = kl(pm, qc), kl(pp, qc)
        radius = intersect(left, right)  # exact capacity values coincide
        if radius.lo <= 0:
            raise ValueError('Positive lower bound needs greater precision')
        return radius


def certify(theta, null, z, relative_tolerance=Fraction(1, 10**6),
            max_nodes=100000, max_depth=60):
    model = Model(theta, null)
    tolerance = Fraction(relative_tolerance)
    if tolerance <= 0:
        raise ValueError('Positive relative tolerance required')
    lower = model.reflected_lower(z)
    # A downward target ensures accepted bounds imply U/L <= 1+tolerance.
    target = (I(lower.lo)*enclosure(1+tolerance)).lo
    pending = [(sign, Fraction(-1), Fraction(1), 0) for sign in (-1, 1)]
    accepted, unresolved = [], []
    visited = 0
    while pending:
        sign, left, right, depth = pending.pop()
        upper = model.interval_upper(left, right, sign)
        visited += 1
        leaf = {'chart': sign, 'left': str(left), 'right': str(right),
                'upper': str(upper), 'depth': depth}
        if upper <= target:
            accepted.append(leaf)
        elif depth >= max_depth or visited >= max_nodes:
            unresolved.append(leaf)
            unresolved.extend({'chart': sg, 'left': str(le), 'right': str(ri),
                               'depth': de, 'upper': None}
                              for sg, le, ri, de in pending)
            pending = []
        else:
            mid = (left+right)/2
            pending.extend(((sign, left, mid, depth+1),
                            (sign, mid, right, depth+1)))
    result = {'status': 'certified' if not unresolved else 'unresolved',
              'theta_exact': list(map(str, model.theta)),
              'null_exact': list(map(str, model.null)),
              'cosine_exact': str(Fraction(z)),
              'pair_information_interval': [str(lower.lo), str(lower.hi)],
              'relative_tolerance_exact': str(tolerance),
              'acceptance_target_downward': str(target),
              'visited_nodes': visited, 'accepted_leaves': accepted,
              'unresolved_leaves': unresolved}
    if not unresolved:
        upper = max(Decimal(leaf['upper']) for leaf in accepted)
        result['all_angle_information_upper'] = str(upper)
        result['upper_over_pair_lower'] = str((I(upper)/I(lower.lo)).hi)
    return result


def run_registry(indices, output):
    source = ROOT/'results/sota_round5/continuous_chernoff_refinement.json'
    registry = json.loads(source.read_text(encoding='utf-8'))
    if output.exists():
        raise FileExistsError('Preserve prior certificates')
    rows = []
    started = time.monotonic()
    for index in indices:
        row = registry['rows'][index]
        theta = [Fraction(row['model'][key]) for key in ('A', 'B', 'C')]
        null = list(map(Fraction, row['reflected_pair']['null_candidate'][:2]))
        z = Fraction(row['reflected_pair']['cosine'])
        one = certify(theta, null, z)
        one['registry_index'] = index
        rows.append(one)
        print(json.dumps({'index': index, 'status': one['status'],
                          'nodes': one['visited_nodes'],
                          'ratio': one.get('upper_over_pair_lower')}), flush=True)
    result = {
        'status': 'complete_fixed_registry_continuum_certification',
        'new_simulation_observations': 0,
        'source_parameter_semantics': 'Every binary64 A,B,C,a,b,z is an exact Fraction; subsequent probabilities are not rounded back to binary64.',
        'proof': [
            'Both charts (cos,sin)=(sign*(1-t^2)/(1+t^2),2*t/(1+t^2)), t in[-1,1], sign in{-1,+1}, cover the circle.',
            'Every accepted interval has a directed-arithmetic KL upper bound, taking the smaller of direct interval evaluation and Taylor value + |first derivative|*halfwidth + sup|second derivative|*halfwidth^2/2.',
            'The exact dyadic leaves partition both entire chart domains. The null is one fixed physical affine function throughout.',
            'A reflected pair at the registered exact cosine achieves its Bernoulli capacity using capacity weights. Its directed lower bound is <= V. For any angle distribution, the proposed null bounds its infimum by the global pointwise upper, so V <= U.',
            'Certification of U/L <= 1.000001 proves only a fixed-model oracle information gap, with arbitrary support sizes allowed.'
        ],
        'rows': rows,
        'certified_models': sum(r['status']=='certified' for r in rows),
        'unresolved_models': sum(r['status']!='certified' for r in rows),
        'elapsed_seconds': time.monotonic()-started,
        'source_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__).resolve(), source,
                                    ROOT/'experiments/circuit_validation/direct_tests.py')},
        'claim_boundary': 'Numerical all-angle certificates for listed exact models only. No all-parameter theorem, novelty claim, paid-learning guarantee, finite-stopping superiority or SOTA success.'
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('certified_models', 'unresolved_models', 'elapsed_seconds')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices', default='all')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    indices = range(48) if args.indices == 'all' else [int(x) for x in args.indices.split(',')]
    run_registry(indices, args.output)
