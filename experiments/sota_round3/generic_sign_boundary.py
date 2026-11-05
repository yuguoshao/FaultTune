"""Finite-horizon sign boundaries shaped by classical sequential statistics.

Floating scores only design a frozen symmetric INTEGER boundary. The actual
test and its null-risk certificate use that table and exact integer path
counts. No claim that a floating level set is an exact real GLR level set is
needed for risk validity. No simulator data are read during calibration.
"""
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/structured_validation'))
from expert_config import expert_config

VERSION = 'finite_integer_tail_boundary_v1'
_VERIFIED = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_hash(value):
    data = {k: v for k, v in value.items() if k != 'artifact_sha256'}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def source_hashes():
    paths = [Path(__file__).resolve(), ROOT/'experiments/structured_validation/expert_config.py']
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def validate_table(table):
    if (not isinstance(table, (list, tuple)) or len(table) < 2
            or type(table[0]) is not int or table[0] != 1):
        raise ValueError('Boundary must contain index zero and a positive horizon')
    for n, b in enumerate(table[1:], 1):
        if isinstance(b, bool) or not isinstance(b, int) or not n//2 < b <= n+1:
            raise ValueError('Invalid symmetric integer tail boundary')


def exact_risk(table):
    """Independent full Pascal-row recurrence, with absorbed states zeroed."""
    validate_table(table)
    survivors = [1]
    for n, b in enumerate(table[1:], 1):
        fresh = [0]*(n+1)
        for k in range(max(0, n-b+1), min(n, b-1)+1):
            fresh[k] = ((survivors[k] if k < n else 0)
                        + (survivors[k-1] if k else 0))
        survivors = fresh
    denominator = 1 << (len(table)-1)
    numerator = denominator-sum(survivors)
    return numerator, denominator


def float_risk(table):
    """Search aid only; risk acceptance always uses exact_risk."""
    survivors = np.ones(1)
    for n, b in enumerate(table[1:], 1):
        fresh = np.zeros(n+1)
        fresh[:-1] += .5*survivors
        fresh[1:] += .5*survivors
        fresh[:max(0, n-b+1)] = 0.
        fresh[min(n+1, b):] = 0.
        survivors = fresh
    return max(0., 1.-float(survivors.sum()))


def score_rows(maximum, shape):
    """Upper tails, from floor(n/2)+1 through n; all shapes are symmetric."""
    rows = []
    if shape in ('pair_grid', 'pairwise_hedge'):
        experts = expert_config()
        odds = np.array([x['numerator']/x['denominator'] for x in experts])
        q = odds/(1+odds)
        lq, lc = np.log(2*q), np.log(2*(1-q))
        weights = np.log([x['weight'][0]/x['weight'][1] for x in experts])
    elif shape.startswith('sprt_'):
        q = float(shape.split('_', 1)[1])
        if not .5 < q < 1:
            raise ValueError('SPRT effect must lie strictly between .5 and 1')
    for n in range(1, maximum+1):
        k = np.arange(n//2+1, n+1, dtype=float)
        c = n-k
        if shape == 'maxsprt':
            score = xlogy(k, 2*k/n)+xlogy(c, 2*c/n)
        elif shape == 'beta':
            score = (gammaln(k+.5)+gammaln(c+.5)-gammaln(n+1)
                     -2*gammaln(.5)+n*math.log(2))
        elif shape in ('pair_grid', 'pairwise_hedge'):
            score = logsumexp(k[:, None]*lq+c[:, None]*lc+weights, axis=1)
            if shape == 'pairwise_hedge':
                beta = (gammaln(k+.5)+gammaln(c+.5)-gammaln(n+1)
                        -2*gammaln(.5)+n*math.log(2))
                score = np.logaddexp(score, beta)-math.log(2)
        elif shape.startswith('sprt_'):
            score = np.logaddexp(k*math.log(2*q)+c*math.log(2*(1-q)),
                                 c*math.log(2*q)+k*math.log(2*(1-q)))-math.log(2)
        else:
            raise ValueError('Unknown score shape: '+shape)
        if np.any(np.diff(score) < -1e-10) or not np.all(np.isfinite(score)):
            raise ValueError('Score upper tail is not finite and increasing')
        rows.append(score)
    return rows


def level_table(rows, threshold):
    return [1]+[n//2+1+int(np.searchsorted(row, threshold, side='left'))
                for n, row in enumerate(rows, 1)]


def calibrate(maximum, shape, delta='0.01', log_threshold_scale=1000000):
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError('Positive integer horizon required')
    risk = Fraction(str(delta))
    if not 0 < risk < 1:
        raise ValueError('Risk must lie strictly between zero and one')
    if type(log_threshold_scale) is not int or log_threshold_scale <= 0:
        raise ValueError('Positive integer threshold scale required')
    rows = score_rows(maximum, shape)
    scale = log_threshold_scale
    lo, hi = 0, int(math.ceil(math.log(1/float(risk))*scale))
    while float_risk(level_table(rows, hi/scale)) > float(risk):
        hi *= 2
    while hi-lo > 1:
        mid = (lo+hi)//2
        if float_risk(level_table(rows, mid/scale)) <= float(risk):
            hi = mid
        else:
            lo = mid
    table = level_table(rows, hi/scale)
    numerator, denominator = exact_risk(table)
    # Float search can only propose a table. Integer arithmetic decides.
    adjustments = 0
    while numerator*risk.denominator > denominator*risk.numerator:
        hi += 1
        adjustments += 1
        table = level_table(rows, hi/scale)
        numerator, denominator = exact_risk(table)
    previous = level_table(rows, (hi-1)/scale)
    pn, pd = exact_risk(previous)
    result = {'version': VERSION, 'shape': shape,
              'status': 'integer_boundary_exact_risk_certified',
              'max_discordances': maximum, 'boundaries': table,
              'delta_exact': {'numerator': str(risk.numerator), 'denominator': str(risk.denominator)},
              'design_log_threshold': hi/scale,
              'search': {'log_threshold_scale': scale, 'selected_tick': hi,
                         'integer_acceptance_adjustments': adjustments,
                         'previous_tick_risk_exceeds_delta': pn*risk.denominator > pd*risk.numerator},
              'fair_null_risk': {'crossing_numerator': str(numerator), 'fair_denominator': str(denominator),
                                 'probability_float_diagnostic': numerator/denominator},
              'boundary_encoding': 'B[n]; reject at plus>=B[n], accept at plus<=n-B[n]; zero score never stops',
              'source_hashes': source_hashes(),
              'claim_boundary': 'Classical score-shaped finite-horizon test. Integer table and risk are exact; floating design level sets are not certified real-arithmetic GLR levels. No novelty or performance claim.'}
    result['artifact_sha256'] = payload_hash(result)
    return result


@dataclass(frozen=True)
class Boundary:
    table: tuple
    artifact_sha256: str
    shape: str


def verify_artifact(artifact):
    if (artifact['version'] != VERSION
            or artifact['status'] != 'integer_boundary_exact_risk_certified'
            or artifact['source_hashes'] != source_hashes()):
        raise ValueError('Unrecognized version, status or changed source binding')
    if artifact['artifact_sha256'] != payload_hash(artifact):
        raise ValueError('Boundary artifact hash mismatch')
    table = artifact['boundaries']
    validate_table(table)
    if (type(artifact['max_discordances']) is not int
            or len(table) != artifact['max_discordances']+1):
        raise ValueError('Boundary horizon mismatch')
    n, d = exact_risk(table)
    saved = artifact['fair_null_risk']
    delta = artifact['delta_exact']
    if not 0 < int(delta['numerator']) < int(delta['denominator']):
        raise ValueError('Invalid declared risk fraction')
    if n != int(saved['crossing_numerator']) or d != int(saved['fair_denominator']):
        raise ValueError('Exact path count mismatch')
    if n*int(delta['denominator']) > d*int(delta['numerator']):
        raise ValueError('Exact risk exceeds budget')
    verified = Boundary(tuple(table), artifact['artifact_sha256'], artifact['shape'])
    _VERIFIED[verified.artifact_sha256] = verified
    return verified


def certificate_from_pairs(counts, boundary):
    if not isinstance(boundary, Boundary) or _VERIFIED.get(boundary.artifact_sha256) != boundary:
        raise ValueError('Runtime boundary must first pass verify_artifact')
    if len(counts) != 4 or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts):
        raise ValueError('Four nonnegative integer pair counts required')
    n, plus = counts[1]+counts[2], counts[1]
    if n >= len(boundary.table):
        raise ValueError('Certified discordance horizon exceeded')
    b = boundary.table[n]
    decision = 'reject' if plus >= b else 'accept' if plus <= n-b else 'abstain'
    return {'decision': decision, 'shots': 2*sum(counts), 'discordances': n,
            'score': counts[1]-counts[2], 'integer_boundary': b,
            'boundary_sha256': boundary.artifact_sha256, 'shape': boundary.shape,
            'finite_horizon_only': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maximum', type=int, default=2048)
    parser.add_argument('--shape', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    artifact = calibrate(args.maximum, args.shape)
    verify_artifact(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'path': str(args.output), 'shape': args.shape,
                      'threshold': artifact['design_log_threshold'],
                      'risk': artifact['fair_null_risk']['probability_float_diagnostic']}))


if __name__ == '__main__':
    main()
