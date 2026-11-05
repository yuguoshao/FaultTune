"""Independent integer-risk audit of the fixed round4 mixture registry."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/sota_round2/theory'))
import verify_integer_tail_dp as verifier


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_capital(n, k, spec):
    answer = Fraction(0)
    for component in spec['components']:
        weight = Fraction(*component['weight'])
        if component['kind'] == 'symmetric_two_point':
            q = Fraction(*component['q'])
            answer += weight*((2*q)**k*(2-2*q)**(n-k)+(2*q)**(n-k)*(2-2*q)**k)/2
        else:
            assert component['kind'] == 'jeffreys_beta' and component['alpha'] == [1, 2]
            # Exact beta moment from the integer rising-factorial identity.
            numerator, denominator = 1, 1
            for j in range(k):
                numerator *= 2*j+1
            for j in range(n-k):
                numerator *= 2*j+1
            denominator = math.factorial(n)
            answer += weight*Fraction(numerator, denominator)
    return answer


def main():
    manifest_path = ROOT/'results/sota_round4/lr_mixture_boundaries/calibration_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete' and manifest['max_discordances'] == 8192
    assert manifest['physical_shot_cap'] == 16384 and len(manifest['records']) == 11
    for name, expected in manifest['source_hashes'].items():
        assert digest(ROOT/name) == expected
    results = []
    for row in manifest['records']:
        path = ROOT/row['path']
        assert digest(path) == row['sha256']
        artifact = json.loads(path.read_text(encoding='utf-8'))
        assert artifact['shape'] == row['shape']
        assert artifact['max_physical_pairs'] == artifact['max_discordances'] == 8192
        assert artifact['physical_shot_cap'] == 16384
        for name, expected in artifact['mixture_source_hashes'].items():
            assert digest(ROOT/name) == expected
        spec = artifact['fixed_prior_spec']
        assert sum((Fraction(*c['weight']) for c in spec['components']), Fraction(0)) == 1
        assert all(Fraction(*c['weight']) > 0 for c in spec['components'])
        for n in range(13):
            capitals = [exact_capital(n, k, spec) for k in range(n+1)]
            assert capitals == list(reversed(capitals))
            assert sum((math.comb(n,k)*c for k,c in enumerate(capitals)), Fraction(0)) == (1 << n)
            assert all(capitals[k] <= capitals[k+1] for k in range(n//2+1,n))
        audit = verifier.audit_artifact(artifact)
        audit.update(shape=row['shape'], input_path=row['path'], input_sha256=row['sha256'],
                     positive_exact_prior_weights_sum_to_one=True,
                     exact_rational_small_count_mean_one_symmetry_and_upper_tail_monotonicity=True)
        results.append(audit)
        print(json.dumps({'shape': row['shape'], 'status': audit['status'],
                          'risk': audit['float_probability_diagnostic']}), flush=True)
    result = {'status': 'passed', 'new_simulation_observations': 0,
              'case_parameters_or_observations_read': False,
              'independent_integer_dp_all_tables': True,
              'manifest_sha256': digest(manifest_path), 'results': results,
              'source_review': {
                  'two_point_mixture': 'Each symmetric q component has half its declared weight at q and half at1-q. The score equals log of the sum of weighted normalized Bernoulli likelihood ratios against .5.',
                  'jeffreys_component': 'The beta component uses B(k+.5,n-k+.5)/B(.5,.5)*2^n. The independently evaluated rising-factorial formula checks its exact moments.',
                  'shape_monotonicity': 'Every symmetric two-point LR is a positive constant times cosh((2k-n)*log(q/(1-q))/2), hence increases with distance from n/2. Integration proves the same for symmetric beta; positive mixtures preserve it.',
                  'finite_risk': 'The saved symmetric integer boundaries, not real score contours, define the tests. Full-vector integer recursion independently recounts first crossings and survivors through8192 for every table and checks risk<=1/100.',
                  'generic_compatibility': 'Producer uses generic VERSION, ordinary generic source bindings, payload hash over extra metadata, max-discordance horizon, four count categories, and identical direction encoding. Separate mixture-source/prior/physical-budget verification is additive.',
                  'runtime_scope': 'All-pair inspection up to16384 physical shots; strict wrong-sign control uses the already reviewed generic symmetric-tail lemma. Disallow using the smaller physical cap merely because discordances remain below8192.',
                  'minimality': 'This independent audit does not recompute floating score rows or the previous threshold tick. It certifies risk of the saved integer tables, not exact real LR contour equality or independent threshold minimality.',
                  'literature_scope': 'Classical fixed LR-mixture priors with finite-horizon calibration. No statistical novelty, oracle parameter access, observed performance, or SOTA claim is inferred.'},
              'source_sha256': {Path(__file__).resolve().relative_to(ROOT).as_posix(): digest(__file__),
                                'experiments/sota_round2/theory/verify_integer_tail_dp.py': digest(verifier.__file__),
                                **manifest['source_hashes']}}
    path = Path(__file__).with_suffix('.json')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'artifact': str(path), 'sha256': digest(path)}), flush=True)


if __name__ == '__main__':
    main()
