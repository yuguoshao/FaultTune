"""Close equal-resource finite budgets and optionally penalize all resources."""
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.circuit_validation.direct_tests import _Bounds as I


def enc(value):
    value=Fraction(value)
    return I(value.numerator)/value.denominator


def bounds(pair):
    return I(Decimal(pair[0]),Decimal(pair[1]))


def limits(value):
    return [str(value.lo),str(value.hi)]


def main():
    path=Path(__file__).with_name('noisy_multidepth_phase_obstruction.json')
    data=json.loads(path.read_text(encoding='utf-8'))
    cap=2**20
    glower=bounds(data['score_points']['any_valid_verifier_expected_gate_uses_lower'])
    tlower=bounds(data['score_points']['any_valid_verifier_expected_device_time_lower'])
    rows=[]
    for baseline in data['qualified_baseline_bounds']:
        k=baseline['depth']
        b=enc(baseline['bet_exact'])
        atanh=((1+b)/(1-b)).log()/2
        mu=bounds(baseline['score_point_demodulated_mean'])
        drift=(1-b*b).log()/2+mu*atanh
        distance=cap*drift-I(200).log()
        assert distance.lo>0
        tail=(-distance*distance/(2*cap*atanh*atanh)).exp()
        expected_gate=bounds(baseline['score_point_expected_gate_uses_upper'])
        expected_time=bounds(baseline['score_point_expected_device_time_upper'])
        penalized_gate=expected_gate+(16-k)*cap*tail
        penalized_time=expected_time+enc(Fraction(161,10)-k)*cap*tail
        rows.append({'depth':k,'point_abstention_probability_upper':limits(tail),
                     'optional_full_gate_cap_penalty_cost_upper':limits(penalized_gate),
                     'optional_full_time_cap_penalty_cost_upper':limits(penalized_time),
                     'optional_gate_penalty_speedup_upper':limits(penalized_gate/glower),
                     'optional_time_penalty_speedup_upper':limits(penalized_time/tlower)})
    assert Decimal(rows[0]['optional_gate_penalty_speedup_upper'][1])<Decimal('1.954')
    assert Decimal(rows[0]['optional_time_penalty_speedup_upper'][1])<Decimal('2.045')
    result={
        'status':'explicit_common_budget_closure_and_optional_penalty_check',
        'new_simulation_observations':0,
        'main_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'common_budgets':{'circuit_shots':cap,'sensing_pulse_uses':16*cap,
                          'phase_switches_at_most':cap,
                          'modeled_device_time_exact':str(Fraction(183,10)*cap)},
        'action_rule':'All methods use the same budget triple. Any positive integer depth is allowed if its next paid circuit fits the remaining shot, pulse and device budgets. A method that exhausts any budget returns an explicit decision already justified or abstains. A phase query cannot be delivered and charged afterward if it exceeds a cap.',
        'baseline_feasibility':'The depth8 and depth16 policies both fit all common budgets pathwise for every nuisance value and observation sequence, including cap abstentions. Their previously proved uniform95% power and1% error bounds therefore remain valid under this concrete equal-budget contract.',
        'information_bound_scope':'The main lower bounds allow even unrestricted depths and omit switch cost, so they remain valid for this smaller common budget class. The original primary shot-cost3x exclusion therefore requires no unspecified future resource budget.',
        'original_acceptance_penalty':'The actual and full-cap-penalized primary shot costs agree because these controls only abstain at the common full shot cap. The existing acceptance files are not changed.',
        'optional_stronger_companion_penalties':'For completeness, even if a cap abstention is also charged the full common gate and time caps, the same rounded gate/time speedup exclusions survive. At a depthk cap abstention, extra gate penalty is(16-k)N; extra time penalty is at most(16.1-k)N since actual time is at least(k+2.2)N. The pointwise Hoeffding bound controls the probability of that event.',
        'optional_penalty_results':rows,
        'scope':'This closes budgets only for the declared new noisy Ramsey family and score pair. It does not replace the existing frozen three-qubit benchmark.',
        'source_sha256':{Path(__file__).resolve().relative_to(ROOT).as_posix():hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'experiments/circuit_validation/direct_tests.py':hashlib.sha256((ROOT/'experiments/circuit_validation/direct_tests.py').read_bytes()).hexdigest()}
    }
    out=Path(__file__).with_suffix('.json')
    with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'optional_penalty_results':rows},indent=2))


if __name__=='__main__':main()
