"""Independent theorem-to-code review of stable resource-controls v3.

All delivered transcripts below are prescribed, not sampled observations.
"""
from fractions import Fraction as F
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.sota_round7 import resource_controls as rc
from experiments.sota_round7.test_resource_controls import ControlChecks


def expect_error(operation,kind=(ValueError,RuntimeError)):
    try:operation()
    except kind:return
    raise AssertionError('Expected rejection')


def exact_crossing(plus,minus):
    assert not(plus>=200 and minus>=200)
    return 1 if plus>=200 else -1 if minus>=200 else 0


def main():
    source=ROOT/'experiments/sota_round7/resource_controls.py'
    expected='6e6e77a8babd41cdf5cb33d6bb33b0fb68cd8094e128900b2dbbc3219ea55d35'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    check_path=ROOT/'results/sota_round7/mechanisms/resource_control_checks_v3.json'
    saved=json.loads(check_path.read_text(encoding='utf-8'))
    for name,digest in saved['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    output=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ControlChecks)
    rerun=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    assert rerun.wasSuccessful() and rerun.testsRun==7,output.getvalue()
    count_checks=0
    for b in (F(1,2),F(3,250),F(7,500)):
        for negative in (0,1,17,101,4096):
            center=math.floor((math.log(200)-negative*math.log(float(1-b)))/math.log(float(1+b)))
            for positive in range(max(0,center-3),center+4):
                evidence=rc.TwoSidedBet(b)
                evidence.positive,evidence.negative,evidence.zero=positive,negative,37
                n,d=b.numerator,b.denominator
                numerator=(d+n)**positive*(d-n)**negative
                denominator=d**(positive+negative)
                reflected=(d+n)**negative*(d-n)**positive
                sign=1 if numerator>=200*denominator else -1 if reflected>=200*denominator else 0
                certificate=evidence.certificate()
                assert certificate['crossing_sign']==sign
                exact,method=evidence._crossed(positive,negative,rc.I(-100,100))
                assert exact==(numerator>=200*denominator) and method=='exact_integer'
                evidence.zero=0
                without=evidence.certificate()
                assert without['crossing_sign']==sign
                assert without['log_plus_interval']==certificate['log_plus_interval']
                count_checks+=1
    ramsey_prefixes=0
    trajectories=[]
    for depth,b in ((8,F(3,250)),(16,F(7,500))):
        for orientation in (-1,1):
            control=rc.RamseyFixedDepthControl(depth,seed=7123,cap=4096)
            plus=minus=F(1)
            last=None
            switches=0
            for index in range(4096):
                z=orientation*((-1 if index%2==0 else 1) if index<30 else 1)
                action=control.next_action()
                issued=control.summary()
                assert issued['actual_paid_shots']==index+1
                assert issued['full_cap_penalized_shots']==4096
                assert issued['sensing_pulse_uses']==depth*(index+1)
                assert len(control.ledger)==index+1 and control.ledger[-1]['status']=='issued'
                switches+=int(last!=action['phase_sign'])
                last=action['phase_sign']
                assert issued['phase_switches']==switches
                assert F(issued['device_time_exact'])==F((22+10*depth)*(index+1)+switches,10)
                expect_error(control.next_action)
                y=(1+action['phase_sign']*z)//2
                plus*=1+b*z
                minus*=1-b*z
                result=control.observe(action['action_id'],y)
                sign=exact_crossing(plus,minus)
                assert result['evidence']['crossing_sign']==sign
                assert result['terminal']==bool(sign)
                assert result['pending_action'] is None
                assert control.ledger[-1]['status']=='delivered'
                ramsey_prefixes+=1
                if sign:
                    assert result['decision']==('reject' if sign>0 else 'accept')
                    assert result['full_cap_penalized_shots']==index+1
                    before=control.summary()
                    expect_error(control.next_action)
                    expect_error(lambda:control.observe(action['action_id'],y))
                    expect_error(lambda:control.finalize_incomplete('cannot erase decided result'))
                    assert control.summary()==before
                    trajectories.append({'depth':depth,'orientation':orientation,'first_crossing_paid_shots':index+1,
                                         'decision':result['decision'],'ledger_replayed':True})
                    break
            else:raise AssertionError('Prescribed path should cross')
    bitstring_prefixes=0
    for orientation in (-1,1):
        control=rc.InvariantBitstringControl('same_duration_disjoint_invariant_spectators',cap=1000)
        plus=minus=F(1)
        counts={'old':[0]*8,'new':[0]*8}
        for pair_index in range(500):
            pair=((7,6),(4,5),(3,3),(0,0))[pair_index] if pair_index<4 else (7,6)
            if orientation<0:pair=tuple(reversed(pair))
            for arm,value in zip(('old','new'),pair):
                before=control.evidence.certificate()
                action=control.next_action()
                assert action['config']==arm and action['duration_exact']=='1'
                issued=control.summary()
                assert issued['actual_paid_shots']==len(control.ledger)
                assert issued['full_cap_penalized_shots']==1000
                assert issued['readout_bits_charged']==3*len(control.ledger)
                counts[arm][value]+=1
                result=control.observe(action['action_id'],value)
                assert result['complete_three_bit_counts']==counts
                assert result['control_window_uses']==len(control.ledger)
                assert result['configuration_switches']==len(control.ledger)
                assert F(result['device_time_exact'])==F(33*len(control.ledger),10)
                if arm=='old':
                    assert result['evidence']==before and not result['terminal']
                else:
                    z=(pair[0]&1)-(pair[1]&1)
                    plus*=1+F(1,2)*z
                    minus*=1-F(1,2)*z
                    sign=exact_crossing(plus,minus)
                    assert result['evidence']['crossing_sign']==sign
                    assert result['terminal']==bool(sign)
                    assert result['evidence']['positive']+result['evidence']['negative']+result['evidence']['zero']==pair_index+1
                bitstring_prefixes+=1
            if result['terminal']:
                assert result['decision']==('accept' if plus>=200 else 'reject')
                assert result['actual_paid_shots']%2==0
                expect_error(control.next_action)
                break
        else:raise AssertionError('Prescribed pair path should cross')
    default=rc.RamseyFixedDepthControl(8)
    assert isinstance(default.rng,random.SystemRandom)
    assert default.summary()['randomness_mode']=='system_random_bits'
    assert 'Full declared eligible domain' in default.summary()['power_scope']
    assert 'Custom cap' in rc.RamseyFixedDepthControl(8,cap=100).summary()['power_scope']
    assert 'no inherited probabilistic' in rc.RamseyFixedDepthControl(16,seed=9).summary()['power_scope']
    for bad in (8.0,True,'8',F(8),32):expect_error(lambda bad=bad:rc.RamseyFixedDepthControl(bad))
    for bad in (0,-2,True,3):expect_error(lambda bad=bad:rc.InvariantBitstringControl('same_duration_disjoint_invariant_spectators',cap=bad))
    for constructor in (lambda:rc.RamseyFixedDepthControl(8,seed=1,cap=100),
                        lambda:rc.InvariantBitstringControl('same_duration_disjoint_invariant_spectators',cap=100)):
        for stage in ('pending_failure','pending_interruption','between_actions_interruption'):
            control=constructor()
            assert control.summary()['full_cap_penalized_shots']==100
            action=control.next_action()
            expect_error(lambda:control.observe(action['action_id'],True))
            assert control.summary()['actual_paid_shots']==1
            assert control.summary()['full_cap_penalized_shots']==100
            if stage=='pending_failure':control.fail_pending(action['action_id'])
            elif stage=='pending_interruption':control.finalize_incomplete('prescribed interrupted action')
            else:
                control.observe(action['action_id'],0)
                control.finalize_incomplete('prescribed interruption between actions')
            result=control.summary()
            assert result['terminal'] and result['decision']=='abstain'
            assert result['actual_paid_shots']==1 and result['full_cap_penalized_shots']==100
            assert result['pending_action'] is None
            expect_error(control.next_action)
    archives=[]
    for version,check_name in ((1,'resource_control_checks.json'),(2,'resource_control_checks_v2.json')):
        archive=ROOT/('results/sota_round7/mechanisms/resource_controls_v%s_sources.zip'%version)
        with zipfile.ZipFile(str(archive)) as handle:
            names=handle.namelist()
            assert any(name.endswith('resource_controls.py') for name in names)
            assert any(name.endswith('test_resource_controls.py') for name in names)
        old=json.loads((ROOT/'results/sota_round7/mechanisms'/check_name).read_text())
        archives.append({'version':version,'source_archive':archive.relative_to(ROOT).as_posix(),
                         'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                         'retained_check_status':old['status']})
    assert archives[1]['retained_check_status']=='failed'
    result={
        'status':'passed_independent_theorem_to_code_review_stable_v3',
        'new_simulation_observations':0,
        'blocking_findings':[],
        'tested_source_sha256':expected,
        'deterministic_checks':{'original_groups_rerun':rerun.testsRun,'original_groups_passed':True,
            'additional_threshold_neighborhood_count_cases':count_checks,
            'exact_LR_prescribed_Ramsey_prefixes':ramsey_prefixes,
            'prescribed_complete_bitstring_prefixes':bitstring_prefixes,
            'Ramsey_trajectories':trajectories,
            'both_controls_failure_and_interruption_paths':6,
            'default_SystemRandom_and_seeded_replay_scope':True,
            'invalid_integer_depth_and_cap_rejected':True},
        'theorem_to_code_review':[
            {'item':'Paid action lifecycle','verdict':'passed',
             'reason':'Every successful next_action increments shots and appends one issued ledger row before returning the action. At most one action is pending. Duplicate, mismatched and malformed deliveries are rejected without creating observations or new charges; the original paid action remains pending. Execution failure and explicit interruption clear pending state, absorb the run and preserve all issued charges.'},
            {'item':'Unfinished-run penalties','verdict':'passed_after_fix',
             'reason':'Only decided runs use actual shots for the conservative score. Running, cap-abstaining, failed and interrupted snapshots all carry the full cap. finalize_incomplete records an explicit reason and absorbs an unfinished run, with or without a currently pending action. This closes the earlier inexpensive-unfinished-snapshot problem.'},
            {'item':'Exact likelihood-ratio crossing and stopping prefixes','verdict':'passed',
             'reason':'The state stores positive/negative increments and reconstructs exact rational products via directed logs. An undecidable log comparison falls back to the integer inequality(d+n)^positive(d-n)^negative>=200*d^(positive+negative), including equality. Zero increments multiply by1. The public controls test after each allowed evidence update and absorb at the first crossing; no later prefix can erase a decision.'},
            {'item':'Ramsey randomization and decision sign','verdict':'passed',
             'reason':'The default SystemRandom randrange(2) maps to a fair phase sign. Given independent phase coins and the declared observation law, z=S(2Y-1) has the demodulated Bernoulli law used in the proof and removes unknown readout bias. Positive evidence correctly rejects positiveDelta; negative evidence accepts. Fixed seeds are explicitly marked deterministic replay and no longer inherit probabilistic power claims. These tests prescribe outcomes and do not validate physical stochastic behavior empirically.'},
            {'item':'Ramsey nuisance and power scope','verdict':'passed_with_declared_contract',
             'reason':'The reviewed depth/bet pairs are exactly(8,3/250) and(16,7/500), threshold200, with the default2^20 cap. Uniform power applies only under the complete noisy-Ramsey model in noisy_multidepth_phase_obstruction.json: |phi|<=.02, gamma in[.06,.065], contrast>=.95 and the common readout model. A custom cap or seeded replay has no inherited95% certificate. The class consumes no hidden physical parameter.'},
            {'item':'Complete-pair bitstring statistic','verdict':'passed_with_declared_contract',
             'reason':'Only the explicitly named same-duration disjoint invariant-spectator contract is accepted. The documented s>0 assumption is essential. Old and new are issued alternately, the X0 statistic is formed from the low bit of each complete output, and evidence z=X_old-X_new is updated only after both separately charged outcomes arrive. Concordant pairs are paid and contribute factor1. Positive crossing correctly accepts. All three-bit counts are retained; the cap is even.'},
            {'item':'Bitstring power claim','verdict':'passed_scoped',
             'reason':'The implementation states power only for the reviewed xo=.03,xn=.01,s=.5 example at cap131072. It does not claim uniform eligible-domain power or robustness to unknown amplitude-dependent spectator crosstalk. The physical observation model must ensure the fresh old/new independence used in the theorem.'},
            {'item':'Gate and device ledgers','verdict':'passed',
             'reason':'Ramsey charges depth sensing-pulse uses on issuance and exactly22+10*depth+switch tenths of time, including the initial phase configuration. Full deterministic ledger replay matches G=depth*N andT=2.2N+G+.1switches, within the common16N/18.3N caps. Bitstring reports one shared control window, three readouts and an alternating configuration per issued action, withT=3.3N. Simultaneous q0/spectator Hamiltonian terms are not double-counted as serial windows.'}
        ],
        'resolved_findings':[
            'Floating depth8.0 previously passed the menu check and later broke exact time serialization; v3 requires an actual non-bool integer.',
            'Running snapshots previously used partial rather than full-cap conservative cost; v3 penalizes all non-decided snapshots and supports explicit interruption.',
            'Seeded replay previously retained the general power label; v3 removes inherited probabilistic power for every seeded replay.',
            'Bitstring device costs were previously only derivable from action records; v3 reports them explicitly.'
        ],
        'retained_history':archives,
        'nonblocking_limits':[
            'The classes are protocol state machines, not a physical-model validator or a security boundary against mutation of their public Python attributes.',
            'They assume the caller executes the issued action and supplies the registered physical outcome. Model violations, outcome-dependent delivery failures, or hardware control errors do not inherit the ideal-model power guarantee; failed runs must be included in the external assessment.',
            'The primary conservative shot score is implemented directly. The optional full-gate/full-time abstention penalties remain assessment formulas from the budget addendum, rather than separate fields of every runtime summary.'
        ],
        'source_sha256':{
            source.relative_to(ROOT).as_posix():expected,
            check_path.relative_to(ROOT).as_posix():hashlib.sha256(check_path.read_bytes()).hexdigest(),
            **saved['source_sha256'],
            'results/sota_round7/mechanisms/noisy_multidepth_phase_obstruction.json':hashlib.sha256((ROOT/'results/sota_round7/mechanisms/noisy_multidepth_phase_obstruction.json').read_bytes()).hexdigest(),
            'results/sota_round7/mechanisms/noisy_multidepth_phase_budget_closure.json':hashlib.sha256((ROOT/'results/sota_round7/mechanisms/noisy_multidepth_phase_budget_closure.json').read_bytes()).hexdigest(),
            'results/sota_round7/mechanisms/bitstring_invariance_cost.json':hashlib.sha256((ROOT/'results/sota_round7/mechanisms/bitstring_invariance_cost.json').read_bytes()).hexdigest(),
            Path(__file__).resolve().relative_to(ROOT).as_posix():hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'claim_boundary':'The reviewed v3 state machines implement the three classical controls under their stated observation contracts. Deterministic checks are not new Monte Carlo observations, operational device validation, or a SOTA performance result.'
    }
    out=Path(__file__).with_suffix('.json')
    with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
                      'status':result['status'],'checks':result['deterministic_checks']},indent=2))


if __name__=='__main__':main()
