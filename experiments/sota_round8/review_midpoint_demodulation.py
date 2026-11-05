"""Independent midpoint-demodulation identity, LR and physical-scope audit."""
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.circuit_validation import quantum_model as qm
from experiments.sota_round8.fixed_stage_guard import iv, enc


def measured_q0(model, amplitude, chi=0.0, analysis_sign=None, damping=0.0):
    """Explicit 3-qubit gates and Kraus branches, not the scalar identity."""
    state = qm._prepare(qm.WORKLOAD_CONTEXT)
    state = qm.apply_single_qubit(state, qm.rx_matrix(model.omega+amplitude), 0)
    state = qm.apply_controlled_phase(state, model.gamma+chi*amplitude)
    branches = (state,)
    phase = -math.expm1(-model.dephasing_rate)/2
    for bit in (1,2):
        branches = qm.apply_kraus_channel(branches, qm.dephasing_kraus(phase), bit)
    branches = tuple(qm._unprepare(s, qm.WORKLOAD_CONTEXT) for s in branches)
    if damping:
        operators = (((1.0,0.0),(0.0,math.sqrt(1-damping))),
                     ((0.0,math.sqrt(damping)),(0.0,0.0)))
        branches = qm.apply_kraus_channel(branches, operators, 0)
    if analysis_sign is not None:
        branches = tuple(qm.apply_single_qubit(s, qm.rx_matrix(analysis_sign*math.pi/2), 0)
                         for s in branches)
    for bit in (0,1,2):
        branches = qm.apply_kraus_channel(branches, qm.bit_flip_kraus(model.readout_flip), bit)
    probabilities = [math.fsum(abs(s[index])**2 for s in branches) for index in range(8)]
    assert abs(math.fsum(probabilities)-1) < 2e-13
    return math.fsum(probabilities[1::2])


def main():
    uo, un = 2*math.asin(math.sqrt(5/198)), 2*math.asin(math.sqrt(1/198))
    umid, half_gap = (uo+un)/2, (uo-un)/2
    sin_half_gap = (iv(985).sqrt()-iv(193).sqrt())/198
    assert sin_half_gap.lo > Decimal('.0883')
    assert sin_half_gap.lo < Decimal('.0884')
    errors = []
    for eta in (0.0, .005, .05):
        for kappa in (0.0, .005, .4):
            for chi in (0.0, .13, -.164):
                for omega in (0.0, .3, -1.0):
                    model = qm.QuantumModel(omega, .7, eta, kappa)
                    old, new = [measured_q0(model, u, chi) for u in (uo, un)]
                    mu = sum(sign*(2*measured_q0(model, umid, chi, sign)-1)/2
                             for sign in (-1,1))
                    err = abs(old-new-math.sin(half_gap)*mu)
                    assert err < 2e-13
                    errors.append(err)

    # Genuine common-channel counterexample: amplitude damping before the
    # final analysis rotation has unequal Y and Z contraction factors.
    gamma_ad = .75
    model = qm.QuantumModel(0.0, .7, .005, .1)
    old, new = [measured_q0(model, u, .13, damping=gamma_ad) for u in (uo,un)]
    mu_ad = sum(sign*(2*measured_q0(model,umid,.13,sign,gamma_ad)-1)/2 for sign in (-1,1))
    assert abs((old-new)-math.sqrt(1-gamma_ad)*math.sin(half_gap)*mu_ad) < 2e-13
    assert abs((old-new)-math.sin(half_gap)*mu_ad) > .004

    d0, sin_lower, mu1 = F(3,500), F(883,10000), F(1,5)
    cut = d0/sin_lower
    assert cut == F(60,883)
    factors = {z:(1+mu1*z)/(1+cut*z) for z in (-1,1)}
    logs = {z:iv(f).log() for z,f in factors.items()}
    good_mu = iv(F(1,50))/sin_half_gap
    drift = ((1+good_mu)*logs[1]+(1-good_mu)*logs[-1])/2
    assert drift.lo > 0
    expected_shot_upper = (iv(200).log()+logs[1])/iv(F(drift.lo))
    # The analytical supremum identity certifies the whole mean-null, not a
    # grid of tested probabilities: E_mu[f]=1+(mu1-cut)*(mu-cut)/(1-cut^2).
    for mu in (F(-1), F(-1,3), F(0), cut, F(1)):
        exact = sum((1+mu*z)*factors[z]/2 for z in (-1,1))
        formula = 1+(mu1-cut)*(mu-cut)/(1-cut*cut)
        assert exact == formula

    result = {
        'schema': 'faulttune.round8.midpoint_demodulation_independent_review.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'identity_and_LR_pass_in_declared_model_generic_Markov_extension_rejected',
        'exact_identity': {
            'definition': 'u_mid=(u_old+u_new)/2, h=(u_old-u_new)/2>0, d=x_old-x_new.',
            'population_law': 'x(u)=r+v*sin^2((omega+u)/2).',
            'trigonometric_difference': 'd=v*sin(omega+u_mid)*sin(h).',
            'paid_probe': 'Run the midpoint duration1 window, then an ideal registered Rx(S*pi/2) analysis primitive on q0 with an independent fair public sign S. After readout Y in{0,1}, record Z=S*(2Y-1).',
            'demodulation_law': 'For the existing readout model v=1-2r, E[Z]=v*sin(omega+u_mid). More generally a shared affine classical readout law r+v*p gives the same demodulated mean, with its offset canceled by S.',
            'sign': 'Using Rx(+pi/2) after Rx(theta), P(Y=1)=(1+sin theta)/2 before readout, so the displayed Z convention has the required positive sign.',
            'exact_sin_h': '(sqrt(985)-sqrt(193))/198',
            'sin_h_interval': enc(sin_half_gap),
            'certified_rational_lower': str(sin_lower),
            'applicability': 'The old three-qubit model has no q0 noise except readout. Its spectator dephasing and the added (gamma+chi*u)P11 term act on a disjoint tensor factor, so they cannot alter this identity.',
        },
        'LR_control': {
            'd0': str(d0), 'mean_cut': str(cut), 'fixed_alternative_mean': str(mu1),
            'factor_minus': str(factors[-1]), 'factor_plus': str(factors[1]),
            'null_mean_identity': 'E_mu[f(Z)]=1+(mu1-cut)*(mu-cut)/(1-cut^2)<=1 for mu<=cut.',
            'target_null_implication': 'd<=d0 implies mu=d/sin(h)<=d0/sin(h)<=d0/.0883=cut. Therefore the likelihood product is a valid nonnegative supermartingale for d<=d0.',
            'risk': 'Crossing200 has probability<=.005 on the d<=d0 null. The mirrored process controls d>=-d0. These can replace the paired target processes in the existing fixed-parameter partition proof with the same echo gate.',
            'good_world_mean_interval': enc(good_mu), 'good_world_log_drift_interval': enc(drift),
            'good_world_necessary_scope': 'Old/new observed q0 rates.03/.01 and no extra q0 channel; hidden truth appears only in this diagnostic cost calculation.',
            'good_world_expected_midpoint_shots_upper': str(expected_shot_upper.hi),
            'cost_proof': 'For log likelihood S_n, positive step overshoot is at most log(f_plus). Applying bounded stopped Wald to T wedge n gives drift*E[T wedge n]<=log200+log(f_plus). Monotone convergence yields the expected-shot bound.',
        },
        'independent_matrix_checks': {'physical_parameter_points':len(errors),
                                     'three_qubit_gate_Kraus_runs':4*len(errors),
                                     'identity_max_absolute_error':max(errors)},
        'common_Markov_channel_counterexample': {
            'channel': 'Amplitude damping of strength g=3/4 after the sensing Rx and before analysis/readout, identical in every arm.',
            'population_contrast': 'v_Z=(1-2r)*(1-g)',
            'demodulation_contrast': 'v_Y=(1-2r)*sqrt(1-g)',
            'actual_identity': 'd=sqrt(1-g)*sin(h)*mu_d; for g=3/4 this is half the proposed generic-Markov value.',
            'matrix_d': old-new, 'matrix_demod_mean':mu_ad,
            'incorrect_generic_formula_value':math.sin(half_gap)*mu_ad,
            'conclusion': 'A shared unspecified q0 Markov channel is insufficient. The scalar population law alone also does not identify the orthogonal measurement contrast.',
        },
        'required_scope': [
            'Use the existing no-extra-q0-noise model, or a proved channel class with the required equal quadrature transfer and calibrated offset. Generic Markov noise is not covered.',
            'Final basis rotations are declared known ideal primitives with explicit cost. An uncalibrated finite pulse with unknown drift/overrotation requires a new error envelope.',
            'The midpoint amplitude, analysis bases and phase cycling are new public actions shared with every strong comparator. The result is measurement-menu improvement, not same-action SOTA superiority.',
            'All midpoint and echo shots are independently paid. Extra basis/control settings, resets and analysis primitive costs must enter the same ledger.',
            'The echo prior, contrast lower bound, spectator preparation and risk assumptions remain necessary for the complete target-sign guard.',
        ],
        'new_monte_carlo_observations':0,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = ROOT/'results/sota_round8/midpoint_demodulation_independent_review.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status':result['status'], 'sin_h_interval':enc(sin_half_gap),
                     'good_mu_interval':enc(good_mu),
                     'midpoint_shot_upper':str(expected_shot_upper.hi),
                     'gate_checks':4*len(errors), 'max_error':max(errors)}, indent=2))


if __name__ == '__main__':
    main()
