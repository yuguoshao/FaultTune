"""Physical echo and midpoint readout identities with paid resource bounds.

No Monte Carlo. This records two public echo menus, exact cancellation under
registered ideal gate primitives, and classical same-action controls.
"""
from decimal import Decimal, ROUND_CEILING
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys

import mpmath as mp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.circuit_validation.direct_tests import _Bounds as I, _PI_LOW, _PI_HIGH


def iv(value):
    value=F(value)
    return I(value.numerator)/value.denominator


def bounds(value):
    return [str(value.lo),str(value.hi)]


def asin_sqrt(probability):
    """Positive arcsin series with an explicit geometric tail bound."""
    p=iv(probability)
    term=p.sqrt()
    total=term
    for n in range(512):
        following=term*p*iv(F((2*n+1)**2,2*(n+1)*(2*n+3)))
        if n==511:
            return I(total.lo,(total+following/(1-p)).hi)
        total=total+following
        term=following


def trig(value,cosine=False):
    """Interval Taylor polynomial and a derivative-bounded remainder."""
    value=I.as_bounds(value)
    if max(value.lo.copy_abs(),value.hi.copy_abs())>2:
        raise ValueError('Registered trig range is [-2,2].')
    term=I(1) if cosine else value
    total=term
    for n in range(1,45):
        denominator=(2*n-1)*(2*n) if cosine else (2*n)*(2*n+1)
        term=term*(-value*value)/denominator
        total=total+term
    n=45
    denominator=(2*n-1)*(2*n) if cosine else (2*n)*(2*n+1)
    following=term*(-value*value)/denominator
    remainder=max(following.lo.copy_abs(),following.hi.copy_abs())
    return I((total-I(remainder)).lo,(total+I(remainder)).hi)


def fair_walk_mean_bound(beta,threshold=200,cap=65536):
    b=iv(beta)
    drift=-(1-b*b).log()/2
    slope=((1+b)/(1-b)).log()/2
    barrier=I(threshold).log()
    ratio=(barrier+slope)/I(drift.lo)
    n0=int(ratio.hi.to_integral_value(rounding=ROUND_CEILING))
    assert (n0*drift-barrier).lo>=slope.hi
    a=drift/slope
    assert 0<a.lo<a.hi<1
    m=n0//2
    if n0%2==0:
        p0=F(math.comb(n0,m),2**n0)
        p1=F(0)
        abs_mean=n0*p0
    else:
        p0=F(0)
        p1=F(2*math.comb(n0,m),2**n0)
        abs_mean=F(n0*math.comb(2*m,m),2**(2*m))
    outside=1-p0-p1
    expected=I(n0)+iv(abs_mean-p1)/I(a.lo)+iv(outside)
    deviation=cap*drift-barrier
    assert deviation.lo>0
    tail=2*(-deviation*deviation/(2*cap*slope*slope)).exp()
    return {'boundary_mean_exact':str(beta),'threshold_each':threshold,'burnin_n0':n0,
            'log_drift_at_chi_zero':bounds(drift),'atanh_boundary_mean':bounds(slope),
            'expected_echo_shots_upper':bounds(expected),'audit_shot_cap':cap,
            'chi_zero_cap_noncertificate_probability_upper':bounds(tail)},expected


def mm(value):
    value=F(value)
    return mp.mpf(value.numerator)/value.denominator


def matrix_checks():
    mp.mp.dps=115
    uo=2*mp.asin(mp.sqrt(mm(F(5,198))))
    un=2*mp.asin(mp.sqrt(mm(F(1,198))))
    eye=mp.eye(4)
    x=mp.matrix([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])
    y=mp.matrix([[0,-1j,0,0],[1j,0,0,0],[0,0,0,-1j],[0,0,1j,0]])
    z1=mp.diag([1,-1,1,-1])
    z2=mp.diag([1,1,-1,-1])
    vector=mp.matrix([0,0,1/mp.sqrt(2),1/mp.sqrt(2)])
    rows=[]
    for label,lo,hi,k in [('original_amplitudes',un,uo,42),('expanded_amplitudes',mp.mpf(-1),mp.mpf(1),3)]:
        for chi,kappa,gamma,alpha0,alpha1 in [
                (mp.mpf(0),mp.mpf(0),mp.mpf('1.9'),mp.mpf('.005'),mp.mpf('.005')),
                (mp.mpf('-.1639587152826606203137657'),mp.mpf('.005'),mp.mpf('-.8'),mp.mpf('.03'),mp.mpf('.02')),
                (mp.mpf('.1'),mp.mpf('.002'),mp.mpf('4.7'),mp.mpf('.01'),mp.mpf('.04'))]:
            rho=vector*vector.transpose_conj()
            flip=(1-mp.exp(-kappa))/2
            for unused in range(k):
                for amplitude in (lo,hi):
                    unitary=mp.diag([1,1,1,mp.exp(-1j*(gamma+chi*amplitude))])
                    rho=unitary*rho*unitary.transpose_conj()
                    for z in (z1,z2):rho=(1-flip)*rho+flip*z*rho*z
                    rho=x*rho*x
            phase=k*(hi-lo)*chi
            attenuation=mp.exp(-2*kappa*k)
            expected=mp.matrix(4)
            expected[2,2]=expected[3,3]=mp.mpf('.5')
            expected[2,3]=attenuation*mp.exp(-1j*phase)/2
            expected[3,2]=mp.conj(expected[2,3])
            error=max(abs(rho[i,j]-expected[i,j]) for i in range(4) for j in range(4))
            assert error<mp.mpf('1e-100')
            visibility=1-alpha0-alpha1
            means={}
            for name,pauli in [('sin',y),('cos',x)]:
                probabilities=[]
                for sign in (-1,1):
                    projector=(eye+sign*pauli)/2
                    ideal=sum((rho*projector)[i,i] for i in range(4)).real
                    probabilities.append(alpha0+visibility*ideal)
                means[name]=probabilities[1]-probabilities[0]
            assert abs(means['sin']-visibility*attenuation*mp.sin(phase))<mp.mpf('1e-100')
            assert abs(means['cos']-visibility*attenuation*mp.cos(phase))<mp.mpf('1e-100')
            rows.append({'menu':label,'chi':str(chi),'kappa':str(kappa),'gamma':str(gamma),
                         'full_spectator_density_matrix_max_error':mp.nstr(error,25),
                         'demodulated_sin_and_cos_checked':True})
    midpoint=[]
    for omega in [mp.mpf('-.8'),mp.mpf(0),mp.mpf('.6')]:
        alpha0,alpha1=mp.mpf('.03'),mp.mpf('.02')
        v=1-alpha0-alpha1
        phi=omega+(uo+un)/2
        h=(uo-un)/2
        rx=lambda angle:mp.matrix([[mp.cos(angle/2),-1j*mp.sin(angle/2)],[-1j*mp.sin(angle/2),mp.cos(angle/2)]])
        initial=mp.matrix([1,0])
        rates=[]
        for angle in (omega+uo,omega+un):
            state=rx(angle)*initial
            rates.append(alpha0+v*abs(state[1])**2)
        outcomes=[]
        for sign in (-1,1):
            state=rx(sign*mp.pi/2)*rx(phi)*initial
            outcomes.append(alpha0+v*abs(state[1])**2)
        mu=outcomes[1]-outcomes[0]
        residual=abs(rates[0]-rates[1]-mp.sin(h)*mu)
        assert residual<mp.mpf('1e-100')
        midpoint.append({'omega':str(omega),'identity_residual':mp.nstr(residual,25)})
    return rows,midpoint


def main():
    uo,un=2*asin_sqrt(F(5,198)),2*asin_sqrt(F(1,198))
    gap=uo-un
    assert gap.hi<Decimal('.177')
    pi=I(_PI_LOW,_PI_HIGH)
    chi_max=F(1,5)
    chi_cut=F(11,531)
    kappa_max=F(1,200)
    v_min=F(9,10)
    menus=[]
    for name,k,gain,lower,upper,beta in [
        ('original_target_amplitudes',42,42*gap,'u_new','u_old',F(85,1000)),
        ('expanded_public_amplitudes',3,I(6),'-1','1',F(106,1000))]:
        windows=2*k
        pulse_count=2*k
        device=F(2*k)+F(2*k,10)+F(12,5)
        alias=gain*iv(chi_max)
        assert alias.hi<(pi/2).lo
        signal=trig(gain*iv(chi_cut))
        # Conservative bound allows dephasing during the full scheduled time.
        amin=iv(v_min)*(-iv(kappa_max*device)).exp()
        minimum_mean=amin*signal
        assert minimum_mean.lo>iv(beta).hi
        audit,mean=fair_walk_mean_bound(beta)
        menus.append({'name':name,'u_low':lower,'u_high':upper,'duration_each_window_exact':'1',
                      'blocks':k,'phase_gain_interval':bounds(gain),
                      'maximum_absolute_phase_over_declared_domain':bounds(alias),
                      'sensing_windows_per_echo_shot':windows,'X1_echo_pulses_per_shot':pulse_count,
                      'additional_X2_preparation_pulses':1,'measured_bits_per_shot':3,
                      'device_time_upper_per_shot_exact':str(device),
                      'minimum_contrast_using_full_scheduled_time':bounds(amin),
                      'minimum_sin_mean_at_abs_chi_cut':bounds(minimum_mean),
                      'classical_boundary_LR_control':audit,
                      'expected_sensing_windows_upper':bounds(windows*mean),
                      'expected_X1_echo_pulses_upper':bounds(pulse_count*mean),
                      'expected_device_time_upper':bounds(iv(device)*mean)})
    half_gap=gap/2
    multiplier=trig(half_gap)
    mu_cut=iv(F(3,500))/multiplier
    null_mean,alternative=F(7,100),F(1,5)
    assert mu_cut.hi<iv(null_mean).lo
    midpoint_phase=(uo+un)/2
    good_mu=iv(F(99,100))*trig(midpoint_phase)
    log_up=(iv(1+alternative)/iv(1+null_mean)).log()
    log_down=(iv(1-alternative)/iv(1-null_mean)).log()
    drift=(1+good_mu)*log_up/2+(1-good_mu)*log_down/2
    cost=(I(200).log()+log_up)/drift
    cap=4096
    deviation=cap*drift-I(200).log()
    tail=(-2*deviation*deviation/(cap*(log_up-log_down)*(log_up-log_down))).exp()
    assert drift.lo>0 and deviation.lo>0
    matrix,midpoint_checks=matrix_checks()
    result={
        'status':'complete_physical_derivation_and_deterministic_resource_bounds_pending_independent_review',
        'new_simulation_observations':0,
        'model':'H(u)=((omega+u)/2)X0+(gamma+chi*u)P11_(1,2), with the existing disjoint subsystem structure and stationary Markov Z-dephasing onq1,q2. Unknown omega,gamma,chi and readout nuisance are never supplied to a policy.',
        'echo_sequence':{
            'preparation':'Reset|000>, applyX2 to prepareq2=|1>, applyH1 to prepareq1=|+>. q0 can remain|0>; its later evolution factors and is not used by the echo statistic.',
            'one_block_in_time_order':['sensing windowH(u_low), durationtau','ideal registered X1 pulse','sensing windowH(u_high), durationtau','ideal registered X1 pulse'],
            'exact_operator':'Onq2=1, V_u=diag(1,exp(-i(gamma+chi*u)tau)) onq1. The block isX V_high X V_low=diag(exp(-i g_high tau),exp(-i g_low tau))=global_phase*Rz(chi*(u_high-u_low)*tau). The scalar global phase containsgamma and disappears from every measurement.',
            'repetition':'Afterk blocks, phase_phi=k*tau*(u_high-u_low)*chi. Both original amplitudes and the expanded amplitudes use actual paid sensing windows, not a free controlled inverse.',
            'dephasing':'D_Z commutes with each diagonal sensing window and is covariant underX becauseXZX=-Z. Thus sensing-window dephasing gives attenuationexp(-2*kappa*k*tau). Additional registered dephasing before analysis can be folded into a common attenuationexp(-kappa*T_eff).',
            'readout':'For a chosen axisP in{X1,Y1} and independently fair signS, measure outcome1 with projector(I+S*P)/2. If classical readout errors arealpha0,alpha1, thenZ=S(2Y-1) has mean A*cos(phi) forX1 or A*sin(phi) forY1, withA=(1-alpha0-alpha1)exp(-kappa*T_eff); the unknown additive bias cancels.',
            'q0_and_q2':'The q0 rotation during every sensing window is charged but cannot affectq1 under this Hamiltonian. q2 dephasing does nothing to its|1> population. Readout ofall3 bits is charged, with no postselection on the observed q2 bit.'},
        'necessary_implementation_assumptions':[
            'X1,X2 and the preparation/analysis gates are registered ideal unitary primitives, as in the existing gate/Kraus simulator. They have nonzero scheduled cost. The exact derivation does not assert that finite-width physicalX pulses automatically refocus an always-on Hamiltonian during their own drive.',
            'A hardware extension needs bounded residual evolution and pulse errors. With unknown unrestrictedgamma, imperfect echo pulses can leakgamma back into the measurement and cannot be absorbed into visibility without proof.',
            'The q2=1 preparation is a physical assumption, not inferred from noisy final readout. Ifq2 is actually1 with probabilityq, the means becomeqA sin(phi) andA[(1-q)+q cos(phi)]; unknown preparation leakage can mimic a smaller phase.',
            'Readout contrast is nonnegative and the same classical channel applies to both randomized signs and quadrature axes. Basis-dependent contrast or drifting nuisance between separately acquired axes requires an enlarged model or per-shot axis randomization.',
            'gamma,chi and the sensing response are stationary during both echo halves. Unmodeled gamma drift creates a residual phase instead of exact cancellation.',
            'A shared bound|chi|<=1/5 is explicit. Without an alias restriction, chi=2pi*m/gain produces the same echo sin/cos law aschi=0. A wider domain needs paid multidepth phase unwrapping shared with every comparator.'
        ],
        'paid_cost_contract':{
            'primary':'One complete prepared/measured echo is one charged circuit shot. Both quadratures, if used, require their own fresh paid shots.',
            'sensing_time':'Each ofthe2k Hamiltonian windows hasduration1 and costs one sensing-window time unit.',
            'ideal_gate_prices':'Each registeredX pulse is charged0.1 time unit, including2k X1 echoes andone X2 preparation. Setup/reset/readout and H/analysis layers use the existing2.2 convention. Up to0.1 is added per shot for analysis-phase configuration.',
            'echo_total':'T_echo<=2k+0.1*(2k+1)+2.2+0.1=2.2k+2.4. This is94.8 for original-menu k42 and9 for expanded-menu k3.',
            'noise_envelope_for_numeric_controls':'For the numerical single-quadrature controls, assumeunknownkappa in[0,.005] andalpha0+alpha1<=.1. The conservative visibility lower bound is.9*exp(-.005*T_echo), allowing dephasing throughout the full charged duration. The ideal sensing-only attenuation is less severe.',
            'same_action_comparison':'Every strong control receives exactly the same amplitude menu, durations, echo pulses, midpoint pulse, analysis phases and cost schedule. One cannot giveFaultTune +/-1 drives while restricting the comparator toold/new.'},
        'numeric_gate':{'chi_safe_cut_exact':str(chi_cut),'target_control_gap_interval':bounds(gap),
                        'gap_strict_upper':'177/1000','prior_abs_chi_max_exact':str(chi_max),
                        'why_not_original_0075':'The initial exploratory .075 cut is superseded. The current conservative guard usesh=11/531, compatible withd0>=11/2000 throughh=2d0/(3*.177).'},
        'echo_menu_comparison':menus,
        'classical_echo_controls':{
            'with_public_noise_envelope':'Choose rationalbeta<=A_min*sin(gain*h). For each demodulated sin outcomeZ, updateE_right*=1/(1+beta*Z),E_left*=1/(1-beta*Z). Underchi>=h, E_right is a supermartingale; underchi<=-h,E_left is one. Certify|chi|<h only when BOTH current products>=200, at or after the stated deterministicburnin. This intersection-union rule has false-certificate probability<=1/200 overthe entire declaredchi/nuisance domain, not twice that number.',
            'no_truth_assumption':'The executed rule never assumeschi=0. Cost and power below are evaluated atchi=0, wherefair demodulation makesZ exactlyRademacher regardless ofunknownreadout bias or attenuation.',
            'closed_cost_proof':'Write d=-log(1-beta^2)/2,l=atanh(beta),L=log200. ThenlogE_right/left=nd-/+l*S_n. Letn0=ceil((L+l)/d),a=d/l<1,so the simultaneous-crossing central band atn0 has radius>=1. Atchi=0, after n0, a surviving positive or negativefair walk hits a boundary moving inward relative toit atspeeda. Bounded stoppedWald gives extra mean at most(distance+1+a)/a. Averaging overS_n0 andusingbandradius>=1 yieldsE[tau]<=n0+[E|S_n0|-P(|S_n0|=1)]/a+P(|S_n0|>1). The retained central-binomial moments are exact rationals. AtcapN, P(no certificate)<=2exp(-(Nd-L)^2/(2N*l^2)).',
            'without_free_kappa_bound':'Use paid sin AND cos quadratures and letW_right=Z_y-t Z_x,W_left=-Z_y-t Z_x,t=tan(gain*h). Their expectations areA[+/-sin(phi)-t cos(phi)]. For0<b<1/(1+t), products of1-bW_right/left are valid against the respective outside-half-interval nulls. Certifying both gives a risk-valid interval without knowingA or kappa, butA approaching0 precludes a uniform finite-cost/power promise.',
            'drift_robust_axis_randomization':'Ifvisibility may vary predictably between shots while remaining common toboth potential axes ofthe current shot, independently randomizeaxisQ aswell asS and useW=2[+/-1(Q=Y)Z-t*1(Q=X)Z]. Then conditional mean isA_t[+/-sin(phi)-t cos(phi)], retaining the halfspace supermartingale witha correctly boundedbet. This still doesnot fix uncontrolled phase orpreparation drift.'},
        'midpoint_q0_mechanism':{
            'public_control':'u_mid=(u_old+u_new)/2, duration1, available equally to all methods.',
            'exact_target_identity':'Withtheta_j=omega+u_j,phi=omega+(u_old+u_new)/2,h0=(u_old-u_new)/2 andcommonreadoutcontrastv, d_q0=x_old-x_new=v*sin(phi)*sin(h0). This holds for arbitrary realomega, without a local phase oralias assumption.',
            'operator_and_sign':'Startq0=|0>, executeRx(phi), then the known analysisRx(S*pi/2), andZ-readout. Equivalentlyoutcome1 projectsonto(I-SY0)/2, since the state afterRx(phi) hasY expectation -sin(phi). WithfairS, Z0=S(2Y-1) hasmeanmu_d=v*sin(phi), so d_q0=sin(h0)*E[Z0]. The projector sign is opposite totheq1 Rz-echo Y convention.',
            'nuisance':'Unknownreadoutbias cancels andunknowncontrastv cancels fromthe identitybetween d_q0 andthe observablemean. gamma andchi act only onq1q2 and cannot affect thisq0 marginal under the declared tensor-factor Hamiltonian. Extraq0 decoherence oranalysis-gate errors require a new derivation.',
            'multiplier_sin_half_gap_interval':bounds(multiplier),
            'example_target_q0_gap_cut_exact':'3/500',
            'corresponding_observable_mean_cut_interval':bounds(mu_cut),
            'classical_LR_null_mean_exact':str(null_mean),'classical_LR_alternative_mean_exact':str(alternative),
            'good_world_observable_mean_interval':bounds(good_mu),
            'one_direction_log_drift_at_good_world':bounds(drift),
            'one_direction_expected_shots_upper':bounds(cost),
            'cap':cap,'good_world_correct_process_non_crossing_upper':bounds(tail),
            'classical_rule':'Forpositive d_q0 use likelihood factors(1+.2Z)/(1+.07Z) andthreshold200;the boundary .07 isrounded UP beyond(.006/sin(h0)). Under d_q0<=.006 the process is a supermartingale. Its sign-reflected version is available for negative d_q0. Bothcan run withfirst-crossing decisions; addingthe wrong-directionVille bound gives a finite-power bound. These areordinary fixed-alternative Bernoulli likelihood controls.',
            'cost':'Onefresh midpoint measurement isone paidshot,one duration1 sensingwindow,oneknown quarter-turnanalysis gate,fullreadout andconfiguration. A conservative separate analysis surcharge gives atmost3.4 time units per shot underthe stated2.2 setup/readout+.1 analysis+.1switch convention.'},
        'link_to_target_guard':{
            'structural_bound':'For the original target amplitudes at common duration, s_old>=1/4 and|s_new-s_old|<=(3/8)|chi|Delta_u in the declared spectator model. Thusd_q0>.006 and|chi|<11/531, together withDelta_u<.177, implyDelta_target<-1/8000. The negative-d_q0 mirror yields the corresponding harmful-update certificate with the analogous decomposition.',
            'not_an_invariance_assumption':'The echo evidence pays toboundchi. It doesnot assumethe good-worldchi=0, andit isnot assertingQ_old=Q_newexactly.',
            'integration_boundary':'Root/statistics controls supply the final combinedstage budget,risk bookkeeping andany performance comparison. These physical identities and component bounds do not by themselves certify a whole implementation or SOTA win.'},
        'deterministic_checks':{'mpmath_digits':mp.mp.dps,'spectator_matrix_sequences':matrix,
                                'midpoint_operator_checks':midpoint_checks,
                                'scope':'Independent explicit matrix/channel checks plus directed64-digit scalar bounds. No physical output law was sampled.'},
        'source_sha256':{Path(__file__).resolve().relative_to(ROOT).as_posix():hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'experiments/circuit_validation/quantum_model.py':hashlib.sha256((ROOT/'experiments/circuit_validation/quantum_model.py').read_bytes()).hexdigest(),
                         'experiments/circuit_validation/direct_tests.py':hashlib.sha256((ROOT/'experiments/circuit_validation/direct_tests.py').read_bytes()).hexdigest(),
                         'results/sota_round7/mechanisms/bitstring_invariance_cost.json':hashlib.sha256((ROOT/'results/sota_round7/mechanisms/bitstring_invariance_cost.json').read_bytes()).hexdigest()},
        'claim_boundary':'Echo cancellation, Ramsey phase readout and fixed likelihood controls are established mechanisms available tostrongsame-action baselines. This result expands the admissible information experiment but doesnot claim algorithmic novelty, hardware validation orsame-action SOTA superiority.'}
    out=Path(__file__).with_suffix('.json')
    with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
                      'menus':[{key:row[key] for key in ('name','phase_gain_interval','device_time_upper_per_shot_exact','minimum_sin_mean_at_abs_chi_cut','classical_boundary_LR_control')} for row in menus],
                      'midpoint_cost':bounds(cost),'midpoint_mu':bounds(good_mu)},indent=2))


if __name__=='__main__':main()
