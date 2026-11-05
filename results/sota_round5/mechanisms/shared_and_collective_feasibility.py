"""Deterministic mechanism screening: shared linear evidence and collective data.

No simulated observations and no SDP solves are performed. Exact rational
checks and outward formula evaluations support the stated screening results.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import mpmath as mp

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/circuit_validation'))
from direct_tests import _Bounds
mp.mp.dps=160


def iv(x):
    x=Fraction(x)
    return _Bounds(x.numerator)/x.denominator


def phi(x):
    return ((1+x)*(1+x).log()+(1-x)*(1-x).log())/2


def describe(x,reference=None):
    result={'lower':str(x.lo),'upper':str(x.hi)}
    if reference is not None:
        assert mp.mpf(str(x.lo))<=reference<=mp.mpf(str(x.hi))
        result.update(mpmath160_inside=True,mpmath160_diagnostic=mp.nstr(reference,110))
    return result


def ceil_exact_decimal(x):
    x=Fraction(x)
    return -((-x.numerator)//x.denominator)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(name,value):
    path=Path(__file__).with_name(name)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')
    return {'path':str(path),'sha256':sha(path)}


def main():
    kappa=iv(Fraction(19,20))*_Bounds(95).log()+iv(Fraction(1,20))*iv(Fraction(5,99)).log()
    # Exact two-extreme-ray decomposition for1001 distinct update directions.
    cone_checks=0
    for j in range(1001):
        t=Fraction(j,1000)-Fraction(1,2)
        wminus,wplus=Fraction(1,2)-t,Fraction(1,2)+t
        assert min(wminus,wplus)>=0 and wminus+wplus==1
        assert -wminus/2+wplus/2==t
        cone_checks+=1
    # A concrete shared fixed-time confidence rectangle independent of M.
    R=Fraction(1,4)
    e=R/8
    nupper=(_Bounds(800).log()/(2*iv(e*e))).hi
    per_arm=ceil_exact_decimal(nupper)
    shared={
        'status':'mechanism_reduced_to_classical_two_dimensional_linear_identification',
        'new_simulation_observations':0,
        'root_implementation_recommendation':'Use shared confidence regions as a correct engineering feature; do not pursue it as the novel SOTA mechanism without an additional task-specific advantage over equally shared structured confidence-sequence/pure-exploration controls.',
        'deterministic_problem':{
            'model':'One static p_theta(x)=A+B cos(x)+C sin(x); every paid observation and every proposed update refers to this SAME function.',
            'update':'For old x_j and candidate y_j, improvement is g_j(theta)=p(x_j)-p(y_j)=d_j dot(B,C), where d_j=(cos x_j-cos y_j,sin x_j-sin y_j). A cancels.',
            'global_safety':'Over the entire finite or adaptively proposed sequence, probability of ever approving a g_j<=0 update must be<=.01.',
            'success':'For a deterministic finite set with strictly positive improvements, require all approved with probability>=.95, or separately each approved with probability>=.95 for the information lower bound.',
            'shared_certificate':'Use one simultaneous anytime confidence region S_t for(A,B,C). Approve any proposed update only if inf_theta in S_t g_j(theta)>0. The single event theta_true in S_t for all t proves the whole-sequence error bound, even for updates selected from past data. No per-update error-budget restart is necessary.'},
        'sharp_geometry':{
            'lemma':'If a finite set of nonzero update directions d_j admits some lambda with every d_j dot lambda>0, its generated cone lies in an open half-plane and has at most two extreme rays. Every d_j is a nonnegative linear combination of these extremes. Certifying the two extreme linear inequalities certifies all M signs.',
            'proof':'Rotate coordinates so lambda points horizontally. Every d_j has positive first coordinate. Normalize each d_j by that coordinate; each becomes(1,t_j). Every t_j lies between the minimum and maximum slopes, so(1,t_j) is a convex combination of those two normalized extremes. Restore the positive normalizations.',
            'phase_form':'Writing(B,C)=R(cos phi,sin phi), each true-positive sign restricts phi to an open semicircle. Their jointly feasible intersection is an angular interval with at most two active endpoints. Radius-zero models must remain in the confidence region until excluded; an unqualified phase interval is not a valid substitute.',
            'exact_checks':cone_checks,
            'exceptions':'This two-ray reduction is for signs of homogeneous linear forms. Different required absolute improvement thresholds produce affine half-spaces with potentially many active facets. Random future directions also cannot be selected as the two extremes before they are proposed.'},
        'concrete_chain':{
            'construction':'Let p(x)=1/2+R cos x and x_j=pi/3+j*pi/(3M), j=0,...,M. Every consecutive update decreases failure probability. Normalized chord directions are(sin midpoint,-cos midpoint), hence lie within angles[-pi/6,pi/6] around(B,C)=(R,0). Two extreme chords certify the entire chain.',
            'shared_four_angle_test':'Sample each of0,pi,+pi/2,-pi/2 exactly n times. Estimate B and C by half-differences of the corresponding means. Hoeffding plus a four-arm union bound gives simultaneous mean errors<=e with probability>=.99 when n>=log800/(2e^2). On this event the coefficient confidence rectangle of radius e around the estimates has B>=R-2e and |C|<=2e.',
            'choice':'Take e=R/8. Every normalized chain direction has dot product at least3 sqrt(3) R/8-R/8>0 throughout the rectangle. Thus all M updates are certified with probability>=.99, at a fixed cost independent of M.',
            'example_R_exact':'1/4','per_arm_n_sufficient':per_arm,'total_preparations_sufficient':4*per_arm,
            'comparison':'A reset endpoint-comparison strategy can be much worse as each short chord has a small raw probability difference. A same-information structured linear confidence region obtains the same M-independent cost immediately. The reuse gain is real but classical.'},
        'information_bound':{
            'null_union':'For fixed directions D, let H_j={eta physical:d_j dot(B_eta,C_eta)<=0}. For any valid policy and any eta in H_j, data processing for the event approving j gives transcript KL>=kappa=kl(.95,.01).',
            'characteristic_time':'If w is the expected paid-angle occupation measure normalized by expected total cost, chain rule gives E[N] integral kl(p_theta(x)||p_eta(x)) w(dx)>=kappa for every eta in union H_j. Thus E[N]>=kappa/I*(theta,D), where I*=sup_w inf_eta in union H_j integral kl(p_theta||p_eta)dw. This is the standard controlled-sensing/linear-identification characteristic-time structure, not a new theorem family.',
            'scope':'This fixed-D information lower bound does not condition on a randomly selected set of successful updates.'},
        'no_uniform_free_sequence_counterexample':{
            'construction':'For any0<epsilon<=1 use theta=(A,B,C)=(1/2,1/4,0) and direction d=(epsilon,1). The improvement is epsilon/4>0. The physical null eta=(1/2,1/4,-epsilon/4) has d dot(B_eta,C_eta)=0. Positive scaling realizes d as a circle chord direction without changing its sign.',
            'per_shot_bound':'For every query angle, |p_theta-p_eta|<=epsilon/4 and q_eta(1-q_eta)>=1/8. The Bernoulli bound kl(p||q)<= (p-q)^2/[q(1-q)] gives per-shot KL<=epsilon^2/2.',
            'cost_obstruction':'Any95%-successful globally safe certificate therefore needs E[N]>=2*kappa/epsilon^2, even with arbitrary adaptive angles and all old data charged. Sending a candidate direction toward the zero-improvement boundary prevents any uniform finite-cost guarantee.',
            'trajectory_warning':'If a chain approaches an unknown optimum, the minimum angular margin can shrink as the number of steps grows; the fixed-margin reuse example does not remove this cost.'},
        'fair_comparators':['A shared structured Bernoulli likelihood confidence sequence over(A,B,C).',
                            'A shared linear/generalized-linear pure-exploration design targeting the two active update constraints.',
                            'A fixed-design shared simultaneous confidence region when active design overhead is not justified.'],
        'novelty_boundary':'Shared confidence sets, confidence sequences, phase inference and fixed-confidence linear identification are established ideas. This reduction alone supplies no basis for an international-leading claim.'}
    # Full same-information local baseline: active flag AND arbitrary one-copy POVMs.
    fstar=(59+_Bounds(1881).sqrt())/1000
    chi=-fstar.log()
    # For0<t<=1/4, beta>=24/25; s<=425/1568*t^2;
    # -log(1-s)<=10625/38416*t^2.
    L=Fraction(10625,38416)
    fixed_block_speedup_bound=2*iv(L)*(_Bounds(200).log()+iv(Fraction(1,4)))/chi
    capped_filter_speedup_bound=fixed_block_speedup_bound/iv(Fraction(19,20))
    assert capped_filter_speedup_bound.hi<iv(Fraction(3,2)).lo
    quantum_rows=[]
    for denominator in (4,8,16,32,64,128):
        t=Fraction(1,denominator)
        beta2=1-t*t-t**4
        r2=1-t**4
        beta,r=iv(beta2).sqrt(),iv(r2).sqrt()
        local=phi(iv(t))
        quantum=phi(r)-phi(beta)
        active_zero_probability=(1+beta)/2
        block_lower=chi/(-active_zero_probability.log())
        local_mean=(_Bounds(200).log()+(1+iv(t)).log())/local
        t_mp=mp.mpf(1)/denominator
        b_mp=mp.sqrt(1-t_mp**2-t_mp**4)
        r_mp=mp.sqrt(1-t_mp**4)
        ph=lambda x:((1+x)*mp.log(1+x)+(1-x)*mp.log(1-x))/2
        q_mp=ph(r_mp)-ph(b_mp)
        l_mp=ph(t_mp)
        f_mp=(59+mp.sqrt(1881))/1000
        n_mp=-mp.log(f_mp)/(-mp.log((1+b_mp)/2))
        lower_formula=(1-iv(t*t))*(iv(Fraction(1,2))*iv(4/(t*t)).log()
                       -iv(2*t*t/3)*(3*_Bounds(1).exp()/iv(t**4)).log())
        quantum_rows.append({
            't_exact':str(t),'strictly_mixed_beta_squared_exact':str(beta2),
            'strictly_mixed_radius_squared_exact':str(r2),
            'local_information_per_active_copy':describe(local,l_mp),
            'quantum_relative_entropy_per_active_copy':describe(quantum,q_mp),
            'quantum_over_local_information_ratio':describe(quantum/local,q_mp/l_mp),
            'rigorous_analytic_information_ratio_lower':describe(lower_formula),
            'fixed_active_block_copies_necessary_fidelity_bound':describe(block_lower,n_mp),
            'necessary_minimum_integer_active_block':ceil_exact_decimal(block_lower.lo),
            'strong_local_sprt_expected_active_copies_upper':str(local_mean.hi),
            'strong_local_sprt_sufficient_finite_active_cap':ceil_exact_decimal((local_mean/iv(Fraction(9,200))).hi),
            'fixed_block_maximum_speedup_over_this_feasible_local_baseline_upper':str((local_mean/block_lower).hi),
            'active_filter_with_cap_abstention_maximum_speedup_upper':str((local_mean/(iv(Fraction(19,20))*block_lower)).hi)})
    quantum={
        'status':'explicit_quantum_mechanism_with_strong_local_baseline_and_fixed_block_rejection',
        'new_simulation_observations':0,'sdp_solved':False,
        'root_implementation_recommendation':'Do NOT launch a fixed-block collective SDP campaign for a3x claim on this near-pure family: the fidelity lower bound plus an explicit fair local SPRT proves a speedup ceiling below1.5. The SDP formulation remains a reusable exact feasibility diagnostic. A genuinely sequential coherent-instrument mechanism is a separate, unresolved problem; the relative-entropy gap does not establish its benefit.',
        'physical_model':{
            'density_matrix':'rho(A,B,C)=diag-block(1-2A, [[A+B,C],[C,A-B]]),0<A<=1/2,B^2+C^2<=A^2.',
            'original_query':'P_x=0_vacuum direct-sum (I+cos(x)Z+sin(x)X)/2. Then Tr rho P_x=A+B cos x+C sin x exactly.',
            'extra_joint_resource':'A nondestructive measurement of the active/vacuum projector, storage of multiple surviving active qubits, and a joint POVM on them. The active conditional state is tau=(I+beta Z+gamma X)/2 with beta=B/A,gamma=C/A.',
            'strong_local_control':'Give the comparator the SAME nondestructive flag, every possible single-copy POVM, and arbitrary classical adaptation from all earlier results. It may use all outcome categories. It does not have coherent storage across preparations or entangling measurements between copies.',
            'resource_accounting':'Count every source preparation, including discarded vacuum events and abstaining runs. Separately report stored-qubit count/lifetime, two-qubit gates, ancillas, measurement depth and error. A preparation reduction bought by unbounded coherent memory or joint gates is not a same-resource SOTA win.'},
        'arbitrary_single_copy_povm_information_theorem':{
            'normalized_null':'For0<|gamma|<1 let lambda=atanh(gamma)/gamma and sigma=(I+beta lambda Z)/2. Physicality implies |beta lambda|<1. Set H*=I+gamma X. The prior symmetric proof gives D log(H*)[tau]=sigma, Tr sigma H*=1, and Tr tau log H*=Phi(gamma).',
            'all_positive_H':'Concavity of Tr tau log H gives Tr tau log H<=Phi(gamma)+Tr sigma(H-H*).',
            'povm_extension':'For any POVM{E_y}, let p_y=Tr tau E_y,q_y=Tr sigma E_y,h_y=p_y/q_y and H=sum h_y E_y. Operator Jensen gives log H >=sum(log h_y) E_y, so KL(p||q)<=Tr tau log H. Also Tr sigma H=sum q_y h_y=1. Hence KL(p||q)<=Phi(gamma). Zero p_y are handled by positive regularization; full-rank sigma makes q_y=0 imply E_y=0.',
            'flag_extension':'Use one qutrit null omega=(1-2A)|vac><vac| direct-sum2A sigma. Both states are block diagonal in the flag. Adding the flag before any POVM does not disturb either state and can only increase classical KL. Conditional vacuum outputs agree; active output KL<=Phi(gamma). Thus EVERY one-copy POVM, including flag access, has KL<=2A Phi(C/A)=U(A,C) against this same physical null.',
            'adaptive_local_policies':'The same fixed null works at every history. Transcript chain rule yields KL<=U E[N] for arbitrary local POVMs with classical adaptation and finite stopping. Within-copy feed-forward is part of a POVM; coherent cross-copy memory is deliberately outside this theorem.',
            'tight_oracle_local_rate':'Flag plus an X-basis measurement of the active qubit gives probabilities(1-2A,A+C,A-C). Its KL to the best C<=0 physical null is exactly2A Phi(C/A), so this local oracle information rate is U. The proposed collective information gap cannot be attributed to withholding the flag.'},
        'collective_quantum_relative_entropy':{
            'projection':'The minimizer of D(rho||eta) over all physical real qutrit eta with C_eta<=0, for C>0, is diag(rho). Thus I_Q=D(rho||diag rho)=2A[Phi(sqrt(B^2+C^2)/A)-Phi(B/A)].',
            'proof':'At eta=diag rho, diagonal perturbations have derivative minus their trace, hence zero under normalization. The off-diagonal derivative is -2C*(log(A+B)-log(A-B))/(2B), with its continuous positive divided-difference value when B=0. For a feasible C_eta<=0 perturbation its contribution is nonnegative. Convexity of D(rho||eta) in eta proves the global minimum. Strictly positive diagonal entries suffice; boundary cases follow by limits.',
            'interpretation':'Quantum relative entropy is an upper bound on measured block KL and supports asymptotic quantum testing/coherence questions. Its size does NOT prove any finite95%-power,1%-risk copy cost, any optimal composite-null test, or an implementable low-cost joint circuit. This artifact makes no finite achievability or novelty claim from I_Q.'},
        'strict_near_pure_family':{
            'definition':'Fix0<A<=1/2 and0<t<=1/4. Let C=A t, B=A sqrt(1-t^2-t^4). The active Bloch radius is sqrt(1-t^4)<1, so the family is strictly physical and strictly mixed, not just a rank-one boundary example.',
            'local_rate':'I_local=2A Phi(t)=A t^2+O(A t^4).',
            'quantum_rate':'I_Q=A t^2[log(1/t)+log2+1/2]+O(A t^4 log(1/t)). Hence I_Q/I_local=log(1/t)+log2+1/2+O(t^2 log(1/t)), which diverges.',
            'derivation':'Write s_B=(1-sqrt(1-t^2-t^4))/2=t^2/4+O(t^4), s_R=(1-sqrt(1-t^4))/2=t^4/4+O(t^8). Use I_Q=2A[h(s_B)-h(s_R)] and h(s)=s(log(1/s)+1)+O(s^2).',
            'strict_lower_bound':'For0<t<=1/4, I_Q/I_local >=(1-t^2)[(1/2)log(4/t^2)-(2t^2/3)log(3e/t^4)]. This follows from s_B>=t^2/4,s_R<=t^4/3, h(s)>=s log(1/s), h(s)<=s log(e/s), and Phi(t)<=t^2/[2(1-t^2)].',
            'not_uniform_finite_error':'The diverging information ratio concerns relative entropy as the state family changes. A Stein-type small-error asymptotic at fixed states cannot be exchanged with t->0 while risk and power stay fixed.'},
        'finite_block_fidelity_dual_obstruction':{
            'legal_null':'The active pure null |0><0| is allowed by the same C<=0 family. For k active target copies, squared fidelity to |0>^tensor k equals a^k, where a=(1+beta)/2=1-s_B; this is true also for the strictly mixed target.',
            'data_processing':'For a report event having target probability>=.95 and null probability<=.01, squared classical fidelity is at most F*=(sqrt(.95*.01)+sqrt(.05*.99))^2=(59+sqrt1881)/1000. Monotonicity under measurement gives a^k<=F*. Therefore k>=chi/[-log a],chi=-log F*.',
            'Fstar':describe(fstar),'chi':describe(chi),
            'near_pure_limit':'The necessary fixed-block k is asymptotic to chi/s_B. By comparison the local KL floor kappa/Phi(t) is asymptotic to kappa/(2s_B). The fidelity constant is larger, so the log(1/t) relative-entropy gain does not yield a fixed-block3x gain at these errors.',
            'duality_use':'Before building any SDP at k, reject infeasible k analytically whenever a^k>F*. This uses a concrete allowed null and data processing, independent of discretizing the nuisance parameter.'},
        'explicit_fair_local_sprt':{
            'design_information':'The effect design value t is fixed before observations, just as a reference target would be fixed in the proposed SDP. Both procedures get the same t; neither reads simulator truth during operation. The test remains risk-valid for every physical state, not only the reference family.',
            'observations':'Measure the flag; for each active qubit measure X. Under the reference family the positive outcome probability is q=(1+t)/2, independent of beta.',
            'rule':'Maintain L+_n=(1+t)^S(1-t)^(n-S) and L-_n=(1-t)^S(1+t)^(n-S). Report a sign on its first crossing of200. At C=0 the union Ville bound is<=2/200=.01; for either nonzero sign the wrong LR is a supermartingale, giving wrong-report probability<=1/200.',
            'expected_cost_bound':'Under the positive reference, the log L+ increments have mean Phi(t) and positive overshoot<=log(1+t). The finite stopped-Wald identity at T+ wedge n and bounded overshoot imply E[T+]<=[log200+log(1+t)]/Phi(t). Stopping at either sign only lowers cost. No Monte Carlo or successful-path conditioning is involved.',
            'finite_cap':'Choose an active cap M>=200/9 times that mean upper bound. Markov gives cap abstention<=.045, so correct probability>=1-.005-.045=.95. A physical-preparation cap uses the same formula with the mean divided by2A. The resulting capped expected cost is no greater than the uncapped bound.',
            'fixed_block_comparison':'For A=.5 a fixed-k joint measurement pays k preparations. For general A and a physical cap, reaching k active copies is necessary for a correct report, so P(reach)>=.95; Wald for flag counts yields E[N]>=.95*k/(2A). The local feasible mean is bounded above by the displayed active mean/(2A).'},
        'uniform_fixed_block_rejection_for_whole_family':{
            'algebra':'For0<t<=1/4, beta>=24/25, a>=49/50, s_B<=425*t^2/1568, and -log a<=10625*t^2/38416. Also Phi(t)>=t^2/2 and log(1+t)<=1/4. Therefore local_mean/fidelity_k_lower <=2*(10625/38416)*(log200+1/4)/chi.',
            'fixed_active_block_speedup_upper':describe(fixed_block_speedup_bound),
            'including_flag_waiting_and_cap_abstention_speedup_upper':describe(capped_filter_speedup_bound),
            'strictly_below_three_halves':True,
            'meaning':'Against this explicit qualified local comparator, no candidate that first collects a fixed number of active qubits then makes one collective decision can deliver3x on this family; even1.5x is excluded by these conservative bounds. This does not cover genuinely sequential coherent instruments or different physical promises.'},
        'deterministic_sdp_formulation':{
            'fixed_k_variables':'Real symmetric effects E+,E- on k active qubits, with E+>=0,E->=0,E++E-<=I. Optional sign symmetry E-=Z^tensor k E+ Z^tensor k and permutation symmetry are valid reductions.',
            'strict_wrong_sign_constraints':'For EVERY(beta,gamma) with beta^2+gamma^2<=1 and gamma<=0 impose Tr[E+ tau(beta,gamma)^tensor k]<=.01. The negative-side counterpart follows by sign symmetry. At gamma=0 impose the stronger combined-null constraint Tr[(E++E-)tau(beta,0)^tensor k]<=.01.',
            'power_constraint':'For a predeclared positive engineering reference tau0 impose Tr[E+ tau0^tensor k]>=.95, or maximize this power subject to the error constraints. Mirrored power follows by symmetry.',
            'physical_cap_cost':'With fixed active block k and preparation capM, success probability equals P(Bin(M,2A)>=k) times conditional block power. Expected preparations are E[min(NB(k,2A),M)], using the already audited exact/outward waiting identity. Charge joint gates and memory separately.',
            'search_only':'A finite nuisance-state grid may propose a POVM but cannot certify continuum error control. Symmetrizing permutation-invariant iid states can reduce variables via Schur-Weyl spin blocks; no numeric feasibility is inferred merely from this reduction.',
            'rigorous_primal_certificate':'Round the final effects to rational matrices with PSD/slack checks. Their acceptance probabilities are exact polynomials in beta,gamma. Certify .01-f>=0 on the closed half-disk with a rational sum-of-squares identity using1-beta^2-gamma^2 and-gamma, or exhaustive outward Bernstein subdivision with explicit boundary handling. Certify the gamma=0 combined constraint separately on[-1,1].',
            'finite_null_dual_certificate':'Choose any finite set of allowed null states, nonnegative weights lambda_i and a PSD Y satisfying Y>=tau0^tensor k-sum_i lambda_i sigma_i^tensor k. For any0<=E<=I with Tr(E sigma_i^tensor k)<=alpha_i, Tr(E tau0^tensor k)<=sum lambda_i alpha_i+Tr Y. Exact/outward verification of a value<.95 proves infeasibility even for this relaxed finite-null problem, hence also for the full SDP.',
            'first_screen':'Apply the fidelity obstruction and the explicit local-SPRT cost bound before solving the SDP. For the stated near-pure fixed-block3x goal this screen already rejects the whole family.',
            'unresolved_extension':'A sequential coherent tester needs CPTP instrument/comb causality constraints and expected preparation cost, plus uniform null control. A one-shot effect SDP is not a valid replacement. No claim of benefit or implementation authorization follows from leaving this extension open.'},
        'same_information_sota_boundary':'Collective hypothesis testing, measured relative entropy and coherence testing are established fields. Any useful FaultTune claim would need a concrete realizable circuit/resource tradeoff against strongest applicable local and collective controls. Giving only FaultTune the flag or coherent memory would not establish same-resource SOTA superiority.',
        'numerical_rows':quantum_rows}
    sources={Path(__file__).resolve().relative_to(ROOT).as_posix():sha(__file__),
             'experiments/circuit_validation/direct_tests.py':sha(ROOT/'experiments/circuit_validation/direct_tests.py'),
             'docs/CONTINUOUS_SYMMETRIC_DESIGN.md':sha(ROOT/'docs/CONTINUOUS_SYMMETRIC_DESIGN.md'),
             'docs/CONTINUOUS_PSR_INFORMATION_BOUND.md':sha(ROOT/'docs/CONTINUOUS_PSR_INFORMATION_BOUND.md')}
    shared['source_sha256']=sources
    quantum['source_sha256']=sources
    print(json.dumps({'shared':write_new('shared_static_updates_analysis.json',shared),
                      'collective':write_new('collective_joint_observation_feasibility.json',quantum),
                      'fixed_block_speedup_upper':str(fixed_block_speedup_bound.hi),
                      'with_filtered_cap_abstention_upper':str(capped_filter_speedup_bound.hi),
                      'new_observations':0},indent=2))


if __name__=='__main__':
    main()
