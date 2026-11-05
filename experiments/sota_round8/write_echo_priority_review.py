"""Write source-bound priority and strong-control requirements for the echo route."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/sota_round8/literature'


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def main():
    delta_u=2*math.asin(math.sqrt(5/198))-2*math.asin(math.sqrt(1/198))
    c=42*delta_u
    report={
        'schema':'faulttune.round8.echo_priority_review.v1',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'reviewer':'stream_sota',
        'status':'targeted_original_review_complete_with_explicit_coverage_limits',
        'new_simulation_observations':0,
        'sota_claim_supported':False,
        'echo_mechanism_novelty_supported':False,
        'headline':'Cancelling a static background by a toggling-frame echo while retaining a control-dependent difference is established differential Ramsey/quantum-sensing methodology. Conditional Hamiltonian tomography and echoed crosstalk calibration are also established. Any FaultTune claim must concern a new, verified decision-cost advantage over equally resourced controls, not the echo construction itself.',
        'original_sources':[
            {
                'authors':'Sarah Sheldon, Easwar Magesan, Jerry M. Chow, Jay M. Gambetta',
                'title':'Procedure for systematically tuning up crosstalk in the cross resonance gate',
                'version':'arXiv1603.04821v1, 2016-03-15',
                'doi':'10.1103/PhysRevA.93.060302',
                'url':'https://arxiv.org/html/1603.04821v1#p8',
                'locations':'Eq.(2); paragraph p8/Eqs.(4)-(5), Fig.2 Hamiltonian tomography; p10 and Fig.1 echoed gate; p14 gate-time accounting.',
                'verified_method':'Prepare the conditioning qubit in 0 or 1; scan drive duration and measure X,Y,Z on the target; fit the conditional Bloch generators; take sums/differences to recover single-qubit and conditional Hamiltonian coefficients. Use an echo and drive-sign reversal, plus a calibrated cancellation drive for residual terms.',
                'direct_overlap':'Conditioning on a spectator state to expose a simple generator, estimating a control-dependent Hamiltonian coefficient, and refocusing unwanted terms are prior techniques.',
                'important_limit':'The original explicitly explains that a noncommuting IY component is not completely removed by the standard echo. Its gate time includes echo pulses, delays and pulse rise/fall. Ideal instantaneous echoes cannot be promoted to robust hardware cancellation without an error model.',
                'not_same_benchmark':'Cross-resonance transmon Hamiltonian and gate-fidelity calibration differ from our diagonal P11 toy model and finite-risk sign decision. Reproducing all six tomography axes is unnecessary for a fair baseline when the simpler model is public.',
            },
            {
                'authors':'C. L. Degen, F. Reinhard, P. Cappellaro',
                'title':'Quantum sensing',
                'version':'arXiv1611.02427v2, 2017-06-06',
                'doi':'10.1103/RevModPhys.89.035002',
                'url':'https://arxiv.org/pdf/1611.02427v2',
                'locations':'Printed p20, Sec.VI.B/C, Fig.5, Eqs.(63)-(64); printed p17, Eq.(41)-(42) sensing-time/readout-overhead tradeoff; Sec.VIII/Eq.(104) phase wrapping and prior range.',
                'verified_method':'Echo accumulation is the first-half phase minus the second-half phase. General pulse sequences accumulate the integral of the signal times a sign-changing modulation function. This directly gives cancellation of a static offset and retention of a control-modulated component.',
                'direct_overlap':'The proposed gamma cancellation and chi differential phase are a special case of the established modulation-function formula. Choosing an evolution time against decoherence and setup/readout overhead, and respecting phase ambiguity, are also established design considerations.',
                'coverage_note':'The fetched arXiv HTML conversion stops at Section III; it was not used as evidence for missing sensing sections. The complete 45-page original PDF was downloaded and text-extracted to verify the cited sections.',
            },
            {
                'authors':'X. Dai et al.',
                'title':'Calibration of flux crosstalk in large-scale flux-tunable superconducting quantum circuits',
                'version':'arXiv2105.14360v2, 2021-10-22',
                'doi':'10.1103/PRXQuantum.2.040313',
                'url':'https://arxiv.org/html/2105.14360v2#S3.SS1',
                'locations':'Sec.III.1/Eqs.(1)-(3) flux-control matrix with offsets; Sec.III.2 general approach; Sec.III.5 CISCIQi iterative coordinate compensation.',
                'verified_method':'Model fluxes as C times control voltages plus offsets, estimate coupling coefficients using circuit symmetries, and iteratively redefine compensated controls to reduce bias from interacting circuits.',
                'direct_overlap':'Learning cross-control slopes while separating offsets and compensating nuisance interactions is established calibration work.',
                'not_same_benchmark':'This is DC flux-crosstalk calibration using loop/resonator data, not the proposed echoed binary-shot experiment. Its reported coefficient errors cannot be compared directly to our 1% reporting-risk or stopping-cost targets.',
            },
        ],
        'exact_model_reduction':{
            'status':'analytic specialization of prior modulation formula; no novelty assertion',
            'hamiltonian':'H(u)=(omega+u)X0/2+(gamma+chi*u)P11_12.',
            'preparation':'q2 exactly in |1>, q1 in |+>; q0 factor does not couple to the spectator block.',
            'single_echo_unit':'Write E_j=exp[-i tau (gamma+chi*u_j)P1] on q1. With rightmost operation first, U=X1 E_hi X1 E_lo=exp[-i tau(g_hi P0+g_lo P1)]. Up to a global phase, the relative phase is tau(g_lo-g_hi)=tau chi(u_lo-u_hi); orientation can be reversed by the readout convention.',
            'repetition':'For k identical units the signed relative angle has magnitude theta=c chi, c=k*tau*|u_hi-u_lo|. Gamma cancels exactly because the two halves have equal duration and the toggled projectors commute.',
            'noise_and_cost':'Static gamma cancellation does not cancel Markov dephasing. The circuit consumes 2k sensing windows and, in the restoring convention, 2k X1 pulses, plus q2 preparation, q1 preparation, analysis, readout, switching and processing.',
        },
        'proposed_round8_contract_as_received_from_design':{
            'not_frozen_acceptance':True,
            'amplitude_difference':delta_u,
            'tau':1,
            'k':42,
            'c':c,
            'public_abs_chi_bound':.2,
            'max_abs_phase':c*.2,
            'safe_abs_chi_threshold':.075,
            'safe_abs_phase_threshold':c*.075,
            'tan_threshold':math.tan(c*.075),
            'phase_branch':'The proposed bound gives |theta|<pi/2, so cos(theta)>0 and both boundary halfspaces represent the desired interval.',
            'contrast':'After independent readout-sign randomization, sin/cos means are (A sin(theta),A cos(theta)), with common unknown A>=0. No upper bound on the dephasing rate is needed for validity, but arbitrarily vanishing A prevents uniform finite-budget power.',
            'cost_received':'Approximately94.8 time units per shot:84 sensing windows+84 X1 at.1 each+2.2 setup/readout+.1 q2 preparation+.1 switching. This is a branch proposal and needs root/model validation; any extra analysis pulses must also be included.',
        },
        'strong_same_action_comparators':[
            {
                'name':'Task-directed dual-quadrature sequential halfspace control',
                'priority':'mandatory, stronger task match than estimating all Hamiltonian coefficients',
                'construction':'Let a=c*h. On every shot independently choose Q=y with probability p and Q=x with probability1-p, and choose the readout sign fairly. For demodulated Z in{-1,1}, define W_plus=1(Q=y)Z/p-tan(a)1(Q=x)Z/(1-p), W_minus=-1(Q=y)Z/p-tan(a)1(Q=x)Z/(1-p).',
                'mean':'Conditionally on past data, if the action randomization does not alter the shared contrast, E[W_plus]=A[sin(theta)-tan(a)cos(theta)] and E[W_minus]=A[-sin(theta)-tan(a)cos(theta)]. The two wrong-side phase nulls imply the corresponding mean is nonnegative.',
                'finite_risk_rule':'For each side use the nonnegative product of factors1-lambda_t W_t, with past-only lambda_t>=0 bounded by the inverse largest possible positive W_t. It is a supermartingale under that side null. Reject at1/alpha by Ville. Fixed valid lambdas, mixtures and paid adaptive bets all qualify as controls.',
                'intersection_union':'A safe interval is certified only after both side tests reject. Since every outside-interval parameter satisfies at least one side null, using level alpha for each side already gives a level-alpha interval certificate; an alpha/2 split is not automatically necessary. Combining this with separate final-Delta certificates still needs correct joint risk accounting.',
                'implementation_status':'Analytic valid-control specification communicated to design/stats; this literature branch has not implemented or benchmarked it.',
            },
            {
                'name':'Direct phase inversion with exact binomial confidence sets',
                'construction':'At a frozen shot budget obtain simultaneous binomial intervals for the two quadrature means, and invert the feasible set {(A,chi):mu_y=A sin(c chi),mu_x=A cos(c chi),A>=0,chi in public range}. Intersect projected chi intervals with the full physical parameter set; certify only when the original task condition holds for every feasible point.',
                'restrictions':'Use a valid simultaneous interval allocation. Repeatedly checking fixed-sample intervals is not optional-stopping safe; replace with confidence sequences/e-process inversion or a prespecified spending schedule. A ratio/atan2 point estimate or Bayesian credible interval alone is insufficient.',
            },
            {
                'name':'Same-action constrained-likelihood and experimental-design control',
                'construction':'Use the public physical sin/cos model and identical paid actions to fit/infer chi and contrast, with a constrained null likelihood or valid mixture/e-process. Optimize evolution count, amplitudes, quadrature allocation and readout phase using only public bounds or paid past data.',
                'local_design_guidance':'For known contrast and sin quadrature, Fisher information is A^2*c^2*cos(theta)^2/[1-A^2*sin(theta)^2]. The relevant quantity is information per total paid cost. If A is independent of amplitude and the public phase range remains on the chosen branch, maximizing the control separation is natural; long k is limited by dephasing, overhead and phase ambiguity.',
                'quadrature_allocation':'For estimating a halfspace mean with fixed total samples, the local variance-minimizing ratio is n_y/n_x=sqrt(1-mu_y^2)/[tan(a)*sqrt(1-mu_x^2)]. It is only an asymptotic design rule; unknown means require paid learning, and the final decision still needs finite-risk certification.',
                'no_oracle':'No use of scorer truth for contrast, chi, decoherence or optimal bets. A faithful strong control may exploit any public prior and nuisance restriction used by FaultTune.',
            },
        ],
        'circuit_prior_noise_constraints':[
            'Equal durations and a common static gamma in both halves are essential. Intra-echo gamma drift contributes tau(gamma_lo-gamma_hi); it must be modeled or bounded.',
            'The cancellation is exact for commuting diagonal toggled terms. Noncommuting Hamiltonian additions or finite-pulse residual evolution require explicit control/error analysis, as the cross-resonance original itself illustrates.',
            'The prior branch |c chi|<pi/2 is essential to the simple halfspace certificate. Without it, larger chi can alias to an apparently small phase. Broader priors require multi-time unwrapping or set-valued phase inference.',
            'The same unknown A must multiply sine and cosine. Choosing the quadrature independently within each shot protects against time-varying contrast when the random measurement choice does not itself change A. Pairing separate sin/cos shots with unrelated contrasts does not justify their cancellation.',
            'Exact q2=1 preparation matters for a magnitude bound. If q2=1 only with unknown probability q and all else is ideal, the means become (q A sin(theta), A[(1-q)+q cos(theta)]). This is an offset circle, not a common contrast times(sin,cos), and can make large |chi| appear small. Use the ideal preparation convention explicitly or include q as a bounded nuisance.',
            'Phase-dependent readout slopes, sign-dependent offset errors, q1 preparation/analysis errors and unequal pulse visibility can distort the assumed circle. Known additive quadrature bounds e_y,e_x induce halfspace uncertainty at most e_y+tan(a)e_x; invert an expanded model or shift the valid null test accordingly.',
            'No finite-budget95% stopping guarantee can be uniform over arbitrarily small A. Power must be checked for the actual frozen evaluation cases or a justified positive visibility lower bound.',
            'Certifying |chi|<h alone is not the original FaultTune acceptance decision. The complete Delta sign/margin must hold throughout the joint confidence/likelihood region, and all diagnostic/learning shots and pulses must be charged.',
        ],
        'claim_and_acceptance_boundaries':{
            'new_algorithm_novelty':'Unresolved. This review identifies a standard mechanism and concrete applicable controls, not a new SOTA candidate.',
            'risk_power_cost':'Retain at most1% uniform reporting risk, at least95% per-case correct stopping, and the original threefold unconditional paid-cost ambition against equally resourced applicable strong controls.',
            'no_field_wide_survey':'Three targeted originals were inspected. Dai provides crosstalk-calibration context; Sheldon and the complete Degen PDF provide the direct mechanism/practice overlap. This is not an exhaustive latest-literature survey.',
            'next_decisive_work':'Implement and qualify the task-directed halfspace and physical likelihood controls, compare paid costs on frozen independent cases, and assess pulse/preparation/noise robustness before attributing an operational gain to a new method.',
        },
    }
    sources=['experiments/sota_round8/fetch_echo_literature.py','experiments/sota_round8/fetch_sensing_pdf.py','experiments/sota_round8/write_echo_priority_review.py','results/sota_round8/literature/public_source_ledger.json','results/sota_round8/literature/sensing_pdf_source_ledger.json','results/sota_round8/literature/degen2017_v2.pdf','results/sota_round8/literature/degen2017_v2_pdf.txt','results/sota_round7/literature/direct_priority_and_resource_comparators.json']
    for name in ['sheldon2016_abs','sheldon2016_full','dai2021_abs','dai2021_full','degen2017_abs','degen2017_full']:
        sources.extend('results/sota_round8/literature/'+name+suffix for suffix in ['.html','.txt'])
    report['source_sha256']={p:sha(p) for p in sources}
    path=OUT/'echo_priority_and_same_action_controls.json'
    with path.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path.relative_to(ROOT)),'source_files':len(sources),'max_abs_phase':c*.2,'threshold_angle':c*.075,'sota_claim_supported':False}))


if __name__=='__main__':main()
