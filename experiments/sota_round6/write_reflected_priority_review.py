"""Write the source-bound priority review after targeted original-text reading."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT/'results/sota_round6/literature'
OUT = DIR/'reflected_pair_priority_review.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    paths = [Path(__file__).resolve(),
        ROOT/'results/sota_round5/theory/reflected_pair_weak_signal_theorem.json',
        ROOT/'results/sota_round5/theory/reflected_pair_weak_signal_peer_review.json',
        ROOT/'results/sota_round5/theory/reflected_pair_global_gap_checklist.json']
    names = ['metadata_search_ledger.json', 'metadata_search_ledger_ssl_ready.json',
        'source_ledger_1.json', 'source_ledger_2.json', 'source_ledger_3.json',
        'kl_discrimination_crossref.json', 'overshift_crossref.json', 'measurement_title_crossref.json',
        'baek2020_abs.html', 'baek2020_full.html', 'baek2020_full.txt',
        'sharpness2026_abs.html', 'sharpness2026_full.html', 'sharpness2026_full.txt',
        'overshift_abs.html', 'overshift_full.html', 'overshift_full.txt',
        'kl_properties_abs.html', 'kl_properties_full.html', 'kl_properties_full.txt',
        'fourier_discrimination_abs.html', 'fourier_discrimination_full.html', 'fourier_discrimination_full.txt',
        'information_radius_arxiv_search.html', 'tendick_arxiv_search.html', 'epsilon_resource_arxiv_search.html']
    paths.extend(DIR/name for name in names)
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    peer = json.loads((ROOT/'results/sota_round5/theory/reflected_pair_weak_signal_peer_review.json').read_text())
    assert peer['status'] == 'passed_independent_analytic_review_fixed_interior_weak_signal_scope'
    result = {
        'status': 'targeted_priority_review_complete_with_material_coverage_limits',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'reviewer': 'stream_sota',
        'new_simulation_observations': 0,
        'novelty_established': False,
        'sota_algorithm_comparison_complete': False,
        'headline': 'The minimax relative-entropy measurement-coherence quantity is already explicitly defined by Baek et al. (2020). The potential contribution is an exact reflected-probe structural solution and physical tangent certificate in the audited fixed-interior weak-signal regime, not a new resource measure or a new KL design criterion.',
        'current_theorem_scope': {
            'accepted': 'For each fixed0<A<1 and|B|<min(A,1-A), there exists epsilon(A,B)>0 such that every0<|C|<epsilon has an exactly globally KL-optimal reflection pair with capacity weights; the optimal cosine is unique and interior and a single physical affine tangent null bounds KL at every angle.',
            'not_accepted': 'No uniform or computable epsilon, all-C reflected-pair theorem, paid-learning procedure, finite-sample SOTA superiority, or priority claim follows.',
            'all_C_status': 'Unresolved in the frozen round5 source and its gap checklist; finite diagnostics are not a global proof.'},
        'direct_measurement_resource_overlap': {
            'work': 'Kyunghyun Baek, Adel Sohbi, Jaehak Lee, Jaewan Kim, Hyunchul Nha, Quantifying coherence of quantum measurements, New Journal of Physics22(2020)093019.',
            'doi': '10.1088/1367-2630/abad7e',
            'original_url': 'https://arxiv.org/html/2008.03999v1#S3.E7',
            'abstract_url': 'https://arxiv.org/abs/2008.03999',
            'version_read': 'arXiv2008.03999v1, submitted2020-08-10; current abstract lists onlyv1.',
            'location': 'Section3.1, Definition2, Eq.(7); channel divergence is defined in the sentence immediately after Eq.(7). Free POVMs are defined in Definition1 / Eq.(4). The different robustness identity is Eq.(12).',
            'verified_original_definition': 'C_S(A)=min_{M in I(d,n)} S(A||M), with S(A||M)=sup_rho H_rho(A||M) and H_rho=sum_a p_A(a)log[p_A(a)/p_M(a)]. The displayed supremum is over the system state rho, without a quantum reference ancilla.',
            'mapping': [
                'Let H=A I+B Z+C X be a binary qubit POVM effect and use the measurement M_H={H,I-H}. H is not a density operator: Tr(H)=2A. The constraint0<=H<=I is exactly B^2+C^2<=min(A,1-A)^2.',
                'Let rho_x=(I+cos(x)Z+sin(x)X)/2. This is the trace-one real pure probe state, and Tr(rho_x H)=A+Bcos(x)+Csin(x).',
                'Every free binary qubit POVM has diagonal first effect Q=aI+bZ, with|b|<=min(a,1-a). Therefore the null family matches exactly.',
                'For fixed H and diagonal Q, output KL depends only on the X,Z coordinates of rho. These coordinates range over the closed Bloch disk. Joint convexity of Bernoulli KL in its two rates makes this a convex function of those coordinates; any disk point is a mixture of boundary points. Hence the supremum over all complex/mixed qubit states equals the supremum over the real pure circle.',
                'Baek Eq.(7) therefore equals D=min_Q sup_x kl(p_H(x)||p_Q(x)), up to a conventional logarithm-base factor. This identifies the minimax object; identifying our maxmin design value V withD requires the separate argument below.'
            ],
            'maxmin_minmax_bridge': {
                'warning': 'Do not replace a classically labeled preparation design w by its averaged density matrix inside KL. The objective integrates per-angle KL and retains the angle label.',
                'regularization': 'For0<e<1 define Q_e=(1-e)Q+e I/2 and let Q_e range over the resulting compact convex subset of physical diagonal effects. Its output probabilities lie in[e/2,1-e/2].',
                'continuous_game': 'The regularized payoff is continuous and finite on the compact circle and null parameter set, linear in the probability measurew and convex inQ. Standard Sion minimax applies, giving V_e=D_e.',
                'uniform_error': 'Both q_e>=(1-e)q and1-q_e>=(1-e)(1-q), so kl(p||q_e)<=kl(p||q)-log(1-e), including boundary conventions. Subset inclusion plus this inequality give V<=V_e<=V-log(1-e) andD<=D_e<=D-log(1-e).',
                'conclusion': 'The constant null I/2 makes both values finite. Sendinge down to0 proves V=D=C_S(M_H) for this physical binary real-qubit model, without assuming unregularized KL is finite on null boundaries. This is an elementary model-matching derivation supplied by this review, not a claim that Baek explicitly states the reflected design game.',
                'review_status': 'Written analytic derivation for parent review; not a proof-assistant formalization.'},
            'what_the_paper_solves': 'It defines the statistical-distance/relative-entropy coherence monotone and proves resource monotonicity. Its explicit two-outcome qubit formula is for robustness/l-infinity coherence, not for the relative-entropy minimax.',
            'robustness_distinction': 'Eq.(12) states C_linf=R_C=C_l1/2 for dichotomic qubit POVMs. For{H,I-H} this is2|C|. This is not C_S, not the Bernoulli capacityR, and cannot substitute for our information rate.',
            'overlap_assessment': 'Direct prior definition of the same minimax resource quantity. The inspected original does not provide the entropy-center tangent null, a reflected two-probe optimizer, or the fixed-(A,B) nonzero weak-C exact structural theorem. That absence in this paper is not a field-wide novelty proof.'},
        'other_work_comparisons': [
            {'work': 'J. Lopez-Fidalgo, C. Tommasi, P. C. Trandafir (2007), An Optimal Experimental Design Criterion for Discriminating Between Non-Normal Models.',
             'doi': '10.1111/j.1467-9868.2007.00586.x',
             'url': 'https://academic.oup.com/jrsssb/article/69/2/231/7109346',
             'evidence_level': 'Publisher-deposited Crossref title/authors/abstract; publisher original returned403. Its criterion is also explicitly reproduced in the full original Aletti-May-Tommasi paper below.',
             'scope': 'General locally specified true-model versus parametric rival-model KL design; equivalence theorem and algorithm, with lognormal/gamma examples.',
             'overlap': 'Our sup_design inf_null expected KL is an instance of the established KL-optimal discrimination-design criterion. A uniform pointwise KL upper certificate with equality at supports is established design-theory methodology.',
             'difference': 'A general equivalence theorem does not furnish the explicit two-reflection support, entropy-capacity weights, or globally feasible tangent affine null for the physical Bernoulli trigonometric family.'},
            {'work': 'Giacomo Aletti, Caterina May, Chiara Tommasi, KL-optimum designs: theoretical properties and practical computation.',
             'doi': '10.1007/s11222-014-9515-8',
             'url': 'https://arxiv.org/html/1212.3556v4',
             'version_read': 'arXiv1212.3556v4,2014-09-29.',
             'evidence_level': 'Full original read for Sections2,4,5.2 and conclusion.',
             'locations': 'Section2 Eq.(2) gives the minimum integrated KL criterion; Section4 / Eq.(11) treats its directional derivative; Section5.2 is a logistic-regression discrimination example.',
             'overlap': 'General KL design, continuity, equivalence checking, singular optimal designs and numerical regularization already have a developed literature. Our reflection design makes the two-dimensional affine nuisance design matrix rank deficient because both points have the same cosine; generic full-rank regular-design algorithms therefore require care.',
             'difference': 'Their binary example uses logistic regression on[0,1], not identity-link Bernoulli p=A+Bcosx+Csinx with the global physical cone. Neither their general algorithm nor that example supplies our exact weak-C support theorem.'},
            {'work': 'Holger Dette, Viatcheslav B. Melas, Petr Shpilev (2015 preprint), T-optimal discriminating designs for Fourier regression models.',
             'url': 'https://arxiv.org/html/1512.07441v1',
             'version_read': 'arXiv1512.07441v1,2015-12-23.',
             'evidence_level': 'Full original Sections2-4 inspected.',
             'locations': 'Section2 Eq.(2.1) and the following T-optimal criterion; Theorem2.1 gives an approximation/equivalence characterization; Theorems3.1-3.3 give explicit Fourier support structures.',
             'scope': 'Additive mean regression with constant variance and squared mean separation; nested Fourier models differing in at most three terms.',
             'overlap': 'Analytic trigonometric discrimination designs, reflected/equispaced supports, and two-versus-three support transitions are prior subjects. A claim of first optimal trigonometric discrimination design would be false.',
             'difference': 'Squared-error T-optimality is not the finite-C Bernoulli KL objective; Bernoulli variance depends on p0 and the physical null cone matters. Its exact support theorems cannot be imported without a reduction. The Fisher/quadratic leading limit is related but does not prove an exact nonzero-C neighborhood.'},
            {'work': 'Leonardo Banchi, Dominic Branford, Chetan Waghela, Overshifted Parameter-Shift Rules: Optimizing Complex Quantum Systems with Few Measurements.',
             'doi': '10.1088/2058-9565/ae73ad',
             'url': 'https://arxiv.org/html/2510.05289v1',
             'version_read': 'arXiv2510.05289v1,2025-10-06; current Crossref publication date2026-06-30.',
             'evidence_level': 'Full original SectionsII andIII inspected; publisher HTML was a captcha, not treated as full text.',
             'locations': 'Eq.(7)-(9) define exact derivative shift identities and symmetric shifts; SectionIII.1/Eq.(16)-(20) optimize measurement overhead via shot allocation and coefficient norms.',
             'scope': 'Linear unbiased derivative estimators for known frequency sets or bandwidth constraints, possibly many/infinite shifts; minimum-variance/measurement-overhead optimization under its noise model.',
             'overlap': 'Reflection shifts, optimal shot allocation and convex optimization over shift rules are established, including broader multi-frequency generators than our single harmonic.',
             'difference': 'The paper does not solve Bernoulli sign identification against a physical composite null, does not provide our KL capacity/tangent-null theorem, and does not certify a sequential95%/1% stopping-cost advantage. Conversely our oracle theorem is not a stronger general derivative-estimation algorithm than overshifted PSR.'},
            {'work': 'Kyunghyun Baek, Yonggi Jo, Hyunchul Nha (2026), Maximal coherence of quantum measurement and the resource theory of sharpness.',
             'url': 'https://arxiv.org/html/2607.05847v1#Thmtheorem2',
             'version_read': 'arXiv2607.05847v1,2026-07-07.',
             'evidence_level': 'Directly relevant definitions, Theorem2, Corollary4 and qubit example inspected; no wider new search initiated after this check.',
             'locations': 'Theorem2 / Eq.(17): maximal coherence over unitary choices of basis equals distance to trivial measurements for commuting POVM elements. Corollary4 uses the fact that binary POVM elements commute.',
             'overlap': 'Every binary POVM{H,I-H} meets its commutativity assumption. It connects basis-optimized measurement coherence to sharpness and reinforces that our object belongs to existing measurement resource theory.',
             'difference': 'Theorem2 optimizes the incoherent basis overU. Our model fixes theZ basis and fixedA,B whileC is small; varyingU changesB,C. Thus the theorem does not give the fixed-basis reflected-pair optimizer or its tangent null. Its commuting-case result includes the orientation-maximized/B=0 capacity special case, so that special case is a weak novelty candidate.'},
            {'work': 'Frank Nielsen (2021), On a Variational Definition for the Jensen-Shannon Symmetrization of Distances based on the Information Radius.',
             'url': 'https://arxiv.org/abs/2102.09728',
             'evidence_level': 'Official arXiv search metadata/abstract only; not a full original-theorem audit in this pass.',
             'overlap': 'Information radius and weighted Jensen-Shannon capacity are classical notions. For a fixed pair of Bernoulli endpoints, maximizing weighted Jensen-Shannon information gives the equal-KL center and the displayed entropy-secant formula by elementary differentiation.',
             'difference': 'This identifies the pairwise subproblem, not the proof that a reflected pair is globally sufficient among all circle designs. Do not claim novelty forq_c=logistic([f(p+)-f(p-)]/(p+-p-)) or its capacity weight alone.'}
        ],
        'classical_capacity_derivation': [
            'For fixed endpointsu<v, letq_w=(1-w)u+w v andf(p)=p logp+(1-p)log(1-p). The two-input Bernoulli channel mutual information isI(w)=(1-w)f(u)+w f(v)-f(q_w).',
            'An interior optimum satisfiesf(v)-f(u)=(v-u)fprime(q_w), henceq_c=logistic([f(v)-f(u)]/(v-u)) andw=(q_c-u)/(v-u). This also equalizeskl(u||q_c)=kl(v||q_c).',
            'Applying this identity separately for eachz computes the best reflected-pair lower bound. The nontrivial model-specific step is a single globally physical affineq*(z) that bounds both branches everywhere and meets this radius.'
        ],
        'next_priority_original': {
            'work': 'Lucas Tendick, Martin Kliesch, Hermann Kampermann, Dagmar Bruss (2023), Distance-based resource quantification for sets of quantum measurements, Quantum7,1003.',
            'doi': '10.22331/q-2023-05-15-1003',
            'url': 'https://arxiv.org/abs/2205.08546',
            'publisher_url': 'https://quantum-journal.org/papers/q-2023-05-15-1003/',
            'why_first': 'It is the direct modern measurement-distance continuation cited by Baek-Jo-Nha2026; AppendixI treats dichotomic measurements. Check whether any relative-entropy specialization or exact optimizer is hidden behind norm-equivalence results before advancing a first-structural-solution claim.',
            'evidence_now': 'Bibliography verified in the2026 original and official arXiv exact-title result obtained. Full text and AppendixI not read in this pass, so their scope is not asserted.',
            'second_if_needed': 'A. Mitra, S. Mukherjee, C. Lee, Distance-based measures and Epsilon-measures for measurement-based quantum resources, arXiv2505.11331, DOI10.1103/wwn7-j8df; bibliographic existence verified, exact theorem overlap still unread.'},
        'priority_conclusions': {
            'established_prior': ['KL-optimal discrimination-design objective and equivalence methodology.',
                                  'The same unassisted relative-entropy distance to incoherent binary measurements.',
                                  'Bernoulli two-input capacity, equal-KL center and entropy-secant weights.',
                                  'Trigonometric discrimination support analysis and variance-optimal symmetric parameter shifts.'],
            'potential_specific_contribution': 'Exact global two-reflection support for every sufficiently small nonzeroC at fixed physical interior(A,B), with unique angle, generally unequal weights, an explicit physical tangent null, and closed weak-C expansions; equivalently an explicit local solution of a previously defined measurement-coherence minimax.',
            'what_is_not_shown_by_search': 'No inspected original contained this exact fixed-basis weak-C theorem or the all-C capacity reduction. This records the inspected corpus only; novelty is not established.',
            'recommended_safe_story': 'We give a model-specific exact structural solution and certificate for KL-optimal sign discrimination, equivalently a local analytic evaluation of an established measurement-coherence measure. The global extension and literature priority remain open.',
            'international_lead_gap': 'A potentially new theorem must clear direct resource-theory priority checks and the all-C proof gap. Separately, a complete fair algorithm-level SOTA comparison remains unfinished; theorem novelty, oracle information optimality, and superiority of a paid finite-sample algorithm are different deliverables.'},
        'coverage_limits': [
            'The current web search connector was attempted once and failed with a transport HTTP send error before returning results.',
            'The first public-fetch attempt lacked the Python SSL runtime and failed. That failure ledger is preserved. A corrected runtime then retrieved public Crossref and arXiv records.',
            'Several broad Crossref requests returned429. Search results were noisy, and later exact-title / citation-chain retrieval was targeted rather than an exhaustive survey.',
            'The2007 publisher original returned403; its deposited abstract and a subsequent original reproducing the KL criterion were inspected. The overshift publisher response was a captcha and was excluded as substantive evidence; its arXiv original was read.',
            'This pass does not cover all citing papers, books, non-English optimal-design literature, all identities under alternative parameterizations, or every measurement-channel divergence. Missing results cannot justify a priority claim.',
            'The2026 maximal-coherence paper was checked only for direct overlap. Tendick2023 and Mitra2025 remain prioritized unread originals. Gour-Winter1906.03517 full HTML was downloaded during the last fixed fetch list but not used to assert a theorem comparison; reference-assisted channel definitions must not be conflated with Baek Eq.(7).'
        ],
        'source_sha256': hashes}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'artifact': str(OUT), 'sha256': sha(OUT),
                      'direct_prior': 'Baek2020 Eq7', 'novelty_established': False}), flush=True)


if __name__ == '__main__':
    main()
