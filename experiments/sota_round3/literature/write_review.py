"""Write the scoped finite-horizon sequential-binomial prior-art review."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT/'results/sota_round3/literature'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    official = 'results/sota_round3/literature/sequential_source_'
    source_names = [
        'experiments/sota_round3/literature/fetch_sources.py',
        'experiments/sota_round3/literature/check_small_binomial_reference.py',
        'experiments/sota_round3/literature/write_review.py',
        'results/sota_round3/literature/sequential_4.6.3.tar.gz',
        'results/sota_round3/literature/sequential_index.txt',
        'results/sota_round3/literature/sequential_manual.pdf',
        'results/sota_round3/literature/sequential_manual.txt',
        'results/sota_round3/literature/sequential_manual_page81.png',
        official+'CV.Binomial.R', official+'Performance.Threshold.Binomial.R', official+'Optimal.Binomial.R',
        'results/sota_round3/literature/small_binomial_reference.json',
        'results/sota_round3/literature/howard_abs.txt',
        'results/sota_round3/literature/howard_full.html',
        'results/sota_round3/literature/howard_full.txt',
        'results/sota_round3/literature/johari_abs.txt',
        'results/sota_round3/literature/johari_full.html',
        'results/sota_round3/literature/johari_full.txt',
        'results/sota_round3/literature/simctest_manual.pdf',
        'results/sota_round3/literature/simctest_manual.txt',
        'results/sota_round3/literature/gandy_crossref.json',
        'results/sota_round3/literature/gandy_literature.txt',
        'results/sota_round3/literature/optimal_spending_crossref.json',
        'results/sota_round3/literature/exact_sequential_crossref.json',
        'results/sota_round3/literature/maxsprt_crossref.json',
        'results/sota_round3/literature/lademets_crossref.json',
        'results/structured_validation/continuous_fisher_literature_review.json',
        'experiments/structured_validation/weighted_fisher.py',
        'experiments/structured_validation/expert_config.py',
        'results/sota_round3/literature/downloads.json',
    ]
    value = {
        'status': 'checked_direct_prior_art_and_implementable_comparators_found',
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'reviewer': 'stream_sota',
        'scope': 'Finite-cap exact calibration of paired Bernoulli discordance-sign boundaries, H=2048 and delta=.01. No simulation observations, no changes to frozen prior rounds.',
        'main_conclusion': 'Finite-horizon binomial absorbing-state calibration, Beta mixture boundaries, mixture SPRTs, and optimized alpha spending all have direct prior art. A new certified implementation can be valuable but these principles are not an originality claim or a demonstrated international-leading result.',
        'source_access': {
            'connector_search': 'One attempt failed with transport HTTP request failure.',
            'fallback': 'Direct primary arXiv pages, official CRAN manuals/source archive, author publication page, and Crossref primary-paper metadata.',
            'downloaded_code_executed': False,
            'R_runtime_result_claimed': False,
            'new_simulation_observations': 0,
            'not_exhaustive_current_sota_search': True,
            'preserved_initial_environment_failure': 'downloads_initial_ssl_environment_failure.json records the first urllib attempt without the Anaconda SSL DLL PATH. The corrected downloads succeeded.'
        },
        'primary_sources': [
            {
                'id': 'Sequential_4_6_3',
                'title': 'Sequential: Exact Sequential Analysis for Poisson and Binomial Data',
                'authors': ['Ivair Ramos Silva', 'Martin Kulldorff'],
                'version': '4.6.3', 'published': '2026-09-14',
                'url': 'https://cran.r-project.org/package=Sequential',
                'manual_url': 'https://cran.r-project.org/web/packages/Sequential/Sequential.pdf',
                'source_url': 'https://cran.r-project.org/src/contrib/Sequential_4.6.3.tar.gz',
                'source_sha256': sha(DIRECTORY/'sequential_4.6.3.tar.gz'),
                'read_scope': ['Manual pp80-82 CV.Binomial', 'Manual pp88-91 Optimal.Binomial',
                               'Manual pp91-94 Performance.AlphaSpend.Binomial',
                               'CV.Binomial.R', 'Performance.Threshold.Binomial.R core statistic and absorption recursion',
                               'Optimal.Binomial.R inputs, constraints, objective and output'],
                'findings': [
                    'Exact finite-horizon sequential binomial critical values and arbitrary integer-boundary operating characteristics are standard implemented functionality.',
                    'Manual p81 claims CV.Binomial accepts upper/lower/two tails, but current source explicitly rejects any Tailed other than upper in both continuous and group branches.',
                    'Performance.Threshold.Binomial has a real two-sided branch and can evaluate supplied integer lower/upper boundaries under p=.5.',
                    'Optimal.Binomial targets chosen alternatives/power constraints using linear programming; it is not a universal unknown-effect optimum.'
                ]
            },
            {
                'id': 'maxsprt_2011',
                'title': 'A Maximized Sequential Probability Ratio Test for Drug and Vaccine Safety Surveillance',
                'authors': ['Martin Kulldorff', 'Robert L. Davis', 'Margarette Kolczak', 'Edwin Lewis', 'Tracy Lieu', 'Richard Platt'],
                'year': 2011, 'doi': '10.1080/07474946.2011.539924',
                'url': 'https://doi.org/10.1080/07474946.2011.539924',
                'read_scope': 'Verified bibliographic metadata; implemented mathematical statistic and algorithm inspected in official CRAN source, not an independently read full journal article.',
                'finding': 'Direct named comparator for unknown Bernoulli effect is a maximized likelihood ratio with a calibrated finite-horizon threshold.'
            },
            {
                'id': 'optimal_spending_2020',
                'title': 'Optimal Alpha Spending for Sequential Analysis with Binomial Data',
                'authors': ['Ivair R. Silva', 'Martin Kulldorff', 'W. Katherine Yih'],
                'year': 2020, 'journal': 'JRSS Series B 82(4):1141-1164',
                'doi': '10.1111/rssb.12379',
                'url': 'https://academic.oup.com/jrsssb/article/82/4/1141/7056022',
                'read_scope': 'Primary abstract via Crossref plus official implementing function/manual. Full paper not inspected.',
                'finding': 'Exact linear-programming alpha spending optimization predates this project; published optimality is conditional on specified objectives, error, power and alternatives.',
                'citation_correction': 'Manual Details mentions Silva and Kulldorff (2018), but the formal reference and DOI identify the three-author 2020 publication. Use the latter as the verified publication.'
            },
            {
                'id': 'gandy_2009',
                'title': 'Sequential Implementation of Monte Carlo Tests With Uniformly Bounded Resampling Risk',
                'authors': ['Axel Gandy'], 'year': 2009, 'doi': '10.1198/jasa.2009.tm08368',
                'url': 'https://doi.org/10.1198/jasa.2009.tm08368',
                'software_url': 'https://cran.r-project.org/package=simctest',
                'read_scope': 'Author bibliography, Crossref metadata, official simctest manual pp4-5 and 24-27. Full journal paper not inspected.',
                'finding': 'Precomputed/on-the-fly sequential Bernoulli decision boundaries with a spending sequence and uniform resampling-risk bound are established. Mapping its Monte Carlo decision convention to a two-sided sign error budget still requires explicit adaptation.'
            },
            {
                'id': 'howard_et_al_2021',
                'title': 'Time-uniform, nonparametric, nonasymptotic confidence sequences',
                'authors': ['Steven R. Howard', 'Aaditya Ramdas', 'Jon McAuliffe', 'Jasjeet Sekhon'],
                'url': 'https://arxiv.org/html/1810.08240',
                'version': 'arXiv v9, revised 2022-08-06; journal publication 2021',
                'read_scope': 'Section 3.2-3.5; Appendix beta-binomial Propositions 7 and 8, equations 56-61.',
                'finding': 'Two-sided Beta-binomial mixtures and one-sided truncated Beta mixtures are explicit standard boundaries. The paper attributes simpler i.i.d. Bernoulli versions to Ville, Robbins, Lai and Shafer et al. It also discusses tuning mixtures and numerical/discrete mixtures.',
                'limit': 'This is not a claim that its published asymptotic/time-uniform boundary equals our particular finite-cap exact crossing table.'
            },
            {
                'id': 'johari_pekelis_walsh',
                'title': 'Always Valid Inference: Bringing Sequential Analysis to A/B Testing',
                'authors': ['Ramesh Johari', 'Leo Pekelis', 'David J. Walsh'],
                'url': 'https://arxiv.org/html/1512.04922',
                'version': 'arXiv v3, 2019-07-16',
                'read_scope': 'Sections 5.1-5.4, especially the mSPRT definition and alternative-prior tuning discussion.',
                'finding': 'Mixture SPRTs and effect-prior-dependent finite-sample performance are established. Matching an alternative prior can matter; a single fixed alternative is not uniformly optimal over unknown effects.',
                'limit': 'Their paper is not a certified FaultTune simulation comparator, nor does its first-order efficiency claim establish exact finite-cap optimality here.'
            },
            {
                'id': 'lan_demets_1983',
                'title': 'Discrete sequential boundaries for clinical trials',
                'authors': ['K. K. G. Lan', 'D. L. DeMets'],
                'year': 1983, 'doi': '10.1093/biomet/70.3.659',
                'url': 'https://doi.org/10.1093/biomet/70.3.659',
                'read_scope': 'Metadata only. Exact binomial spending implementations inspected in Sequential instead.',
                'finding': 'Alpha spending is historical background, not a new local principle.'
            }
        ],
        'official_CV_Binomial_source_crosswalk': {
            'path': official+'CV.Binomial.R',
            'source_sha256': sha(ROOT/(official+'CV.Binomial.R')),
            'two_sided_doc_source_disagreement': {'manual_page': 81, 'rejecting_source_lines': [24, 196],
                'interpretation': 'Do not claim a successful official CV.Binomial(Tailed=two) runtime reproduction.'},
            'statistic': 'For p0=1/(1+z), LLR(D,K)=D KL(K/D,p0) when K/D>p0 and 0 otherwise; K=D uses D log(1+z). No factor 2.',
            'statistic_source_lines': [81, 90],
            'crossing': 'LLR >= cv, not strict >; integer upper absorption boundary is first K satisfying this and K>=M.',
            'crossing_source_lines': [102, 112],
            'minimum_events': 'M defaults to 1 and is the minimum number of upper-side successes, not a minimum total D; some early impossible looks are suppressed.',
            'groups': 'GroupSizes defaults to 1, testing at each D; a group vector tests at cumulative group sizes and must sum to N.',
            'cap': 'N is a positive integer. There is no N<=1000 hard check in CV.Binomial source; manual p80 recommends <=1000 for runtime. N=2048 was not executed in R.',
            'critical_value_search': 'Searches ordered attainable floating LLR values with a binomial Markov recursion. Final cv is rounded upward to 5 decimals (line174); compare resulting integer absorbing states rather than expecting identical float thresholds.',
            'numerical_scope': 'Official exact means exact-distribution computation, not arbitrary-precision outward arithmetic. Our dyadic path-count certificate should independently certify the frozen integer table.',
            'official_two_sided_evaluator': {
                'path': official+'Performance.Threshold.Binomial.R',
                'statistic_lines': [239, 249], 'two_sided_recursion_lines': [400, 459],
                'integer_boundary_branch_lines': [544, 573],
                'recipe_not_executed': "Performance.Threshold.Binomial(N=H,CV.lower=lower,CV.upper=upper,p=.5,GroupSizes=rep(1,H),Tailed='two',Statistic='Cases',RR=1)",
                'note': 'At p=.5 its two-sided LLR is D KL(K/D,.5), including symmetric endpoints. Supplied integer boundaries avoid float threshold conversion.'
            }
        },
        'small_reference_check': {
            'path': 'results/sota_round3/literature/small_binomial_reference.json',
            'sha256': sha(DIRECTORY/'small_binomial_reference.json'),
            'official_published_example': 'Manual p82 example N25,alpha.05,M1,z7/2,GroupSizes5 has cv1.9852 and displayed risk.04775995; independent integer transition calculation gives .04775995282015197.',
            'two_sided_fair_alpha_01_risks': {'N12': .0078125, 'N20': .00994873046875,
                'N32': .008275758475065231, 'N50': .009620236661112003},
            'scope': 'Exact integer/rational probability of the included boundary tables. Symmetric tables are a transparent mathematical extension; no R package code was executed.'
        },
        'strong_comparators_same_information_and_budget': [
            {'name': 'exact_finite_cap_Jeffreys', 'statistic': '2^D B(K+1/2,D-K+1/2)/B(1/2,1/2)',
             'status': 'Candidate, built from an established mixture with a finite-horizon exact boundary.'},
            {'name': 'exact_finite_cap_binomial_MaxSPRT', 'statistic': 'D KL(K/D,1/2), both tails',
             'status': 'Required direct strong comparator; does not require a chosen nonnull effect.',
             'adaptation': 'Use the two-sided statistic verified in the official Performance.Threshold source and certify its frozen integer boundary independently.'},
            {'name': 'exact_finite_cap_symmetric_point_SPRT',
             'statistic': '0.5[(2q)^K(2(1-q))^(D-K)+(2(1-q))^K(2q)^(D-K)]',
             'q_values_to_freeze': [.6, .75, .9, .99],
             'status': 'Strong effect-tuned controls. No oracle selection of q for the evaluated case; report each or select using separate development information.'},
            {'name': 'exact_finite_cap_pair_grid',
             'statistic': 'Existing fixed symmetric rational-odds prior, rewritten as Bernoulli sign LR mixture at q=omega/(1+omega).',
             'status': 'Gives the existing discrete mixture the same finite-cap calibration advantage.'},
            {'name': 'exact_finite_cap_pairwise_hedge',
             'statistic': '(Jeffreys sign mixture + existing pair_grid mixture)/2',
             'status': 'Exactly the m=1 Fisher-plus-Beta hedge, so the same (D,K) DP is available.',
             'adaptation': 'Explicit pairwise protocol; not silently identified with the previous variable-block-size Fisher hedge.'},
            {'name': 'exact_binomial_alpha_spending',
             'statistic': 'Integer two-sided boundaries with a predeclared cumulative spending schedule, evaluated by the same exact absorbing-state recursion.',
             'status': 'Useful additional boundary shapes such as prespecified power spending; all receive the same horizon/error budget.'},
            {'name': 'Optimal_Binomial_design_matched_LP',
             'status': 'Stronger optimization reference when objective, alternatives and power constraints are matched; not currently a ready H2048 executable comparator.'}
        ],
        'comparison_contract': [
            'All finite-budget methods use the same H=2048 discordances, delta=.01 and every-D eligibility, with a physical cap of4096 shots.',
            'A physical pair costs2 shots whether concordant or discordant; calibrating in D never makes concordant observations free.',
            'Freeze exact integer rejection states, including tie and terminal behavior. Floating statistics can propose shapes; exact path counts certify the actual table.',
            'Cap without a directional crossing is abstain for every comparator. It is not evidence accepting a nonpositive or negative sign.',
            'Count wrong decisions and cap abstentions in unconditional cost, and report each separately.',
            'Finite-horizon thresholds cannot be continued beyond H without a newly justified risk contract; retain an infinite-horizon baseline if such flexibility is a product requirement.',
            'Selecting an effect, prior or boundary after viewing evaluation outcomes is development selection, not held-out superiority.'
        ],
        'Optimal_Binomial_scope_for_FaultTune': {
            'verified_publication': 'Silva, Kulldorff and Yih (2020), DOI10.1111/rssb.12379',
            'ESampleSize': 'Source lines367-383 includes N*(1-power), matching capped nonrejection cost in Bernoulli-event units.',
            'ETimeToSignal': 'Source divides signal-time sum by power; this conditions on signaling and does not match unconditional capped cost.',
            'two_sided_design': 'Requires two fixed alternatives RR[1]<1<RR[2] and two target powers. Its objective averages the two alternative-specific measures, not every unknown effect.',
            'unknown_alternative': 'No single quoted LP optimum applies to a heterogeneous unknown-effect benchmark without a prespecified prior, minimax objective, or matched power constraints.',
            'physical_cost_mismatch': 'The Bernoulli-event horizon is not the same as a fixed physical-pair horizon with a random nuisance-dependent discordance count; physical expected shots need a separate matched objective.',
            'directional_errors': 'Equality-null two-sided rejection is not by itself a uniform directional-error theorem; a monotone-boundary coupling argument must be checked for the returned stopping rule.',
            'randomization_and_numerics': 'The inspected LP source uses continuous solution variables and a numerical simplex solver. A rounded nonrandomized integer boundary cannot inherit claimed LP optimality automatically; certify its actual risk and recompute cost/power.',
            'size_limits': 'Manual says continuous N<=240. Current source lines122-124 sets Nref300 when the supplied GroupSizes length<=20 and Nref240 otherwise, then rejects N>Nref. It cannot run H2048 unchanged.',
            'not_established': ['An exact H2048 optimal alpha-spending implementation', 'Global minimum physical shots over all effects',
                                'The validity or optimality of an uninspected randomized LP decision rule after conversion to our directional-abstention protocol']
        },
        'Fisher_and_hedge_fair_upgrade': {
            'pairwise_m1_identity': 'For one Bernoulli observation per side, conditioning on a discordance gives a fair sign under equality. Each fixed Fisher odds expert is a Bernoulli sign LR; its mixture depends only on(D,K). The Beta+pair_grid mixture is therefore eligible for the same symmetric-table DP.',
            'larger_blocks': 'For m>1 Fisher factors depend on each block total as well as cumulative sign; past block totals change the accumulated CGF. The full process is not a function of(D,K) alone.',
            'conditional_nuisance_limit': 'Conditioning retrospectively on an entire future block-total sequence is not automatically valid under adaptive actions or probabilities; future totals can depend on prior signs. An exact finite-horizon guarantee needs the appropriate joint or worst-case conditional-state recursion.',
            'invalid_shortcut': 'Replacing the original hedge threshold100 by a smaller Beta-only calibrated threshold, or scaling the Beta capital before adding it to the Fisher e-value, has no inherited hedge guarantee.',
            'simple_valid_alternative': 'Use an OR of separately valid Beta finite-boundary and Fisher original-boundary decisions with delta_beta+delta_fisher=delta, predeclared conflict handling and no post-hoc risk split. This is a new combined stopping protocol, not the old arithmetic hedge.',
            'scope_of_original_false_stop': 'The prior development fine-block hedge false stop remains preserved. Recalibration does not erase that observation or demonstrate a nominal-risk violation from one event.'
        },
        'novelty_crosswalk': [
            {'claim': 'Exact finite-cap absorbing-state binomial risk calibration', 'assessment': 'Direct established prior art in Sequential/MaxSPRT and sequential Bernoulli spending.'},
            {'claim': 'Jeffreys/Beta-binomial mixture or symmetric point-mixture SPRT', 'assessment': 'Established methods; no new statistical principle.'},
            {'claim': 'Optimizing finite-budget alpha spending', 'assessment': 'Direct published optimization prior art, with objective-specific scope.'},
            {'claim': 'A hash-bound integer probability certificate for a specific adaptive physical sign-decision pipeline', 'assessment': 'Potentially useful engineering/scientific implementation; priority and superiority remain unestablished.'},
            {'claim': 'International-leading performance', 'assessment': 'Requires matched strong controls, held-out cases/seeds, error/abstention reporting and a declared target metric; not implied by this literature audit.'}
        ],
        'sources': {name: sha(ROOT/name) for name in source_names},
    }
    target = DIRECTORY/'finite_horizon_prior_art.json'
    assert not target.exists()
    target.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':value['status'],'path':str(target),'sha256':sha(target)}))


if __name__ == '__main__':
    main()
