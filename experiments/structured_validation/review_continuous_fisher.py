"""Persist the stream_stats review and independent deterministic references.

The mathematical/source review was performed separately by stream_stats.
This script corroborates exact toy identities without simulator or ledger I/O;
it does not rerun the author's check suite or assert rigorous mpmath intervals.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import mpmath as mp

import continuous_fisher as implementation


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/"results/structured_validation/continuous_fisher_independent_review.json"


def main():
    started = time.perf_counter()
    assert not OUTPUT.exists(), "Preserve the completed review"
    paths = ["docs/CONTINUOUS_FISHER_EXPLORATION.md",
             "experiments/structured_validation/continuous_fisher.py",
             "experiments/structured_validation/continuous_fisher_checks.py",
             "results/structured_validation/continuous_fisher_checks.json",
             "experiments/structured_validation/conditional_fisher.py",
             "experiments/circuit_validation/direct_tests.py",
             Path(__file__).resolve().relative_to(ROOT).as_posix()]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    sources = {name: digest(ROOT/name) for name in paths}
    author = json.loads((ROOT/paths[3]).read_text(encoding="utf-8"))
    assert author["status"] == "passed"
    for name, expected in author["sources"].items():
        assert digest(ROOT/name) == expected, name
    coarse = implementation.geometric_nodes((2, 1), 24)
    nodes = tuple(a+(b-a)*Fraction(k, 4) for a, b in zip(coarse[:-1], coarse[1:])
                  for k in range(4))+(coarse[-1],)
    references = []
    with mp.workdps(120):
        root6 = mp.sqrt(6)
        fixtures = [
            ("m1_three_positive", [{"m":1,"minus_successes":0,"plus_successes":1}]*3,
             lambda x: 8*(x**6+1)/(1+x*x)**4, mp.mpf(5)/2, "5/2"),
            ("m2_positive_extreme", [{"m":2,"minus_successes":0,"plus_successes":2}],
             lambda x: 6*(x**4+1)/((1+x*x)*(x**4+4*x*x+1)), 2*root6-3, "2*sqrt(6)-3"),
            ("m2_central", [{"m":2,"minus_successes":1,"plus_successes":1}],
             lambda x: 12*x*x/((1+x*x)*(x**4+4*x*x+1)), 3-root6, "3-sqrt(6)"),
            ("two_m2_central", [{"m":2,"minus_successes":1,"plus_successes":1}]*2,
             lambda x: 72*x**4/((1+x*x)*(x**4+4*x*x+1)**2), 9-mp.mpf(7)*root6/2,
             "9-(7/2)*sqrt(6)")]
        for name, blocks, rational_integrand, exact, formula in fixtures:
            integrated = 2/mp.pi*mp.quad(rational_integrand, [0, mp.mpf(1)/4, mp.mpf(1)/2, 1])
            difference = abs(integrated-exact)
            assert difference < mp.mpf("1e-100"), name
            certificate = implementation.certificate_from_blocks(blocks, nodes)
            score = sum(b["plus_successes"]-b["minus_successes"] for b in blocks)
            assert certificate["score"] == score
            assert certificate["conditional_score"] == {"numerator":score,"denominator":2}
            assert certificate["shots"] == 2*sum(b["m"] for b in blocks)
            log_lower = mp.mpf(certificate["log_evidence_lower_bound"])
            assert log_lower < mp.log(exact), name
            references.append({"case":name, "blocks":blocks, "complete_mixture_formula":formula,
                               "complete_mixture_reference":mp.nstr(exact, 110),
                               "independent_integral_absolute_error":mp.nstr(difference, 8),
                               "implementation_certified_log_lower":certificate["log_evidence_lower_bound"],
                               "lower_fraction_of_complete_reference":mp.nstr(mp.exp(log_lower)/exact, 24),
                               "score_and_physical_shot_units_checked":True})
    assert sources == {name:digest(ROOT/name) for name in paths}, "Review sources changed"
    result = {
        "status":"passed", "reviewer":"stream_stats independent mathematical and source review",
        "sources":sources,
        "scope":"New continuous-mixture exploration only. No simulator calls, no prospective files, no stored trial ledgers.",
        "mathematical_checks":[
            "The normalized density is exp(lambda/2)/(pi*(1+exp(lambda)))=1/(2*pi*cosh(lambda/2)); it is the Bernoulli-q Jeffreys prior after q=logistic(lambda), not a claim about the design-dependent hypergeometric Jeffreys prior.",
            "Central CGFs are convex and nonnegative; log density has second derivative -1/(4*cosh(lambda/2)^2). Thus log integrand is strictly concave.",
            "Endpoint lower values and the exact positive log-odds width yield the exponential-chord integral lower bound. Nonoverlapping intervals can be summed and outside positive mass discarded without renormalizing.",
            "Data-dependent integration nodes leave the underlying fixed continuous mixture unchanged; the numerical lower process itself need not be a supermartingale.",
            "Wrong-direction events are dominated by 1/2 plus the opposite-half continuous expert mixture. Tonelli and Ville justify threshold 1/delta=100 at delta=.01; equality uses the complete mixture for either direction together.",
            "The conditional score is U=sum(plus-minus)/2 while the API score is 2*U; physical shots are twice the sum of per-side block counts.",
            "When every nonempty block has m=1, the complete integral equals the exact Jeffreys paired-Beta capital, including its rational factorial formula. Finite chord calculations generally give strict lower values.",
            "The m=2,T=2 complete-mixture identities 2*sqrt(6)-3 and 3-sqrt(6) follow independently from positive rational integrals and give exact central expectation one.",
            "Under a uniform positive true gap d and sum(m)->infinity, the without-replacement Hoeffding bound is K_b<=m_b*lambda^2/8. Bounded martingale SLLN gives liminf U/sum(m)>=d/2. A compact interval inside (0,4d) proves ideal continuous-mixture growth.",
            "That sufficient condition gives an almost surely finite ideal first stop with directional error at most delta, not an almost surely correct first stop. Rates stated per sum(m) must be divided by two for physical shots.",
            "For n=sum(m), |log L derivative|<=n and |log rho derivative|<=1/2. Integrating the global Lipschitz lower cone yields M>=L(lambda*)/[pi*(n+1/2)*cosh(lambda*/2)] for each finite, possibly data-selected lambda*.",
            "The explicit expert comparison is O(log N) for fixed lambda*. A data-dependent growing lambda* retains the log cosh(lambda*/2) term; finite chord Q does not automatically inherit the bound."
        ],
        "implementation_checks":[
            "Machin pi uses exact Fraction alternating-series brackets and directed conversion.",
            "Positive rational odds nodes are deduplicated and sorted exactly; unresolved log widths are rejected.",
            "All CGFs and endpoint integrands use outward arithmetic; the final decision uses the lower log bound and upper log threshold only.",
            "The exponential chord uses a scaled negative exponential at large differences and a positive exprel series with geometric remainder near equality.",
            "The reported upper quadrature endpoint encloses the computed lower quadrature, not the complete mixture; that distinction is explicit.",
            "The independent rectangle checker correctly removes the minimum support power, uses reciprocal polynomial symmetry, and integrates the whole transformed [0,1] interval with exact rational rectangle bounds.",
            "The finite wrong-crossing DP deliberately keeps paths after earlier correct crossings, so it bounds the larger event relevant to a delayed numerical rule."
        ],
        "author_checks_inspected":{
            "status":author["status"], "m1_cases":len(author["m1_beta_cases"]),
            "complete_integral_enclosures":len(author["full_integral_enclosures"]),
            "exact_central_normalizations":author["m1_exact_central_normalizations"],
            "finite_ever_wrong_enumerations":len(author["exact_finite_wrong_crossings"]),
            "invalid_inputs_rejected":author["invalid_inputs_rejected"],
            "suite_independently_rerun_by_reviewer":False},
        "independent_deterministic_references":references,
        "reference_arithmetic":"120-digit mpmath rational-function integration corroborates algebraic closed forms; it is not an outward integration certificate. These four references were independently executed, not the author's full suite.",
        "blocking_findings":[],
        "limits":[
            "No empirical superiority, stopping-cost comparison, theorem novelty or full-field SOTA claim.",
            "No power guarantee for fixed finite nodes, fixed coverage, fixed precision, fixed cap, or arbitrary adaptive vanishing gaps.",
            "The implementation and author's rectangle checks reuse outward arithmetic and pi primitives; the normalizer is an existing frozen dependency.",
            "The current prospective candidate remains the previously frozen protocol; this continuous method is not substituted into that study."],
        "new_simulator_observations":0, "prospective_files_read":False,
        "stored_trial_ledgers_read":False, "elapsed_seconds":time.perf_counter()-started}
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"status":result["status"], "output":str(OUTPUT),
                      "independent_reference_cases":len(references), "elapsed_seconds":result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
