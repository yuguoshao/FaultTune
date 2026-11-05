"""Record checked sources, executable adaptation, and unresolved SOTA scope."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DESTINATION = ROOT/"results/sota_round2/literature"


def reference(name):
    path = DESTINATION/name
    return {"path":path.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    dkm_source = ROOT/"experiments/sota_round2/literature/dkm_baseline.py"
    source_sha = hashlib.sha256(dkm_source.read_bytes()).hexdigest()
    checks = json.loads((DESTINATION/"dkm_checks.json").read_text(encoding="utf-8"))
    assert source_sha == checks["source_sha256"] == "e8799facf1494dcf6355f2f2e6069be350f35f788a70381201f55c53e5479c9d"
    result = {
        "status":"targeted primary-source verification and executable baseline completed; independent peer review separate",
        "checked_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"Named structured pure-exploration families, not an exhaustive 2026 SOTA survey or a superiority claim.",
        "selected_baseline":"DKM Algorithm 1 / Section 3.1 numerical finite-grid implementation with separately selected valid stopping calibration",
        "implementation":{"path":dkm_source.relative_to(ROOT).as_posix(),"sha256":source_sha,
                          "checks":reference("dkm_checks.json"),
                          "initial_oracle_issue_preserved":reference("dkm_checks_initial_oracle_issue.json")},
        "sources":{
            "dkm":{"title":"Non-Asymptotic Pure Exploration by Solving Games", "authors":["Remy Degenne","Wouter M. Koolen","Pierre Menard"],
                   "publication":"NeurIPS 2019", "url":"https://proceedings.neurips.cc/paper_files/paper/2019/file/8d1de7457fa769ece8d93a13a59c8552-Paper.pdf",
                   "main_pdf":reference("dkm_neurips.pdf"),"appendix_pdf":reference("dkm_arxiv.pdf"),
                   "inspected":"Sections 2.1-2.3, Algorithm 1, Section 3.1, Section 4; Algorithm 1 page rendered and visually checked",
                   "main_claim":"General finite-arm structured one-parameter exponential-family identification, with optimistic game-based allocation and finite-confidence sample-complexity analysis under assumptions.",
                   "important_experiment_limit":"Section 4 uses beta(t,delta)=log((1+log(t))/delta), f(t)=log(t), expressly outside the paper's proved calibration. Our strict-risk comparisons do not silently adopt these choices."},
            "kk21":{"title":"Mixture Martingales Revisited with Applications to Sequential Tests and Confidence Intervals", "authors":["Emilie Kaufmann","Wouter M. Koolen"],
                    "publication":"JMLR 22, 2021", "url":"https://jmlr.org/papers/v22/18-798.html", "pdf":reference("kk_arxiv.pdf"),
                    "inspected":"Theorem 7, equations (9)-(10), Remark 6, discussion of Gaussian-only refinements; theorem page rendered and visually checked",
                    "applicable_result":"Two-sided general-exponential-family threshold. The smaller Gaussian-specific calibration is not used for Bernoulli data."},
            "adahedge":{"title":"Follow the Leader If You Can, Hedge If You Must", "publication":"JMLR 15, 2014",
                        "url":"https://jmlr.org/papers/volume15/rooij14a/rooij14a.pdf", "pdf":reference("adahedge_jmlr.pdf"),
                        "inspected":"Section 2.2, equations (8)-(9): eta=log(K)/past cumulative mixability gap, uniform-initial exponential weights, mix loss."},
            "fwsp":{"title":"Pure Exploration via Frank-Wolfe Self-Play", "authors":["Xinyu Liu","Chao Qin","Wei You"],
                      "url":"https://arxiv.org/html/2509.19901v1", "publication_status":"arXiv v1 submitted 2025-09-24; current abstract page lists v1 only",
                      "html":reference("fw_latest.html"),"abstract":reference("fw_abs.html"),
                      "inspected":"Assumptions 1-4, Theorem 1, Section 6 / Algorithm 1, Appendix B",
                      "limits":["Theorem 1 covers unstructured exponential-family BAI and Gaussian linear-bandit BAI, not a proved Bernoulli linear-mean sign learner.",
                                "Section 6 learning algorithm assumes a structure-respecting posterior update/sample oracle and leaves its theoretical analysis for future work.",
                                "No complete delta-correct stopping procedure for this project's Bernoulli sign task was established by this review."],
                      "task_specific_inference":"For a fixed sign answer, the opposite-sign physical set is one convex component. The skeptic's simplex over alternative components is therefore a singleton. This inference does not supply the missing learning/stopping guarantee."}},
        "public_code":{
            "official_repository":"https://bitbucket.org/wmkoolen/tidnabbil", "pinned_commit":"fdb307afcee834cee17fa849f3f6ca7e4d518a64",
            "language":"Julia", "paper_readme_environment":"Julia 1.1.0", "repository_updated_on":"2020-10-26T12:59:45.636297+00:00",
            "provenance":"Koolen's software page names tidnabbil as successor to purex_games, and his DKM publication entry links it as code. Repository purex_games_paper/README explicitly identifies DKM 2019.",
            "author_software_page":reference("koolen_software.html"),"author_publications":reference("koolen_publications.html"),
            "repository_metadata":reference("tidnabbil_repo.json"),"readme":reference("tidnabbil_purex_README.txt"),
            "inspected_files":[reference(x) for x in ("tidnabbil_samplingrules.jl","tidnabbil_tracking.jl","tidnabbil_thresholds.jl","tidnabbil_runit.jl","tidnabbil_peps.jl")],
            "execution_status":"Public source inspected only; Julia experiments not executed and no downloaded external code was executed.",
            "license_status":"No license file was present in inspected root listing; no claim that this is a permissively licensed vendored implementation.",
            "mapping_differences":["Our tracking follows the paper's ratio argmin N/sum(w). Official CTracking uses argmin N-sum(w).",
                                   "Official optimistic_gradient uses f(t)=log(t); ours follows the larger theorem setting from DKM Section 2.3.",
                                   "Official thresholds.jl exposes GK16 log((1+log(t))/delta); our KK21 certificate is a different, formally calibrated Bernoulli threshold.",
                                   "Our structured Bernoulli physical-cone oracle is a new task adapter; the official examples implement other identification structures."],
            "fwsp_code_status":"No author-linked executable FWSP repository located in the inspected paper/abstract pages. Generic GitHub searches were inconclusive; this is not a claim that no code exists."},
        "model_and_algorithm":{
            "model":"p_a=A+B*cos(x_a)+C*sin(x_a), Omega={0<=A<=1, hypot(B,C)<=min(A,1-A)}",
            "answers":"sign(C); C=0 remains a null/abstention case and is outside unique-answer finite-termination claims",
            "default_grid":"Six exact rational arms: cos=-3/5,0,3/5 with both signs of sin=4/5,1,4/5. Grid fixed before any observations; caller may register another grid.",
            "paid_state":"One paid initial observation per arm, then per-arm integer successes S_a and totals N_a. No true A,B,C passed to allocation or certification.",
            "sampling_estimate":"Clip empirical means only for design to [eta,1-eta]; confidence-constrained physical projection selects the answer. eta=1e-6 is an explicit numerical design margin, not a restriction of the stopping null.",
            "confidence_intervals":"I_a={q in [eta,1-eta]:N_a*kl(mu_hat_a,q)<=f(n)}",
            "theoretical_exploration":"x=(1+a)(1+b)*log(n), f(n)=x+log(x)+1/2, a=b=0.1>0; DKM Section 2.3 with n=t-1",
            "best_response":"lambda=argmin over opposite-sign Omega_eta of sum_a w_a kl(mu_hat_a,p_a(lambda)); numerical convex oracle with gap<=1e-5 or explicit failure",
            "optimism":"U_a=max(f(n)/N_a, kl(lower_a,p_a(lambda)), kl(upper_a,p_a(lambda)))",
            "learner":"Separate AdaHedge for each selected sign; loss vector -U. eta_H=log(K)/Delta_previous; Delta accumulates expected loss minus mix loss.",
            "tracking":"Initialize cumulative target to paid initial counts; add current w; pull argmin_a N_a/cumulative_target_a",
            "order":"Project/select sign and test on the SAME current counts; if continuing, execute the next planned paid draw. Final unused allocation computation is explicitly logged as unexecuted.",
            "batching":"Default batch_size=1 and check_every=1. Batch>1 is labelled explicit_batch_adaptation and has no inherited DKM efficiency theorem."},
        "stopping_rules":{
            "shared_numerical_certificate":"Exact rational query coordinates; certified null log-likelihood upper from the frozen concave-tangent/support primitive. Optimization proposals never authorize a stop.",
            "kk21":{"statistic":"Z_i=sum_a sup_q log L_a(q) - sup_{theta in Omega, i*C<=0} log L(theta)",
                    "threshold":"sum_a 3*log(1+log N_a)+K*C_exp(log(1/delta)/K)",
                    "calibration":"h(u)=u-log u; C_exp(x)=2*h_tilde_(3/2)((h_inverse(1+x)+log(2*zeta(2)))/2); exact branches as KK21 equations (9)-(10)",
                    "rounding":"Machin alternating-series enclosure for pi/zeta(2), monotone verified h-inverse bisection, outward Decimal logs/exponentials; strict lower-statistic > upper-threshold",
                    "positive_part_note":"KK21 controls sum [N_a kl - correction_a]_+. Subtracting all nonnegative corrections from the total KL is no larger, giving the implemented sufficient condition.",
                    "risk":"Any selected false sign implies the same all-arm true-mean concentration event. Therefore delta is shared across both answers, including C=0, rather than silently union-bounding two delta budgets."},
            "kt_safe":{"status":"Explicit certificate adaptation, not the plotted DKM threshold",
                       "numerator":"Q=product_a B(S_a+1/2,N_a-S_a+1/2)/B(1/2,1/2)",
                       "statistic":"log Q - sup_{theta in Omega,i*C<=0} log L(theta)",
                       "threshold":"log(1/delta)",
                       "risk":"For predictable selection among the fixed arms, Q/L_true is a common nonnegative likelihood-ratio martingale (supermartingale at support boundaries). Every false-sign crossing is dominated by this same process; Ville controls their union at delta."},
            "boundary_extension":"For finite fixed-arm histories, event probabilities are continuous in the true Bernoulli means. Approximate physical-boundary nulls by interior physical nulls, then take increasing finite horizons. Efficiency at those boundaries is not asserted."},
        "validation_and_runtime":{
            "tests":"16 independent 100-digit calibration/threshold checks; 49 rounds of 90-digit AdaHedge checks; per-step optimism/causality/tracking and RNG replay; low-probability oracle-gap checks; 4096-shot null timing.",
            "fixed_initial_issue":"An early SLSQP success flag hid huge oracle gaps; its check record is preserved. Correct convex objective extension, nondegenerate cone constraints, restarts and feasible line search fixed the tested cases. Final checks enforce gap<=1e-5 and no failed responses.",
            "measured_one_shot_4096_null_seconds":18.2858845,
            "representative_per_shot_milliseconds":"About 4.4-5.3 ms excluding tiny stopped trajectories; measurements include per-shot stop screens and no batching.",
            "cap4096_estimate_seconds":"Approximately 18-22 seconds for inspected representative regimes on this machine; difficult new oracles can take longer."},
        "comparison_requirements":["Retain closed_hedge as the strongest already implemented comparison; previous failure to beat it remains unchanged.",
                                   "Add DKM with a named grid, allocation rule and stopping calibration. A gain caused by a more conservative K-dependent threshold is not evidence of practical full-field SOTA superiority.",
                                   "To separate design from evidence, compare allocation strategies with the same fixed grid and same valid certificate.",
                                   "Include all paid initialization, exploration, errors and cap abstentions. Record CPU time separately from shot cost.",
                                   "Use fresh validation seeds after development and predeclare the primary comparison. These engineering probes are development data."],
        "remaining_limits":["This is a paper-based numerical adapter, not a claim of bit-for-bit reproduction of the official Julia experiments.",
                            "Finite grid coverage must be quantified before conclusions about unrestricted continuous-angle designs.",
                            "Exact-oracle and compact-interior assumptions required for DKM sample-efficiency results are not automatically inherited by SLSQP or batch adaptations.",
                            "Neither the current implementation checks nor the reviewed literature establish that FaultTune has exceeded SOTA."]}
    (DESTINATION/"baseline_literature_review.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"implementation_sha256":source_sha,
                      "review_sha256":hashlib.sha256((DESTINATION/"baseline_literature_review.json").read_bytes()).hexdigest()},indent=2))


if __name__ == "__main__":
    main()
