"""Independent analytic and symbolic review of the weak-signal theorem.

No producer functions, optimizers, simulated outcomes, or measured performance
are used. The proof review is mathematical, not proof-assistant verification.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import sympy as s

ROOT = Path(__file__).resolve().parents[3]
SUBJECT = ROOT / "results/sota_round5/theory/reflected_pair_weak_signal_theorem.json"
SOURCE = SUBJECT.with_suffix(".py")
OUT = Path(__file__).with_suffix(".json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def symbolic_checks():
    A, B, z, p = s.symbols("A B z p", real=True)
    v = A*(1-A); d = v-B**2; e = B*(1-2*A)
    p0 = A+B*z
    w = p0*(1-p0)/(1-z**2)
    f2 = 1/(p*(1-p))
    f3 = s.diff(f2, p)
    residuals = {
        "d_plus_e_positive_factorization": d+e-(A+B)*(1-A-B),
        "d_minus_e_positive_factorization": d-e-(A-B)*(1-A+B),
        "discriminant_factorization": d**2-e**2-(A**2-B**2)*((1-A)**2-B**2),
        "reciprocal_objective_identity": w-(B**2+(d+e*z)/(1-z**2)),
        "reciprocal_derivative_numerator": s.diff(w, z)-(e*(1+z**2)+2*d*z)/(1-z**2)**2,
        "entropy_third_derivative": f3-(2*p-1)/(p**2*(1-p)**2),
        "capacity_center_delta_squared_coefficient": p*(1-p)*f3/6-(2*p-1)/(6*p*(1-p)),
        "capacity_radius_delta_squared_coefficient": f2/2-1/(2*p*(1-p)),
        "plus_weight_delta_coefficient_denominator12": p*(1-p)*f3/12-(2*p-1)/(12*p*(1-p)),
    }
    result = {}
    for name, residual in residuals.items():
        simplified = s.factor(residual)
        assert simplified == 0
        result[name] = "exact symbolic residual0"
    # Avoid assuming a sign convention for sqrt inside a CAS. Reduce identities
    # modulo D^2=d0^2-e0^2, while positivity D>0,d0>|e0| is proved in prose.
    d0, e0, D = s.symbols("d0 e0 D", real=True)
    z0 = -e0/(d0+D)
    identities = {
        "closed_z0_stationarity": e0*(1+z0**2)+2*d0*z0,
        "curvature_numerator_at_z0": d0+e0*z0-D,
        "closed_hmax_denominator": 2*(d0+e0*z0)/(1-z0**2)-(d0+D),
    }
    modulus = s.Poly(D**2-d0**2+e0**2, D)
    for name, residual in identities.items():
        numerator = s.together(residual).as_numer_denom()[0]
        remainder = s.rem(s.Poly(s.expand(numerator), D), modulus).as_expr()
        assert s.factor(remainder) == 0
        result[name] = "exact polynomial remainder0 modulo D^2-d0^2+e0^2"
    return result


def main():
    subject = json.loads(SUBJECT.read_text(encoding="utf-8"))
    hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in (SUBJECT, SOURCE, Path(__file__).resolve())}
    for path, expected in subject["source_sha256"].items():
        assert sha(ROOT/path) == expected
        hashes[path] = expected
    algebra = symbolic_checks()
    result = {
        "status": "passed_independent_analytic_review_fixed_interior_weak_signal_scope",
        "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        "new_simulation_observations": 0, "producer_functions_imported": False,
        "input_sha256": hashes,
        "accepted_theorem_scope": "For each fixed0<A<1 and|B|<min(A,1-A), there exists a possibly parameter-dependent positive epsilon such that every0<|C|<epsilon has an exactly globally KL-optimal reflected two-angle oracle design, with the stated capacity weights and physical tangent affine null certificate.",
        "blocking_findings": [],
        "proof_checks": [
            {"item": "Compact strict interior margin and analytic extension", "verdict": "passed",
             "argument": "Set tau=min(A-|B|,1-A-|B|)>0. For all z in[-1,1], p0=A+Bz lies in[tau,1-tau]. The entropy f and its derivatives are analytic on a common neighborhood of this compact interval. The symmetric secant has the uniformly convergent even series l=f'(p0)+f'''(p0)delta^2/6+f^(5)(p0)delta^4/120+..., delta=C sqrt(1-z^2). Replacing delta^2 by t(1-z^2) yields a real-analytic extension in(z,t) around[-1,1]x{0}; this formal extension is legitimate even outside the physical t>=0 domain."},
            {"item": "Capacity evenness and normalized R convergence", "verdict": "passed",
             "argument": "The secant identity gives KL(p0+delta||qc)=KL(p0-delta||qc) exactly. Hence R equals the average of these two KLs: [f(p0+delta)+f(p0-delta)]/2-p0 log qc-(1-p0)log(1-qc). Both terms are analytic in delta^2, proving evenness without differentiating sqrt at an endpoint. R(z,0)=0 identically, so R(z,C)/C^2 extends analytically in(z,C^2). Its limiting value is h(z)=(1-z^2)/(2p0(1-p0)). Therefore R/C^2=h+O(C^2) uniformly with any fixed finite number of z derivatives on[-1,1], in particular globalC0 and localC2."},
            {"item": "Unique nondegenerate limiting maximizer", "verdict": "passed",
             "argument": "With v=A(1-A),d=v-B^2,e=B(1-2A), d+e=(A+B)(1-A-B)>0 and d-e=(A-B)(1-A+B)>0; hence d>|e|. For w=1/(2h), w'=N/(1-z^2)^2 where N=e(1+z^2)+2dz. N'=2(d+ez)>0 on[-1,1], while N(-1)<0<N(1). Thus exactly one critical point occurs, z0=-e/(d+sqrt(d^2-e^2)), including z0=0 when e=0. At z0, w''=2sqrt(d^2-e^2)/(1-z0^2)^2>0 and h''=-w''/(2w^2)<0. Since h>0 inside and h=0 at endpoints, this is the unique global maximum."},
            {"item": "Persistence and uniqueness of z*(C)", "verdict": "passed",
             "argument": "Apply the analytic implicit-function theorem to partial_z(R/C^2)=0 at(z0,0), using h''(z0)<0. This gives a unique local branch z*(t)=z0+O(t), t=C^2. Choose a small interval J with h'' uniformly negative. GlobalC1 convergence keeps the derivative nonzero on the compact complement of the interior ofJ; localC2 convergence keeps strict concavity onJ. Thus no additional maximizers occur and z* is the unique interior global maximizer for sufficiently smallC."},
            {"item": "Physical tangent affine null and expansions", "verdict": "passed",
             "argument": "The center expansion is qc=p0+C^2 g(z)+O(C^4), g=(1-z^2)(2p0-1)/(6p0(1-p0)), uniformly with derivatives. Since z*=z0+O(C^2), b*=partial_z qc(z*,C)=B+C^2g'(z0)+O(C^4) and a*=qc(z*,C)-b*z*=A+C^2[g(z0)-z0g'(z0)]+O(C^4). The strict fixed margin min(A,1-A)-|B|>0 therefore implies min(a*,1-a*)>|b*| for smallC. Taking epsilon also below sqrt(min(A,1-A)^2-B^2) ensures the true model is physical."},
            {"item": "Two branches have the same exact stationary contact", "verdict": "passed",
             "argument": "For each fixedC the tangent line qt matches both qc and its first z derivative atz*. The chain rule for Fplus/minus(z)=KL(pplus/minus(z)||qt(z)) therefore gives Fplus(z*)=Fminus(z*)=R(z*,C) and Fplus'(z*)=Fminus'(z*)=R'(z*,C)=0. No derivative of the optimizing point z* is taken when differentiating z at fixedC. There is no missing branch-dependent stationarity condition."},
            {"item": "Local C2 convergence of Fplus/minus/C^2", "verdict": "passed",
             "argument": "On any compact J inside(-1,1), s=sqrt(1-z^2) is analytic. Put ell(z)=g(z0)+g'(z0)(z-z0), so qt=p0+C^2 ell+O(C^4) in every local derivative norm. Taylor expansion of the Bernoulli Bregman divergence gives Fplus/minus/C^2=h +/- C*s*(g-ell)/(p0(1-p0))+O_J(C^2), including two z derivatives. Thus the normalized branches converge in the function-space C2 norm, usually at O(|C|), not O(C^2). The producer correctly distinguishes these notions. Since g-ell and its first derivative vanish atz0, the leading branch asymmetry does not contradict the shared stationary contact."},
            {"item": "Uniform endpoint-safe gap and global dominance", "verdict": "passed",
             "argument": "Globally s is bounded even though its endpoint derivatives diverge. The uniform interior margin and pplus/minus-qt=+/-Cs+O(C^2) give Fplus/minus/C^2=h+O(|C|) in C0([-1,1]); no globalC2 assertion for these branches is needed. Choose J compact aroundz0 with h''<=-lambda<0 and z* in its interior. LocalC2 convergence gives strict concavity of both branches onJ, making their common stationary point their unique local maximum. On K=[-1,1] minus interior(J), compactness and uniqueness give gamma=h(z0)-max_K h>0. UniformC0 convergence and R(z*,C)/C^2->h(z0) preserve this gap outsideJ. Hence every angle satisfies KL(p(x)||qt(cos x))<=R(z*,C)."},
            {"item": "Oracle lower/upper sandwich", "verdict": "passed",
             "argument": "At any reflection pair the affine C_null=0 family has a common rate at its two angles and contains every constant rate in[0,1]. The weighted KL minimizer is their weighted mean. Choosing omega=(qc-pminus)/(pplus-pminus) makes that mean qc and the two KL values equalR, furnishing V>=max_z R. The single physical tangent null supplies the pointwise upper bound for every sampling distribution, giving V<=R(z*,C). No unjustified exchange of infimum and supremum is needed."},
            {"item": "Opposite-sign null enlargement", "verdict": "passed",
             "argument": "For C>0 and the reflection pair with positive sine labelledplus, any physical C_null<=0 has qplus<=qminus. The strictly convex weighted Bernoulli-KL projection of pplus>pminus onto this order cone pools both rates at the weighted meanqc. The feasible constant physical null attains the pooled minimumR. The tangent null has C_null=0 and still supplies the universal upper bound. Thus the extension is valid; negativeC follows by relabelling."},
            {"item": "Closed C^2 value and plus-weight factor12", "verdict": "passed",
             "argument": "The exact discriminant identity is d^2-e^2=(A^2-B^2)((1-A)^2-B^2). Since w(z0)=(v+B^2+sqrt(d^2-e^2))/2, h(z0)=1/(v+B^2+sqrt(d^2-e^2)), giving the stated V=C^2 h(z0)+O(C^4). Also qc-p0=delta^2(2p0-1)/(6p0(1-p0))+O(delta^4); dividing by2delta in omega=1/2+(qc-p0)/(2delta) gives the denominator12. Replacing z* byz0 changes this linear-inC term only byO(C^3). There is no missing factor2."},
        ],
        "symbolic_identity_checks": algebra,
        "nonblocking_clarifications": [
            "When J is closed, write sup outsideJ or max over the complement of interior(J); the literal complement of a closedJ need not be compact. The gap argument above supplies the precise compact formulation.",
            "The derivative exclusion in the uniqueness argument uses uniformC1 convergence, already available from the analytic expansion; uniform convergence of function values alone would not suffice.",
            "The proof supplies an existential parameter-dependent neighborhood, not a usable certified numerical epsilon. High-precision stationarity checks at selected finiteC values cannot certify that those values lie in that neighborhood.",
        ],
        "remaining_open_scope": [
            "The unrestricted all-physical-C reflected-pair conjecture remains unresolved.",
            "No uniformity is established when A,B move toward the physical boundary as C tends to0.",
            "No finite-shot saving, paid parameter-learning policy, new-circuit performance, novelty, SOTA or3x claim follows from this oracle theorem.",
            "Other globally exact special cases and the proposed global curvature reduction are contextual claims; this review's acceptance covers the fixed-interior weak-signal theorem and its displayed asymptotics.",
        ],
        "review_status_limit": "Independent mathematical argument review and exact symbolic identities; not formal proof-assistant verification or a numerical certificate of a particular finiteC neighborhood.",
    }
    assert all(sha(ROOT/path) == expected for path, expected in hashes.items())
    with OUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False); handle.write("\n")
    print(json.dumps({"output": str(OUT), "sha256": sha(OUT), "status": result["status"],
                      "symbolic_checks": len(algebra), "blocking_findings": result["blocking_findings"]}))


if __name__ == "__main__":
    main()
