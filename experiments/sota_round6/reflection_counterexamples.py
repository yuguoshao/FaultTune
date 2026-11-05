"""Preregistered deterministic stress search for the all-C reflection claim.

Float scans are screening only. Entropy centers use the corrected secant and
high-precision checks use closed entropy expressions, never fixed quadrature.
All parameter rows and failures are retained. No Monte Carlo data are created.
"""
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import mpmath as mp
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/sota_round5"))
import continuous_chernoff_refinement as refined

DIRECTORY = ROOT / "results/sota_round6/reflection_counterexamples"
PLAN = DIRECTORY / "search_plan.json"
ROWS = DIRECTORY / "all_parameter_rows.jsonl"
SUMMARY = DIRECTORY / "summary.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write("\n")


def registry():
    points = []
    for A in ("1/1000000", "1/10000", "1/1000", "1/100", "1/10", "1/4", "49/100", "1/2"):
        for rho in ("3/4", "99/100", "9999/10000", "99999999/100000000", "999999999999/1000000000000"):
            for u in ("1/100", "1/5", "1/2", "3/5", "4/5", "9/10", "99/100", "9999/10000"):
                points.append({"kind": "interior_grid", "A": A, "rho": rho, "B_over_R": u})
    for A in ("1/1000000", "1/100", "49/100"):
        for v in ("1/10", "1/100", "1/1000", "1/100000"):
            for power in (2, 4):
                points.append({"kind": "nonuniform_weak_boundary", "A": A,
                               "rho": str(1-Fraction(v)**power), "C_over_R": v, "rho_gap_power": power})
    for A in ("1/100", "1/4", "49/100"):
        for u in ("1/5", "3/5", "9/10"):
            points.append({"kind": "pure_boundary", "A": A, "rho": "1", "B_over_R": u})
    assert len(points) == 353
    return [{"parameter_id": i, **point} for i, point in enumerate(points)]


def rational_mp(value):
    value = Fraction(value)
    return mp.mpf(value.numerator)/value.denominator


def model(point):
    A, rho = rational_mp(point["A"]), rational_mp(point["rho"])
    radius = min(A, 1-A)*rho
    if "B_over_R" in point:
        u = rational_mp(point["B_over_R"])
        return A, radius*u, radius*mp.sqrt(1-u*u)
    v = rational_mp(point["C_over_R"])
    return A, radius*mp.sqrt(1-v*v), radius*v


def plan():
    sources = (Path(__file__).resolve(), ROOT/"experiments/sota_round5/continuous_chernoff_refinement.py",
               ROOT/"experiments/sota_round5/continuous_chernoff_probe.py",
               ROOT/"results/sota_round5/theory/reflected_pair_global_gap_checklist.json",
               ROOT/"experiments/structured_validation/design_policy.py")
    return {"status": "fixed_before_round6_counterexample_search", "created_utc": datetime.now(timezone.utc).isoformat(),
            "registry": registry(), "parameter_count": 353,
            "parameter_semantics": "A,rho and normalized B or C are exact rationals; the other coordinate is the positive algebraic square root. Float-rounded models only screen; high-precision reconstruction uses these exact expressions.",
            "pair_search": "257 uniform angles plus both endpoint distances10^-1,...10^-12; evaluate all. Refine every sampled local maximum if at most16, otherwise refine the16 highest and explicitly retain the incomplete-search flag. Bounded scipy search tolerance1e-12. No sampled maximum is called a global certificate.",
            "angle_witness_grid": "127 interior uniform angles j*pi/128 plus endpoint distances10^-2,...10^-6 and the selected pair angle.",
            "curvature": "At best sampled/refined radius, evaluate both ell_R(pplus) curvature and ell_R(1-pminus) curvature; the latter is negative of U''. Record normalized sign [(ell''/ell')*(p')^2*s^3/C]-1, which has the same sign as L'' or -U''.",
            "high_precision_trigger": "100-digit reconstruction for every nonuniform/boundary row, every A<=1e-4, every rho>=1-1e-8, or any normalized curvature>1e-6, tangent relative excess>1e-7, or nonpositive tangent margin. Also reconstruct any float-screen failure. Check the strongest witness of each type; retain numerical failure rather than omitting it.",
            "high_precision_pair_rule": "Use the screened maximum basin, or the same257-plus-log mesh in mp if screening failed; seek an R' sign-changing bracket and bisect at most150 times. This only certifies numerical stationarity, not global optimality of the reflected-pair radius.",
            "three_support_screen": "Run the existing129-arm rational-circle classical oracle at every16th interior parameter_id, and whenever high-precision tangent excess>1e-8 or null margin<0. Save support, weights, null and solver status. Any apparent lower bound exceeding pair radius needs separate high-precision/interval global-pair verification before being called a counterexample.",
            "evidence_separation": ["curvature sufficient-condition violation", "tangent-null pointwise all-angle violation", "genuine multi-support value above globally certified best pair"],
            "retention": "Every registered parameter row, sampled scan and all failures are appended. No selective deletion, no threshold or registry changes after results.",
            "no_proof_from_mesh": True, "new_simulation_observations": 0,
            "source_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in sources}}


def finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def entropy(p):
    return p*mp.log(p)+(1-p)*mp.log1p(-p) if p not in (0, 1) else mp.mpf(0)


def mp_at(theta, angle):
    A, B, C = theta
    middle, delta = A+B*mp.cos(angle), C*mp.sin(angle)
    if not delta:
        return mp.mpf(0), middle
    pm, pp = middle-delta, middle+delta
    if not (0 <= pm < pp <= 1):
        raise ArithmeticError("Nonphysical high-precision endpoint")
    secant = (entropy(pp)-entropy(pm))/(pp-pm)
    center = 1/(1+mp.exp(-secant))
    # The average avoids depending on either branch alone when checking equality.
    radius = (entropy(pp)+entropy(pm))/2-middle*mp.log(center)-(1-middle)*mp.log1p(-center)
    return radius, center


def float_at(theta, angle):
    A, B, C = theta
    middle, delta = A+B*math.cos(angle), C*math.sin(angle)
    if not delta:
        return {"radius": 0., "center": middle}
    return refined.radius(middle-delta, middle+delta)


def tangent_float(theta, angle, center):
    A, B, C = theta
    z, sine = math.cos(angle), math.sin(angle)
    delta, derivative = C*sine, -C*z/sine
    middle = A+B*z
    pm, pp = middle-delta, middle+delta
    if 2*delta < .01*min(pm, 1-pp):
        probabilities = middle+delta*refined.initial.GL_X
        lprime = float(refined.initial.GL_W.dot((B+derivative*refined.initial.GL_X)/(probabilities*(1-probabilities)))/2)
    else:
        secant = float(logit(center))
        lprime = (float(logit(pp))*(B+derivative)-float(logit(pm))*(B-derivative))/(2*delta)-secant*derivative/delta
    slope = center*(1-center)*lprime
    return center-slope*z, slope


def float_lower_curvature(p, derivative, sine, C, radius):
    if not 0 < p < 1:
        raise ArithmeticError("Boundary rate is not twice differentiable in this screen")
    log_odds = float(logit(p))
    f = p*math.log(p)+(1-p)*math.log1p(-p)
    def difference(v):
        y = log_odds-v
        return f-p*y+float(np.logaddexp(0., y))-radius
    high = 2+2*radius/p
    for _ in range(20):
        if difference(high) > 0:
            break
        high *= 2
    v = brentq(difference, 0., high, xtol=1e-12, rtol=1e-13, maxiter=150)
    q = float(expit(log_odds-v)); gap = p-q; P = p*(1-p)
    if v <= 0 or gap <= 0:
        raise ArithmeticError("Unresolved weak inverse-KL root")
    ratio = v+(P*v-gap)**2/(P*v*gap*gap)
    return ratio*derivative*derivative*sine**3/C-1


def float_screen(theta):
    mesh = sorted(set(np.linspace(1e-12, math.pi-1e-12, 257).tolist()+[10.**(-k) for k in range(1, 13)]+[math.pi-10.**(-k) for k in range(1, 13)]))
    radii, failures = [], []
    for x in mesh:
        try:
            radii.append(float_at(theta, x)["radius"])
        except Exception as exc:
            radii.append(None); failures.append({"angle": x, "error": str(exc)})
    peaks = [i for i in range(1, len(mesh)-1) if radii[i] is not None and
             (radii[i-1] is None or radii[i] >= radii[i-1]) and (radii[i+1] is None or radii[i] >= radii[i+1])]
    if not peaks:
        raise ArithmeticError("No screened radius maximum; retained high-precision fallback required")
    selected = sorted(peaks, key=lambda i: (-radii[i], i))[:16]
    solutions = []
    for i in selected:
        try:
            fit = minimize_scalar(lambda x: -float_at(theta, float(x))["radius"], bounds=(mesh[i-1], mesh[i+1]),
                                  method="bounded", options={"xatol": 1e-12, "maxiter": 150})
            record = float_at(theta, float(fit.x))
            solutions.append({"angle": float(fit.x), "radius": record["radius"], "center": record["center"],
                              "bracket": [mesh[i-1], mesh[i+1]], "success": bool(fit.success)})
        except Exception as exc:
            failures.append({"radius_basin": i, "error": str(exc)})
    if not solutions:
        raise ArithmeticError("All radius refinements failed")
    best = max(solutions, key=lambda row: row["radius"])
    aa, bb = tangent_float(theta, best["angle"], best["center"])
    A, B, C = theta
    angle_grid = sorted(set([j*math.pi/128 for j in range(1, 128)]+[10.**(-k) for k in range(2, 7)]+
                            [math.pi-10.**(-k) for k in range(2, 7)]+[best["angle"]]))
    witnesses = []
    for x in angle_grid:
        z, sine = math.cos(x), math.sin(x)
        pm, pp = A+B*z-C*sine, A+B*z+C*sine
        row = {"angle": x}
        try:
            row["L_curvature_scaled"] = finite_float(float_lower_curvature(pp, B-C*z/sine, sine, C, best["radius"]))
        except Exception as exc:
            row["L_curvature_error"] = str(exc)
        try:
            row["minus_U_curvature_scaled"] = finite_float(float_lower_curvature(1-pm, -B-C*z/sine, sine, C, best["radius"]))
        except Exception as exc:
            row["minus_U_curvature_error"] = str(exc)
        try:
            rates = refined.initial.design.bernoulli_kl(np.asarray([pm, pp]), aa+bb*z)
            row["tangent_relative_excess"] = float(max(rates)/best["radius"]-1)
        except Exception as exc:
            row["tangent_error"] = str(exc)
        witnesses.append(row)
    best_witness = {}
    for key in ("L_curvature_scaled", "minus_U_curvature_scaled", "tangent_relative_excess"):
        available = [row for row in witnesses if isinstance(row.get(key), (int, float))]
        if available:
            best_witness[key] = max(available, key=lambda row: row[key])
    return {"best_pair": best, "all_radius_candidates": solutions, "sampled_peak_count": len(peaks),
            "all_peak_basins_refined": len(peaks) <= 16, "radius_mesh_angles": mesh, "radius_mesh_values": radii,
            "null_candidate": [aa, bb], "null_physical_margin": min(aa, 1-aa)-abs(bb),
            "angle_witnesses": witnesses, "best_witnesses": best_witness, "failures": failures}


def mp_lower_curvature(p, derivative, sine, C, radius):
    if not 0 < p < 1:
        raise ArithmeticError("Nondifferentiable boundary inverse endpoint")
    odds = mp.log(p/(1-p))
    def difference(v):
        y = odds-v
        return entropy(p)-p*y+mp.log1p(mp.exp(y))-radius
    lo, hi = mp.mpf(0), 2+2*radius/p
    while difference(hi) <= 0:
        hi *= 2
    for _ in range(260):
        middle = (lo+hi)/2
        if difference(middle) > 0:
            hi = middle
        else:
            lo = middle
    v = (lo+hi)/2; q = 1/(1+mp.exp(-(odds-v)))
    gap, P, Q = p-q, p*(1-p), q*(1-q)
    K = Q*v/gap
    stable_ratio = v+(P*v-gap)**2/(P*v*gap*gap)
    original_second = (1/P-2*K/Q+(p/q**2+(1-p)/(1-q)**2)*K*K)*Q/gap
    algebra_error = abs(original_second/K-stable_ratio)/max(1, abs(stable_ratio))
    if algebra_error > mp.mpf("1e-55"):
        raise ArithmeticError("Independent inverse-curvature expressions disagree")
    scaled = stable_ratio*derivative**2*sine**3/C-1
    return {"scaled_sign": mp.nstr(scaled, 70), "ell_second": mp.nstr(original_second, 70),
            "implicit_root_relative_residual": mp.nstr(abs(difference(v))/max(radius, mp.mpf("1e-100")), 12),
            "formula_relative_disagreement": mp.nstr(algebra_error, 12)}


def high_precision(point, screening):
    mp.mp.dps = 100
    theta = model(point)
    if screening:
        left, right = map(mp.mpf, screening["best_pair"]["bracket"])
    else:
        mesh = sorted(set([mp.pi*j/256 for j in range(1, 256)]+[mp.mpf(10)**(-k) for k in range(1, 13)]+
                          [mp.pi-mp.mpf(10)**(-k) for k in range(1, 13)]))
        values = [mp_at(theta, x)[0] for x in mesh]
        i = max(range(1, len(mesh)-1), key=lambda j: values[j])
        left, right = mesh[i-1], mesh[i+1]
    derivative = lambda x: mp.diff(lambda t: mp_at(theta, t)[0], x)
    if not derivative(left) > 0 > derivative(right):
        raise ArithmeticError("High-precision local maximum bracket lacks derivative sign change")
    for _ in range(150):
        middle = (left+right)/2
        if derivative(middle) > 0:
            left = middle
        else:
            right = middle
    angle = (left+right)/2
    R, center = mp_at(theta, angle)
    z = mp.cos(angle)
    slope = -mp.diff(lambda x: mp_at(theta, x)[1], angle)/mp.sin(angle)
    intercept = center-slope*z
    margin = min(intercept, 1-intercept)-abs(slope)
    result = {"precision_digits": 100, "angle": mp.nstr(angle, 75), "local_angle_bracket": [mp.nstr(left, 75), mp.nstr(right, 75)],
              "radius": mp.nstr(R, 75), "center": mp.nstr(center, 75), "null_intercept": mp.nstr(intercept, 75),
              "null_slope": mp.nstr(slope, 75), "null_physical_margin": mp.nstr(margin, 75),
              "stationarity_residual_relative": mp.nstr(abs(derivative(angle))/R, 12),
              "pair_globality": "Best sampled basin and high-precision stationary root; NOT a certified global maximum"}
    witnesses = screening["best_witnesses"] if screening else {}
    A, B, C = theta
    result["witnesses"] = {}
    for key, row in witnesses.items():
        x = mp.mpf(row["angle"]); zz, ss = mp.cos(x), mp.sin(x)
        pm, pp = A+B*zz-C*ss, A+B*zz+C*ss
        try:
            if key == "L_curvature_scaled":
                checked = mp_lower_curvature(pp, B-C*zz/ss, ss, C, R)
            elif key == "minus_U_curvature_scaled":
                checked = mp_lower_curvature(1-pm, -B-C*zz/ss, ss, C, R)
            else:
                q = intercept+slope*zz
                def kl(p):
                    return entropy(p)-p*mp.log(q)-(1-p)*mp.log1p(-q)
                excess = max(kl(pm), kl(pp))/R-1
                checked = {"relative_excess": mp.nstr(excess, 75)}
            result["witnesses"][key] = {"angle_exact_binary64": row["angle"], **checked}
        except Exception as exc:
            result["witnesses"][key] = {"error": str(exc)}
    return result


def finite_grid_screen(theta, pair_radius):
    design = refined.initial.design
    result = design.design_from_fit(theta, grid=design.rational_grid(129, max_denominator=1024))
    return {key: result[key] for key in ("support", "null_candidate", "solver_status", "epigraph_success", "lp_success", "info_lower", "info_upper", "relative_gap")} | {"lower_over_best_numeric_pair": result["info_lower"]/pair_radius}


def main():
    started = time.perf_counter()
    frozen = plan()
    write_new(PLAN, frozen)
    outcomes = []
    with ROWS.open("x", encoding="utf-8") as stream:
        for point in frozen["registry"]:
            row = {"parameter": point}
            mp.mp.dps = 100
            exact_theta = model(point)
            theta = tuple(map(float, exact_theta))
            row["float_screen_theta"] = theta
            screen = None
            try:
                screen = float_screen(theta); row["float_screen"] = screen
            except Exception as exc:
                row["float_screen_failure"] = str(exc)
            force = point["kind"] != "interior_grid" or Fraction(point["A"]) <= Fraction(1, 10000) or Fraction(point["rho"]) >= Fraction(99999999, 100000000)
            if screen:
                best = screen["best_witnesses"]
                force = force or screen["null_physical_margin"] <= 0 or any(
                    best.get(key, {}).get(key, -1.) > threshold for key, threshold in (
                        ("L_curvature_scaled", 1e-6), ("minus_U_curvature_scaled", 1e-6), ("tangent_relative_excess", 1e-7)))
            else:
                force = True
            if force:
                try:
                    row["high_precision"] = high_precision(point, screen)
                except Exception as exc:
                    row["high_precision_failure"] = str(exc)
            trigger_grid = point["kind"] == "interior_grid" and point["parameter_id"] % 16 == 0
            hp = row.get("high_precision", {})
            excess = hp.get("witnesses", {}).get("tangent_relative_excess", {}).get("relative_excess", "0")
            trigger_grid = trigger_grid or mp.mpf(excess) > mp.mpf("1e-8") or mp.mpf(hp.get("null_physical_margin", "1")) < 0
            if trigger_grid and screen:
                try:
                    row["finite_grid_design"] = finite_grid_screen(theta, screen["best_pair"]["radius"])
                except Exception as exc:
                    row["finite_grid_failure"] = str(exc)
            row["evidence_status"] = "deterministic_screen_and_high_precision_diagnostic_not_global_certificate"
            stream.write(json.dumps(row, allow_nan=False)+"\n"); stream.flush(); outcomes.append(row)
            if len(outcomes) % 16 == 0:
                print(json.dumps({"complete": len(outcomes), "total": len(frozen["registry"]), "elapsed_seconds": time.perf_counter()-started}), flush=True)
    witnesses, tangent = [], []
    for row in outcomes:
        for key, value in row.get("high_precision", {}).get("witnesses", {}).items():
            if "scaled_sign" in value and mp.mpf(value["scaled_sign"]) > mp.mpf("1e-20"):
                witnesses.append({"parameter_id": row["parameter"]["parameter_id"], "type": key, **value})
            if "relative_excess" in value and mp.mpf(value["relative_excess"]) > mp.mpf("1e-15"):
                tangent.append({"parameter_id": row["parameter"]["parameter_id"], **value})
    result = {"status": "complete_preregistered_deterministic_boundary_stress_screen", "parameter_count": len(outcomes),
              "high_precision_curvature_candidates": witnesses, "high_precision_tangent_candidates": tangent,
              "float_screen_failures": sum("float_screen_failure" in row for row in outcomes),
              "high_precision_failures": sum("high_precision_failure" in row for row in outcomes),
              "finite_grid_attempts": sum("finite_grid_design" in row or "finite_grid_failure" in row for row in outcomes),
              "actual_all_C_counterexample_certified": False,
              "proof_boundary": "A curvature witness at a numerical best-pair radius still needs a certified R_star interval or a justified monotonic-radius argument. A tangent violation does not by itself supply a primal multi-support lower bound above globally certified reflected capacity. All points retained; no finite mesh proves the conjecture.",
              "source_sha256": frozen["source_sha256"], "plan_sha256": sha(PLAN), "rows_sha256": sha(ROWS),
              "elapsed_seconds": time.perf_counter()-started, "new_simulation_observations": 0}
    assert all(sha(ROOT/path) == value for path, value in frozen["source_sha256"].items())
    write_new(SUMMARY, result)
    print(json.dumps({key: result[key] for key in ("status", "parameter_count", "float_screen_failures", "high_precision_failures", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
