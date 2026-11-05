"""Independent high-precision and analytic checks of the DKM certificate API.

These compare the frozen candidate implementation against KK21's mathematical
formula evaluated independently with mpmath, and against an analytic two-arm
physical null likelihood. No trial simulation or new validation seed is used.
"""
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

import mpmath as mp


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"experiments/sota_round2/literature"))
import dkm_baseline as dkm

OUT = ROOT/"results/sota_round2/theory/dkm_numeric_checks.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inverse_h(value):
    left, right = mp.mpf(1), max(mp.mpf(2), 2*value)
    while right-mp.log(right) < value:
        right *= 2
    for unused in range(400):
        middle = (left+right)/2
        if middle-mp.log(middle) < value:
            left = middle
        else:
            right = middle
    return (left+right)/2


def reference_calibration(delta, arms):
    x = -mp.log(delta)/arms
    argument = (inverse_h(1+x)+mp.log(2*mp.zeta(2)))/2
    z = mp.mpf(3)/2
    cutoff = 1/mp.log(z)-mp.log(1/mp.log(z))
    if argument >= cutoff:
        root = inverse_h(argument)
        value = root*mp.exp(1/root)
    else:
        value = z*(argument-mp.log(mp.log(z)))
    return 2*arms*value


def run_checks():
    mp.mp.dps = 100
    calibrations = []
    for delta in ("0.5", "0.1", "0.01", "0.000001", "0.000000000001"):
        for arms in (1, 2, 6, 65):
            exact = reference_calibration(mp.mpf(delta), arms)
            upper = mp.mpf(str(dkm.kk_calibration_upper(delta, arms)))
            assert upper >= exact
            assert upper-exact < mp.mpf("1e-35")*max(1, exact)
            counts = [1+(index+1)**2 for index in range(arms)]
            threshold = mp.mpf(str(dkm.kk_threshold_upper(counts, float(delta))))
            reference = exact+sum(3*mp.log(1+mp.log(n)) for n in counts)
            assert threshold >= reference
            assert threshold-reference < mp.mpf("1e-34")*max(1, reference)
            calibrations.append({"delta":delta,"arms":arms,
                                 "calibration_upper":str(upper),
                                 "reference":mp.nstr(exact,75),
                                 "outward_gap":mp.nstr(upper-exact,8)})
    minus = dict(cos_numerator=0,sin_numerator=-1,denominator=1)
    plus = dict(cos_numerator=0,sin_numerator=1,denominator=1)
    arms = [minus, plus]
    analytic = []
    for n, k, q in (([64,64],[0,64],Decimal("0.5")),
                    ([7,13],[1,10],Decimal("0.55"))):
        rows = dkm.observations(arms,n,k)
        null = dkm.likelihood.null_loglik_upper(rows,1,candidate=[q,Decimal(0),Decimal(0)])
        reference = sum(k)*mp.log(str(q))+(sum(n)-sum(k))*mp.log(1-mp.mpf(str(q)))
        upper = mp.mpf(null["loglik_upper_decimal"])
        assert upper >= reference
        assert upper-reference < mp.mpf("1e-45")
        analytic.append({"trials":n,"successes":k,"null_pooled_mean":str(q),
                         "analytic_null_loglik":mp.nstr(reference,75),
                         "certified_upper":str(upper),"outward_gap":mp.nstr(upper-reference,8)})
    decisions = []
    for calibration in ("kk21","kt_safe"):
        for sign,successes in ((1,[0,64]),(-1,[64,0])):
            cert = dkm.certificate(arms,[64,64],successes,sign,calibration=calibration,
                                   proposal=[Decimal("0.5"),Decimal(0),Decimal(0)])
            assert cert["decision"] == ("reject" if sign==1 else "accept")
            wrong = dkm.certificate(arms,[64,64],successes,-sign,calibration=calibration,
                                    proposal=[Decimal("0.5"),Decimal(0),Decimal(0)])
            assert wrong["decision"] == "abstain"
            decisions.append({"calibration":calibration,"sign":sign,
                              "correct_certificate":cert,"opposite_decision":wrong["decision"]})
    return {"calibration_reference_checks":len(calibrations),
            "count_dependent_threshold_checks":len(calibrations),
            "calibration_cases":calibrations,"analytic_global_null_checks":analytic,
            "two_sided_certificate_cases":decisions}


def main():
    if OUT.exists():
        raise RuntimeError("Preserve existing review checks")
    source = Path(dkm.__file__).resolve()
    before = digest(source)
    checks = run_checks()
    assert digest(source)==before,"Source changed during independent checks"
    paths = [Path(__file__).resolve(),source,ROOT/"experiments/structured_validation/certified_likelihood.py",
             ROOT/"experiments/circuit_validation/direct_tests.py",
             ROOT/"results/sota_round2/literature/kk_arxiv.txt"]
    result = {"status":"passed","created_utc":datetime.now(timezone.utc).isoformat(),
              "sources":{p.relative_to(ROOT).as_posix():digest(p) for p in paths},"checks":checks,
              "scope":"Independent 100-digit mathematical-reference checks and analytic null maxima; no Monte Carlo, performance comparison, or exact-oracle efficiency claim"}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2)
        f.write("\n")
    print(json.dumps({"status":"passed","output":str(OUT),"sha256":digest(OUT),
                      "calibration_checks":len(checks["calibration_cases"]),
                      "null_checks":len(checks["analytic_global_null_checks"]),
                      "certificate_cases":len(checks["two_sided_certificate_cases"])},indent=2))


if __name__ == "__main__":
    main()
