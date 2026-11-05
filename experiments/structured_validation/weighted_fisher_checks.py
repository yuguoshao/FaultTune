"""Independent exact finite-state checks of the frozen weighted mixture."""
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random

import weighted_fisher as w
from expert_config import expert_config
from predictive import ROOT


def exact_factor(block, odds):
    nm, np_ = block["nminus"], block["nplus"]
    total = block["minus_successes"]+block["plus_successes"]
    lo, hi = max(0, total-nm), min(np_, total)
    normalizer = sum((math.comb(np_, k)*math.comb(nm, total-k)*odds**k
                      for k in range(lo, hi+1)), Fraction(0))
    return odds**block["plus_successes"]*math.comb(nm+np_, total)/normalizer


def main():
    rng = random.Random(821907)
    experts = expert_config()
    assert sum(Fraction(*e["weight"]) for e in experts) == 1
    errors = []
    for index in range(24):
        state = w.empty_state()
        for unused in range(1+index%3):
            nm, np_ = rng.randrange(1, 5), rng.randrange(1, 5)
            w.update_block(state, nm, np_, rng.randrange(nm+1), rng.randrange(np_+1))
        exact = Fraction(0)
        for expert in experts:
            capital = Fraction(*expert["weight"])
            for block in state["blocks"]:
                capital *= exact_factor(block, Fraction(expert["numerator"], expert["denominator"]))
            exact += capital
        with localcontext() as ctx:
            ctx.prec = 110
            reference = (Decimal(exact.numerator)/Decimal(exact.denominator)).ln()
        certificate = w.certificate_from_blocks(state["blocks"])
        low, high = map(Decimal, certificate["evidence_bounds"])
        assert low <= reference <= high, (index, low, reference, high)
        error = abs(w.screen(state)["log_evidence"]-float(reference))
        assert error < 1e-10
        errors.append(error)
    counts = [3, 7, 2, 5]
    state = w.empty_state()
    for (minus, plus), count in zip(((0, 0), (0, 1), (1, 0), (1, 1)), counts):
        for unused in range(count):
            w.update_block(state, 1, minus, plus)
    assert abs(w.screen(state)["log_evidence"]-w.pair_grid_log(counts)) < 1e-12
    cb, pb = w.certificate_from_blocks(state["blocks"]), w.pair_grid_bounds(counts)
    assert Decimal(cb["evidence_bounds"][0]) <= pb.hi and pb.lo <= Decimal(cb["evidence_bounds"][1])
    path = ROOT / "results/structured_validation/weighted_fisher_checks.json"
    result = {"status": "passed", "exact_fraction_mixture_trajectories": 24,
              "m1_matched_prior_identity": True, "exact_weights_sum_to_one": True,
              "maximum_float_error": max(errors), "experts": len(experts),
              "sources": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__).resolve(), Path(w.__file__).resolve(),
                                    Path(__file__).resolve().with_name("expert_config.py"))},
              "scope": "Exact finite-state arithmetic and weighted evidence composition; no performance or SOTA claim."}
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
