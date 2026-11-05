"""Outward arithmetic table for analytical full-Bernoulli information bounds.

No observations are generated and no controller is evaluated. The table is
a deterministic evaluation of proved formulas, not a Monte Carlo study.
"""
from decimal import Decimal, ROUND_CEILING
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"experiments/circuit_validation"))
from direct_tests import _Bounds


def bounded(value):
    return _Bounds(value.numerator)/value.denominator


def phi(value):
    value = Fraction(value)
    assert 0 <= value <= 1
    if value == 0:
        return _Bounds(0)
    if value == 1:
        return _Bounds(2).log()
    x = bounded(value)
    return ((1+x)*(1+x).log()+(1-x)*(1-x).log())/2


def uniform_ratio(m):
    m = Fraction(m)
    if m == Fraction(1, 2):
        return _Bounds(1)
    t = m/(1-m)
    return _Bounds(2)/(1+phi(t)/(bounded(t)*_Bounds(2).log()))


def fixed_signal_ratio(m, u):
    m, u = Fraction(m), Fraction(u)
    if m == Fraction(1, 2):
        return _Bounds(1)
    if u == 0:
        return bounded(2*(1-m))
    t = m/(1-m)
    return _Bounds(2)/(1+phi(t*u)/(bounded(t)*phi(u)))


def describe(value):
    rounded = value.hi.quantize(Decimal(".000001"), rounding=ROUND_CEILING)
    if rounded == 0:
        rounded = abs(rounded)
    return {"lower_decimal": str(value.lo), "upper_decimal": str(value.hi),
            "upper_rounded_six_decimals": str(rounded)}


def main():
    means = [Fraction(value) for value in (".5", ".4999", ".499", ".495", ".49", ".48", ".45", ".4", ".25", ".1", ".01")]
    uniform = []
    for m in means:
        value = uniform_ratio(m)
        row = {"m": str(m), "m_decimal": str(Decimal(m.numerator)/Decimal(m.denominator)),
               "ratio": describe(value), "maximum_additional_information_percent": describe(100*(value-1))}
        uniform.append(row)
    fixed = []
    for m in map(Fraction, (".49", ".45", ".25", ".1")):
        for u in map(Fraction, ("0", ".01", ".1", ".5", ".9", "1")):
            fixed.append({"m": str(m), "u": str(u), "signal_interpretation": "u=|C|/m; u=0 is the C->0 limit",
                          "ratio": describe(fixed_signal_ratio(m, u))})
    result = {"status": "computed_outward", "scope": "Outward deterministic evaluation of analytical full-Bernoulli oracle bounds; no sampling",
              "phi": "((1+x) log(1+x)+(1-x) log(1-x))/2",
              "definitions": {"m": "min(A,1-A)", "t": "m/(1-m)", "u": "|C|/m"},
              "uniform_formula": "2/(1+Phi(t)/(t*log(2)))",
              "fixed_signal_formula": "2/(1+Phi(t*u)/(t*Phi(u)))",
              "rounding": "Displayed bounds use the upper endpoint of an outward Decimal interval, rounded toward +infinity to six decimals",
              "uniform": uniform, "fixed_signal": fixed,
              "sources": {"experiments/structured_validation/psr_information_bound_table.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "experiments/circuit_validation/direct_tests.py": hashlib.sha256((ROOT/"experiments/circuit_validation/direct_tests.py").read_bytes()).hexdigest()}}
    target = ROOT/"results/structured_validation/psr_information_bound_table.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"uniform": [{"m": row["m_decimal"], "G_upper": row["ratio"]["upper_rounded_six_decimals"],
                                   "gain_percent_upper": row["maximum_additional_information_percent"]["upper_rounded_six_decimals"]}
                                  for row in uniform],
                      "fixed": [{"m": row["m"], "u": row["u"], "ratio_upper": row["ratio"]["upper_rounded_six_decimals"]}
                                for row in fixed]}, indent=2))


if __name__ == "__main__":
    main()
