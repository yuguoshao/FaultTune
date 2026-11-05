"""Executable frozen LP policy and independent folded-null risk verification.

Direction-error lemma for FIXED iid sign probability: every prefix reporting
plus has k>d/2. For any q<=1/2 its path likelihood q**k*(1-q)**(d-k) is at most
2**(-d), since that likelihood increases up to k/d>1/2. Multiply by the same
coin/stopping probability and sum the disjoint first-report prefixes. Thus
P_q(report plus)<=P_.5(report plus)=alpha/2; the minus case follows by symmetry.
This does not establish directional control for arbitrary time-varying q.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "results/sota_round5/optimal_sign_stopping/prototype.json"
AUDIT = ARTIFACT.with_name("runtime_audit.json")
VERIFIED = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rational(record):
    return Fraction(int(record["numerator"]), int(record["denominator"]))


@dataclass(frozen=True)
class Policy:
    numerators: tuple
    denominator: int
    artifact_sha256: str

    @property
    def horizon(self):
        return len(self.numerators)-1


def folded_null(policy):
    """Independent exact recursion on minority count, with no full Pascal rows."""
    live = {0: Fraction(1)}
    risk = expected = Fraction(0)
    for d in range(policy.horizon):
        expected += sum(live.values(), Fraction(0))
        fresh = {}
        for j, mass in live.items():
            branches = ((j, mass),) if 2*j == d else ((j, mass/2), (j+1, mass/2))
            for k, probability in branches:
                fresh[k] = fresh.get(k, Fraction(0))+probability
        live = {}
        for j, mass in fresh.items():
            stop = mass*Fraction(policy.numerators[d+1][j], policy.denominator)
            risk += stop
            live[j] = mass-stop
    abstain = sum(live.values(), Fraction(0))
    assert risk+abstain == 1
    return risk, expected


def load_policy(path=ARTIFACT):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    assert value["status"] == "exact_audited_classical_development_prototype"
    protocol = value["protocol"]
    denominator = protocol["probability_denominator"]
    assert type(denominator) is int and denominator > 0
    rows = protocol["policy_numerators"]
    assert len(rows) == value["development_plan"]["discordance_horizon"]+1
    for d, row in enumerate(rows):
        assert len(row) == d//2+1
        assert all(type(n) is int and 0 <= n <= denominator for n in row)
        if d % 2 == 0:
            assert row[d//2] == 0
    policy = Policy(tuple(tuple(row) for row in rows), denominator, digest(path))
    risk, _ = folded_null(policy)
    assert risk == rational(value["results"][0]["exact_null_any_report"])
    assert risk <= Fraction(1, 100)
    VERIFIED[policy.artifact_sha256] = policy
    return policy


def decide(policy, discordances, plus, uniform_integer):
    """Input coin must be fresh and independent, uniform on range(denominator)."""
    if VERIFIED.get(policy.artifact_sha256) != policy:
        raise ValueError("Load and exact-verify this frozen policy before use")
    if (type(discordances) is not int or type(plus) is not int or
            not 1 <= discordances <= policy.horizon or not 0 <= plus <= discordances):
        raise ValueError("Invalid counts or certified discordance cap exceeded")
    if type(uniform_integer) is not int or not 0 <= uniform_integer < policy.denominator:
        raise ValueError("Coin must lie in the certified integer range")
    j = min(plus, discordances-plus)
    if uniform_integer < policy.numerators[discordances][j]:
        return "plus" if 2*plus > discordances else "minus"
    return "abstain" if discordances == policy.horizon else "continue"


def main():
    policy = load_policy()
    checked = 0
    for d in range(1, policy.horizon+1):
        for k in range(d+1):
            threshold = policy.numerators[d][min(k, d-k)]
            for coin in sorted({0, policy.denominator-1, max(0, threshold-1), min(policy.denominator-1, threshold)}):
                observed = decide(policy, d, k, coin)
                if coin < threshold:
                    assert observed == ("plus" if 2*k > d else "minus")
                    assert 2*k != d
                else:
                    assert observed == ("abstain" if d == policy.horizon else "continue")
                reflected = decide(policy, d, d-k, coin)
                assert reflected == {"plus": "minus", "minus": "plus"}.get(observed, observed)
                checked += 1
    try:
        decide(policy, policy.horizon+1, 0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-horizon decision was accepted")
    risk, cost = folded_null(policy)
    result = {
        "status": "passed_independent_exact_null_recursion_and_runtime_checks",
        "artifact_sha256": policy.artifact_sha256,
        "runtime_source_sha256": digest(Path(__file__).resolve()),
        "checked_state_coin_combinations": checked,
        "exact_null_risk_matches_full_state_dp": True,
        "null_risk_numerator": str(risk.numerator), "null_risk_denominator": str(risk.denominator),
        "null_risk_float_diagnostic": float(risk),
        "null_expected_discordances_float_diagnostic": float(cost),
        "horizon_overrun_rejected": True, "zero_majority_never_reported": True,
        "reflection_symmetry_checked": True, "new_observations": 0,
        "fixed_iid_directional_error_lemma": "For q<=.5 each first plus-report prefix has k>d/2 and q^k(1-q)^(d-k)<=.5^d. Summing disjoint prefixes with identical policy-coin weights gives P_q(report plus)<=P_.5(report plus)=null_any_report/2; mirror for the other direction. This uses fixed iid q, not a changing alternative.",
    }
    with AUDIT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(AUDIT), "checked": checked, "null_risk": float(risk)}))


if __name__ == "__main__":
    main()
