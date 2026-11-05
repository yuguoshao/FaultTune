"""Frozen-plan access, deterministic selection and prospective integrity checks."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PLAN = ROOT / "results/structured_validation/prospective_plan.json"
PLAN_SHA256 = "c9baab61fc33594ac21d3231ae51fc40bc3824506145ae9f45c0c71960e81d37"
DIRECTORY = ROOT / "results/structured_validation/prospective_v1"
WRAPPERS = ("common.py", "run.py", "analyze.py")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+"\n")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_plan():
    require(digest(PLAN) == PLAN_SHA256, "Frozen prospective plan changed")
    plan = read_json(PLAN)
    require(all(digest(ROOT/name) == value for name, value in
                plan["algorithm_sources_sha256"].items()), "Algorithm source changed")
    control = ROOT / "results/structured_validation/run_control.json"
    require(digest(control) == plan["run_control_sha256"], "Run deadline changed")
    return plan


def wrapper_hashes():
    return {path.relative_to(ROOT).as_posix(): digest(path)
            for path in (HERE/name for name in WRAPPERS)}


def select_from_complete_development(plan):
    """Do not use any incomplete file, provisional statistic or unaudited result."""
    scores, inputs, reference_cases = {}, {}, None
    for entry in plan["selection"]["development_inputs"]:
        directory = ROOT / entry["directory"]
        paths = {name: directory/name for name in
                 ("spec.json", "summary.json", "trials.jsonl", "audit.json")}
        require(all(path.is_file() for path in paths.values()),
                "Development summary and passed audit are required: "+str(directory))
        hashes = {name: digest(path) for name, path in paths.items()}
        require(hashes["spec.json"] == entry["spec_sha256"], "Development spec changed")
        audit, spec, summary = (read_json(paths[name]) for name in
                                ("audit.json", "spec.json", "summary.json"))
        require(audit.get("status") == "passed", "Development audit did not pass")
        required_hashes = {name: hashes[name] for name in ("spec.json", "summary.json", "trials.jsonl")}
        require(audit[entry["audit_hash_field"]] == required_hashes,
                "Development audit does not bind the current complete input files")
        require(summary["spec"] == spec, "Development summary/spec mismatch")
        require(summary["new_trajectories"] == entry["expected_records"],
                "Development record count is incomplete")
        require(spec["repetitions"] == entry["repetitions"] and
                spec["seed_base"] == entry["seed_base"], "Development replicate design changed")
        require(spec["cap"] == 2**22 and spec["delta"] == plan["delta"],
                "Development caps or risks are not comparable")
        require(all(plan["algorithm_sources_sha256"].get(name) == value
                    for name, value in spec["sources"].items()), "Development algorithm hashes differ")
        cases = sorted(spec["cases"], key=lambda c: c["case_id"])
        require(len(cases) == entry["expected_cases"] and
                all(c["kind"] == "main" and c["C"] > 0 for c in cases),
                "Wrong development case matrix")
        if reference_cases is None:
            reference_cases = cases
        require(cases == reference_cases, "Development case matrices differ")
        by_method, identities = defaultdict(list), set()
        for cell in summary["cells"]:
            key = cell["case_id"], cell["method"]
            require(key not in identities, "Duplicate development cell")
            identities.add(key)
            require(cell["kind"] == "main" and cell["repetitions"] == entry["repetitions"],
                    "Incomplete development cell")
            require(math.isfinite(cell["mean_shots"]) and 0 < cell["mean_shots"] <= spec["cap"],
                    "Invalid unconditional development mean")
            by_method[cell["method"]].append(cell)
        require(len(by_method) == entry["expected_methods"], "Wrong development method count")
        expected_ids = {c["case_id"] for c in cases}
        for method, cells in by_method.items():
            require(len(cells) == len(cases) and {c["case_id"] for c in cells} == expected_ids,
                    "Development method lacks cases")
            require(method not in scores, "Development method repeated across datasets")
            scores[method] = math.exp(sum(math.log(c["mean_shots"]) for c in
                                         sorted(cells, key=lambda c: c["case_id"]))/len(cells))
        inputs[entry["directory"]] = hashes
    expected_methods = set(plan["selection"]["candidate_pool"]+plan["selection"]["comparator_pool"])
    require(set(scores) == expected_methods, "Development methods differ from the frozen pools")
    choose = lambda pool: min(pool, key=lambda name: (scores[name], name))
    candidate = choose(plan["selection"]["candidate_pool"])
    comparator = choose(plan["selection"]["comparator_pool"])
    secondary = plan["selection"]["fixed_secondary_comparator"]
    return {"candidate": candidate, "primary_comparator": comparator,
            "secondary_comparator": secondary, "methods": sorted({candidate, comparator, secondary}),
            "development_scores": scores, "development_input_sha256": inputs,
            "selection_rule": plan["selection"]["metric"],
            "tie_break": plan["selection"]["tie_break"]}


def validate_release(recheck_development=True):
    plan = load_plan()
    selection_path, release_path = DIRECTORY/"selection.json", DIRECTORY/"release.json"
    selection, release = read_json(selection_path), read_json(release_path)
    require(release["explicit_root_release"] is True and bool(release["release_reference"].strip()),
            "Explicit root release is missing")
    require(release["plan_sha256"] == PLAN_SHA256 and
            release["selection_sha256"] == digest(selection_path), "Release binding changed")
    require(release["wrapper_sources_sha256"] == wrapper_hashes(), "Wrapper or analysis changed after release")
    require(release["algorithm_sources_sha256"] == plan["algorithm_sources_sha256"],
            "Released algorithm hashes differ")
    require(selection["plan_sha256"] == PLAN_SHA256 and
            release["selection"] == selection["selection"], "Released method selection differs")
    if recheck_development:
        require(selection["selection"] == select_from_complete_development(plan),
                "Development selection inputs or selection changed")
    return plan, selection["selection"], release


def build_cells(records, cases, methods):
    """Reconstruct all costs and decisions without conditioning on correctness."""
    case_map = {c["case_id"]: c for c in cases}
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["case_id"], record["method"])].append(record)
    cells = []
    for case_id in sorted(case_map):
        case = case_map[case_id]
        expected = "accept" if case["C"] < 0 else "reject" if case["C"] > 0 else "abstain"
        for method in methods:
            local = grouped[(case_id, method)]
            require(bool(local), "Missing prospective cell")
            cells.append({"case_id": case_id, "kind": case["kind"], "method": method,
                          "repetitions": len(local),
                          "mean_shots": sum(r["shots"] for r in local)/len(local),
                          "wrong": sum(r["decision"] not in ("abstain", expected) for r in local),
                          "abstained": sum(r["decision"] == "abstain" for r in local)})
    return cells
