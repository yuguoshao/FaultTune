"""Package the audited development evidence without altering prior studies."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/"results"/"learned_validation"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verified=0
    for name in ("development_v1","development_v2"):
        folder=RESULTS/name
        spec=json.loads((folder/"spec.json").read_text(encoding="utf-8"))
        for relative,expected in spec["source_sha256"].items():
            assert digest(ROOT/relative)==expected, relative
            verified+=1
        audit=json.loads((folder/"audit.json").read_text(encoding="utf-8"))
        assert audit.get("status")=="passed" or audit.get("passed") is True
        assert audit.get("ledger_records",audit.get("records_checked"))==55296
        for relative,expected in audit["artifact_sha256"].items():
            assert digest(folder/relative)==expected,relative
            verified+=1
        audit_source=audit.get("audit_source_path","experiments/learned_validation/audit_validation.py")
        assert digest(ROOT/audit_source)==audit["audit_source_sha256"]
        verified+=1
    theory=json.loads((RESULTS/"global_kl_manifest.json").read_text(encoding="utf-8"))
    for relative,expected in theory["files"].items():
        assert digest(ROOT/relative)==expected,relative
        verified+=1
    documents=[ROOT/"docs"/name for name in ("LEARNED_VALIDATION_RESEARCH.md",
                                            "LEARNED_VALIDATION_THEORY.md",
                                            "ADAPTIVE_PAIRED_EVIDENCE_LEMMA.md")]
    assert all(p.is_file() for p in documents)
    sources=sorted((ROOT/"experiments"/"learned_validation").glob("*.py"))
    sources += [ROOT/"experiments"/"circuit_validation"/"direct_tests.py"]
    archive=RESULTS/"learned_validation_sources.zip"
    with zipfile.ZipFile(str(archive),"w",compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sources+documents:
            bundle.write(str(path),str(path.relative_to(ROOT)).replace("\\","/"))
    with zipfile.ZipFile(str(archive)) as bundle:
        for path in sources+documents:
            assert bundle.read(str(path.relative_to(ROOT)).replace("\\","/"))==path.read_bytes()
    path=RESULTS/"evidence_manifest.json"
    artifacts=sorted(p for p in RESULTS.rglob("*") if p.is_file() and p!=path)
    all_files=sorted(set(sources+documents+artifacts))
    manifest={"status":"audited development evidence; no SOTA claim","client_date":"2026-11-03",
              "new_simulated_protocol_trajectories":92160,
              "reused_baseline_trajectories_in_v2":18432,
              "per_run_ledger_records":55296,
              "v1_seed_replay_records":2304,"v2_independent_seed_replay_records":1536,
              "upstream_source_and_theory_hashes_verified":verified,
              "source_archive_files":len(sources)+len(documents),
              "files":{str(p.relative_to(ROOT)).replace("\\","/"):digest(p) for p in all_files}}
    path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    for relative,expected in manifest["files"].items():
        assert digest(ROOT/relative)==expected,relative
    print(json.dumps({"status":"passed","manifest_files":len(all_files),
                      "source_archive_files":manifest["source_archive_files"],
                      "new_trajectories":92160,"manifest":str(path)},indent=2))


if __name__=="__main__":
    main()
