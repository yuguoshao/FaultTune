"""Check ledger append boundaries and tamper rejection on existing smoke data."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time

import audit_unequal_incremental as incremental


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/"results/structured_validation"
SMOKE = RESULTS/"unequal_smoke_v1"


def main():
    started = time.perf_counter()
    output = RESULTS/"incremental_audit_checks.json"
    assert not output.exists()
    old = json.loads((SMOKE/"audit.json").read_text())
    newer = json.loads((SMOKE/"incremental_audit.json").read_text())
    for key in ("trajectories", "checks", "outcomes", "input_sha256"):
        assert old[key] == newer[key], key
    checks = ["Existing complete smoke audit agrees exactly with original audit on counts, outcomes and input hashes."]
    spec = (SMOKE/"spec.json").read_bytes()
    summary = (SMOKE/"summary.json").read_bytes()
    raw = (SMOKE/"trials.jsonl").read_bytes()
    first_end = raw.index(b"\n")+1

    with tempfile.TemporaryDirectory(prefix="incremental_audit_checks_", dir=str(RESULTS)) as temporary:
        base = Path(temporary).resolve()
        assert RESULTS.resolve() in base.parents

        def fixture(name, initial, with_summary=False):
            directory = base/name
            directory.mkdir()
            (directory/"spec.json").write_bytes(spec)
            (directory/"trials.jsonl").write_bytes(initial)
            if with_summary:
                (directory/"summary.json").write_bytes(summary)
            return directory

        partial = fixture("partial", raw[:first_end//2])
        with contextlib.redirect_stdout(io.StringIO()):
            assert incremental.audit(partial, partial/"audit.json") is None
        assert not (partial/"audit.json").exists()
        progress = json.loads((partial/"audit_progress.json").read_text())
        assert progress["trajectories"] == 0 and progress["audited_ledger_bytes"] == 0
        checks.append("A partial JSON line is never parsed or marked audited, and produces no complete audit.")

        growing = fixture("growing", raw[:first_end//2])
        def append_rest():
            time.sleep(.2)
            with (growing/"trials.jsonl").open("ab") as stream:
                stream.write(raw[first_end//2:])
                stream.flush()
            (growing/"summary.json").write_bytes(summary)
        writer = threading.Thread(target=append_rest)
        writer.start()
        with contextlib.redirect_stdout(io.StringIO()):
            audited = incremental.audit(growing, growing/"audit.json", watch=True)
        writer.join()
        for key in ("trajectories", "checks", "outcomes", "input_sha256"):
            assert audited[key] == old[key], key
        checks.append("An initially partial line followed by the remaining flushed ledger yields the identical complete audit.")

        trailing = fixture("trailing", raw+b"unexpected trailing bytes", True)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                incremental.audit(trailing, trailing/"audit.json")
        except AssertionError as error:
            assert "Final ledger differs" in str(error)
        else:
            raise AssertionError("Trailing unaudited bytes were accepted")
        assert not (trailing/"audit.json").exists()
        checks.append("Unexpected bytes after the expected records prevent a passed audit.")

        tampered = fixture("tampered", raw)
        writer_errors = []
        def tamper_after_all_records():
            try:
                until = time.monotonic()+60
                while time.monotonic() < until:
                    path = tampered/"audit_progress.json"
                    if path.exists() and json.loads(path.read_text())["trajectories"] == old["trajectories"]:
                        changed = bytearray(raw)
                        changed[10] ^= 1
                        (tampered/"trials.jsonl").write_bytes(changed)
                        (tampered/"summary.json").write_bytes(summary)
                        return
                    time.sleep(.05)
                raise AssertionError("No audit progress before fixture deadline")
            except Exception as error:
                writer_errors.append(repr(error))
        writer = threading.Thread(target=tamper_after_all_records)
        writer.start()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                incremental.audit(tampered, tampered/"audit.json", watch=True)
        except AssertionError as error:
            assert "Final ledger differs" in str(error)
        else:
            raise AssertionError("Previously audited bytes changed without rejection")
        writer.join()
        assert not writer_errors and not (tampered/"audit.json").exists()
        checks.append("Changing an already audited prefix while waiting for the summary is rejected by the final ledger hash.")

    result = {"status": "passed", "scope": "Existing smoke records only; no new simulated observations",
              "checks": checks, "audit_source_sha256": hashlib.sha256(Path(incremental.__file__).read_bytes()).hexdigest(),
              "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "elapsed_seconds": time.perf_counter()-started}
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
