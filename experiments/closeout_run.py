"""Deadline-bounded postprocessing of the already frozen experiment.

No new benchmark observations: only full raw audits, deterministic replay,
frozen statistical aggregation, and delivery integrity checks.
"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "experiments" / "circuit_validation"
RESULTS = BASE / "results"


def main():
    state = json.loads((BASE / "run_state.json").read_text(encoding="utf-8"))
    deadline = state["closeout_extension"]["deadline_epoch"]
    if time.time() >= deadline:
        raise RuntimeError("authorized closeout deadline has elapsed")
    spec = str(BASE / "confirmation_spec_v1.json")
    raw = str(RESULTS / "confirmation_v1_trials.jsonl.gz")
    tasks = {
        "records_audit": ["audit_records.py", "--spec", spec, "--trials", raw,
                          "--output", str(RESULTS / "confirmation_v1_records_audit.json"),
                          "--workers", "12", "--chunk-records", "16", "--progress-every", "25000"],
        "summary": ["summarize_results.py", "--spec", spec, "--trials", raw,
                    "--output-prefix", str(RESULTS / "confirmation_v1")],
        "seed_replay": ["replay_registered.py", "--spec", spec, "--trials", raw,
                        "--output", str(RESULTS / "confirmation_v1_replay_audit.json"), "--workers", "2"],
        "delivery_integrity": ["delivery_integrity.py", "--spec", spec, "--trials", raw,
                               "--status", str(RESULTS / "confirmation_v1_status.json"),
                               "--expected-spec-sha256", "ab2c711aacca68e84972ac5f6163a08bbc2f3e64544f90d26c9777904e146235",
                               "--output", str(RESULTS / "delivery_integrity.json")],
    }
    report = {"schema": "faulttune.closeout_jobs.v1", "start_epoch": time.time(),
              "deadline_epoch": deadline, "purpose": "authorized 45-minute closeout only",
              "jobs": {}, "state": "running"}
    processes, logs = {}, {}
    report_path = RESULTS / "closeout_jobs.json"
    def save():
        report["updated_epoch"] = time.time()
        temp = report_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(str(temp), str(report_path))
    try:
        for name, arguments in tasks.items():
            arguments[0] = str(BASE / arguments[0])
            command = [sys.executable, "-B"] + arguments
            log = (RESULTS / ("closeout_" + name + ".log")).open("w", encoding="utf-8")
            logs[name] = log
            process = subprocess.Popen(command, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            processes[name] = process
            report["jobs"][name] = {"pid": process.pid, "state": "running", "command": command,
                                   "started_epoch": time.time(), "log": str(Path(log.name))}
            print(json.dumps({"started": name, "pid": process.pid}), flush=True)
        save()
        while any(process.poll() is None for process in processes.values()):
            if time.time() >= deadline:
                report["state"] = "deadline_incomplete"
                for name, process in processes.items():
                    if process.poll() is None:
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        report["jobs"][name]["state"] = "terminated_at_authorized_deadline"
                break
            for name, process in processes.items():
                code = process.poll()
                if code is not None and report["jobs"][name]["state"] == "running":
                    report["jobs"][name].update(state="complete" if code == 0 else "failed",
                                               exit_code=code, finished_epoch=time.time())
                    print(json.dumps({"finished": name, "exit_code": code}), flush=True)
                    save()
            time.sleep(1)
        for name, process in processes.items():
            code = process.poll()
            if code is not None and report["jobs"][name]["state"] == "running":
                report["jobs"][name].update(state="complete" if code == 0 else "failed",
                                           exit_code=code, finished_epoch=time.time())
        if report["state"] == "running":
            report["state"] = "complete" if all(p.returncode == 0 for p in processes.values()) else "failed"
        save()
        print(json.dumps({"state": report["state"], "elapsed_seconds": time.time() - report["start_epoch"]}), flush=True)
    finally:
        for log in logs.values():
            log.close()


if __name__ == "__main__":
    main()
