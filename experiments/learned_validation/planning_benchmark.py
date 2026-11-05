"""Isolated local CPU benchmark, distinct from paid measurement cost."""
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from designs import choose_design
from kl_designs import choose_kl_design
from run_validation import ROOT, case_matrix, pilot_for_case


def main():
    counts=np.concatenate([pilot_for_case(c,8)[0] for c in case_matrix() if c["kind"]=="main"])
    totals=np.full_like(counts,256)
    methods=("learned_closed_form","learned_grid_design","learned_kl_grid")
    iterations, rounds=10, 7
    observations={m:[] for m in methods}
    def evaluate(method):
        return choose_kl_design(counts,totals) if method=="learned_kl_grid" else choose_design(method,counts,totals)
    for method in methods:
        evaluate(method)
    for repeat in range(rounds):
        # Rotate order to reduce systematic warm-up/background-load effects.
        order=methods[repeat%len(methods):]+methods[:repeat%len(methods)]
        for method in order:
            start=time.perf_counter()
            for unused in range(iterations):
                evaluate(method)
            observations[method].append((time.perf_counter()-start)/iterations)
    medians={m:float(np.median(observations[m])) for m in methods}
    result={"status":"local CPU microbenchmark","batch_designs":len(counts),"rounds":rounds,
            "iterations_per_round":iterations,"python":platform.python_version(),"numpy":np.__version__,
            "median_seconds_per_batch":medians,"all_seconds_per_batch":observations,
            "runtime_over_closed":{m:medians[m]/medians["learned_closed_form"] for m in methods},
            "scope":"includes identical pilot fitting and public caps; 513-point grids; one Windows machine, background processes may affect timings; not a measurement-saving or SOTA claim",
            "source_sha256":{name:hashlib.sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()
                             for name in ("planning_benchmark.py","designs.py","kl_designs.py")}}
    (ROOT/"results"/"learned_validation"/"planning_benchmark.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
