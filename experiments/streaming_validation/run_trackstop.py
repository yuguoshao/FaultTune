"""Run the unchanged two-arm Track-and-Stop control on common budgets."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from benchmark import ROOT,DELTA,case_matrix,budget_schedule,aggregate
from sota_baseline import run_cell,SOURCE,METHOD


def run(repetitions,output):
    folder=Path(output)
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"spec.json").exists():
        raise RuntimeError("Refusing to overwrite existing experiment")
    cases=case_matrix()
    paths=[Path(__file__).resolve(),Path(__file__).resolve().with_name("sota_baseline.py"),
           Path(__file__).resolve().with_name("benchmark.py"),SOURCE]
    spec={"status":"development fixed two-arm Track-and-Stop-style control",
          "cases":cases,"methods":[METHOD],"repetitions":repetitions,"delta":DELTA,
          "checkpoints":budget_schedule(),"sources":{str(p.relative_to(ROOT)).replace("\\","/"):
              hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
          "scope":"existing frozen two-arm allocation and audited GLR; fixed±pi/2; not full all-angle structured SOTA"}
    (folder/"spec.json").write_text(json.dumps(spec,indent=2),encoding="utf-8")
    start=time.time()
    cells=[]
    with (folder/"trials.jsonl").open("w",encoding="utf-8") as ledger:
        for case in cases:
            cell,records=run_cell(case,repetitions,budget_schedule(),170000000+case["case_id"]*1009,DELTA)
            cells.append(cell)
            for row in records:
                ledger.write(json.dumps(row,separators=(",",":"))+"\n")
            ledger.flush()
            print(json.dumps({"case":case["case_id"],"mean":cell["mean_shots"],
                              "wrong":cell["wrong"],"abstained":cell["abstained"]}),flush=True)
    result={"spec":spec,"cells":cells,"aggregate":aggregate(cells,[METHOD]),
            "trial_count":len(cases)*repetitions,"elapsed_seconds":time.time()-start}
    (folder/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"done":True,"seconds":result["elapsed_seconds"],"aggregate":result["aggregate"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--repetitions",type=int,default=128)
    parser.add_argument("--output",default=str(ROOT/"results"/"streaming_validation"/"trackstop_v1"))
    args=parser.parse_args()
    if args.repetitions<1:
        parser.error("repetitions must bepositive")
    run(args.repetitions,args.output)
