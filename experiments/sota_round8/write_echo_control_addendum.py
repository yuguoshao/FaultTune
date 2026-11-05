"""Record the stronger public-action control received after the literature review."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def main():
    prior='results/sota_round8/literature/echo_priority_and_same_action_controls.json'
    guard='results/sota_round8/fixed_stage_guard.json'
    data=json.loads((ROOT/guard).read_text(encoding='utf-8'))
    report={
        'schema':'faulttune.round8.echo_control_prior_addendum.v1',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'source_bound_comparator_contract_clarification_not_independent_guard_audit',
        'new_simulation_observations':0,
        'sota_claim_supported':False,
        'clarification':'The k42/original-update-gap dual-quadrature design in the preceding review is an unfrozen branch proposal. The now-implemented stats control uses the stronger allowed amplitude separation2, three echo blocks, gain6 and sensing duration6. These settings and positive visibility/dephasing bounds are shared public resources.',
        'public_contract_read_from_guard':data['public_frozen_design'],
        'stronger_control_implication':'With public A>=.9*exp(-6*.005)>0 and |6chi|<=1.2<pi/2, every |chi|>=h implies |E[Z]|>=A_min*sin(6h). Consequently a single randomized sine quadrature supports a direct two-sided equivalence certificate. Requiring a comparator to learn A or perform both quadratures would impose avoidable overhead under this stronger public contract.',
        'when_dual_quadrature_remains_relevant':'The phase-halfspace sin/cos construction remains applicable when only common positive contrast is justified, without a positive lower bound, provided the preparation/readout circle model holds. It does not by itself promise bounded-cost power as contrast vanishes.',
        'implemented_control_status':data['status'],
        'finite_control_scope':'The fixed-stage guard gives finite-risk and case-specific power bounds, with larger costs than anticipated sequential variants. It is a classical same-action baseline and does not establish SOTA. Pulse/readout/setup overhead remains additional to the guard sensing-window count.',
        'validation_boundary':'This addendum read the produced artifact and its public contract; it is not an independent proof/code audit of fixed_stage_guard.py. Root/stats own that verification.',
        'source_sha256':{p:sha(p) for p in [prior,guard,'experiments/sota_round8/fixed_stage_guard.py','experiments/sota_round8/write_echo_control_addendum.py']},
    }
    path=ROOT/'results/sota_round8/literature/echo_control_prior_addendum.json'
    with path.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path.relative_to(ROOT))}))


if __name__=='__main__':main()
