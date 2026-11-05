"""Certified fixed randomized prefix followed by a frozen sign boundary.

The fallback is inspected first at discordance H+1. There is one decision
opportunity per discordance, and all still-undecided paths pay the physical
cap. Validity requires a separate joint risk certificate for this whole rule.
"""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.sota_round4.interval_sign_cost import (
    rational,enclose_fraction,down,up,tree_sum,dot_interval,binomial_intervals,validate_table)
from experiments.sota_round5.interval_randomized_stopping import canonical_policy,_probability_arrays


def first_passage_intervals(policy,boundary,q,steps):
    canonical_policy(*policy)
    validate_table(boundary,steps)
    q=rational(q)
    qlo,qhi=enclose_fraction(q); clo,chi=enclose_fraction(1-q)
    horizon=len(policy[1])-1
    probabilities=_probability_arrays(policy)
    lo,hi=np.ones(1),np.ones(1)
    origin=0
    pluslo,plushi=np.zeros(steps+1),np.zeros(steps+1)
    minuslo,minushi=np.zeros(steps+1),np.zeros(steps+1)
    for d in range(1,steps+1):
        freshlo,freshhi=np.zeros(len(lo)+1),np.zeros(len(hi)+1)
        freshlo[:-1],freshhi[:-1]=down(lo*clo),up(hi*chi)
        freshlo[1:]=down(freshlo[1:]+down(lo*qlo))
        freshhi[1:]=up(freshhi[1:]+up(hi*qhi))
        counts=np.arange(origin,origin+len(freshlo))
        if d<=horizon:
            alo,ahi,blo,bhi=probabilities[d]
            stoppedlo,stoppedhi=down(freshlo*alo),up(freshhi*ahi)
            positive,negative=2*counts>d,2*counts<d
            lo,hi=down(freshlo*blo),up(freshhi*bhi)
        else:
            cutoff=boundary[d]
            positive,negative=counts>=cutoff,counts<=d-cutoff
            stoppedlo,stoppedhi=freshlo,freshhi
            survive=~positive & ~negative
            lo,hi=freshlo[survive],freshhi[survive]
            if np.any(survive): origin=int(counts[survive][0])
        pluslo[d],plushi[d]=tree_sum(stoppedlo[positive],False),tree_sum(stoppedhi[positive],True)
        minuslo[d],minushi[d]=tree_sum(stoppedlo[negative],False),tree_sum(stoppedhi[negative],True)
        if not len(lo): break
    return {'plus':(pluslo,plushi),'minus':(minuslo,minushi),
            'survival':(tree_sum(lo,False),min(1.,tree_sum(hi,True)))}


def operating_rq(policy,boundary,r,q,physical_pair_cap):
    pairs=physical_pair_cap
    canonical_policy(*policy)
    validate_table(boundary,pairs)
    r,q=rational(r),rational(q)
    if not r:
        return {'plus':[0.,0.],'minus':[0.,0.],'abstain':[1.,1.],
                'expected_shots':[float(2*pairs),float(2*pairs)]}
    passage=first_passage_intervals(policy,boundary,q,pairs)
    _,_,headlo,headhi,taillo,tailhi=binomial_intervals(pairs,r)
    _,_,_,_,nextlo,nexthi=binomial_intervals(pairs+1,r)
    d=np.arange(1,pairs+1,dtype=float)
    rlo,rhi=enclose_fraction(r)
    waitinglo=down(down(down(nextlo[2:]/rhi)*d)+down(pairs*headlo[:-1]))
    if rlo==0:
        waitinghi=np.full(pairs,float(pairs))
    else:
        with np.errstate(over='ignore'):
            firsthi=np.minimum(float(pairs),up(up(nexthi[2:]/rlo)*d))
        waitinghi=np.minimum(float(pairs),up(firsthi+up(pairs*headhi[:-1])))
    plus=dot_interval((passage['plus'][0][1:],passage['plus'][1][1:]),(taillo[1:],tailhi[1:]))
    minus=dot_interval((passage['minus'][0][1:],passage['minus'][1][1:]),(taillo[1:],tailhi[1:]))
    masslo=down(passage['plus'][0][1:]+passage['minus'][0][1:])
    masshi=up(passage['plus'][1][1:]+passage['minus'][1][1:])
    spentlo,spenthi=dot_interval((masslo,masshi),(waitinglo,waitinghi))
    spentlo=float(down(spentlo+down(pairs*passage['survival'][0])))
    spenthi=min(float(pairs),float(up(spenthi+up(pairs*passage['survival'][1]))))
    return {'plus':[max(0.,plus[0]),min(1.,plus[1])],
            'minus':[max(0.,minus[0]),min(1.,minus[1])],
            'abstain':[float(down(down(1-plus[1])-minus[1])),min(1.,float(up(up(1-plus[0])-minus[0])))],
            'expected_shots':[float(down(2*spentlo)),float(up(2*spenthi))],
            'physical_pair_cap':pairs,'randomized_prefix_discordances':len(policy[1])-1,
            'first_fallback_discordance':len(policy[1]),'exact_r':str(r),'exact_q':str(q)}


def operating_intervals(policy,boundary,pminus,pplus,physical_pair_cap):
    pm,pp=rational(pminus),rational(pplus)
    r=pp*(1-pm)+pm*(1-pp)
    output=operating_rq(policy,boundary,r,pp*(1-pm)/r if r else 0,physical_pair_cap)
    output['exact_arm_probabilities']=[str(pm),str(pp)]
    return output
