"""Outward operating characteristics for a frozen iid-sign stopping policy.

After the policy's discordance horizon no new decision is possible; every
undecided physical path pays the full common physical-pair cap. This avoids
crediting cheap early abstentions. No observations are generated here.
"""
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sota_round4.interval_sign_cost import (
    rational, enclose_fraction, down, up, tree_sum, dot_interval,
    binomial_intervals)


def canonical_policy(denominator, numerators):
    if type(denominator) is not int or denominator < 1 or len(numerators)<2:
        raise ValueError('A positive integer denominator and nonempty horizon are required')
    rows=[]
    for d,row in enumerate(numerators):
        if len(row)!=d//2+1:
            raise ValueError('Folded row length must be floor(d/2)+1')
        if any(type(n) is not int or not 0<=n<=denominator for n in row):
            raise ValueError('Invalid stopping probability numerator')
        if d%2==0 and row[-1]!=0:
            raise ValueError('A tie cannot report a direction')
        rows.append(tuple(row))
    return denominator,tuple(rows)


@lru_cache(maxsize=8)
def _probability_arrays(policy):
    denominator,rows=policy
    output=[]
    for d,row in enumerate(rows):
        exact=[F(row[min(k,d-k)],denominator) for k in range(d+1)]
        probability=np.array([enclose_fraction(a) for a in exact])
        complement=np.array([enclose_fraction(1-a) for a in exact])
        output.append((probability[:,0],probability[:,1],complement[:,0],complement[:,1]))
    return output


def policy_passage_intervals(policy, q, steps=None):
    denominator,rows=policy
    canonical_policy(denominator,rows)
    horizon=len(rows)-1
    steps=horizon if steps is None else steps
    if type(steps) is not int or not 0<=steps<=horizon:
        raise ValueError('Sign horizon exceeds frozen policy')
    q=rational(q)
    qlo,qhi=enclose_fraction(q)
    clo,chi=enclose_fraction(1-q)
    lo,hi=np.ones(1),np.ones(1)
    pluslo,plushi=np.zeros(steps+1),np.zeros(steps+1)
    minuslo,minushi=np.zeros(steps+1),np.zeros(steps+1)
    for d in range(1,steps+1):
        freshlo,freshhi=np.zeros(d+1),np.zeros(d+1)
        freshlo[:-1],freshhi[:-1]=down(lo*clo),up(hi*chi)
        freshlo[1:]=down(freshlo[1:]+down(lo*qlo))
        freshhi[1:]=up(freshhi[1:]+up(hi*qhi))
        alo,ahi,blo,bhi=_probability_arrays(policy)[d]
        stoppedlo,stoppedhi=down(freshlo*alo),up(freshhi*ahi)
        positive=np.arange(d+1)>d/2
        negative=np.arange(d+1)<d/2
        pluslo[d],plushi[d]=tree_sum(stoppedlo[positive],False),tree_sum(stoppedhi[positive],True)
        minuslo[d],minushi[d]=tree_sum(stoppedlo[negative],False),tree_sum(stoppedhi[negative],True)
        lo,hi=down(freshlo*blo),up(freshhi*bhi)
    return {'plus':(pluslo,plushi),'minus':(minuslo,minushi),
            'survival':(tree_sum(lo,False),min(1.,tree_sum(hi,True)))}


def operating_intervals(policy, pminus, pplus, physical_pair_cap):
    pairs=physical_pair_cap
    if type(pairs) is not int or pairs<1:
        raise ValueError('Positive integer physical-pair cap required')
    denominator,rows=policy
    canonical_policy(denominator,rows)
    pm,pp=rational(pminus),rational(pplus)
    r=pp*(1-pm)+pm*(1-pp)
    if not r:
        return {'plus':[0.,0.],'minus':[0.,0.],'abstain':[1.,1.],
                'expected_shots':[float(2*pairs),float(2*pairs)]}
    horizon=min(pairs,len(rows)-1)
    passage=policy_passage_intervals(policy,pp*(1-pm)/r,horizon)
    _,_,headlo,headhi,taillo,tailhi=binomial_intervals(pairs,r)
    _,_,_,_,nextlo,nexthi=binomial_intervals(pairs+1,r)
    d=np.arange(1,horizon+1,dtype=float)
    rlo,rhi=enclose_fraction(r)
    waitinglo=down(down(down(nextlo[2:horizon+2]/rhi)*d)+down(pairs*headlo[:horizon]))
    if rlo==0:
        waitinghi=np.full(horizon,float(pairs))
    else:
        with np.errstate(over='ignore'):
            firsthi=np.minimum(float(pairs),up(up(nexthi[2:horizon+2]/rlo)*d))
        waitinghi=np.minimum(float(pairs),up(firsthi+up(pairs*headhi[:horizon])))
    plus=dot_interval((passage['plus'][0][1:],passage['plus'][1][1:]),
                      (taillo[1:horizon+1],tailhi[1:horizon+1]))
    minus=dot_interval((passage['minus'][0][1:],passage['minus'][1][1:]),
                       (taillo[1:horizon+1],tailhi[1:horizon+1]))
    masslo=down(passage['plus'][0][1:]+passage['minus'][0][1:])
    masshi=up(passage['plus'][1][1:]+passage['minus'][1][1:])
    spentlo,spenthi=dot_interval((masslo,masshi),(waitinglo,waitinghi))
    spentlo=float(down(spentlo+down(pairs*passage['survival'][0])))
    spenthi=min(float(pairs),float(up(spenthi+up(pairs*passage['survival'][1]))))
    abstain=[float(down(down(1-plus[1])-minus[1])),
             min(1.,float(up(up(1-plus[0])-minus[0])))]
    return {'plus':[max(0.,plus[0]),min(1.,plus[1])],
            'minus':[max(0.,minus[0]),min(1.,minus[1])],
            'abstain':abstain,'expected_shots':[float(down(2*spentlo)),float(up(2*spenthi))],
            'physical_pair_cap':pairs,'frozen_policy_discordance_horizon':len(rows)-1,
            'exact_arm_probabilities':[str(pm),str(pp)],
            'protocol':'Every discordance inspected; after frozen sign horizon no further reports; all abstentions pay full physical cap.',
            'arithmetic':'Outward interval sign recursion and independently audited rational-ratio binomial tail enclosures.'}
