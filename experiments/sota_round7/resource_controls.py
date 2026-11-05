"""Executable paid-observation controls for the two screened mechanisms.

These are classical controls, not new SOTA methods. No true model parameter
enters either class. `next_action` records a paid attempt immediately; a failed
attempt ends with an explicit error and full-cap penalty. A delivered action
must receive exactly one outcome before another action can be issued.
"""
from fractions import Fraction as F
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.circuit_validation.direct_tests import _Bounds as I


def iv(value):
    value = F(value)
    return I(value.numerator)/value.denominator


def integer(value, name, low=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(name+' must be an integer >= '+str(low))
    return value


class TwoSidedBet:
    """Exact rational decisions, with fast directed-log bounds first."""
    def __init__(self, bet):
        self.bet = F(bet)
        if not 0 < self.bet < 1:
            raise ValueError('Bet must lie in (0,1)')
        self.positive = self.negative = self.zero = 0
        self.up = iv(1+self.bet).log()
        self.down = iv(1-self.bet).log()
        self.threshold = I(200).log()

    def observe(self, z):
        if isinstance(z, bool) or z not in (-1, 0, 1):
            raise ValueError('Evidence increment must be -1,0,1')
        if z == 1:
            self.positive += 1
        elif z == -1:
            self.negative += 1
        else:
            self.zero += 1

    def _crossed(self, positive, negative, value):
        if value.lo >= self.threshold.hi:
            return True, 'directed_log'
        if value.hi < self.threshold.lo:
            return False, 'directed_log'
        # This fallback resolves even an exact threshold equality, so the
        # executable stopping rule equals the rule used in the power proof.
        numerator, denominator = self.bet.numerator, self.bet.denominator
        lhs = pow(denominator+numerator, positive)*pow(denominator-numerator, negative)
        rhs = 200*pow(denominator, positive+negative)
        return lhs >= rhs, 'exact_integer'

    def certificate(self):
        plus = self.positive*self.up+self.negative*self.down
        minus = self.negative*self.up+self.positive*self.down
        yes_plus, how_plus = self._crossed(self.positive, self.negative, plus)
        yes_minus, how_minus = self._crossed(self.negative, self.positive, minus)
        assert not (yes_plus and yes_minus)
        return {'crossing_sign': 1 if yes_plus else -1 if yes_minus else 0,
                'positive': self.positive, 'negative': self.negative, 'zero': self.zero,
                'bet_exact': str(self.bet), 'threshold_exact': '200',
                'log_plus_interval': [str(plus.lo), str(plus.hi)],
                'log_minus_interval': [str(minus.lo), str(minus.hi)],
                'decision_arithmetic': [how_plus, how_minus]}


class _PaidControl:
    def __init__(self, cap):
        self.cap = integer(cap, 'cap', 1)
        self.shots = 0
        self.pending = None
        self.status = 'running'
        self.decision = 'abstain'
        self.ledger = []

    def _before_action(self):
        if self.pending is not None:
            raise RuntimeError('An issued paid action still needs an outcome')
        if self.status != 'running' or self.shots >= self.cap:
            raise RuntimeError('Control has stopped')

    def _issue(self, action):
        self.shots += 1
        action = dict(action, action_id=self.shots)
        self.pending = action
        self.ledger.append(dict(action, outcome=None, status='issued'))
        return dict(action)

    def _validate_delivery(self, action_id, outcome, maximum):
        integer(action_id, 'action_id', 1)
        integer(outcome, 'outcome')
        if outcome > maximum:
            raise ValueError('Outcome outside registered alphabet')
        if self.pending is None or action_id != self.pending['action_id']:
            raise RuntimeError('Outcome does not match the pending paid action')

    def _delivered(self, outcome):
        self.ledger[-1].update(outcome=outcome, status='delivered')
        self.pending = None

    def _stop(self, certificate, plus_decision):
        sign = certificate['crossing_sign']
        if sign:
            self.decision = plus_decision if sign > 0 else ('reject' if plus_decision == 'accept' else 'accept')
            self.status = 'decided'
        elif self.shots == self.cap:
            self.status = 'cap_abstention'

    def fail_pending(self, action_id):
        integer(action_id, 'action_id', 1)
        if self.pending is None or action_id != self.pending['action_id']:
            raise RuntimeError('Failure does not match the pending paid action')
        self.ledger[-1]['status'] = 'execution_failed'
        self.pending = None
        self.status = 'execution_error'
        self.decision = 'abstain'

    def finalize_incomplete(self, reason):
        if self.status != 'running':
            raise RuntimeError('Only an unfinished run can be finalized')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('An explicit interruption reason is required')
        if self.pending is not None:
            self.ledger[-1].update(status='interrupted', reason=reason)
        self.pending = None
        self.status = 'incomplete'
        self.decision = 'abstain'
        self.interruption_reason = reason

    def summary(self):
        return {'status': self.status, 'decision': self.decision,
                'actual_paid_shots': self.shots, 'shot_cap': self.cap,
                # Running/interrupted records must not look artificially cheap
                # if persisted before the executor can finalize them.
                'full_cap_penalized_shots': self.shots if self.status == 'decided' else self.cap,
                'terminal': self.status != 'running',
                'interruption_reason': getattr(self, 'interruption_reason', None),
                'pending_action': None if self.pending is None else dict(self.pending),
                'evidence': self.evidence.certificate()}


class RamseyFixedDepthControl(_PaidControl):
    """Depth8/16 control under the explicit round7 noisy-Ramsey contract."""
    def __init__(self, depth, seed=None, cap=2**20):
        integer(depth, 'depth', 1)
        if depth not in (8, 16):
            raise ValueError('Only reviewed depths8 and16 are registered')
        if seed is not None:
            integer(seed, 'seed')
        super().__init__(cap)
        self.depth = depth
        self.rng = random.SystemRandom() if seed is None else random.Random(seed)
        self.randomness_mode = 'system_random_bits' if seed is None else 'seeded_pseudorandom_replay'
        self.evidence = TwoSidedBet(F(3, 250) if depth == 8 else F(7, 500))
        self.gates = self.switches = self.device_tenths = 0
        self.last_phase_sign = None

    def next_action(self):
        self._before_action()
        sign = 2*self.rng.randrange(2)-1
        switch = int(self.last_phase_sign != sign)
        self.last_phase_sign = sign
        self.gates += self.depth
        self.switches += switch
        self.device_tenths += 22+10*self.depth+switch
        assert self.gates <= 16*self.cap
        assert self.device_tenths <= 183*self.cap
        return self._issue({'kind': 'ramsey_quadrature', 'depth': self.depth, 'phase_sign': sign})

    def observe(self, action_id, outcome):
        self._validate_delivery(action_id, outcome, 1)
        z = self.pending['phase_sign']*(2*outcome-1)
        self.evidence.observe(z)
        self._delivered(outcome)
        self._stop(self.evidence.certificate(), 'reject')
        return self.summary()

    def summary(self):
        result = super().summary()
        result.update({'sensing_pulse_uses': self.gates, 'phase_switches': self.switches,
                       'device_time_exact': str(F(self.device_tenths, 10)),
                       'randomness_mode': self.randomness_mode,
                       'randomness_assumption': 'The theoretical guarantee uses independent fair phase coins; a fixed seeded replay is not itself a probabilistic guarantee.',
                       'model_contract': 'round7 stationary commuting-dephasing Ramsey with |phi|<=.02',
                       'power_scope': ('Seeded deterministic replay has no inherited probabilistic power guarantee'
                                       if self.randomness_mode == 'seeded_pseudorandom_replay'
                                       else 'Full declared eligible domain at epsilon=.001 under the independent fair-coin and physical observation assumptions'
                                       if self.cap == 2**20 else 'Custom cap has no inherited95percent power certificate')})
        return result


class InvariantBitstringControl(_PaidControl):
    """Classical paired X0 control; requires invariant spectator success >0.

The constructor demands the named structural promise explicitly. This is NOT
a validator for unknown amplitude-dependent spectator crosstalk.
"""
    def __init__(self, model_contract, cap=131072):
        if model_contract != 'same_duration_disjoint_invariant_spectators':
            raise ValueError('An explicit reviewed invariance contract is required')
        super().__init__(cap)
        if cap % 2:
            raise ValueError('Pair control needs an even physical cap')
        self.evidence = TwoSidedBet(F(1, 2))
        self.old_x = None
        self.outcome_counts = {'old': [0]*8, 'new': [0]*8}

    def next_action(self):
        self._before_action()
        config = 'old' if self.shots % 2 == 0 else 'new'
        return self._issue({'kind': 'three_bit_target', 'config': config, 'duration_exact': '1'})

    def observe(self, action_id, outcome):
        self._validate_delivery(action_id, outcome, 7)
        config = self.pending['config']
        self.outcome_counts[config][outcome] += 1
        if config == 'old':
            self.old_x = outcome & 1
        else:
            self.evidence.observe(self.old_x-(outcome & 1))
            self.old_x = None
        self._delivered(outcome)
        if config == 'new':
            self._stop(self.evidence.certificate(), 'accept')
        return self.summary()

    def summary(self):
        result = super().summary()
        result.update({'complete_three_bit_counts': {k: list(v) for k, v in self.outcome_counts.items()},
                       'readout_bits_charged': 3*self.shots,
                       'model_contract': 'same_duration_disjoint_invariant_spectators with s>0',
                       'power_scope': 'Reviewed cost/power example x_old=.03,x_new=.01,s=.5 at cap131072 only; no uniform eligible-domain power claim'})
        return result
