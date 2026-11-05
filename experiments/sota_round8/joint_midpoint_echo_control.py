"""Paid joint betting on midpoint and echo outcomes; no private truth inputs.

This is a classical same-action candidate, not a SOTA claim. Four current
products are used; evidence that crossed earlier is never latched. The
physical sign certificate requires the explicit public contract below.
"""
from fractions import Fraction as F
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sota_round7.resource_controls import (
    _PaidControl, TwoSidedBet, integer,
)

CONTRACT = 'round8_midpoint_echo_full_time_9_ideal_primitives'
DEFAULT_CAP = 65536


class JointEvidence:
    """Four rational wealth processes, updated after each paid outcome."""
    def __init__(self):
        self.same = TwoSidedBet(F(3, 25))
        self.opposite = TwoSidedBet(F(3, 25))
        self.counts = {'midpoint': [0, 0], 'echo': [0, 0]}

    def observe(self, kind, z):
        if kind not in self.counts:
            raise ValueError('Unregistered measurement kind')
        if isinstance(z, bool) or not isinstance(z, int) or z not in (-1, 1):
            raise ValueError('Demodulated outcome must be integer -1 or +1')
        self.counts[kind][(z+1)//2] += 1
        self.same.observe(z)
        self.opposite.observe(z if kind == 'midpoint' else -z)

    def certificate(self):
        same, opposite = self.same.certificate(), self.opposite.certificate()
        a, b = same['crossing_sign'], opposite['crossing_sign']
        direction = a if a != 0 and a == b else 0
        return {
            'crossing_sign': direction,
            'rule': 'Both CURRENT products for one direction must be >= 200',
            'same_echo_sign_products': same,
            'opposite_echo_sign_products': opposite,
            'counts_minus_plus': {k: list(v) for k, v in self.counts.items()},
        }


class JointMidpointEchoControl(_PaidControl):
    """Issue fresh randomized actions and accept exactly one outcome each.

    Public assumptions: fixed old/new amplitudes 2 asin sqrt(5/198) and
    2 asin sqrt(1/198); disjoint H(u)=((omega+u)/2)X0+
    (gamma+chi*u)P11; |chi|<=.2; symmetric readout r<=.05;
    spectator dephasing kappa<=.005; ideal known preparation and analysis;
    no extra q0 dynamics/noise; echo contrast >=.9 exp(-.005*9).
    No probability/power guarantee is inherited by a fixed seeded replay.
    """
    def __init__(self, model_contract, seed=None, cap=DEFAULT_CAP):
        if model_contract != CONTRACT:
            raise ValueError('The explicit reviewed physical contract is required')
        if seed is not None:
            integer(seed, 'seed')
        super().__init__(cap)
        self.evidence = JointEvidence()
        self.rng = random.SystemRandom() if seed is None else random.Random(seed)
        self.randomness_mode = ('system_random_bits' if seed is None
                                else 'seeded_pseudorandom_replay')
        self.kind_counts = {'midpoint': 0, 'echo': 0}
        self.sensing_windows = self.x1_pulses = self.x2_pulses = 0
        self.waveform_configurations = self.internal_amplitude_transitions = 0
        self.device_tenths = 0
        self.last_configuration = None

    def next_action(self):
        self._before_action()
        # randrange implements exact discrete uniform coins, without a
        # floating probability comparison. Device outcomes follow the draw.
        kind = 'midpoint' if self.rng.randrange(7) < 4 else 'echo'
        sign = 2*self.rng.randrange(2)-1
        configuration = (kind, sign)
        switch = int(configuration != self.last_configuration)
        self.last_configuration = configuration
        self.waveform_configurations += switch
        self.kind_counts[kind] += 1
        if kind == 'midpoint':
            self.sensing_windows += 1
            self.device_tenths += 32+switch
            detail = {
                'amplitude': '(2*asin(sqrt(5/198))+2*asin(sqrt(1/198)))/2',
                'window_duration_exact': '1', 'sensing_windows': 1,
                'preparation': 'reset |000>',
                'analysis': 'Rx(phase_sign*pi/2) on q0',
                'statistic_bit': 0, 'x1_echo_pulses': 0,
                'x2_preparation_pulses': 0,
            }
        else:
            self.sensing_windows += 6
            self.x1_pulses += 6
            self.x2_pulses += 1
            self.internal_amplitude_transitions += 5
            self.device_tenths += 89+switch
            detail = {
                'preparation': 'reset |000>, X2, H1',
                'window_amplitudes_exact': ['-1', '1']*3,
                'window_duration_exact': '1', 'sensing_windows': 6,
                'pulse_after_every_window': 'X1',
                'analysis': 'Rx(-phase_sign*pi/2) on q1',
                'statistic_bit': 1, 'x1_echo_pulses': 6,
                'x2_preparation_pulses': 1,
            }
        detail.update(kind=kind, phase_sign=sign,
                      readout='all three bits, q0 least significant',
                      waveform_configuration_charged=switch)
        return self._issue(detail)

    def observe(self, action_id, outcome):
        self._validate_delivery(action_id, outcome, 7)
        action = self.pending
        bit = (outcome >> action['statistic_bit']) & 1
        z = action['phase_sign']*(2*bit-1)
        self.evidence.observe(action['kind'], z)
        self._delivered(outcome)
        self._stop(self.evidence.certificate(), 'accept')
        return self.summary()

    def summary(self):
        result = super().summary()
        result.update({
            'model_contract': CONTRACT,
            'randomness_mode': self.randomness_mode,
            'randomness_assumption': 'Independent fresh kind coins (4/7 midpoint, 3/7 echo), independent fair phase coins, and conditionally physical outcomes are required.',
            'kind_attempt_counts': dict(self.kind_counts),
            'readout_bits_charged': 3*self.shots,
            'sensing_window_time_exact': str(self.sensing_windows),
            'x1_echo_pulses': self.x1_pulses,
            'x2_preparation_pulses': self.x2_pulses,
            'preparation_and_analysis_layers': 2*self.shots,
            'waveform_configurations': self.waveform_configurations,
            'internal_amplitude_transitions': self.internal_amplitude_transitions,
            'device_time_exact': str(F(self.device_tenths, 10)),
            'device_cost_model': 'Per issued shot: setup/reset/readout and preparation/analysis layers 2.2; each sensing window 1; each X pulse .1; waveform/phase reconfiguration .1. Internal waveform amplitude transitions are counted and included in the sensing-window price. Prices are an explicit simulation convention, not measured hardware timings.',
            'device_time_maximum_per_shot': '9',
            'conservative_device_score_exact': str(F(self.device_tenths, 10)) if self.status == 'decided' else str(9*self.cap),
            'conservative_device_score_scope': 'Undecided records receive the worst-action full-cap score; this penalty is not actual executed device time. The shot-cost theorem is not a theorem for this additional device score.',
            'risk_scope': 'At most .005 for each wrong direction, at most .01 for any report at equality, under the registered physical and randomization contract; no arbitrary pulse-error or q0-noise extension.',
            'power_scope': ('Fixed seeded replay has no inherited probabilistic power certificate' if self.randomness_mode != 'system_random_bits' else 'Only the reference and sign-mirrored reference, with no execution failures or interruptions' if self.cap == DEFAULT_CAP else 'Custom cap has no inherited 95 percent power certificate'),
            'benchmark_status': 'Development candidate; no independent benchmark or SOTA superiority established',
        })
        return result
