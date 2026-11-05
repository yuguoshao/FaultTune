"""Graph-constrained simultaneous Bernoulli measurements, Python 3.8+.

PublicGroupConfig contains only quantities available to a measurement policy.
The runner gives a policy that config and oracle.measure, never the private
probability vector or oracle.audit. This is an API convention, not a security
sandbox against introspection of arbitrary Python objects.

One successful measure(group) call costs one round and len(group) object
observations. The group must be an independent set of the public conflict
graph. The returned tuple follows SORTED arm indices, regardless of the order
in which the action was supplied. Budget or transcript preflight failures
produce no observations and consume neither counters nor random numbers.
"""

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from .probe import BudgetExhausted, TranscriptExhausted, stable_seed
except ImportError:
    from probe import BudgetExhausted, TranscriptExhausted, stable_seed


Edges = Tuple[Tuple[int, int], ...]
Partition = Tuple[Tuple[int, ...], ...]


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("{} must be a positive integer".format(name))


def _arm_index(index: int, n: int) -> None:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < n:
        raise ValueError("arm index must be an integer in [0, n)")


def canonical_group(group: Iterable[int], n: int) -> Tuple[int, ...]:
    """Validate bounds/nonemptiness/uniqueness, then sort an action's indices."""
    _positive_integer(n, "n")
    try:
        items = tuple(group)
    except TypeError as error:
        raise ValueError("group must be an iterable of arm indices") from error
    if not items:
        raise ValueError("group must not be empty")
    for index in items:
        _arm_index(index, n)
    if len(set(items)) != len(items):
        raise ValueError("group must not contain duplicate indices")
    return tuple(sorted(items))


def canonical_edges(n: int, edges: Iterable[Sequence[int]]) -> Edges:
    """Canonical undirected edges; repeated/reversed edges collapse to one."""
    _positive_integer(n, "n")
    unique = set()
    try:
        for edge in edges:
            pair = tuple(edge)
            if len(pair) != 2:
                raise ValueError("each conflict edge must contain two indices")
            left, right = pair
            _arm_index(left, n)
            _arm_index(right, n)
            if left == right:
                raise ValueError("conflict graph must not contain self-loops")
            unique.add((min(left, right), max(left, right)))
    except TypeError as error:
        raise ValueError("conflict_edges must be an iterable of index pairs") from error
    return tuple(sorted(unique))


def _adjacency(n: int, edges: Edges) -> Tuple[frozenset, ...]:
    neighbours = [set() for _ in range(n)]
    for left, right in edges:
        neighbours[left].add(right)
        neighbours[right].add(left)
    return tuple(frozenset(row) for row in neighbours)


def _independent(group: Tuple[int, ...], neighbours: Tuple[frozenset, ...]) -> bool:
    members = frozenset(group)
    return not any(neighbours[index].intersection(members) for index in group)


def greedy_partition(n: int, conflict_edges: Iterable[Sequence[int]] = ()) -> Partition:
    """Stable first-fit coloring: ascending arm order, smallest available color."""
    edges = canonical_edges(n, conflict_edges)
    neighbours = _adjacency(n, edges)
    colors = [-1] * n
    groups = []  # type: List[List[int]]
    for index in range(n):
        forbidden = {colors[j] for j in neighbours[index] if colors[j] >= 0}
        color = 0
        while color in forbidden:
            color += 1
        if color == len(groups):
            groups.append([])
        groups[color].append(index)
        colors[index] = color
    return tuple(tuple(group) for group in groups)


def validate_partition(n: int, conflict_edges: Iterable[Sequence[int]], partition: Iterable[Iterable[int]]) -> Partition:
    """Require a disjoint cover of all arms, with each group independent."""
    edges = canonical_edges(n, conflict_edges)
    neighbours = _adjacency(n, edges)
    try:
        groups = tuple(canonical_group(group, n) for group in partition)
    except TypeError as error:
        raise ValueError("partition must be an iterable of groups") from error
    seen = set()
    for group in groups:
        if not _independent(group, neighbours):
            raise ValueError("partition contains a group with a conflict edge")
        if seen.intersection(group):
            raise ValueError("partition groups must be disjoint")
        seen.update(group)
    if seen != set(range(n)):
        raise ValueError("partition must cover every arm exactly once")
    return groups


@dataclass(frozen=True)
class PublicGroupConfig:
    n: int
    p0: float = 0.1
    local_gain: float = 0.15
    delta: float = 0.05
    conflict_edges: Edges = ()
    partition: Optional[Partition] = None

    def __post_init__(self) -> None:
        _positive_integer(self.n, "n")
        if not (0.0 < self.p0 < 1.0):
            raise ValueError("p0 must lie strictly between zero and one")
        if not math.isfinite(self.local_gain) or self.local_gain <= 0.0:
            raise ValueError("local_gain must be finite and positive")
        if not (0.0 < self.delta < 0.5):
            raise ValueError("delta must lie strictly between zero and one half")
        edges = canonical_edges(self.n, self.conflict_edges)
        supplied = greedy_partition(self.n, edges) if self.partition is None else self.partition
        groups = validate_partition(self.n, edges, supplied)
        # Frozen is also deep with respect to the caller's graph/partition
        # lists: no mutable list supplied by the caller is retained.
        object.__setattr__(self, "conflict_edges", edges)
        object.__setattr__(self, "partition", groups)

    def validate_group(self, group: Iterable[int]) -> Tuple[int, ...]:
        action = canonical_group(group, self.n)
        members = frozenset(action)
        if any(left in members and right in members for left, right in self.conflict_edges):
            raise ValueError("measurement group is not an independent set")
        return action


def empty_graph(n: int) -> Edges:
    _positive_integer(n, "n")
    return ()


def path_graph(n: int) -> Edges:
    _positive_integer(n, "n")
    return tuple((index, index + 1) for index in range(n - 1))


def clique_graph(n: int) -> Edges:
    _positive_integer(n, "n")
    return tuple((left, right) for left in range(n) for right in range(left + 1, n))


def grid_graph(rows: int, columns: int) -> Edges:
    """Open rectangular grid, orthogonal neighbours, row-major arm indices."""
    _positive_integer(rows, "rows")
    _positive_integer(columns, "columns")
    edges = []  # type: List[Tuple[int, int]]
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            if column + 1 < columns:
                edges.append((index, index + 1))
            if row + 1 < rows:
                edges.append((index, index + columns))
    return tuple(sorted(edges))


def singleton_partition(n: int) -> Partition:
    _positive_integer(n, "n")
    return tuple((index,) for index in range(n))


def full_partition(n: int, conflict_edges: Iterable[Sequence[int]] = ()) -> Partition:
    """All arms in one group; rejected if any conflict edge is present."""
    _positive_integer(n, "n")
    return validate_partition(n, conflict_edges, (tuple(range(n)),))


def checkerboard_partition(rows: int, columns: int) -> Partition:
    """Two legal color classes of grid_graph; omit empty classes for 1x1."""
    edges = grid_graph(rows, columns)
    groups = [[], []]  # type: List[List[int]]
    for row in range(rows):
        for column in range(columns):
            groups[(row + column) % 2].append(row * columns + column)
    return validate_partition(rows * columns, edges, (group for group in groups if group))


def _budget(value: Optional[int], name: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise ValueError("{} must be None or a nonnegative integer".format(name))


class GroupMeasurementOracle:
    """Private Bernoulli environment for legal simultaneous actions.

    All budgets count observations actually delivered. Switches count changes
    in the canonical action, excluding the first successful action. They do
    not add a time cost unless a downstream evaluator explicitly assigns one.
    A public partition is a suggested legal schedule, not an action whitelist:
    any independent set is measurable, including legal groups spanning colors.
    """

    def __init__(
        self,
        config: PublicGroupConfig,
        probabilities: Sequence[float],
        seed: int,
        round_budget: Optional[int] = None,
        object_budget: Optional[int] = None,
        transcripts: Optional[Mapping[int, Sequence[int]]] = None,
    ) -> None:
        if not isinstance(config, PublicGroupConfig):
            raise ValueError("config must be a PublicGroupConfig")
        actual = tuple(probabilities)
        if len(actual) != config.n or any(not 0.0 <= value <= 1.0 for value in actual):
            raise ValueError("probabilities must contain n finite values in [0, 1]")
        _budget(round_budget, "round_budget")
        _budget(object_budget, "object_budget")
        self._config = config
        self._probabilities = actual
        self._seed = seed
        self._round_budget = round_budget
        self._object_budget = object_budget
        self._counts = [0] * config.n
        self._streams = [None] * config.n  # type: List[Optional[random.Random]]
        self._rounds = 0
        self._objects = 0
        self._switches = 0
        self._last_action = None  # type: Optional[Tuple[int, ...]]
        self._neighbours = _adjacency(config.n, config.conflict_edges)
        self._transcripts = None  # type: Optional[Dict[int, Tuple[int, ...]]]
        if transcripts is not None:
            copied = {}  # type: Dict[int, Tuple[int, ...]]
            for index, values in transcripts.items():
                _arm_index(index, config.n)
                observations = tuple(values)
                if any(value not in (0, 1) for value in observations):
                    raise ValueError("transcript observations must be zero or one")
                copied[index] = tuple(int(value) for value in observations)
            self._transcripts = copied

    def measure(self, group: Iterable[int]) -> Tuple[int, ...]:
        action = canonical_group(group, self._config.n)
        if not _independent(action, self._neighbours):
            raise ValueError("measurement group is not an independent set")
        if self._round_budget is not None and self._rounds + 1 > self._round_budget:
            raise BudgetExhausted("group measurement would exceed round budget")
        if self._object_budget is not None and self._objects + len(action) > self._object_budget:
            raise BudgetExhausted("group measurement would exceed object budget")
        if self._transcripts is not None:
            # Check ALL arms before delivering any observation or advancing an
            # arm count. A missing late-arm transcript cannot partially charge.
            for index in action:
                if self._counts[index] >= len(self._transcripts.get(index, ())):
                    raise TranscriptExhausted("arm {} has no observation {}".format(index, self._counts[index]))
            observations = tuple(self._transcripts[index][self._counts[index]] for index in action)
        else:
            draws = []  # type: List[int]
            for index in action:
                stream = self._streams[index]
                if stream is None:
                    stream = random.Random(stable_seed(self._seed, "arm", index))
                    self._streams[index] = stream
                draws.append(int(stream.random() < self._probabilities[index]))
            observations = tuple(draws)
        self._rounds += 1
        self._objects += len(action)
        if self._last_action is not None and action != self._last_action:
            self._switches += 1
        self._last_action = action
        for index in action:
            self._counts[index] += 1
        return observations

    @property
    def rounds(self) -> int:
        return self._rounds

    @property
    def objects(self) -> int:
        return self._objects

    @property
    def switches(self) -> int:
        return self._switches

    @property
    def counts(self) -> Tuple[int, ...]:
        return tuple(self._counts)

    @property
    def covered(self) -> int:
        return sum(count > 0 for count in self._counts)

    @property
    def max_samples(self) -> int:
        return max(self._counts)

    def audit(self) -> Dict[str, Any]:
        remote_change = math.fsum(probability - self._config.p0 for probability in self._probabilities)
        change = math.fsum((remote_change, -self._config.local_gain))
        return {
            "probabilities": self._probabilities,
            "remote_change": remote_change,
            "true_delta": change,
            "expected_decision": "accept" if change <= 0.0 else "reject",
            "observation_source": "random" if self._transcripts is None else "fixed_transcripts",
        }
