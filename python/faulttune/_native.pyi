"""Type stubs of the native extension module ``faulttune._native``.

The stubs are checked against the compiled module with ``python -m mypy.stubtest
faulttune._native`` and by ``python/tests/test_typing.py``, which also compares the public
names and the parameter names of this file with the runtime: adding or changing an argument
of the extension without changing this file fails the tests.

Names that start with an underscore are helper types that exist only here (type aliases, the
protocols of duck-typed user code, the shape of ``Snake.config``). They are importable under
``typing.TYPE_CHECKING`` from this module for annotations, but they do not exist at runtime.
"""

import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import (
    Any,
    ClassVar,
    Literal,
    Protocol,
    Self,
    TypeAlias,
    TypedDict,
    TypeVar,
    final,
    overload,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "FaultTuneError",
    "InvalidGraphError",
    "InvalidConfigError",
    "SearchSpaceTooLargeError",
    "ModelError",
    "NonFiniteError",
    "NoProgressError",
    "GraphMismatchError",
    "CancelledError",
    "FormatError",
    "__version__",
    "ElementKind",
    "Element",
    "ElementSet",
    "ProcessorGraph",
    "GraphBuilder",
    "Step",
    "StepRecord",
    "Invalidation",
    "Report",
    "TraversalContext",
    "CoordinateDescent",
    "Exhaustive",
    "RandomSearch",
    "SearchSpace",
    "Objective",
    "ParameterDatabase",
    "Retry",
    "AlgorithmSubgraphs",
    "StitchPlan",
    "Snake",
    "read_journal",
]

__version__: str

_T = TypeVar("_T")

# ---------------------------------------------------------------------------------------
# Helper types that exist only in the stubs
# ---------------------------------------------------------------------------------------

_PathLike: TypeAlias = str | os.PathLike[str]
"""A file system path, as accepted by every ``path`` argument."""

_ElementLike: TypeAlias = Element | str
"""An element or its label."""

_Elements: TypeAlias = ElementSet | Iterable[Element]
"""An :class:`ElementSet` or any iterable of :class:`Element` of the same graph."""

_KindOrKinds: TypeAlias = ElementKind | Iterable[ElementKind]
"""One element kind or an iterable of kinds."""

_IntegerArray: TypeAlias = Sequence[int] | Sequence[Sequence[int]] | NDArray[np.integer[Any]]
"""Option indices: a 1-D (one candidate) or 2-D (a batch of candidates) integer array."""

class _ErrorModelLike(Protocol):
    """What :meth:`Snake.calibrate` needs from an error model (duck typing).

    Subclasses of :class:`faulttune.ErrorModel` satisfy it. A model may also define
    ``prepare(step)``; it is called once per step before any ``error`` call, and what it
    returns is available as ``step.data``.
    """

    def options(self, element: Element, /) -> ArrayLike:
        """The candidate values of one element: a non-empty 1-D array of finite floats."""

    def error(self, step: Step, candidates: NDArray[np.float64], /) -> ArrayLike:
        """The errors of ``candidates`` (shape ``(n, len(step.parameters))``) as shape ``(n,)``."""

class _PriorityHeuristic(Protocol):
    """A traversal heuristic with ``frontier="priority"``: it also scores elements."""

    frontier: Literal["priority"]

    def __call__(
        self, context: TraversalContext, options: list[Element], /
    ) -> Iterable[Element]: ...
    def priority(self, context: TraversalContext, element: Element, /) -> float:
        """The priority of ``element``; the frontier is walked in ascending priority."""

_HeuristicFunction: TypeAlias = Callable[[TraversalContext, list[Element]], Iterable[Element]]
"""A custom traversal heuristic: ``(context, options) -> the options in the order to visit``.

The callable may carry a ``frontier`` attribute (``"fifo"`` (default), ``"lifo"`` or
``"priority"``) and, for ``"priority"``, a ``priority(context, element)`` method. It must
return exactly the elements it was given, in the order they are to be visited.
"""

_OptimizerFunction: TypeAlias = Callable[
    [SearchSpace, Objective, np.random.Generator],
    tuple[Sequence[int] | NDArray[np.integer[Any]], float | np.floating[Any]],
]
"""A custom step optimizer: ``(space, objective, rng) -> (indices, error)``."""

_SeedFunction: TypeAlias = Callable[[list[Element]], Element]
"""A custom seed policy: ``(thread_elements) -> the element to start from``."""

_Heuristic: TypeAlias = (
    Literal["breadth_first", "depth_first", "random"] | _HeuristicFunction | _PriorityHeuristic
)
"""The ``heuristic`` argument of :class:`Snake`."""

_Optimizer: TypeAlias = (
    Literal["coordinate_descent", "exhaustive", "random_search"]
    | CoordinateDescent
    | Exhaustive
    | RandomSearch
    | _OptimizerFunction
)
"""The ``optimizer`` argument of :class:`Snake`."""

_SeedPolicy: TypeAlias = Literal["lowest_id", "random"] | _Elements | _SeedFunction
"""The ``seed_policy`` argument of :class:`Snake`."""

_ModelErrorPolicy: TypeAlias = Literal["abort", "skip"] | tuple[Literal["retry"], int] | Retry
"""The ``on_model_error`` argument of :class:`Snake`."""

_CachePolicy: TypeAlias = Literal["step", "across_steps"] | int
"""The ``cache`` argument of :class:`Snake`: a mode or the capacity of the cross-step cache."""

_OnStep: TypeAlias = Callable[[StepRecord], object]
"""The ``on_step`` callback: called on the calling thread with every committed step."""

class _SnakeConfig(TypedDict):
    """The dict returned by :attr:`Snake.config` (a copy; the arguments as they were resolved)."""

    parameter_distance: int
    constraint_distance: int
    traversal_distance: int
    expiry_radius: int
    seed: int
    threads: int | None
    on_model_error: Literal["abort", "skip"] | tuple[Literal["retry"], int]
    cache: Literal["step"] | int
    evaluation_budget: int
    heuristic: _Heuristic
    optimizer: _Optimizer
    seed_policy: _SeedPolicy
    algorithm: AlgorithmSubgraphs | None

# ---------------------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------------------

class FaultTuneError(Exception):
    """Base class of every error raised by FaultTune itself."""

    partial: ParameterDatabase | None
    """The database with every step committed before the failure.

    ``calibrate``, ``recalibrate`` and ``stitch`` attach it to the error they raise (also to
    exceptions that are not FaultTune errors, such as the ones raised by ``on_step``). It is
    missing on errors raised elsewhere, for example by ``ProcessorGraph.from_json``.
    """

class InvalidGraphError(FaultTuneError, ValueError):
    """The processor graph is malformed."""

class InvalidConfigError(FaultTuneError, ValueError):
    """The configuration, goal or another input is invalid."""

class SearchSpaceTooLargeError(FaultTuneError):
    """The exhaustive optimizer was asked to search too many combinations."""

    size: int
    budget: int

class ModelError(FaultTuneError):
    """The error model (or another piece of user code) failed.

    The original exception is the ``__cause__``.
    """

    element: Element | None

class NonFiniteError(FaultTuneError):
    """The error model returned NaN."""

    element: Element | None

class NoProgressError(FaultTuneError):
    """A calibration pass made no progress."""

class GraphMismatchError(FaultTuneError, ValueError):
    """A saved calibration belongs to a different graph."""

class CancelledError(FaultTuneError):
    """The calibration was cancelled. Every committed step is kept."""

    report: Report | None

class FormatError(FaultTuneError, ValueError):
    """A file or string is not in a recognised format."""

# ---------------------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------------------

@final
class ElementKind:
    """What an element is: a node or an edge of some type."""

    NODE: ClassVar[ElementKind]
    """A node, usually a qubit."""
    ENGINEERED_EDGE: ClassVar[ElementKind]
    """A designed coupling."""
    PARASITIC_EDGE: ClassVar[ElementKind]
    """Unwanted crosstalk: carries constraints but has no parameters of its own."""
    CUSTOM_EDGE: ClassVar[ElementKind]
    """A user-defined edge type (see :attr:`Element.custom_type`)."""

    def __int__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object, /) -> bool: ...

@final
class Element:
    """A node or an edge of a processor graph: a hashable, comparable handle.

    Elements are equal when they have the same id in the same graph. They order by id within
    one graph. ``str(element)`` is the label.
    """

    @property
    def label(self) -> str:
        """The label, for example ``"q1_3"`` or ``"q1_3-q1_4"``."""
    @property
    def kind(self) -> ElementKind:
        """The kind of the element."""
    @property
    def index(self) -> int:
        """The dense integer id of the element within its graph."""
    @property
    def endpoints(self) -> tuple[Element, Element] | None:
        """The two end nodes of an edge, or ``None`` for a node."""
    @property
    def custom_type(self) -> int | None:
        """The number of a user-defined edge type, or ``None`` for other elements."""
    @property
    def is_node(self) -> bool:
        """Whether this element is a node."""
    @property
    def is_edge(self) -> bool:
        """Whether this element is an edge."""

    def __str__(self) -> str: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...
    def __lt__(self, other: Element, /) -> bool: ...
    def __le__(self, other: Element, /) -> bool: ...
    def __gt__(self, other: Element, /) -> bool: ...
    def __ge__(self, other: Element, /) -> bool: ...

@final
class ElementSet:
    """An immutable set of elements of one graph.

    Supports ``len``, iteration (ascending id order), ``in``, ``|``, ``&``, ``-``, ``^``, the
    subset comparisons and :meth:`isdisjoint`. Operators accept another ``ElementSet`` or any
    iterable of ``Element`` of the same graph.
    """

    @property
    def graph(self) -> ProcessorGraph:
        """The graph these elements belong to."""

    def __len__(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __iter__(self) -> Iterator[Element]: ...
    def __contains__(self, item: object, /) -> bool: ...
    def __or__(self, other: _Elements, /) -> ElementSet: ...
    def __ror__(self, other: _Elements, /) -> ElementSet: ...
    def __and__(self, other: _Elements, /) -> ElementSet: ...
    def __rand__(self, other: _Elements, /) -> ElementSet: ...
    def __sub__(self, other: _Elements, /) -> ElementSet: ...
    def __rsub__(self, other: _Elements, /) -> ElementSet: ...
    def __xor__(self, other: _Elements, /) -> ElementSet: ...
    def __rxor__(self, other: _Elements, /) -> ElementSet: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...
    def __lt__(self, other: ElementSet, /) -> bool: ...
    def __le__(self, other: ElementSet, /) -> bool: ...
    def __gt__(self, other: ElementSet, /) -> bool: ...
    def __ge__(self, other: ElementSet, /) -> bool: ...
    def __hash__(self) -> int: ...
    def isdisjoint(self, other: _Elements) -> bool:
        """Whether the set shares no element with ``other``."""
    def to_list(self) -> list[Element]:
        """The elements as a list, in ascending id order."""

@final
class ProcessorGraph:
    """An immutable processor graph: nodes and typed edges, all addressed as elements.

    Build one with :meth:`square_lattice`, :meth:`from_json` or :class:`GraphBuilder`.
    """

    @staticmethod
    def square_lattice(rows: int, cols: int) -> ProcessorGraph:
        """A ``rows x cols`` square lattice.

        Nodes ``q{row}_{col}`` come first (row-major), then the nearest-neighbour engineered
        edges, then the diagonal parasitic edges.
        """
    @staticmethod
    def from_json(text: str) -> ProcessorGraph:
        """Parses a graph from its JSON description (see :meth:`to_json`)."""
    def to_json(self) -> str:
        """The JSON description of the graph."""
    def node(self, row: int, col: int) -> Element:
        """The node ``q{row}_{col}`` of a lattice-labelled graph (``KeyError`` if none)."""
    def edge(self, a: _ElementLike, b: _ElementLike, kind: ElementKind | None = None) -> Element:
        """The edge joining nodes ``a`` and ``b`` (``KeyError`` if there is none)."""
    def element(self, label: str) -> Element:
        """The element with this label (``KeyError`` if there is none)."""
    @overload
    def __getitem__(self, key: str, /) -> Element: ...
    @overload
    def __getitem__(self, key: int, /) -> Element: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Element]: ...
    def __contains__(self, item: object, /) -> bool: ...
    def nodes(self) -> list[Element]:
        """All nodes, in id order."""
    def edges(self) -> list[Element]:
        """All edges, in id order."""
    def elements(self) -> list[Element]:
        """All elements, in id order."""
    @property
    def fingerprint(self) -> str:
        """A hash of the labels, kinds and incidence (16 hex digits)."""
    @property
    def coords(self) -> NDArray[np.float32] | None:
        """Layout coordinates, shape ``(n_elements, 2)``, or ``None``."""
    def goal(
        self,
        exclude: _KindOrKinds | None = None,
        include: _KindOrKinds | None = None,
        predicate: Callable[[Element], object] | None = None,
    ) -> ElementSet:
        """The set of elements to calibrate.

        An element is selected when its kind is in ``include`` (default: every kind), not in
        ``exclude``, and ``predicate(element)`` is true (default: always).
        """
    def all_elements(self) -> ElementSet:
        """The set of every element."""
    def element_set(self, elements: _Elements) -> ElementSet:
        """An :class:`ElementSet` of this graph from any iterable of its elements."""
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class GraphBuilder:
    """Builds a :class:`ProcessorGraph` of any topology. Elements get ids in insertion order."""

    def __new__(cls) -> Self: ...
    def add_node(self, label: str, pos: tuple[float, float] | None = None) -> str:
        """Adds a node and returns its label; ``pos`` is an optional ``(x, y)`` coordinate."""
    def add_edge(
        self,
        a: str,
        b: str,
        kind: ElementKind = ...,
        *,
        label: str | None = None,
        custom_type: int | None = None,
    ) -> str:
        """Adds an edge between the nodes labelled ``a`` and ``b`` and returns its label.

        ``kind`` is ``ENGINEERED_EDGE`` (default), ``PARASITIC_EDGE`` or ``CUSTOM_EDGE``; a
        custom edge also needs ``custom_type``. The label defaults to ``"{a}-{b}"``.
        """
    def build(self) -> ProcessorGraph:
        """Finishes the graph. The builder cannot be used afterwards."""

# ---------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------

@final
class Step:
    """One calibration step as seen by the error model."""

    @property
    def index(self) -> int:
        """Ordinal of the step within its subgoal and pass, starting at 0."""
    @property
    def centre(self) -> Element:
        """The element the step is centred on."""
    @property
    def parameters(self) -> list[Element]:
        """The parameters being optimized: the columns of every candidate, in this order."""
    @property
    def constraints(self) -> dict[Element, float]:
        """The active constraints as ``{element: fixed value}``."""
    @property
    def data(self) -> Any:
        """What ``prepare`` returned (``None`` while it runs, and if it returned nothing)."""
    @property
    def graph(self) -> ProcessorGraph:
        """The processor graph."""

@final
class StepRecord:
    """One committed calibration step."""

    @property
    def index(self) -> int:
        """Position in the history (or the ordinal within the subgoal in ``on_step``)."""
    @property
    def pass_(self) -> int:
        """The pass of the ``calibrate`` call that ran the step (0-based)."""
    @property
    def pass_index(self) -> int:
        """Alias of :attr:`pass_`."""
    @property
    def subgoal(self) -> int:
        """The subgoal within the pass (0-based)."""
    @property
    def centre(self) -> Element:
        """The element the step was centred on."""
    @property
    def parameters(self) -> list[Element]:
        """The parameters that were optimized, in candidate-column order."""
    @property
    def option_indices(self) -> list[int]:
        """The committed option index of each parameter."""
    @property
    def constraints(self) -> list[Element]:
        """The elements that constrained the step."""
    @property
    def error(self) -> float:
        """The error of the committed assignment (possibly ``inf``)."""
    @property
    def evaluations(self) -> int:
        """Number of distinct candidates evaluated."""
    @property
    def duration(self) -> float:
        """Total time of the step in seconds."""
    @property
    def prepare_time(self) -> float:
        """Time spent in ``prepare``, in seconds."""
    @property
    def error_time(self) -> float:
        """Time spent evaluating errors, in seconds."""
    @property
    def values(self) -> dict[Element, float] | None:
        """The committed values as ``{parameter: value}``, or ``None`` if not known."""
    def same_outcome(self, other: StepRecord) -> bool:
        """Whether ``other`` describes the same outcome, ignoring the timings."""

@final
class Invalidation:
    """Elements removed from the calibration by ``recalibrate``."""

    @property
    def elements(self) -> list[Element]:
        """The elements that were forgotten, ascending."""

@final
class Report:
    """Summary of a calibration run."""

    @property
    def passes(self) -> int:
        """Number of passes of the completion loop."""
    @property
    def steps(self) -> int:
        """Number of steps committed."""
    @property
    def evaluations(self) -> int:
        """Total candidates evaluated."""
    @property
    def cache_hits(self) -> int:
        """Cross-step cache hits (0 when the cache is off)."""
    @property
    def cache_misses(self) -> int:
        """Cross-step cache misses (0 when the cache is off)."""
    @property
    def cache_hit_rate(self) -> float | None:
        """Fraction of cross-step cache lookups that hit, or ``None`` if there were none."""
    @property
    def wall_time(self) -> float:
        """Wall-clock time of the run in seconds."""
    @property
    def prepare_time(self) -> float:
        """Total time spent in ``prepare``, summed over steps, in seconds."""
    @property
    def error_time(self) -> float:
        """Total time spent evaluating errors, summed over steps, in seconds."""
    @property
    def skipped(self) -> list[Element]:
        """Elements given up on after a model error (``on_model_error="skip"``)."""
    @property
    def worst_steps(self) -> list[StepRecord]:
        """The steps with the largest error of the run, worst first (at most 10)."""

# ---------------------------------------------------------------------------------------
# Traversal and optimizers
# ---------------------------------------------------------------------------------------

@final
class TraversalContext:
    """What a heuristic sees: the graph, the calibration status and the step history.

    It is only valid during the heuristic call it is passed to.
    """

    @property
    def graph(self) -> ProcessorGraph:
        """The processor graph."""
    @property
    def current(self) -> Element:
        """The element whose step was just committed."""
    def is_calibrated(self, element: Element) -> bool:
        """Whether ``element`` is calibrated."""
    @property
    def calibrated_count(self) -> int:
        """Number of calibrated elements."""
    @property
    def calibrated(self) -> ElementSet:
        """The calibrated elements."""
    def history(self) -> list[StepRecord]:
        """The steps committed so far by the current subgoal in this pass, in commit order."""

@final
class CoordinateDescent:
    """Coordinate descent (the default optimizer): sweeps one parameter at a time."""

    def __new__(
        cls, random_starts: int = 4, max_sweeps: int = 3, max_start_attempts: int = 20_000
    ) -> Self: ...
    @property
    def random_starts(self) -> int:
        """Random starting points evaluated first."""
    @property
    def max_sweeps(self) -> int:
        """Maximum number of sweeps over all parameters."""
    @property
    def max_start_attempts(self) -> int:
        """Extra random points tried when every start is forbidden."""

@final
class Exhaustive:
    """Exhaustive search: tries every combination, in batches."""

    def __new__(cls, batch_size: int = 4096, max_size: int = 1000000) -> Self: ...
    @property
    def batch_size(self) -> int:
        """Candidates per objective call."""
    @property
    def max_size(self) -> int:
        """Largest search space that is searched."""

@final
class RandomSearch:
    """Random search: samples a fixed number of candidates."""

    def __new__(cls, samples: int = 1000, batch_size: int = 1000) -> Self: ...
    @property
    def samples(self) -> int:
        """Number of candidates sampled."""
    @property
    def batch_size(self) -> int:
        """Candidates per objective call."""

@final
class SearchSpace:
    """The search space of one step, as handed to a custom optimizer."""

    @property
    def sizes(self) -> list[int]:
        """Number of options of each parameter, in candidate-column order."""
    @property
    def budget(self) -> int:
        """The most candidates a budget-aware optimizer should evaluate."""
    @property
    def width(self) -> int:
        """Number of parameters."""
    @property
    def size(self) -> int:
        """Number of combinations (the product of ``sizes``)."""

@final
class Objective:
    """The objective handed to a custom optimizer.

    ``objective(indices)`` takes an integer array of shape ``(n, width)`` (or ``(width,)``)
    of option indices and returns the float64 errors of shape ``(n,)``. It is only valid
    during the optimizer call it was passed to.
    """

    def __call__(self, indices: _IntegerArray, /) -> NDArray[np.float64]: ...
    @property
    def sizes(self) -> list[int]:
        """The option counts of the step's parameters."""

# ---------------------------------------------------------------------------------------
# Configuration objects
# ---------------------------------------------------------------------------------------

@final
class Retry:
    """Retry a failing model call ``times`` more times before giving up (``on_model_error``)."""

    def __new__(cls, times: int = 1) -> Self: ...
    @property
    def times(self) -> int:
        """Number of retries after the first failure."""
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class AlgorithmSubgraphs:
    """The algorithm subgraphs ``A_i``: the elements that are active at the same time.

    Only elements that can be active together constrain each other.
    """

    def __new__(cls, graph: ProcessorGraph, subgraphs: Iterable[_Elements]) -> Self: ...
    @staticmethod
    def all_active(graph: ProcessorGraph) -> AlgorithmSubgraphs:
        """A single subgraph holding every element: everything can be active at once."""
    @staticmethod
    def xeb_square_lattice(graph: ProcessorGraph, rows: int, cols: int) -> AlgorithmSubgraphs:
        """The five XEB subgraphs of a ``rows x cols`` lattice built by ``square_lattice``."""
    @property
    def graph(self) -> ProcessorGraph:
        """The processor graph."""
    @property
    def subgraphs(self) -> list[ElementSet]:
        """The subgraphs."""
    def __len__(self) -> int: ...

@final
class StitchPlan:
    """A plan for calibrating constraint-disjoint regions in parallel and stitching them."""

    @property
    def regions(self) -> list[ElementSet]:
        """The regions the plan was made from."""
    @property
    def stage_one(self) -> ElementSet:
        """Elements calibrated in stage one (the regions minus the gap)."""
    @property
    def gap(self) -> ElementSet:
        """Elements calibrated in stage two."""

# ---------------------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------------------

@final
class ParameterDatabase:
    """Learned values and the ordered calibration history for one graph.

    ``db[element]`` is the value of a calibrated element (``KeyError`` otherwise). While a
    ``calibrate``, ``recalibrate`` or ``stitch`` call that was given this database runs, the
    engine owns it and any other use raises ``RuntimeError``.
    """

    def __new__(cls, graph: ProcessorGraph) -> Self: ...
    @property
    def graph(self) -> ProcessorGraph:
        """The processor graph."""
    @property
    def fingerprint(self) -> str:
        """Fingerprint of the graph this database belongs to (16 hex digits)."""
    def __getitem__(self, element: _ElementLike, /) -> float: ...
    @overload
    def get(self, element: _ElementLike, default: None = None) -> float | None: ...
    @overload
    def get(self, element: _ElementLike, default: _T) -> float | _T: ...
    def __contains__(self, element: object, /) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Element]: ...
    @overload
    def to_dict(self, labels: Literal[False] = False) -> dict[Element, float]: ...
    @overload
    def to_dict(self, labels: Literal[True]) -> dict[str, float]: ...
    @overload
    def to_dict(self, labels: bool) -> dict[Element, float] | dict[str, float]: ...
    def option_index(self, element: _ElementLike) -> int:
        """The index of the committed value among the element's options."""
    def step_of(self, element: _ElementLike) -> int:
        """The index (in the history) of the step that set the element."""
    @property
    def history(self) -> list[StepRecord | Invalidation]:
        """The append-only history: steps and invalidations in order."""
    def steps(self) -> list[StepRecord]:
        """The committed steps in order (invalidations left out)."""
    @property
    def step_count(self) -> int:
        """Number of steps ever committed."""
    @property
    def calibrated(self) -> ElementSet:
        """The set of calibrated elements."""
    def save(self, path: _PathLike) -> None:
        """Writes the database as JSON, atomically."""
    @classmethod
    def load(cls, path: _PathLike, graph: ProcessorGraph) -> Self:
        """Reads a database written by :meth:`save` (``GraphMismatchError`` on another graph)."""
    def to_json(self) -> str:
        """The database as a JSON string (the format of :meth:`save`)."""
    @classmethod
    def from_json(cls, graph: ProcessorGraph, text: str) -> Self:
        """Parses the JSON written by :meth:`to_json` or :meth:`save`."""
    def same_outcome(self, other: ParameterDatabase) -> bool:
        """Whether ``other`` holds the same values and the same history, ignoring timings."""
    def copy(self) -> Self:
        """A copy of the database (same graph, same values, same history)."""
    def replay_journal(self, path: _PathLike, model: _ErrorModelLike) -> int:
        """Applies the entries of a checkpoint journal that this database lacks.

        Returns how many were applied. The values come from ``model.options``.
        """

def read_journal(path: _PathLike, graph: ProcessorGraph) -> list[StepRecord]:
    """Reads a checkpoint journal written by ``Snake.calibrate(..., checkpoint=path)``."""

# ---------------------------------------------------------------------------------------
# The Snake
# ---------------------------------------------------------------------------------------

@final
class Snake:
    """The Snake Optimizer on ``graph``, calibrating the elements of ``goal``.

    Keyword arguments (``None`` selects the default shown): ``parameter_distance`` (``d_P``,
    1), ``constraint_distance`` (``d_R``, 2), ``traversal_distance`` (``d_T``, 2),
    ``expiry_radius`` (used by ``recalibrate``, 1), ``seed`` (0), ``threads`` (all cores; the
    result does not depend on it), ``on_model_error`` (``"abort"``), ``cache`` (``"step"``),
    ``evaluation_budget`` (1000000), ``heuristic`` (``"breadth_first"``), ``optimizer``
    (``"coordinate_descent"``), ``seed_policy`` (``"lowest_id"``) and ``algorithm`` (everything
    active at once).
    """

    def __new__(
        cls,
        graph: ProcessorGraph,
        goal: _Elements,
        *,
        parameter_distance: int | None = None,
        constraint_distance: int | None = None,
        traversal_distance: int | None = None,
        expiry_radius: int | None = None,
        seed: int | None = None,
        threads: int | None = None,
        on_model_error: _ModelErrorPolicy | None = None,
        cache: _CachePolicy | None = None,
        evaluation_budget: int | None = None,
        heuristic: _Heuristic | None = None,
        optimizer: _Optimizer | None = None,
        seed_policy: _SeedPolicy | None = None,
        algorithm: AlgorithmSubgraphs | None = None,
    ) -> Self: ...
    @property
    def graph(self) -> ProcessorGraph:
        """The processor graph."""
    @property
    def goal(self) -> ElementSet:
        """The calibration goal."""
    @property
    def config(self) -> _SnakeConfig:
        """The configuration as a dict (a copy: changing it does not change the Snake)."""
    @property
    def last_report(self) -> Report | None:
        """The report of the last finished (or cancelled) run, or ``None``."""
    def calibrate(
        self,
        model: _ErrorModelLike,
        *,
        database: ParameterDatabase | None = None,
        on_step: _OnStep | None = None,
        checkpoint: _PathLike | None = None,
    ) -> ParameterDatabase:
        """Calibrates the whole goal and returns the database.

        With ``database=`` it continues that database and returns it. ``on_step(record)`` is
        called on the calling thread for every committed step. ``checkpoint=path`` appends
        every committed step to a journal file.
        """
    def recalibrate(
        self,
        database: ParameterDatabase,
        elements: Element | Iterable[Element],
        model: _ErrorModelLike,
        *,
        on_step: _OnStep | None = None,
        checkpoint: _PathLike | None = None,
    ) -> Report:
        """Forgets the neighbourhood of each expired element in ``database`` and re-learns it."""
    def stitch_plan(self, regions: Iterable[_Elements]) -> StitchPlan:
        """Plans the stitching of constraint-disjoint ``regions``."""
    def stitch(
        self,
        database: ParameterDatabase,
        regions_or_plan: StitchPlan | Iterable[_Elements],
        model: _ErrorModelLike,
        *,
        on_step: _OnStep | None = None,
        checkpoint: _PathLike | None = None,
    ) -> Report:
        """Calibrates the regions minus the gap between them, then the gap."""
    def cancel(self) -> None:
        """Asks the running ``calibrate``/``recalibrate``/``stitch`` call to stop."""
