"""The base class for error models written in Python."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    import numpy as np

    from faulttune._native import Element, Step

__all__ = ["ErrorModel"]


class ErrorModel(abc.ABC):
    """The physics of a calibration problem: candidate values and the error of assignments.

    Subclass it and implement :meth:`options` and :meth:`error`; implement :meth:`prepare` if
    a step needs calibration data (for example by measuring hardware). Any object with
    ``options`` and ``error`` methods also works with :meth:`faulttune.Snake.calibrate`.

    **Threads.** During a run the methods are called from the engine's threads, not from the
    thread that called ``calibrate``. Calls from the several subgoals of a run may *overlap*: the GIL serializes Python bytecode, but it is released inside ``time.sleep``,
    I/O (a hardware driver, a socket) and long NumPy operations, and a call that is waiting
    there lets the next one start. A model must therefore not assume that its methods run one
    at a time, and must not assume which thread calls them. Guard shared mutable state
    (counters, scratch buffers, a device handle that is not thread-safe) with a lock, or
    create the :class:`~faulttune.Snake` with ``threads=1`` to have the calls one after the
    other. The result of a run does not depend on the number of threads.
    """

    @abc.abstractmethod
    def options(self, element: Element) -> ArrayLike:
        """The candidate values of one element, as a non-empty 1-D array of finite floats.

        Called once per element per run. Continuous parameters are discretized here; a model
        with structured options can return the indices ``0..k`` and interpret them itself.
        """

    def prepare(self, step: Step) -> Any:
        """Collect the calibration data a step needs (optional).

        Called once per step, before any :meth:`error` call of that step. Whatever it returns
        is available as ``step.data`` in :meth:`error`. The default does nothing and costs
        nothing: the engine does not call it unless a subclass overrides it.
        """
        return None

    @abc.abstractmethod
    def error(self, step: Step, candidates: NDArray[np.float64]) -> ArrayLike:
        """The estimated errors of many candidate assignments at once (lower is better).

        ``candidates`` is a float64 array of shape ``(n_candidates, len(step.parameters))``;
        column ``j`` holds the values of ``step.parameters[j]``. ``step.constraints`` maps
        each constraining element to its fixed value. Return an array of shape
        ``(n_candidates,)``. Return ``inf`` for forbidden candidates; ``nan`` is an error.
        """
