"""The Python examples and the notebook run, and print what the documentation says they print.

Every ``examples/*.py`` is run as a subprocess with the current interpreter (exit code 0, a few
expected lines). The examples end with their own self-checks, so a regression in the package
makes them exit non-zero. The notebook ``examples/gate_frequency.ipynb`` is validated (nbformat
4.5, no stored outputs) and all its code cells are executed in order in one fresh namespace. The
whole module takes a few seconds with a release build and well under a minute with a debug one.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import io
import json
import os
import subprocess
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
SCRIPTS = sorted(p.name for p in EXAMPLES.glob("*.py"))
NOTEBOOK = EXAMPLES / "gate_frequency.ipynb"

#: Lines (as substrings) that each script must print. A new example needs only an exit code 0
#: until it is listed here.
EXPECTED_LINES: dict[str, list[str]] = {
    "quickstart.py": [
        "Processor: 16 nodes, 42 edges (58 elements in total)",
        "Calibration goal: 40 parameters",
        "Calibrated 40 parameters:",
        "Qubit frequencies (GHz), laid out like the lattice:",
        "Saved and loaded 40 parameters; identical to the original: True",
        "Re-calibrated q1_3:",
    ],
    "gate_frequency.py": [
        "Device: 8x8 lattice, 274 elements, 176 of them to calibrate, 5 algorithm subgraphs",
        "== 1. Calibrate (d_P = 1, d_R = 2, d_T = 2, breadth-first), with progress ==",
        "calibrated 176 of 176 elements.",
        "== 2. Comparison of settings",
        "  naive: every gate at 6 GHz",
        "  d_P = 2, breadth-first",
        "q3_4 runs at ",
        "Loaded it back: same values and history (bit for bit): True",
        "Loading it onto a 4x4 lattice is refused: graph mismatch",
        "Run cancelled after 26 steps; the database kept all of them (78 of 176 elements).",
        "the replayed database equals the interrupted one: True",
        "all 176 goal elements calibrated.",
        "identical databases, history included: True",
        "All self-checks passed.",
    ],
}


def run_script(name: str) -> subprocess.CompletedProcess[str]:
    """Runs an example in a fresh interpreter, with a timeout, and returns the process."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        env=env,
        cwd=EXAMPLES.parent,
        check=False,
    )


@functools.cache
def script_output(name: str) -> str:
    """The standard output of a successful run of an example (run once per test session)."""
    done = run_script(name)
    assert done.returncode == 0, (
        f"{name} exited with {done.returncode}\n--- stdout ---\n{done.stdout}"
        f"\n--- stderr ---\n{done.stderr}"
    )
    return done.stdout


def test_the_examples_are_there() -> None:
    assert {"quickstart.py", "gate_frequency.py"} <= set(SCRIPTS)
    assert NOTEBOOK.is_file()
    assert set(EXPECTED_LINES) <= set(SCRIPTS)


@pytest.mark.parametrize("name", SCRIPTS)
def test_example_runs_and_prints_the_expected_lines(name: str) -> None:
    output = script_output(name)
    assert output.strip(), f"{name} printed nothing"
    for expected in EXPECTED_LINES.get(name, []):
        assert any(expected in line for line in output.splitlines()), (
            f"{name}: no line contains {expected!r}\n--- stdout ---\n{output}"
        )


@pytest.mark.parametrize("name", SCRIPTS)
def test_example_output_is_deterministic(name: str) -> None:
    """Same inputs and seed, same output: nothing in the output depends on timing or threads."""
    second = run_script(name)
    assert second.returncode == 0, second.stderr
    assert second.stdout == script_output(name)


@pytest.mark.parametrize("name", SCRIPTS)
def test_example_leaves_no_files_behind(name: str, tmp_path: Path) -> None:
    """Examples write to a temporary directory, clean it up and do not depend on the cwd."""
    done = subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        capture_output=True,
        text=True,
        timeout=300,
        env=dict(os.environ, TMPDIR=str(tmp_path), TEMP=str(tmp_path), TMP=str(tmp_path)),
        cwd=tmp_path,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert list(tmp_path.iterdir()) == [], "the example left files in its temporary directory"


# -------------------------------------------------------------------------------------------
# The notebook
# -------------------------------------------------------------------------------------------


def load_notebook() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return document


def code_cells(notebook: dict[str, Any]) -> list[str]:
    """The sources of the code cells, in order."""
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def is_magic(source: str) -> bool:
    """Whether a cell is an IPython magic cell (``%pip install ...``), which is not run here."""
    return source.lstrip().startswith("%")


def test_notebook_is_valid_nbformat_4_5() -> None:
    nbformat = pytest.importorskip("nbformat")
    notebook = nbformat.read(str(NOTEBOOK), as_version=4)
    nbformat.validate(notebook)
    assert (notebook.nbformat, notebook.nbformat_minor) == (4, 5)


def test_notebook_has_no_outputs_and_explains_its_steps() -> None:
    notebook = load_notebook()
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] >= 5
    cells = notebook["cells"]
    ids = [c["id"] for c in cells]
    assert len(set(ids)) == len(ids), "cell ids must be unique"
    for cell in cells:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], "no stored outputs"
            assert cell["execution_count"] is None, "no execution counts"
    assert cells[0]["cell_type"] == "markdown"
    markdown = [c for c in cells if c["cell_type"] == "markdown"]
    code = [c for c in cells if c["cell_type"] == "code"]
    assert len(markdown) >= 8 and len(code) >= 10
    # A markdown cell introduces every numbered section.
    text = "\n".join("".join(c["source"]) for c in markdown)
    for heading in ("## 1. ", "## 2. ", "## 3. ", "## 4. ", "## 5. ", "## 6. "):
        assert heading in text
    for source in code_cells(notebook):
        if not is_magic(source):
            ast.parse(source)


def class_sources(source: str, names: set[str]) -> dict[str, str]:
    """The source text of the named top-level classes."""
    found: dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            found[node.name] = segment
    return found


def test_notebook_models_are_the_models_of_the_script() -> None:
    """The notebook and the script share their model code: they cannot drift apart."""
    names = {"StepPlan", "GateFrequencyModel", "DriftedModel", "HoldingModel"}
    script = class_sources((EXAMPLES / "gate_frequency.py").read_text(encoding="utf-8"), names)
    notebook: dict[str, str] = {}
    for source in code_cells(load_notebook()):
        if not is_magic(source):
            notebook.update(class_sources(source, names))
    assert set(script) == set(notebook) == names
    for name in names:
        # The dataclass decorator is not part of the class node's source segment.
        assert script[name] == notebook[name], f"{name} differs between script and notebook"


@contextlib.contextmanager
def fresh_module(name: str) -> Iterator[types.ModuleType]:
    """A throw-away module whose namespace the notebook cells run in.

    Registering it in ``sys.modules`` is what ``dataclasses`` and ``typing`` expect of a module.
    """
    module = types.ModuleType(name)
    sys.modules[name] = module
    try:
        yield module
    finally:
        del sys.modules[name]


def test_notebook_runs_top_to_bottom() -> None:
    """Executes all code cells in order in one namespace (cells starting with ``%`` are skipped)."""
    cells = code_cells(load_notebook())
    output = io.StringIO()
    with fresh_module("faulttune_example_notebook") as module, contextlib.redirect_stdout(output):
        namespace = module.__dict__
        for index, source in enumerate(cells):
            if is_magic(source):
                continue
            try:
                exec(compile(source, f"<notebook cell {index}>", "exec"), namespace)
            except BaseException as err:  # report which cell failed
                raise AssertionError(f"notebook code cell {index} failed:\n{source}") from err

        # The notebook's own state, not only its output.
        goal = namespace["goal"]
        assert len(goal) == 176
        assert goal <= namespace["calibrated"].calibrated
        assert goal <= namespace["stitched"].calibrated
        assert namespace["restored"].same_outcome(namespace["rebuilt"])
        assert not Path(namespace["scratch"]).exists(), "the notebook cleans up after itself"

    printed = output.getvalue()
    for expected in (
        "Device: 8x8 lattice, 274 elements, 176 of them to calibrate, 5 algorithm subgraphs",
        "calibrated 176 of 176 elements.",
        "Loaded it back: same values and history (bit for bit): True",
        "Run cancelled after 26 steps; the database kept all of them (78 of 176 elements).",
        "the replayed database equals the interrupted one: True",
        "all 176 goal elements calibrated;",
        "Identical databases with 1 and 4 threads, history included: True",
    ):
        assert expected in printed, f"{expected!r} not in the notebook output:\n{printed}"

    # The notebook computes the same numbers as the script (same model code, same seed): the
    # rows of its comparison table and its re-calibration are lines of the script's output.
    script_lines = set(script_output("gate_frequency.py").splitlines())
    shared = [
        line
        for line in printed.splitlines()
        if line.startswith(("  d_P = ", "  naive:", "  q3_4", "q3_4 runs at"))
    ]
    assert len(shared) >= 8
    for line in shared:
        assert line in script_lines, f"notebook line not in the script output: {line!r}"
