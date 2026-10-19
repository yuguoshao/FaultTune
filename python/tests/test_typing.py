"""The type stubs of the native extension, and the type checking of the package.

Four kinds of checks keep ``python/faulttune/_native.pyi`` honest:

* ``mypy --strict`` passes on the package, and ``python -m mypy.stubtest faulttune._native``
  passes against the compiled module;
* the examples in ``typing_examples/`` are type-checked with ``mypy --strict``: the ``ok_*``
  examples must produce no diagnostic, and every line of the ``bad_*`` examples that must be
  rejected carries an ``# E: <error code>[: <text of the message>]`` comment (several on one
  line are separated by ``;;``) -- the diagnostics must be exactly those. The ``ok_*``
  examples are also run, so the stubs do not accept what the extension rejects;
* the stub is compared with the runtime: the public names, the parameter names, kinds and
  defaults of every function, the properties, the dunder methods, the base classes and
  whether a class can be subclassed (parsed with ``ast`` and compared with ``inspect``), so
  that adding an argument to the extension without a change of the stub fails;
* the types are checked against values: the value of every property and the result of every
  method of a real calibration run must have the type the stub declares.
"""

from __future__ import annotations

import ast
import collections.abc
import inspect
import io
import os
import re
import runpy
import subprocess
import sys
import tokenize
import types
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import faulttune as ft
import faulttune._native as native

mypy_api = pytest.importorskip("mypy.api")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON_DIR = ROOT / "python"
PACKAGE = PYTHON_DIR / "faulttune"
STUB = PACKAGE / "_native.pyi"
EXAMPLES = HERE / "typing_examples"
PYPROJECT = ROOT / "pyproject.toml"
ALLOWLIST = PYTHON_DIR / "stubtest_allowlist.txt"
CACHE_DIR = ROOT / ".mypy_cache"

# ---------------------------------------------------------------------------------------
# Running mypy
# ---------------------------------------------------------------------------------------

DIAGNOSTIC = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+): error: (?P<message>.*?)(?:  \[(?P<code>[a-z0-9-]+)\])?$"
)


@dataclass(frozen=True)
class Diagnostic:
    """One error reported by mypy."""

    path: Path
    line: int
    code: str
    message: str


def run_mypy(*targets: Path | str) -> tuple[list[Diagnostic], str, int]:
    """Runs ``mypy --strict`` (with the settings of pyproject.toml); returns its errors."""
    stdout, stderr, status = mypy_api.run(
        [
            "--config-file",
            str(PYPROJECT),
            "--cache-dir",
            str(CACHE_DIR),
            "--strict",
            "--show-absolute-path",
            "--show-error-codes",
            "--no-error-summary",
            "--no-pretty",
            "--no-color-output",
            *map(str, targets),
        ]
    )
    diagnostics = []
    for text in stdout.splitlines():
        match = DIAGNOSTIC.match(text)
        if match:
            diagnostics.append(
                Diagnostic(
                    Path(match["path"]).resolve(),
                    int(match["line"]),
                    match["code"] or "none",
                    match["message"],
                )
            )
    return diagnostics, stdout + stderr, status


EXPECTATION = re.compile(r"^(?:E:\s*)?(?P<code>[a-z][a-z0-9-]*)(?::\s*(?P<text>.*?))?\s*$")


def expectations(source: str) -> dict[int, list[tuple[str, str]]]:
    """The ``# E: code[: text]`` comments of a source: ``{line: [(code, text), ...]}``."""
    found: dict[int, list[tuple[str, str]]] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT or not token.string.startswith("# E:"):
            continue
        for part in token.string[2:].split(";;"):
            match = EXPECTATION.match(part.strip())
            assert match, f"line {token.start[0]}: cannot parse the expectation {part!r}"
            found.setdefault(token.start[0], []).append((match["code"], match["text"] or ""))
    return found


def compare(expected: dict[int, list[tuple[str, str]]], actual: list[Diagnostic]) -> list[str]:
    """The differences between the expected and the reported diagnostics, as messages."""
    problems = []
    by_line: dict[int, list[Diagnostic]] = {}
    for diagnostic in actual:
        by_line.setdefault(diagnostic.line, []).append(diagnostic)
    for line in sorted(set(expected) | set(by_line)):
        remaining = list(by_line.get(line, []))
        for code, text in expected.get(line, []):
            hit = next((d for d in remaining if d.code == code and text in d.message), None)
            if hit is None:
                reported = [f"[{d.code}] {d.message}" for d in remaining] or ["no error"]
                problems.append(f"line {line}: expected [{code}] {text!r}, got {reported}")
            else:
                remaining.remove(hit)
        problems.extend(
            f"line {line}: unexpected [{d.code}] {d.message}" for d in remaining
        )
    return problems


@pytest.fixture(scope="module")
def example_diagnostics() -> dict[Path, list[Diagnostic]]:
    """The diagnostics of one mypy run over all the examples, by file."""
    files = sorted(EXAMPLES.glob("*.py"))
    diagnostics, output, _ = run_mypy(*files)
    by_file: dict[Path, list[Diagnostic]] = {path.resolve(): [] for path in files}
    for diagnostic in diagnostics:
        assert diagnostic.path in by_file, f"error outside the examples:\n{output}"
        by_file[diagnostic.path].append(diagnostic)
    return by_file


EXAMPLE_FILES = sorted(EXAMPLES.glob("*.py"))
POSITIVE = [p for p in EXAMPLE_FILES if p.name.startswith("ok_")]
NEGATIVE = [p for p in EXAMPLE_FILES if p.name.startswith("bad_")]


def test_the_examples_are_named_ok_or_bad() -> None:
    assert POSITIVE and NEGATIVE
    assert set(POSITIVE) | set(NEGATIVE) == set(EXAMPLE_FILES)


# ---------------------------------------------------------------------------------------
# mypy --strict on the package and on the examples
# ---------------------------------------------------------------------------------------


def test_the_package_passes_mypy_strict() -> None:
    diagnostics, output, status = run_mypy(PACKAGE)
    assert not diagnostics and status == 0, output


@pytest.mark.parametrize("path", POSITIVE, ids=lambda p: p.name)
def test_positive_examples_have_no_diagnostics(
    path: Path, example_diagnostics: dict[Path, list[Diagnostic]]
) -> None:
    assert not expectations(path.read_text()), "an ok_ example must not expect errors"
    problems = compare({}, example_diagnostics[path.resolve()])
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", NEGATIVE, ids=lambda p: p.name)
def test_negative_examples_have_exactly_the_expected_diagnostics(
    path: Path, example_diagnostics: dict[Path, list[Diagnostic]]
) -> None:
    expected = expectations(path.read_text())
    assert sum(map(len, expected.values())) >= 8, "a negative example needs expectations"
    problems = compare(expected, example_diagnostics[path.resolve()])
    assert not problems, "\n".join(problems)


def test_the_negative_examples_cover_the_required_mistakes() -> None:
    """Wrong argument types, missing methods and bad literal options are all exercised."""
    seen: set[tuple[str, str]] = set()
    for path in NEGATIVE:
        for items in expectations(path.read_text()).values():
            seen.update(items)
    codes = {code for code, _ in seen}
    assert {"arg-type", "call-arg", "attr-defined", "abstract", "misc", "assignment"} <= codes
    texts = " ".join(text for _, text in seen)
    for fragment in ("Literal['bfs']", "Literal['nelder_mead']", "OnlyOptions", "OnlyError"):
        assert fragment in texts or any(
            fragment in path.read_text() for path in NEGATIVE
        ), fragment


def test_the_checker_reports_missing_unexpected_and_wrong_diagnostics(tmp_path: Path) -> None:
    """The comparison fails when a diagnostic is missing, unexpected or of another kind."""
    source = (
        "import faulttune as ft\n"
        "graph = ft.ProcessorGraph.square_lattice(2, 2)\n"
        "goal = graph.goal()\n"
        'ft.Snake(graph, goal, heuristic="bfs")\n'  # line 4: an error
        "ft.Snake(graph, goal, threads=2)\n"  # line 5: fine
        "ft.Snake(graph, goal, threads=2.5)\n"  # line 6: an error
    )
    path = tmp_path / "snippet.py"
    path.write_text(source)
    diagnostics, output, _ = run_mypy(path)
    assert {d.line for d in diagnostics} == {4, 6}, output
    assert not compare({4: [("arg-type", "'bfs'")], 6: [("arg-type", "float")]}, diagnostics)
    # unexpected: nothing expected on line 4
    assert any("unexpected" in p for p in compare({6: [("arg-type", "")]}, diagnostics))
    # missing: an error expected on line 5
    assert any(
        "line 5" in p and "expected" in p
        for p in compare(
            {4: [("arg-type", "")], 5: [("arg-type", "")], 6: [("arg-type", "")]}, diagnostics
        )
    )
    # wrong code and wrong text
    assert compare({4: [("call-arg", "")], 6: [("arg-type", "")]}, diagnostics)
    assert compare({4: [("arg-type", "no such text")], 6: [("arg-type", "")]}, diagnostics)
    assert expectations("x  # E: arg-type: \"int\" ;; E: misc\ny") == {
        1: [("arg-type", '"int"'), ("misc", "")]
    }


@pytest.mark.parametrize("path", POSITIVE, ids=lambda p: p.name)
def test_positive_examples_run(path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """What the stubs accept in the positive examples is also accepted at runtime."""
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(path), run_name=path.stem)


# ---------------------------------------------------------------------------------------
# stubtest
# ---------------------------------------------------------------------------------------


def test_stubtest_passes() -> None:
    command = [
        sys.executable,
        "-m",
        "mypy.stubtest",
        "faulttune._native",
        "--mypy-config-file",
        str(PYPROJECT),
    ]
    if ALLOWLIST.exists():
        command += ["--allowlist", str(ALLOWLIST)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(PYTHON_DIR), *sys.path[:0]])}
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=ROOT, env=env, timeout=600, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Success" in result.stdout


def test_the_allowlist_is_justified() -> None:
    """Every entry of the allowlist has a comment above it saying why it is unavoidable."""
    if not ALLOWLIST.exists():
        return
    lines = ALLOWLIST.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.strip() and not line.startswith("#"):
            assert index > 0 and lines[index - 1].startswith("#"), f"unjustified: {line}"


# ---------------------------------------------------------------------------------------
# The stub, parsed
# ---------------------------------------------------------------------------------------


@dataclass
class StubFunction:
    """A function or method of the stub: its overloads and its kind."""

    name: str
    kind: str  # "function", "method", "static", "class" or "property"
    definitions: list[ast.FunctionDef] = field(default_factory=list)


@dataclass
class StubClass:
    """A class of the stub."""

    name: str
    node: ast.ClassDef
    functions: dict[str, StubFunction] = field(default_factory=dict)
    attributes: dict[str, ast.expr] = field(default_factory=dict)

    @property
    def final(self) -> bool:
        return any(decorator_name(d) == "final" for d in self.node.decorator_list)

    @property
    def bases(self) -> list[str]:
        return [ast.unparse(b) for b in self.node.bases]

    def properties(self) -> dict[str, ast.expr]:
        """The properties and their declared types."""
        found = {}
        for name, function in self.functions.items():
            if function.kind == "property":
                assert function.definitions[0].returns is not None
                found[name] = function.definitions[0].returns
        return found


@dataclass
class Stub:
    """The module-level content of the stub."""

    classes: dict[str, StubClass] = field(default_factory=dict)
    functions: dict[str, StubFunction] = field(default_factory=dict)
    variables: dict[str, ast.expr] = field(default_factory=dict)
    aliases: dict[str, ast.expr] = field(default_factory=dict)
    all_names: list[str] = field(default_factory=list)


def decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ast.unparse(node)


def function_kind(node: ast.FunctionDef, in_class: bool) -> str:
    decorators = {decorator_name(d) for d in node.decorator_list}
    if "property" in decorators:
        return "property"
    if "staticmethod" in decorators:
        return "static"
    if "classmethod" in decorators:
        return "class"
    return "method" if in_class else "function"


def add_function(into: dict[str, StubFunction], node: ast.FunctionDef, in_class: bool) -> None:
    kind = function_kind(node, in_class)
    entry = into.setdefault(node.name, StubFunction(node.name, kind))
    assert entry.kind == kind, f"{node.name}: overloads of different kinds"
    entry.definitions.append(node)


def parse_stub() -> Stub:
    tree = ast.parse(STUB.read_text(), filename=str(STUB))
    stub = Stub()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            cls = StubClass(node.name, node)
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    add_function(cls.functions, item, in_class=True)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    cls.attributes[item.target.id] = item.annotation
            stub.classes[node.name] = cls
        elif isinstance(node, ast.FunctionDef):
            add_function(stub.functions, node, in_class=False)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if ast.unparse(node.annotation) == "TypeAlias":
                assert node.value is not None
                stub.aliases[node.target.id] = node.value
            else:
                stub.variables[node.target.id] = node.annotation
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "__all__":
                stub.all_names = list(ast.literal_eval(node.value))
    return stub


STUB_TREE = parse_stub()
PUBLIC_CLASSES = {n: c for n, c in STUB_TREE.classes.items() if not n.startswith("_")}


def runtime_class(name: str) -> type:
    cls = getattr(native, name)
    assert isinstance(cls, type)
    return cls


# ---------------------------------------------------------------------------------------
# The stub against the runtime: names and signatures
# ---------------------------------------------------------------------------------------

UNKNOWN: Any = object()


@dataclass(frozen=True)
class Param:
    """A parameter, as compared between the stub and the runtime."""

    name: str
    kind: str  # "positional" or "keyword"
    has_default: bool
    default: Any = UNKNOWN


def literal(node: ast.expr | None) -> Any:
    """The value of a literal default, or ``UNKNOWN`` (also for ``...``)."""
    if node is None or (isinstance(node, ast.Constant) and node.value is Ellipsis):
        return UNKNOWN
    try:
        return ast.literal_eval(node)
    except ValueError:
        return UNKNOWN


def stub_params(node: ast.FunctionDef, drop_first: bool) -> list[Param]:
    """The parameters of a stub function (without ``self``/``cls`` if ``drop_first``)."""
    args = node.args
    assert args.vararg is None and args.kwarg is None, f"{node.name}: unexpected *args/**kwargs"
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
    defaults += args.defaults
    params = [
        Param(a.arg, "positional", d is not None, literal(d))
        for a, d in zip(positional, defaults, strict=True)
    ]
    params += [
        Param(a.arg, "keyword", d is not None, literal(d))
        for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True)
    ]
    return params[1:] if drop_first else params


def runtime_params(function: Any, drop_first: bool) -> list[Param] | None:
    """The parameters of a runtime callable, or ``None`` if they cannot be known."""
    try:
        signature = inspect.signature(function)
    except (ValueError, TypeError):
        return None
    params = []
    for p in signature.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            return None  # (*args, **kwargs): the extension gives no signature
        kind = "keyword" if p.kind is p.KEYWORD_ONLY else "positional"
        has_default = p.default is not p.empty
        params.append(Param(p.name, kind, has_default, p.default if has_default else UNKNOWN))
    return params[1:] if drop_first else params


def compare_signature(
    label: str, function: StubFunction, runtime: Any, *, runtime_has_self: bool
) -> None:
    """Asserts that a stub function (all its overloads) matches the runtime signature.

    Names, order, positional/keyword-only kind and the presence and literal value of
    defaults must agree; for overloaded functions every overload must be a sub-signature
    and together they must mention every runtime parameter.
    """
    actual = runtime_params(runtime, drop_first=runtime_has_self)
    if actual is None:
        return  # existence is all that can be checked
    stub_has_self = function.kind in ("method", "class")
    overloaded = len(function.definitions) > 1
    mentioned: set[str] = set()
    runtime_by_name = {p.name: p for p in actual}
    for definition in function.definitions:
        declared = stub_params(definition, drop_first=stub_has_self)
        if overloaded:
            for p in declared:
                assert p.name in runtime_by_name, f"{label}: the runtime has no parameter {p.name}"
                assert p.kind == runtime_by_name[p.name].kind, f"{label}: {p.name} kind differs"
            mentioned.update(p.name for p in declared)
            continue
        names = [p.name for p in declared]
        assert names == [p.name for p in actual], (
            f"{label}: the stub has parameters {names}, the runtime {[p.name for p in actual]}"
        )
        for stub_p, run_p in zip(declared, actual, strict=True):
            assert stub_p.kind == run_p.kind, (
                f"{label}: parameter {stub_p.name} is {stub_p.kind} in the stub but "
                f"{run_p.kind} at runtime"
            )
            assert stub_p.has_default == run_p.has_default, (
                f"{label}: parameter {stub_p.name} has a default in the stub: "
                f"{stub_p.has_default}, at runtime: {run_p.has_default}"
            )
            if UNKNOWN not in (stub_p.default, run_p.default) and run_p.default is not Ellipsis:
                assert stub_p.default == run_p.default, (
                    f"{label}: parameter {stub_p.name}: default {stub_p.default!r} in the "
                    f"stub, {run_p.default!r} at runtime"
                )
    if overloaded:
        assert mentioned == set(runtime_by_name), (
            f"{label}: the overloads mention {sorted(mentioned)}, the runtime has "
            f"{sorted(runtime_by_name)}"
        )


def test_the_public_names_are_those_of_the_runtime() -> None:
    declared = set(STUB_TREE.all_names)
    assert declared == set(native.__all__)
    defined = (
        {n for n in STUB_TREE.classes if not n.startswith("_")}
        | {n for n in STUB_TREE.functions if not n.startswith("_")}
        | {n for n in STUB_TREE.variables if not n.startswith("_") or n == "__version__"}
    )
    assert defined == declared, f"defined but not in __all__ (or vice versa): {defined ^ declared}"
    public_runtime = {n for n in dir(native) if not n.startswith("_") or n == "__version__"}
    assert public_runtime == declared, f"runtime names differ: {public_runtime ^ declared}"
    # the package re-exports the extension and adds ErrorModel only
    assert set(ft.__all__) == declared | {"ErrorModel"}
    for name in declared:
        assert getattr(ft, name) is getattr(native, name)


def test_the_module_function_matches() -> None:
    assert set(STUB_TREE.functions) == {"read_journal"}
    compare_signature(
        "read_journal",
        STUB_TREE.functions["read_journal"],
        native.read_journal,
        runtime_has_self=False,
    )


NOT_REQUIRED_IN_THE_STUB = {
    "__new__",  # declared exactly when the class has a constructor (checked apart)
    "__repr__",
    "__doc__",
    "__module__",
    "__dict__",
    "__weakref__",
    "__lt__",  # richcmp slots exist on every class with __eq__, often returning NotImplemented
    "__le__",
    "__gt__",
    "__ge__",
    "__ne__",
}


def runtime_member_kind(owner: type, name: str) -> str:
    """How the extension defines ``name`` on ``owner``: property, method, static, class, attribute."""
    raw = vars(owner)[name]
    if isinstance(raw, (types.GetSetDescriptorType, property)):
        return "property"
    if isinstance(raw, staticmethod):
        return "static"
    if isinstance(raw, (classmethod, types.ClassMethodDescriptorType)):
        return "class"
    if isinstance(raw, (types.MethodDescriptorType, types.FunctionType)):
        return "method"
    if isinstance(raw, owner):
        return "attribute"
    raise AssertionError(f"{owner.__name__}.{name}: unexpected {type(raw)}")


def is_subclassable(cls: type) -> bool:
    try:
        types.new_class("Probe", (cls,))
    except TypeError:
        return False
    return True


@pytest.mark.parametrize("name", sorted(PUBLIC_CLASSES))
def test_class_members_match_the_runtime(name: str) -> None:
    stub = PUBLIC_CLASSES[name]
    cls = runtime_class(name)
    is_exception = issubclass(cls, BaseException)

    runtime_bases = [b.__name__ for b in cls.__bases__]
    assert stub.bases == ([] if runtime_bases == ["object"] else runtime_bases)
    assert stub.final != is_subclassable(cls), f"{name}: @final must mean 'not subclassable'"

    # public members: the same names of the same kinds (the attributes of exceptions are set
    # on the instances, and checked on real exceptions below)
    runtime_kinds = {n: runtime_member_kind(cls, n) for n in vars(cls) if not n.startswith("_")}
    stub_kinds = {n: f.kind for n, f in stub.functions.items() if not n.startswith("_")}
    if not is_exception:
        stub_kinds.update({n: "attribute" for n in stub.attributes if not n.startswith("_")})
    assert stub_kinds == runtime_kinds, f"{name}: stub {stub_kinds}, runtime {runtime_kinds}"

    # dunders: everything the stub declares exists, everything defined at runtime is declared
    stub_dunders = {n for n in stub.functions if n.startswith("__")} - {"__new__"}
    runtime_dunders = {n for n in vars(cls) if n.startswith("__") and n.endswith("__")}
    for dunder in stub_dunders:
        assert hasattr(cls, dunder), f"{name}.{dunder} is in the stub but not at runtime"
    missing = runtime_dunders - NOT_REQUIRED_IN_THE_STUB - stub_dunders
    assert not missing, f"{name}: dunder methods missing in the stub: {sorted(missing)}"

    # the constructor: PyO3 classes define __new__, never __init__
    assert ("__new__" in vars(cls)) == ("__new__" in stub.functions), f"{name}: __new__ differs"
    assert "__init__" not in stub.functions, f"{name}: PyO3 classes have no __init__"
    if "__new__" in stub.functions:
        compare_signature(
            f"{name}.__new__", stub.functions["__new__"], cls, runtime_has_self=False
        )

    for method, function in stub.functions.items():
        if function.kind != "property" and not method.startswith("__"):
            compare_signature(
                f"{name}.{method}",
                function,
                getattr(cls, method),
                runtime_has_self=function.kind == "method",
            )


def test_every_class_of_the_runtime_is_in_the_stub() -> None:
    runtime_classes = {n for n in native.__all__ if isinstance(getattr(native, n), type)}
    assert runtime_classes == set(PUBLIC_CLASSES)


def test_the_snake_signature_is_spelled_out() -> None:
    """A change of the runtime signature must also be made here, deliberately."""
    parameters = inspect.signature(native.Snake).parameters
    assert list(parameters) == [
        "graph",
        "goal",
        "parameter_distance",
        "constraint_distance",
        "traversal_distance",
        "expiry_radius",
        "seed",
        "threads",
        "on_model_error",
        "cache",
        "evaluation_budget",
        "heuristic",
        "optimizer",
        "seed_policy",
        "algorithm",
    ]
    assert all(p.kind is p.KEYWORD_ONLY and p.default is None for p in list(parameters.values())[2:])
    for method, keywords in {
        "calibrate": {"database", "on_step", "checkpoint"},
        "recalibrate": {"on_step", "checkpoint"},
        "stitch": {"on_step", "checkpoint"},
    }.items():
        found = inspect.signature(getattr(native.Snake, method)).parameters
        assert {n for n, p in found.items() if p.kind is p.KEYWORD_ONLY} == keywords


def changed_function(function: StubFunction, old: str, new: str) -> StubFunction:
    text = ast.unparse(function.definitions[0])
    # Not an AssertionError: the callers expect `compare_signature` to raise one, and a pattern
    # that is not in the text must not make that check pass without running it.
    if old not in text:
        raise LookupError(f"{old!r} is not in the stub text {text!r}")
    node = ast.parse(text.replace(old, new)).body[0]
    assert isinstance(node, ast.FunctionDef)
    return StubFunction(function.name, function.kind, [node])


def test_the_signature_comparison_detects_changes() -> None:
    """compare_signature fails for an extra, missing, renamed or re-defaulted parameter."""
    recalibrate = STUB_TREE.classes["Snake"].functions["recalibrate"]
    compare_signature("ok", recalibrate, native.Snake.recalibrate, runtime_has_self=True)
    for old, new in [
        ("on_step", "callback"),  # renamed
        ("checkpoint:", "extra: int | None=None, checkpoint:"),  # an extra parameter
        ("*, ", ""),  # keyword-only marker dropped
        (", on_step: _OnStep | None=None", ""),  # a parameter missing
        ("checkpoint: _PathLike | None=None", "checkpoint: _PathLike | None"),  # default lost
        ("elements: Element | Iterable[Element], model: _ErrorModelLike", "model: _ErrorModelLike, elements: Element | Iterable[Element]"),
    ]:  # fmt: skip
        with pytest.raises(AssertionError):
            compare_signature(
                "changed",
                changed_function(recalibrate, old, new),
                native.Snake.recalibrate,
                runtime_has_self=True,
            )
    constructor = STUB_TREE.classes["CoordinateDescent"].functions["__new__"]
    with pytest.raises(AssertionError, match="default"):
        compare_signature(
            "changed",
            changed_function(constructor, "max_sweeps: int=3", "max_sweeps: int=4"),
            native.CoordinateDescent,
            runtime_has_self=False,
        )
    getter = STUB_TREE.classes["ParameterDatabase"].functions["get"]  # overloaded
    compare_signature("ok", getter, native.ParameterDatabase.get, runtime_has_self=True)
    single = StubFunction("get", "method", getter.definitions[:1])
    with pytest.raises(AssertionError, match="overloads mention|runtime has no parameter"):
        compare_signature(
            "changed",
            StubFunction("get", "method", [changed_function(single, "default", "other").definitions[0]] * 2),
            native.ParameterDatabase.get,
            runtime_has_self=True,
        )  # fmt: skip


# ---------------------------------------------------------------------------------------
# The stub against values: do the declared types describe what the extension returns?
# ---------------------------------------------------------------------------------------


class Matcher:
    """Checks a runtime value against an annotation of the stub."""

    def __init__(self, stub: Stub) -> None:
        self.stub = stub

    def matches(self, value: Any, annotation: ast.expr, owner: type | None = None) -> bool:
        node = annotation
        if isinstance(node, ast.Constant):
            return value is None if node.value is None else bool(value == node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return self.matches(value, node.left, owner) or self.matches(value, node.right, owner)
        if isinstance(node, ast.Name):
            return self.matches_name(value, node.id, owner)
        if isinstance(node, ast.Attribute) and ast.unparse(node) == "np.random.Generator":
            return isinstance(value, np.random.Generator)
        if isinstance(node, ast.Subscript):
            return self.matches_generic(value, node, owner)
        raise AssertionError(f"unsupported annotation {ast.unparse(node)}")

    def matches_name(self, value: Any, name: str, owner: type | None) -> bool:
        if name == "int":
            return isinstance(value, int) and not isinstance(value, bool)
        simple: dict[str, type | tuple[type, ...]] = {
            "float": float,
            "bool": bool,
            "str": str,
            "object": object,
            "Any": object,
            "_T": object,
        }
        if name in simple:
            return isinstance(value, simple[name])
        if name == "Self":
            assert owner is not None
            return isinstance(value, owner)
        if name in self.stub.aliases:
            return self.matches(value, self.stub.aliases[name], owner)
        if name in self.stub.classes:
            stub_class = self.stub.classes[name]
            if "TypedDict" in stub_class.bases:
                fields = stub_class.attributes
                return (
                    isinstance(value, dict)
                    and set(value) == set(fields)
                    and all(self.matches(value[k], a, owner) for k, a in fields.items())
                )
            if "Protocol" in stub_class.bases:
                return callable(value)
            return isinstance(value, runtime_class(name))
        raise AssertionError(f"unknown name {name} in an annotation")

    def matches_generic(self, value: Any, node: ast.Subscript, owner: type | None) -> bool:
        base = ast.unparse(node.value)
        args = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if base == "Literal":
            return any(isinstance(a, ast.Constant) and value == a.value for a in args)
        if base == "Callable":
            return callable(value)
        if base == "ClassVar":
            return self.matches(value, args[0], owner)
        if base == "list":
            return isinstance(value, list) and all(self.matches(i, args[0], owner) for i in value)
        if base == "dict":
            return isinstance(value, dict) and all(
                self.matches(k, args[0], owner) and self.matches(v, args[1], owner)
                for k, v in value.items()
            )
        if base == "tuple":
            return (
                isinstance(value, tuple)
                and len(value) == len(args)
                and all(self.matches(v, a, owner) for v, a in zip(value, args, strict=True))
            )
        if base == "Iterator":
            return isinstance(value, collections.abc.Iterator) and all(
                self.matches(item, args[0], owner) for item in list(value)
            )
        if base in ("Iterable", "Sequence", "Collection"):
            if not isinstance(value, collections.abc.Iterable):
                return False
            if isinstance(value, collections.abc.Iterator):
                return True  # consuming it would change the value
            return all(self.matches(item, args[0], owner) for item in value)
        if base == "NDArray":
            dtype = ast.unparse(args[0]).removeprefix("np.")
            return isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype)
        raise AssertionError(f"unsupported generic annotation {ast.unparse(node)}")


MATCHER = Matcher(STUB_TREE)


def annotation_of(text: str) -> ast.expr:
    return ast.parse(text, mode="eval").body


def returns_of(function: StubFunction) -> ast.expr:
    """The return annotation of a function (the union of those of its overloads)."""
    annotations = [d.returns for d in function.definitions if d.returns is not None]
    node = annotations[0]
    for other in annotations[1:]:
        node = ast.BinOp(left=node, op=ast.BitOr(), right=other)
    return node


def test_the_value_matcher_can_fail() -> None:
    element = ft.ProcessorGraph.square_lattice(2, 2).node(0, 0)
    ok = MATCHER.matches
    assert ok(3, annotation_of("int")) and not ok(True, annotation_of("int"))
    assert not ok(3, annotation_of("float")) and not ok(3.0, annotation_of("int"))
    assert ok(None, annotation_of("Element | None")) and not ok(None, annotation_of("Element"))
    assert ok([element], annotation_of("list[Element]"))
    assert not ok([element, 1], annotation_of("list[Element]"))
    assert not ok((element,), annotation_of("list[Element]"))
    assert ok({element: 1.0}, annotation_of("dict[Element, float]"))
    assert not ok({element: 1}, annotation_of("dict[Element, float]"))
    assert not ok({"q0_0": 1.0}, annotation_of("dict[Element, float]"))
    assert ok("skip", annotation_of('Literal["abort", "skip"]'))
    assert not ok("retry", annotation_of('Literal["abort", "skip"]'))
    assert ok(np.zeros(2), annotation_of("NDArray[np.float64]"))
    assert not ok(np.zeros(2, np.float32), annotation_of("NDArray[np.float64]"))
    assert ok((element, element), annotation_of("tuple[Element, Element] | None"))
    assert not ok((element,), annotation_of("tuple[Element, Element] | None"))
    assert ok([element], annotation_of("Iterable[Element]"))
    assert ok({element}, annotation_of("Iterable[Element]"))
    assert not ok([element, 1], annotation_of("Iterable[Element]"))
    assert not ok(3, annotation_of("Iterable[Element]"))
    assert ok(iter([element]), annotation_of("Iterator[Element]"))
    assert not ok(iter([1]), annotation_of("Iterator[Element]"))
    assert ok("breadth_first", annotation_of("_Heuristic"))
    assert not ok("breadth", annotation_of("_Heuristic"))


@dataclass
class Samples:
    """Instances to read properties from and to call methods on."""

    objects: dict[str, list[Callable[[], Any]]] = field(default_factory=dict)

    def add(self, class_name: str, factory: Callable[[], Any]) -> None:
        self.objects.setdefault(class_name, []).append(factory)


class World:
    """A small real run whose objects are checked against the stub.

    Objects that are only valid during a callback (the traversal context, the search space,
    the objective) are checked inside the callbacks; ``problems`` collects what is wrong.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.counter = 0
        self.problems: list[str] = []
        self.read: Counter[str] = Counter()  # "Class.property" -> times read
        self.called: Counter[str] = Counter()  # "Class.method" -> times called
        graph = ft.ProcessorGraph.square_lattice(3, 3)
        self.graph = graph
        self.goal = graph.goal(exclude=ft.ElementKind.PARASITIC_EDGE)
        self.nodes = graph.goal(include=ft.ElementKind.NODE)
        self.edges = graph.goal(include=ft.ElementKind.ENGINEERED_EDGE)
        self.parasitic = graph.goal(include=ft.ElementKind.PARASITIC_EDGE)
        self.node = graph.node(1, 1)
        builder = ft.GraphBuilder()
        builder.add_node("a", (0.0, 1.0))
        builder.add_node("b", (1.0, 1.0))
        builder.add_edge("a", "b", ft.ElementKind.CUSTOM_EDGE, label="ab", custom_type=5)
        self.custom_graph = builder.build()
        self.steps: list[ft.Step] = []
        self.delivered: list[ft.StepRecord] = []
        world = self

        class Model(ft.ErrorModel):
            def options(self, element: ft.Element) -> Any:
                return np.linspace(0.0, 1.0, 5)

            def prepare(self, step: ft.Step) -> Any:
                return {"index": step.index}

            def error(self, step: ft.Step, candidates: Any) -> Any:
                world.steps.append(step)
                return np.abs(candidates - 0.3).sum(axis=1)

        def heuristic(ctx: ft.TraversalContext, options: list[ft.Element]) -> list[ft.Element]:
            world.inspect(ctx, "TraversalContext")
            return options

        def optimizer(
            space: ft.SearchSpace, objective: ft.Objective, rng: np.random.Generator
        ) -> tuple[list[int], float]:
            world.inspect(space, "SearchSpace")
            world.inspect(objective, "Objective")
            assert isinstance(rng, np.random.Generator)
            zero = [0] * space.width
            return zero, float(objective(zero)[0])

        def on_step(record: ft.StepRecord) -> None:
            world.delivered.append(record)
            world.inspect(record, "StepRecord")

        self.model = Model()
        self.snake = ft.Snake(
            graph,
            self.goal,
            heuristic=heuristic,
            optimizer=optimizer,
            seed_policy=lambda thread: thread[0],
            algorithm=ft.AlgorithmSubgraphs.all_active(graph),
            cache="across_steps",
            threads=1,
            seed=4,
        )
        self.db = self.snake.calibrate(self.model, on_step=on_step)
        self.report = self.snake.recalibrate(self.db, self.node, self.model)
        self.plan = self.snake.stitch_plan([self.nodes, self.edges])

    def path(self, name: str) -> Path:
        """A fresh file name."""
        self.counter += 1
        return self.directory / f"{self.counter}-{name}"

    # -- checking ------------------------------------------------------------------------

    def inspect(self, obj: Any, class_name: str, once: bool = True) -> None:
        """Reads every property of ``obj`` and calls its methods, checking the types."""
        if once and self.read[f"{class_name}.#"]:
            return
        self.read[f"{class_name}.#"] += 1
        self.check_properties(obj, class_name)
        self.check_methods(lambda: obj, class_name)

    def check_properties(self, obj: Any, class_name: str) -> None:
        stub = STUB_TREE.classes[class_name]
        assert isinstance(obj, runtime_class(class_name))
        for name, annotation in stub.properties().items():
            value = getattr(obj, name)
            self.read[f"{class_name}.{name}"] += 1
            if not MATCHER.matches(value, annotation, type(obj)):
                self.problems.append(
                    f"{class_name}.{name}: {value!r} is not {ast.unparse(annotation)}"
                )

    def check_methods(self, factory: Callable[[], Any], class_name: str) -> None:
        """Runs the recipes of ``class_name`` on fresh objects and checks the results."""
        stub = STUB_TREE.classes[class_name]
        for method, recipe in RECIPES.get(class_name, {}).items():
            annotation = returns_of(stub.functions[method])
            try:
                results = recipe(self, factory())
            except Exception as error:  # noqa: BLE001
                self.problems.append(f"{class_name}.{method}: raised {error!r}")
                continue
            self.called[f"{class_name}.{method}"] += 1
            for value in results:
                if not MATCHER.matches(value, annotation, runtime_class(class_name)):
                    self.problems.append(
                        f"{class_name}.{method}: {value!r} is not {ast.unparse(annotation)}"
                    )

    def samples(self) -> Samples:
        """Instances of every class that can be had outside a callback."""
        s = Samples()
        records = self.db.steps()
        s.add("ElementKind", lambda: ft.ElementKind.NODE)
        s.add("Element", lambda: self.node)
        s.add("Element", lambda: self.graph.edge("q0_0", "q0_1"))
        s.add("Element", lambda: self.custom_graph["ab"])
        s.add("ElementSet", lambda: self.goal)
        s.add("ProcessorGraph", lambda: self.graph)
        s.add("ProcessorGraph", lambda: self.custom_graph)

        def builder() -> ft.GraphBuilder:
            b = ft.GraphBuilder()
            b.add_node("a")
            b.add_node("b")
            return b

        s.add("GraphBuilder", builder)
        s.add("StepRecord", lambda: records[0])
        s.add("StepRecord", lambda: self.report.worst_steps[0])
        s.add("StepRecord", lambda: self.delivered[-1])
        s.add("Step", lambda: self.steps[0])
        s.add("Invalidation", lambda: next(i for i in self.db.history if isinstance(i, ft.Invalidation)))
        s.add("Report", lambda: self.report)
        s.add("CoordinateDescent", ft.CoordinateDescent)
        s.add("Exhaustive", ft.Exhaustive)
        s.add("RandomSearch", ft.RandomSearch)
        s.add("Retry", lambda: ft.Retry(3))
        s.add("AlgorithmSubgraphs", lambda: ft.AlgorithmSubgraphs.all_active(self.graph))
        s.add("StitchPlan", lambda: self.plan)
        s.add("ParameterDatabase", lambda: self.db)
        s.add("ParameterDatabase", lambda: ft.ParameterDatabase(self.graph))
        s.add("Snake", lambda: self.snake)
        s.add("Snake", lambda: ft.Snake(self.graph, self.goal, on_model_error=("retry", 2), cache=100))
        s.add("Snake", lambda: ft.Snake(self.graph, self.goal))
        return s  # fmt: skip


Recipe = Callable[[World, Any], list[Any]]


def _graph_recipes() -> dict[str, Recipe]:
    return {
        "square_lattice": lambda w, o: [ft.ProcessorGraph.square_lattice(2, 3)],
        "from_json": lambda w, o: [ft.ProcessorGraph.from_json(w.graph.to_json())],
        "to_json": lambda w, o: [o.to_json()],
        "node": lambda w, o: [w.graph.node(1, 2)],
        "edge": lambda w, o: [
            w.graph.edge("q0_0", w.graph.node(0, 1)),
            w.graph.edge("q0_0", "q1_1", ft.ElementKind.PARASITIC_EDGE),
        ],
        "element": lambda w, o: [w.graph.element("q1_1")],
        "nodes": lambda w, o: [o.nodes()],
        "edges": lambda w, o: [o.edges()],
        "elements": lambda w, o: [o.elements()],
        "goal": lambda w, o: [o.goal(exclude=[ft.ElementKind.NODE], predicate=bool)],
        "all_elements": lambda w, o: [o.all_elements()],
        "element_set": lambda w, o: [w.graph.element_set([w.node])],
        "__getitem__": lambda w, o: [o[0], o[len(o) // 2], o[-1], o[next(iter(o)).label]],
        "__len__": lambda w, o: [len(o)],
        "__iter__": lambda w, o: [iter(o)],
        "__contains__": lambda w, o: [w.node in o, "zz" in o],
        "__eq__": lambda w, o: [o == w.graph],
        "__hash__": lambda w, o: [hash(o)],
    }


def _set_recipes() -> dict[str, Recipe]:
    return {
        "isdisjoint": lambda w, o: [o.isdisjoint(w.parasitic)],
        "to_list": lambda w, o: [o.to_list()],
        "__len__": lambda w, o: [len(o)],
        "__bool__": lambda w, o: [bool(o)],
        "__iter__": lambda w, o: [iter(o)],
        "__contains__": lambda w, o: [w.node in o],
        "__or__": lambda w, o: [o | w.edges, o | [w.node]],
        "__ror__": lambda w, o: [[w.node] | o],
        "__and__": lambda w, o: [o & w.edges, o & [w.node]],
        "__rand__": lambda w, o: [[w.node] & o],
        "__sub__": lambda w, o: [o - w.edges, o - [w.node]],
        "__rsub__": lambda w, o: [[w.node] - o],
        "__xor__": lambda w, o: [o ^ w.edges, o ^ [w.node]],
        "__rxor__": lambda w, o: [[w.node] ^ o],
        "__eq__": lambda w, o: [o == w.edges],
        "__ne__": lambda w, o: [o != w.edges],
        "__lt__": lambda w, o: [o < w.graph.all_elements()],
        "__le__": lambda w, o: [o <= w.goal],
        "__gt__": lambda w, o: [o > w.edges],
        "__ge__": lambda w, o: [o >= w.edges],
        "__hash__": lambda w, o: [hash(o)],
    }


def _element_recipes() -> dict[str, Recipe]:
    return {
        "__str__": lambda w, o: [str(o)],
        "__hash__": lambda w, o: [hash(o)],
        "__eq__": lambda w, o: [o == o, o == 3],
        "__ne__": lambda w, o: [o != o, o != 3],
        "__lt__": lambda w, o: [o < o],
        "__le__": lambda w, o: [o <= o],
        "__gt__": lambda w, o: [o > o],
        "__ge__": lambda w, o: [o >= o],
    }


def _database_recipes() -> dict[str, Recipe]:
    def replay(w: World, o: Any) -> list[Any]:
        journal = w.path("replay.journal")
        w.snake.calibrate(w.model, database=ft.ParameterDatabase(w.graph), checkpoint=journal)
        return [ft.ParameterDatabase(w.graph).replay_journal(journal, w.model)]

    def save_and_load(w: World, o: Any) -> list[Any]:
        path = w.path("db.json")
        return [o.save(path)]

    def load(w: World, o: Any) -> list[Any]:
        path = w.path("db.json")
        o.save(path)
        return [ft.ParameterDatabase.load(path, w.graph)]

    return {
        "get": lambda w, o: [
            o.get(w.node),
            o.get("q0_0"),
            o.get(w.parasitic.to_list()[0]),
            o.get(w.parasitic.to_list()[0], "absent"),
            o.get(w.node, 0.0),
        ],
        "to_dict": lambda w, o: [o.to_dict(), o.to_dict(labels=True), o.to_dict(labels=False)],
        "option_index": lambda w, o: [o.option_index(e) for e in list(o)[:1]],
        "step_of": lambda w, o: [o.step_of(e.label) for e in list(o)[:1]],
        "steps": lambda w, o: [o.steps()],
        "save": save_and_load,
        "load": load,
        "to_json": lambda w, o: [o.to_json()],
        "from_json": lambda w, o: [ft.ParameterDatabase.from_json(w.graph, o.to_json())],
        "same_outcome": lambda w, o: [o.same_outcome(o.copy())],
        "copy": lambda w, o: [o.copy()],
        "replay_journal": replay,
        "__getitem__": lambda w, o: [o[w.node] if w.node in o else 0.0, o["q0_0"] if "q0_0" in o else 0.0],
        "__contains__": lambda w, o: [w.node in o, "q0_0" in o, 3 in o],
        "__len__": lambda w, o: [len(o)],
        "__iter__": lambda w, o: [iter(o)],
    }  # fmt: skip


def _snake_recipes() -> dict[str, Recipe]:
    def calibrate(w: World, o: Any) -> list[Any]:
        fresh = ft.ParameterDatabase(w.graph)
        journal = w.path("run.journal")
        return [o.calibrate(w.model), o.calibrate(w.model, database=fresh, checkpoint=journal)]

    def recalibrate(w: World, o: Any) -> list[Any]:
        db = o.calibrate(w.model)
        return [o.recalibrate(db, w.node, w.model), o.recalibrate(db, [w.node], w.model)]

    def stitch(w: World, o: Any) -> list[Any]:
        plan = o.stitch_plan([w.nodes, w.edges])
        return [
            o.stitch(ft.ParameterDatabase(w.graph), plan, w.model),
            o.stitch(ft.ParameterDatabase(w.graph), [w.nodes, w.edges], w.model),
        ]

    return {
        "calibrate": calibrate,
        "recalibrate": recalibrate,
        "stitch_plan": lambda w, o: [o.stitch_plan([w.nodes, w.edges])],
        "stitch": stitch,
        "cancel": lambda w, o: [o.cancel()],
    }


# How to call every method (and dunder method) of the stub, by class: each recipe returns the
# values to check against the declared return type.
RECIPES: dict[str, dict[str, Recipe]] = {
    "ElementKind": {
        "__int__": lambda w, o: [int(o)],
        "__hash__": lambda w, o: [hash(o)],
        "__eq__": lambda w, o: [o == o, o == ft.ElementKind.NODE],
    },
    "Element": _element_recipes(),
    "ElementSet": _set_recipes(),
    "ProcessorGraph": _graph_recipes(),
    "GraphBuilder": {
        "add_node": lambda w, o: [o.add_node("c"), o.add_node("d", pos=(0.0, 0.0))],
        "add_edge": lambda w, o: [o.add_edge("a", "b"), o.add_edge("b", "a", ft.ElementKind.PARASITIC_EDGE, label="x")],
        "build": lambda w, o: [o.build()],
    },
    "StepRecord": {"same_outcome": lambda w, o: [o.same_outcome(o)]},
    "TraversalContext": {
        "is_calibrated": lambda w, o: [o.is_calibrated(o.current)],
        "history": lambda w, o: [o.history()],
    },
    "Objective": {
        "__call__": lambda w, o: [o([0] * len(o.sizes)), o([[0] * len(o.sizes)] * 2)],
    },
    "Retry": {
        "__eq__": lambda w, o: [o == ft.Retry(o.times), o == 3],
        "__hash__": lambda w, o: [hash(o)],
    },
    "AlgorithmSubgraphs": {
        "all_active": lambda w, o: [ft.AlgorithmSubgraphs.all_active(w.graph)],
        "xeb_square_lattice": lambda w, o: [ft.AlgorithmSubgraphs.xeb_square_lattice(w.graph, 3, 3)],
        "__len__": lambda w, o: [len(o)],
    },
    "ParameterDatabase": _database_recipes(),
    "Snake": _snake_recipes(),
}  # fmt: skip


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    w = World(tmp_path_factory.mktemp("world"))
    for class_name, factories in w.samples().objects.items():
        for factory in factories:
            w.inspect(factory(), class_name, once=False)
    # read_journal is a module function: the journal of a real run
    journal = w.path("module.journal")
    w.snake.calibrate(w.model, database=ft.ParameterDatabase(w.graph), checkpoint=journal)
    records = ft.read_journal(journal, w.graph)
    if not MATCHER.matches(records, annotation_of("list[StepRecord]")):
        w.problems.append(f"read_journal: {records!r}")
    w.called["read_journal"] += 1
    return w


def test_the_run_of_the_world_is_what_the_checks_assume(world: World) -> None:
    assert len(world.db) == len(world.goal)
    assert world.steps and world.delivered
    assert any(isinstance(i, ft.Invalidation) for i in world.db.history)
    assert world.report.worst_steps


def test_the_values_have_the_declared_types(world: World) -> None:
    """Every property read and every method call of the world returned the declared type."""
    assert not world.problems, "\n".join(world.problems)


def test_every_property_and_method_of_the_stub_was_exercised(world: World) -> None:
    properties = {
        f"{cls}.{name}" for cls, stub in PUBLIC_CLASSES.items() for name in stub.properties()
    }
    methods = {
        f"{cls}.{name}"
        for cls, stub in PUBLIC_CLASSES.items()
        for name, function in stub.functions.items()
        if function.kind != "property" and name != "__new__"
    } | {"read_journal"}
    assert not properties - set(world.read), f"never read: {sorted(properties - set(world.read))}"
    assert not methods - set(world.called), f"never called: {sorted(methods - set(world.called))}"
    stale = {k for k in world.called if k != "read_journal" and k not in methods}
    assert not stale, f"recipes for methods that are not in the stub: {sorted(stale)}"


def test_the_callbacks_saw_the_scoped_objects(world: World) -> None:
    """The objects that live only during a callback were checked inside the callbacks."""
    for key in ["TraversalContext.history", "SearchSpace.sizes", "Objective.sizes", "Step.data"]:
        assert world.read[key] or world.called[key], key
    assert world.called["TraversalContext.history"] and world.called["Objective.__call__"]


def test_the_defaults_are_those_of_the_documentation() -> None:
    graph = ft.ProcessorGraph.square_lattice(2, 2)
    assert ft.Snake(graph, []).config == {
        "parameter_distance": 1,
        "constraint_distance": 2,
        "traversal_distance": 2,
        "expiry_radius": 1,
        "seed": 0,
        "threads": None,
        "on_model_error": "abort",
        "cache": "step",
        "evaluation_budget": 1_000_000,
        "heuristic": "breadth_first",
        "optimizer": "coordinate_descent",
        "seed_policy": "lowest_id",
        "algorithm": None,
    }


# ---------------------------------------------------------------------------------------
# Exceptions: their attributes are set per instance, so stubtest cannot see them
# ---------------------------------------------------------------------------------------


def raised_errors() -> list[BaseException]:
    graph = ft.ProcessorGraph.square_lattice(3, 3)
    goal = graph.goal(exclude=ft.ElementKind.PARASITIC_EDGE)
    errors: list[BaseException] = []

    class Model(ft.ErrorModel):
        def __init__(self, mode: str) -> None:
            self.mode = mode

        def options(self, element: ft.Element) -> Any:
            return np.arange(4.0)

        def error(self, step: ft.Step, candidates: Any) -> Any:
            if self.mode == "raise":
                raise ValueError("boom")
            if self.mode == "nan":
                return np.full(len(candidates), np.nan)
            return candidates.sum(axis=1)

    def capture(call: Callable[[], object]) -> None:
        try:
            call()
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    snake = ft.Snake(graph, goal, threads=1)
    exhaustive = ft.Snake(graph, goal, optimizer=ft.Exhaustive(max_size=2))
    other = ft.ProcessorGraph.square_lattice(2, 2)
    capture(lambda: snake.calibrate(Model("raise")))
    capture(lambda: snake.calibrate(Model("nan")))
    capture(lambda: exhaustive.calibrate(Model("")))
    capture(lambda: snake.calibrate(Model(""), on_step=lambda record: snake.cancel()))
    capture(lambda: snake.stitch_plan([graph.goal(include=ft.ElementKind.PARASITIC_EDGE)]))
    capture(lambda: snake.calibrate(Model(""), database=ft.ParameterDatabase(other)))
    capture(lambda: ft.ProcessorGraph.from_json("{"))
    capture(lambda: ft.ParameterDatabase.from_json(graph, "{"))
    capture(lambda: ft.ParameterDatabase.from_json(other, ft.ParameterDatabase(graph).to_json()))
    return errors


def test_exception_attributes_exist_and_have_the_declared_types() -> None:
    errors = raised_errors()
    seen: set[str] = set()
    for error in errors:
        for klass in type(error).__mro__:
            stub = STUB_TREE.classes.get(klass.__name__)
            if stub is None or klass.__module__ != "faulttune":
                continue
            seen.add(klass.__name__)
            for name, annotation in stub.attributes.items():
                if name == "partial" and not hasattr(error, "partial"):
                    continue  # documented: only the errors of a run carry it
                assert hasattr(error, name), f"{type(error).__name__} has no {name}"
                value = getattr(error, name)
                assert MATCHER.matches(value, annotation), (
                    f"{type(error).__name__}.{name} = {value!r}, not {ast.unparse(annotation)}"
                )
    expected = {
        "FaultTuneError",
        "ModelError",
        "NonFiniteError",
        "SearchSpaceTooLargeError",
        "CancelledError",
        "FormatError",
        "InvalidConfigError",
        "GraphMismatchError",
    }
    assert expected <= seen, sorted(expected - seen)
    carriers = [e for e in errors if hasattr(e, "partial")]
    assert len(carriers) >= 4, "calibrate attaches the partial database to its errors"
    kinds = {type(e).__name__ for e in carriers}
    assert {"ModelError", "NonFiniteError", "CancelledError"} <= kinds


# ---------------------------------------------------------------------------------------
# What the negative examples promise is true at runtime too
# ---------------------------------------------------------------------------------------


def test_the_mistakes_rejected_by_the_stubs_fail_at_runtime() -> None:
    graph = ft.ProcessorGraph.square_lattice(3, 3)
    goal = graph.goal()
    node = graph.node(0, 0)
    db = ft.ParameterDatabase(graph)
    snake = ft.Snake(graph, goal)

    class OnlyOptions:
        def options(self, element: ft.Element) -> Any:
            return np.arange(3.0)

    class OnlyError:
        def error(self, step: ft.Step, candidates: Any) -> Any:
            return np.zeros(len(candidates))

    wrong_type: list[Callable[[], object]] = [
        lambda: ft.Snake(graph, goal, parameter_distance="1"),  # type: ignore[arg-type]
        lambda: ft.Snake(graph, goal, threads=1.5),  # type: ignore[arg-type]
        lambda: ft.Snake(graph, goal, seed="abc"),  # type: ignore[arg-type]
        lambda: ft.Snake("graph", goal),  # type: ignore[arg-type]
        lambda: ft.Snake(graph, node),  # type: ignore[arg-type]
        lambda: ft.Snake(graph, goal, algorithm=graph),  # type: ignore[arg-type]
        lambda: ft.Retry("3"),  # type: ignore[arg-type]
        lambda: ft.CoordinateDescent(random_starts=2.5),  # type: ignore[arg-type]
        lambda: ft.ParameterDatabase(goal),  # type: ignore[arg-type]
        lambda: ft.ProcessorGraph.square_lattice(3.0, 3),  # type: ignore[arg-type]
        lambda: graph.goal(exclude="node"),  # type: ignore[arg-type]
        lambda: graph.goal(predicate=3),  # type: ignore[arg-type]
        lambda: graph.element_set(node),  # type: ignore[arg-type]
        lambda: ft.AlgorithmSubgraphs(graph, goal),  # type: ignore[arg-type]
        lambda: snake.stitch_plan(goal),  # type: ignore[arg-type]
        lambda: snake.calibrate(OnlyOptions()),  # type: ignore[arg-type]
        lambda: snake.calibrate(OnlyError()),  # type: ignore[arg-type]
        lambda: snake.calibrate(object()),  # type: ignore[arg-type]
        lambda: snake.calibrate(OnlyOptions(), on_step="print"),  # type: ignore[arg-type]
        lambda: snake.recalibrate(db, "q0_0", OnlyOptions()),  # type: ignore[arg-type]
        lambda: snake.recalibrate(db, node),  # type: ignore[call-arg]
        lambda: ft.Snake(graph, goal, parameter_dist=2),  # type: ignore[call-arg]
        lambda: ft.Snake(graph, goal, 1, 2, 2),  # type: ignore[call-arg]
        lambda: ft.Snake(graph),  # type: ignore[call-arg]
        lambda: ft.read_journal("path"),  # type: ignore[call-arg]
    ]  # fmt: skip
    for call in wrong_type:
        with pytest.raises(TypeError):
            call()

    bad_literal: list[Callable[[], object]] = [
        lambda: ft.Snake(graph, goal, heuristic="bfs"),  # type: ignore[arg-type]
        lambda: ft.Snake(graph, goal, optimizer="nelder_mead"),  # type: ignore[arg-type]
        lambda: ft.Snake(graph, goal, seed_policy="highest_id"),  # type: ignore[arg-type]
        lambda: ft.Snake(graph, goal, on_model_error="ignore"),  # type: ignore[arg-type]
        lambda: ft.Snake(graph, goal, on_model_error="retry"),  # type: ignore[arg-type]
        lambda: ft.Snake(graph, goal, cache="always"),  # type: ignore[arg-type]
    ]
    for call in bad_literal:
        with pytest.raises(ValueError):
            call()

    with pytest.raises(AttributeError):
        node.label = "other"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        db.step_count = 0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        ft.ElementKind.OTHER  # type: ignore[attr-defined]  # noqa: B018
    # the classes of the extension cannot be subclassed, as the stub says (@final)
    for cls in (ft.Snake, ft.Retry):
        with pytest.raises(TypeError):
            types.new_class("Sub", (cls,))
