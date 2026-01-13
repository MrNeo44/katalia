from __future__ import annotations
import ast
import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import tokenize
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from math import floor, ceil, log
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP

# -----------------------------
# Optional deps
# -----------------------------
# Radon (recommended)
try:
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit
    from radon.raw import analyze as raw_analyze

    RADON_AVAILABLE = True
except Exception:
    RADON_AVAILABLE = False

# NetworkX (architecture graph helpers)
try:
    import networkx as nx  # type: ignore

    NETWORKX_AVAILABLE = True
except Exception:
    NETWORKX_AVAILABLE = False

mcp = FastMCP("katalia-metrics", json_response=True)

DEFAULT_IGNORES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    "build",
    ".eggs",
    "site-packages",
    "node_modules",
}

ALLOWED_ROOT: Optional[Path] = None

CACHE_DIRNAME = ".katalia"
CACHE_FILENAME = "metrics_cache.json"


# -----------------------------
# Data model (file-level)
# -----------------------------
@dataclass
class FileMetrics:
    path: str
    loc: int
    sloc: int
    comments: int
    multi: int
    blank: int
    cc_sum: float
    cc_avg: float
    cc_max: float
    cc_blocks: int
    mi: Optional[float]

    # --- AST-derived signals (stdlib; always available) ---
    ast_ok: bool
    stmt_count: int
    n_classes: int
    n_functions: int  # top-level funcs
    n_methods: int  # methods inside classes
    max_class_methods: int
    max_class_wmc: float  # WMC proxy: sum cyclomatic(method) in a class
    ast_cyc_sum: float  # sum cyclomatic(proxy) across top-level funcs + methods
    ast_cyc_max: float  # max cyclomatic(proxy) among funcs/methods

    # --- Smell-ready file aggregates (computed from entities) ---
    # These are *aggregates*; detailed evidence is stored in cache["files"][path]["entities"]
    max_nesting_depth: int
    bool_op_count: int
    max_stmt_tokens: int
    magic_number_count: int
    match_without_wildcard: bool
    empty_except_handlers: int
    max_identifier_length: int
    overridable_call_in_constructor: bool

    # clone metrics (filled by repo-level clone analysis tool)
    clone_ratio: Optional[float]
    dup_lines: Optional[int]
    dup_blocks: Optional[int]

    # static lint metrics (optional: ruff)
    unused_symbols: Optional[int]

    analyzed_at: str
    error: Optional[str] = None


@dataclass
class FileAnalysis:
    metrics: FileMetrics
    entities: Dict[str, Any]  # {"functions":[...], "classes":[...], "file_examples":{...}}


# -----------------------------
# Helpers
# -----------------------------
def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _resolve_repo(repo_path: str) -> Path:
    repo = Path(repo_path).expanduser().resolve()
    if not repo.exists() or not repo.is_dir():
        raise ValueError(f"repo_path no existe o no es carpeta: {repo}")
    if ALLOWED_ROOT is not None:
        root = ALLOWED_ROOT.resolve()
        if repo != root and root not in repo.parents:
            raise ValueError(f"repo_path fuera de root permitido: {repo} (root: {root})")
    return repo


def _safe_read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="strict")
    except Exception:
        return p.read_text(encoding="latin-1", errors="replace")


def _is_ignored(repo: Path, path: Path, include_tests: bool) -> bool:
    try:
        rel = path.relative_to(repo)
    except ValueError:
        return True

    parts = set(rel.parts)
    if any(x in parts for x in DEFAULT_IGNORES):
        return True
    if not include_tests and ("tests" in parts or "test" in parts):
        return True
    return False


def _iter_py_files(repo: Path, include_tests: bool) -> Iterable[Path]:
    for f in repo.rglob("*.py"):
        if f.is_file() and not _is_ignored(repo, f, include_tests):
            yield f


def _cache_base_dir(repo: Path) -> Path:
    """
    Por defecto: cache dentro del repo => <repo>/.katalia/metrics_cache.json
    Si defines KATALIA_CACHE_DIR, cache afuera (menos invasivo).
    """
    external = os.environ.get("KATALIA_CACHE_DIR")
    if external:
        base = Path(external).expanduser().resolve()
        repo_hash = hashlib.sha1(str(repo).encode("utf-8")).hexdigest()[:12]
        return base / f"katalia_{repo_hash}"
    return repo / CACHE_DIRNAME


def _cache_path(repo: Path) -> Path:
    return _cache_base_dir(repo) / CACHE_FILENAME


def _load_cache(repo: Path) -> Dict[str, Any]:
    p = _cache_path(repo)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"meta": {}, "files": {}, "repo": {}}


def _save_cache(repo: Path, cache: Dict[str, Any]) -> None:
    p = _cache_path(repo)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def _file_stat_key(f: Path) -> Dict[str, Any]:
    st = f.stat()
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}


def _cached_ok(cache_entry: Dict[str, Any], stat_key: Dict[str, Any]) -> bool:
    prev = cache_entry.get("stat") or {}
    return prev.get("mtime_ns") == stat_key.get("mtime_ns") and prev.get("size") == stat_key.get("size")


# -----------------------------
# Artifacts (JSON outputs)
# -----------------------------
def _artifact_dir(repo: Path) -> Path:
    """
    Si defines KATALIA_ARTIFACT_DIR, escribimos afuera (ideal para muchos repos).
    Si no, escribimos dentro del repo en artifacts/katalia/
    """
    external = os.environ.get("KATALIA_ARTIFACT_DIR")
    if external:
        base = Path(external).expanduser().resolve()
        repo_hash = hashlib.sha1(str(repo).encode("utf-8")).hexdigest()[:12]
        return base / f"katalia_{repo_hash}"
    return repo / "artifacts" / "katalia"


def _write_artifact(repo: Path, filename: str, payload: Dict[str, Any]) -> str:
    d = _artifact_dir(repo)
    d.mkdir(parents=True, exist_ok=True)
    p = d / filename
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(p)


# -----------------------------
# AST analysis (stdlib)
# -----------------------------
class _BranchVisitor(ast.NodeVisitor):
    """
    Proxy de puntos de decisión (McCabe-like) para estimar CC.

    Nota: NO descendemos a defs/clases anidadas dentro del scope que analizamos.
    """

    def __init__(self) -> None:
        self.points = 0

    def visit_If(self, node: ast.If) -> None:
        self.points += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.points += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AST) -> None:
        self.points += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.points += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.points += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.points += len(getattr(node, "handlers", []) or [])
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self.points += 1
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AST) -> None:
        self.points += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        vals = getattr(node, "values", []) or []
        self.points += max(0, len(vals) - 1)
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.points += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.points += 1
        self.points += len(getattr(node, "ifs", []) or [])
        self.generic_visit(node)

    def visit_Match(self, node: Any) -> None:
        cases = getattr(node, "cases", []) or []
        add = 0
        for c in cases:
            pat = getattr(c, "pattern", None)
            if pat is None:
                continue
            cls_name = pat.__class__.__name__
            if cls_name == "MatchAs" and getattr(pat, "name", None) is None and getattr(pat, "pattern", None) is None:
                continue
            add += 1
        self.points += add
        self.generic_visit(node)

    # Do NOT descend into nested defs/classes
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        return

    def visit_AsyncFunctionDef(self, node: ast.AST) -> None:  # noqa: N802
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        return


def _cyclomatic_proxy(fn_node: ast.AST) -> float:
    v = _BranchVisitor()
    v.visit(fn_node)
    return float(1 + v.points)


def _count_bool_ops(node: ast.AST) -> int:
    class V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.c = 0

        def visit_BoolOp(self, n: ast.BoolOp) -> None:
            vals = getattr(n, "values", []) or []
            self.c += max(0, len(vals) - 1)
            self.generic_visit(n)

        def visit_FunctionDef(self, n: ast.FunctionDef) -> None:  # avoid nested
            return

        def visit_AsyncFunctionDef(self, n: ast.AST) -> None:
            return

        def visit_ClassDef(self, n: ast.ClassDef) -> None:
            return

    v = V()
    v.visit(node)
    return int(v.c)


def _max_nesting_depth(fn_node: ast.AST) -> int:
    """
    Max block nesting depth inside a function/method.
    Count typical nesting statements; ignore nested defs/classes.
    """

    BLOCKS = (
        ast.If,
        ast.For,
        ast.While,
        ast.Try,
        ast.With,
        ast.Match,
        ast.AsyncFor,
        ast.AsyncWith,
    )

    maxd = 0

    def walk(n: ast.AST, depth: int) -> None:
        nonlocal maxd
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not fn_node:
            return

        if isinstance(n, BLOCKS):
            depth += 1
            maxd = max(maxd, depth)

        for child in ast.iter_child_nodes(n):
            walk(child, depth)

    walk(fn_node, 0)
    return int(maxd)


def _stmt_token_count(stmt_src: str) -> int:
    if not stmt_src.strip():
        return 0
    try:
        toks = tokenize.generate_tokens(io.StringIO(stmt_src).readline)
        count = 0
        for t in toks:
            if t.type in (tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT):
                continue
            count += 1
        return int(count)
    except Exception:
        # fallback: rough split
        return int(len(re.findall(r"\w+|[^\s\w]", stmt_src)))


def _max_stmt_tokens_in_node(code: str, node: ast.AST) -> int:
    max_tokens = 0

    class V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.max_tokens = 0

        def generic_visit(self, n: ast.AST) -> Any:
            if isinstance(n, ast.stmt):
                seg = ast.get_source_segment(code, n) or ""
                self.max_tokens = max(self.max_tokens, _stmt_token_count(seg))
            return super().generic_visit(n)

        def visit_FunctionDef(self, n: ast.FunctionDef) -> None:  # avoid nested
            return

        def visit_AsyncFunctionDef(self, n: ast.AST) -> None:
            return

        def visit_ClassDef(self, n: ast.ClassDef) -> None:
            return

    v = V()
    v.visit(node)
    max_tokens = v.max_tokens
    return int(max_tokens)


def _match_without_wildcard_in_node(node: ast.AST) -> bool:
    found = False
    missing = False

    class V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.missing = False
            self.found = False

        def visit_Match(self, n: Any) -> None:
            self.found = True
            cases = getattr(n, "cases", []) or []
            has_wild = False
            for c in cases:
                pat = getattr(c, "pattern", None)
                if pat is None:
                    continue
                cls_name = pat.__class__.__name__
                if cls_name == "MatchAs" and getattr(pat, "name", None) is None and getattr(pat, "pattern", None) is None:
                    has_wild = True
            if not has_wild:
                self.missing = True
            # keep searching
            self.generic_visit(n)

        def visit_FunctionDef(self, n: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, n: ast.AST) -> None:
            return

        def visit_ClassDef(self, n: ast.ClassDef) -> None:
            return

    v = V()
    v.visit(node)
    found = v.found
    missing = v.missing
    return bool(found and missing)


def _empty_except_handlers_in_node(node: ast.AST) -> int:
    """
    Count except handlers whose body is empty-ish (pass or ellipsis only).
    """
    class V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.count = 0

        def visit_ExceptHandler(self, n: ast.ExceptHandler) -> None:
            body = getattr(n, "body", []) or []
            if not body:
                self.count += 1
                return
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                self.count += 1
                return
            if len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                if getattr(body[0].value, "value", None) is Ellipsis:
                    self.count += 1
                    return
            self.generic_visit(n)

        def visit_FunctionDef(self, n: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, n: ast.AST) -> None:
            return

        def visit_ClassDef(self, n: ast.ClassDef) -> None:
            return

    v = V()
    v.visit(node)
    return int(v.count)


def _max_identifier_length_in_tree(tree: ast.AST) -> Tuple[int, List[str]]:
    maxlen = 0
    examples: List[str] = []

    class V(ast.NodeVisitor):
        def visit_Name(self, n: ast.Name) -> None:
            nonlocal maxlen, examples
            if isinstance(n.id, str):
                L = len(n.id)
                if L > maxlen:
                    maxlen = L
                    examples = [n.id]
                elif L == maxlen and n.id not in examples and len(examples) < 10:
                    examples.append(n.id)

        def visit_Attribute(self, n: ast.Attribute) -> None:
            nonlocal maxlen, examples
            if isinstance(n.attr, str):
                L = len(n.attr)
                if L > maxlen:
                    maxlen = L
                    examples = [n.attr]
                elif L == maxlen and n.attr not in examples and len(examples) < 10:
                    examples.append(n.attr)
            self.generic_visit(n)

    V().visit(tree)
    return int(maxlen), examples


def _magic_number_stats(tree: ast.AST) -> Tuple[int, List[Any]]:
    """
    Heurística razonable:
    - Cuenta literales numéricas (int/float) excluyendo -1/0/1
    - Reporta ejemplos frecuentes (top 10)
    """
    counts: Counter[Any] = Counter()

    class V(ast.NodeVisitor):
        def visit_Constant(self, n: ast.Constant) -> None:
            if isinstance(n.value, (int, float)) and n.value not in (-1, 0, 1):
                counts[n.value] += 1

    V().visit(tree)
    total = sum(counts.values())
    examples = [x for x, _ in counts.most_common(10)]
    return int(total), examples


def _count_sloc_range(lines: List[str], start: int, end: int) -> int:
    """
    start/end are 1-based inclusive. SLOC: non-blank, non-comment-only.
    """
    s = max(1, int(start))
    e = max(s, int(end))
    sl = 0
    for i in range(s - 1, min(e, len(lines))):
        ln = lines[i]
        stripped = ln.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        sl += 1
    return int(sl)


def _infer_component_from_path(rel_path: str) -> str:
    parts = rel_path.replace("\\", "/").split("/")
    if not parts:
        return "."
    # choose top-level folder as component; fallback "."
    return parts[0] if len(parts) > 1 else "."


def _extract_import_targets(tree: ast.AST) -> List[str]:
    """
    Return top-level module names imported (e.g., "requests", "pkg.sub" -> "pkg").
    """
    targets: List[str] = []

    class V(ast.NodeVisitor):
        def visit_Import(self, n: ast.Import) -> None:
            for a in getattr(n, "names", []) or []:
                name = getattr(a, "name", "") or ""
                if name:
                    targets.append(name.split(".")[0])
            self.generic_visit(n)

        def visit_ImportFrom(self, n: ast.ImportFrom) -> None:
            mod = getattr(n, "module", None)
            if isinstance(mod, str) and mod:
                targets.append(mod.split(".")[0])
            self.generic_visit(n)

    V().visit(tree)
    return targets


def _ruff_unused_symbols(repo: Path, rel_path: str) -> Optional[int]:
    """
    Optional: uses `ruff` if available to count unused imports/vars quickly.
    Select codes:
      - F401: unused import
      - F841: local variable assigned but never used
    """
    try:
        res = subprocess.run(
            ["ruff", "check", "--select", "F401,F841", "--quiet", str((repo / rel_path).resolve())],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return 0
        out = (res.stdout or "") + "\n" + (res.stderr or "")
        # rough count: lines that look like diagnostics "path:line:col ..."
        cnt = 0
        for ln in out.splitlines():
            if ":" in ln and ("F401" in ln or "F841" in ln):
                cnt += 1
        return int(cnt)
    except Exception:
        return None


def _entropy_from_counter(c: Counter[Any]) -> float:
    total = float(sum(c.values()))
    if total <= 0:
        return 0.0
    h = 0.0
    for v in c.values():
        p = float(v) / total
        if p > 0:
            h -= p * log(p)
    return float(h)


# -----------------------------
# Entity extraction (functions/classes)
# -----------------------------
def _overridable_calls_in_init(class_node: ast.ClassDef, code: str) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Detect calls like self.foo() inside __init__ where foo is defined as a method in the class
    (potential overridable call in constructor).
    """
    method_names = set()
    decorators_map: Dict[str, List[str]] = {}

    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_names.add(item.name)
            decs = []
            for d in getattr(item, "decorator_list", []) or []:
                if isinstance(d, ast.Name):
                    decs.append(d.id)
                elif isinstance(d, ast.Attribute):
                    decs.append(d.attr)
            decorators_map[item.name] = decs

    init_node: Optional[ast.AST] = None
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
            init_node = item
            break

    if init_node is None:
        return False, []

    call_sites: List[Dict[str, Any]] = []

    class V(ast.NodeVisitor):
        def visit_Call(self, n: ast.Call) -> None:
            fn = getattr(n, "func", None)
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "self":
                name = fn.attr
                if name in method_names and name != "__init__":
                    decs = decorators_map.get(name, [])
                    if "staticmethod" not in decs and "classmethod" not in decs:
                        call_sites.append(
                            {
                                "lineno": getattr(n, "lineno", None),
                                "method": name,
                                "src": (ast.get_source_segment(code, n) or "").strip(),
                            }
                        )
            self.generic_visit(n)

        def visit_FunctionDef(self, n: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, n: ast.AST) -> None:
            return

        def visit_ClassDef(self, n: ast.ClassDef) -> None:
            return

    V().visit(init_node)
    return (len(call_sites) > 0), call_sites


def _class_field_access_sets(class_node: ast.ClassDef) -> Tuple[Dict[str, set], set, Dict[str, int]]:
    """
    Build:
      - method_fields: method_name -> set(fields accessed via self.<field>)
      - all_fields: set(fields assigned (self.x=...) in __init__ and class assignments)
      - method_getset_flags: counts for getters/setters (heuristic)
    """
    all_fields: set = set()
    method_fields: Dict[str, set] = {}
    getset = {"getter": 0, "setter": 0, "methods": 0}

    # class-level assignments: x = ...
    for item in class_node.body:
        if isinstance(item, ast.Assign):
            for t in item.targets:
                if isinstance(t, ast.Name):
                    all_fields.add(t.id)

    class MethodVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.fields: set = set()
            self.assign_fields: set = set()

        def visit_Attribute(self, n: ast.Attribute) -> None:
            if isinstance(n.value, ast.Name) and n.value.id == "self":
                self.fields.add(n.attr)
            self.generic_visit(n)

        def visit_Assign(self, n: ast.Assign) -> None:
            for t in n.targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                    self.assign_fields.add(t.attr)
            self.generic_visit(n)

        def visit_AnnAssign(self, n: ast.AnnAssign) -> None:
            t = getattr(n, "target", None)
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                self.assign_fields.add(t.attr)
            self.generic_visit(n)

        def visit_FunctionDef(self, n: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, n: ast.AST) -> None:
            return

        def visit_ClassDef(self, n: ast.ClassDef) -> None:
            return

    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            getset["methods"] += 1
            name = item.name
            if name.startswith("get_") or name.startswith("is_"):
                getset["getter"] += 1
            if name.startswith("set_"):
                getset["setter"] += 1

            mv = MethodVisitor()
            mv.visit(item)
            method_fields[name] = set(mv.fields)
            all_fields |= set(mv.assign_fields)

    return method_fields, all_fields, getset


def _lcom4_from_method_fields(method_fields: Dict[str, set]) -> float:
    """
    LCOM4: number of connected components in the graph of methods connected by shared field usage.
    1 = cohesive, higher = less cohesive.
    """
    methods = list(method_fields.keys())
    if len(methods) <= 1:
        return 1.0

    # build adjacency
    adj: Dict[str, set] = {m: set() for m in methods}
    for i, m1 in enumerate(methods):
        f1 = method_fields.get(m1, set())
        for m2 in methods[i + 1 :]:
            f2 = method_fields.get(m2, set())
            if f1 and f2 and (f1 & f2):
                adj[m1].add(m2)
                adj[m2].add(m1)

    # count connected components
    seen = set()
    comps = 0
    for m in methods:
        if m in seen:
            continue
        comps += 1
        stack = [m]
        seen.add(m)
        while stack:
            x = stack.pop()
            for y in adj.get(x, set()):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
    return float(max(1, comps))


def _analyze_entities(code: str, use_radon: bool) -> Dict[str, Any]:
    """
    Returns:
      {
        "functions":[{... function-level metrics ...}],
        "classes":[{... class-level metrics ...}],
        "file_examples": { "magic_numbers": [...], "long_identifiers": [...] }
      }
    """
    lines = code.splitlines()
    tree = ast.parse(code)

    # Radon blocks mapping (optional) for more realistic CC per function/method
    radon_cc_by_name_line: Dict[Tuple[str, int], float] = {}
    if use_radon and RADON_AVAILABLE:
        try:
            blocks = cc_visit(code) or []
            for b in blocks:
                # b has attributes: name, lineno, complexity
                nm = getattr(b, "name", None)
                ln = getattr(b, "lineno", None)
                cx = getattr(b, "complexity", None)
                if isinstance(nm, str) and isinstance(ln, int) and isinstance(cx, (int, float)):
                    radon_cc_by_name_line[(nm, ln)] = float(cx)
        except Exception:
            radon_cc_by_name_line = {}

    # file-level AST extras
    magic_count, magic_examples = _magic_number_stats(tree)
    max_ident, ident_examples = _max_identifier_length_in_tree(tree)

    functions: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []

    class ModuleVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, n: ast.FunctionDef) -> None:
            start = int(getattr(n, "lineno", 1))
            end = int(getattr(n, "end_lineno", start))
            sloc = _count_sloc_range(lines, start, end)
            cc = radon_cc_by_name_line.get((n.name, start), _cyclomatic_proxy(n))
            nesting = _max_nesting_depth(n)
            bool_ops = _count_bool_ops(n)
            max_stmt_tokens = _max_stmt_tokens_in_node(code, n)
            n_params = (
                len(getattr(n.args, "posonlyargs", []) or [])
                + len(getattr(n.args, "args", []) or [])
                + len(getattr(n.args, "kwonlyargs", []) or [])
                + (1 if getattr(n.args, "vararg", None) is not None else 0)
                + (1 if getattr(n.args, "kwarg", None) is not None else 0)
            )
            empty_excepts = _empty_except_handlers_in_node(n)
            match_wo = _match_without_wildcard_in_node(n)

            functions.append(
                {
                    "scope": "function",
                    "name": n.name,
                    "lineno": start,
                    "end_lineno": end,
                    "function_sloc": sloc,
                    "function_cc": float(round(float(cc), 2)),
                    "nesting_depth": nesting,
                    "bool_op_count": bool_ops,
                    "max_stmt_tokens": max_stmt_tokens,
                    "n_params": int(n_params),
                    "empty_except_handlers": int(empty_excepts),
                    "match_without_wildcard": bool(match_wo),
                }
            )

        def visit_AsyncFunctionDef(self, n: ast.AST) -> None:
            # treat similarly
            start = int(getattr(n, "lineno", 1))
            end = int(getattr(n, "end_lineno", start))
            nm = getattr(n, "name", "async_fn")
            sloc = _count_sloc_range(lines, start, end)
            cc = radon_cc_by_name_line.get((nm, start), _cyclomatic_proxy(n))
            nesting = _max_nesting_depth(n)
            bool_ops = _count_bool_ops(n)
            max_stmt_tokens = _max_stmt_tokens_in_node(code, n)
            n_args = getattr(n, "args", None)
            n_params = 0
            if n_args is not None:
                n_params = (
                    len(getattr(n_args, "posonlyargs", []) or [])
                    + len(getattr(n_args, "args", []) or [])
                    + len(getattr(n_args, "kwonlyargs", []) or [])
                    + (1 if getattr(n_args, "vararg", None) is not None else 0)
                    + (1 if getattr(n_args, "kwarg", None) is not None else 0)
                )
            empty_excepts = _empty_except_handlers_in_node(n)
            match_wo = _match_without_wildcard_in_node(n)

            functions.append(
                {
                    "scope": "function",
                    "name": nm,
                    "lineno": start,
                    "end_lineno": end,
                    "function_sloc": sloc,
                    "function_cc": float(round(float(cc), 2)),
                    "nesting_depth": nesting,
                    "bool_op_count": bool_ops,
                    "max_stmt_tokens": max_stmt_tokens,
                    "n_params": int(n_params),
                    "empty_except_handlers": int(empty_excepts),
                    "match_without_wildcard": bool(match_wo),
                }
            )

        def visit_ClassDef(self, n: ast.ClassDef) -> None:
            start = int(getattr(n, "lineno", 1))
            end = int(getattr(n, "end_lineno", start))
            loc = int(max(1, end - start + 1))

            # methods + WMC proxy (sum method cc)
            method_nodes: List[ast.AST] = []
            for item in n.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_nodes.append(item)

            method_count = len(method_nodes)
            wmc = 0.0
            methods_detail: List[Dict[str, Any]] = []
            max_m_nesting = 0

            for m in method_nodes:
                mname = getattr(m, "name", "method")
                mstart = int(getattr(m, "lineno", 1))
                mend = int(getattr(m, "end_lineno", mstart))
                msloc = _count_sloc_range(lines, mstart, mend)
                mcc = radon_cc_by_name_line.get((mname, mstart), _cyclomatic_proxy(m))
                nesting = _max_nesting_depth(m)
                bool_ops = _count_bool_ops(m)
                max_stmt_tokens = _max_stmt_tokens_in_node(code, m)
                n_args = getattr(m, "args", None)
                n_params = 0
                if n_args is not None:
                    n_params = (
                        len(getattr(n_args, "posonlyargs", []) or [])
                        + len(getattr(n_args, "args", []) or [])
                        + len(getattr(n_args, "kwonlyargs", []) or [])
                        + (1 if getattr(n_args, "vararg", None) is not None else 0)
                        + (1 if getattr(n_args, "kwarg", None) is not None else 0)
                    )
                empty_excepts = _empty_except_handlers_in_node(m)
                match_wo = _match_without_wildcard_in_node(m)

                wmc += float(mcc)
                max_m_nesting = max(max_m_nesting, int(nesting))

                methods_detail.append(
                    {
                        "scope": "method",
                        "class_name": n.name,
                        "name": mname,
                        "lineno": mstart,
                        "end_lineno": mend,
                        "function_sloc": msloc,
                        "function_cc": float(round(float(mcc), 2)),
                        "nesting_depth": int(nesting),
                        "bool_op_count": int(bool_ops),
                        "max_stmt_tokens": int(max_stmt_tokens),
                        "n_params": int(n_params),
                        "empty_except_handlers": int(empty_excepts),
                        "match_without_wildcard": bool(match_wo),
                    }
                )

            # cohesion proxy: LCOM4
            method_fields, all_fields, getset = _class_field_access_sets(n)
            lcom4 = _lcom4_from_method_fields(method_fields)

            # public_field_ratio heuristic (naming)
            public_fields = [f for f in all_fields if isinstance(f, str) and not f.startswith("_")]
            public_field_ratio = (len(public_fields) / max(1, len(all_fields))) if all_fields else 0.0

            getter_setter_ratio = (
                float(getset["getter"] + getset["setter"]) / float(max(1, getset["methods"]))
                if getset["methods"] > 0
                else 0.0
            )

            # overridable call in ctor
            ov, call_sites = _overridable_calls_in_init(n, code)

            classes.append(
                {
                    "scope": "class",
                    "name": n.name,
                    "lineno": start,
                    "end_lineno": end,
                    "class_loc": loc,
                    "class_method_count": int(method_count),
                    "class_wmc": float(round(float(wmc), 2)),
                    "lcom4": float(round(float(lcom4), 2)),
                    "n_fields": int(len(all_fields)),
                    "public_field_ratio": float(round(float(public_field_ratio), 4)),
                    "getter_setter_ratio": float(round(float(getter_setter_ratio), 4)),
                    "overridable_call_in_constructor": bool(ov),
                    "overridable_call_sites": call_sites,
                    # convenience aggregates for smells
                    "max_method_nesting_depth": int(max_m_nesting),
                }
            )

            # Also push method details into global functions list (so smells can search uniformly)
            functions.extend(methods_detail)

            # do not descend into nested classes here; but allow nested? keep simple
            for item in n.body:
                if isinstance(item, ast.ClassDef):
                    # nested class: treat as separate (rare)
                    self.visit_ClassDef(item)

    ModuleVisitor().visit(tree)

    return {
        "functions": functions,
        "classes": classes,
        "file_examples": {
            "magic_numbers": magic_examples,
            "long_identifiers": ident_examples,
        },
    }


# -----------------------------
# AST stats visitor (file-level summary)
# -----------------------------
class _AstStatsVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: List[ast.AST] = []
        self.stmt_count = 0
        self.n_classes = 0
        self.n_functions = 0
        self.n_methods = 0
        self.max_class_methods = 0
        self.max_class_wmc = 0.0
        self.ast_cyc_sum = 0.0
        self.ast_cyc_max = 0.0
        self._class_method_count_stack: List[int] = []
        self._class_wmc_stack: List[float] = []

    def visit(self, node: ast.AST) -> Any:
        self.stack.append(node)
        try:
            method = "visit_" + node.__class__.__name__
            visitor = getattr(self, method, None)
            if visitor is None:
                return super().generic_visit(node)
            return visitor(node)
        finally:
            self.stack.pop()

    def generic_visit(self, node: ast.AST) -> Any:
        if isinstance(node, ast.stmt):
            self.stmt_count += 1
        return super().generic_visit(node)

    def _parent(self) -> Optional[ast.AST]:
        return self.stack[-2] if len(self.stack) >= 2 else None

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.n_classes += 1
        self._class_method_count_stack.append(0)
        self._class_wmc_stack.append(0.0)

        for child in node.body:
            self.visit(child)

        mcount = self._class_method_count_stack.pop()
        wmc = self._class_wmc_stack.pop()
        self.max_class_methods = max(self.max_class_methods, mcount)
        self.max_class_wmc = max(self.max_class_wmc, float(wmc))
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        parent = self._parent()
        cyc = _cyclomatic_proxy(node)

        if isinstance(parent, ast.Module):
            self.n_functions += 1
            self.ast_cyc_sum += cyc
            self.ast_cyc_max = max(self.ast_cyc_max, cyc)
        elif isinstance(parent, ast.ClassDef):
            self.n_methods += 1
            self.ast_cyc_sum += cyc
            self.ast_cyc_max = max(self.ast_cyc_max, cyc)
            if self._class_method_count_stack:
                self._class_method_count_stack[-1] += 1
            if self._class_wmc_stack:
                self._class_wmc_stack[-1] += float(cyc)
        return None

    def visit_AsyncFunctionDef(self, node: ast.AST) -> Any:
        parent = self._parent()
        cyc = _cyclomatic_proxy(node)

        if isinstance(parent, ast.Module):
            self.n_functions += 1
            self.ast_cyc_sum += cyc
            self.ast_cyc_max = max(self.ast_cyc_max, cyc)
        elif isinstance(parent, ast.ClassDef):
            self.n_methods += 1
            self.ast_cyc_sum += cyc
            self.ast_cyc_max = max(self.ast_cyc_max, cyc)
            if self._class_method_count_stack:
                self._class_method_count_stack[-1] += 1
            if self._class_wmc_stack:
                self._class_wmc_stack[-1] += float(cyc)
        return None


def _ast_metrics_for_code(code: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(code)
        v = _AstStatsVisitor()
        v.visit(tree)
        return {
            "ast_ok": True,
            "stmt_count": int(v.stmt_count),
            "n_classes": int(v.n_classes),
            "n_functions": int(v.n_functions),
            "n_methods": int(v.n_methods),
            "max_class_methods": int(v.max_class_methods),
            "max_class_wmc": round(float(v.max_class_wmc), 2),
            "ast_cyc_sum": round(float(v.ast_cyc_sum), 2),
            "ast_cyc_max": round(float(v.ast_cyc_max), 2),
            "ast_error": None,
        }
    except Exception as e:
        return {
            "ast_ok": False,
            "stmt_count": 0,
            "n_classes": 0,
            "n_functions": 0,
            "n_methods": 0,
            "max_class_methods": 0,
            "max_class_wmc": 0.0,
            "ast_cyc_sum": 0.0,
            "ast_cyc_max": 0.0,
            "ast_error": f"{type(e).__name__}: {e}",
        }


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)
    if n == 1:
        return float(xs[0])
    pos = (n - 1) * float(q)
    lo = int(floor(pos))
    hi = int(ceil(pos))
    if lo == hi:
        return float(xs[lo])
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


# -----------------------------
# Metrics computation (file + entities)
# -----------------------------
def _compute_metrics_for_code(code: str, require_radon: bool, use_radon: bool) -> Dict[str, Any]:
    if require_radon and not RADON_AVAILABLE:
        raise RuntimeError("Radon no está disponible (instala: pip install radon).")

    astm = _ast_metrics_for_code(code)

    if (not use_radon) or (not RADON_AVAILABLE):
        lines = code.splitlines()
        loc = len(lines)
        blank = sum(1 for x in lines if not x.strip())
        comments = sum(1 for x in lines if x.strip().startswith("#"))
        sloc = loc - blank

        return {
            "loc": loc,
            "sloc": sloc,
            "comments": comments,
            "multi": 0,
            "blank": blank,
            "mi": None,
            "cc_sum": float(astm.get("ast_cyc_sum", 0.0)),
            "cc_avg": 0.0,
            "cc_max": float(astm.get("ast_cyc_max", 0.0)),
            "cc_blocks": 0,
            **{k: astm[k] for k in [
                "ast_ok",
                "stmt_count",
                "n_classes",
                "n_functions",
                "n_methods",
                "max_class_methods",
                "max_class_wmc",
                "ast_cyc_sum",
                "ast_cyc_max",
            ]},
        }

    raw = raw_analyze(code)
    mi = float(mi_visit(code, multi=False))
    blocks = cc_visit(code)
    ccs = [b.complexity for b in blocks] if blocks else []

    cc_sum = float(sum(ccs)) if ccs else 0.0
    cc_max = float(max(ccs)) if ccs else 0.0
    cc_avg = float(cc_sum / len(ccs)) if ccs else 0.0

    return {
        "loc": int(raw.loc),
        "sloc": int(raw.sloc),
        "comments": int(raw.comments),
        "multi": int(raw.multi),
        "blank": int(raw.blank),
        "mi": round(mi, 2),
        "cc_sum": round(cc_sum, 2),
        "cc_avg": round(cc_avg, 2),
        "cc_max": round(cc_max, 2),
        "cc_blocks": int(len(ccs)),
        **{k: astm[k] for k in [
            "ast_ok",
            "stmt_count",
            "n_classes",
            "n_functions",
            "n_methods",
            "max_class_methods",
            "max_class_wmc",
            "ast_cyc_sum",
            "ast_cyc_max",
        ]},
    }


def _compute_file_analysis(
    repo: Path,
    rel_path: str,
    include_tests: bool,
    require_radon: bool,
    use_radon: bool,
    max_file_bytes: int,
    compute_entities: bool,
    compute_ruff_unused: bool,
) -> FileAnalysis:
    analyzed_at = _now_iso()
    rel_norm = rel_path.replace("\\", "/")
    f = (repo / rel_norm).resolve()

    ast_defaults = dict(
        ast_ok=False,
        stmt_count=0,
        n_classes=0,
        n_functions=0,
        n_methods=0,
        max_class_methods=0,
        max_class_wmc=0.0,
        ast_cyc_sum=0.0,
        ast_cyc_max=0.0,
    )

    # smell aggregates defaults
    smell_defaults = dict(
        max_nesting_depth=0,
        bool_op_count=0,
        max_stmt_tokens=0,
        magic_number_count=0,
        match_without_wildcard=False,
        empty_except_handlers=0,
        max_identifier_length=0,
        overridable_call_in_constructor=False,
        clone_ratio=None,
        dup_lines=None,
        dup_blocks=None,
        unused_symbols=None,
    )

    entities: Dict[str, Any] = {"functions": [], "classes": [], "file_examples": {}}

    try:
        if not f.exists() or f.suffix != ".py":
            fm = FileMetrics(
                path=rel_norm,
                loc=0, sloc=0, comments=0, multi=0, blank=0,
                cc_sum=0.0, cc_avg=0.0, cc_max=0.0, cc_blocks=0, mi=None,
                analyzed_at=analyzed_at,
                error="file_not_found_or_not_py",
                **ast_defaults,
                **smell_defaults,
            )
            return FileAnalysis(metrics=fm, entities=entities)

        if _is_ignored(repo, f, include_tests):
            fm = FileMetrics(
                path=rel_norm,
                loc=0, sloc=0, comments=0, multi=0, blank=0,
                cc_sum=0.0, cc_avg=0.0, cc_max=0.0, cc_blocks=0, mi=None,
                analyzed_at=analyzed_at,
                error="ignored",
                **ast_defaults,
                **smell_defaults,
            )
            return FileAnalysis(metrics=fm, entities=entities)

        if max_file_bytes > 0 and f.stat().st_size > max_file_bytes:
            fm = FileMetrics(
                path=rel_norm,
                loc=0, sloc=0, comments=0, multi=0, blank=0,
                cc_sum=0.0, cc_avg=0.0, cc_max=0.0, cc_blocks=0, mi=None,
                analyzed_at=analyzed_at,
                error="file_too_large",
                **ast_defaults,
                **smell_defaults,
            )
            return FileAnalysis(metrics=fm, entities=entities)

        code = _safe_read_text(f)
        m = _compute_metrics_for_code(code, require_radon=require_radon, use_radon=use_radon)

        if compute_entities:
            entities = _analyze_entities(code, use_radon=use_radon and RADON_AVAILABLE)

        # aggregates for smell thresholds
        max_nesting = 0
        bool_ops_total = 0
        max_stmt_tokens = 0
        match_wo_any = False
        empty_excepts_total = 0
        overridable_any = False

        for fn in entities.get("functions", []) or []:
            max_nesting = max(max_nesting, int(fn.get("nesting_depth", 0) or 0))
            bool_ops_total += int(fn.get("bool_op_count", 0) or 0)
            max_stmt_tokens = max(max_stmt_tokens, int(fn.get("max_stmt_tokens", 0) or 0))
            match_wo_any = bool(match_wo_any or bool(fn.get("match_without_wildcard", False)))
            empty_excepts_total += int(fn.get("empty_except_handlers", 0) or 0)

        for cl in entities.get("classes", []) or []:
            overridable_any = bool(overridable_any or bool(cl.get("overridable_call_in_constructor", False)))

        # file-level AST extras (magic numbers, id length) are in entities["file_examples"] already,
        # but we also store counts in FileMetrics.
        try:
            tree = ast.parse(code)
            magic_count, _magic_examples = _magic_number_stats(tree)
            max_ident, _ident_examples = _max_identifier_length_in_tree(tree)
        except Exception:
            magic_count, max_ident = 0, 0

        unused = None
        if compute_ruff_unused:
            unused = _ruff_unused_symbols(repo, rel_norm)

        fm = FileMetrics(
            path=rel_norm,
            loc=int(m["loc"]),
            sloc=int(m["sloc"]),
            comments=int(m["comments"]),
            multi=int(m["multi"]),
            blank=int(m["blank"]),
            cc_sum=float(m["cc_sum"]),
            cc_avg=float(m["cc_avg"]),
            cc_max=float(m["cc_max"]),
            cc_blocks=int(m["cc_blocks"]),
            mi=(float(m["mi"]) if m["mi"] is not None else None),
            ast_ok=bool(m.get("ast_ok", False)),
            stmt_count=int(m.get("stmt_count", 0)),
            n_classes=int(m.get("n_classes", 0)),
            n_functions=int(m.get("n_functions", 0)),
            n_methods=int(m.get("n_methods", 0)),
            max_class_methods=int(m.get("max_class_methods", 0)),
            max_class_wmc=float(m.get("max_class_wmc", 0.0)),
            ast_cyc_sum=float(m.get("ast_cyc_sum", 0.0)),
            ast_cyc_max=float(m.get("ast_cyc_max", 0.0)),
            max_nesting_depth=int(max_nesting),
            bool_op_count=int(bool_ops_total),
            max_stmt_tokens=int(max_stmt_tokens),
            magic_number_count=int(magic_count),
            match_without_wildcard=bool(match_wo_any),
            empty_except_handlers=int(empty_excepts_total),
            max_identifier_length=int(max_ident),
            overridable_call_in_constructor=bool(overridable_any),
            clone_ratio=None,
            dup_lines=None,
            dup_blocks=None,
            unused_symbols=unused,
            analyzed_at=analyzed_at,
            error=None,
        )
        return FileAnalysis(metrics=fm, entities=entities)

    except Exception as e:
        fm = FileMetrics(
            path=rel_norm,
            loc=0, sloc=0, comments=0, multi=0, blank=0,
            cc_sum=0.0, cc_avg=0.0, cc_max=0.0, cc_blocks=0, mi=None,
            analyzed_at=analyzed_at,
            error=f"{type(e).__name__}: {e}",
            **ast_defaults,
            **smell_defaults,
        )
        return FileAnalysis(metrics=fm, entities=entities)


def _summary_view(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "loc": m.get("loc", 0),
        "sloc": m.get("sloc", 0),
        "mi": m.get("mi", None),
        "cc_max": m.get("cc_max", 0),
        "cc_sum": m.get("cc_sum", 0.0),
        "cc_blocks": m.get("cc_blocks", 0),
        # AST explainability
        "ast_ok": m.get("ast_ok", None),
        "stmt_count": m.get("stmt_count", 0),
        "n_classes": m.get("n_classes", 0),
        "n_functions": m.get("n_functions", 0),
        "n_methods": m.get("n_methods", 0),
        "max_class_methods": m.get("max_class_methods", 0),
        "max_class_wmc": m.get("max_class_wmc", 0.0),
        "ast_cyc_sum": m.get("ast_cyc_sum", 0.0),
        "ast_cyc_max": m.get("ast_cyc_max", 0.0),
        # smell-ready aggregates
        "max_nesting_depth": m.get("max_nesting_depth", 0),
        "bool_op_count": m.get("bool_op_count", 0),
        "max_stmt_tokens": m.get("max_stmt_tokens", 0),
        "magic_number_count": m.get("magic_number_count", 0),
        "match_without_wildcard": m.get("match_without_wildcard", False),
        "empty_except_handlers": m.get("empty_except_handlers", 0),
        "max_identifier_length": m.get("max_identifier_length", 0),
        "overridable_call_in_constructor": m.get("overridable_call_in_constructor", False),
        # clone + lint (optional)
        "clone_ratio": m.get("clone_ratio", None),
        "dup_lines": m.get("dup_lines", None),
        "dup_blocks": m.get("dup_blocks", None),
        "unused_symbols": m.get("unused_symbols", None),
        "error": m.get("error", None),
    }


# -----------------------------
# Repo-level: clone detection (simple, practical)
# -----------------------------
def _normalized_code_lines_for_clone(code: str) -> List[str]:
    """
    Normalize code for clone detection:
    - strip comments
    - replace strings with STR
    - replace numbers with NUM
    - keep keywords/operators reasonably
    Output: normalized "lines" as token-joined strings.
    """
    out_lines: List[List[str]] = [[]]
    try:
        toks = tokenize.generate_tokens(io.StringIO(code).readline)
        for t in toks:
            if t.type in (tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE):
                if out_lines and out_lines[-1]:
                    out_lines.append([])
                continue
            if t.type in (tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT):
                continue
            if t.type == tokenize.STRING:
                out_lines[-1].append("STR")
                continue
            if t.type == tokenize.NUMBER:
                out_lines[-1].append("NUM")
                continue
            txt = (t.string or "").strip()
            if not txt:
                continue
            out_lines[-1].append(txt)
    except Exception:
        # fallback: whitespace normalized
        return [re.sub(r"\s+", " ", ln.strip()) for ln in code.splitlines() if ln.strip()]

    normalized = [" ".join(x) for x in out_lines if x]
    return normalized


def _compute_clone_metrics_for_repo(repo: Path, include_tests: bool, min_block_lines: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    Winnowing-ish by hashing blocks of N normalized lines.
    Returns per-file:
      { "clone_ratio": float, "dup_lines": int, "dup_blocks": int }
    """
    files = [p.relative_to(repo).as_posix() for p in _iter_py_files(repo, include_tests)]
    norm_lines_by_file: Dict[str, List[str]] = {}
    sloc_by_file: Dict[str, int] = {}

    for rel in files:
        code = _safe_read_text(repo / rel)
        nl = _normalized_code_lines_for_clone(code)
        norm_lines_by_file[rel] = nl
        sloc_by_file[rel] = sum(1 for x in nl if x.strip())

    # map hash -> occurrences
    occ: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for rel, lines in norm_lines_by_file.items():
        if len(lines) < min_block_lines:
            continue
        for i in range(0, len(lines) - min_block_lines + 1):
            block = "\n".join(lines[i : i + min_block_lines])
            h = hashlib.sha1(block.encode("utf-8")).hexdigest()
            occ[h].append((rel, i))

    dup_line_marks: Dict[str, set] = {rel: set() for rel in files}
    dup_blocks_count: Counter[str] = Counter()

    for h, positions in occ.items():
        if len(positions) <= 1:
            continue
        # duplicate block across >=2 places
        for rel, i in positions:
            dup_blocks_count[rel] += 1
            for ln in range(i, i + min_block_lines):
                dup_line_marks[rel].add(ln)

    result: Dict[str, Dict[str, Any]] = {}
    for rel in files:
        dup_lines = len(dup_line_marks.get(rel, set()))
        sloc = max(1, int(sloc_by_file.get(rel, 0)))
        clone_ratio = float(dup_lines) / float(sloc) if sloc > 0 else 0.0
        result[rel] = {
            "clone_ratio": round(clone_ratio, 4),
            "dup_lines": int(dup_lines),
            "dup_blocks": int(dup_blocks_count.get(rel, 0)),
        }
    return result


# -----------------------------
# Repo-level: architecture via import graph
# -----------------------------
def _compute_import_graph_metrics(repo: Path, include_tests: bool) -> Dict[str, Any]:
    """
    Build a component graph using top-level folder as component.
    Edge A->B if a file in component A imports module B that exists as a top-level folder component.
    Outputs:
      - dependency_cycles (count + cycles list)
      - graph_density
      - fanin/fanout, centrality (if networkx)
      - instability per component (Ce/(Ca+Ce))
      - stable_depends_on_unstable edges
      - entrypoint heuristic: component with max incoming deps + api size
    """
    py_files = [p for p in _iter_py_files(repo, include_tests)]
    existing_components = set()
    for p in py_files:
        rel = p.relative_to(repo).as_posix()
        existing_components.add(_infer_component_from_path(rel))

    edges: set = set()
    comp_public_api_size: Counter[str] = Counter()  # heuristic
    comp_entrypoints: Counter[str] = Counter()

    for p in py_files:
        rel = p.relative_to(repo).as_posix()
        comp = _infer_component_from_path(rel)
        code = _safe_read_text(p)
        try:
            tree = ast.parse(code)
        except Exception:
            continue
        targets = _extract_import_targets(tree)
        for t in targets:
            if t in existing_components and t != comp:
                edges.add((comp, t))

        # API heuristic: count public defs/classes in module
        pub = 0
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nm = getattr(n, "name", "")
                if nm and not nm.startswith("_"):
                    pub += 1
        comp_public_api_size[comp] += pub

    comps = sorted(list(existing_components))
    n = len(comps)
    edge_list = sorted(list(edges))

    # density
    density = 0.0
    if n > 1:
        density = float(len(edges)) / float(n * (n - 1))

    # fanin/fanout
    fanout: Counter[str] = Counter()
    fanin: Counter[str] = Counter()
    for a, b in edges:
        fanout[a] += 1
        fanin[b] += 1

    instability: Dict[str, float] = {}
    for c in comps:
        ce = float(fanout.get(c, 0))
        ca = float(fanin.get(c, 0))
        denom = ca + ce
        instability[c] = float(round((ce / denom) if denom > 0 else 0.0, 4))

    stable_depends_on_unstable: List[Dict[str, Any]] = []
    for a, b in edges:
        ia = instability.get(a, 0.0)
        ib = instability.get(b, 0.0)
        if ia < ib:  # stable-ish depending on more unstable
            stable_depends_on_unstable.append({"from": a, "to": b, "instability_from": ia, "instability_to": ib})

    cycles: List[List[str]] = []
    if NETWORKX_AVAILABLE:
        G = nx.DiGraph()
        G.add_nodes_from(comps)
        G.add_edges_from(edge_list)
        try:
            cycles = [list(c) for c in nx.simple_cycles(G)]
        except Exception:
            cycles = []
        # entrypoint heuristic: max incoming
        if comps:
            top_in = max(comps, key=lambda c: fanin.get(c, 0))
            comp_entrypoints[top_in] += 1
        degree_centrality = nx.degree_centrality(G)
    else:
        # minimal cycle detection (DFS)
        graph: Dict[str, List[str]] = {c: [] for c in comps}
        for a, b in edge_list:
            graph[a].append(b)

        stack: List[str] = []
        visited = set()
        onstack = set()

        def dfs(u: str) -> None:
            visited.add(u)
            onstack.add(u)
            stack.append(u)
            for v in graph.get(u, []):
                if v not in visited:
                    dfs(v)
                elif v in onstack:
                    # cycle found
                    try:
                        idx = stack.index(v)
                        cyc = stack[idx:] + [v]
                        cycles.append(cyc)
                    except Exception:
                        pass
            stack.pop()
            onstack.remove(u)

        for c in comps:
            if c not in visited:
                dfs(c)

        degree_centrality = {}

    entrypoint = None
    if comps:
        entrypoint = max(comps, key=lambda c: fanin.get(c, 0))
    entrypoint_count = 1 if entrypoint else 0
    entrypoint_api_size = int(comp_public_api_size.get(entrypoint, 0)) if entrypoint else 0

    return {
        "components": comps,
        "edges": edge_list,
        "edge_count": len(edge_list),
        "graph_density": round(float(density), 6),
        "fanin": {k: int(v) for k, v in fanin.items()},
        "fanout": {k: int(v) for k, v in fanout.items()},
        "instability": instability,
        "stable_depends_on_unstable": stable_depends_on_unstable,
        "dependency_cycles": {"count": len(cycles), "cycles": cycles[:50]},  # cap
        "degree_centrality": degree_centrality,
        "entrypoint_count": entrypoint_count,
        "entrypoint": entrypoint,
        "entrypoint_api_size": entrypoint_api_size,
        "networkx_available": NETWORKX_AVAILABLE,
    }


# -----------------------------
# Repo-level: git history (shotgun/divergent + feature scatter)
# -----------------------------
ISSUE_RE = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")
CONVENTIONAL_RE = re.compile(r"^(feat|fix|refactor|chore|docs|test|build|ci|perf|style|revert)(\(.+\))?:", re.IGNORECASE)


def _git_available(repo: Path) -> bool:
    try:
        res = subprocess.run(["git", "-C", str(repo), "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
        return res.returncode == 0 and "true" in (res.stdout or "").lower()
    except Exception:
        return False


def _compute_git_history_metrics(repo: Path, include_tests: bool, max_commits: int = 3000) -> Dict[str, Any]:
    """
    Produces:
      - files_touched_per_change distribution (commits)
      - per-file change_reason_entropy based on Conventional Commit types (proxy)
      - feature_scatter based on issue IDs in commit messages (proxy "real feature" when IDs exist)
    """
    if not _git_available(repo):
        return {"git": False, "error": "not_a_git_repo_or_git_missing"}

    cmd = [
        "git",
        "-C",
        str(repo),
        "log",
        f"-n{int(max_commits)}",
        "--no-merges",
        "--name-only",
        "--pretty=format:%H%x1f%s%x1f%ad",
        "--date=iso-strict",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        return {"git": True, "error": "git_log_failed", "stderr": (res.stderr or "")[:4000]}

    files_touched_per_change: List[int] = []
    commit_records: List[Dict[str, Any]] = []

    file_type_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    issue_components: Dict[str, set] = defaultdict(set)

    cur_hash = None
    cur_subject = None
    cur_date = None
    cur_files: List[str] = []

    def flush_commit() -> None:
        nonlocal cur_hash, cur_subject, cur_date, cur_files
        if not cur_hash:
            return
        # filter files by include_tests + ignored
        files = []
        for fp in cur_files:
            fp = fp.strip()
            if not fp or not fp.endswith(".py"):
                continue
            p = (repo / fp).resolve()
            if not p.exists():
                continue
            if _is_ignored(repo, p, include_tests):
                continue
            files.append(fp.replace("\\", "/"))

        touched = len(set(files))
        files_touched_per_change.append(int(touched))

        # type from conventional commits
        ctype = "other"
        if isinstance(cur_subject, str):
            m = CONVENTIONAL_RE.match(cur_subject.strip())
            if m:
                ctype = (m.group(1) or "other").lower()

        for fp in files:
            file_type_counts[fp][ctype] += 1

        # issue IDs -> feature scatter
        issues = set()
        if isinstance(cur_subject, str):
            issues |= set(ISSUE_RE.findall(cur_subject))
        for iss in issues:
            for fp in files:
                issue_components[iss].add(_infer_component_from_path(fp))

        commit_records.append(
            {
                "hash": cur_hash,
                "subject": cur_subject,
                "date": cur_date,
                "files_touched": int(touched),
                "type": ctype,
                "issues": sorted(list(issues)),
                "files": files[:200],
            }
        )

        cur_hash, cur_subject, cur_date, cur_files = None, None, None, []

    for ln in (res.stdout or "").splitlines():
        if "\x1f" in ln:
            flush_commit()
            parts = ln.split("\x1f")
            cur_hash = parts[0] if len(parts) > 0 else None
            cur_subject = parts[1] if len(parts) > 1 else ""
            cur_date = parts[2] if len(parts) > 2 else ""
            cur_files = []
        else:
            cur_files.append(ln)

    flush_commit()

    # per-file entropy over change "types"
    file_entropy: Dict[str, float] = {}
    for fp, c in file_type_counts.items():
        file_entropy[fp] = float(round(_entropy_from_counter(c), 6))

    # feature scatter from issues
    feature_scatter = {iss: int(len(comps)) for iss, comps in issue_components.items()}
    feature_scatter_top = sorted(feature_scatter.items(), key=lambda x: x[1], reverse=True)[:200]

    # summary stats
    def pct(vals: List[int], q: float) -> Optional[float]:
        if not vals:
            return None
        xs = sorted(vals)
        n = len(xs)
        pos = int(round((n - 1) * q))
        return float(xs[max(0, min(n - 1, pos))])

    payload = {
        "git": True,
        "commits_analyzed": len(commit_records),
        "files_touched_per_change": {
            "avg": round(sum(files_touched_per_change) / max(1, len(files_touched_per_change)), 4),
            "p95": pct(files_touched_per_change, 0.95),
            "p99": pct(files_touched_per_change, 0.99),
            "max": max(files_touched_per_change) if files_touched_per_change else 0,
        },
        "change_reason_entropy": file_entropy,  # proxy for DIVERGENT_CHANGE
        "feature_scatter": dict(feature_scatter_top),
        "feature_scatter_raw_count": len(feature_scatter),
        "commits_sample": commit_records[:200],  # cap
    }
    return payload


# -----------------------------
# Ranking logic (unchanged)
# -----------------------------
def _score_file(mi: Optional[float], loc: int, cc_max: float, mi_threshold: float = 65.0) -> Tuple[float, str]:
    reason = "signal"
    score = 0.0

    if mi is not None and mi < mi_threshold:
        score += (mi_threshold - mi) * 4.0
        reason = f"mi<{int(mi_threshold)}"

    score += loc * 0.5
    score += float(cc_max) * 0.6
    return round(score, 4), reason


# -----------------------------
# MCP Tools
# -----------------------------
@mcp.tool()
def ping() -> Dict[str, Any]:
    return {"ok": True, "ts": _now_iso(), "radon_available": RADON_AVAILABLE, "networkx_available": NETWORKX_AVAILABLE}


@mcp.tool()
def list_python_files(repo_path: str, include_tests: bool = False) -> Dict[str, Any]:
    repo = _resolve_repo(repo_path)
    files = [p.relative_to(repo).as_posix() for p in _iter_py_files(repo, include_tests)]
    return {"repo": str(repo), "count": len(files), "files": files}


@mcp.tool()
def cache_info(repo_path: str) -> Dict[str, Any]:
    repo = _resolve_repo(repo_path)
    cache = _load_cache(repo)
    return {
        "repo": str(repo),
        "cache_path": str(_cache_path(repo)),
        "files_cached": len(cache.get("files", {})),
        "repo_cached_keys": sorted(list((cache.get("repo") or {}).keys())),
        "meta": cache.get("meta", {}),
        "radon_available": RADON_AVAILABLE,
        "networkx_available": NETWORKX_AVAILABLE,
    }


@mcp.tool()
def analyze_python_batch(
    repo_path: str,
    files: List[str],
    include_tests: bool = False,
    use_cache: bool = True,
    force_recompute: bool = False,
    require_radon: bool = False,
    use_radon: bool = True,
    summary_only: bool = True,
    max_workers: int = 1,
    max_file_bytes: int = 1_000_000,
    compute_entities: bool = True,
    compute_ruff_unused: bool = False,
) -> Dict[str, Any]:
    """
    Analiza SOLO los archivos indicados (batch-friendly).
    Ahora puede almacenar entity metrics (functions/classes) para tus smells.
    """
    repo = _resolve_repo(repo_path)
    req_files_norm = [x.replace("\\", "/") for x in files]

    if require_radon and not RADON_AVAILABLE:
        raise RuntimeError("Radon no está disponible. Instala: pip install radon")

    cache = _load_cache(repo) if use_cache else {"meta": {}, "files": {}, "repo": {}}
    cache_files: Dict[str, Any] = cache.get("files", {})

    tasks: List[str] = []
    skipped: List[str] = []

    for rel_norm in req_files_norm:
        f = (repo / rel_norm).resolve()
        if not f.exists() or f.suffix != ".py":
            skipped.append(rel_norm)
            continue
        if _is_ignored(repo, f, include_tests):
            skipped.append(rel_norm)
            continue

        st_key = _file_stat_key(f)
        entry = cache_files.get(rel_norm)

        if not force_recompute and entry and _cached_ok(entry, st_key):
            continue
        tasks.append(rel_norm)

    computed: List[FileAnalysis] = []
    if tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(
                    _compute_file_analysis,
                    repo,
                    rel,
                    include_tests,
                    require_radon,
                    use_radon,
                    max_file_bytes,
                    compute_entities,
                    compute_ruff_unused,
                ): rel
                for rel in tasks
            }
            for fut in as_completed(futs):
                computed.append(fut.result())

    for fa in computed:
        rel_norm = fa.metrics.path
        f = (repo / rel_norm)
        cache_files[rel_norm] = {
            "metrics": asdict(fa.metrics),
            "entities": fa.entities,
            "stat": _file_stat_key(f),
        }

    out_files: Dict[str, Any] = {}
    for rel_norm in req_files_norm:
        entry = cache_files.get(rel_norm)
        if not entry:
            continue
        m = entry.get("metrics", {})
        if summary_only:
            out_files[rel_norm] = _summary_view(m)
        else:
            out_files[rel_norm] = {
                "metrics": m,
                "entities": entry.get("entities", {}),
            }

    cache["files"] = cache_files
    cache["meta"] = {
        "repo": str(repo),
        "updated_at": _now_iso(),
        "radon_available": RADON_AVAILABLE,
        "use_radon": bool(use_radon and RADON_AVAILABLE),
        "max_file_bytes": max_file_bytes,
        "cache_path": str(_cache_path(repo)),
    }
    if use_cache:
        _save_cache(repo, cache)

    return {
        "repo": str(repo),
        "requested": len(req_files_norm),
        "returned": len(out_files),
        "skipped": skipped,
        "meta": cache["meta"],
        "files": out_files,
    }


@mcp.tool()
def analyze_python_repo_full(
    repo_path: str,
    include_tests: bool = False,
    use_cache: bool = True,
    force_recompute: bool = False,
    require_radon: bool = False,
    use_radon: bool = True,
    max_workers: int = 1,
    max_file_bytes: int = 1_000_000,
    compute_entities: bool = True,
    compute_clone_metrics: bool = True,
    compute_architecture: bool = True,
    compute_git_history: bool = False,
    compute_ruff_unused: bool = False,
) -> Dict[str, Any]:
    """
    Full pipeline:
      1) analiza todos los .py del repo (file + entity metrics)
      2) opcional: clone metrics (dup_lines/blocks/clone_ratio)
      3) opcional: import graph (dependency_cycles, instability, density, etc.)
      4) opcional: git history proxies (files_touched_per_change, change_reason_entropy, feature_scatter)
    """
    repo = _resolve_repo(repo_path)
    files = [p.relative_to(repo).as_posix() for p in _iter_py_files(repo, include_tests)]

    batch = analyze_python_batch(
        repo_path=repo_path,
        files=files,
        include_tests=include_tests,
        use_cache=use_cache,
        force_recompute=force_recompute,
        require_radon=require_radon,
        use_radon=use_radon,
        summary_only=True,
        max_workers=max_workers,
        max_file_bytes=max_file_bytes,
        compute_entities=compute_entities,
        compute_ruff_unused=compute_ruff_unused,
    )

    cache = _load_cache(repo) if use_cache else {"meta": {}, "files": {}, "repo": {}}
    cache_files: Dict[str, Any] = cache.get("files", {})

    repo_out: Dict[str, Any] = {}

    if compute_clone_metrics:
        clones = _compute_clone_metrics_for_repo(repo, include_tests=include_tests, min_block_lines=5)
        for rel, cm in clones.items():
            entry = cache_files.get(rel)
            if not entry:
                continue
            m = entry.get("metrics", {})
            m["clone_ratio"] = cm.get("clone_ratio")
            m["dup_lines"] = cm.get("dup_lines")
            m["dup_blocks"] = cm.get("dup_blocks")
            entry["metrics"] = m
            cache_files[rel] = entry
        repo_out["clone_metrics"] = {"ok": True, "min_block_lines": 5, "files": len(clones)}

    if compute_architecture:
        arch = _compute_import_graph_metrics(repo, include_tests=include_tests)
        repo_out["architecture"] = arch

    if compute_git_history:
        hist = _compute_git_history_metrics(repo, include_tests=include_tests)
        repo_out["history"] = hist

    cache["files"] = cache_files
    cache["repo"] = {**(cache.get("repo", {}) or {}), **repo_out}
    cache["meta"] = {**(cache.get("meta", {}) or {}), "repo_updated_at": _now_iso()}

    if use_cache:
        _save_cache(repo, cache)

    artifact = _write_artifact(repo, "analysis_repo_full.json", {"repo": str(repo), "generated_at": _now_iso(), "repo_metrics": repo_out})
    return {"status": "ok", "artifact": artifact, "files": len(files), "repo_metrics_keys": sorted(list(repo_out.keys())), "batch": batch}


@mcp.tool()
def export_metrics_json(repo_path: str, include_errors: bool = True) -> Dict[str, Any]:
    """
    Exporta TODO lo que hay en cache a metrics_python.json (file-level summary view).
    """
    repo = _resolve_repo(repo_path)
    cache = _load_cache(repo)
    files_map: Dict[str, Any] = cache.get("files", {})

    out = {}
    for path, entry in files_map.items():
        m = (entry or {}).get("metrics") or {}
        if (not include_errors) and m.get("error"):
            continue
        out[path] = _summary_view(m)

    payload = {
        "repo": str(repo),
        "generated_at": _now_iso(),
        "meta": cache.get("meta", {}),
        "repo_metrics": cache.get("repo", {}),
        "files": out,
    }
    artifact = _write_artifact(repo, "metrics_python.json", payload)
    return {"status": "ok", "artifact": artifact, "files": len(out)}


@mcp.tool()
def export_entities_json(repo_path: str, include_errors: bool = False) -> Dict[str, Any]:
    """
    Exporta entities (functions/classes) a entities_python.json para evidencias de smells:
      - function_sloc, function_cc, nesting_depth, bool_op_count, n_params, ...
      - class_method_count, class_wmc, lcom4, overridable_call_in_constructor, ...
    """
    repo = _resolve_repo(repo_path)
    cache = _load_cache(repo)
    files_map: Dict[str, Any] = cache.get("files", {})

    out_files: Dict[str, Any] = {}
    total_fn = 0
    total_cls = 0

    for path, entry in files_map.items():
        m = (entry or {}).get("metrics") or {}
        if (not include_errors) and m.get("error"):
            continue
        ents = (entry or {}).get("entities") or {}
        fns = ents.get("functions", []) or []
        cls = ents.get("classes", []) or []
        total_fn += len(fns)
        total_cls += len(cls)
        out_files[path] = {
            "functions": fns,
            "classes": cls,
            "file_examples": ents.get("file_examples", {}),
        }

    payload = {
        "repo": str(repo),
        "generated_at": _now_iso(),
        "meta": cache.get("meta", {}),
        "counts": {"functions": total_fn, "classes": total_cls},
        "files": out_files,
    }
    artifact = _write_artifact(repo, "entities_python.json", payload)
    return {"status": "ok", "artifact": artifact, "files": len(out_files), "functions": total_fn, "classes": total_cls}


@mcp.tool()
def rank_python_files(
    repo_path: str,
    top_k: int = 20,
    mi_threshold: float = 65.0,
    include_errors: bool = False,
) -> Dict[str, Any]:
    """
    Ranking Top K desde cache + export hotspots_python.json
    """
    repo = _resolve_repo(repo_path)
    cache = _load_cache(repo)
    files_map: Dict[str, Any] = cache.get("files", {})

    ranked: List[Dict[str, Any]] = []
    for path, entry in files_map.items():
        m = (entry or {}).get("metrics") or {}
        if (not include_errors) and m.get("error"):
            continue
        mi = m.get("mi", None)
        loc = int(m.get("loc", 0))
        cc_max = float(m.get("cc_max", 0.0))
        score, reason = _score_file(mi, loc, cc_max, mi_threshold=mi_threshold)
        ranked.append(
            {
                "path": path,
                "score": score,
                "reason": reason,
                "cc_max": m.get("cc_max", 0),
                "cc_sum": m.get("cc_sum", 0.0),
                "mi": m.get("mi", None),
                "loc": m.get("loc", 0),
                "sloc": m.get("sloc", 0),
                # smell aggregates
                "clone_ratio": m.get("clone_ratio", None),
                "dup_lines": m.get("dup_lines", None),
                "dup_blocks": m.get("dup_blocks", None),
                "max_nesting_depth": m.get("max_nesting_depth", 0),
                "max_stmt_tokens": m.get("max_stmt_tokens", 0),
                "magic_number_count": m.get("magic_number_count", 0),
                "unused_symbols": m.get("unused_symbols", None),
                # AST explainability
                "n_classes": m.get("n_classes", 0),
                "n_methods": m.get("n_methods", 0),
                "max_class_methods": m.get("max_class_methods", 0),
                "max_class_wmc": m.get("max_class_wmc", 0.0),
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    top = ranked[: max(0, int(top_k))]

    payload = {
        "repo": str(repo),
        "generated_at": _now_iso(),
        "mi_threshold": mi_threshold,
        "top_k": top_k,
        "ranking": top,
    }
    artifact = _write_artifact(repo, "hotspots_python.json", payload)
    return {"status": "ok", "artifact": artifact, "top": top, "ranked_total": len(ranked)}


@mcp.tool()
def detect_large_module_candidates(
    repo_path: str,
    q: float = 0.95,
    require_both: bool = True,
    top_k: int = 50,
    include_errors: bool = False,
) -> Dict[str, Any]:
    """
    Detecta candidatos "Large module/class-ish" usando umbrales relativos (percentil q):
      - sloc >= p(q)
      - complexity_sum >= p(q)   (usa cc_sum; si no hay radon, cc_sum cae a AST proxy)
    """
    repo = _resolve_repo(repo_path)
    cache = _load_cache(repo)
    files_map: Dict[str, Any] = cache.get("files", {})

    rows: List[Dict[str, Any]] = []
    slocs: List[float] = []
    ccsums: List[float] = []

    for path, entry in files_map.items():
        m = (entry or {}).get("metrics") or {}
        if (not include_errors) and m.get("error"):
            continue

        sl = float(m.get("sloc", 0) or 0)
        cs = float(m.get("cc_sum", 0.0) or 0.0)
        slocs.append(sl)
        ccsums.append(cs)

        rows.append(
            {
                "path": path,
                "sloc": sl,
                "cc_sum": cs,
                "cc_max": float(m.get("cc_max", 0.0) or 0.0),
                "mi": m.get("mi", None),
                # smell aggregates
                "clone_ratio": m.get("clone_ratio", None),
                "dup_lines": m.get("dup_lines", None),
                "max_nesting_depth": m.get("max_nesting_depth", 0),
                "unused_symbols": m.get("unused_symbols", None),
                # AST explainability
                "n_classes": int(m.get("n_classes", 0) or 0),
                "n_methods": int(m.get("n_methods", 0) or 0),
                "max_class_methods": int(m.get("max_class_methods", 0) or 0),
                "max_class_wmc": float(m.get("max_class_wmc", 0.0) or 0.0),
                "stmt_count": int(m.get("stmt_count", 0) or 0),
                "ast_ok": m.get("ast_ok", None),
            }
        )

    p_sloc = _percentile([x for x in slocs if x > 0], q)
    p_ccsum = _percentile([x for x in ccsums if x > 0], q)

    candidates: List[Dict[str, Any]] = []
    for r in rows:
        cond_sloc = (p_sloc is not None) and (r["sloc"] >= p_sloc)
        cond_cc = (p_ccsum is not None) and (r["cc_sum"] >= p_ccsum)
        ok = (cond_sloc and cond_cc) if require_both else (cond_sloc or cond_cc)
        if ok:
            r["rule"] = f"sloc>={q:.2f} & cc_sum>={q:.2f}" if require_both else f"sloc>={q:.2f} OR cc_sum>={q:.2f}"
            candidates.append(r)

    candidates.sort(key=lambda x: (x["sloc"], x["cc_sum"]), reverse=True)
    candidates = candidates[: max(0, int(top_k))]

    payload = {
        "repo": str(repo),
        "generated_at": _now_iso(),
        "q": q,
        "p_sloc": p_sloc,
        "p_cc_sum": p_ccsum,
        "require_both": require_both,
        "candidates": candidates,
    }
    artifact = _write_artifact(repo, "large_module_candidates.json", payload)
    return {
        "status": "ok",
        "artifact": artifact,
        "thresholds": {"p_sloc": p_sloc, "p_cc_sum": p_ccsum},
        "candidates": candidates,
    }


@mcp.tool()
def export_repo_metrics_json(repo_path: str) -> Dict[str, Any]:
    """
    Exporta solo repo-level metrics (clones/architecture/history) a repo_metrics.json
    """
    repo = _resolve_repo(repo_path)
    cache = _load_cache(repo)
    payload = {
        "repo": str(repo),
        "generated_at": _now_iso(),
        "meta": cache.get("meta", {}),
        "repo_metrics": cache.get("repo", {}),
    }
    artifact = _write_artifact(repo, "repo_metrics.json", payload)
    return {"status": "ok", "artifact": artifact, "keys": sorted(list((cache.get("repo") or {}).keys()))}


# -----------------------------
# Entrypoint
# -----------------------------
def main() -> None:
    global ALLOWED_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Limita repo_path a esta carpeta (seguridad).")
    args = parser.parse_args()

    if args.root:
        ALLOWED_ROOT = Path(args.root).expanduser().resolve()

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
