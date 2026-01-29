#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KatalIA Metrics MCP Server (katalia-metrics)

This module provides:
- File-level metrics (LOC/SLOC/comments/blank, MI/CC via Radon when available)
- AST-derived signals (always available) to support smell detection
- Entity metrics (functions/classes): CC, SLOC, nesting, params, LCOM4 proxy, WMC proxy, etc.
- Repo-level metrics: clones, architecture import graph, git history proxies, churn
- Advanced design/OO-ish metrics (repo-level unless noted):
  - data_class_ratio
  - public_api_size (file-level + repo-level aggregates)
  - usage_count (repo-level; optional heavy)
  - abstract_has_logic_ratio
  - topic_entropy
  - api_overlap
  - direct_field_access_ratio (file-level + repo-level aggregate)
  - children_per_node
  - inheritance_depth
  - multiple_paths_count
  - inheritance_cycles
  - dup_across_subclasses
  - override_contract_violations
  - type_similarity_clusters
  - contract_breaking_overrides
"""

from __future__ import annotations

import argparse
import ast
import builtins
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
from math import ceil, floor, log
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP

# -----------------------------
# Optional deps
# -----------------------------
try:
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit
    from radon.raw import analyze as raw_analyze

    RADON_AVAILABLE = True
except Exception:
    RADON_AVAILABLE = False

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
    max_nesting_depth: int
    bool_op_count: int
    max_stmt_tokens: int
    magic_number_count: int
    match_without_wildcard: bool
    empty_except_handlers: int
    max_identifier_length: int
    overridable_call_in_constructor: bool

    # --- NEW: public API + direct field access (file-level) ---
    public_api_size: int
    attribute_access_count: int
    direct_field_access_count: int
    direct_field_access_ratio: float

    # clone metrics (repo-level clone analysis)
    clone_ratio: Optional[float]
    dup_lines: Optional[int]
    dup_blocks: Optional[int]

    # static lint metrics (optional: ruff)
    unused_symbols: Optional[int]

    analyzed_at: str
    error: Optional[str] = None

    # churn metrics (git history; used for refactor prioritization)
    churn_commits: Optional[int] = None
    churn_added: Optional[int] = None
    churn_deleted: Optional[int] = None
    churn_total: Optional[int] = None
    churn_last_modified: Optional[str] = None


@dataclass
class FileAnalysis:
    metrics: FileMetrics
    entities: Dict[str, Any]  # {"functions":[...], "classes":[...], "file_examples":{...}, ...}


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
    Default: cache inside repo => <repo>/.katalia/metrics_cache.json
    If KATALIA_CACHE_DIR is defined, cache is placed outside (less invasive).
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
    If KATALIA_ARTIFACT_DIR is set, write outside.
    If not, write inside repo in artifacts/katalia/
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
    """Proxy of decision points (McCabe-like) to estimate CC. Avoid nested defs/classes."""

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
            # wildcard case: MatchAs(name=None, pattern=None)
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

        def visit_FunctionDef(self, n: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, n: ast.AST) -> None:
            return

        def visit_ClassDef(self, n: ast.ClassDef) -> None:
            return

    v = V()
    v.visit(node)
    return int(v.c)


def _max_nesting_depth(fn_node: ast.AST) -> int:
    """Max block nesting depth inside a function/method. Ignore nested defs/classes."""
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
            if t.type in (
                tokenize.ENCODING,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.COMMENT,
            ):
                continue
            count += 1
        return int(count)
    except Exception:
        return int(len(re.findall(r"\\w+|[^\\s\\w]", stmt_src)))


def _max_stmt_tokens_in_node(code: str, node: ast.AST) -> int:
    """
    Compute max tokens among statements under a node.
    IMPORTANT: For module-level calls, pass the Module tree to avoid 0s on files without funcs.
    """
    class V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.max_tokens = 0

        def generic_visit(self, n: ast.AST) -> Any:
            if isinstance(n, ast.stmt):
                seg = ast.get_source_segment(code, n) or ""
                self.max_tokens = max(self.max_tokens, _stmt_token_count(seg))
            return super().generic_visit(n)

        def visit_FunctionDef(self, n: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, n: ast.AST) -> None:
            return

        def visit_ClassDef(self, n: ast.ClassDef) -> None:
            return

    v = V()
    v.visit(node)
    return int(v.max_tokens)


def _match_without_wildcard_in_node(node: ast.AST) -> bool:
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
            self.generic_visit(n)

        def visit_FunctionDef(self, n: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, n: ast.AST) -> None:
            return

        def visit_ClassDef(self, n: ast.ClassDef) -> None:
            return

    v = V()
    v.visit(node)
    return bool(v.found and v.missing)


def _empty_except_handlers_in_node(node: ast.AST) -> int:
    """Count except handlers whose body is empty-ish (pass or ellipsis only)."""

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
    Heuristic:
    - Count numeric literals (int/float) excluding -1/0/1
    - Return top 10 frequent values
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
    """start/end 1-based inclusive. SLOC: non-blank, non-comment-only."""
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
    parts = rel_path.replace("\\\\", "/").split("/")
    if not parts:
        return "."
    if parts[0] in {"src", "lib", "app"} and len(parts) > 2:
        return parts[1]
    return parts[0] if len(parts) > 1 else "."


def _extract_import_targets(tree: ast.AST) -> List[str]:
    """Return top-level module names imported (e.g., 'requests', 'pkg.sub' -> 'pkg')."""
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
    Codes:
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
        out = (res.stdout or "") + "\\n" + (res.stderr or "")
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


def _public_api_size_from_tree(tree: ast.AST) -> int:
    cnt = 0
    for n in getattr(tree, "body", []) or []:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nm = getattr(n, "name", "") or ""
            if nm and not nm.startswith("_"):
                cnt += 1
    return int(cnt)


def _direct_field_access_counts(tree: ast.AST) -> Tuple[int, int]:
    """
    Heuristic for 'direct field access':
    - attribute access where value is a Name not equal to 'self'/'cls' and attr is public (no leading '_')
    Returns: (attribute_access_total, direct_field_access_count)
    """
    total = 0
    direct = 0

    class V(ast.NodeVisitor):
        def visit_Attribute(self, n: ast.Attribute) -> None:
            nonlocal total, direct
            total += 1
            v = getattr(n, "value", None)
            if isinstance(v, ast.Name) and v.id not in {"self", "cls"}:
                if isinstance(n.attr, str) and n.attr and not n.attr.startswith("_"):
                    direct += 1
            self.generic_visit(n)

    V().visit(tree)
    return int(total), int(direct)


# -----------------------------
# Entity extraction (functions/classes)
# -----------------------------
_ABSTRACT_DECORATORS = {"abstractmethod", "abstractproperty"}


def _decorator_names(decorator_list: List[ast.AST]) -> List[str]:
    names: List[str] = []
    for d in decorator_list or []:
        if isinstance(d, ast.Name):
            names.append(d.id)
        elif isinstance(d, ast.Attribute):
            names.append(d.attr)
        elif isinstance(d, ast.Call):
            fn = getattr(d, "func", None)
            if isinstance(fn, ast.Name):
                names.append(fn.id)
            elif isinstance(fn, ast.Attribute):
                names.append(fn.attr)
    return names


def _is_abstract_class(node: ast.ClassDef) -> bool:
    # bases
    for b in getattr(node, "bases", []) or []:
        if isinstance(b, ast.Name) and b.id in {"ABC", "ABCMeta"}:
            return True
        if isinstance(b, ast.Attribute) and b.attr in {"ABC", "ABCMeta"}:
            return True
    # keywords: metaclass=ABCMeta
    for kw in getattr(node, "keywords", []) or []:
        if getattr(kw, "arg", None) == "metaclass":
            v = getattr(kw, "value", None)
            if isinstance(v, ast.Name) and v.id == "ABCMeta":
                return True
            if isinstance(v, ast.Attribute) and v.attr == "ABCMeta":
                return True
    # abstract methods
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decs = set(_decorator_names(getattr(item, "decorator_list", []) or []))
            if decs & _ABSTRACT_DECORATORS:
                return True
    return False


def _is_method_abstract(fn_node: ast.AST) -> bool:
    decs = set(_decorator_names(getattr(fn_node, "decorator_list", []) or []))
    return bool(decs & _ABSTRACT_DECORATORS)


def _signature_info(fn_node: ast.AST) -> Dict[str, Any]:
    """
    Lightweight signature shape for override checks.
    Includes required counts and whether *args/**kwargs exist.
    """
    args = getattr(fn_node, "args", None)
    if args is None:
        return {
            "req_pos_excl_self": 0,
            "pos_excl_self": 0,
            "req_kwonly": 0,
            "kwonly": 0,
            "vararg": False,
            "kwarg": False,
            "param_ann": {},
            "ret_ann": None,
        }

    posonly = list(getattr(args, "posonlyargs", []) or [])
    pos = list(getattr(args, "args", []) or [])
    kwonly = list(getattr(args, "kwonlyargs", []) or [])
    defaults = list(getattr(args, "defaults", []) or [])
    kw_defaults = list(getattr(args, "kw_defaults", []) or [])

    # positional params include posonly + args
    pos_params = posonly + pos
    # drop self/cls from left if present
    pos_names = [getattr(a, "arg", "") for a in pos_params]
    drop0 = 1 if pos_names and pos_names[0] in {"self", "cls"} else 0

    total_pos_excl = max(0, len(pos_params) - drop0)

    # defaults apply to last len(defaults) positional params (including self possibly)
    n_pos_total = len(pos_params)
    n_required_pos_total = max(0, n_pos_total - len(defaults))
    req_pos_excl = max(0, n_required_pos_total - drop0)

    # kwonly required: kw_defaults entries that are None mean required
    req_kwonly = 0
    for d in kw_defaults:
        if d is None:
            req_kwonly += 1

    # annotations (by name, excluding self/cls)
    param_ann: Dict[str, str] = {}

    def _ann_str(a: Optional[ast.AST]) -> Optional[str]:
        if a is None:
            return None
        try:
            return ast.unparse(a)
        except Exception:
            return None

    for i, a in enumerate(pos_params):
        nm = getattr(a, "arg", None)
        if not nm or (i == 0 and nm in {"self", "cls"}):
            continue
        s = _ann_str(getattr(a, "annotation", None))
        if s:
            param_ann[nm] = s

    for a in kwonly:
        nm = getattr(a, "arg", None)
        if not nm:
            continue
        s = _ann_str(getattr(a, "annotation", None))
        if s:
            param_ann[nm] = s

    ret_ann = _ann_str(getattr(fn_node, "returns", None))

    return {
        "req_pos_excl_self": int(req_pos_excl),
        "pos_excl_self": int(total_pos_excl),
        "req_kwonly": int(req_kwonly),
        "kwonly": int(len(kwonly)),
        "vararg": bool(getattr(args, "vararg", None) is not None),
        "kwarg": bool(getattr(args, "kwarg", None) is not None),
        "param_ann": param_ann,
        "ret_ann": ret_ann,
    }


def _hash_method_body(code: str, fn_node: ast.AST) -> str:
    """
    Normalize method body source for comparing duplicates across subclasses.
    - Replace strings with STR, numbers with NUM
    - Keep names/operators reasonably
    """
    body = getattr(fn_node, "body", []) or []
    # drop docstring statement if present
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        if isinstance(getattr(body[0].value, "value", None), str):
            body = body[1:]
    parts: List[str] = []
    for st in body:
        seg = ast.get_source_segment(code, st) or ""
        if seg.strip():
            parts.append(seg)
    src = "\\n".join(parts).strip()
    if not src:
        return hashlib.sha1(b"").hexdigest()
    try:
        out: List[str] = []
        toks = tokenize.generate_tokens(io.StringIO(src).readline)
        for t in toks:
            if t.type in (tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT):
                continue
            if t.type == tokenize.STRING:
                out.append("STR")
                continue
            if t.type == tokenize.NUMBER:
                out.append("NUM")
                continue
            s = (t.string or "").strip()
            if s:
                out.append(s)
        norm = " ".join(out)
        return hashlib.sha1(norm.encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha1(src.encode("utf-8")).hexdigest()


def _overridable_calls_in_init(class_node: ast.ClassDef, code: str) -> Tuple[bool, List[Dict[str, Any]]]:
    """Detect calls like self.foo() inside __init__ where foo is a method in the class."""
    method_names = set()
    decorators_map: Dict[str, List[str]] = {}

    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_names.add(item.name)
            decorators_map[item.name] = _decorator_names(getattr(item, "decorator_list", []) or [])

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
      - all_fields: set(fields assigned (self.x=...) in __init__ + class assignments
      - getset counters
    """
    all_fields: set = set()
    method_fields: Dict[str, set] = {}
    getset = {"getter": 0, "setter": 0, "methods": 0}

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
            if name.startswith("get_") or name.startswith("is_") or name.startswith("has_"):
                getset["getter"] += 1
            if name.startswith("set_"):
                getset["setter"] += 1

            mv = MethodVisitor()
            mv.visit(item)
            method_fields[name] = set(mv.fields)
            all_fields |= set(mv.assign_fields)

    return method_fields, all_fields, getset


def _lcom4_from_method_fields(method_fields: Dict[str, set]) -> float:
    """LCOM4: number of connected components of methods connected by shared field usage."""
    methods = list(method_fields.keys())
    if len(methods) <= 1:
        return 1.0
    adj: Dict[str, set] = {m: set() for m in methods}
    for i, m1 in enumerate(methods):
        f1 = method_fields.get(m1, set())
        for m2 in methods[i + 1 :]:
            f2 = method_fields.get(m2, set())
            if f1 and f2 and (f1 & f2):
                adj[m1].add(m2)
                adj[m2].add(m1)
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
        "functions":[{...}],
        "classes":[{...}],
        "file_examples": {...}
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
                nm = getattr(b, "name", None)
                ln = getattr(b, "lineno", None)
                cx = getattr(b, "complexity", None)
                cls = getattr(b, "classname", None)
                if isinstance(nm, str) and isinstance(ln, int) and isinstance(cx, (int, float)):
                    radon_cc_by_name_line[(nm, ln)] = float(cx)
                    if cls:
                        radon_cc_by_name_line[(f"{cls}.{nm}", ln)] = float(cx)
        except Exception:
            radon_cc_by_name_line = {}

    magic_count, magic_examples = _magic_number_stats(tree)
    max_ident, ident_examples = _max_identifier_length_in_tree(tree)

    functions: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []

    def _fn_metrics(n: ast.AST, scope: str, class_name: Optional[str] = None) -> Dict[str, Any]:
        start = int(getattr(n, "lineno", 1) or 1)
        end = int(getattr(n, "end_lineno", start) or start)
        nm = getattr(n, "name", "fn") or "fn"
        sloc = _count_sloc_range(lines, start, end)
        cc = radon_cc_by_name_line.get((nm, start), _cyclomatic_proxy(n))
        if class_name:
            cc = radon_cc_by_name_line.get((f"{class_name}.{nm}", start), cc)

        nesting = _max_nesting_depth(n)
        bool_ops = _count_bool_ops(n)
        max_stmt_tokens = _max_stmt_tokens_in_node(code, n)
        empty_excepts = _empty_except_handlers_in_node(n)
        match_wo = _match_without_wildcard_in_node(n)
        sig = _signature_info(n)
        body_hash = _hash_method_body(code, n)

        out = {
            "scope": scope,
            "name": nm,
            "lineno": start,
            "end_lineno": end,
            "function_sloc": int(sloc),
            "function_cc": float(round(float(cc), 2)),
            "nesting_depth": int(nesting),
            "bool_op_count": int(bool_ops),
            "max_stmt_tokens": int(max_stmt_tokens),
            "n_params": int(
                (len(getattr(getattr(n, "args", None), "posonlyargs", []) or [])
                 + len(getattr(getattr(n, "args", None), "args", []) or [])
                 + len(getattr(getattr(n, "args", None), "kwonlyargs", []) or [])
                 + (1 if getattr(getattr(n, "args", None), "vararg", None) is not None else 0)
                 + (1 if getattr(getattr(n, "args", None), "kwarg", None) is not None else 0))
                if getattr(n, "args", None) is not None else 0
            ),
            "empty_except_handlers": int(empty_excepts),
            "match_without_wildcard": bool(match_wo),
            "signature": sig,
            "body_hash": body_hash,
            "is_abstract": bool(_is_method_abstract(n)),
        }
        if class_name:
            out["class_name"] = class_name
        return out

    class ModuleVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, n: ast.FunctionDef) -> None:
            functions.append(_fn_metrics(n, scope="function"))
            # do not recurse into nested defs automatically; keep top-level
            return

        def visit_AsyncFunctionDef(self, n: ast.AST) -> None:
            functions.append(_fn_metrics(n, scope="function"))
            return

        def visit_ClassDef(self, n: ast.ClassDef) -> None:
            start = int(getattr(n, "lineno", 1) or 1)
            end = int(getattr(n, "end_lineno", start) or start)
            loc = int(max(1, end - start + 1))

            method_nodes: List[ast.AST] = []
            for item in n.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_nodes.append(item)

            method_count = len(method_nodes)
            wmc = 0.0
            methods_detail: List[Dict[str, Any]] = []
            max_m_nesting = 0

            # method names for API similarity clusters
            method_name_set: set = set()

            for m in method_nodes:
                mname = getattr(m, "name", "method") or "method"
                method_name_set.add(mname)
                mstart = int(getattr(m, "lineno", 1) or 1)
                mcc = radon_cc_by_name_line.get(
                    (mname, mstart),
                    radon_cc_by_name_line.get((f"{n.name}.{mname}", mstart), _cyclomatic_proxy(m)),
                )
                wmc += float(mcc)
                nesting = _max_nesting_depth(m)
                max_m_nesting = max(max_m_nesting, int(nesting))
                methods_detail.append(_fn_metrics(m, scope="method", class_name=n.name))

            method_fields, all_fields, getset = _class_field_access_sets(n)
            lcom4 = _lcom4_from_method_fields(method_fields)

            public_fields = [f for f in all_fields if isinstance(f, str) and not f.startswith("_")]
            public_field_ratio = (len(public_fields) / max(1, len(all_fields))) if all_fields else 0.0

            getter_setter_ratio = (
                float(getset["getter"] + getset["setter"]) / float(max(1, getset["methods"]))
                if getset["methods"] > 0 else 0.0
            )

            ov, call_sites = _overridable_calls_in_init(n, code)

            bases: List[str] = []
            for b in getattr(n, "bases", []) or []:
                try:
                    bases.append(ast.unparse(b))
                except Exception:
                    if isinstance(b, ast.Name):
                        bases.append(b.id)
                    elif isinstance(b, ast.Attribute):
                        bases.append(b.attr)

            abstract_class = _is_abstract_class(n)

            # abstract_has_logic: abstract class with non-trivial concrete methods
            has_logic = False
            if abstract_class:
                for m in method_nodes:
                    if _is_method_abstract(m):
                        continue
                    # logic heuristic: cc>1 or nesting>0 or sloc>3
                    mm = _fn_metrics(m, scope="method", class_name=n.name)
                    if (mm.get("function_cc", 1.0) or 1.0) > 1.0 or (mm.get("nesting_depth", 0) or 0) > 0 or (mm.get("function_sloc", 0) or 0) > 3:
                        has_logic = True
                        break

            # data class heuristic
            # - has fields
            # - most methods are getters/setters OR small WMC
            # - low cohesion not required, but help
            data_class_candidate = bool(
                len(all_fields) >= 2
                and (
                    getter_setter_ratio >= 0.6
                    or (method_count <= 6 and wmc <= 10.0)
                )
            )

            # public API size for class: public methods + public fields
            public_methods = [mn for mn in method_name_set if mn and not mn.startswith("_")]
            class_public_api_size = int(len(public_methods) + len(public_fields))

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
                    "field_names": sorted([x for x in all_fields if isinstance(x, str)])[:500],
                    "public_field_ratio": float(round(float(public_field_ratio), 4)),
                    "getter_setter_ratio": float(round(float(getter_setter_ratio), 4)),
                    "overridable_call_in_constructor": bool(ov),
                    "overridable_call_sites": call_sites,
                    "max_method_nesting_depth": int(max_m_nesting),
                    # NEW for advanced metrics
                    "bases": bases,
                    "abstract_class": bool(abstract_class),
                    "abstract_has_logic": bool(has_logic),
                    "data_class_candidate": bool(data_class_candidate),
                    "class_public_api_size": int(class_public_api_size),
                    "method_names": sorted(list(method_name_set))[:1000],
                }
            )

            functions.extend(methods_detail)

            # recurse into nested classes
            for item in n.body:
                if isinstance(item, ast.ClassDef):
                    self.visit_ClassDef(item)

    ModuleVisitor().visit(tree)

    return {
        "functions": functions,
        "classes": classes,
        "file_examples": {
            "magic_numbers": magic_examples,
            "long_identifiers": ident_examples,
            "magic_number_count": magic_count,
            "max_identifier_length": max_ident,
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

        # Improved stmt_count: count statements but don't descend into nested defs inside funcs.
        class _StmtV(ast.NodeVisitor):
            def __init__(self) -> None:
                self.count = 0
                self.func_depth = 0

            def generic_visit(self, node: ast.AST) -> Any:
                if isinstance(node, ast.stmt):
                    self.count += 1
                return super().generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
                self.count += 1
                if self.func_depth >= 1:
                    return None
                self.func_depth += 1
                for ch in node.body:
                    self.visit(ch)
                self.func_depth -= 1
                return None

            def visit_AsyncFunctionDef(self, node: ast.AST) -> Any:
                self.count += 1
                if self.func_depth >= 1:
                    return None
                self.func_depth += 1
                for ch in getattr(node, "body", []) or []:
                    self.visit(ch)
                self.func_depth -= 1
                return None

            def visit_ClassDef(self, node: ast.ClassDef) -> Any:
                self.count += 1
                if self.func_depth >= 1:
                    return None
                for ch in node.body:
                    self.visit(ch)
                return None

        sv = _StmtV()
        sv.visit(tree)

        return {
            "ast_ok": True,
            "stmt_count": int(sv.count),
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
        sloc = max(0, loc - blank - comments)
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
                "ast_ok", "stmt_count", "n_classes", "n_functions", "n_methods",
                "max_class_methods", "max_class_wmc", "ast_cyc_sum", "ast_cyc_max"
            ]},
        }

    raw = raw_analyze(code)
    mi = float(mi_visit(code, multi=False))
    blocks = cc_visit(code) or []

    # Keep CC only for top-level funcs + class methods (exclude nested defs)
    wanted: set[tuple[str, int]] = set()
    try:
        tree = ast.parse(code)
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                wanted.add((n.name, int(getattr(n, "lineno", 0) or 0)))
            elif isinstance(n, ast.ClassDef):
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        ln = int(getattr(m, "lineno", 0) or 0)
                        wanted.add((m.name, ln))
                        wanted.add((f"{n.name}.{m.name}", ln))
    except Exception:
        wanted = set()

    ccs: List[float] = []
    for b in blocks:
        nm = getattr(b, "name", None)
        ln = int(getattr(b, "lineno", 0) or 0)
        cx = float(getattr(b, "complexity", 0.0) or 0.0)
        cls = getattr(b, "classname", None)

        key1 = (str(nm), ln) if nm else None
        key2 = (f"{cls}.{nm}", ln) if (cls and nm) else None

        if wanted:
            if (key1 and key1 in wanted) or (key2 and key2 in wanted):
                ccs.append(cx)
        else:
            ccs.append(cx)

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
            "ast_ok", "stmt_count", "n_classes", "n_functions", "n_methods",
            "max_class_methods", "max_class_wmc", "ast_cyc_sum", "ast_cyc_max"
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
    rel_norm = rel_path.replace("\\\\", "/")
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
    smell_defaults = dict(
        max_nesting_depth=0,
        bool_op_count=0,
        max_stmt_tokens=0,
        magic_number_count=0,
        match_without_wildcard=False,
        empty_except_handlers=0,
        max_identifier_length=0,
        overridable_call_in_constructor=False,
        public_api_size=0,
        attribute_access_count=0,
        direct_field_access_count=0,
        direct_field_access_ratio=0.0,
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
                loc=0,
                sloc=0,
                comments=0,
                multi=0,
                blank=0,
                cc_sum=0.0,
                cc_avg=0.0,
                cc_max=0.0,
                cc_blocks=0,
                mi=None,
                analyzed_at=analyzed_at,
                error="file_not_found_or_not_py",
                **ast_defaults,
                **smell_defaults,
            )
            return FileAnalysis(metrics=fm, entities=entities)

        if _is_ignored(repo, f, include_tests):
            fm = FileMetrics(
                path=rel_norm,
                loc=0,
                sloc=0,
                comments=0,
                multi=0,
                blank=0,
                cc_sum=0.0,
                cc_avg=0.0,
                cc_max=0.0,
                cc_blocks=0,
                mi=None,
                analyzed_at=analyzed_at,
                error="ignored",
                **ast_defaults,
                **smell_defaults,
            )
            return FileAnalysis(metrics=fm, entities=entities)

        if max_file_bytes > 0 and f.stat().st_size > max_file_bytes:
            fm = FileMetrics(
                path=rel_norm,
                loc=0,
                sloc=0,
                comments=0,
                multi=0,
                blank=0,
                cc_sum=0.0,
                cc_avg=0.0,
                cc_max=0.0,
                cc_blocks=0,
                mi=None,
                analyzed_at=analyzed_at,
                error="file_too_large",
                **ast_defaults,
                **smell_defaults,
            )
            return FileAnalysis(metrics=fm, entities=entities)

        code = _safe_read_text(f)
        m = _compute_metrics_for_code(code, require_radon=require_radon, use_radon=use_radon)

        # parse AST once for file-level extras (public api, direct field access, magic/id len)
        try:
            tree = ast.parse(code)
            public_api_size = _public_api_size_from_tree(tree)
            attr_total, direct_fields = _direct_field_access_counts(tree)
            magic_count, _ = _magic_number_stats(tree)
            max_ident, _ = _max_identifier_length_in_tree(tree)
            module_stmt_tokens = _max_stmt_tokens_in_node(code, tree)  # FIX: module-level max stmt tokens
        except Exception:
            tree = None
            public_api_size = 0
            attr_total, direct_fields = 0, 0
            magic_count, max_ident = 0, 0
            module_stmt_tokens = 0

        if compute_entities:
            entities = _analyze_entities(code, use_radon=use_radon and RADON_AVAILABLE)

        # attach module/component info to entities (helps repo-level metrics without reparsing)
        module = rel_norm[:-3].replace("/", ".")
        component = _infer_component_from_path(rel_norm)
        for cl in entities.get("classes", []) or []:
            cl["module"] = module
            cl["component"] = component
            cl["qualname"] = f"{module}.{cl.get('name','')}".strip(".")
        for fn in entities.get("functions", []) or []:
            fn["module"] = module
            fn["component"] = component
            if fn.get("scope") == "method" and fn.get("class_name"):
                fn["qualname"] = f"{module}.{fn.get('class_name')}.{fn.get('name')}"
            else:
                fn["qualname"] = f"{module}.{fn.get('name')}"

        # file aggregates from entities
        max_nesting = 0
        bool_ops_total = 0
        max_stmt_tokens = int(module_stmt_tokens)  # start with module-level
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

        unused = None
        if compute_ruff_unused:
            unused = _ruff_unused_symbols(repo, rel_norm)

        ratio = float(direct_fields) / float(attr_total) if attr_total > 0 else 0.0

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
            public_api_size=int(public_api_size),
            attribute_access_count=int(attr_total),
            direct_field_access_count=int(direct_fields),
            direct_field_access_ratio=float(round(ratio, 6)),
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
            loc=0,
            sloc=0,
            comments=0,
            multi=0,
            blank=0,
            cc_sum=0.0,
            cc_avg=0.0,
            cc_max=0.0,
            cc_blocks=0,
            mi=None,
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
        "ast_ok": m.get("ast_ok", None),
        "stmt_count": m.get("stmt_count", 0),
        "n_classes": m.get("n_classes", 0),
        "n_functions": m.get("n_functions", 0),
        "n_methods": m.get("n_methods", 0),
        "max_class_methods": m.get("max_class_methods", 0),
        "max_class_wmc": m.get("max_class_wmc", 0.0),
        "ast_cyc_sum": m.get("ast_cyc_sum", 0.0),
        "ast_cyc_max": m.get("ast_cyc_max", 0.0),
        "max_nesting_depth": m.get("max_nesting_depth", 0),
        "bool_op_count": m.get("bool_op_count", 0),
        "max_stmt_tokens": m.get("max_stmt_tokens", 0),
        "magic_number_count": m.get("magic_number_count", 0),
        "match_without_wildcard": m.get("match_without_wildcard", False),
        "empty_except_handlers": m.get("empty_except_handlers", 0),
        "max_identifier_length": m.get("max_identifier_length", 0),
        "overridable_call_in_constructor": m.get("overridable_call_in_constructor", False),
        "public_api_size": m.get("public_api_size", 0),
        "attribute_access_count": m.get("attribute_access_count", 0),
        "direct_field_access_count": m.get("direct_field_access_count", 0),
        "direct_field_access_ratio": m.get("direct_field_access_ratio", 0.0),
        "clone_ratio": m.get("clone_ratio", None),
        "dup_lines": m.get("dup_lines", None),
        "dup_blocks": m.get("dup_blocks", None),
        "unused_symbols": m.get("unused_symbols", None),
        "churn_commits": m.get("churn_commits", None),
        "churn_added": m.get("churn_added", None),
        "churn_deleted": m.get("churn_deleted", None),
        "churn_total": m.get("churn_total", None),
        "churn_last_modified": m.get("churn_last_modified", None),
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
        return [re.sub(r"\\s+", " ", ln.strip()) for ln in code.splitlines() if ln.strip()]

    normalized = [" ".join(x) for x in out_lines if x]
    return normalized


def _compute_clone_metrics_for_repo(repo: Path, include_tests: bool, min_block_lines: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    Hash blocks of N normalized lines.
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

    occ: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for rel, lines in norm_lines_by_file.items():
        if len(lines) < min_block_lines:
            continue
        for i in range(0, len(lines) - min_block_lines + 1):
            block = "\\n".join(lines[i : i + min_block_lines])
            h = hashlib.sha1(block.encode("utf-8")).hexdigest()
            occ[h].append((rel, i))

    dup_line_marks: Dict[str, set] = {rel: set() for rel in files}
    dup_blocks_count: Counter[str] = Counter()

    for h, positions in occ.items():
        if len(positions) <= 1:
            continue
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
    Component graph using top-level folder as component.
    Edge A->B if file in component A imports module B that exists as a top-level component.
    Adds: public_api_size (repo + per-component), topic_entropy, api_overlap (by component).
    """
    py_files = [p for p in _iter_py_files(repo, include_tests)]
    existing_components = set()
    for p in py_files:
        rel = p.relative_to(repo).as_posix()
        existing_components.add(_infer_component_from_path(rel))

    edges: set = set()
    comp_public_api_size: Counter[str] = Counter()
    comp_api_names: Dict[str, set] = {c: set() for c in existing_components}

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

        pub = 0
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nm = getattr(n, "name", "") or ""
                if nm and not nm.startswith("_"):
                    pub += 1
                    comp_api_names.setdefault(comp, set()).add(nm)
        comp_public_api_size[comp] += pub

    comps = sorted(list(existing_components))
    n = len(comps)
    edge_list = sorted(list(edges))

    density = 0.0
    if n > 1:
        density = float(len(edges)) / float(n * (n - 1))

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
        if ia < ib:
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
        degree_centrality = nx.degree_centrality(G)
    else:
        degree_centrality = {}

    entrypoint = max(comps, key=lambda c: fanin.get(c, 0)) if comps else None
    entrypoint_api_size = int(comp_public_api_size.get(entrypoint, 0)) if entrypoint else 0

    # NEW: topic_entropy (Shannon entropy over public API name tokens per repo)
    def tokenize_name(name: str) -> List[str]:
        if not name:
            return []
        # snake_case + camelCase split
        tmp = re.sub(r"([a-z0-9])([A-Z])", r"\\1_\\2", name)
        parts = re.split(r"[^A-Za-z0-9]+", tmp)
        toks = [p.lower() for p in parts if p]
        return toks

    tok_counts: Counter[str] = Counter()
    for comp, names in comp_api_names.items():
        for nm in names:
            for t in tokenize_name(nm):
                tok_counts[t] += 1
    topic_entropy = round(_entropy_from_counter(tok_counts), 6)

    # NEW: api_overlap (Jaccard overlap between components public APIs)
    overlaps: List[Dict[str, Any]] = []
    comps_list = comps[:]
    for i, a in enumerate(comps_list):
        A = comp_api_names.get(a, set())
        for b in comps_list[i + 1 :]:
            B = comp_api_names.get(b, set())
            if not A and not B:
                continue
            inter = len(A & B)
            union = len(A | B) if (A | B) else 1
            j = float(inter) / float(union)
            if j > 0:
                overlaps.append({"a": a, "b": b, "jaccard": round(j, 6), "shared": inter})
    overlaps.sort(key=lambda x: x["jaccard"], reverse=True)
    api_overlap = {
        "pairs_considered": int(len(comps) * (len(comps) - 1) / 2) if len(comps) > 1 else 0,
        "nonzero_pairs": int(len(overlaps)),
        "max_jaccard": overlaps[0]["jaccard"] if overlaps else 0.0,
        "top_pairs": overlaps[:20],
    }

    return {
        "components": comps,
        "edges": edge_list,
        "edge_count": len(edge_list),
        "graph_density": round(float(density), 6),
        "fanin": {k: int(v) for k, v in fanin.items()},
        "fanout": {k: int(v) for k, v in fanout.items()},
        "instability": instability,
        "stable_depends_on_unstable": stable_depends_on_unstable,
        "dependency_cycles": {"count": len(cycles), "cycles": cycles[:50]},
        "degree_centrality": degree_centrality,
        "entrypoint": entrypoint,
        "entrypoint_api_size": entrypoint_api_size,
        "networkx_available": NETWORKX_AVAILABLE,
        # NEW
        "public_api_size": int(sum(comp_public_api_size.values())),
        "public_api_size_by_component": {k: int(v) for k, v in comp_public_api_size.items()},
        "topic_entropy": topic_entropy,
        "api_overlap": api_overlap,
    }


# -----------------------------
# Repo-level: git history + churn + improved commit classification + shotgun surgery
# -----------------------------
ISSUE_RE = re.compile(r"\\b([A-Z][A-Z0-9]+-\\d+)\\b")
CONVENTIONAL_RE = re.compile(r"^(feat|fix|refactor|chore|docs|test|build|ci|perf|style|revert)(\\(.+\\))?:", re.IGNORECASE)

_HEURISTIC_TYPES = {
    "fix": {"fix", "bug", "hotfix", "issue", "error", "crash", "patch"},
    "feat": {"feat", "feature", "add", "added", "new", "implement"},
    "refactor": {"refactor", "cleanup", "restructure", "rework", "simplify"},
    "docs": {"docs", "doc", "readme"},
    "test": {"test", "tests", "unittest", "pytest"},
    "perf": {"perf", "performance", "optimize", "optimization"},
    "ci": {"ci", "pipeline", "github", "actions", "jenkins"},
    "build": {"build", "deps", "dependency", "bump", "upgrade", "downgrade"},
    "style": {"style", "format", "lint", "ruff", "black", "isort"},
    "chore": {"chore", "misc", "housekeeping"},
    "revert": {"revert"},
}


def _classify_commit_type(subject: str) -> str:
    """
    Improved commit classifier:
    - Conventional Commits first
    - Heuristics fallback when repo doesn't cooperate
    """
    s = (subject or "").strip()
    if not s:
        return "other"
    m = CONVENTIONAL_RE.match(s)
    if m:
        return (m.group(1) or "other").lower()

    low = s.lower()
    # strip leading tags like "[XYZ]" "(...)"
    low = re.sub(r"^[\\[\\(].*?[\\]\\)]\\s*", "", low)

    # quick rules
    if low.startswith("merge ") or low.startswith("merged "):
        return "other"
    if low.startswith("revert"):
        return "revert"

    tokens = set(re.findall(r"[a-zA-Z]+", low))
    for typ, keys in _HEURISTIC_TYPES.items():
        if tokens & keys:
            return typ
    return "other"


def _git_available(repo: Path) -> bool:
    try:
        res = subprocess.run(["git", "-C", str(repo), "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
        return res.returncode == 0 and "true" in (res.stdout or "").lower()
    except Exception:
        return False


def _git_head(repo: Path) -> Optional[str]:
    try:
        res = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if res.returncode != 0:
            return None
        return (res.stdout or "").strip() or None
    except Exception:
        return None


def _normalize_git_numstat_path(p: str) -> str:
    """
    Normalize weird rename formats in git numstat:
      - "old => new"
      - "src/{old => new}/file.py"
    """
    s = (p or "").strip()
    if not s:
        return s

    brace_re = re.compile(r"\\{([^{}]*?)\\s*=>\\s*([^{}]*?)\\}")
    while True:
        m = brace_re.search(s)
        if not m:
            break
        s = s[: m.start()] + (m.group(2) or "") + s[m.end() :]

    if "=>" in s:
        s = s.split("=>")[-1].strip()

    return s.replace("\\\\", "/").strip()


def _compute_churn_metrics_for_repo(repo: Path, include_tests: bool, max_commits: int = 5000) -> Dict[str, Any]:
    """
    Per-file churn from git history:
    - churn_commits
    - churn_added/deleted/total
    - churn_last_modified
    """
    if not _git_available(repo):
        return {"git": False, "error": "not_a_git_repo_or_git_missing", "commits_scanned": 0, "files": {}, "head": None}

    head = _git_head(repo)

    cmd = [
        "git", "-C", str(repo), "log",
        f"-n{int(max_commits)}",
        "--no-merges",
        "--numstat",
        "--pretty=format:%H%x1f%ad",
        "--date=iso-strict",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        return {"git": True, "error": "git_log_failed", "stderr": (res.stderr or "")[:2000], "commits_scanned": 0, "files": {}, "head": head}

    files: Dict[str, Dict[str, Any]] = {}
    cur_date: Optional[str] = None
    commits_scanned = 0

    for ln in (res.stdout or "").splitlines():
        if "\\x1f" in ln:
            commits_scanned += 1
            parts = ln.split("\\x1f")
            cur_date = parts[1] if len(parts) > 1 else None
            continue

        if not ln.strip():
            continue

        parts = ln.split("\\t")
        if len(parts) < 3:
            continue

        add_s, del_s, path_s = parts[0].strip(), parts[1].strip(), "\\t".join(parts[2:]).strip()
        path_s = _normalize_git_numstat_path(path_s)

        if not path_s.endswith(".py"):
            continue

        p_abs = (repo / path_s).resolve()
        if p_abs.exists() and _is_ignored(repo, p_abs, include_tests):
            continue

        try:
            added = int(add_s) if add_s != "-" else 0
            deleted = int(del_s) if del_s != "-" else 0
        except Exception:
            continue

        e = files.get(path_s)
        if not e:
            e = {
                "churn_commits": 0,
                "churn_added": 0,
                "churn_deleted": 0,
                "churn_total": 0,
                "churn_last_modified": None,
            }
            files[path_s] = e

        e["churn_commits"] = int(e.get("churn_commits", 0) or 0) + 1
        e["churn_added"] = int(e.get("churn_added", 0) or 0) + int(added)
        e["churn_deleted"] = int(e.get("churn_deleted", 0) or 0) + int(deleted)
        e["churn_total"] = int(e.get("churn_total", 0) or 0) + int(added) + int(deleted)
        if e.get("churn_last_modified") is None and cur_date:
            e["churn_last_modified"] = cur_date

    return {"git": True, "head": head, "commits_scanned": commits_scanned, "files": files}


def _compute_git_history_metrics(repo: Path, include_tests: bool, max_commits: int = 3000) -> Dict[str, Any]:
    """
    Produces:
      - SHOTGUN_SURGERY instances (files_touched per commit; avg/max + sample)
      - per-file change_reason_entropy based on commit classifier (Conventional + heuristics)
      - feature_scatter from issue IDs in commit messages
    """
    if not _git_available(repo):
        return {"git": False, "error": "not_a_git_repo_or_git_missing"}

    cmd = [
        "git", "-C", str(repo), "log",
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
            files.append(fp.replace("\\\\", "/"))

        touched = len(set(files))
        files_touched_per_change.append(int(touched))

        ctype = _classify_commit_type(cur_subject or "")

        for fp in set(files):
            file_type_counts[fp][ctype] += 1

        issues = set()
        if isinstance(cur_subject, str):
            issues |= set(ISSUE_RE.findall(cur_subject))

        for iss in issues:
            for fp in set(files):
                issue_components[iss].add(_infer_component_from_path(fp))

        commit_records.append(
            {
                "hash": cur_hash,
                "subject": cur_subject,
                "date": cur_date,
                "files_touched": int(touched),
                "type": ctype,
                "issues": sorted(list(issues)),
                "files": sorted(list(set(files)))[:200],
            }
        )

        cur_hash, cur_subject, cur_date, cur_files[:] = None, None, None, []

    for ln in (res.stdout or "").splitlines():
        if "\\x1f" in ln:
            flush_commit()
            parts = ln.split("\\x1f")
            cur_hash = parts[0] if len(parts) > 0 else None
            cur_subject = parts[1] if len(parts) > 1 else ""
            cur_date = parts[2] if len(parts) > 2 else ""
            cur_files = []
        else:
            cur_files.append(ln)

    flush_commit()

    file_entropy: Dict[str, float] = {}
    for fp, c in file_type_counts.items():
        file_entropy[fp] = float(round(_entropy_from_counter(c), 6))

    feature_scatter = {iss: int(len(comps)) for iss, comps in issue_components.items()}
    feature_scatter_top = sorted(feature_scatter.items(), key=lambda x: x[1], reverse=True)[:200]

    def pct(vals: List[int], q: float) -> Optional[float]:
        if not vals:
            return None
        xs = sorted(vals)
        n = len(xs)
        pos = int(round((n - 1) * q))
        return float(xs[max(0, min(n - 1, pos))])

    # SHOTGUN_SURGERY instances (avg/max, plus top commits)
    commits_by_touched = sorted(commit_records, key=lambda r: r.get("files_touched", 0), reverse=True)
    shotgun = {
        "avg_files_touched": round(sum(files_touched_per_change) / max(1, len(files_touched_per_change)), 4),
        "max_files_touched": max(files_touched_per_change) if files_touched_per_change else 0,
        "p95_files_touched": pct(files_touched_per_change, 0.95),
        "p99_files_touched": pct(files_touched_per_change, 0.99),
        "instances_top": commits_by_touched[:200],
    }

    payload = {
        "git": True,
        "commits_analyzed": len(commit_records),
        "shotgun_surgery": shotgun,
        "files_touched_per_change": {
            "avg": shotgun["avg_files_touched"],
            "p95": shotgun["p95_files_touched"],
            "p99": shotgun["p99_files_touched"],
            "max": shotgun["max_files_touched"],
        },
        "change_reason_entropy": file_entropy,
        "feature_scatter": dict(feature_scatter_top),
        "feature_scatter_raw_count": len(feature_scatter),
        "commits_sample": commit_records[:200],
    }
    return payload


# -----------------------------
# Ranking logic (includes churn)
# -----------------------------
def _score_file(
    mi: Optional[float],
    loc: int,
    cc_max: float,
    churn_total: Optional[int] = None,
    churn_commits: Optional[int] = None,
    mi_threshold: float = 65.0,
) -> Tuple[float, str]:
    """
    Base hotspots: MI + size + max CC
    Add churn as log-scaled signal: high churn => prioritize (ROI).
    """
    reason = "signal"
    score = 0.0

    if mi is not None and mi < mi_threshold:
        score += (mi_threshold - mi) * 4.0
        reason = f"mi<{int(mi_threshold)}"

    score += loc * 0.5
    score += float(cc_max) * 0.6

    ct = int(churn_total or 0)
    cc = int(churn_commits or 0)
    if ct > 0 or cc > 0:
        score += log(1.0 + float(ct)) * 2.0
        score += log(1.0 + float(cc)) * 6.0
        if reason == "signal":
            reason = "churn"

    return round(score, 4), reason


# -----------------------------
# Advanced metrics (repo-level)
# -----------------------------

def _tokenize_name_for_topics(name: str) -> List[str]:
    if not name:
        return []
    tmp = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    parts = re.split(r"[^A-Za-z0-9]+", tmp)
    return [p.lower() for p in parts if p]


def _compute_topic_entropy_and_api_overlap_from_entities(
    classes: List[Dict[str, Any]],
    functions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute:
      - topic_entropy: entropy over tokens from public API names (top-level funcs + classes), across repo
      - api_overlap: Jaccard overlap of public API sets between components

    This is intentionally independent of the import-graph step.
    """
    api_by_comp: Dict[str, set] = defaultdict(set)
    tok_counts: Counter[str] = Counter()

    # classes: public class names
    for c in classes:
        nm = c.get("name")
        if isinstance(nm, str) and nm and not nm.startswith("_"):
            comp = str(c.get("component", "."))
            api_by_comp[comp].add(nm)
            for t in _tokenize_name_for_topics(nm):
                tok_counts[t] += 1

    # functions: only top-level public functions
    for fn in functions:
        if fn.get("scope") != "function":
            continue
        nm = fn.get("name")
        if isinstance(nm, str) and nm and not nm.startswith("_"):
            comp = str(fn.get("component", "."))
            api_by_comp[comp].add(nm)
            for t in _tokenize_name_for_topics(nm):
                tok_counts[t] += 1

    topic_entropy = round(_entropy_from_counter(tok_counts), 6)

    comps = sorted(api_by_comp.keys())
    overlaps: List[Dict[str, Any]] = []
    for i, a in enumerate(comps):
        A = api_by_comp.get(a, set())
        for b in comps[i + 1 :]:
            B = api_by_comp.get(b, set())
            if not A and not B:
                continue
            inter = len(A & B)
            union = len(A | B) if (A | B) else 1
            j = float(inter) / float(union)
            if j > 0:
                overlaps.append({"a": a, "b": b, "jaccard": round(j, 6), "shared": inter})
    overlaps.sort(key=lambda x: x["jaccard"], reverse=True)

    api_overlap = {
        "components": int(len(comps)),
        "pairs_considered": int(len(comps) * (len(comps) - 1) / 2) if len(comps) > 1 else 0,
        "nonzero_pairs": int(len(overlaps)),
        "max_jaccard": overlaps[0]["jaccard"] if overlaps else 0.0,
        "top_pairs": overlaps[:20],
    }

    return {"topic_entropy": topic_entropy, "api_overlap": api_overlap}
def _collect_cached_entities(cache_files: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (all_classes, all_functions) from cache."""
    classes: List[Dict[str, Any]] = []
    functions: List[Dict[str, Any]] = []
    for _, entry in (cache_files or {}).items():
        ents = (entry or {}).get("entities") or {}
        classes.extend(list(ents.get("classes", []) or []))
        functions.extend(list(ents.get("functions", []) or []))
    return classes, functions


def _build_inheritance_graph(classes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build an inheritance graph among project classes (best-effort resolution).
    Nodes are class qualnames.
    """
    by_qual: Dict[str, Dict[str, Any]] = {}
    by_name: Dict[str, List[str]] = defaultdict(list)
    for c in classes:
        qn = c.get("qualname")
        nm = c.get("name")
        if not qn or not nm:
            continue
        by_qual[qn] = c
        by_name[str(nm)].append(qn)

    def resolve_base(base_expr: str, child: Dict[str, Any]) -> Optional[str]:
        if not base_expr:
            return None
        b = base_expr.strip()
        if b in by_qual:
            return b
        short = b.split(".")[-1]
        cands = by_name.get(short, [])
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        # try same component
        child_comp = child.get("component")
        same_comp = [q for q in cands if (by_qual.get(q, {}).get("component") == child_comp)]
        if len(same_comp) == 1:
            return same_comp[0]
        # try same module
        child_mod = child.get("module")
        same_mod = [q for q in cands if (by_qual.get(q, {}).get("module") == child_mod)]
        if len(same_mod) == 1:
            return same_mod[0]
        return None

    edges: List[Tuple[str, str]] = []
    parents_of: Dict[str, set] = defaultdict(set)
    children_of: Dict[str, set] = defaultdict(set)

    for qn, c in by_qual.items():
        bases = c.get("bases", []) or []
        for b in bases:
            parent = resolve_base(str(b), c)
            if parent and parent != qn:
                edges.append((parent, qn))
                parents_of[qn].add(parent)
                children_of[parent].add(qn)

    nodes = sorted(list(by_qual.keys()))
    return {"nodes": nodes, "edges": edges, "parents_of": parents_of, "children_of": children_of, "by_qual": by_qual}


def _compute_inheritance_metrics(classes: List[Dict[str, Any]]) -> Dict[str, Any]:
    g = _build_inheritance_graph(classes)
    nodes: List[str] = g["nodes"]
    edges: List[Tuple[str, str]] = g["edges"]
    parents_of: Dict[str, set] = g["parents_of"]
    children_of: Dict[str, set] = g["children_of"]

    n = len(nodes)
    e = len(edges)
    children_per_node_avg = float(e) / float(n) if n > 0 else 0.0
    children_per_node_max = max((len(children_of.get(x, set())) for x in nodes), default=0)

    cycles: List[List[str]] = []
    if NETWORKX_AVAILABLE and n <= 5000:
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        try:
            cycles = [list(c) for c in nx.simple_cycles(G)]
        except Exception:
            cycles = []
    else:
        # simple cycle detection (limited)
        graph = {u: list(children_of.get(u, set())) for u in nodes}
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
                    try:
                        idx = stack.index(v)
                        cyc = stack[idx:] + [v]
                        if len(cycles) < 50:
                            cycles.append(cyc)
                    except Exception:
                        pass
            stack.pop()
            onstack.remove(u)

        for u in nodes:
            if u not in visited:
                dfs(u)
            if len(cycles) >= 50:
                break

    # inheritance depth (longest path) ignoring cycles via memo with cycle guard
    depth_memo: Dict[str, int] = {}
    visiting: set = set()

    def depth(u: str) -> int:
        if u in depth_memo:
            return depth_memo[u]
        if u in visiting:
            return 0
        visiting.add(u)
        dmax = 0
        for ch in children_of.get(u, set()):
            dmax = max(dmax, 1 + depth(ch))
        visiting.remove(u)
        depth_memo[u] = dmax
        return dmax

    depths = [depth(u) for u in nodes]
    max_depth = max(depths) if depths else 0
    avg_depth = float(sum(depths)) / float(len(depths)) if depths else 0.0

    # multiple_paths_count heuristic (diamond-ish): nodes with >=2 parents whose parents share an ancestor
    # ancestors computed via DFS
    ancestors_memo: Dict[str, set] = {}

    def ancestors(u: str) -> set:
        if u in ancestors_memo:
            return ancestors_memo[u]
        anc = set()
        for p in parents_of.get(u, set()):
            anc.add(p)
            anc |= ancestors(p)
        ancestors_memo[u] = anc
        return anc

    diamond = 0
    multi_inh = 0
    for u in nodes:
        ps = list(parents_of.get(u, set()))
        if len(ps) >= 2:
            multi_inh += 1
            a0 = ancestors(ps[0]) | {ps[0]}
            for p in ps[1:]:
                if a0 & (ancestors(p) | {p}):
                    diamond += 1
                    break

    return {
        "children_per_node": {"avg": round(children_per_node_avg, 6), "max": int(children_per_node_max)},
        "inheritance_depth": {"max": int(max_depth), "avg": round(avg_depth, 6)},
        "multiple_paths_count": int(diamond),
        "multiple_inheritance_nodes": int(multi_inh),
        "inheritance_cycles": {"count": int(len(cycles)), "cycles": cycles[:50]},
        "edge_count": int(e),
        "node_count": int(n),
    }


def _compute_dup_across_subclasses(classes: List[Dict[str, Any]], functions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Duplication across subclasses:
    For a base class, if 2+ subclasses contain identical bodies for the same method name.
    Uses method body_hash.
    """
    # map class qualname -> methods {name -> body_hash}
    methods_by_class: Dict[str, Dict[str, str]] = defaultdict(dict)
    sig_by_class: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for fn in functions:
        if fn.get("scope") != "method":
            continue
        qn = fn.get("qualname")
        if not qn:
            continue
        # qn is module.Class.method
        parts = str(qn).split(".")
        if len(parts) < 3:
            continue
        class_qn = ".".join(parts[:-1])
        mname = parts[-1]
        methods_by_class[class_qn][mname] = str(fn.get("body_hash", ""))
        sig_by_class[class_qn][mname] = fn.get("signature", {}) or {}

    # inheritance edges for parent->child
    g = _build_inheritance_graph(classes)
    edges: List[Tuple[str, str]] = g["edges"]

    base_to_children: Dict[str, List[str]] = defaultdict(list)
    for base, child in edges:
        base_to_children[base].append(child)

    dups: List[Dict[str, Any]] = []
    dup_groups = 0
    dup_instances = 0

    for base, kids in base_to_children.items():
        if len(kids) < 2:
            continue
        # method name -> hash -> list of subclasses
        per_method: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        for k in kids:
            for mname, h in methods_by_class.get(k, {}).items():
                if not h:
                    continue
                per_method[mname][h].append(k)
        for mname, hmap in per_method.items():
            for h, subs in hmap.items():
                if len(subs) >= 2:
                    dup_groups += 1
                    dup_instances += len(subs)
                    dups.append(
                        {
                            "base": base,
                            "method": mname,
                            "hash": h,
                            "subclasses": subs[:50],
                            "count": len(subs),
                        }
                    )

    dups.sort(key=lambda x: x["count"], reverse=True)
    return {
        "dup_groups": int(dup_groups),
        "dup_instances": int(dup_instances),
        "top": dups[:50],
    }


def _compare_signatures(base_sig: Dict[str, Any], sub_sig: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Return (ok, reasons) for override compatibility.
    Conservative: flags obvious LSP-ish breaks.
    """
    reasons: List[str] = []
    b_req = int(base_sig.get("req_pos_excl_self", 0) or 0)
    b_pos = int(base_sig.get("pos_excl_self", 0) or 0)
    b_req_kw = int(base_sig.get("req_kwonly", 0) or 0)
    b_kw = int(base_sig.get("kwonly", 0) or 0)
    b_var = bool(base_sig.get("vararg", False))
    b_kwarg = bool(base_sig.get("kwarg", False))

    s_req = int(sub_sig.get("req_pos_excl_self", 0) or 0)
    s_pos = int(sub_sig.get("pos_excl_self", 0) or 0)
    s_req_kw = int(sub_sig.get("req_kwonly", 0) or 0)
    s_kw = int(sub_sig.get("kwonly", 0) or 0)
    s_var = bool(sub_sig.get("vararg", False))
    s_kwarg = bool(sub_sig.get("kwarg", False))

    # cannot require more positional params than base
    if s_req > b_req:
        reasons.append("requires_more_positional_args")
    # if base has *args, override should also allow varargs or accept enough
    if b_var and not s_var and s_pos < b_pos:
        reasons.append("drops_varargs_support")
    # if override accepts fewer positional without vararg
    if (not s_var) and s_pos < b_req:
        reasons.append("accepts_fewer_positional_than_required")
    # kwonly required should not increase
    if s_req_kw > b_req_kw:
        reasons.append("requires_more_kwonly_args")
    # if base has **kwargs, override should not drop it
    if b_kwarg and not s_kwarg:
        reasons.append("drops_kwargs_support")

    ok = len(reasons) == 0
    return ok, reasons


def _compute_override_contract_metrics(classes: List[Dict[str, Any]], functions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    override_contract_violations:
      - signature incompatibility between base and overriding method

    contract_breaking_overrides:
      - annotation mismatch between base and overriding method (when both specified)
    """
    # build class->method signature/annotation
    methods: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for fn in functions:
        if fn.get("scope") != "method":
            continue
        qn = fn.get("qualname")
        if not qn:
            continue
        parts = str(qn).split(".")
        if len(parts) < 3:
            continue
        class_qn = ".".join(parts[:-1])
        mname = parts[-1]
        methods[class_qn][mname] = fn.get("signature", {}) or {}

    g = _build_inheritance_graph(classes)
    edges: List[Tuple[str, str]] = g["edges"]

    violations = 0
    breaking = 0
    details_v: List[Dict[str, Any]] = []
    details_b: List[Dict[str, Any]] = []

    for base, sub in edges:
        base_methods = methods.get(base, {})
        sub_methods = methods.get(sub, {})
        if not base_methods or not sub_methods:
            continue
        for mname, sub_sig in sub_methods.items():
            if mname.startswith("__") and mname.endswith("__"):
                continue
            if mname not in base_methods:
                continue
            base_sig = base_methods[mname] or {}
            ok, reasons = _compare_signatures(base_sig, sub_sig)
            if not ok:
                violations += 1
                if len(details_v) < 200:
                    details_v.append({"base": base, "sub": sub, "method": mname, "reasons": reasons})

            # contract breaking by annotations
            b_ret = base_sig.get("ret_ann")
            s_ret = sub_sig.get("ret_ann")
            b_pa = base_sig.get("param_ann") or {}
            s_pa = sub_sig.get("param_ann") or {}

            ann_break = False
            ann_reasons: List[str] = []
            if b_ret and s_ret and str(b_ret) != str(s_ret):
                ann_break = True
                ann_reasons.append("return_annotation_changed")

            # compare param annotations on overlapping names
            for pname, bann in b_pa.items():
                sann = s_pa.get(pname)
                if bann and sann and str(bann) != str(sann):
                    ann_break = True
                    ann_reasons.append(f"param_annotation_changed:{pname}")
                    break

            if ann_break:
                breaking += 1
                if len(details_b) < 200:
                    details_b.append({"base": base, "sub": sub, "method": mname, "reasons": ann_reasons})

    return {
        "override_contract_violations": int(violations),
        "contract_breaking_overrides": int(breaking),
        "violations_sample": details_v,
        "breaking_sample": details_b,
    }


def _compute_data_and_abstract_ratios(classes: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(classes)
    data_cnt = sum(1 for c in classes if c.get("data_class_candidate"))
    abs_cnt = sum(1 for c in classes if c.get("abstract_class"))
    abs_logic = sum(1 for c in classes if c.get("abstract_class") and c.get("abstract_has_logic"))
    data_ratio = float(data_cnt) / float(total) if total > 0 else 0.0
    abs_logic_ratio = float(abs_logic) / float(abs_cnt) if abs_cnt > 0 else 0.0
    return {
        "data_class_ratio": round(data_ratio, 6),
        "data_class_count": int(data_cnt),
        "class_count": int(total),
        "abstract_class_count": int(abs_cnt),
        "abstract_has_logic_ratio": round(abs_logic_ratio, 6),
        "abstract_has_logic_count": int(abs_logic),
    }


def _compute_type_similarity_clusters(classes: List[Dict[str, Any]], threshold: float = 0.75, max_classes_per_component: int = 600) -> Dict[str, Any]:
    """
    Cluster classes by similarity of (method_names + field_names).
    Best effort, O(n^2) per component with cap.
    """
    # union-find
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return parent.get(x, x)

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # group by component
    by_comp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in classes:
        qn = c.get("qualname")
        if not qn:
            continue
        parent.setdefault(qn, qn)
        by_comp[str(c.get("component", "."))].append(c)

    def features(c: Dict[str, Any]) -> set:
        meth = set([m for m in (c.get("method_names", []) or []) if isinstance(m, str)])
        fields = set([f for f in (c.get("field_names", []) or []) if isinstance(f, str)])
        # ignore dunders
        meth = {m for m in meth if not (m.startswith("__") and m.endswith("__"))}
        return meth | fields

    # clustering
    compared = 0
    merged = 0
    for comp, lst in by_comp.items():
        # cap
        lst = lst[: max_classes_per_component]
        fs = [(c.get("qualname"), features(c)) for c in lst if c.get("qualname")]
        for i in range(len(fs)):
            a_qn, A = fs[i]
            if not A:
                continue
            for j in range(i + 1, len(fs)):
                b_qn, B = fs[j]
                if not B:
                    continue
                compared += 1
                inter = len(A & B)
                union_sz = len(A | B)
                if union_sz <= 0:
                    continue
                sim = float(inter) / float(union_sz)
                if sim >= threshold:
                    union(a_qn, b_qn)
                    merged += 1

    clusters: Dict[str, List[str]] = defaultdict(list)
    for qn in parent.keys():
        clusters[find(qn)].append(qn)

    groups = [v for v in clusters.values() if len(v) >= 2]
    groups.sort(key=lambda g: len(g), reverse=True)

    return {
        "type_similarity_clusters": int(len(groups)),
        "threshold": float(threshold),
        "pairwise_compared": int(compared),
        "merged_pairs": int(merged),
        "top_clusters": [{"size": len(g), "classes": g[:50]} for g in groups[:20]],
    }


def _compute_usage_count(repo: Path, include_tests: bool, max_symbols: int = 2000) -> Dict[str, Any]:
    """
    Repo-level usage_count (optional heavier):
    - Collects repo-defined public top-level functions/classes (by name)
    - Counts Name/Attribute references across repo (approx)
    """
    # gather definitions
    defs: Dict[str, set] = defaultdict(set)  # symbol -> set(modules)
    py_files = [p for p in _iter_py_files(repo, include_tests)]
    for p in py_files:
        rel = p.relative_to(repo).as_posix()
        mod = rel[:-3].replace("/", ".")
        try:
            tree = ast.parse(_safe_read_text(p))
        except Exception:
            continue
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nm = getattr(n, "name", "") or ""
                if nm and not nm.startswith("_"):
                    defs[nm].add(mod)

    # cap symbols
    symbols = sorted(list(defs.keys()))[:max_symbols]
    symbol_set = set(symbols)

    # avoid counting builtins
    builtin_names = set(dir(builtins))

    counts: Counter[str] = Counter()

    class V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.in_def = 0

        def visit_FunctionDef(self, n: ast.FunctionDef) -> Any:
            self.in_def += 1
            # visit body but ignore nested def names
            for ch in n.body:
                self.visit(ch)
            self.in_def -= 1
            return None

        def visit_AsyncFunctionDef(self, n: ast.AST) -> Any:
            self.in_def += 1
            for ch in getattr(n, "body", []) or []:
                self.visit(ch)
            self.in_def -= 1
            return None

        def visit_ClassDef(self, n: ast.ClassDef) -> Any:
            self.in_def += 1
            for ch in n.body:
                self.visit(ch)
            self.in_def -= 1
            return None

        def visit_Name(self, n: ast.Name) -> None:
            if self.in_def >= 0:
                nm = n.id
                if nm in symbol_set and nm not in builtin_names:
                    counts[nm] += 1

        def visit_Attribute(self, n: ast.Attribute) -> None:
            nm = getattr(n, "attr", None)
            if isinstance(nm, str) and nm in symbol_set:
                counts[nm] += 1
            self.generic_visit(n)

    for p in py_files:
        try:
            tree = ast.parse(_safe_read_text(p))
        except Exception:
            continue
        V().visit(tree)

    # stats
    vals = [counts.get(s, 0) for s in symbols]
    used = sum(1 for v in vals if v > 0)
    zero = sum(1 for v in vals if v == 0)
    avg = float(sum(vals)) / float(len(vals)) if vals else 0.0
    top = counts.most_common(50)

    return {
        "computed": True,
        "symbols": int(len(symbols)),
        "used_symbols": int(used),
        "unused_symbols": int(zero),
        "avg_usage": round(avg, 6),
        "top": [{"symbol": k, "count": int(v), "defined_in": sorted(list(defs.get(k, set())))[:10]} for k, v in top],
    }


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
    Analyze ONLY specified files (batch-friendly).
    Cache-aware: invalidates if file changes or relevant flags change.
    """
    repo = _resolve_repo(repo_path)
    req_files_norm = [x.replace("\\\\", "/") for x in files]

    if require_radon and not RADON_AVAILABLE:
        raise RuntimeError("Radon no está disponible. Instala: pip install radon")

    cache = _load_cache(repo) if use_cache else {"meta": {}, "files": {}, "repo": {}}
    cache_files: Dict[str, Any] = cache.get("files", {})

    expected_cfg = {
        "use_radon": bool(use_radon and RADON_AVAILABLE),
        "compute_entities": bool(compute_entities),
        "compute_ruff_unused": bool(compute_ruff_unused),
        "max_file_bytes": int(max_file_bytes),
        "schema_version": "2026-01-18",  # helps invalidate old cache when fields change
    }

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

        if (not force_recompute) and entry and _cached_ok(entry, st_key) and entry.get("config") == expected_cfg:
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
            "config": expected_cfg,
        }

    out_files: Dict[str, Any] = {}
    for rel_norm in req_files_norm:
        entry = cache_files.get(rel_norm)
        if not entry:
            continue
        m = entry.get("metrics", {}) or {}
        if summary_only:
            out_files[rel_norm] = _summary_view(m)
        else:
            out_files[rel_norm] = {"metrics": m, "entities": entry.get("entities", {})}

    cache["files"] = cache_files
    cache["meta"] = {
        "repo": str(repo),
        "updated_at": _now_iso(),
        "radon_available": RADON_AVAILABLE,
        "use_radon": bool(use_radon and RADON_AVAILABLE),
        "max_file_bytes": max_file_bytes,
        "cache_path": str(_cache_path(repo)),
        "schema_version": expected_cfg["schema_version"],
    }
    if use_cache:
        _save_cache(repo, cache)

    return {"repo": str(repo), "requested": len(req_files_norm), "returned": len(out_files), "skipped": skipped, "meta": cache["meta"], "files": out_files}


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
    compute_churn: bool = True,
    churn_max_commits: int = 5000,
    compute_usage_count: bool = False,
) -> Dict[str, Any]:
    """
    Full pipeline:
      1) analyze all .py files (file + entity metrics)
      2) clone metrics
      3) import graph metrics (architecture + public_api/topic/api_overlap)
      4) git history proxies (optional; includes shotgun surgery instances)
      5) churn per-file (commits + added/deleted)
      6) advanced design metrics (always; usage_count optional)
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
            m = entry.get("metrics", {}) or {}
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

    if compute_churn:
        churn = _compute_churn_metrics_for_repo(repo, include_tests=include_tests, max_commits=int(churn_max_commits))
        churn_files = (churn or {}).get("files", {}) or {}
        updated = 0
        for rel, entry in cache_files.items():
            m = (entry or {}).get("metrics", {}) or {}
            ch = churn_files.get(rel)
            if not ch:
                continue
            m["churn_commits"] = ch.get("churn_commits")
            m["churn_added"] = ch.get("churn_added")
            m["churn_deleted"] = ch.get("churn_deleted")
            m["churn_total"] = ch.get("churn_total")
            m["churn_last_modified"] = ch.get("churn_last_modified")
            entry["metrics"] = m
            cache_files[rel] = entry
            updated += 1
        repo_out["churn"] = {
            "ok": bool(churn.get("git", False)),
            "head": churn.get("head", None),
            "commits_scanned": churn.get("commits_scanned", 0),
            "files_with_churn": int(len(churn_files)),
            "files_updated_in_cache": int(updated),
            "max_commits": int(churn_max_commits),
            "error": churn.get("error", None),
        }

    # --- Advanced metrics computed from cached entities/metrics ---
    classes, functions = _collect_cached_entities(cache_files)

    ratios = _compute_data_and_abstract_ratios(classes)
    inheritance = _compute_inheritance_metrics(classes)
    dups = _compute_dup_across_subclasses(classes, functions)
    contracts = _compute_override_contract_metrics(classes, functions)
    clusters = _compute_type_similarity_clusters(classes)

    topic_and_overlap = _compute_topic_entropy_and_api_overlap_from_entities(classes, functions)

    # repo-level public_api_size (sum of file metrics)
    repo_public_api = 0
    repo_attr_total = 0
    repo_direct_total = 0
    for _, entry in cache_files.items():
        m = (entry or {}).get("metrics", {}) or {}
        if m.get("error"):
            continue
        repo_public_api += int(m.get("public_api_size", 0) or 0)
        repo_attr_total += int(m.get("attribute_access_count", 0) or 0)
        repo_direct_total += int(m.get("direct_field_access_count", 0) or 0)

    direct_ratio = float(repo_direct_total) / float(repo_attr_total) if repo_attr_total > 0 else 0.0

    advanced: Dict[str, Any] = {
        **ratios,
        **topic_and_overlap,
        "public_api_size": int(repo_public_api),
        "direct_field_access_ratio": round(direct_ratio, 6),
        "direct_field_access_count": int(repo_direct_total),
        "attribute_access_count": int(repo_attr_total),
        **inheritance,
        **{"dup_across_subclasses": dups},
        **contracts,
        **clusters,
        # usage_count prepared (optional)
        "usage_count": {"computed": False},
    }

    if compute_usage_count:
        try:
            advanced["usage_count"] = _compute_usage_count(repo, include_tests=include_tests)
        except Exception as e:
            advanced["usage_count"] = {"computed": False, "error": f"{type(e).__name__}: {e}"}

    repo_out["advanced_metrics"] = advanced

    cache["files"] = cache_files
    cache["repo"] = {**(cache.get("repo", {}) or {}), **repo_out}
    cache["meta"] = {**(cache.get("meta", {}) or {}), "repo_updated_at": _now_iso()}

    if use_cache:
        _save_cache(repo, cache)

    artifact = _write_artifact(repo, "analysis_repo_full.json", {"repo": str(repo), "generated_at": _now_iso(), "repo_metrics": repo_out})
    return {"status": "ok", "artifact": artifact, "files": len(files), "repo_metrics_keys": sorted(list(repo_out.keys())), "batch": batch}


@mcp.tool()
def export_metrics_json(repo_path: str, include_errors: bool = True) -> Dict[str, Any]:
    """Export ALL cache to metrics_python.json (file-level summary view)."""
    repo = _resolve_repo(repo_path)
    cache = _load_cache(repo)
    files_map: Dict[str, Any] = cache.get("files", {}) or {}

    out = {}
    for path, entry in files_map.items():
        m = (entry or {}).get("metrics") or {}
        if (not include_errors) and m.get("error"):
            continue
        out[path] = _summary_view(m)

    payload = {"repo": str(repo), "generated_at": _now_iso(), "meta": cache.get("meta", {}), "repo_metrics": cache.get("repo", {}), "files": out}
    artifact = _write_artifact(repo, "metrics_python.json", payload)
    return {"status": "ok", "artifact": artifact, "files": len(out)}


@mcp.tool()
def export_entities_json(repo_path: str, include_errors: bool = False) -> Dict[str, Any]:
    """
    Export entities to entities_python.json for smell evidence:
      - functions/methods: sloc, cc, nesting, bool ops, params, signature, body_hash, ...
      - classes: wmc, lcom4, bases, data_class_candidate, abstract flags, ...
    """
    repo = _resolve_repo(repo_path)
    cache = _load_cache(repo)
    files_map: Dict[str, Any] = cache.get("files", {}) or {}

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
        out_files[path] = {"functions": fns, "classes": cls, "file_examples": ents.get("file_examples", {})}

    payload = {"repo": str(repo), "generated_at": _now_iso(), "meta": cache.get("meta", {}), "counts": {"functions": total_fn, "classes": total_cls}, "files": out_files}
    artifact = _write_artifact(repo, "entities_python.json", payload)
    return {"status": "ok", "artifact": artifact, "files": len(out_files), "functions": total_fn, "classes": total_cls}


@mcp.tool()
def rank_python_files(repo_path: str, top_k: int = 20, mi_threshold: float = 65.0, include_errors: bool = False) -> Dict[str, Any]:
    """Ranking Top K from cache + export hotspots_python.json (includes churn)."""
    repo = _resolve_repo(repo_path)
    cache = _load_cache(repo)
    files_map: Dict[str, Any] = cache.get("files", {}) or {}

    ranked: List[Dict[str, Any]] = []
    for path, entry in files_map.items():
        m = (entry or {}).get("metrics", {}) or {}
        if (not include_errors) and m.get("error"):
            continue
        mi = m.get("mi", None)
        loc = int(m.get("loc", 0) or 0)
        cc_max = float(m.get("cc_max", 0.0) or 0.0)
        churn_total = m.get("churn_total", None)
        churn_commits = m.get("churn_commits", None)

        score, reason = _score_file(mi, loc, cc_max, churn_total=churn_total, churn_commits=churn_commits, mi_threshold=mi_threshold)

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
                "clone_ratio": m.get("clone_ratio", None),
                "dup_lines": m.get("dup_lines", None),
                "dup_blocks": m.get("dup_blocks", None),
                "max_nesting_depth": m.get("max_nesting_depth", 0),
                "max_stmt_tokens": m.get("max_stmt_tokens", 0),
                "magic_number_count": m.get("magic_number_count", 0),
                "unused_symbols": m.get("unused_symbols", None),
                "public_api_size": m.get("public_api_size", 0),
                "direct_field_access_ratio": m.get("direct_field_access_ratio", 0.0),
                "churn_commits": churn_commits,
                "churn_total": churn_total,
                "churn_added": m.get("churn_added", None),
                "churn_deleted": m.get("churn_deleted", None),
                "churn_last_modified": m.get("churn_last_modified", None),
                "n_classes": m.get("n_classes", 0),
                "n_methods": m.get("n_methods", 0),
                "max_class_methods": m.get("max_class_methods", 0),
                "max_class_wmc": m.get("max_class_wmc", 0.0),
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    top = ranked[: max(0, int(top_k))]

    payload = {"repo": str(repo), "generated_at": _now_iso(), "mi_threshold": mi_threshold, "top_k": top_k, "ranking": top}
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
    Detect candidates using percentile q:
      - sloc >= p(q)
      - cc_sum >= p(q)
    """
    repo = _resolve_repo(repo_path)
    cache = _load_cache(repo)
    files_map: Dict[str, Any] = cache.get("files", {}) or {}

    rows: List[Dict[str, Any]] = []
    slocs: List[float] = []
    ccsums: List[float] = []

    for path, entry in files_map.items():
        m = (entry or {}).get("metrics", {}) or {}
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
                "clone_ratio": m.get("clone_ratio", None),
                "dup_lines": m.get("dup_lines", None),
                "max_nesting_depth": m.get("max_nesting_depth", 0),
                "unused_symbols": m.get("unused_symbols", None),
                "churn_commits": m.get("churn_commits", None),
                "churn_total": m.get("churn_total", None),
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

    payload = {"repo": str(repo), "generated_at": _now_iso(), "q": q, "p_sloc": p_sloc, "p_cc_sum": p_ccsum, "require_both": require_both, "candidates": candidates}
    artifact = _write_artifact(repo, "large_module_candidates.json", payload)
    return {"status": "ok", "artifact": artifact, "thresholds": {"p_sloc": p_sloc, "p_cc_sum": p_ccsum}, "candidates": candidates}


@mcp.tool()
def export_repo_metrics_json(repo_path: str) -> Dict[str, Any]:
    """Export repo-level metrics (clones/architecture/history/churn/advanced) to repo_metrics.json."""
    repo = _resolve_repo(repo_path)
    cache = _load_cache(repo)
    payload = {"repo": str(repo), "generated_at": _now_iso(), "meta": cache.get("meta", {}), "repo_metrics": cache.get("repo", {})}
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