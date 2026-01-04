#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP

# -----------------------------
# Radon (optional)
# -----------------------------
try:
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit
    from radon.raw import analyze as raw_analyze
    RADON_AVAILABLE = True
except Exception:
    RADON_AVAILABLE = False

mcp = FastMCP("katalia-metrics", json_response=True)

DEFAULT_IGNORES = {
    ".git", ".hg", ".svn",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "venv", ".venv", "env", ".env",
    "dist", "build", ".eggs", "site-packages",
    "node_modules",
}

ALLOWED_ROOT: Optional[Path] = None

CACHE_DIRNAME = ".katalia"
CACHE_FILENAME = "metrics_cache.json"


# -----------------------------
# Data model
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
    analyzed_at: str
    error: Optional[str] = None


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
    return {"meta": {}, "files": {}}


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
# Metrics computation
# -----------------------------
def _compute_metrics_for_code(code: str, require_radon: bool, use_radon: bool) -> Dict[str, Any]:
    if require_radon and not RADON_AVAILABLE:
        raise RuntimeError("Radon no está disponible (instala: pip install radon).")

    if (not use_radon) or (not RADON_AVAILABLE):
        lines = code.splitlines()
        loc = len(lines)
        blank = sum(1 for x in lines if not x.strip())
        comments = sum(1 for x in lines if x.strip().startswith("#"))
        sloc = loc - blank
        return {
            "loc": loc, "sloc": sloc, "comments": comments, "multi": 0, "blank": blank,
            "mi": None, "cc_sum": 0.0, "cc_avg": 0.0, "cc_max": 0.0, "cc_blocks": 0,
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
    }


def _compute_file_metrics(
    repo: Path,
    rel_path: str,
    require_radon: bool,
    use_radon: bool,
    max_file_bytes: int,
) -> FileMetrics:
    analyzed_at = _now_iso()
    rel_norm = rel_path.replace("\\", "/")
    f = (repo / rel_norm).resolve()

    try:
        if not f.exists() or f.suffix != ".py":
            return FileMetrics(
                path=rel_norm, loc=0, sloc=0, comments=0, multi=0, blank=0,
                cc_sum=0.0, cc_avg=0.0, cc_max=0.0, cc_blocks=0, mi=None,
                analyzed_at=analyzed_at, error="file_not_found_or_not_py",
            )

        if max_file_bytes > 0 and f.stat().st_size > max_file_bytes:
            return FileMetrics(
                path=rel_norm, loc=0, sloc=0, comments=0, multi=0, blank=0,
                cc_sum=0.0, cc_avg=0.0, cc_max=0.0, cc_blocks=0, mi=None,
                analyzed_at=analyzed_at, error="file_too_large",
            )

        code = _safe_read_text(f)
        m = _compute_metrics_for_code(code, require_radon=require_radon, use_radon=use_radon)

        return FileMetrics(
            path=rel_norm,
            loc=m["loc"], sloc=m["sloc"], comments=m["comments"], multi=m["multi"], blank=m["blank"],
            cc_sum=float(m["cc_sum"]), cc_avg=float(m["cc_avg"]), cc_max=float(m["cc_max"]), cc_blocks=int(m["cc_blocks"]),
            mi=(float(m["mi"]) if m["mi"] is not None else None),
            analyzed_at=analyzed_at,
            error=None,
        )
    except Exception as e:
        return FileMetrics(
            path=rel_norm, loc=0, sloc=0, comments=0, multi=0, blank=0,
            cc_sum=0.0, cc_avg=0.0, cc_max=0.0, cc_blocks=0, mi=None,
            analyzed_at=analyzed_at, error=f"{type(e).__name__}: {e}",
        )


def _summary_view(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "loc": m.get("loc", 0),
        "sloc": m.get("sloc", 0),
        "mi": m.get("mi", None),
        "cc_max": m.get("cc_max", 0),
        "cc_blocks": m.get("cc_blocks", 0),
        "error": m.get("error", None),
    }


# -----------------------------
# Ranking logic
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
    return {"ok": True, "ts": _now_iso(), "radon_available": RADON_AVAILABLE}


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
        "meta": cache.get("meta", {}),
        "radon_available": RADON_AVAILABLE,
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
) -> Dict[str, Any]:
    """
    Analiza SOLO los archivos indicados (batch-friendly). Sin Git.
    """
    repo = _resolve_repo(repo_path)
    req_files_norm = [x.replace("\\", "/") for x in files]

    if require_radon and not RADON_AVAILABLE:
        raise RuntimeError("Radon no está disponible. Instala: pip install radon")

    cache = _load_cache(repo) if use_cache else {"meta": {}, "files": {}}
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

    computed: List[FileMetrics] = []
    if tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(_compute_file_metrics, repo, rel, require_radon, use_radon, max_file_bytes): rel
                for rel in tasks
            }
            for fut in as_completed(futs):
                computed.append(fut.result())

    for fm in computed:
        rel_norm = fm.path
        f = (repo / rel_norm)
        cache_files[rel_norm] = {"metrics": asdict(fm), "stat": _file_stat_key(f)}

    out_files: Dict[str, Any] = {}
    for rel_norm in req_files_norm:
        entry = cache_files.get(rel_norm)
        if not entry:
            continue
        m = entry.get("metrics", {})
        out_files[rel_norm] = _summary_view(m) if summary_only else m

    cache["files"] = cache_files
    cache["meta"] = {
        "repo": str(repo),
        "updated_at": _now_iso(),
        "radon_available": RADON_AVAILABLE,
        "use_radon": use_radon and RADON_AVAILABLE,
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
def export_metrics_json(
    repo_path: str,
    include_errors: bool = True,
) -> Dict[str, Any]:
    """
    Exporta TODO lo que hay en cache a metrics_python.json
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
        "files": out,
    }
    artifact = _write_artifact(repo, "metrics_python.json", payload)
    return {"status": "ok", "artifact": artifact, "files": len(out)}


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
        ranked.append({
            "path": path,
            "score": score,
            "reason": reason,
            "cc_max": m.get("cc_max", 0),
            "mi": m.get("mi", None),
            "loc": m.get("loc", 0),
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    top = ranked[:max(0, int(top_k))]

    payload = {
        "repo": str(repo),
        "generated_at": _now_iso(),
        "mi_threshold": mi_threshold,
        "top_k": top_k,
        "ranking": top,
    }
    artifact = _write_artifact(repo, "hotspots_python.json", payload)
    return {"status": "ok", "artifact": artifact, "top": top, "ranked_total": len(ranked)}


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
