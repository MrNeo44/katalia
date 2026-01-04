#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("katalia-fs", json_response=True)

ALLOWED_ROOT: Optional[Path] = None

DEFAULT_ALLOWED_EXTS = {
    ".py", ".pyi",
    ".json", ".yml", ".yaml",
    ".md", ".txt",
    ".toml", ".ini",
    ".env",  # opcional, si prefieres bloquearlo, quítalo
}


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


def _resolve_in_repo(repo: Path, rel_path: str) -> Path:
    # normaliza slashes
    rel = rel_path.replace("\\", "/").lstrip("/")
    p = (repo / rel).resolve()
    # evita path traversal
    if p != repo and repo not in p.parents:
        raise ValueError(f"Path fuera del repo: {rel_path}")
    return p


def _check_ext(path: Path, allowed_exts: Optional[List[str]] = None, allow_any: bool = False) -> None:
    if allow_any:
        return
    exts = set(allowed_exts or list(DEFAULT_ALLOWED_EXTS))
    if path.suffix and path.suffix.lower() in exts:
        return
    # archivos sin extensión (ej: .env) -> permitir si está listado tal cual por nombre
    if path.suffix == "" and path.name in exts:
        return
    raise ValueError(f"Extensión no permitida: {path.name} ({path.suffix})")


def _safe_read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="strict")
    except Exception:
        return p.read_text(encoding="latin-1", errors="replace")


def _with_line_numbers(lines: List[str], start_line_no: int) -> List[str]:
    width = len(str(start_line_no + len(lines) - 1))
    out = []
    for i, line in enumerate(lines):
        ln = start_line_no + i
        out.append(f"{ln:>{width}} | {line}")
    return out


# -----------------------------
# Tools
# -----------------------------
@mcp.tool()
def ping() -> Dict[str, Any]:
    return {"ok": True, "ts": _now_iso()}


@mcp.tool()
def read_file(
    repo_path: str,
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    max_chars: int = 12000,
    with_line_numbers: bool = True,
    allowed_exts: Optional[List[str]] = None,
    allow_any_ext: bool = False,
) -> Dict[str, Any]:
    """
    Lee archivo completo o un rango de líneas [start_line, end_line] (1-indexed).
    """
    repo = _resolve_repo(repo_path)
    p = _resolve_in_repo(repo, file_path)
    _check_ext(p, allowed_exts=allowed_exts, allow_any=allow_any_ext)

    if not p.exists() or not p.is_file():
        return {"ok": False, "error": "file_not_found", "path": file_path}

    text = _safe_read_text(p)
    lines = text.splitlines()

    s = 1 if start_line is None else max(1, int(start_line))
    e = len(lines) if end_line is None else min(len(lines), int(end_line))
    if e < s:
        return {"ok": False, "error": "invalid_range", "path": file_path, "start_line": s, "end_line": e}

    slice_lines = lines[s - 1 : e]
    if with_line_numbers:
        view_lines = _with_line_numbers(slice_lines, s)
        view = "\n".join(view_lines)
    else:
        view = "\n".join(slice_lines)

    if max_chars > 0 and len(view) > max_chars:
        view = view[:max_chars] + "\n...<truncated>"

    return {
        "ok": True,
        "repo": str(repo),
        "path": file_path.replace("\\", "/"),
        "start_line": s,
        "end_line": e,
        "lines_total": len(lines),
        "content": view,
    }


@mcp.tool()
def get_hotspot_code(
    repo_path: str,
    file_path: str,
    center_line: int,
    context: int = 60,
    max_chars: int = 12000,
    with_line_numbers: bool = True,
) -> Dict[str, Any]:
    """
    Devuelve un snippet centrado en center_line con +/- context líneas.
    """
    center = max(1, int(center_line))
    ctx = max(0, int(context))
    start = max(1, center - ctx)
    # end lo ajusta read_file con el total de líneas
    end = center + ctx
    return read_file(
        repo_path=repo_path,
        file_path=file_path,
        start_line=start,
        end_line=end,
        max_chars=max_chars,
        with_line_numbers=with_line_numbers,
    )


@mcp.tool()
def write_file(
    repo_path: str,
    file_path: str,
    content: str,
    create_dirs: bool = True,
    overwrite: bool = True,
    max_bytes: int = 2_000_000,
    allowed_exts: Optional[List[str]] = None,
    allow_any_ext: bool = False,
) -> Dict[str, Any]:
    """
    Escribe un archivo dentro del repo. Por defecto crea directorios.
    """
    repo = _resolve_repo(repo_path)
    p = _resolve_in_repo(repo, file_path)
    _check_ext(p, allowed_exts=allowed_exts, allow_any=allow_any_ext)

    data = content.encode("utf-8", errors="ignore")
    if max_bytes > 0 and len(data) > max_bytes:
        return {"ok": False, "error": "content_too_large", "bytes": len(data), "max_bytes": max_bytes}

    if p.exists() and not overwrite:
        return {"ok": False, "error": "file_exists", "path": file_path}

    if create_dirs:
        p.parent.mkdir(parents=True, exist_ok=True)

    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": file_path.replace("\\", "/"), "bytes_written": len(data)}


@dataclass
class PatchOp:
    """
    Operaciones simples y auditables (más seguro que un unified diff genérico).
    """
    op: str                 # "replace_lines" | "replace_text"
    file: str
    # replace_lines:
    start_line: Optional[int] = None  # 1-indexed
    end_line: Optional[int] = None    # inclusive
    new_text: Optional[str] = None
    # replace_text:
    old_text: Optional[str] = None
    replace_all: bool = False


@mcp.tool()
def apply_patch_ops(
    repo_path: str,
    ops: List[Dict[str, Any]],
    dry_run: bool = False,
    allowed_exts: Optional[List[str]] = None,
    allow_any_ext: bool = False,
) -> Dict[str, Any]:
    """
    Aplica una lista de operaciones de patch.
    - replace_lines: reemplaza un rango de líneas por new_text
    - replace_text: reemplaza una ocurrencia (o todas) de old_text por new_text
    """
    repo = _resolve_repo(repo_path)
    parsed: List[PatchOp] = [PatchOp(**x) for x in ops]

    results = []
    for op in parsed:
        p = _resolve_in_repo(repo, op.file)
        _check_ext(p, allowed_exts=allowed_exts, allow_any=allow_any_ext)

        if not p.exists() or not p.is_file():
            results.append({"ok": False, "file": op.file, "error": "file_not_found"})
            continue

        original = _safe_read_text(p)

        if op.op == "replace_lines":
            if op.start_line is None or op.end_line is None or op.new_text is None:
                results.append({"ok": False, "file": op.file, "error": "missing_fields_replace_lines"})
                continue
            lines = original.splitlines()
            s = max(1, int(op.start_line))
            e = min(len(lines), int(op.end_line))
            if e < s:
                results.append({"ok": False, "file": op.file, "error": "invalid_range", "start_line": s, "end_line": e})
                continue

            new_lines = (op.new_text or "").splitlines()
            updated_lines = lines[: s - 1] + new_lines + lines[e:]
            updated = "\n".join(updated_lines) + ("\n" if original.endswith("\n") else "")
            changed = (updated != original)

        elif op.op == "replace_text":
            if op.old_text is None or op.new_text is None:
                results.append({"ok": False, "file": op.file, "error": "missing_fields_replace_text"})
                continue
            if op.old_text not in original:
                results.append({"ok": False, "file": op.file, "error": "old_text_not_found"})
                continue
            if op.replace_all:
                updated = original.replace(op.old_text, op.new_text)
            else:
                updated = original.replace(op.old_text, op.new_text, 1)
            changed = (updated != original)

        else:
            results.append({"ok": False, "file": op.file, "error": f"unknown_op:{op.op}"})
            continue

        if not dry_run and changed:
            p.write_text(updated, encoding="utf-8")

        results.append({
            "ok": True,
            "file": op.file.replace("\\", "/"),
            "changed": changed,
            "dry_run": dry_run,
            "op": op.op,
        })

    return {"ok": True, "repo": str(repo), "results": results}


# -----------------------------
# Command execution (safe-ish)
# -----------------------------
_ALLOWED_COMMANDS = [
    # python -m compileall .
    re.compile(r"^python(\.exe)?$"),
    re.compile(r"^pytest(\.exe)?$"),
    re.compile(r"^ruff(\.exe)?$"),
    re.compile(r"^mypy(\.exe)?$"),
]

# Allowed “templates”: first token must match, and we validate known safe args patterns
def _is_allowed_command(cmd: List[str]) -> Tuple[bool, str]:
    if not cmd:
        return False, "empty_command"

    exe = cmd[0].lower()

    if not any(r.match(exe) for r in _ALLOWED_COMMANDS):
        return False, f"exe_not_allowed:{exe}"

    # Allow compileall
    if exe.startswith("python"):
        # python -m compileall .
        if len(cmd) >= 4 and cmd[1:3] == ["-m", "compileall"] and cmd[3] in [".", "./"]:
            return True, "ok"
        # opcional: python -m pytest -q
        if len(cmd) >= 3 and cmd[1:3] == ["-m", "pytest"]:
            return True, "ok"
        return False, "python_args_not_allowed"

    # pytest patterns
    if exe.startswith("pytest"):
        # pytest -q  OR pytest
        return True, "ok"

    # ruff patterns
    if exe.startswith("ruff"):
        # ruff check .
        return True, "ok"

    # mypy patterns
    if exe.startswith("mypy"):
        return True, "ok"

    return False, "not_allowed"


@mcp.tool()
def run_command(
    repo_path: str,
    cmd: List[str],
    timeout_sec: int = 60,
    capture_limit: int = 12000,
) -> Dict[str, Any]:
    """
    Ejecuta comando en el repo con allowlist y timeout.
    """
    repo = _resolve_repo(repo_path)
    ok, reason = _is_allowed_command(cmd)
    if not ok:
        return {"ok": False, "error": "command_not_allowed", "reason": reason, "cmd": cmd}

    try:
        cp = subprocess.run(
            cmd,
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
            shell=False,
        )
        out = (cp.stdout or "")
        err = (cp.stderr or "")

        if capture_limit > 0:
            if len(out) > capture_limit:
                out = out[:capture_limit] + "\n...<truncated>"
            if len(err) > capture_limit:
                err = err[:capture_limit] + "\n...<truncated>"

        return {
            "ok": True,
            "repo": str(repo),
            "cmd": cmd,
            "returncode": cp.returncode,
            "stdout": out,
            "stderr": err,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "timeout_sec": timeout_sec, "cmd": cmd}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "cmd": cmd}


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
