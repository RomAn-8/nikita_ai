"""Target project helpers for Day 14 security loop MVP."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

TARGET_PROJECT_PATH = Path(r"D:\vetonline\guardsar_claude")
_COPY_IGNORE = shutil.ignore_patterns(".git", ".claude", "build", "screenshots", "docker-shot", "docker-web")
_BUILD_TIMEOUT = 120   # секунды
_TEST_TIMEOUT  = 90


def _npm_executable() -> str:
    """Return the correct npm executable for the current OS."""
    return "npm.cmd" if os.name == "nt" else "npm"


@dataclass
class CommandResult:
    """Result of a shell command executed inside the temp target workspace."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass
class ValidationResult:
    """Combined build/test validation result."""

    build: CommandResult
    test: CommandResult

    @property
    def ok(self) -> bool:
        return self.build.returncode == 0 and self.test.returncode == 0


def ensure_target_project_available() -> None:
    """Validate that the fixed target project exists."""
    if not TARGET_PROJECT_PATH.exists():
        raise FileNotFoundError(f"Target project not found: {TARGET_PROJECT_PATH}")
    if not (TARGET_PROJECT_PATH / "package.json").exists():
        raise FileNotFoundError(f"package.json not found in target project: {TARGET_PROJECT_PATH}")


def _link_or_copy_node_modules(source_root: Path, temp_root: Path) -> None:
    """Reuse node_modules from the source project without mutating the source tree."""
    source_node_modules = source_root / "node_modules"
    if not source_node_modules.exists():
        return

    target_node_modules = temp_root / "node_modules"
    if target_node_modules.exists():
        return

    try:
        os.symlink(source_node_modules, target_node_modules, target_is_directory=True)
        return
    except Exception:
        pass

    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target_node_modules), str(source_node_modules)],
            check=True,
            capture_output=True,
            text=True,
        )
        return
    except Exception:
        logger.warning("Failed to link node_modules, falling back to copytree")

    shutil.copytree(source_node_modules, target_node_modules)


def create_temp_copy(run_id: str) -> Path:
    """Create an isolated temporary copy of the fixed target project."""
    ensure_target_project_available()

    temp_root = Path(tempfile.mkdtemp(prefix=f"day14_{run_id}_"))
    workspace_root = temp_root / TARGET_PROJECT_PATH.name
    shutil.copytree(TARGET_PROJECT_PATH, workspace_root, ignore=_COPY_IGNORE)
    _link_or_copy_node_modules(TARGET_PROJECT_PATH, workspace_root)
    return workspace_root


def cleanup_temp_copy(workspace_root: Path) -> None:
    """Best-effort cleanup for the temporary workspace."""
    try:
        shutil.rmtree(workspace_root.parent, ignore_errors=True)
    except Exception as exc:
        logger.exception("cleanup_temp_copy failed: %s", exc)


def validate_edit_path(workspace_root: Path, relative_path: str) -> Path:
    """Resolve a relative edit path and ensure it stays inside the temp workspace."""
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError(f"Invalid edit path: {relative_path!r}")

    resolved = (workspace_root / relative_path).resolve()
    workspace_resolved = workspace_root.resolve()
    if workspace_resolved not in resolved.parents and resolved != workspace_resolved:
        raise ValueError(f"Edit path escapes workspace: {relative_path!r}")
    return resolved


def apply_edit(workspace_root: Path, edit: dict) -> None:
    """Apply one strict JSON edit to the temp workspace."""
    action = edit.get("action")
    relative_path = edit.get("path", "")
    content = edit.get("content")

    if action not in {"replace_file", "create_file"}:
        raise ValueError(f"Unsupported action: {action}")
    if not isinstance(content, str):
        raise ValueError("Edit content must be a string")

    target_path = validate_edit_path(workspace_root, relative_path)
    if action == "replace_file" and not target_path.exists():
        raise ValueError(f"replace_file requires an existing file: {relative_path}")
    if action == "create_file" and target_path.exists():
        raise ValueError(f"create_file requires a new file: {relative_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")


def apply_edits(workspace_root: Path, edits: list[dict]) -> list[str]:
    """Apply validated edits and return touched paths."""
    touched_paths: list[str] = []
    for edit in edits:
        apply_edit(workspace_root, edit)
        touched_paths.append(edit["path"])
    return touched_paths


def read_changed_files(workspace_root: Path, relative_paths: list[str]) -> list[dict[str, str]]:
    """Read final contents of touched files for security review."""
    changed: list[dict[str, str]] = []
    for relative_path in relative_paths:
        target_path = validate_edit_path(workspace_root, relative_path)
        if not target_path.exists():
            continue
        changed.append(
            {
                "path": relative_path,
                "content": target_path.read_text(encoding="utf-8"),
            }
        )
    return changed


def _run_command(
    workspace_root: Path,
    command: list[str],
    timeout: int = 120,
    extra_env: dict[str, str] | None = None,
) -> CommandResult:
    """Run one command in the workspace and capture stdout/stderr."""
    env = {**os.environ, **(extra_env or {})}
    try:
        completed = subprocess.run(
            command,
            cwd=workspace_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            command=command,
            returncode=-1,
            stdout="",
            stderr=f"Команда прервана по таймауту ({timeout}s).",
        )
    return CommandResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout[-8000:],
        stderr=completed.stderr[-8000:],
    )


def backup_files(workspace_root: Path, relative_paths: list[str]) -> dict[str, str | None]:
    """Читает текущее содержимое файлов перед изменением. None если файл не существует."""
    snapshot: dict[str, str | None] = {}
    for rel_path in relative_paths:
        try:
            target = validate_edit_path(workspace_root, rel_path)
            snapshot[rel_path] = target.read_text(encoding="utf-8") if target.exists() else None
        except Exception:
            snapshot[rel_path] = None
    return snapshot


def restore_files(workspace_root: Path, snapshot: dict[str, str | None]) -> None:
    """Восстанавливает файлы из snapshot. None → удаляет файл если он появился."""
    for rel_path, original_content in snapshot.items():
        try:
            target = validate_edit_path(workspace_root, rel_path)
            if original_content is None:
                if target.exists():
                    target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(original_content, encoding="utf-8")
        except Exception as exc:
            logger.exception("restore_files failed for %s: %s", rel_path, exc)


def run_validation(workspace_root: Path) -> ValidationResult:
    """Run the fixed MVP validation pipeline for the target project."""
    npm_exec = _npm_executable()
    build_result = _run_command(
        workspace_root,
        [npm_exec, "run", "build"],
        timeout=_BUILD_TIMEOUT,
        extra_env={"CI": "true"},
    )
    if build_result.returncode != 0:
        test_result = CommandResult(
            command=[npm_exec, "test", "--", "--watchAll=false"],
            returncode=-1,
            stdout="",
            stderr="Skipped because build failed.",
        )
        return ValidationResult(build=build_result, test=test_result)

    test_result = _run_command(
        workspace_root,
        [npm_exec, "test", "--", "--watchAll=false"],
        timeout=_TEST_TIMEOUT,
        extra_env={"CI": "true"},
    )
    return ValidationResult(build=build_result, test=test_result)
