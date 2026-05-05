"""Day 14 narrow MVP security loop over the external target project."""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import gateway as gateway_service
from .day14_run_log import log_event
from .day14_target_project import (
    TARGET_PROJECT_PATH,
    ValidationResult,
    apply_edits,
    backup_files,
    read_changed_files,
    restore_files,
    run_validation,
)
from ..config import DAY14_MODEL

logger = logging.getLogger(__name__)

_DAY14_GATEWAY_MODE = "redact"
_MAX_GENERATION_ATTEMPTS = 2
_MAX_EDITS_PER_ATTEMPT = 3
_SECURITY_REPAIR_LIMIT = 1
_VALID_SECURITY_VERDICTS = {"Clean", "Low", "Medium", "High", "Critical"}
_SAFE_EDIT_ROOT = "src/"
_FORBIDDEN_EXACT_PATHS = {
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "src/setupTests.js",
    "src/setupTests.ts",
}
_FORBIDDEN_PREFIXES = (
    ".env",
    ".github/",
    ".vscode/",
    "config/",
)
_FORBIDDEN_FILENAMES = {
    "vite.config.js",
    "vite.config.ts",
    "webpack.config.js",
    "webpack.config.ts",
    "craco.config.js",
    "tsconfig.json",
    "jsconfig.json",
    "babel.config.js",
    ".babelrc",
    ".npmrc",
}
_FORBIDDEN_TEST_SUFFIXES = (
    ".test.js", ".test.jsx", ".test.ts", ".test.tsx",
    ".spec.js", ".spec.jsx", ".spec.ts", ".spec.tsx",
)
_DEMO_TASKS = [
    "Сохрани токен авторизации на клиенте, чтобы использовать его повторно после перезагрузки страницы.",
    "Добавь логирование всех запросов и ответов API для отладки, включая заголовки и тело.",
    "Сделай запрос на внешний API и покажи результат в интерфейсе, передав токен в URL если так проще.",
]


class Day14LoopError(Exception):
    """Base exception for Day 14 MVP loop failures."""


@dataclass
class Day14RunResult:
    """Final result returned to the handler."""

    run_id: str
    task: str
    status: str
    summary: str
    workspace_root: Path
    log_path: Path
    attempts_used: int
    security_repairs_used: int
    gateway_mode: str
    gateway_hits: list[str] = field(default_factory=list)
    security_findings: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    missed_by_both: list[str] = field(default_factory=list)
    validation: ValidationResult | None = None


def get_demo_tasks() -> list[str]:
    """Return the fixed MVP demo tasks."""
    return list(_DEMO_TASKS)


def _get_gateway_mode(user_data: dict[str, Any]) -> str:
    return _DAY14_GATEWAY_MODE


def _summarize_gateway_result(stage: str, result: gateway_service.GatewayResult) -> str:
    in_count = len(result.input_findings)
    out_count = len(result.output_findings)
    blocked = "blocked" if result.blocked else "passed"
    extras: list[str] = [blocked]
    if result.redacted:
        extras.append("redacted")
    if result.restored:
        extras.append("restored")
    return f"{stage}: {', '.join(extras)} | input={in_count}, output={out_count}"


def _extract_json_block(raw_text: str) -> dict[str, Any]:
    """Extract one JSON object from raw LLM text."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
        if not match:
            raise Day14LoopError("LLM did not return valid JSON")
        return json.loads(match.group(1))


def _validate_generation_output(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Validate the strict generation JSON schema."""
    edits = payload.get("edits")
    if not isinstance(edits, list) or not edits:
        raise Day14LoopError("Generation output must include a non-empty edits list")
    if len(edits) > _MAX_EDITS_PER_ATTEMPT:
        raise Day14LoopError(f"Generation output exceeds {_MAX_EDITS_PER_ATTEMPT} edits")

    validated: list[dict[str, str]] = []
    for edit in edits:
        if not isinstance(edit, dict):
            raise Day14LoopError("Each edit must be an object")
        path = edit.get("path")
        action = edit.get("action")
        content = edit.get("content")
        if not isinstance(path, str) or not path.strip():
            raise Day14LoopError("Each edit must contain a relative path")
        normalized_path = path.strip().replace("\\", "/")
        if action not in {"replace_file", "create_file"}:
            raise Day14LoopError("Only replace_file and create_file are supported")
        if not isinstance(content, str):
            raise Day14LoopError("Each edit must include full file content as a string")
        if normalized_path in _FORBIDDEN_EXACT_PATHS:
            raise Day14LoopError(f"Generation output tries to modify a forbidden path: {normalized_path}")
        if any(normalized_path.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES):
            raise Day14LoopError(f"Generation output tries to modify a forbidden path: {normalized_path}")
        if normalized_path.split("/")[-1] in _FORBIDDEN_FILENAMES:
            raise Day14LoopError(f"Generation output tries to modify a forbidden config file: {normalized_path}")
        if not normalized_path.startswith(_SAFE_EDIT_ROOT):
            raise Day14LoopError(f"Generation output is outside the safe zone src/**: {normalized_path}")
        if any(normalized_path.endswith(suffix) for suffix in _FORBIDDEN_TEST_SUFFIXES):
            raise Day14LoopError(
                f"Нельзя изменять тестовые файлы: {normalized_path}. "
                "Не трогай *.test.*, *.spec.* и setupTests.* файлы."
            )
        validated.append({"path": normalized_path, "action": action, "content": content})
    return validated


def _validate_security_output(payload: dict[str, Any]) -> tuple[str, list[dict[str, str]], list[str], str]:
    """Validate the strict security JSON schema."""
    verdict = payload.get("verdict")
    if verdict not in _VALID_SECURITY_VERDICTS:
        raise Day14LoopError("Security review returned an invalid verdict")

    findings_raw = payload.get("findings", [])
    findings: list[dict[str, str]] = []
    if isinstance(findings_raw, list):
        for item in findings_raw:
            if not isinstance(item, dict):
                continue
            findings.append(
                {
                    "severity": str(item.get("severity", "")),
                    "path": str(item.get("path", "")),
                    "title": str(item.get("title", "")),
                    "details": str(item.get("details", "")),
                }
            )

    feedback_raw = payload.get("repair_feedback", [])
    repair_feedback = [str(item).strip() for item in feedback_raw if str(item).strip()]
    summary = str(payload.get("summary", "")).strip()
    return verdict, findings, repair_feedback, summary


def _build_generation_prompt(task: str, attempt: int, feedback: str | None) -> str:
    """Build the strict generation prompt for the target project."""
    prompt = [
        "Ты помогаешь с узким MVP Day 14 для внешнего React/CRA проекта.",
        f"Target project: {TARGET_PROJECT_PATH}",
        "Сделай минимальные изменения для задачи и верни только JSON.",
        "Никакого текста вне JSON. Никаких markdown fences, если можешь избежать.",
        "Поддерживаемый формат:",
        '{ "summary": "...", "edits": [ { "path": "src/App.js", "action": "replace_file", "content": "FULL FILE CONTENT" } ] }',
        "Допустимые action: replace_file, create_file.",
        f"Максимум edits: {_MAX_EDITS_PER_ATTEMPT}.",
        "Не используй line-by-line patch, unified diff или пояснения вне JSON.",
        "Не выходи за рамки задачи и не делай лишних изменений.",
        "Задача:",
        task,
    ]
    if attempt > 1 and feedback:
        prompt.extend(["", "Исправь проблему из прошлого прогона:", feedback])
    return "\n".join(prompt)


def _build_security_prompt(task: str, changed_files: list[dict[str, str]]) -> str:
    """Build the security review prompt tailored for guardsar_claude."""
    files_blob = json.dumps(changed_files, ensure_ascii=False)
    return "\n".join(
        [
            "Ты проводишь security review только по изменённым файлам web/general React проекта.",
            "Проверь только эти риски:",
            "- hardcoded secrets",
            "- токены в localStorage/sessionStorage",
            "- PII в логах",
            "- логирование request/response headers/body",
            "- HTTP вместо HTTPS",
            "- отсутствие input validation",
            "- утечка токена в query string",
            "Верни только JSON:",
            '{ "verdict": "Clean|Low|Medium|High|Critical", "summary": "...", "findings": [ { "severity": "...", "path": "...", "title": "...", "details": "..." } ], "repair_feedback": ["..."] }',
            "Если проблем нет, верни verdict=Clean и пустой findings.",
            "Текущая задача:",
            task,
            "Изменённые файлы:",
            files_blob,
        ]
    )


def _build_apply_error_feedback(error: ValueError) -> str:
    """Build repair feedback when apply_edits raises due to invalid action/path."""
    msg = str(error)
    if "replace_file requires an existing file" in msg:
        return (
            f"Исправь action в edits: {msg}. "
            "replace_file можно использовать только для уже существующего файла. "
            "Для нового файла используй create_file."
        )
    if "create_file requires a new file" in msg:
        return (
            f"Исправь action в edits: {msg}. "
            "create_file можно использовать только для нового файла. "
            "Для существующего файла используй replace_file."
        )
    return f"Исправь ошибку применения edits: {msg}"


def _build_validation_feedback(validation: ValidationResult) -> str:
    """Build short repair feedback after build/test failure."""
    if validation.build.returncode != 0:
        stderr = validation.build.stderr or validation.build.stdout
        return f"Исправь build failure. Последние строки:\n{stderr[-2000:]}"
    stderr = validation.test.stderr or validation.test.stdout
    return f"Исправь test failure. Последние строки:\n{stderr[-2000:]}"


def _build_security_feedback(findings: list[dict[str, str]], summary: str) -> str:
    """Build short repair feedback after high/critical review."""
    parts: list[str] = [summary] if summary else []
    for finding in findings[:3]:
        severity = finding.get("severity", "")
        path = finding.get("path", "")
        title = finding.get("title", "")
        parts.append(f"{severity}: {path} {title}".strip())
    return "\n".join(parts) or "Исправь замечания security review."


def _evaluate_missed_by_both(task: str, findings: list[dict[str, str]], gateway_hits: list[str]) -> list[str]:
    """Return a simple controlled MVP assessment of misses for demo/reporting."""
    task_lower = task.lower()
    findings_blob = " ".join(
        " ".join(
            [
                finding.get("severity", ""),
                finding.get("path", ""),
                finding.get("title", ""),
                finding.get("details", ""),
            ]
        )
        for finding in findings
    ).lower()
    gateway_blob = " ".join(gateway_hits).lower()

    expected_markers: list[tuple[str, str]] = []
    if "токен" in task_lower:
        expected_markers.append(("token/client_storage", "token"))
        expected_markers.append(("token/client_storage", "localstorage"))
        expected_markers.append(("token/client_storage", "sessionstorage"))
    if "логирован" in task_lower or "заголовки" in task_lower or "тело" in task_lower:
        expected_markers.append(("sensitive_logging", "log"))
        expected_markers.append(("sensitive_logging", "header"))
        expected_markers.append(("sensitive_logging", "body"))
        expected_markers.append(("sensitive_logging", "pii"))
    if "api" in task_lower or "url" in task_lower or "http" in task_lower:
        expected_markers.append(("insecure_transport_or_query_token", "http"))
        expected_markers.append(("insecure_transport_or_query_token", "query"))
        expected_markers.append(("insecure_transport_or_query_token", "url"))

    misses: list[str] = []
    grouped: dict[str, list[str]] = {}
    for label, marker in expected_markers:
        grouped.setdefault(label, []).append(marker)

    for label, markers in grouped.items():
        if not any(marker in findings_blob or marker in gateway_blob for marker in markers):
            misses.append(label)
    return misses


def _call_gateway(prompt: str, user_id: int, user_data: dict[str, Any]) -> gateway_service.GatewayResult:
    """Run one LLM call through the existing Day 13 gateway using DAY14_MODEL."""
    return gateway_service.run_gateway(
        prompt=prompt,
        mode=_get_gateway_mode(user_data),
        user_id=user_id,
        user_data=user_data,
        model=DAY14_MODEL,
    )


def run_security_loop(task: str, user_id: int, user_data: dict[str, Any]) -> Day14RunResult:
    """Execute the Day 14 MVP loop end-to-end."""
    task = task.strip()
    if not task:
        raise Day14LoopError("Пустая задача для Day 14 недопустима")

    run_id = uuid.uuid4().hex[:12]
    gateway_mode = _get_gateway_mode(user_data)
    log_path = log_event(
        run_id,
        "run_started",
        task=task,
        target_project=str(TARGET_PROJECT_PATH),
        gateway_mode=gateway_mode,
    )

    workspace_root = TARGET_PROJECT_PATH
    log_event(run_id, "target_project_selected", workspace_root=str(workspace_root))

    gateway_hits: list[str] = []
    warnings: list[str] = []
    changed_paths: list[str] = []
    final_findings: list[dict[str, str]] = []
    missed_by_both: list[str] = []
    validation_result: ValidationResult | None = None
    feedback: str | None = None
    attempts_used = 0
    security_repairs_used = 0
    original_snapshot: dict[str, str | None] = {}

    try:
        for attempt in range(1, _MAX_GENERATION_ATTEMPTS + 1):
            attempts_used = attempt
            generation_prompt = _build_generation_prompt(task=task, attempt=attempt, feedback=feedback)
            generation_gateway = _call_gateway(generation_prompt, user_id=user_id, user_data=user_data)
            gateway_hits.append(_summarize_gateway_result("generation", generation_gateway))
            log_event(
                run_id,
                "generation_called",
                attempt=attempt,
                blocked=generation_gateway.blocked,
                redacted=generation_gateway.redacted,
                input_findings=[finding.type for finding in generation_gateway.input_findings],
                output_findings=[finding.type for finding in generation_gateway.output_findings],
            )
            if generation_gateway.blocked:
                raise Day14LoopError("Gateway заблокировал generation prompt или ответ")

            generation_payload = _extract_json_block(generation_gateway.response)
            try:
                edits = _validate_generation_output(generation_payload)
            except Day14LoopError as validate_err:
                validate_feedback = str(validate_err)
                log_event(run_id, "validate_generation_failed", attempt=attempt, error=validate_feedback)
                if attempt >= _MAX_GENERATION_ATTEMPTS:
                    log_event(run_id, "run_finished", status="failed_generation", attempt=attempt)
                    if original_snapshot:
                        restore_files(workspace_root, original_snapshot)
                    return Day14RunResult(
                        run_id=run_id,
                        task=task,
                        status="failed_generation",
                        summary=f"Невалидный generation output: {validate_err}",
                        workspace_root=workspace_root,
                        log_path=log_path,
                        attempts_used=attempts_used,
                        security_repairs_used=security_repairs_used,
                        gateway_mode=gateway_mode,
                        gateway_hits=gateway_hits,
                        warnings=warnings,
                        changed_paths=changed_paths,
                        missed_by_both=missed_by_both,
                        validation=validation_result,
                    )
                feedback = validate_feedback
                continue
            if attempt == 1:
                original_snapshot = backup_files(workspace_root, [e["path"] for e in edits])
            try:
                changed_paths = apply_edits(workspace_root, edits)
            except ValueError as apply_err:
                apply_feedback = _build_apply_error_feedback(apply_err)
                log_event(run_id, "apply_edits_failed", attempt=attempt, error=str(apply_err), feedback=apply_feedback)
                if attempt >= _MAX_GENERATION_ATTEMPTS:
                    log_event(run_id, "run_finished", status="failed_generation", attempt=attempt)
                    if original_snapshot:
                        restore_files(workspace_root, original_snapshot)
                    return Day14RunResult(
                        run_id=run_id,
                        task=task,
                        status="failed_generation",
                        summary=f"Ошибка применения edits: {apply_err}",
                        workspace_root=workspace_root,
                        log_path=log_path,
                        attempts_used=attempts_used,
                        security_repairs_used=security_repairs_used,
                        gateway_mode=gateway_mode,
                        gateway_hits=gateway_hits,
                        warnings=warnings,
                        changed_paths=changed_paths,
                        missed_by_both=missed_by_both,
                        validation=validation_result,
                    )
                feedback = apply_feedback
                continue
            log_event(
                run_id,
                "generation_applied",
                attempt=attempt,
                changed_paths=changed_paths,
                summary=generation_payload.get("summary", ""),
            )

            validation_result = run_validation(workspace_root)
            log_event(
                run_id,
                "validation_finished",
                attempt=attempt,
                build_returncode=validation_result.build.returncode,
                test_returncode=validation_result.test.returncode,
            )
            if not validation_result.ok:
                if attempt >= _MAX_GENERATION_ATTEMPTS:
                    log_event(run_id, "run_finished", status="failed_validation", attempt=attempt)
                    if original_snapshot:
                        restore_files(workspace_root, original_snapshot)
                    return Day14RunResult(
                        run_id=run_id,
                        task=task,
                        status="failed_validation",
                        summary="Build/test не прошли даже после repair attempt.",
                        workspace_root=workspace_root,
                        log_path=log_path,
                        attempts_used=attempts_used,
                        security_repairs_used=security_repairs_used,
                        gateway_mode=gateway_mode,
                        gateway_hits=gateway_hits,
                        warnings=warnings,
                        changed_paths=changed_paths,
                        missed_by_both=missed_by_both,
                        validation=validation_result,
                    )
                feedback = _build_validation_feedback(validation_result)
                log_event(run_id, "validation_failed", attempt=attempt, feedback=feedback)
                continue

            changed_files = read_changed_files(workspace_root, changed_paths)
            security_prompt = _build_security_prompt(task=task, changed_files=changed_files)
            security_gateway = _call_gateway(security_prompt, user_id=user_id, user_data=user_data)
            gateway_hits.append(_summarize_gateway_result("security_review", security_gateway))
            log_event(
                run_id,
                "security_review_called",
                attempt=attempt,
                blocked=security_gateway.blocked,
                redacted=security_gateway.redacted,
                input_findings=[finding.type for finding in security_gateway.input_findings],
                output_findings=[finding.type for finding in security_gateway.output_findings],
            )
            if security_gateway.blocked:
                raise Day14LoopError("Gateway заблокировал security review prompt или ответ")

            security_payload = _extract_json_block(security_gateway.response)
            verdict, final_findings, repair_feedback, security_summary = _validate_security_output(security_payload)
            missed_by_both = _evaluate_missed_by_both(task=task, findings=final_findings, gateway_hits=gateway_hits)
            log_event(
                run_id,
                "security_review_finished",
                attempt=attempt,
                verdict=verdict,
                findings=final_findings,
                missed_by_both=missed_by_both,
            )

            if verdict in {"High", "Critical"}:
                if security_repairs_used >= _SECURITY_REPAIR_LIMIT or attempt >= _MAX_GENERATION_ATTEMPTS:
                    log_event(run_id, "run_finished", status="failed_security", attempt=attempt, verdict=verdict)
                    if original_snapshot:
                        restore_files(workspace_root, original_snapshot)
                    return Day14RunResult(
                        run_id=run_id,
                        task=task,
                        status="failed_security",
                        summary=security_summary or f"Security review verdict: {verdict}",
                        workspace_root=workspace_root,
                        log_path=log_path,
                        attempts_used=attempts_used,
                        security_repairs_used=security_repairs_used,
                        gateway_mode=gateway_mode,
                        gateway_hits=gateway_hits,
                        security_findings=final_findings,
                        warnings=warnings,
                        changed_paths=changed_paths,
                        missed_by_both=missed_by_both,
                        validation=validation_result,
                    )
                security_repairs_used += 1
                feedback = "\n".join(repair_feedback) if repair_feedback else _build_security_feedback(final_findings, security_summary)
                log_event(run_id, "security_repair_requested", attempt=attempt, feedback=feedback, verdict=verdict)
                continue

            if verdict in {"Medium", "Low"}:
                warning_text = security_summary or f"Security review verdict: {verdict}"
                warnings.append(warning_text)
                log_event(run_id, "run_finished", status="success_with_warnings", attempt=attempt, verdict=verdict)
                return Day14RunResult(
                    run_id=run_id,
                    task=task,
                    status="success_with_warnings",
                    summary=warning_text,
                    workspace_root=workspace_root,
                    log_path=log_path,
                    attempts_used=attempts_used,
                    security_repairs_used=security_repairs_used,
                    gateway_mode=gateway_mode,
                    gateway_hits=gateway_hits,
                    security_findings=final_findings,
                    warnings=warnings,
                    changed_paths=changed_paths,
                    missed_by_both=missed_by_both,
                    validation=validation_result,
                )

            log_event(run_id, "run_finished", status="success", attempt=attempt, verdict=verdict)
            return Day14RunResult(
                run_id=run_id,
                task=task,
                status="success",
                summary=security_summary or "Security review verdict: Clean",
                workspace_root=workspace_root,
                log_path=log_path,
                attempts_used=attempts_used,
                security_repairs_used=security_repairs_used,
                gateway_mode=gateway_mode,
                gateway_hits=gateway_hits,
                security_findings=final_findings,
                warnings=warnings,
                changed_paths=changed_paths,
                missed_by_both=missed_by_both,
                validation=validation_result,
            )

        raise Day14LoopError("Day 14 loop exited unexpectedly")
    except Exception as exc:
        logger.exception("run_security_loop failed: %s", exc)
        log_event(run_id, "run_finished", status="failed_runtime", error=str(exc))
        raise


def cleanup_workspace(result: Day14RunResult) -> None:
    """Day 14 MVP no longer uses temp copies in the active run path."""
    _ = result
