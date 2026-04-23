"""Fine-tuning pipeline service — управление датасетом и валидацией."""

import json
import logging
from pathlib import Path

from ..config import FT_DATA_PATH, FT_BASE_MODEL, FT_POLL_INTERVAL

logger = logging.getLogger(__name__)

EXPECTED_ROLES = ["system", "user", "assistant"]

_DATASET_FILES = {
    "train": "train.jsonl",
    "eval": "eval.jsonl",
    "baseline": "baseline_results.jsonl",
}


def validate_file(path: Path) -> list[dict]:
    """Валидирует JSONL-файл для fine-tuning. Возвращает список ошибок."""
    errors: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                errors.append({"line": lineno, "reason": f"Invalid JSON: {e}"})
                continue
            if "messages" not in obj:
                errors.append({"line": lineno, "reason": "Missing key 'messages'"})
                continue
            msgs = obj["messages"]
            if not isinstance(msgs, list):
                errors.append({"line": lineno, "reason": "'messages' is not a list"})
                continue
            if len(msgs) != 3:
                errors.append({"line": lineno, "reason": f"'messages' has {len(msgs)} items, expected 3"})
                continue
            roles = [m.get("role") for m in msgs]
            if roles != EXPECTED_ROLES:
                errors.append({"line": lineno, "reason": f"Wrong roles order: {roles}"})
                continue
            for m in msgs:
                if "content" not in m:
                    errors.append({"line": lineno, "reason": f"Message role='{m.get('role')}' missing 'content'"})
                    break
                if not isinstance(m["content"], str) or not m["content"].strip():
                    errors.append({"line": lineno, "reason": f"Message role='{m.get('role')}' has empty content"})
                    break
    return errors


def _count_lines(path: Path) -> int:
    """Считает непустые строки в файле."""
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def get_dataset_status() -> dict:
    """Возвращает метаданные файлов датасета: количество строк и размер."""
    result: dict[str, dict] = {}
    for key, filename in _DATASET_FILES.items():
        path = FT_DATA_PATH / filename
        if not path.exists():
            result[key] = {"filename": filename, "exists": False}
            continue
        stat = path.stat()
        lines = _count_lines(path)
        result[key] = {
            "filename": filename,
            "exists": True,
            "lines": lines,
            "size_kb": round(stat.st_size / 1024, 1),
        }
    return result


def validate_datasets() -> dict:
    """Валидирует train.jsonl и eval.jsonl. Возвращает сводку по каждому."""
    result: dict[str, dict] = {}
    for key in ("train", "eval"):
        filename = _DATASET_FILES[key]
        path = FT_DATA_PATH / filename
        if not path.exists():
            result[key] = {"filename": filename, "exists": False}
            continue
        total = _count_lines(path)
        errors = validate_file(path)
        result[key] = {
            "filename": filename,
            "exists": True,
            "total": total,
            "errors": errors,
        }
    return result


def get_baseline_summary() -> dict:
    """Краткая статистика по baseline_results.jsonl без вывода полных текстов."""
    path = FT_DATA_PATH / _DATASET_FILES["baseline"]
    if not path.exists():
        return {"exists": False, "filename": _DATASET_FILES["baseline"]}

    baseline_lens: list[int] = []
    expected_lens: list[int] = []

    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            bl = obj.get("baseline_response", "")
            ex = obj.get("expected_response", "")
            if bl:
                baseline_lens.append(len(bl))
            if ex:
                expected_lens.append(len(ex))

    count = len(baseline_lens)
    if count == 0:
        return {"exists": True, "count": 0}

    return {
        "exists": True,
        "count": count,
        "baseline_avg": round(sum(baseline_lens) / count),
        "baseline_min": min(baseline_lens),
        "baseline_max": max(baseline_lens),
        "expected_avg": round(sum(expected_lens) / count),
        "expected_min": min(expected_lens),
        "expected_max": max(expected_lens),
    }


def run_finetune_dryrun() -> dict:
    """Воспроизводит dry-run: проверяет параметры pipeline без API-вызовов."""
    train_path = FT_DATA_PATH / _DATASET_FILES["train"]
    if not train_path.exists():
        raise FileNotFoundError(f"Файл не найден: {train_path}")

    lines = _count_lines(train_path)
    size_kb = round(train_path.stat().st_size / 1024, 1)

    return {
        "file": str(train_path),
        "lines": lines,
        "size_kb": size_kb,
        "model": FT_BASE_MODEL,
        "interval": FT_POLL_INTERVAL,
    }
