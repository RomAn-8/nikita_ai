"""Text processing utilities."""

TELEGRAM_MESSAGE_LIMIT = 4096


def split_telegram_text(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """
    Split text into chunks that fit Telegram message limit.
    
    Args:
        text: Text to split
        limit: Maximum length per chunk
        
    Returns:
        List of text chunks
    """
    t = (text or "").strip()
    if not t:
        return [""]

    if len(t) <= limit:
        return [t]

    parts: list[str] = []
    cur = t
    while len(cur) > limit:
        cut = cur.rfind("\n", 0, limit)
        if cut < 200:
            cut = limit
        parts.append(cur[:cut].rstrip())
        cur = cur[cut:].lstrip()
    if cur:
        parts.append(cur)
    return parts


def looks_like_json(text: str) -> bool:
    """Check if text looks like JSON."""
    t = (text or "").lstrip()
    return (t.startswith("{") and t.endswith("}")) or t.startswith("{")


def is_forest_final(text: str) -> bool:
    """Check if text starts with FINAL marker."""
    t = (text or "").lstrip()
    return t.upper().startswith("FINAL")


def strip_forest_final_marker(text: str) -> str:
    """Remove FINAL marker from text."""
    lines = (text or "").splitlines()
    if not lines:
        return ""
    if lines[0].strip().upper() == "FINAL":
        return "\n".join(lines[1:]).strip()
    return (text or "").strip()


def _short_model_name(m: str) -> str:
    """Get short model name from full model path."""
    m = (m or "").strip()
    if not m:
        return "default"
    return m.split("/")[-1]
