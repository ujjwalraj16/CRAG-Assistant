"""Text normalization and cleanup utilities for ingested documents."""

import re


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HORIZONTAL_SPACE_RE = re.compile(r"[ \t]+")
EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str | None) -> str:
    """Normalize line endings, strip control characters, and collapse extra whitespace."""
    if not text:
        return ""

    # Standardize line endings first so regexes handle Windows/Mac/Unix identically.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = CONTROL_CHARS_RE.sub("", normalized)

    # Clean whitespace per line while preserving meaningful line breaks and table delimiters.
    cleaned_lines = [
        HORIZONTAL_SPACE_RE.sub(" ", line).strip()
        for line in normalized.split("\n")
    ]
    joined = "\n".join(cleaned_lines)
    joined = EXCESS_NEWLINES_RE.sub("\n\n", joined)
    return joined.strip()

