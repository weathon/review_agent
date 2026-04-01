import re
import sys
from pathlib import Path


STATUS_PATTERNS = [
    re.compile(r"Published as a conference paper at ICLR \d{4}\s*\n?", re.IGNORECASE),
    re.compile(r"Under review as a conference paper at ICLR \d{4}\s*\n?", re.IGNORECASE),
]


def clean_text(text: str) -> tuple[str, int]:
    replacements = 0
    cleaned = text
    for pattern in STATUS_PATTERNS:
        cleaned, n = pattern.subn("", cleaned)
        replacements += n
    return cleaned, replacements


def main(target_dir: str = "iclr2025_data/papers") -> None:
    papers_dir = Path(target_dir)
    if not papers_dir.exists():
        raise FileNotFoundError(f"Directory not found: {papers_dir}")

    changed_files = 0
    total_replacements = 0

    for path in sorted(papers_dir.glob("*.txt")):
        original = path.read_text(encoding="utf-8", errors="ignore")
        cleaned, replacements = clean_text(original)
        if replacements > 0:
            path.write_text(cleaned, encoding="utf-8")
            changed_files += 1
            total_replacements += replacements

    print(f"Updated {changed_files} files")
    print(f"Removed {total_replacements} leaked status headers")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "iclr2025_data/papers"
    main(target)
