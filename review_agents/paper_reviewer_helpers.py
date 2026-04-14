import re
from pathlib import Path


LEAKAGE_WARNING_PATTERNS = [
    r"\bsame paper\b",
    r"\bexact same paper\b",
    r"\bthis exact paper\b",
    r"\bcontains this exact paper\b",
    r"\bthe exact same paper\b",
    r"\bcalibration copy\b",
]

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def score_to_decision(score: float | None) -> str | None:
    if score is None:
        return None
    return "Accept" if float(score) >= 5.0 else "Reject"


def decision_match(predicted: str | None, gt_binary: str) -> bool | None:
    if predicted in (None, "", "N/A"):
        return None
    return predicted == gt_binary


def match_label(match: bool | None) -> str:
    if match is None:
        return "N/A"
    return "YES" if match else "NO"


def detect_leakage_warning_phrases(text: str) -> list[str]:
    matches: list[str] = []
    for pattern in LEAKAGE_WARNING_PATTERNS:
        found = re.search(pattern, text, flags=re.IGNORECASE)
        if found:
            matches.append(found.group(0))
    return matches


def load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


HARSH_CRITIC_PROMPT = load_prompt("harsh_critic.txt")
NEUTRAL_REVIEWER_PROMPT = load_prompt("neutral_reviewer.txt")
SPARK_FINDER_PROMPT = load_prompt("spark_finder.txt")
RELATED_WORK_PROMPT = load_prompt("related_work.txt")
RELATED_WORK_FILTER_PROMPT = load_prompt("related_work_filter.txt")
_MERGER_PROMPT_TEMPLATE = load_prompt("merger.txt")
SCORE_PROMPT = load_prompt("scorer.txt")


def build_merger_prompt(
    skip_neutral: bool = False,
    skip_spark: bool = False,
    skip_related_work: bool = False,
) -> str:
    num = 1
    neutral_line = ""
    spark_line = ""
    related_work_line = ""
    if not skip_neutral:
        num += 1
        neutral_line = f"{num}. A **neutral/balanced** review\n"
    if not skip_spark:
        num += 1
        spark_line = f"{num}. A **spark finder** report (focuses on insights, not flaws)\n"
    if not skip_related_work:
        num += 1
        related_work_line = (
            f"{num}. A **potentially missed related work** report (these are SUGGESTIONS, not "
            f"definitive omissions — the authors may have good reasons for not citing them)\n"
        )
    return _MERGER_PROMPT_TEMPLATE.format(
        input_count=num,
        neutral_line=neutral_line,
        spark_line=spark_line,
        related_work_line=related_work_line,
    )


MERGER_PROMPT = build_merger_prompt()


def sanitize_text(text: str) -> str:
    return text.replace("\x00", "")
