#!/usr/bin/env python3
"""Build the merger system prompt with dynamic sections.

Usage:
  python agents/build_merger_prompt.py --output /tmp/work/merger_system.txt \
      [--skip-neutral] [--skip-spark] [--skip-related-work]
"""
from pathlib import Path


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--skip-neutral", action="store_true")
    p.add_argument("--skip-spark", action="store_true")
    p.add_argument("--skip-related-work", action="store_true")
    args = p.parse_args()

    template = (Path(__file__).parent.parent / "prompts" / "merger.txt").read_text(encoding="utf-8")

    num = 1
    neutral_line = ""
    spark_line = ""
    related_work_line = ""
    if not args.skip_neutral:
        num += 1
        neutral_line = f"{num}. A **neutral/balanced** review\n"
    if not args.skip_spark:
        num += 1
        spark_line = f"{num}. A **spark finder** report (focuses on insights, not flaws)\n"
    if not args.skip_related_work:
        num += 1
        related_work_line = (
            f"{num}. A **potentially missed related work** report (these are SUGGESTIONS, not "
            f"definitive omissions — the authors may have good reasons for not citing them)\n"
        )

    result = template.format(
        input_count=num,
        neutral_line=neutral_line,
        spark_line=spark_line,
        related_work_line=related_work_line,
    )
    Path(args.output).write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
