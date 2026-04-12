from pathlib import Path


def build_reviewer_query(system_prompt: str, paper_path: str, venue: str = "") -> tuple[str, str, str]:
    paper_abs = str(Path(paper_path).resolve())
    paper_dir = str(Path(paper_abs).parent)
    venue_line = (
        f"This paper was submitted to **{venue}**. "
        f"You MUST evaluate it against {venue}'s specific standards, acceptance bar, "
        f"and expectations. Consider what {venue} reviewers typically look for.\n\n"
    ) if venue else ""
    query = (
        f"{system_prompt}\n\n"
        f"---\n\n"
        f"{venue_line}"
        f"Review the following paper thoroughly.\n\n"
        f"NOTE: This paper was extracted from PDF by an automated parser. "
        f"There may be formatting artifacts such as broken equations, garbled "
        f"tables, misplaced figure references, or OCR errors. These are parser "
        f"issues, NOT problems with the paper itself. Do NOT treat formatting "
        f"artifacts as weaknesses.\n\n"
        f"The paper is located at: {paper_abs}\n"
        f"Use the read_file tool to inspect the paper file. You may also use "
        f"grep_files and glob_files if helpful. Then produce your review."
    )
    return query, paper_abs, paper_dir
