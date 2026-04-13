import glob as glob_module
import os
import re


def _resolve_allowed_paths(allowed_paths: list[str]) -> list[str]:
    return [os.path.abspath(path) for path in allowed_paths]


def _check_allowed_path(path: str, allowed_paths: list[str]) -> str | None:
    resolved = os.path.abspath(path)
    resolved_allowed = _resolve_allowed_paths(allowed_paths)
    if any(resolved.startswith(allowed_path) for allowed_path in resolved_allowed):
        return None
    return (
        f"ERROR: Access denied. Path '{resolved}' is not under any allowed directory: "
        f"{resolved_allowed}"
    )


def read_file_text(abs_path: str, start_line: int = 1, end_line: int = 0, allowed_paths: list[str] | None = None) -> str:
    if allowed_paths is None:
        raise ValueError("allowed_paths is required")
    error = _check_allowed_path(abs_path, allowed_paths)
    if error:
        return error
    try:
        with open(abs_path, "r", errors="replace") as file_handle:
            lines = file_handle.readlines()
    except FileNotFoundError:
        return f"ERROR: File not found: {abs_path}"
    selected = lines[max(0, start_line - 1):end_line if end_line > 0 else len(lines)]
    return "".join(f"{start_line + index}: {line}" for index, line in enumerate(selected))


def grep_files_text(
    pattern: str,
    directory: str = ".",
    file_glob: str = "**/*",
    allowed_paths: list[str] | None = None,
) -> str:
    if allowed_paths is None:
        raise ValueError("allowed_paths is required")
    error = _check_allowed_path(directory, allowed_paths)
    if error:
        return error
    matches: list[str] = []
    files = sorted(glob_module.glob(file_glob, root_dir=directory, recursive=True))
    for relative_path in files[:500]:
        file_path = os.path.join(directory, relative_path)
        if not os.path.isfile(file_path):
            continue
        try:
            with open(file_path, "r", errors="replace") as file_handle:
                for line_number, line in enumerate(file_handle, 1):
                    if re.search(pattern, line):
                        matches.append(f"{file_path}:{line_number}: {line.rstrip()}")
        except Exception:
            continue
        if len(matches) >= 50:
            break
    if not matches:
        return "No matches found."
    return "\n".join(matches)


def glob_files_text(pattern: str, directory: str = ".", allowed_paths: list[str] | None = None) -> str:
    if allowed_paths is None:
        raise ValueError("allowed_paths is required")
    error = _check_allowed_path(directory, allowed_paths)
    if error:
        return error
    matches = sorted(glob_module.glob(pattern, root_dir=directory, recursive=True))
    if not matches:
        return "No files matched."
    return "\n".join(os.path.join(directory, match) for match in matches)
