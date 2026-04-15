"""
Claude Agent SDK merger for main.py.
Used when MERGER_MODEL starts with 'claude_sdk:'.
"""
from __future__ import annotations

import fcntl
import os
import numpy as np
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from openai import OpenAI
import dotenv
dotenv.load_dotenv()

HUMAN_REVIEW_DIR = os.path.abspath("../human_reviews/")

# ── Build indexes (mirrors tools.py) ──────────────────────────────────
_bm25_db: dict = {}
_review_db: dict = {}   # {"filenames": [...], "vectors": np.ndarray, "dir": str}
_or_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

def _ensure_indexes():
    global _bm25_db
    if _bm25_db:
        return
    all_files = []
    all_file_paths = []
    for root, _, files in os.walk(HUMAN_REVIEW_DIR):
        for fname in files:
            if fname.endswith(".txt") or fname.endswith(".md"):
                fpath = os.path.join(root, fname)
                with open(fpath, "r", errors="replace") as fh:
                    content = fh.read()
                if content.strip():
                    all_files.append(content)
                    all_file_paths.append(fpath)
    tokenized = [doc.split(" ") for doc in all_files]
    _bm25_db["bm25"] = BM25Okapi(tokenized)
    _bm25_db["files"] = all_file_paths

    with open("./human_reviews_embeddings.pkl", "rb") as f:
        db = pickle.load(f)
    _bm25_db["filenames"] = list(db.keys())
    _bm25_db["vectors"] = np.array(list(db.values()))


def _load_review_db(reviews_dir: str):
    """Load the per-run self-consistency embedding DB from reviews_dir."""
    global _review_db
    pkl_path = os.path.join(reviews_dir, "embeddings.pkl")
    _review_db["dir"] = reviews_dir
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            db = pickle.load(f)
        if db:
            _review_db["filenames"] = list(db.keys())
            _review_db["vectors"] = np.array(list(db.values()))
        else:
            _review_db["filenames"] = []
            _review_db["vectors"] = np.zeros((0, 0))
    else:
        _review_db["filenames"] = []
        _review_db["vectors"] = np.zeros((0, 0))


async def _embed(text: str) -> np.ndarray:
    """Embed text using Gemini via OpenRouter."""
    response = _or_client.embeddings.create(
        model="google/gemini-embedding-001",
        input=text,
        encoding_format="float",
    )
    return np.array(response.data[0].embedding)


async def _save_review_embedding(reviews_dir: str, filename: str, review_text: str):
    """Embed a review and append it to the per-run DB (race-safe via flock)."""
    global _review_db
    vec = await _embed(review_text)
    pkl_path = os.path.join(reviews_dir, "embeddings.pkl")

    # Atomic read-modify-write with exclusive file lock
    with open(pkl_path, "a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read()
            db = pickle.loads(raw) if raw else {}
            db[filename] = vec
            f.seek(0)
            f.truncate()
            pickle.dump(db, f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    # Update in-memory DB so subsequent searches in this process see the new entry
    _review_db["filenames"].append(filename)
    if _review_db["vectors"].shape[0] == 0:
        _review_db["vectors"] = vec.reshape(1, -1)
    else:
        _review_db["vectors"] = np.vstack([_review_db["vectors"], vec])


def _make_merger_mcp_server(paper_dir: str, reviews_dir: str):
    from claude_agent_sdk import create_sdk_mcp_server, tool

    _ensure_indexes()
    allowed_paths = [paper_dir, HUMAN_REVIEW_DIR, reviews_dir]

    def _check_path(path: str) -> str | None:
        resolved = os.path.abspath(path)
        if any(resolved.startswith(ap) for ap in allowed_paths):
            return None
        return f"ERROR: Access denied. Path '{resolved}' is not under any allowed directory: {allowed_paths}"

    @tool(
        "read_file",
        "Read lines from a file. Returns lines numbered start_line to end_line (1-based). If end_line is 0, reads to EOF.",
        {"abs_path": str, "start_line": int, "end_line": int},
    )
    async def _read_file(args: dict) -> dict:
        abs_path = args["abs_path"]
        start_line = args.get("start_line", 1) or 1
        end_line = args.get("end_line", 0) or 0
        print(f"  [merger:read_file] {abs_path} lines {start_line}-{end_line or 'EOF'}")
        err = _check_path(abs_path)
        if err:
            return {"content": [{"type": "text", "text": err}], "is_error": True}
        try:
            with open(abs_path, "r", errors="replace") as fh:
                lines = fh.readlines()
            selected = lines[max(0, start_line - 1):end_line if end_line > 0 else len(lines)]
            text = "".join(f"{start_line + i}: {line}" for i, line in enumerate(selected))
            return {"content": [{"type": "text", "text": text}]}
        except FileNotFoundError:
            return {"content": [{"type": "text", "text": f"ERROR: File not found: {abs_path}"}], "is_error": True}

    @tool(
        "grep_file",
        "Search a single file for a substring pattern. Returns matching lines with line numbers.",
        {"pattern": str, "abs_path": str},
    )
    async def _grep_file(args: dict) -> dict:
        import re as _re
        pattern = args["pattern"]
        abs_path = args["abs_path"]
        print(f"  [merger:grep_file] pattern='{pattern}' in '{abs_path}'")
        err = _check_path(abs_path)
        if err:
            return {"content": [{"type": "text", "text": err}], "is_error": True}
        if not os.path.isfile(abs_path):
            return {"content": [{"type": "text", "text": f"ERROR: '{abs_path}' is not a file."}], "is_error": True}
        matches = []
        try:
            with open(abs_path, "r", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if _re.search(pattern, line):
                        matches.append(f"{i}: {line.rstrip()}")
        except Exception as e:
            return {"content": [{"type": "text", "text": f"ERROR: {e}"}], "is_error": True}
        text = "\n".join(matches) if matches else "No matches found."
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "search_file",
        "Search for a query in human reviews using BM25 or vector embeddings. Returns top n matching files. Set mode='bm25' or 'vector'.",
        {"query": str, "n": int, "mode": str},
    )
    async def _search_file(args: dict) -> dict:
        query = args["query"]
        n = args.get("n", 5)
        mode = args.get("mode", "vector")
        print(f"  [merger:search_file] query='{query}' n={n} mode='{mode}'")
        if mode == "bm25":
            bm25 = _bm25_db["bm25"]
            files = _bm25_db["files"]
            tokenized_query = query.split(" ")
            doc_scores = bm25.get_scores(tokenized_query)
            top_indices = doc_scores.argsort()[-n:][::-1]
            results = []
            for idx in top_indices:
                fpath = os.path.abspath(files[idx])
                score = doc_scores[idx]
                with open(fpath, "r", errors="replace") as fh:
                    content = fh.read()
                results.append(f"{fpath}\nscore: {score:.2f}\nfirst 1000 chars:\n{content[:1000]}\n")
            text = "\n---\n".join(results) if results else "No relevant files found."
        else:
            query_vector = await _embed(query)
            vectors = _bm25_db["vectors"]
            filenames = _bm25_db["filenames"]
            similarities = vectors @ query_vector.T
            top_indices = similarities.argsort()[-n:][::-1]
            results = []
            for idx in top_indices:
                fpath = os.path.abspath(f"../human_reviews/{filenames[idx]}")
                score = similarities[idx]
                with open(fpath, "r", errors="replace") as fh:
                    content = fh.read()
                results.append(f"{fpath}\nscore: {score:.2f}\nfirst 1000 chars:\n{content[:1000]}\n")
            text = "\n---\n".join(results) if results else "No relevant files found."

        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "search_review",
        "Search your own past reviews for self-consistency calibration. Returns top-n similar reviews by vector similarity. Use this to anchor your score against reviews you have already produced in this run.",
        {"query": str, "n": int},
    )
    async def _search_review(args: dict) -> dict:
        query = args["query"]
        n = args.get("n", 5)
        print(f"  [merger:search_review] query='{query}' n={n}")
        filenames = _review_db.get("filenames", [])
        vectors = _review_db.get("vectors", np.zeros((0, 0)))
        if len(filenames) == 0 or vectors.shape[0] == 0:
            return {"content": [{"type": "text", "text": "No past reviews yet. This is the first paper in the run — rely on your training knowledge of ICLR standards."}]}
        query_vector = await _embed(query)
        similarities = vectors @ query_vector.T
        top_n = min(n, len(filenames))
        top_indices = similarities.argsort()[-top_n:][::-1]
        results = []
        for idx in top_indices:
            fname = filenames[idx]
            fpath = os.path.abspath(os.path.join(_review_db["dir"], fname))
            score = similarities[idx]
            try:
                with open(fpath, "r", errors="replace") as fh:
                    content = fh.read()
                results.append(f"{fpath}\nscore: {score:.2f}\nfirst 1000 chars:\n{content[:1000]}\n")
            except FileNotFoundError:
                results.append(f"{fpath}\nscore: {score:.2f}\n[File not found on disk yet]\n")
        text = "\n---\n".join(results) if results else "No past reviews found."
        return {"content": [{"type": "text", "text": text}]}

    return create_sdk_mcp_server(
        name="merger_fs",
        version="1.0.0",
        tools=[_read_file, _grep_file, _search_file, _search_review],
    )


async def run_merger_claude_sdk(model_id: str, merger_prompt: str, paper_dir: str, reviews_dir: str, paper_filename: str) -> str:
    """
    Run the merger agent via Claude Agent SDK.
    Returns the final merged review text.
    After completion, embeds the review and saves it to the per-run DB.
    """
    from claude_agent_sdk import (
        ClaudeSDKClient,
        ClaudeAgentOptions,
        AssistantMessage,
        TextBlock,
    )

    print(f"  [Merger] starting Claude Agent SDK ({model_id}) ...")

    _load_review_db(reviews_dir)

    with open("prompts/merger.md", "r") as f:
        system_prompt = f.read()
    system_prompt = system_prompt.replace(
        "{{PAPER_ACCESS_INSTRUCTION}}",
        "The paper path is provided in the user message. Use read_file to read the paper and verify reviewer claims directly.",
    )

    mcp_server = _make_merger_mcp_server(paper_dir, reviews_dir)

    options = ClaudeAgentOptions(
        model=model_id,
        allowed_tools=[
            "mcp__merger_fs__read_file",
            "mcp__merger_fs__grep_file",
            "mcp__merger_fs__search_file",
            "mcp__merger_fs__search_review",
        ],
        permission_mode="bypassPermissions",
        disallowed_tools=["Read", "Glob", "Grep", "Bash", "Edit", "Write", "Agent"],
        mcp_servers={"merger_fs": mcp_server},
        max_turns=30,
    )

    full_prompt = f"{system_prompt}\n\n---\n\n{merger_prompt}"

    result_text = ""
    async with ClaudeSDKClient(options=options) as sdk_client:
        await sdk_client.query(full_prompt)
        async for message in sdk_client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        result_text += block.text

    if not result_text.strip():
        raise RuntimeError("[Merger] Claude Agent SDK returned empty output")

    print(f"  [Merger] done — {model_id} (Claude Agent SDK)")

    # Embed and save this review to the per-run DB
    print(f"  [Merger] saving review embedding for '{paper_filename}' ...")
    try:
        await _save_review_embedding(reviews_dir, paper_filename, result_text)
        print(f"  [Merger] embedding saved.")
    except Exception as e:
        print(f"  [Merger] WARNING: failed to save embedding: {e}")

    return result_text
