import asyncio
import os
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock, ResultMessage
import pandas as pd
from claude_agent_sdk import tool
import csv


MODEL = "claude-sonnet-4-6"
with open("prompts/bare_baseline.txt", "r") as f:
    prompt = f.read()

save_path = "bare_baseline_results.csv"
direct_scores_csv = "direct_scores.csv" # use to extract papers need to be reviewed
bench_dir = "iclr2025"

gt = pd.read_csv(f"{bench_dir}/ratings.csv")

direct_scores_df = pd.read_csv(direct_scores_csv)
papers_to_review = direct_scores_df["paper_id"].tolist()



def _make_sandboxed_tools(paper_path):
    allowed_paper_path = os.path.abspath(paper_path)

    @tool(
        "read_file",
        "Read a file. Returns the full content with line numbers. Restricted to allowed directories only.",
        {"abs_path": str, "start_line": int, "end_line": int},
    )
    async def _read_file(args: dict) -> dict:
        print(f"  [read_file] requested: {args['abs_path']} (lines {args.get('start_line', 1)}-{args.get('end_line', 'end')})")
        abs_path = args["abs_path"]
        start_line = args.get("start_line", 1) or 1
        end_line = args.get("end_line", 0) or 0
        if allowed_paper_path != os.path.abspath(abs_path):
            err = f"Access denied: {abs_path} is not the paper path."
            print(f"  [read_file] ERROR: {err}")
            return {"content": [{"type": "text", "text": err}], "is_error": True}
        try:
            with open(abs_path, "r", errors="replace") as file_handle:
                lines = file_handle.readlines()
            selected = lines[max(0, start_line - 1):end_line if end_line > 0 else len(lines)]
            text = "".join(f"{start_line + i}: {line}" for i, line in enumerate(selected))
            return {"content": [{"type": "text", "text": text}]}
        except:
            print(f"  [read_file] ERROR: {abs_path}")
            return {"content": [{"type": "text", "text": f"ERROR: {abs_path}"}], "is_error": True}
    return [_read_file]


with open(save_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["paper_id", "pred_score", "pred_decision", "gt_avg_score", "gt_decision", "gt_binary", "match", "cost", "sdk_savings",
                    "gt_score_0", "gt_score_1", "gt_score_2", "gt_score_3", "gt_score_4", "gt_score_5", "gt_score_6"])

async def review_paper(paper_id):

    from claude_agent_sdk import create_sdk_mcp_server
    paper_path = f"{bench_dir}/papers/{paper_id}.txt"
    server = create_sdk_mcp_server(
        name="reviewer_fs",
        version="1.0.0",
        tools=_make_sandboxed_tools(paper_path),
    )

    options = ClaudeAgentOptions(
        model=MODEL,
        allowed_tools=[
            "mcp__reviewer_fs__read_file",
        ],
        permission_mode="bypassPermissions",
        disallowed_tools=["Read", "Glob", "Grep", "Bash", "Edit", "Write", "Agent", "WebFetch"],
        mcp_servers={"reviewer_fs": server},
        effort="medium",
        max_turns=15,
    )

    formatted_prompt = prompt.format(paper_path=os.path.abspath(paper_path))

    review_text = ""
    cost = 0
    async with ClaudeSDKClient(options=options) as sdk_client:
        await sdk_client.query(f"{formatted_prompt}")
        async for message in sdk_client.receive_response():
            if isinstance(message, AssistantMessage):
                print(message.content)
            if isinstance(message, ResultMessage):
                cost += message.total_cost_usd
                review_text = message.result


    review = review_text.strip()
    score = float(review.split("<pineapple>")[1].split("</pineapple>")[0].strip())
    gt_scores = gt[gt["paper_id"] == paper_id][["score_0","score_1","score_2","score_3","score_4","score_5"]].values.flatten().tolist()
    gt_avg_score = gt[gt["paper_id"] == paper_id]["avg_score"].values[0]

    with open(save_path, "a", newline="") as f:
        w = csv.writer(f)
        gt_scores_padded = gt_scores + [""] * (7 - len(gt_scores))
        w.writerow([
            paper_id,
            score,
            "Accept" if score >= 5 else "Reject",
            f"{gt_avg_score:.2f}",
            gt[gt["paper_id"] == paper_id]["decision"].values[0],
            gt[gt["paper_id"] == paper_id]["gt_binary"].values[0],
            "",
            f"{cost:.4f}",
            f"{cost}",
            *gt_scores_padded,
        ])

    with open(save_path.replace(".csv", f"/{paper_id}.txt"), "w") as f:
        f.write(review_text)

import tqdm
if __name__ == "__main__":
    os.makedirs(save_path.replace(".csv", ""), exist_ok=True)

    for paper in tqdm.tqdm(papers_to_review):
        asyncio.run(review_paper(paper))

    
