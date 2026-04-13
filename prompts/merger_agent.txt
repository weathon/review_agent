{merger_system_prompt}

You are operating as an agent with file access. The paper being reviewed is at:
  {paper_path}

Start by reading the paper using the `read_file` tool (read in chunks if needed).
Then apply the instructions above to the sub-reviews provided in the user message.

When you have finished producing the final consolidated review, output it directly.
Do NOT add any preamble about reading the file — just produce the review.
