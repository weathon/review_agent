# export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
# export HARSH_MODEL="claude_sdk:claude-sonnet-4-6"
# export MERGER_MODEL="claude_sdk:claude-sonnet-4-6"
# export NEUTRAL_MODEL="ollama:glm-5.1:cloud"


# export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
# export HARSH_MODEL="ollama:kimi-k2.5:cloud"
# export MERGER_MODEL="ollama:kimi-k2.5:cloud"
# export NEUTRAL_MODEL="ollama:kimi-k2.5:cloud"
export ANTHROPIC_API_KEY=""
export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="claude_sdk:claude-sonnet-4-6"
export MERGER_MODEL="claude_sdk:claude-sonnet-4-6"
export NEUTRAL_MODEL="glm-5.1"

ollama serve &
python main.py --single_paper ./papers/2601.22638v1.pdf # --accept_csv results/bench_scores_kimi.csv
# python main.py --single_paper ./papers/paper.md