# export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
# export HARSH_MODEL="claude_sdk:claude-sonnet-4-6"
# export MERGER_MODEL="claude_sdk:claude-sonnet-4-6"
# export NEUTRAL_MODEL="ollama:glm-5.1:cloud"


# export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
# export HARSH_MODEL="ollama:kimi-k2.5:cloud"
# export MERGER_MODEL="ollama:kimi-k2.5:cloud"
# export NEUTRAL_MODEL="ollama:kimi-k2.5:cloud"
# export ANTHROPIC_API_KEY=""
# export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
# export HARSH_MODEL="claude_sdk:claude-sonnet-4-6"
# export MERGER_MODEL="claude_sdk:claude-sonnet-4-6"
# export NEUTRAL_MODEL="glm-5.1"
source ../.env
export ANTHROPIC_API_KEY
export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="claude_sdk:claude-sonnet-4-6"
export MERGER_MODEL="claude_sdk:claude-sonnet-4-6"
export NEUTRAL_MODEL="ollama:glm-5.1:cloud"
export OUTPUT_CSV="bench_scores_claude_sonnet_3.csv"
export MERGE_LOG="pipeline_whole_claude_3.log"

ollama serve &
python main.py --single_paper ../paper.md --accept_csv bench_scores_claude_sonnet_3.csv
# python main.py --single_paper ./papers/paper.md