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
export HARSH_MODEL="ollama:glm-5.1:cloud" 
export MERGER_MODEL="ollama:glm-5.1:cloud"
export NEUTRAL_MODEL="ollama:glm-5.1:cloud"
export SUBAGENT_MODEL="ollama:glm-5.1:cloud"
export CALIBRATION_SET="2025"

ollama serve &
python main.py --single_paper ../new/gated.md
# python main.py --single_paper f4oAYJxrgH.pdf  # --accept_csv bench_scores_claude_sonnet_3.csv
# python main.py --single_paper