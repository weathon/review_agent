set -a && source "$(dirname "$0")/.env" && set +a

export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY"
export ANTHROPIC_API_KEY="" # Important: Must be explicitly empty

export ANTHROPIC_DEFAULT_OPUS_MODEL="qwen/qwen3.6-plus"
export ANTHROPIC_DEFAULT_SONNET_MODEL="qwen/qwen3.5-flash-02-23"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="qwen/qwen3.5-flash-02-23"
export CLAUDE_CODE_SUBAGENT_MODEL="qwen/qwen3.6-plus"


claude