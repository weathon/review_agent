export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="gpt-5.4"
export MERGER_MODEL="claude_sdk:claude-sonnet-4-6"
export NEUTRAL_MODEL="ollama:glm-5.1:cloud"
ollama serve &
python main.py --single_paper 2503.11651.pdf --accept_csv bench_scores_claude_sonnet.csv