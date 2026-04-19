export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="claude_sdk:claude-sonnet-4-6"
export MERGER_MODEL="qwen3.5-plus-02-15"
# export MERGER_MODEL="qwen3.6-plus"
export NEUTRAL_MODEL="ollama:glm-5.1:cloud"
export OUTPUT_CSV="bench_scores_fixed.csv" 
export MERGE_LOG="pipeline_whole_2025.log"
rm pipeline_whole_2025.log
ollama serve & 
python main.py --n_samples 500 --benchmark ../iclr2025/ --seed $(cksum <<< 'who are you' | cut -f 1 -d ' ') 