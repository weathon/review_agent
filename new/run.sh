export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="gpt-5.1"
# export MERGER_MODEL="claude_sdk:claude-sonnet-4-6"
export MERGER_MODEL="qwen3.5-plus-02-15"
export NEUTRAL_MODEL="qwen3.5-plus-02-15"
export OUTPUT_CSV="bench_scores.csv" 
export MERGE_LOG="pipeline_whole_2025.log" 
rm pipeline_whole_2025.log
ollama serve & 
python main.py --n_samples 100 --benchmark ../iclr2025/ --seed $(cksum <<< 'who are you' | cut -f 1 -d ' ') 