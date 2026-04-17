export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="gpt-5.1"
export HUMAN_FINDER="ollama:glm-5.1:cloud"
export MERGER_MODEL="claude_sdk:claude-sonnet-4-6"
# export MERGER_MODEL="ollama:glm-5.1:cloud" 
export OUTPUT_CSV="bench_scores.csv" 
export MERGE_LOG="pipeline_whole.log"
ollama serve & 
python main.py --n_samples 500 --benchmark ../iclr2026_cspaper/ --seed $(cksum <<< 'who are you' | cut -f 1 -d ' ') # use 2026 because it is pre-rebuttal scores