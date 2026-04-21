export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="claude_sdk:claude-sonnet-4-6"
export MERGER_MODEL="claude_sdk:claude-sonnet-4-6"
export NEUTRAL_MODEL="ollama:glm-5.1:cloud"
export OUTPUT_CSV="bench_scores_claude_sonnet_2.csv"
export MERGE_LOG="pipeline_whole_claude.log"
ollama serve & 
# use balanced to see quality in the whole spectrum of scores
export CONCURRENCY=1
python main.py --n_samples 100 --benchmark ../iclr2025/ --seed $(cksum <<< 'who are you' | cut -f 1 -d ' ') 
 