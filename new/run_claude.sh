export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="gpt-5.1"
export HUMAN_FINDER="ollama:glm-5.1:cloud"
export MERGER_MODEL="claude_sdk:claude-sonnet-4-6"
export NEUTRAL_MODEL="ollama:glm-5.1:cloud"
export SPARK_MODEL="ollama:glm-5.1:cloud" 
export OUTPUT_CSV="bench_scores_claude_opus.csv" 
export MERGE_LOG="pipeline_whole_claude.log"
rm pipeline_whole_claude.log
ollama serve & 
python main.py --n_samples 200 --benchmark ../iclr2025/ --seed $(cksum <<< 'who are you' | cut -f 1 -d ' ') --balanced 
# use balanced to see quality in the whole spectrum of scores