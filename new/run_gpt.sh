
export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="gpt-5.4"
export MERGER_MODEL="gpt-5.4"
export SUBAGENT_MODEL="ollama:glm-5.1:cloud"
export NEUTRAL_MODEL="ollama:glm-5.1:cloud"
export OUTPUT_CSV="bench_scores_claude_gpt.csv"
export MERGE_LOG="pipeline_whole_gpt.log"
ollama serve & 
# use balanced to see quality in the whole spectrum of scores
export CONCURRENCY=1
python main.py --n_samples 100 --benchmark ../iclr2025/ --seed $(cksum <<< 'who 3r324f34 you' | cut -f 1 -d ' ') 
 