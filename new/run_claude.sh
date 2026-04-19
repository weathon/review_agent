export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="gpt-5.4"
export MERGER_MODEL="claude_sdk:claude-sonnet-4-6"
export NEUTRAL_MODEL="ollama:glm-5.1:cloud"
export OUTPUT_CSV="bench_scores_claude_sonnet.csv"
export MERGE_LOG="pipeline_whole_claude.log"
rm pipeline_whole_claude.log
ollama serve & 
python main.py --n_samples 200 --benchmark ../iclr2025/ --seed $(cksum <<< 'asfwekhjfwefw' | cut -f 1 -d ' ')
# use balanced to see quality in the whole spectrum of scores