export OPENAI_DEFAULT_MODEL="gpt-5.1"
export HARSH_MODEL="gpt-5.1" 
export MERGER_MODEL="gpt-5.1"
export NEUTRAL_MODEL="gpt-5.1"
export OUTPUT_CSV="results/bench_scores_gpt_or.csv"
export MERGE_LOG="results/pipeline_whole_gpt_or.log"
export SUBAGENT_MODEL="gpt-5.4-mini"
export CONCURRENCY=10
# rm bench_scores_qwen.log
ollama serve & 
git commit -am "run.sh: $(date)"
python main.py --n_samples 300 --benchmark ../iclr2025/ --seed $(cksum <<< '🍍' | cut -f 1 -d ' ') 
# python main.py --n_samples 100 --benchmark ../iclr2025/ --seed $(cksum <<< 'who 3r324f34 you' | cut -f 1 -d ' ') 
