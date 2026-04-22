export OPENAI_DEFAULT_MODEL="glm-5.1"
export HARSH_MODEL="ollama:glm-5.1:cloud" 
export MERGER_MODEL="ollama:glm-5.1:cloud"
export NEUTRAL_MODEL="ollama:glm-5.1:cloud"
export OUTPUT_CSV="results/bench_scores_glm_2026.csv"
export MERGE_LOG="results/pipeline_whole_glm_2026.log"
export SUBAGENT_MODEL="glm-5.1"
export CONCURRENCY=10
# rm bench_scores_qwen.log
ollama serve & 
git commit -am "run.sh: $(date)"
python main.py --n_samples 300 --benchmark ../iclr2026_cspaper/ --seed $(cksum <<< '🍍' | cut -f 1 -d ' ') 
# python main.py --n_samples 100 --benchmark ../iclr2025/ --seed $(cksum <<< 'who 3r324f34 you' | cut -f 1 -d ' ') 
