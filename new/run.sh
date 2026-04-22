export OPENAI_DEFAULT_MODEL="ollama:glm-5.1:cloud"
export HARSH_MODEL="ollama:glm-5.1:cloud" 
export MERGER_MODEL="ollama:glm-5.1:cloud"
export NEUTRAL_MODEL="ollama:glm-5.1:cloud"
export OUTPUT_CSV="results/bench_scores_glm.csv"
export MERGE_LOG="results/pipeline_whole_glm.log"
export SUBAGENT_MODEL="ollama:glm-5.1:cloud"
export CONCURRENCY=5
# rm bench_scores_qwen.log
ollama serve & 
git commit -am "run.sh: $(date) back to test 2025 benchmark using ollama"
# python main.py --n_samples 300 --benchmark ../iclr2026_cspaper/ --seed $(cksum <<< '🍍' | cut -f 1 -d ' ') 
python main.py --n_samples 800 --benchmark ../iclr2025/ --seed $(cksum <<< '23456789876543234567' | cut -f 1 -d ' ') 
