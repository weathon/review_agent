export OPENAI_DEFAULT_MODEL="glm-5.1"
export HARSH_MODEL="glm-5.1" 
export MERGER_MODEL="glm-5.1"
export NEUTRAL_MODEL="glm-5.1"
export OUTPUT_CSV="results/bench_scores_glm_2026.csv"
export MERGE_LOG="results/pipeline_whole_glm_2026.log"
export SUBAGENT_MODEL="glm-5.1"
export CONCURRENCY=5
# rm bench_scores_qwen.log
ollama serve & 
git add .
git commit -m "run.sh: $(date) test 2026"
python main.py --n_samples 300 --benchmark ../iclr2026_cspaper/ --seed $(cksum <<< '45678765456787654567' | cut -f 1 -d ' ') 