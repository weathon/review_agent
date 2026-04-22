export OPENAI_DEFAULT_MODEL="glm-5.1"
export HARSH_MODEL="gpt-5.2" 
export MERGER_MODEL="gpt-5.2"
export NEUTRAL_MODEL="gpt-5.2"
export CALIBRATION_SET="2026"
export OUTPUT_CSV="results/bench_scores_glm_2026_new.csv"
export MERGE_LOG="results/pipeline_whole_glm_2026.log"
export SUBAGENT_MODEL="gpt-5.4-mini"
export CONCURRENCY=5
ollama serve & 
git add .
git commit -m "run.sh: $(date) test 2026"
python main.py --n_samples 300 --benchmark ../iclr2026_cspaper_new/ --seed $(cksum <<< '45678765456787654567' | cut -f 1 -d ' ') 
