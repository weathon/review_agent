export OPENAI_DEFAULT_MODEL="gpt-5.1"
export HARSH_MODEL="moonshotai/kimi-k2.6" 
export MERGER_MODEL="moonshotai/kimi-k2.6"
export NEUTRAL_MODEL="moonshotai/kimi-k2.6"
export CALIBRATION_SET="2026"
export OUTPUT_CSV="results/bench_scores_kimi-k2.6_2026.csv"
export MERGE_LOG="results/pipeline_whole_kimi-k2.6_2026.log"
export CONCURRENCY=10
ollama serve & 
git add .
git commit -m "run.sh: $(date) test 2026"
python main.py --n_samples 200 --benchmark ../iclr2026_cspaper/ --seed $(cksum <<< '45678765456787654567' | cut -f 1 -d ' ') 
