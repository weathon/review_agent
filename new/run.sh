export OPENAI_DEFAULT_MODEL="glm-5.1"
export HARSH_MODEL="hy3-preview:free" 
export MERGER_MODEL="hy3-preview:free"
export NEUTRAL_MODEL="hy3-preview:free"
export CALIBRATION_SET="2025"
export OUTPUT_CSV="results/bench_scores_hy_2025.csv"
export MERGE_LOG="results/pipeline_whole_hy_2025.log"
export SUBAGENT_MODEL="hy3-preview:free"
export CONCURRENCY=10
# rm bench_scores_qwen.log
ollama serve & 
git commit -am "run.sh: $(date) do not change anymore, wait till final"
# python main.py --n_samples 300 --benchmark ../iclr2026_cspaper/ --seed $(cksum <<< '🍍' | cut -f 1 -d ' ') 
python main.py --n_samples 300 --benchmark ../iclr2025/ --seed $(cksum <<< '23456789876543234567' | cut -f 1 -d ' ') 
