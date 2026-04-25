export OPENAI_DEFAULT_MODEL="glm-5.1"
export HARSH_MODEL="deepseek-v4-flash" 
export MERGER_MODEL="deepseek-v4-flash"
export NEUTRAL_MODEL="deepseek-v4-flash"
export CALIBRATION_SET="2025"
export OUTPUT_CSV="results/bench_scores_deepseek_2025.csv"
export MERGE_LOG="results/pipeline_whole_deepseek_2025.log"
export SUBAGENT_MODEL="deepseek-v4-flash"
export CONCURRENCY=20
# rm bench_scores_qwen.log
ollama serve & 
git commit -am "run.sh: $(date)"
# python main.py --n_samples 300 --benchmark ../iclr2026_cspaper/ --seed $(cksum <<< '🍍' | cut -f 1 -d ' ') 
python main.py --n_samples 300 --benchmark ../iclr2025/ --seed $(cksum <<< '23456789876543234567' | cut -f 1 -d ' ') 
