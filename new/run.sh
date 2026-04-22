export OPENAI_DEFAULT_MODEL="gpt-5.1"
export HARSH_MODEL="gpt-5.1" 
export MERGER_MODEL="gpt-5.1"
export NEUTRAL_MODEL="gpt-5.1"
export CALIBRATION_SET="2025"
export OUTPUT_CSV="results/bench_scores_gpt-5.1_2025.csv"
export MERGE_LOG="results/pipeline_whole_gpt-5.1_2025.log"
export SUBAGENT_MODEL="gpt-5.1"
export CONCURRENCY=20
# rm bench_scores_qwen.log
ollama serve & 
git commit -am "run.sh: $(date)"
# python main.py --n_samples 300 --benchmark ../iclr2026_cspaper/ --seed $(cksum <<< '🍍' | cut -f 1 -d ' ') 
python main.py --n_samples 300 --benchmark ../iclr2025/ --seed $(cksum <<< '23456789876543234567' | cut -f 1 -d ' ') 
