export OPENAI_DEFAULT_MODEL="glm-5.1"
export HARSH_MODEL="ollama:gemma4:31b-cloud" 
export MERGER_MODEL="ollama:gemma4:31b-cloud"
export NEUTRAL_MODEL="ollama:gemma4:31b-cloud"
export CALIBRATION_SET="2025"
export OUTPUT_CSV="results/bench_scores_gemma.csv"
export MERGE_LOG="results/pipeline_whole_gemma.log"
export SUBAGENT_MODEL="ollama:gemma4:31b-cloud"
export CONCURRENCY=5
# rm bench_scores_qwen.log
ollama serve & 
git add -A
git commit -am "run.sh: $(date)"
# python main.py --n_samples 300 --benchmark ../iclr2026_cspaper/ --seed $(cksum <<< '🍍' | cut -f 1 -d ' ') 
python main.py --n_samples 800 --benchmark ../iclr2025/ --seed $(cksum <<< '23456789876543234567' | cut -f 1 -d ' ') 
