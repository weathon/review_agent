export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="ollama:glm-5.1:cloud"
export MERGER_MODEL="ollama:glm-5.1:cloud"
export NEUTRAL_MODEL="ollama:glm-5.1:cloud"
export OUTPUT_CSV="results/bench_scores_glm.csv"
export MERGE_LOG="results/pipeline_whole_glm.log"
export CONCURRENCY=3
# rm bench_scores_qwen.log
ollama serve & 
git commit -am "run.sh: $(date) switch back to ollama"
python main.py --n_samples 300 --benchmark ../iclr2025/ --seed $(cksum <<< '🍍' | cut -f 1 -d ' ') 
# python main.py --n_samples 200 --benchmark ../iclr2025/ --seed $(cksum <<< 'asfwekhjfwefw' | cut -f 1 -d ' ')
# python main.py --n_samples 500 --benchmark ../iclr2026_cspaper/ --seed $(cksum <<< 'asfwekhjfwefw' | cut -f 1 -d ' ') 
# python main.py --n_samples 500 --benchmark ../iclr2025/ --seed $(cksum <<< 'asfwekhjfwefw' | cut -f 1 -d ' ')
