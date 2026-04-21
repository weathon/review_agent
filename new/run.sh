export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="glm-5.1"
export MERGER_MODEL="glm-5.1"
export NEUTRAL_MODEL="glm-5.1"
export OUTPUT_CSV="results/bench_scores_glm.csv"
export MERGE_LOG="results/pipeline_whole_glm.log"
export CONCURRENCY=10
# rm bench_scores_qwen.log
ollama serve &  
git commit -am "run.sh: $(date)"
python main.py --n_samples 100 --benchmark ../iclr2025/ --seed $(cksum <<< 'ewfuhdweyuf3eou' | cut -f 1 -d ' ') 
# python main.py --n_samples 200 --benchmark ../iclr2025/ --seed $(cksum <<< 'asfwekhjfwefw' | cut -f 1 -d ' ')
# python main.py --n_samples 500 --benchmark ../iclr2026_cspaper/ --seed $(cksum <<< 'asfwekhjfwefw' | cut -f 1 -d ' ') 
# python main.py --n_samples 500 --benchmark ../iclr2025/ --seed $(cksum <<< 'asfwekhjfwefw' | cut -f 1 -d ' ')
