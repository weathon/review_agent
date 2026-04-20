export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="qwen3.6-plus"
export MERGER_MODEL="qwen3.6-plus"
export NEUTRAL_MODEL="ollama:glm-5.1:cloud"
export OUTPUT_CSV="bench_scores_qwen.csv"
export MERGE_LOG="pipeline_whole_qwen.log"
export CONCURRENCY=10
rm bench_scores_qwen.log
ollama serve & 
# python main.py --n_samples 100 --benchmark ../iclr2025/ --seed $(cksum <<< 'who are you' | cut -f 1 -d ' ') 
python main.py --n_samples 200 --benchmark ../iclr2025/ --seed $(cksum <<< 'asfwekhjfwefw' | cut -f 1 -d ' ')
 