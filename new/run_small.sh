export OPENAI_DEFAULT_MODEL="qwen3.5-flash-02-23" 
export HARSH_MODEL="qwen3.5-flash-02-23"
export HUMAN_FINDER="qwen3.5-flash-02-23"
export MERGER_MODEL="qwen3.5-flash-02-23"
export OUTPUT_CSV="bench_scores_flash.csv"
export MERGE_LOG="pipeline_flash.log"
python main.py --n_samples 200 --benchmark ../iclr2025/ --seed $(cksum <<< 'who are you' | cut -f 1 -d ' ')