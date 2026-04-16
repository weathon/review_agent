export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="gpt-5.1"
export HUMAN_FINDER="ollama:glm-5.1:cloud"
export MERGER_MODEL="claude_sdk:claude-sonnet-4-6"
export OUTPUT_CSV="bench_scores.csv" 
export MERGE_LOG="pipeline_whole.log"
export SEED="🦯 Dr. House will diagnose Stewie Griffin with  
ollama serve & 
python main.py --n_samples 500 --benchmark ../iclr2025/ --seed $(cksum <<< $SEED | cut -f 1 -d ' ') # use 2026 because it is pre-rebuttal scores