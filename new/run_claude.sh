source ../.env
export ANTHROPIC_API_KEY=""
export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="claude_sdk:claude-sonnet-4-6"
export MERGER_MODEL="claude_sdk:claude-sonnet-4-6"
export NEUTRAL_MODEL="glm-5.1"
export CALIBRATION_SET="2026"
export OUTPUT_CSV="bench_scores_claude_sonnet_2026.csv"
export MERGE_LOG="pipeline_whole_claude_2026.log"
ollama serve & 
# use balanced to see quality in the whole spectrum of scores
export CONCURRENCY=2    
# python main.py --n_samples 100 --benchmark ../iclr2025/ --seed $(cksum <<< 'who 3r324f34 you' | cut -f 1 -d ' ') 
# python main.py --n_samples 300 --benchmark ../iclr2025/ --seed $(cksum <<< '🍍' | cut -f 1 -d ' ') 
# python main.py --n_samples 300 --benchmark ../iclr2025/ --seed $(cksum <<< '23456789876543234567' | cut -f 1 -d ' ') 

python main.py --n_samples 200 --benchmark ../iclr2026_cspaper/ --seed $(cksum <<< '45678765456787654567' | cut -f 1 -d ' ') 
