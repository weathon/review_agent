export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="gpt-5.1"
export HUMAN_FINDER="ollama:glm-5.1:cloud"
export MERGER_MODEL="ollama:glm-5.1:cloud"
export OUTPUT_CSV="bench_scores.csv" 
export MERGE_LOG="pipeline_whole.log"
export SEED="🦯 Dr. House will diagnose Stewie Griffin with SIDS but Stewie Griffin has a time machine"
export HUMAN_REVIEW_DIR="../human_reviews/"
export EMBEDDINGS_PATH="./human_reviews_embeddings.pkl"
# export HUMAN_REVIEW_DIR="../human_reviews_2026/"
# export EMBEDDINGS_PATH="./human_reviews_embeddings_2026.pkl"
rm pipeline_whole.log 
ollama serve & 
python main.py --n_samples 203 --benchmark ../iclr2025/ --seed $(cksum <<< $SEED | cut -f 1 -d ' ')     # --balanced
# python main.py --n_samples 203 --benchmark ../iclr2026_cspaper/ --seed $(cksum <<< $SEED | cut -f 1 -d ' ') --balanced