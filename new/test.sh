export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="gpt-5.1"
export HUMAN_FINDER="ollama:glm-5.1:cloud"
export MERGER_MODEL="ollama:glm-5.1:cloud" 
export OUTPUT_CSV="bench_scores_glm.csv" 
export MERGE_LOG="pipeline_whole_2026.log"
python main.py --single_paper ../paper.md --accept_csv bench_scores_glm.csv