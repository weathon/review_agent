export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="gpt-5.1"
export HUMAN_FINDER="ollama:glm-5.1:cloud"
export MERGER_MODEL="ollama:glm-5.1:cloud" 
export NEUTRAL_MODEL="ollama:glm-5.1:cloud"
export SPARK_MODEL="ollama:glm-5.1:cloud"
python main.py --single_paper ./papers/2602.10095v1.pdf --accept_csv bench_scores_glm.csv