export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="gpt-5.4"
export HUMAN_FINDER="ollama:glm-5.1:cloud"
export MERGER_MODEL="gpt-5.4"
export OUTPUT_CSV="bench_scores.csv" 
export MERGE_LOG="pipeline_whole.log"
ollama serve & 
ollama pull gemini-3-flash-preview
python main.py --single_paper ../cvprw.md/2503.02910v3.md 