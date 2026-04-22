# DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Reinforcement learning (RL) with large language models shows promise in complex reasoning. However, its progress is hindered by the lack of large-scale training data that is sufficiently challenging, contamination-free and verifiable. To solve this problem, we introduce DeepMath-103K, a large-scale mathematical dataset designed with high difficulty (primarily levels 5-9), rigorous decontamination against numerous benchmarks, and verifiable answers for rule-based RL reward. It further includes three distinct R1 solutions adaptable for diverse training paradigms such as supervised fine-tuning. Spanning a wide range of mathematical topics, DeepMath-103K fosters the development of generalizable and advancing reasoning. Notably, models trained on DeepMath-103K achieve leading results on challenging mathematical benchmarks and demonstrate generalization beyond math such as biology, physics and chemistry, underscoring its broad efficacy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a carefully constructed, high‑difficulty, decontaminated, and verifiable dataset for math RL and demonstrates strong small‑model results and promising cross‑domain transfer. The authors curate problems primarily from Math StackExchange‑style web sources and pass them through a four‑stage pipeline. The result is 95K challenging items augmented with 8K level‑3–5 problems, each annotated with a hierarchical topic label and three diverse solutions. Models trained on DeepMath‑103K achieve substantial gains on several existing benchmarks such as MATH‑500 and  AMC23. Also, results on GPQA‑Diamond indicate transfer beyond mathematics.

### Strengths
S1: This paper builds a high‑difficulty dataset with a clear distribution skew toward level‑5–9 problems, which directly addresses a key bottleneck for training strong reasoners.

S2: The dataset shows strong topical breadth via a hierarchical taxonomy that spans core and advanced areas. Each problem includes a rule‑verifiable final answer and three R1 solution paths, enabling RLVR and SFT without manual cleaning.

S3: The models trained on this dataset achieve state‑of‑the‑art small‑model results on AIME and other math benchmarks and show transfer to GPQA‑Diamond.

### Weaknesses
W1: The paper briefly cites contemporaneous datasets (e.g., BigMath and OpenMathReasoning) but does not provide head‑to‑head dataset ablations or training‑data mixing studies that would establish relative benefits.

W2: The claims of cross‑domain generalization rely mainly on GPQA‑Diamond. It would be more convincing if the authors evaluated on additional scientific or reasoning benchmarks

W3: Several closely related works are missing from the discussion [1–4].

W4: The data curation pipeline heavily depends on LLMs—an LLM-judge (Llama-3.3-70B) for decontamination, GPT-4o for difficulty labeling, and GPT-4o again for rewriting and answer verification. Since LLMs are not perfectly reliable, hallucinations or propagation errors may occur. The authors should clarify whether human verification was performed for each stage.

W5: The experiments are restricted to small and medium-sized models. It would strengthen the paper if the authors included results on larger models (e.g., 72B)



Typos: 

+ In abstract, advancing reasoning  -> advanced reasoning 

+ SITLL‑3‑RL or STILL‑3‑RL should be made consistent throughout the paper.



Reference

[1] Albalak et al., Big‑Math: A Large‑Scale, High‑Quality Math Dataset for Reinforcement Learning in Language Models, arXiv:2502.17387, 2025.

[2] Moshkov et al., AIMO‑2 Winning Solution: Building State‑of‑the‑Art Mathematical Reasoning Models with OpenMathReasoning Dataset, arXiv:2504.16891, 2025.

[3] Paster et al., OpenWebMath: An Open Dataset of High‑Quality Mathematical Web Text, NeurIPS 2023.

[4] Zhou et al., MegaMath: Pushing the Limits of Open Math Corpora, COLM 2025.

### Questions
see above.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces DeepMath-103K, a large-scale and challenging mathematical reasoning dataset for reinforcement learning with verifiable rewards. It emphasizes benchmark decontamination, difficulty filtering, and verified final answers. The authors also train a series of DeepMath models on this dataset and show performance gains on AIME, AMC, MATH-500, OlympiadBench, and GPQA-Diamond.

### Strengths
(1) The work is well-motivated and tackles a meaningful gap in mathematical reasoning datasets, focusing on challenge level, contamination issues, and result verifiability.

(2) A thorough decontamination process is implemented, combining semantic matching and LLM-based assessment to effectively minimize data leakage. 

(3) Experiments demonstrate consistent gains over baseline methods across various model sizes and learning settings—both in zero-shot and reinforcement learning scenarios.

### Weaknesses
(1)	Difficulty annotation relies entirely on GPT-4o without expert calibration, and no inter-rater agreement or error analysis is provided.

(2)	Lack of ablation on the curation pipeline—unclear which components (decontamination, difficulty filtering, answer verification) are most critical to performance gains.

(3)	R1 solution generation bias – R1-generated reasoning paths may reinforce self-biased reasoning structures, but the paper lacks discussion of potential reasoning homogenization or confirmation bias.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposed DeepMath-103k, a large-scale and challenging mathematical reasoning dataset. The dataset is constructed to emphasize the difficulty of the problems, which is proven to be essential to the improvement of finetuning models on these data. The authors show with data that their dataset is also the most diverse dataset that contains most unique questions. Above all, the extensive experiments show that models finetuned with this new dataset are promised to have significant improvement over existing open-source baselines.

### Strengths
1. The paper is clearly written and demonstrated.
2. The authors clearly declared and proved the unique advantage of their dataset, with its number, difficulty and diversity.
3. The experimental results proved that the improvment brought by the dataset is promising and consistent.

### Weaknesses
There is no significant weakness within the dataset scope.

### Questions
1. The table 2 results are promising, but it seems lack of solid baselines with previous existing datasets. I understand that it is not easy to find appropriate baselines sometimes, but I would like the authors to clarify this point with some discussions.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a large-scale dataset, DeepMath-103K, with high difficulty (mainly levels 5-9) and verifiable answers for rule-based RL, compared with other datasets (DAPO-17K, DSR-Preview, Open-R1, ORZ-129K). Using DeepMath-103K, the models under zero RL and RL, outperform than others, such as ORZ-7B, Qwen-2.5-Math-7B. It seems a very useful dataset for AI community. The construction of DeepMath-103K is clear. However, more experiments should be conducted, such as Qwen-2.5-7B-DAPO-17K, Qwen-2.5-7B-ORZ-129K, etc.

### Strengths
1. This paper introduces a useful dataset DeepMath-103k with high difficulty, and the construction of dataset is very clear.
2. DeepMath-103k performs better than other datasets, with same training method (zero RL from base model, and RL from instruct models)

### Weaknesses
The experiments is not enough. A more comprehensive comparison with existing datasets is needed, including ORZ, DAPO-MATH-17K, etc.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3
