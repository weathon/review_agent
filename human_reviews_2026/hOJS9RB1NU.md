# Webscale-RL: Automated Data Pipeline for Scaling RL Data to Pretraining Levels

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
Large Language Models (LLMs) have achieved remarkable success through imitation learning on vast text corpora, but this paradigm creates a training-generation gap and limits robust reasoning. Reinforcement learning (RL) offers a more data-efficient solution capable of bridging this gap, yet its application has been constrained by a critical data bottleneck: existing RL datasets are orders of magnitude smaller and less diverse than web-scale pre-training corpora. To address this, we introduce the \textbf{\texttt{Webscale-RL} pipeline}, a scalable data engine that systematically converts large-scale pre-training documents into millions of diverse, verifiable question-answer pairs for RL. Using this pipeline, we construct the \textbf{\texttt{Webscale-RL} dataset}, containing 1.2 million examples across more than 9 domains. Our experiments show that the model trained on this dataset significantly outperforms continual pretraining and strong data refinement baselines across a suite of benchmarks. Notably, RL training with our dataset proves substantially more efficient, achieving the performance of continual pre-training with up to 100$\times$ fewer tokens. Our work presents a viable path toward scaling RL to pre-training levels, enabling more capable and efficient language models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces Webscale-RL, an automated and scalable data pipeline designed to bridge the data bottleneck in reinforcement learning (RL) for large language models (LLMs). The core idea is to systematically convert large-scale pretraining corpora into verifiable question–answer (QA) pairs, enabling RL training at web scale. Using this pipeline, the authors construct the Webscale-RL dataset, consisting of 1.2 million QA pairs across nine or more domains. Empirical results demonstrate that models trained with RL on this dataset outperform continual pretraining and advanced data refinement baselines across a wide range of benchmarks (MMLU-Pro, BigBench, MATH500, etc.), achieving comparable or superior performance with up to 100× fewer tokens. Overall, the work aims to make RL training as scalable as pretraining, thus improving both capability and efficiency of LLMs.

### Strengths
1. The paper addresses one of the central challenges in applying RL to LLMs — the lack of large, diverse, and verifiable datasets. The proposed Webscale-RL pipeline is a substantial step toward scaling RL data generation to the same order of magnitude as pretraining corpora, a problem of clear importance for the community.

2. The data engine is well structured, with clearly defined stages — data filtering, domain classification with persona assignment, QA generation, and multi-stage verification. This modularity enhances reproducibility and potential adaptability to new domains.

3. The proposed framework could substantially reduce human labeling costs and enable scalable RL training for future LLM development. The paper provides sufficient implementation detail for reproducibility.

4. Experiments are extensive and convincing. RL training with Webscale-RL yields consistent gains across general reasoning, math, and STEM benchmarks. The results on data efficiency (100× fewer tokens to reach equivalent performance) are particularly impressive and suggest genuine benefits beyond data refinement or imitation-based methods.

### Weaknesses
1. Although multi-domain, the dataset underrepresents certain crucial areas (e.g., coding, tool use). This imbalance is reflected in weaker improvements on programming benchmarks (e.g., MBPP), limiting generalizability across task types.

2. The pipeline relies heavily on GPT-4.1 models for data filtering, classification, generation, and verification. This dependency may restrict reproducibility for groups without access to such models and raises cost and sustainability concerns.

3. Experiments are performed mainly on a 3B-parameter model (Qwen2.5-3B). While the improvements are substantial, it remains unclear how well the approach scales to larger or smaller models, or to different RL algorithms. Real-world downstream evaluations (dialogue, reasoning under uncertainty, tool use) are also absent.

### Questions
none

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Webscale-RL pipeline, a scalable data engine that systematically converts large-scale pre-training documents into millions of diverse, verifiable question-answer pairs for RL. Using this pipeline, the paper constructs the Webscale-RL dataset, containing 1.2 million examples across more than 9 domains.

### Strengths
1. The construction pipeline of this paper is solid and the proposed method is sound.
2. The manuscript is also well-written and easy to follow.

### Weaknesses
1. The motivation and methodology of this paper are quite similar to NaturalReasoning (Yuan et al., 2025), which significantly limits the novelty of this work.

2. The entire data generation and annotation pipeline heavily relies on GPT-4.1 and GPT-4.1-mini. This raises concerns about potential biases introduced by these models, and whether the associated costs are acceptable, especially when compared to the scale and cost-efficiency of large-scale pretraining corpora.

3. The experiments and analysis in this paper are quite limited — almost no in-depth analysis is provided. At the very least, a comparative experiment with NaturalReasoning should be included. Moreover, using RL as a post-training method and comparing it against continued pretraining may not be a fair comparison.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This study presents an automated pipeline for scaling reinforcement-learning (RL) data generation to pretraining-level diversity and scale, and it  identifies a fundamental bottleneck: RL training data (<10 B tokens) are vastly smaller and narrower than web-scale pretraining corpora (>1 T tokens). To bridge this gap, the authors propose a scalable RL-data engine that converts pretraining documents into verifiable question–answer pairs; and provide a webscale-RL dataset, containing ≈ 1.2 M verified QA pairs across 9 + domains (math, code, science, lifestyle, commerce, etc.), derived directly from pretraining corpora and easily extendable to trillions of tokens;  Empirical evidence that RL training on Webscale-RL yields large gains over continual pretraining and data-refinement baselines.

### Strengths
- Novel scalable data pipeline: first systematic approach to transform general pretraining corpora into verifiable RL data; elegant combination of domain and persona-based generation with automated quality control.

- Massive coverage & diversity: dataset spans 9 + domains, addressing the narrow-domain limitation of prior RL datasets (mostly math / code).

- Practical contribution: reusable, reproducible pipeline that could standardize RL-data construction for open-source models.

### Weaknesses
- Limited transparency of generation details: prompt templates, filtering thresholds, and judge-LLM configurations are summarized only in appendices; reproducibility may depend on access to proprietary models (GPT-4.1).


- Evaluation bias: The SFT “warm-up” and differing token budgets between RL and continual pretraining make it difficult to ensure the strict fairness of comparison.


- Domain imbalance: coding and certain STEM domains remain underrepresented, reflected in smaller gains on code benchmarks.


- The cost and environmental impact of multi-stage LLM generation and verification are not analyzed.

### Questions
- How robust are results if GPT-4.1 is replaced by a smaller verifier? Does verification bias the dataset toward specific model families?


- Have you tested pipeline scalability beyond 1.2 M pairs, e.g., to 100 M, regarding quality degradation, cost, or RL performance saturation?


- Since persona assignment drives diversity, do personas from one domain (e.g., healthcare) transfer to unseen domains?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the scaling of reinforcement learning dataset. They propose to utilize LLM to convert pretraining dataset into verifiable QA pairs. The empirical evaluation shows that this is much more efficient than naive continual pretraining, providing a potential way to scaling rl and the possibility to change the pretraining/rl paradigm.

### Strengths
1. This paper demonstrates the viability of converting pretraining data into reinforcement data. This is a rather novel research direction compared to ones that simply do filtering or rephrasing of pretraining data.

### Weaknesses
1. There seems to be a limitation of diversity for generating verifiable answers. The correctness also might depend on the powerfulness of the LLM used for conversion.
2. The description of the proposed method seems to be a bit overly succinct. Even though some parts of the pipelines might share similarities with the paper cited. For completeness, it is better to expand the description in Sec 3.2 and clearly discuss the difference with the prior works. The section for verifiable generation seems to be especially important for further discussions.

### Questions
1. Could the authors expand the descriptions of Sec 3.2 to specify more details of each pipeline stage?
2. Do the authors have any ablation to show the diversity of the generated rl data and how it correlates with the final performance (Are there any redundancy in the generation thus the performance seems to be saturating or are there  room for further improvements).

### Soundness
3

### Presentation
3

### Contribution
3
