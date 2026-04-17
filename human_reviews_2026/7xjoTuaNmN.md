# OpenThoughts: Data Recipes for Reasoning Models

- Decision: Accept (Oral)
- Scores: 6, 6, 6, 8

## Abstract
Reasoning models have made rapid progress on many benchmarks involving math,
code, and science. Yet, there are still many open questions about the best train-
ing recipes for reasoning since state-of-the-art models often rely on proprietary
datasets with little to no public information available. To address this, the goal of
the OpenThoughts project is to create open-source datasets for training reasoning
models. Our OpenThoughts2-1M dataset led to OpenThinker2-32B, the first model
trained on public reasoning data to match DeepSeek-R1-Distill-32B on standard
reasoning benchmarks such as AIME and LiveCodeBench. We then improve
our dataset further by systematically investigating each step of our data genera-
tion pipeline with 1,000+ controlled experiments, which led to OpenThoughts3.
Scaling the pipeline to 1.2M examples and using QwQ-32B as teacher yields
our OpenThinker3-7B model, which achieves state-of-the-art results: 53% on
AIME 2025, 51% on LiveCodeBench 06/24-01/25, and 54% on GPQA Dia-
mond – improvements of 15.3, 17.2, and 20.5 percentage points compared to the
DeepSeek-R1-Distill-Qwen-7B. All of our datasets and models are available on
openthoughts.ai.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the challenge of creating publicly available data recipes for training reasoning models. It introduces a data generation pipeline for creating open-source reasoning data by empirically selecting the most effective approach at each stage. Using this pipeline, the authors construct the OpenThoughts2-1M and OpenThoughts3 datasets. A model trained on this data, named OpenThinker3-7B, achieves state-of-the-art performance.

### Strengths
S1: The paper conducts thorough experiments and constructs open-source datasets, which paves the way for future research on reasoning models.

S2: The paper is well-structured and easy to follow.

S3: The paper addresses a practical data problem for reasoning model training and research, which is an important contribution to the development of AI.

### Weaknesses
W1: The paper's aim to create open-source datasets is undermined by the proposed pipeline's high dependency on closed-source LLM APIs, such as GPT-4o. This creates a contradiction and harms the reproducibility of the work, as results can vary significantly depending on the specific API version used.

W2: Some experimental conclusions lack rigor. For instance, in Section 3.6, the paper dismisses answer filtering strategies as ineffective because they did not outperform the baseline. However, this conclusion is not well-supported because the comparison is unfair: the baseline was trained on 63,200 samples, while the answer filtering strategies used only 31,600. This confounding variable makes it impossible to rule out that the baseline's superior performance stems from having more training data, rather than the ineffectiveness of filtering.

W3: The paper lacks an in-depth analysis of some interesting and counterintuitive experimental results, failing to provide deeper insights. For example, Section 3.7 shows that while Qwen-32B has a lower average score than DeepSeek-R1, it outperforms all other models when used as a teacher. This phenomenon is counterintuitive, and a more thorough analysis of why a lower-scoring model makes a better teacher could offer valuable insights for future work.

W4: While it is understandable that some experimental results are placed in the appendix due to space limitations, the paper frequently has a significant separation between the presentation of results and their corresponding analysis. This disjointed structure makes the paper difficult to read.

### Questions
See above.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a large-scale empirical study aimed at identifying effective data recipes for improving LLM reasoning performance. The authors systematically analyze the impact of various data curation strategies, including question sourcing, filtering, deduplication, teacher selection, and dataset scaling. Many controlled experiments are conducted to evaluate how different design choices affect model reasoning ability under supervised fine-tuning. The paper also introduces an open dataset and provides detailed ablations to ensure reproducibility and transparency.

### Strengths
- The paper constructs a large and well-curated dataset specifically designed to enhance LLM reasoning capabilities. This contribution is practically meaningful, as data quality and composition have become increasingly crucial for reasoning-oriented LLM development. The released dataset and accompanying analysis provide a useful foundation for future research in reasoning and instruction tuning.

- The empirical study is extensive and carefully executed. The authors evaluate a wide range of data curation factors, such as filtering, question mixing, deduplication, teacher selection, and dataset scaling. The conclusions are supported with detailed ablation studies. The experiments are systematic and transparent, offering clear evidence for each design choice and contributing valuable insights for the broader community.

### Weaknesses
- This work feels closer to a technical report than a research paper. The study is solid and well-executed, but it lacks significant technical novelty. The proposed pipeline is straightforward and largely follows existing practices in data curation and reasoning dataset construction, with limited conceptual innovation.

- The proposed dataset provides only marginal improvements over existing baselines. As shown in Table 1, the average performance gain over previous datasets is relatively small (around 2.1), which raises questions about the practical significance of the proposed data recipes. While the results are consistent and reproducible, the improvement is quite modest and may not justify the large experimental effort.

### Questions
I do not have specific questions for the authors.

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
4

### Summary
This paper presents OpenThoughts3, a scalable and systematic data curation pipeline for building reasoning models through supervised fine-tuning. The authors carefully analyze each step of the data generation process—question sourcing, filtering, deduplication, answer sampling, and teacher selection—through more than 1,000 controlled experiments, resulting in a high-quality open dataset of 1.2M examples. The resulting model, OpenThinker3-7B, achieves state-of-the-art results among open-data models, surpassing comparable baselines like DeepSeek-R1-Distill-7B and Nemotron-Nano-8B across math, code, and science benchmarks. The paper emphasizes that careful dataset design can rival or exceed RL-based approaches, and provides a reproducible open-source recipe for training reasoning models.

### Strengths
1.	The work excels in its thorough dissection of the data curation process, covering multiple strategies at every pipeline stage. This level of experimental rigor goes beyond most existing work, making the conclusions highly credible and reproducible.
2.	By scaling the dataset to 1.2M examples and carefully selecting teacher models, the authors deliver competitive results on multiple reasoning benchmarks. The scaling curves and ablation studies clearly demonstrate the effectiveness of their approach.
3.	Unlike many proprietary reasoning models, the authors commit to full open-source release of datasets and models. This can significantly accelerate community research and lower the entry barrier for reasoning model development.

### Weaknesses
1.	I appreciate the design of the SFT data pipeline, but it’s hard not to notice how the entire work leans almost exclusively on this single training paradigm. In recent reasoning models, RL or curriculum strategies often play a big role in pushing performance further. Even if the authors didn’t run those experiments, a more thoughtful discussion or positioning would have made the contribution feel less one-dimensional.
2.	I’m left unsure whether the gains translate to broader reasoning capabilities. A few results on less standard or more language-heavy tasks would help a lot here. Right now, the claims around generalization feel more suggestive than demonstrated.
3.	The paper runs a huge number of experiments, but says very little about why certain design choices actually help. For instance, why does a relatively simple teacher mix outperform a seemingly stronger one? What kinds of examples drive the improvements? Without some interpretability or qualitative perspective, the work risks feeling like an “empirical recipe” rather than a deeper contribution.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces OpenThoughts2-1M and OpenThinker2-32B, the first publicly available open source dataset and model for reasoning tasks that match the performance of DeepSeek-R1-Distill-32B on AIME and LiveCodeBench. They also perform a series of 1000+ experiments to systematically improve their data generation pipeline to develop OpenThoughts3, which, combined with scaling to 1.2M examples and using QwQ-3B, yields OpenThinker3-7B, which vastly outperforms DeepSeek-R1-Distill-Qwen-7B on AIME, LiveCodeBench, and GPQA. Key findings include that sampling multiple answers from a teacher model is an effective strategy to scale the size of the training data, models with better performance are not necessarily better teachers, verification and answer filtering methods don’t lead to significant performance improvements, data quality trumps data diversity, and filtering questions based on LLM labeled difficulty or response length yields better results than using embedding based filters typical to pre-training.

### Strengths
1. Well-motivated and carefully designed experiments leading to clear takeaways about best practices for SFT training data design for reasoning models.
2. OpenThinker3-7B achieves the best average performance across all the evaluation benchmarks.
3. Efforts to decontaminate training data by removing samples with high similarity to the benchmark instances. 
4. Interesting results: Question sourcing (simple synthetic questions perform comparably or even better than complex or manually curated pipelines), Question filtering (difficulty filtering and response length filtering work well for code and math, compared to fasttext classifiers or embedding-based methods that work well for pre-training), Best teacher model is not necessarily the best performing model.

### Weaknesses
1. Should further investigate the impact of deduplication and sampling multiple answers because the results are very inconclusive, and exhibit too much variance across code, math, and science. It is also weird that the authors choose varying deduplication strategies across math, code, and science, but choose to pick x16 answer sampling per question for all domains, even though it is not the best across the board. This choice seems arbitrary, and the reasoning that it is better for scaling seems weird since training on much more data for essentially the same or worse performance is potentially just a waste of compute.
2. Again, the conclusions about answer filtering seem weird, since they bring up the point that training on all the data instead of filtering low-quality data makes no difference, so training on all the data is better for scaling. However, an alternative perspective would be that training on these low-quality instances doesn’t add to the performance, so why should one waste compute on training with these instances? In other words, answer filtering strategies could be used to find a smaller, more effective dataset that retains the same level of performance as the full dataset.
3. The difficulty filters used for answer filtering do not use any verifiers, like code execution with test suites, etc., which, while hard to scale, have shown a lot of promise in rejection fine-tuning-based pipelines (like RAFT) [1, 2, 3]. 
4. There is seemingly a big gap in performance between the best-performing OpenThoughts3-7B model and the teacher models used (QwQ, DeepSeekReasoner). While this is reasonable given the parameter size difference, the 10-point difference is pretty significant.
5. Some of the more intriguing results warrant further exploration, like why a less capable teacher model is a better teacher or why different question filtering strategies work for specific domains.

[1] Zheng, Kunhao, et al. "What Makes Large Language Models Reason in (Multi-Turn) Code Generation?." arXiv preprint arXiv:2410.08105 (2024).  
[2] Xiong, Wei, et al. "A minimalist approach to llm reasoning: from rejection sampling to reinforce." arXiv preprint arXiv:2504.11343 (2025).  
[3] Dong, Hanze, et al. "Raft: Reward ranked finetuning for generative foundation model alignment." arXiv preprint arXiv:2304.06767 (2023).

### Questions
Do you explore potential explanations of the QwQ being the best teacher model as an observation of the capacity gap phenomenon [1]?

[1] Zhang, Chen, et al. "Towards the law of capacity gap in distilling language models." arXiv preprint arXiv:2311.07052 (2023).

### Soundness
4

### Presentation
3

### Contribution
3
