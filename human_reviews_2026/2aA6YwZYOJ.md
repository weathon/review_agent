# AgenticMath: Enhancing LLM Reasoning via Agentic-based Math Data Generation

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 2

## Abstract
The creation of high-quality datasets to improve Large Language Model (LLM) reasoning remains a significant challenge, as current methods often suffer from generating low-quality/incorrect answers and limited information richness from available data sources. To address this, we propose AgenticMath, a novel agentic pipeline for generating high-quality mathematical question-answer pairs to enhance the supervised fine-tuning of LLMs. Our method operates through four stages: (1) Seed Question Filter that selects questions with high information richness, complexity, and clarity; (2) an Agentic Question Rephrase step that employs a multi-agent system to generate diverse, logically consistent paraphrases; (3) an Answer Augment step where rewrite answers using chain-of-thought reasoning to enhance numerical and logical correctness, without reliance on human-provided labels; and (4) a final Question and Answer Evaluation that retains only the most superior pairs. Extensive experiments demonstrate that, fine-tuning 3B-8B parameter LLMs on AgenticMath generated datasets (comprising only 30-60K math samples) achieves competitive or superior performance on diverse in domain and out-of-domain mathematical reasoning benchmarks compared to baselines trained on much more data (e.g., 400K or 2.3M samples). Our work demonstrates that targeted, high-quality data generation is a more efficient path to improving mathematical reasoning in LLMs than large-scale, low-quality alternatives.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Large language models can achieve significant performance improvements by learning from high-quality synthetic data. 

Based on this conclusion, the paper proposes an agent-based method to generate training data related to math tasks. Specifically, the method first filters problems, removing low-quality or overly simple ones. Then, it rewrites the problem descriptions to standardize the expressions. Next, a large model is used to generate solutions for the problems. Finally, both the problems and solutions are evaluated to determine the final training data. 

By training with this synthetic data, the model achieves improved performance on math tasks. The authors also provide further analysis of the characteristics of the algorithm.

### Strengths
+ The authors were able to successfully generate training data related to math tasks for the model's instruction fine-tuning process.

+ The model was trained on synthetic data and achieved improvements.

### Weaknesses
+ This method of data synthesis is highly similar to previous works such as JiuZhang 3.0, Dart-Math, ScaleQuest, and MAmmoTH2.0, and it does not show significant improvement in training results. The contribution of this paper is very limited.

+ This paper uses the concept of an Agent to package the method, but in reality, it does not utilize Agent-related features such as memory. The method has little to do with Agents.

+ The evaluation is unreliable. Both the in-domain and out-of-domain tasks are math tasks, which is an unreasonable setup. The base model has weak capabilities, making the experiments unconvincing. The performance in the ablation study is unstable, so it cannot demonstrate whether the process is effective.

### Questions
+ Please further explain the contribution of this paper and describe how it differs from previous works.

+ Please use a stronger base model, such as Qwen2.5-7B, Qwen2.5-7B-Math, Llama3.2, or Gemma 3.

+ The experimental results in the ablation study may be caused by task fluctuations. Please further explain the effectiveness of each stage in the method.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an agentic pipeline for generating high-quality mathematical question–answer pairs. The pipeline consists of four stages: filtering, rephrasing, augmentation, and evaluation, resulting in the construction of the AgenticMathQA dataset. Fine-tuning 3–8B LLMs on this dataset achieves competitive or superior performance on a variety of in-domain and out-of-domain mathematical reasoning benchmarks, outperforming baselines trained on substantially larger datasets.

### Strengths
1. This work proposed a clear agentic pipeline to construct high-quality reasoning corpora.
2. Fine-tuning on the AgenticMathQA dataset outperforms baselines trained on larger datasets.

### Weaknesses
1. The four-stage pipeline, which involves filtering, synthesizing, refining mathematical problems and solutions, and evaluation, appears rather conventional and straightforward, lacking clear novelty.
2. The overall quality of the dataset is determined by the scores assigned at each stage of the pipeline. However, since the score-based filtering relies entirely on human-designed priors, the process is overly heuristic and lacks robustness.
3. The framework is essentially a form of data distillation from agentic models. In this work, all agentic models are based on GPT-4o-mini (2024-07-18), yet the experimental results do not report the performance of GPT-4o-mini itself.

### Questions
1. During the Problem Rephrase stage, is it necessary to ensure that the answers remain unchanged? Additionally, how is the correctness of the synthetic problems and solutions guaranteed?
2. How is the diversity of the synthetic problems ensured, and is it possible to observe the semantic distribution between seed problems and synthetic problems?  
3. The constructed AgenticMathQA has only been shown to be effective in the SFT stage; its usefulness for RLVR remains unclear. This assumes that the questions and answers in AgenticMathQA are completely correct.
4. In lines 301–303, for the 30K setting, the final dataset consists of 15K seed problems and 15K AgenticMath-synthesized problems. This indicates that no filtering was applied to the seed problems, as the total training data for MATH and GSM8K is roughly 15K.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a multi-agent framework designed to generate high-quality mathematical datasets that significantly improve the reasoning abilities of large language models (LLMs). The proposed system, AgenticMath, operates through four coordinated stages—filtering, rephrasing, solving, and evaluation—to ensure logical coherence, precision, and diversity in synthetic question–answer pairs. Using agents to assess and refine problem complexity and correctness, the method prioritizes data quality over scale. Experimental results across several benchmarks (GSM8K, MATH, CollegeMath, DeepMind Mathematics, OlympiadBench, and TheoremQA) show that models fine-tuned on only 30K–60K AgenticMath samples outperform or match baselines trained on hundreds of thousands to millions of examples. The findings highlight that rigorous, agentic data curation is a more efficient path to improving mathematical reasoning in LLMs than simply expanding dataset size.

### Strengths
- AgenticMath at 30K–60K samples consistently matches or beats baselines across Qwen2.5-3B, DeepSeekMath-7B, Mistral-7B, and Llama3-8B,  trained on hundreds of thousands to millions of examples, demonstrating high efficiency.
- Comprehensive analysis. The paper quantifies incremental gains from each stage: solution augmentation delivers the biggest single jump, while filtering, rephrasing, and review/revise provide additive improvements.

### Weaknesses
- The seed filter threshold score τ=3 seems not to be optimal according to Table 4. Other key hyperparameters ( review threshold τrev=4.5, max three review–revise iterations) lack ablation studies.
- Both the problem proposer and evaluator are GPT-4o-mini, which may create self-judging biases.
- The appendix lacks substantial details. It should include more design details (e.g., prompt tuning), ablations on threshold values, and ablations on the implementation of “long-tail” diversity selector (e.g., why not use random/clustered selection methods). I would also expect an analysis of the pipeline's robustness to agent model choice, e.g., swapping gpt-4o-mini to open models for better reproducibility.

### Questions
- How do you choose the one-shot exemplar in stage 3?
- I feel the "problem quality" can be quantified better here. Did you do any analysis of the difficulty distribution, topic distribution, and required skills of the generated data?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes a novel agentic pipeline for generating high quality synthetic data for math reasoning in LLMs. The pipeline consists of four stages: (1) An existing dataset of seed questions is filtered based on the quality of the questions (2) the seed questions are rephrased into diverse variants (3) the solutions (potentially multiple per question, depending on the desired size) of the new as well as the old questions are rewritten using an LLM (4) the final data is further filtered according to the quality of the problems, retaining only the most high quality ones. The resulting dataset of only 30k-90k questions , when used for finetuning various base models, outperforms (on an average across 6 tasks) baseline synthetic data generation approaches in data-sized matched as well as settings where the baseline approaches have upto 2.3M examples.

### Strengths
* The method is simple and results in very sample efficient synthetic data 
* The use of DS$^2$ score curation method in the context of synthetic data generation is novel and interesting
* I liked the ablation studies included in Section 4.3

### Weaknesses
* Clarity of writing in Section 3.5 could be improved (specifically in Lines 288-291)
* My biggest concern Is that comparisons with MetaMath, Dart-MATH etc. are not fair - because the question generation and more importantly, the solution generation models aren't the same. As evident from the results, most of the gains come from solution augmentation. MetaMath and DART-Math both use weaker teacher models to sample solutions. A fairer comparison would be using GPT-4o-mini as the generation model for both baselines (this can be done on a smaller subset, for eg. 10000 questions, for computational feasibility).
* Multiple important, more recent and stronger baselines seem to be missing from the experimental evaluation. Particularly, Personas based synthetic data generation [1] and  ScaleQuest [2] are relevant baselines (note that similarity of question generation models may not be ensured for ScaleQuest since it trains a special question generator).
* Several experimental details seem to be missing: Number of eval seeds, inference sampling temperature, inference max output length, etc. aren't mentioned

### References. 
[1] https://arxiv.org/abs/2406.20094. 
[2] https://arxiv.org/abs/2410.18693.

### Questions
* Can the authors provide a comparison of the the quality of pure synthetic question with the original set of seed questions (solutions can be re-written bby GPT-4o-mini in both cases)?
* Line 241: How was the revise threshold \tau = 4.5 arrived upon
* What are the statistics of other stages?
* Lines 145-146: What do the authors mean by "eliminating ground truth labels"? 
* Line 161 - What do the authors mean by "label-free filtering method?
* On line 89, would  "we evaluate models finetuned on data generated by agentic math….." be more correct? Is AgenticMath the name of the dataset or the proposed pipeline?

### Other Comments. 
* Typos in Lines 135-137

### Soundness
2

### Presentation
2

### Contribution
3
