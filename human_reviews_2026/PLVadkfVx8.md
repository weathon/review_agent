# CASPO: Confidence-aware Step-wise Preference Optimization for Reliable Reasoning in Large Language Models

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Large language models (LLMs) have demonstrated strong performance on tasks requiring multi-step reasoning, from mathematical derivations to knowledge-intensive open-domain generation. However, even when LLMs produce correct final answers, their reasoning processes often involve uncertain or inconsistent steps, which makes them prone to failure when facing similar problems again. To address this issue, we introduce CASPO, a framework that incorporates step-wise confidence into both training and inference. During training, CASPO constructs confidence-filtered preference pairs that capture both correct but low-confidence predictions and incorrect predictions, and optimizes them through iterative Direct Preference Optimization. During inference, we propose Confidence-aware Thought (CaT) strategy that prunes low-confidence reasoning trajectories to enhance reliability. Experiments on 10 reasoning benchmarks and across diverse model families show that CASPO yields improvements in both step-wise faithfulness and final-answer accuracy. We also release a step-wise dataset with confidence annotations to facilitate fine-grained analysis of model reasoning and expose hidden inconsistencies in existing benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces CASPO, a framework that incorporates step-wise confidence into both training and inference. During training, CASPO constructs confidence-filtered preference pairs that capture both correct but low-confidence predictions and incorrect predictions, and optimizes them through iterative Direct Preference Optimization. During inference, they propose Confidence-aware Thought (CaT) strategy that prunes low-confidence reasoning trajectories to enhance reliability.

### Strengths
1. They performed comprehensive analysis of the token length and reasoning pattern evolution, which provides a clear picture of how the models’ output evolved.
2. The writing is clear and easy to follow
3. They provided the code, improving the reproducibility of the work

### Weaknesses
1. Confusing choice of base models. In Section 4.1 the authors state that they use Qwen2.5-7B-Base, yet the tables report results only for Qwen2.5-7B-Math and Qwen2.5-7B-Instruct. Qwen2.5-7B-Math appears to be the math-tuned base model, whereas Qwen2.5-7B-Instruct is the instruction-tuned model. Why was Qwen2.5-7B-Math-Instruct not evaluated?

2. Qwen2.5-7B-Math-Instruct attains a MATH accuracy of 85.2, a maj@8 accuracy of 89.9, and an rm@8 accuracy of 91.4—substantially higher than the scores reported for the proposed method. Consequently, the effectiveness, scalability, and necessity of the method are questionable, given that the trained 7 B model fails to surpass the model used for annotation.

3. Tree-search-based iterative evolution has already been explored in prior work with far greater efficiency. For instance, [1] employed MCTS to bootstrap its own accuracy and achieved 90.0 on MATH500 with Qwen2.5-Math-7B. In comparison, the paper’s approach seems neither novel nor efficient.

[1] Guan, Xinyu, et al. "rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking." arXiv preprint arXiv:2501.04519 (2025).

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes CASPO (Confidence-Aware Step-wise Preference Optimization) to make multi-step reasoning both more faithful and more accurate. The key idea is to inject a calibrated, step-level confidence signal into both training and inference. During training, the authors build step-wise preference pairs that explicitly capture two failure modes—“correct but low-confidence” steps and “confidently wrong” steps. During inference, they introduce Confidence-aware Thought (CaT): a search over reasoning paths where expansion/pruning is governed by cumulative step confidence; paths whose cumulative confidence falls below a threshold are pruned early. Experimental results show strong performance across ten reasoning benchmarks and several model families.

### Strengths
1. CASPO integrates confidence at data construction, optimization, and search—offering a coherent story that targets the common gap between getting a correct final answer and maintaining faithful intermediate steps.

2. The cumulative-confidence pruning threshold formalizes a simple, tunable mechanism that improves the reliability of search without heavyweight external reward models.

3. Results span multiple benchmarks and base models, and the paper claims consistent gains in both accuracy and step-wise faithfulness.

4. Code is available in the anonymous link.

### Weaknesses
1. Step correctness relies on an external model. The sensitivity of CASPO to evaluator errors/biases and cross-domain applicability beyond math-style steps are not fully quantified.

2. CaT introduces a confidence-guided search; however, the paper could better characterize latency/compute overheads versus simpler decoders or self-consistency baselines at comparable budgets.

3. The narrative generalizes to “ten reasoning benchmarks,” but many examples and the evaluator choice skew toward math-style reasoning; broader coverage (code, scientific QA, multi-hop open-domain) would reinforce generality claims.

### Questions
Please see the weaknesses.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper "CASPO: Confidence-Aware Step-Wise Preference Optimization for Reliable Reasoning in Large Language Models" introduces a unified framework that enhances the reliability and faithfulness of reasoning in large language models (LLMs). CASPO integrates step-wise confidence modeling into both training and inference. During training, it constructs confidence-filtered preference pairs—capturing both correct-but-uncertain and confidently incorrect steps—and optimizes them using iterative Direct Preference Optimization (DPO). During inference, the proposed Confidence-aware Thought (CaT) strategy prunes low-confidence reasoning trajectories to ensure reliable reasoning paths. Experiments across ten reasoning benchmarks show consistent improvements in both step-wise correctness and final-answer accuracy, with strong generalization to out-of-domain tasks. The paper also releases a step-wise dataset annotated with confidence scores for further analysis. Overall, CASPO represents a principled approach to aligning confidence with correctness, promoting more trustworthy and interpretable multi-step reasoning in LLMs.

### Strengths
1. The paper is clearly written and well-organized, with intuitive explanations and helpful visualizations (e.g., the CASPO pipeline). The step-wise construction and inference design are presented in a manner which is easily understandable.

2. Experimentation as well as implementation details are well documented, which makes reproducibility of the approach easier. 

3. Generalisation to non-math benchmarks even if the model is trained on math benchmarks makes the approach promising.

### Weaknesses
1. The proposed framework CASPO relies on an external model, i.e., Qwen2.5-Math-7B-Instruct to verify the correctness of each reasoning step. This dependence on these LLMs as evaluators or judge might lead to incorrect reasoning steps being flagged as correct and influencing the training dynamics towards incorrect reasoning. A qualitative evaluation and quantitative evaluation (correlation with human judgement) of the judge will reflect the reliability of the data curated for DPO. 

2. The results show that CASPO’s performance plateaus after the second iteration of DPO training. This behavior is a known limitation of offline RL–style optimization (such as DPO), where the fixed dataset constrains policy improvement. An online or on-policy variant of DPO could mitigate this saturation and reduce the need for multiple iterative training stages.

3. While the reported results across model families suggest generality, the paper does not explicitly analyze whether CASPO’s effectiveness depends on model size. An ablation over smaller and larger models would clarify whether the gains arise from the method itself or from scale-related capacity effects, thereby strengthening the empirical claims.

4. There are a few papers that use this notion of preferential tuning and distillation for optimising rationales or step generation. A discussion around them in the related work and introduction should make the contribution of the work clear and distinctive. (https://arxiv.org/abs/2506.02519, https://arxiv.org/abs/2401.01335, https://arxiv.org/abs/2305.02301, https://arxiv.org/abs/2503.02463)

5. Minor grammatical error in line 400: The evolution of training accuracy and loss are closely mirror the reward dynamics...

### Questions
Look at the weaknesses above. I also have a few additional questions enumerated below. 

1. Was the model trained separately on the training set of each of the benchmarks using CASPO or on a combined dataset using the training set of all benchmarks?

2. What was the number of samples used to train the model for each benchmark. I suppose AIME-24 has just 30 training examples, which is too less for training the model and showcasing such performance. Can you please clarify a bit more on the training datasets used.?

3. Does CASPO ever over-prune reasoning paths that are low-confidence but actually correct? Has there been some analysis on these lines? Essentially, the reliability of the judge model used needs to analyzed explicitly through qualitative evaluation.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work aims to solve the task of mathematical reasoning. It introduces CASPO to incorporate step-wise confidence into both training and inference. It targets constructing a preference dataset for more effective step-wise DPO. Experiments show good results on various math benchmarks.

### Strengths
1. Writing is clear.

### Weaknesses
1. In line 161, the authors mentioned "sub question q_i", which is a little confusing. What's this? Do you mean the question plus the prior reasoning steps? Or something else? The authors should explicitly elaborate on this.

2. In the process of collecting data, the quality of so-called "confidence-aware" data actually heavily relies on the performance of the teacher model (in the paper setting, that is Qwen-Math, which is a strong open-source math model pretrained with enormous amount of math data). In my opinion, without a strong teacher model, judging a low-confidence but correct answer is extremely difficult. So I doubt whether the data quality is good enough, especially on hard problems.

3. In line 179, y_l is set by the second highest prob word. Is that a single word? Won't doing that cause low exploration?

4. In Table 1, the improvement seems limited.

5. Overall, the method is not systematic, and won't be used by the community.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2
