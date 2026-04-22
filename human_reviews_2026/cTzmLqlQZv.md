# SynAdapt: Learning Adaptive Reasoning in Large Language Models via Synthetic Continuous Chain-of-Thought

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 2, 6

## Abstract
While Chain-of-Thought (CoT) reasoning improves model performance, it incurs significant time costs due to the generation of discrete CoT tokens (DCoT). Continuous CoT (CCoT) offers a more efficient alternative, but existing CCoT methods are hampered by indirect fine-tuning, limited alignment, or inconsistent targets. To overcome these limitations, we propose ***SynAdapt***, an innovative efficient reasoning framework. Specifically, *SynAdapt* generates the synthetic CCoT to serve as a precise and effective alignment target for LLMs. This synthetic CCoT explicitly guides the LLM to learn CCoT and derive accurate answers directly. Furthermore, relying solely on CCoT is insufficient for solving hard questions. To address this, *SynAdapt* integrates a difficulty classifier that leverages both question context and CCoT to identify hard questions. CCoT can effectively help identify hard questions after some brief reasoning. We then adaptively prompt the LLM to re-think these hard questions for improved performance. Extensive experimental results across various benchmarks from different difficulty levels strongly demonstrate the effectiveness of our method, achieving the best accuracy-efficiency trade-off.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an innovative efficient reasoning framework for adaptive reasoning in LLMs via synthetic continuous CoT. The synthetic CCoT explicitly guides the LLM to learn CCoT and derive accurate answers directly. Further, the paper proposes a classifier to distinguish easy and hard queries. Finally, extensive experiments are conducted to verify the usefulness of the proposed method.

### Strengths
1. Adaptive reasoning is of practical use due to the need to save resources.
2. The paper is very well written and easy to follow.
3. Experiments have clearly demonstrated the capability of the proposed model in balancing effectiveness and efficiency for inference.

### Weaknesses
1. The proposed method is relatively weak in effectiveness. 
2. CCoT method lacks explainability, which deviates from explainable inference in daily use.
3. The classifier introduces an additional hyperparameter to control easy and hard queries, which is difficult to set.

### Questions
1. In my eyes, one of the major weakness in this paper is the deficiency in model accuracy. The strengths of CCoT lies in generating less tokens, which improves the efficiency. However, it harms the effectiveness, as shown in Table 1.
2. The classifier is used to predict easy and hard queries. However, how to set the threshold is difficult. In other words, how hard is hard and how easy is easy?
3. The generated synthetic CCoT is used to fine-tune LLM for better CCoT understanding, which is reasonable. But the generation of synthetic CCoT is separately optimized from fine-tuning. Hence, I am concerned about the problem that the generated $Z_{syn}$ could be in different subspace from $Z_{final}$, which leads to sub-optimal results.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces SynAdapt, a framework for efficient and adaptive reasoning in LLMs by generating and aligning synthetic continuous Chain-of-Thought (CCoT) representations. The authors propose synthesizing a continuous CoT to serve as an explicit and optimized alignment target, rather than relying on discrete or partial alignments as in prior work. SynAdapt also integrates a difficulty classifier that uses both the input question and its CCoT to identify challenging problems, adaptively prompting the LLM to re-think harder cases using discrete, step-wise reasoning.

### Strengths
1. The paper brings a new perspective to CCoT learning in LLMs, addressing weaknesses of prior partial or indirect alignment approaches. Specifically, SynAdapt’s use of synthetic, explicitly optimized CCoTs as full alignment targets for fine-tuning is clearly articulated and represents a concrete methodological advance.
2. The inclusion of an adaptive, CCoT-informed difficulty classifier is well-motivated and shows robust empirical performance over alternatives that rely on perplexity, prompting, or question-only signals.

### Weaknesses
1. Missing Discussion of Key Related Recent Works. [1] also proposed a module to classify the questions based on the questions’ complexity. 
[1] X. Chen, S. Zhou, K. Liang, and X. Liu, “Distilling reasoning ability from large language models with adaptive thinking,” IEEE Transactions on Neural Networks and Learning Systems, pp. 1–14, 2025.
2. Clarity and Interpretation of Mathematical Descriptions: 
- the motivation for aligning only the eot token’s hidden states is given for overfitting prevention, but why this is preferable to multi-token or more structured alignments is left vague.
- The reason why iteratively refine is needed is not clearly stated in the text.
- The hidden dimensions such as Zsyn are not marked, which greatly affects the reader's understanding.
3. The authors did not conduct experiments on the sensitivity of the length of the synthetic CCoT, denoted as m. In my opinion, the value of m should have a significant impact on the performance of CCoT. Moreover, since m is set to a fixed value, I wonder whether this is appropriate for chain-of-thought reasoning, whose information content can vary greatly across different problems.
4. Generalization beyond mathematical reasoning is claimed but not demonstrated. The method is never tested on tasks outside math QA
5. Using token count or sequence length as an efficiency metric is unfair, since the proposed method includes several preprocessing steps—such as iterative refine and difficulty estimation—before generating the chain of thought. Therefore, inference time would be a fairer basis for comparison.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an efficient reasoning framework that replaces long discrete Chain-of-Thought (DCoT) with a compact continuous CoT (CCoT) plus adaptive routing. First, it optimizes a synthetic CCoT per question while lightly aligning hidden states with the DCoT endpoint. Then it fine-tunes the LLM with LoRA to iteratively refine a draft CCoT to match the synthetic target, teaching the model to think in latent space without autoregressively emitting CoT tokens. A difficulty classifier, conditioned on the question and the refined CCoT, routes “hard” items to re-think with discrete CoT and lets “easy” ones answer directly from CCoT. Across math, scientific QA, and coding, SynAdapt reports shorter generations with competitive accuracy.

### Strengths
The authors implement wide experiments to evaluate the performance of their method.The proposed method is empirically shown to be effective in reasoning tasks in the author’s setting. The authors compare their method with various baselines on different datasets.

### Weaknesses
This paper introduces a quite complex framework based on different components proposed by past works, while it lacks really interesting or important insights/ findings.

The effectiveness of the framework may be questionable. Based on Table 1, when fully using CCoT, the performance is basically the same as directly prompting the model to give the output. The better performance in the accuracy-sensitive scenario is because the full CoT of the model is used. So this naturally questions why we ever need the proposed framework? A difficulty estimator is good enough if it can decide when to directly output the answer and when to use full CoT.

The experiments are only done on a single model. It remains unclear whether the framework is ad-hoc to that model.

There are many parameters to tune to make the framework work, for example, the number of iterations in refinement.  The refinement process also takes time, which is not taken into consideration for efficiency.

The paper implies good CCoT should be distilled step by step from DCoT (this assumption is not very intuitive and lacks evidence). However, the framework itself is not distilling each step from DCoT but last token (which is similar to CoDI). This is a bit against the advantages they claimed in their framework.  The motivation for Synthetic CCoT is not strong or maybe this is because of writing.

### Questions
How do you decide which baselines are for accuracy-sensitive
scenario or efficiency-sensitive scenario? The current division does not make sense to me. For example, tokenbudget is designed to be efficient.  Isn’t accuracy-sensitive scenario for reasoning models that take super long reasoning time to get superior performance? 

What is the length here? Is it the number of tokens?

What is difference between CoT-SFT and raw model? How do you do SFT here? 

What are some particular reasons for refinement? Is it better than directly optimizing  toward the target CCoT?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces SynAdapt, a framework that synthesises CCoT for a target reasoning LLM, where a adapter LLM is trained iteratively to predict a good CCoT needed for reasoning problem solving. The authors show high sequence length efficiency while preserving most accuracy.

### Strengths
1. Novel, tightly-coupled design: The three-stage pipeline (find optimal CCoT for LLM A → train LLM B to mimic it → deploy B+A) is novel and interesting.

2. The authors conduct extensive experiments to prove the superiority of SynAdapt against baselines.

### Weaknesses
> Compute cost is only partially accounted for

Training LLM B requires n iterations and each forward pass concatenates the full current CCoT (length m) with the question.  Complexity O(n*m) is paid during synthesis, yet Table 1 quotes only the final inference length.  If n≈4 and m≈512, the total FLOPs *before seeing a single test example can be already large.  A FLOPs count that includes the iterative stage is needed to argue for true efficiency.

> Performance concerns

Although high token efficiency, the accuracy drop in Table 1 is equally large. Since the “length” column only counts the final tokens for reasoning parts, any claimed trade-off between accuracy and sequence length is meaningless.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
