# Efficient Inference with Large Reasoning Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Large reasoning models (LRMs) achieve state-of-the-art performance by generating long chains-of-thought, but often waste computation on redundant reasoning after the correct answer has already been reached. We introduce Early-Stopping for Token-Aware Reasoning (ESTAR) that detects and reduces such reasoning redundancy to improve efficiency without sacrificing accuracy. Our method combines (i) a trajectory-based classifier that identifies when reasoning can be safely stopped, (ii) supervised fine-tuning to teach LRMs to propose self-generated <stop> signals, and (iii) <stop>-aware reinforcement learning that truncates rollouts at self-generated stop points with compute-aware rewards. Experiments on four reasoning datasets show that ESTAR reduces reasoning length by about x3.7 (from 4799 to 1290) while preserving accuracy (74.9% vs. 74.2%), with strong cross-domain generalization. These results highlight early stopping as a simple yet powerful mechanism for improving reasoning efficiency in LRMs

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes ESTAR (Early-Stopping for Token-Aware Reasoning) to address the reasoning redundancy problem in Large Reasoning Models (LRMs). The main idea is to detect and reduce this redundancy using a LightGBM classifier. The tuning process includes SFT on a dataset with the <stop> token inserted, and the finetuned model is further trained using GRPO with a compute-aware reward function. ESTAR can largely reduce output tokens and maintain accuracy, providing a better accuracy-efficiency trade-off than other methods.

### Strengths
- This paper tackles a practical problem. Redundant reasoning is a major source of inefficiency in LRMs, and the goal of stopping early without sacrificing accuracy is a valuable insight.
- The multi-stage design of ESTAR is a strong point. Combining a lightweight classifier with a model trained to propose its own stop tokens is a clever way.
- The reported results are good. A 3.7x token reduction while maintaining 98.9% of the original accuracy would be a strong contribution.

### Weaknesses
- This paper mainly claims efficiency, but the experiments simply translate this to output tokens, which ignores the newly involved overhead. When using highly optimized inference engines like vllm or sglang, this method may break the workload, so I would like to see if there is end-to-end testing.
- The consistency and accuracy definitions may have problems and even bring contradiction in training. The LITE classifier and SFT focus on consistency, but the RL step is using an accuracy signal, which may conflict when the model's answer is wrong.
- The robustness of both training and evaluation is a huge concern. You mentioned temperature and top-p, but it seems only sampled once for both training and evaluation. Sampling can have a huge impact on reasoning output and even the final answer.
- As your method evolves through its 3 stages, there is a lack of ablations on how each component helps with the final result.
- The paper lacks details, especially for baseline methods. The repository you provided is also empty (as of 10/31/2025).

### Questions
- How does this method deal with batch scenarios?
- How do you deal with wrong answers, specifically in the LITE classifier step and the SFT step?
- Have you tried random sampling for the training data? How will different samples from the same question influence the result?
- Will the classifier be updated in the SFT and RL steps?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a novel three-stage approach called Early-Stopping for Token-Aware Reasoning (ESTAR) that detects and reduces reasoning redundancy without sacrificing accuracy for LRMS. The approach consists of: (1) Using a lightweight classifier (LightGBM) to identify optimal early-stopping positions in reasoning trajectories; (2) Training the LRM via Supervised Fine-Tuning (SFT) to autonomously generate `<stop>` signals based on these positions; and (3) Employing Reinforcement Learning (RL) to further refine the model's self-generation of these `<stop>` signals. Experimental results across four benchmarks indicate that ESTAR achieves a significant reduction in reasoning length (approximately 3.8x) while maintaining task accuracy.

### Strengths
1. The motivation is clear and the topic is well presented.

2. The proposed ESATR method is described in sufficient detail and appears technically sound.

3. The experimental results are strong, demonstrating the effectiveness of the proposed method.

### Weaknesses
### Major Concerns:

1. Overly General Title: The paper's title is too broad. "Efficient inference with LRMs" can be achieved through many different approaches (e.g., architectural changes). The current title does not accurately reflect the paper's contribution.

2. Empty Code Repository: The provided anonymous repository for code is currently empty.

3. Presentation and Formatting Issues: The core methodology sections (Preliminaries and Methods) suffer from presentation issues. The text appears "messy" and contains inconsistent formatting. Specific issues include:

- Awkward or unusual line breaks in the Preliminaries section.
- The use of non-standard markers, such as "(RQ1)", within the body of the Methods section, which seem out of place in a formal paper.
- Inconsistent formatting of mathematical equations between the Preliminaries and Methods sections.

4. Unclear Statements: Several statements in the paper are confusing or incomplete and require clarification:
- L123, ”... update ESTAR-LITE to stay aligned with the new trajectories.“
- L195, "... the tabular classifier also confirms the consistency of the model’s earlystop answer."
- L251, "Then we apply regular teacher ..."

### Minor Issues:

1. L121, ESTAR-FT is used before its formal definition.

2. L203, inconsistent notations vs. Eq. (1).

### Questions
The core methodology and the reported results are promising, and this work appears to be a valuable contribution to the community. However, the paper's current writing is a significant concern. In order to raise my rating, I would like the authors to address the different points in the major concerns.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work combines truncation-base and fine-tuning-base approaches for efficient reasoning.

### Strengths
- Efficient reasoning is one of the more important fields for LRMs.
- Testing consistency + accuracy is a good point. I am convinced.

### Weaknesses
The experiment execution seems to be on the weaker side.
- Only evaluated on two Qwen3 models.
- Baselines are mainly only featured in Table 2, but not all tasks and models. 
- Questionable decoding budget as indicated in Table 1—the vanilla decoding is often <5k.
- No end-to-end latency/throughput efficiency report.

Also, there should be more discussion and comparison of other efficient reasoning methods. Without digging too much, this work is already missing some key comparisons.
- o1-pruner is a well-recognized fine-tuning-based method for efficient reasoning.
- FlashThink, which is cited in the paper, is in fact not a binary (think or no think) method but a truncation one. Similarly, https://arxiv.org/abs/2506.02536 is another one. There are definitely many more early stopping methods applied to CoT, and they should be properly discussed and compared.
- AutoL2S also utilized the idea of incorporating special token for efficient reasoning, which should also be compared and discussed.

### Questions
- Which model is Table 2 testing?
- How many run per each question?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on improving the reasoning efficiency of large reasoning models (LRMs). First, ESTAR-LITE trains a classifier to predict whether reasoning should stop or continue. Then ESTAR-FT fine-tunes LRMs on curated CoT to enable them to determine their own stopping points. Finally, ESTAR adapts GRPO to reward correct "stop" emissions, resulting in a more efficient reasoning model.

### Strengths
-	The paper is well-written and easy to follow.
-	The three research questions (when to truncate reasoning, how to let LRMs decide stopping points, and how to leverage self-generated stop signals with reinforcement learning) are well articulated and explored.
-	Experimental results are promising, demonstrating substantial reductions in tokens while maintaining original accuracy levels.

### Weaknesses
- Limited evaluations. Experiments are conducted only on Qwen3-8B and Qwen3-14B. Evaluations on additional LRMs would strengthen the conclusions. The results in Tables 1 and 2 appear to be repeated. Moreover, Table 2 seems to include only the results of Qwen3-8B. It would be helpful to also compare different methods with Qwen3-14B and other models (e.g., DeepSeek-R1).
- Limited discussion of baseline methods (AdaptThink and Length-Penalty). It would be helpful to clarify whether these are state-of-the-art approaches and to justify their selection as baselines. It would also be beneficial to compare with additional baseline methods, such as [1, 2, 3].
- Insufficient evaluation of reasoning quality. While efficiency has been improved, it would be useful to assess whether shortened reasoning remains coherent and logically sound.
- Minor: Figure 3 is somewhat difficult to read.

[1] Yang, C., Si, Q., Duan, Y., Zhu, Z., Zhu, C., Li, Q., ... & Wang, W. (2025). Dynamic Early Exit in Reasoning Models. arXiv preprint arXiv:2504.15895.

[2] Chen, R., Zhang, Z., Hong, J., Kundu, S., & Wang, Z. (2025). Seal: Steerable reasoning calibration of large language models for free. arXiv preprint arXiv:2504.07986.

[3] Wang, C., Feng, Y., Chen, D., Chu, Z., Krishna, R., & Zhou, T. (2025). Wait, We Don't Need to" Wait"! Removing Thinking Tokens Improves Reasoning Efficiency. arXiv preprint arXiv:2506.08343.

### Questions
-	In Table 2, why are comparisons across different methods on the GPQA dataset missing?

### Soundness
2

### Presentation
3

### Contribution
2
