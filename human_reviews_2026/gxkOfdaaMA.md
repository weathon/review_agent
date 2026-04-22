# DLER: Doing Length pEnalty Right — Incentivizing More Intelligence per Token via Reinforcement Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 8, 4

## Abstract
Reasoning language models such as OpenAI-o1, DeepSeek-R1, and Qwen achieve strong performance via extended chains of thought but often generate unnecessarily long outputs. Maximizing intelligence per token—accuracy relative to response length—remains an open problem. We revisit reinforcement learning (RL) with the simplest length penalty—truncation—and show that accuracy degradation arises not from the lack of sophisticated penalties but from inadequate RL optimization. We identify three key challenges—large bias in advantage estimation, entropy collapse, and sparse reward signal—and address them with $\textbf{D}\text{oing} \textbf{L}\text{ength} \text{p}\textbf{E}\text{nalty} \textbf{R}\text{ight}$ ($\textbf{DLER}$), a training recipe combining batch-wise reward normalization, higher clipping, dynamic sampling, and simple truncation length penalty. DLER achieves state-of-the-art accuracy–efficiency trade-offs, cutting output length by over 70\% while surpassing all previous baseline accuracy. It also improves test-time scaling: compared to DeepSeek-R1-7B, DLER-7B generates multiple concise responses in parallel with 28\% higher accuracy and lower latency. We further propose Difficulty-Aware DLER, which adaptively tightens truncation on easier questions for additional efficiency gains, and an update-selective merging method that preserves baseline accuracy  while retaining the concise reasoning ability of the DLER model which is useful for scenarios where RL training data is scarce.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This manuscript focuses on improving the reasoning efficiency of large reasoning models. The authors identify 3 major challenges including biased advantage estimation, entropy collapse and sparse rewards. To this end, the authors propose DLER, a RL training recipe for efficient reasoning capability. DLER consists a suite of tricks including reward shaping for length penalty, dynamic sampling, clip high and batch normalization for advantage estimation. Experiments on DeepSeek-R1-1.5B/7B and mathmetical benchmarks are carried out to evaluate the proposed training recipe.

### Strengths
- Reasoning efficiency is a widely acknowledged common challenge for large reasoning models. Thus the manuscript is well motivated and highly in-time.
- Most parts of this manuscripts are easy to follow up with. The organization and writting is clear.

### Weaknesses
- Novelty and contribution. From my reading of this manuscript, the contribution and novelty of DLER is hard for me to recognize. For instance, the majority of DLER, i.e. dynamic sampling, batch normalization of rewards, and clip high are proposed by DAPO. To my best knowledge, DAPO has been acccepted by RL community as the most commonly used baseline. Thus the novelty and contribution of DLER is very limited from my view. The manuscript seems to be a reproduction report of DAPO rather than a technical paper for possible publication on ICLR.

- Continue with my last point, the only newly proposed trick of DLER seems to be the truncation length penalty. However, a soft length penalty is also adopted in DAPO. There is no direct comparison between DLER and DAPO in the experiment section. I doubt if DLER can beat DAPO.
- Many other ``findings'' in this manucript are also not new for the reviewer. For example, there has been many previous works try to improve reasoning efficiency by model merging.
- The experiments are only conducted on DeepSeek-R1-1.5B/7B, which is relatively limited. The reviewer suggest more comparison on other model family such as Llama, OctoThinker or Mistral.
- The resulted DLER-R1-1.5B is trained on the same dataset with DeepScaleR-1.5B-Preview. However, the AIME24 accuracy is noticeably lower than DeepScaleR-1.5B-Preview (of which the pass@1 accuracy is 43.1), which raising further concerns about the effectiveness of DLER as well as limited baselines.

### Questions
- Could the authors further explain how the proposed DLER differ from DAPO in performance or method level?
- Could the authors compare DLER with DeepScaleR-1.5B-preview as these two models are trained on the same dataset.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents an integrated method to enhance reasoning efficiency in LLMs by focusing on the length of reasoning outputs. Existing long-to-short methods through RL often degrade accuracy due to suboptimal optimization strategies. The authors propose DLER, which optimizes RL with simple truncation penalties and batch-wise reward normalization, mitigating issues like bias in advantage estimation. Other widely adopted techniques such as clipping higher and dynamic sampling are also integrated. They demonstrate that DLER cuts output length by over 70% and achieves highest accuracy over other length penalty baselines.

### Strengths
1. **Well-considered integration:** The framework integrates length penalty, clipping regions, and dynamic sampling techniques, solving existing issues in RL. 
2. **Empirical significance and extensive experimentation:** Significant reductions in reasoning length (up to 70%) while maintaining or improving accuracy.
3. **Analysis on different length penalties:** Ablation studies include replacing different length penalties are clear and thorough.

### Weaknesses
1. **Lack of focus:** The method integrates a bundle of existing techniques in long-to-short RL without explicit focus. The analysis of different length penalties is most contributory to me, but is not the focus of the paper.
2. **Experimental result interpretation:** Some claims are at least exaggerated from the emprical results. For instance, the experiments in Sec.4.5 only show the different trade-offs and  some of the length penalty do have better "per token performance" than truncation; more rigorous experiments including adjusting differnt truncation lengths and penalty strengths are required for the current claim. Also Sec.4.6 is confusing since DLER does not lose much performance compared to the base model.
3. **Limited scope:** The paper only considers math problems and does not discuss further generalization.

### Questions
1. Why is Fig.4(c) has similar avg. response lengths at step 0 and step 400? And how do the savings of reasoning tokens establish under this circumstance?
2. How do you envision the objective of this area? Should the per-token performance or the best accuracy under whatever cost reduction be the focus?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper revisits reinforcement learning for length-efficient reasoning in LLMs. It argues that performance drops under length penalties come from suboptimal optimization rather than the penalty itself. Using simple truncation, the authors identify three causes—biased advantage estimation, entropy collapse, and sparse reward signals—and propose DLER, which combines batch-wise normalization, higher clipping, dynamic sampling, and truncation. DLER shortens responses by about 70% while maintaining or improving accuracy across math reasoning benchmarks. A difficulty-aware variant further reduces length, and a weight-merging method restores accuracy when training data are limited.

### Strengths
- Clear diagnosis of why RL with truncation fails and how each fix addresses it.
- Consistent gains across multiple datasets and model sizes.
- Useful weight-merging strategy for limited-data scenarios.

### Weaknesses
- Evaluations limited to math reasoning; unclear generalization to other non-math reasoning tasks.
- Difficulty-aware truncation may bias the training distribution toward easier samples, reducing exposure to genuinely hard problems that need longer reasoning.

### Questions
- What is the results of DLER on non-math or multimodal reasoning tasks?
- Does the normalization or dynamic sampling amplify certain reward trends?
- Could the difficulty-aware mechanism introduce potential biases or unintended drawbacks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes DLER, a training recipe for RL-based reasoning models aimed at shortening solution traces while maintaining accuracy. The recipe pairs a truncation-style length penalty with three stabilizers: batch-wise advantage normalization, a higher clipping threshold intended to preserve gradients for exploratory tokens, and dynamic sampling to avoid degenerate reward batches. Experiments on math-reasoning benchmarks report competitive accuracy with noticeably more concise reasoning trace and improved test-time efficiency.

### Strengths
- The empirical setup is described in a way that allows a reader to trace hyperparameters, datasets, and evaluation choices end to end. Ablations and reporting are organized so the effect of each component is inspectable. This clarity materially improves reproducibility and makes it easier to audit design choices and replicate results.
- The produced models achieve solid reasoning performance while reducing unnecessary verbosity in the generated traces. The improvements are presented consistently across tasks, indicating that the gains are not limited to a single benchmark. In short, the evidence suggests the method yields better performance per token without obvious accuracy collapse.

### Weaknesses
I appreciate the authors revisit a few techniques utilized in prior papers and package them into an effective recipe for training a reasoning model. However, the paper’s distinct technical contribution is hard to identify: the clip-high and dynamic sampling components follow DAPO, and batch-wise advantage normalization appears in Hu et al. (2025). In this work these elements largely behave as expected and are not clearly repurposed or extended for a new objective. Overall, this reads as a strong engineering consolidation rather than a substantive research advance.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
