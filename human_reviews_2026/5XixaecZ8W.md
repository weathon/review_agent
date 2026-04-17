# SPECS: Faster Test-Time Scaling through Speculative Drafts and Dynamic Switching

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Scaling test-time compute has driven the recent advances in the reasoning capabilities of large language models (LLMs). However, increased compute often comes at the expense of higher user-facing latency, directly impacting user experience. Current test-time scaling methods primarily optimize for accuracy based on total compute resources (FLOPs), often overlooking latency constraints. To address this gap, we propose SPECS, a latency-aware test-time scaling method. SPECS builds upon beam search, which generates multiple reasoning traces for each step with a reasoning model, and selects one to continue from based on the scores from a dedicated reward model. Inspired by speculative decoding, SPECS uses a smaller, faster model to generate candidate traces efficiently, and evaluates these candidates with both the reasoning model and the reward model. We design novel strategies to select candidate drafts using these model evaluations, including reward-guided soft verification, and a dynamic switching mechanism to defer to the larger model on harder steps. Empirical results on MATH500, AMC23 and OlympiadBench datasets show that SPECS matches or surpasses the accuracy of beam search while reducing latency by up to $\sim$18\%. Our theoretical analysis shows that our algorithm converges to the solution of a KL-regularized reinforcement learning objective as the beam width grows.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes SPECS, a new test-time scaling algorithm for LLMs.
SPECS combines best-of-n sampling with guidance from a process reward model (PRM) alongside a few other ideas: using a draft model (similar to speculative decoding) to accelerate inference; *tilting* the reward by the log-density ratio between the target and draft model; and adding dynamic switching between generation from the draft and target model (depending on whether the problem is deemed difficult or easy, based on the rewards).
The authors provide an extensive theoretical framework for their algorithm, although much of the theory does not seem to be applicable to the algorithm in practice (see below). The authors validate the proposed algorithm with experiments on three datasets (AMC23, MATH500, OlympiadBench), which show that it performs better than beam search with the draft model and sometimes even with the target model, and reward-guided speculative decoding (RSD) [1], a recently proposed alternative algorithm with many similarities to SPECS.

As outlined below, I believe SPECS proposes some interesting algorithmic ideas, but the novelty of the contribution is limited, and the supporting theory is not directly applicable as it hinges on assumptions that do not hold in practice. **I lean towards rejection of the paper, but believe it could also be accepted.**

[1] Baohao Liao, Yuhui Xu, Hanze Dong, Junnan Li, Christof Monz, Silvio Savarese, Doyen Sahoo, and Caiming Xiong. Reward-Guided Speculative Decoding for Efficient LLM Reasoning. 2025. URL: https://arxiv.org/abs/2501.19324.

### Strengths
### Algorithm
The proposed algorithm is interesting and, as the authors highlight in the Appendix, unlike other methods from the literature like RSD [1], SPECS *starts* by decoding from the target model, and only falls back to the draft model if deemed advantageous (i.e., when the reward is high). This idea is novel and nice.

### Theory
The authors prove that under certain assumptions, SPECS converges to the optimal tilted policy as $n\to\infty$. However, these assumptions do not hold in practice (see below).

### Presentation
The paper is well written, structured, and easy to read.

### Experiments
The authors provide a reasonable amount of experiments to validate the performance of SPECS (although more datasets never hurt!). The results include confidence intervals (although as far as I see, the authors do not specify how these are computed, i.e. over how many runs and what degree of confidence). Furthermore, the authors provide ablations over the latency and the effect of the dynamic switching, which show that dynamically switching based on the rewards is better than switching randomly between target and draft model (with identical proportions of models). The Appendix contains additional experiments.


[1] Baohao Liao, Yuhui Xu, Hanze Dong, Junnan Li, Christof Monz, Silvio Savarese, Doyen Sahoo, and Caiming Xiong. Reward-Guided Speculative Decoding for Efficient LLM Reasoning. 2025. URL: https://arxiv.org/abs/2501.19324.

### Weaknesses
### Contribution
The contribution seems relatively limited to me. Large parts of the method have already been proposed in previous works: RSD [1] proposes combining a draft model with a reward model for stepwise speculative generation (although only for $n=1$). GSI [2] proposes using the same tilted rewards as SPECS. The main novelty in SPECS seems to be that generation *starts* with the target model, and can *dynamically switch* between draft and target model.

### Theory
Theorem 1 shows that SPECS can approximate the optimal tilted distribution. However, this theorem hinges on two crucial assumptions, neither of which holds in practice: (1) It assumes that the given PRM is "idealized", meaning it corresponds to the true value function of the  policy w.r.t. a reward model (Definition 1). (2) The number of drafts generated (which is fixed in practice) is assumed to be Poisson distributed.

Furthermore, it is not clear to me what "for reasoning problems over H blocks" means exactly. Does this assume that all steps have the same length? Also, I have not checked the proofs carefully, as they are extremely extensive.


[1] Baohao Liao, Yuhui Xu, Hanze Dong, Junnan Li, Christof Monz, Silvio Savarese, Doyen Sahoo, and Caiming Xiong. Reward-Guided Speculative Decoding for Efficient LLM Reasoning. 2025. URL: https://arxiv.org/abs/2501.19324.

[2] Jonathan Geuter, Youssef Mroueh, and David Alvarez-Melis. Guided Speculative Inference for Efficient Test-Time Alignment of LLMs. 2025. URL: https://arxiv.org/abs/2506.04118.

### Questions
- in Table 2, does "No Target" simply correspond to best-of-n sampling with the draft model? If so, it would help to clarify this (if it is something else, please also clarify).
- in the experiments, the authors say they compare to beam search. However, looking at Figure 1, I think what they actually compare against is the following baseline: In each step, create $n$ drafts, select the one with the highest reward, and use that as a starting point in the next step. (This is different from beam search, since in beam search, all beams can have different histories). In particular, comparing against this baseline seems much more reasonable than comparing against beam search. Could the authors clarify this, please?
- could the authors explain why they believe they see an improvement over the "beam search with the target model" baseline? If "beam search" is what I think it means (see above), I don't see why SPECS should ever perform better than beam search with the target model. Could this be attributed to the draft model actually performing better than the target model on these datasets or on certain samples?
- the authors say they analyze the algorithm with the idealized PRM "for the purpose of keeping the theoretical presentation cleanest". Does this mean a similar result also holds with any "out-of-the-box" PRM? If so, this should certainly be included in the paper. If not, the authors should clarify this.
- it would be interesting to see how SPECS scales beyond $n=16$

Note: In the appendix, line 721: not a complete sentence (probably a period instead of a comma)

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces SPECS, an inference framework to speed up reasoning for LLMs. Building on beam search and speculative decoding, SPECS uses a smaller model to draft candidate reasoning traces and a reward model to guide selection, dynamically switching to a larger model when needed. Experiments on math reasoning benchmarks show that SPECS achieves comparable or higher accuracy than beam search while reducing latency by up to 18%.

### Strengths
1. The paper is well-written and easy to follow.
2. The approach balanced between theory and practice, where the theoretically-rooted method also works well in practice, advancing the pareto frontier of accuracy and latency.
3. The experiments cover a wide range of models from different families.

### Weaknesses
1. As far as the reviewer understands, the practical difference between SPECS and RSD is that (1) SPECS integrates LLM logits for reward modeling, and (2) SPECS uses the large model in the initial step, whereas RSD uses the small model in the initial step. It remains questionable which of the two actually contributes to the accuracy gains. This is especially important, as the theory-grounded reward design is the main novelty of the paper.

2. The evaluation is limited to math benchmarks. It remains questionable if the approach will generalize to non-mathematics benchmarks, such as GPQA or coding benchmarks.

3. Some claims need to be toned down.

3-1. The authors claim that existing test-time scaling approaches usually optimize for total FLOPs, and SPECS optimizes for latency. However, many existing works on efficient test-time scaling focus on optimizing practical efficiency metrics, such as inference latency or number of generated tokens (which translates to decoding latency) [1][2]. This should be toned down throughout the paper.

3-2. Observation 1 in Section 2 is already well-known, especially given that it is the precise motivation of speculative decoding (which has already become a large field of study [3]). This needs to be toned down, as the current presentation suggests that it is a new observation made by the authors.

[1] Yang et. al., Dynamic Early Exit in Reasoning Models, preprint

[2] Yu et. al., Think Smarter not Harder: Adaptive Reasoning with Inference Aware Optimization, preprint

[3] Xia et al., Unlocking Efficiency in Large Language Model Inference:A Comprehensive Survey of Speculative Decoding, ACL 2024 Findings

### Questions
Will SPECS outperform RSD + using large model in the initial step? Also, will SPECS + using small model in the initial step outperform RSD? (This result will clearly present the practical effectiveness of the proposed scoring function.)

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
3

### Summary
This paper proposes SPECS, improving the latency-accuracy trade-off for test-time scaling in LLMs, particularly for reasoning tasks. It starts by generating from the target model and dynamically switches to speculative drafting for high-reward (from a Process Reward Model) traces to minimize accuracy drop. The two main components are a *soft verification* scoring function that combines the PRM score with a log-density ratio, and a *dynamic switching* mechanism that starts with the target model and only switches to the draft model once the PRM score exceeds a certain threshold. The authors claim this method matches or exceeds the accuracy of a large-model beam search while reducing latency by up to ~18% on math reasoning benchmarks.

### Strengths
1. Unlike traditional speculative decoding, SPECS starts with the more capable target model. It features a novel dynamic switching mechanism that transitions to the faster draft model only after a high-reward reasoning trace has been identified.
2. The insight that a draft model is sufficient for easier steps (identified by a high PRM score) and that the system should start with the target model is well-motivated. 
3. The authors provide a clear theoretical motivation for their method.

### Weaknesses
1. As shown in Table 3, the hyperparameters β (inverse-temperature for the soft verification) and τ ( dynamic switching threshold) are tuned for each specific dataset and beam width. This raises questions about the method's sensitivity to the choice of the parameters and the practical overhead of deploying it, as it would seem to require a new, exhaustive hyperparameter search for any new task. The paper provides no robustness analysis to these choices, justification for the specific numbers, or a simple heuristic for setting these parameters (why MATH500 and OlympiadBench use 100 questions and AMC23 uses the whole 40 questions). Furthermore, the tuning methodology itself is inconsistent: validation sets are small and inconsistent (100 questions for two datasets, but only 40 for AMC23), and selecting parameters that give the "best accuracy vs. latency" seems a flexible metric.  In addition, the authors state that "When we condition on the traces having high reward (as computed by a PRM) at the 8th step, we observe that completing the trace using the draft model affects accuracy very minimally compared to completing with the target model", which implies that the decision to switch seems to be example-dependent. Will a different example have a different dynamic switching threshold?

2. The entire dynamic switching mechanism depends on a reliable PRM. The paper uses a strong, existing PRM, but will the performance of SPECS degrade if the PRM is of lower quality, e.g. PRM confidently assigns a high score to an incorrect "easy" path? In addition, will using a different PRM fundamentally change the accuracy vs. latency trade-offs reported in the paper?

3. The method is only evaluated on mathematical reasoning, a highly structured domain where "correctness" is well-defined and PRMs are known to be effective. It remains unclear how SPECS would perform on other types of reasoning tasks (commonsense reasoning tasks), open-ended generation tasks (e.g., summarization), or other domains like code generation. 

4. There are also concerns about the presentation of the results in Table 1. The authors only bold the SPECS accuracy score even when the accuracy is the same as the baselines. For example, on AMC23 (n=4), BS(B) achieves 59.2±1.2 and SPECS achieves 59.2±3.1, but only the SPECS result is bolded. This can be misleading. Furthermore, many accuracy improvements are marginal. Are these improvements statistically significant? In addition, the authors selectively bold only accuracy, while latency and latency/step metrics (where SPECS is not always the best performer) are left unbolded.

### Questions
See in weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
