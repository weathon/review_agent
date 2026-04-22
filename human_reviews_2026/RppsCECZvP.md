# Don’t Waste Mistakes: Leveraging Negative RL-Groups via Confidence Reweighting

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Reinforcement learning with verifiable rewards (RLVR) has become a standard recipe for improving large language models (LLMs) on reasoning tasks, with Group Relative Policy Optimization (GRPO) widely used in practice. Yet GRPO wastes substantial compute on negative groups: groups in which no sampled response is correct yield zero advantage and thus no gradient. We ask whether negative groups can be leveraged without extra supervision. Starting from a maximum-likelihood (MLE) objective in reward modeling, we show that the MLE gradient is equivalent to a policy gradient for a modified value function. This value function adds a confidence-weighted penalty on incorrect responses, imposing larger penalties on more confident mistakes. We refer to this as **L**ikelihood **E**stimation with **N**egative **S**amples (**LENS**). LENS modifies GRPO to assign non-zero, confidence-dependent rewards to incorrect generations, making negative groups informative and converting previously wasted samples into useful gradient updates. On the MATH benchmark with Llama-3.1-8B and Qwen-2.5-3B, the proposed variant consistently outperforms GRPO baseline, with significant gains on harder items. These results demonstrate a principled and practical way to “rescue” negative groups, improving efficiency and performance in RLVR.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses sample inefficiencies in RLVR for reasoning tasks, particularly the GRPO framework. In GRPO, all positive and all negative groups yield zero gradients and thus waste compute.

The authors propose LENS (Likelihood Estimation with Negative Samples), which connects MLE in reward modeling with policy gradient methods. By reinterpreting MLE gradients, they derive a confidence-weighted penalty for incorrect generations: the more confident the model is in a wrong answer, the larger the penalty. This converts previously uninformative negative groups into useful training signals.

LENS is implemented as a drop-in modification to GRPO, introducing nonzero, confidence-dependent rewards for incorrect responses with minimal computational overhead. Experiments on MATH with LLaMA-3.1-8B and Qwen-2.5-3B show consistent performance gains across all Pass@k metrics, especially on harder problems (Levels 4–5).

Overall, the paper is good on the theoretical side and could be further improved on the experimental side. I hence evaluate it as a borderline reject at this stage. I would be happy to increase my evaluation if the experimental side is improved in the next version.

### Strengths
1. **Clear motivation and practical relevance.** The inefficiency of negative groups in GRPO is a well-recognized issue, and LENS directly tackles this without additional supervision.

2. **Theoretical derivation connecting MLE and policy gradients.** The reparameterization and gradient equivalence analysis (Eq. 8–10, Theorem 1) are elegant and make the proposed modification principled rather than heuristic.

3. **Simple and easily adoptable algorithm.** The proposed confidence-based correction term integrates naturally into existing GRPO setups (and potentially, many other variants), making it practical for real-world RLVR pipelines.

4. **The paper is well written and easy to follow.** Figures (especially Fig. 1 and Fig. 4) effectively illustrate how negative samples are “rescued” and how the modified rewards reshape the learning dynamics.

### Weaknesses
1. **Limited empirical scope.**
Experiments are restricted to MATH reasoning; no evidence is given for generalization to other RLVR tasks (e.g., code or symbolic reasoning) or at least, on other commonly used math datasets (e.g., AIME, AMC, Minerva, etc.). The claim of generality would be stronger with broader benchmarks.

2. **Absence of ablation and sensitivity studies, simple baseline.**
It’s unclear how sensitive performance is to the choice of alpha. In other words, do the benefits mainly come from the mixed group or the negative groups? Also, no comparison is made to simple baselines that also take advantage of the negative examples. (The paper mentioned some, but does not compare them.)

### Questions
1. Is the proposed reward shaping applied at the response level or the token level? In other words, do all tokens within a single response share the same effective learning signal (or equivalent learning rate), or is the penalty distributed across tokens differently?
This distinction is important because prior work [1] has shown that in a “negative” response, most intermediate reasoning tokens may still be correct, with only the final token being wrong. Penalizing the entire sequence uniformly might therefore suppress useful intermediate reasoning behaviors. Could the authors clarify how LENS handles this situation, specifically, whether it can assign token-wise negative gradients more selectively or whether the entire response receives a uniform penalty?

2. LENS takes good use of the all-negative groups, what about the all-positive group? Can the methodbe  extended to this case?

[1] Deng, Wenlong, et al. "On the Effect of Negative Gradient in Group Relative Deep Reinforcement Optimization." *arXiv preprint arXiv:2505.18830* (2025).

### Soundness
3

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
The paper proposes LENS, a modification to GRPO that introduces a confidence-dependent penalty for incorrect responses derived from an MLE reformulation. The authors argue that GRPO “wastes” negative groups (groups where all responses are incorrect) because their advantages collapse to zero. LENS uses the model's predicted probabilities to build a calibrated reward that assigns non-zero penalties for wrong answers.

### Strengths
1. Addresses a real inefficiency in GRPO: lack of signal in all-negative groups.

2. Attempts to reframe reward modeling and RLVR through an MLE lens.

### Weaknesses
## Weaknesses

1. **Limited Experimental Scope.**  
   The empirical evaluation covers only the MATH dataset and two model sizes. This limited scope is insufficient to support claims of generality or broad applicability across reasoning tasks or model scales.

2. **Unrealistic Assumptions About the Output Space.**  
   The theoretical formulation assumes a finite, enumerable answer space. In LLM reasoning, however, the number of textual realizations of the same correct answer (e.g., *“The answer is 42.”*, *“It is 42.”*, *“42.”*, etc.) is effectively unbounded. Consequently, \(D(q)\) may be infinite or undefined, making the proposed reparameterization inapplicable to real generative settings.

3. **Breakdown of the MLE/IS Derivation for Negative Groups.**  
   I was enjoying the MLE and importance-sampling derivation, but the analysis breaks down abruptly for negative groups (where all \( r_i = 0 \)). Instead of providing a principled extension of the theory for difficulty measure (such as mixed group), the paper introduces an ad-hoc fallback  
   \[
   D(q) = 2 \cdot \max_i \pi_{\text{old}}(o_i \mid q),
   \]  
   which has no theoretical justification. This weakens the claimed theoretical connection between reward modeling and policy optimization, especially in emphasizing resolving the learning signal from the incorrect group.

4. **Only Marginal Gains in Mixed Groups.**  
   Appendix Table 2 shows that LENS provides only minor improvements on mixed groups. This suggests that most of the observed empirical gains stem from the heuristic handling of all-negative groups, rather than from the theoretically grounded likelihood-based calibration intended for mixed groups, raising a concern about the validity of the theoretical framework.

5. **Missing Comparisons to Strong Negative-Reward Baselines.**  

   The paper does not compare LENS against simpler, well-established negative reinforcement strategies (e.g., penalizing all incorrect responses in all-negative groups). Prior work [1] demonstrates the strength of this method. Without these comparisons, it is unclear whether LENS provides meaningful advantages over existing methods.

[1] The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning,

### Questions
See weakness

### Soundness
3

### Presentation
2

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
The paper targets RLVR training with GRPO, arguing that “negative groups” (all sampled answers wrong) are wasted because GRPO assigns them zero advantage. The authors derive a calibrated policy gradient whose reward for incorrect samples is down-weighted by model confidence and an instance-level “difficulty” term. Their modified reward yields non-zero, confidence-dependent signals for wrong answers and they implement can simply swap GRPO’s reward, using length-normalized probabilities and a simple estimator for difficulty term. They claim negligible overhead and better Pass@k on MATH with Llama-3.1-8B and Qwen-2.5-3B, especially on harder problems.

### Strengths
- The paper identifies a intuitive and significant inefficiency in a widely used algorithm. The concept of "wasted mistakes" is clearly state and the paper provides strong, data-driven motivation in Figure 2. 1 is an excellent framing of the problem. 

- The authors provide theoretical insights to conceptually bridge from a reward modeling MLE objective to a policy gradient form, and the theoretical results provides a solid starting point for the subsequent algorithmic development.

### Weaknesses
- While the theoretical results are interesting, their connection to the proposed method feels weak. Some aspects of the design appear to be heuristic and lack clear justification (see details in the questions).

- The evaluation overlooks several important baselines, and some claims made in the paper are not adequately supported by evidence (see details in the questions).

### Questions
- Concurrent work (Xiong et al. 2025, arXiv:2504.11343) provides empirical evidence that discarding negative groups is GRPO's main advantage. Can the authors comment on this fundamental contradiction and justify why their premise holds in light of this conflicting evidence?

- The key difference between LENS and cited work (Zhu et al. 2025) is the stratification of negative rewards (confidence-weighted vs. uniform). Why was a uniform negative reward baseline (i.e., NSR from Zhu et al.) omitted from the ablation study in Table 2? 

- Section 5 introduces at least four major heuristics (length-norm, a complex $D(q)$ estimator, $1/G$ scaling, and an $\alpha=0.25$ hyperparameter) that do not appear in the " derivation in Section 4. Does the theoretical reward from Eq. 10 work in practice? If not, doesn't this imply the theoretical framework is a loose inspiration rather than a principled derivation?

- The paper claims “negligible computational overhead”. Can the authors please provide quantitative data?

- Can the authors provide a theoretical justification for the number 2 in the $D(q)$ estimator 1 (line 370) and explain why the estimator must be defined differently for mixed vs. negative groups?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on solving the issue of original GRPO objective that the negative groups contribute no gradient to the update. The authors propose a method called Likelihood Estimation with Negative Samples (LENS), which assigns non-zero advantage to responses within negative groups based on a confidence-weighted penalty. Experiments on two models verify the effectiveness of the proposed GRPO-variant objective.

### Strengths
- The paper provides a conceptually clear connection between reward modeling and policy learning.
- The derivations are rigorous, with clear assumptions and logical consistency.

### Weaknesses
- Can the authors prove that results in Table 1 are statistically significant (at least 2-sigma or 95% confidence level deviations)?
- Only one benchmark is evaluated in this paper, how does the methods generalize to more benchmarks (e.g., AIME, AMC, etc.)?

### Questions
1. The proposed method modifies the GRPO objective by introducing a confidence penalty to the negative groups, I'm wondering how this connects with the entropy/confidence-related methods like [1][2][3].
2. [4] shows that simply assigning -1 rewards to negative responses and dropping the correct ones can improve LLM reasoning, what if the proposed objective in this paper is applied to negative-only groups and dropping other samples? Will this be better than simply assigning -1 rewards?

[1] Learning to Reason without External Rewards

[2] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning

[3] Maximizing Confidence Alone Improves Reasoning

[4] The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning

### Soundness
3

### Presentation
3

### Contribution
2
