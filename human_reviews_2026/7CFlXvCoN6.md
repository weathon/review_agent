# Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Off-policy reinforcement learning (RL) for large language models (LLMs) is attracting growing interest, driven by practical constraints in real-world applications, the complexity of LLM-RL infrastructure, and the need for further innovations of RL methodologies. While classic REINFORCE and its modern variants like Group Relative Policy Optimization (GRPO) are typically regarded as on-policy algorithms with limited tolerance of off-policyness, we present in this work a first-principles derivation for group-relative REINFORCE — a REINFORCE variant that uses the within-group mean reward as the baseline for advantage calculation — without assuming a specific training data distribution, showing that it admits a native off-policy interpretation. This perspective yields two general principles for adapting REINFORCE to truly off-policy settings: regularizing policy updates, and actively shaping the data distribution. Our analysis demystifies some myths about the roles of importance sampling and clipping in GRPO, unifies and reinterprets two recent algorithms — Online Policy Mirror Descent and Asymmetric REINFORCE — as regularized forms of the REINFORCE loss, and offers theoretical justification for seemingly heuristic data-weighting strategies. Our findings lead to actionable insights that are validated with extensive empirical studies, and open up new opportunities for principled algorithm design in off-policy RL for LLMs. Source code for this work is available at https://github.com/agentscope-ai/Trinity-RFT/tree/main/examples/rec_gsm8k.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a new perspective that REINFORCE-based algorithms, including GRPO, can be interpreted as inherently 'off-policy' algorithms with a solid theoretical foundation. Based on this off-policy interpretation, it proposes two core principles for stable learning: regularizing policy updates and actively shaping the data distribution. It frames previously disparate GRPO-like RL algorithms into a single, unified framework, proposing several new findings and using this interpretation to suggest a insight about how performance can be improved.

### Strengths
1. The paper's core contribution is its rigorous 'first-principles derivation' that recasts GRPO as a native off-policy algorithm. This reinterprets GRPO(and its friends) not as a on-policy method but as a principled 1-step gradient descent on a KL-regularized surrogate objective.

2. The paper successfully reinterprets and unifies recent algorithms like Kimi's OPMD and Meta's AsymRE. This demonstrates the framework's utility as an analytical tool beyond just GRPO, offering a new lens through which to understand a broader class of REINFORCE-based methods.

3. The paper provides a solid theoretical justification for previously heuristic data shaping techniques, such as "dropping negatives" (RED-DROP) or "up-weighting positives" (RED-WEIGHT). By showing that these established methods can be formally derived from their proposed loss, the authors successfully frame them as principled components of their unified theory, rather than just ad-hoc empirical tricks.

### Weaknesses
1. As the authors acknowledge, the paper proposes a novel interpretation but does not provide a theoretical bridge guaranteeing convergence to an optimal policy.


2. The proposal to widen the clipping range (e.g., to (0.6, 2.0)) is presented purely as an empirical finding without theoretical backing (unlike the data shaping principle). The paper provides no detailed analysis on why this wider range is effective, whether it relates to task properties, or how one might determine an optimal "loose" range etc. Even though, Appendix B.4 shows that widening the clipping range is different from just controlling learning rate, it still remains a task-specific hyperparameter that requires empirical tuning, rather than a generalizable principle.

### Questions
1. You claim that token-level IS is "non-essential," but your appendix figures demonstrate different aspects between NoIS and IS. Why do you think these phenomena have been observed?

2. Your paper argues that at the token level, clipping is important for stable training of GRPO[1] while IS is "non-essential." Given recent work like GSPO, which moves to a sequence-level framework, do you hypothesize that your findings(especially finding1) generalize? Would you predict that sequence-level IS can also be interpreted as non-essential in that framework under off-policy settings?

[1] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, & Junyang Lin. (2025). Group Sequence Policy Optimization.

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
3

### Summary
This paper presents a novel theoretical interpretation of Group-Relative REINFORCE (GRPO), showing that it naturally admits an off-policy formulation without assuming on-policy data. The authors derive this view from first principles, unify several recent algorithms (such as Kimi’s OPMD and Meta’s AsymRE) under the same framework, and support the theory with extensive experiments on reasoning and tool-use benchmarks.

### Strengths
- Provides a fresh and insightful theoretical reinterpretation of GRPO as an off-policy algorithm.
- Unifies several recent algorithms (e.g., OPMD, AsymRE) under a consistent framework.
- Writing is clear.

### Weaknesses
- Innovation lies mainly in conceptual reinterpretation, rather than proposing new algorithms.
- The relation to DPO, Mirror Descent, and other gradient-based RL theories could be expanded.

### Questions
- I think the writing style of this paper is quite different from that of typical research papers. Could the authors provide a more systematic discussion of prior work built upon this foundation and establish clearer connections to previous methods?
- The paper mainly analyzes existing group-based algorithms from a mechanistic perspective. If the authors are unable to design more effective and efficient algorithms based on their proposed theory, then the contribution of this work appears to be overstated.

### Soundness
3

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
This paper argues that popular reinforcement learning (RL) algorithms for LLMs, like Group Relative Policy Optimization (GRPO), are mistakenly treated as on-policy methods. The authors provide a new theoretical derivation showing that group-relative REINFORCE (the basis for GRPO) is "secretly an off-policy algorithm".

### Strengths
The paper's main contribution is a new theoretical framework that re-interprets group-relative REINFORCE algorithms, such as GRPO, as natively off-policy. While these methods are typically viewed as on-policy algorithms that can only tolerate small deviations from the data-generating policy, this work provides a "first-principles derivation" that does not assume a specific training data distribution.

### Weaknesses
1. The analysis in sec 2.2 is novel and interesting, but as they stated in the paper, this deduction needs the surrogate objective in (11) holds. However, this objective normally does not hold in reality since the policy only takes gradient descent for each sample and won't achieve at the optimal point. Hence, the clear conditions of when the derivation will make one gradient descent improve should be rigorously proved as in TRPO [1].

Since TRPO provides rigorous proofs to discuss when importance sampling can make the algorithm converge, the work to disagree this approach should also prove convergence. Otherwise, the results are not convincing enough.

[1] John Schulman, et al, Trust Region Policy Optimization

2. The experimenters are not valid enough. In Figure 2, they only do experiments for 1.5b model and only one group of sync parameters. They are suggested to compare more base models and more different sync_interval and offset parameters to validate their results.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

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
The authors frame a GRPO variant as an off-policy algorithm, and use it to justify and unify several recent empirical heuristics under different themes.

### Strengths
(+) The exposition of this paper was crystal-clear, well done! I appreciated the structuring and how clearly key results were called out. 

(+) It feels like every day on arXiv, we see a new GRPO variant. I appreciate the author's efforts to bring some clarity to this fast-moving space.

### Weaknesses
(-) A couple citations are missing. I'd suggest adding in one for MaxEnt IRL above Eq. 4, and to https://arxiv.org/abs/2406.01462 on line 176 re: coverage conditions for the unique minimizer to be the soft-optimal policy. Up to rescaling, I believe the square loss (Eq. 6) is the same as the loss function defined in https://arxiv.org/abs/2404.16767, so I would also add in a reference to that. I might also suggest adding in a reference to https://arxiv.org/abs/2505.20686 in Sec. 4.2.

(-) I find the characterization of REINFORCE as "inherently off-policy" a bit strange. I believe what Sec. 2.2 actually proved is that given off-policy data, performing a single step of square-loss minimization leads to group-relative REINFORCE. Whether this data was generated on- or off-policy is an entirely orthogonal point as far as I can see. Let me know if I've misunderstood things.

(-) It would be good to clarify early on in this paper that group-relative REINFORCE and GRPO refer to different things -- I believe the former doesn't include the standard deviation term in the advantage estimate?

(-) I'm not sure how the theory implies that the square loss penalty in Sec. 4.2 is "principled." Similarly, I don't understand how the theory implies that randomly up-weighting samples is "well-justified."

### Questions
(1) In https://arxiv.org/abs/2404.16767, authors analyze a similar square-loss objective and note that taking a single-step of Gauss Newton recovers the Natural Policy Gradient. They also discuss the variance-reduction benefit of using other samples in the same "group" as a baseline. Could you comment on the similarities and differences between your work and theirs? Also, I believe the REBEL algorithm in that paper is precisely computing the $\tilde{\theta}s$ in Figure 1 -- could you confirm this?

(2) Can you spell out the point of Sec. 4.1's experiments more explicitly? Also, re: AsymRE, it might be good to explicitly write out how this shift effects the closed-form solution -- I think it should boil down to a change in $\tau$.

### Soundness
2

### Presentation
4

### Contribution
2
