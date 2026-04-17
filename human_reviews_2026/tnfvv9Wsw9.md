# JURY-RL: Votes Propose, Proofs Dispose for Label-Free RLVR

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Reinforcement learning with verifiable rewards (RLVR) enhances the reasoning of large language models (LLMs), but its scalability is hampered by the high cost of human-annotated labels. Label-free alternatives, such as majority voting or LLM-as-a-judge, are susceptible to false positives that lead to reward hacking and training collapse. We introduce JURY-RL, a label-free RLVR framework that separates answer proposal from reward disposal: votes from model rollouts propose a consensus answer, while a formal theorem prover disposes the final reward. Specifically, a rollout is rewarded only if the majority-voted answer is formally verified by a Lean prover. When verification is inconclusive, we activate our proposed ResZero (Residual-Zero) reward: it drops the unverifiable majority proposal and assigns a zero-mean, variance-preserving reward to the remaining (residual) answers. This design maintains a stable optimization gradient for RL algorithms without reinforcing spurious consensus. Experiments across mathematical reasoning, code generation, and multi-task benchmarks show that JURY-RL consistently outperforms label-free baselines and attains performance comparable to supervised ground-truth training in pass@1, with superior generalization demonstrated by higher pass@k and response diversity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces JURY-RL, a framework for label-free RLVR. A formal Lean theorem prover is used to dispose the reward by verifying whether the proposed answer is provably correct. If verification succeeds, the model is rewarded for correctness. When verification is inconclusive, the paper proposes a ResZero fallback reward, which maintains a zero-mean, variance-preserving gradient signal to stabilize rewards. Experiments across different benchmarks show that JURY-RL improves stability over self-reward and LLM-judge baselines.

### Strengths
1. Using Lean as a proof gate bridges RLVR with formal methods.

2. The proposed zero-mean fallback reward stabilizes optimization even when verification fails.

3. The experiments are broad, covering multiple backbones and domains. JURY-RL consistently outperforms other label-free baselines.

### Weaknesses
1. The proposed pipeline uses a 32B auto-formalizer (8 candidates), a 32B consistency checker, and a 32B prover (16 proofs), making each rollout more expensive than LLM-as-Judge. Yet the performance gain is less than 3 points on average, questioning the cost–benefit balance.

2. Although Lean is rigorous, the upstream translation and proof-generation models may introduce false positives or negatives. The paper does not quantify conversion accuracy, proof validity rates, or error propagation.

3. The impact of each module (formalizer, checker, prover) is not isolated, making it unclear which contributes to the observed improvements.

### Questions
1. What is the total verification cost (e.g. GPU hours/ LLM calls per rollout) compared to LLM-as-Judge? Can you provide a more detailed cost-benefit curve or analysis, e.g., the impact of different autoformalization/ proof candidate number on RL performance?

2. Can you report the accuracy of auto-formalization and model generated Lean proofs? 

3. Can you justify why your method performance is better than the ground truth answer setting in table 1 and 2?

I will increase the score if all the questions are well-justified.

### Soundness
3

### Presentation
3

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
The submission proposes a hybrid of LLM-voting and label-free RLVR, where a committee proposes a derivation and then a formal theorem prover verifies it. When the proposed derivation cannot be verified, they fall back to a heuristic reward that achieves zero mean by assigning negative reward to the plurality answer and positive reward to other candidate answers. In this way it is possible to learn even from answers that cannot be verified.

Experiments show that this approach improves significantly over LLM-as-judge, and also outperforms RLVR from ground truth rewards in most cases. Ablations suggest that simply assigning zero reward to unverified derivations does not work, so the proposed approach is justified empirically. Experiments also demonstrate the stability of the method (figure 2), the ability to preserve answer diversity (figure 3), and the impact of the main hyperparameter (figure 4).

### Strengths
- The paper tackles an important problem: relaxing the need for human annotation while preserving the validity of the annotations.
- The proposed approach is mathematically sound and appears to be practical to implement
- The motivation of scalability, truth-alignment, and optimization-stability is persuasive
- The paper includes unusually comprehensive experimental validation, including ablations and robustness checks
- The worked example in B.2 is helpful for understanding.

### Weaknesses
It would be great to have more formal understanding of the ResZero reward. In particular, I don't have an intuition for why it is a good idea to  penalize the majority and amplifying the residuals in proportion to their frequency. It seems like this could even lead to an oscillatory behavior where two hypotheses alternate as the "majority" and top "residual".

Minor: "Majority" typically refers to >50%, but here I think what is meant is "plurality"

### Questions
- How is $\overline{u}$ computed?
- What happens in ResZero when all the samples are different answers? Do you pick a "majority" answer at random?
- Can you please formally express the proposition being proved in A.2? 
- Table 2 shows that assigning zero reward to unverified answers is generally less effective than ResZero. What about assigning a zero-mean random reward?
- How are the CIs computed for the averages in tables 1 & 2?

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
This paper introduce a novel label-free RL framework (JURY-RL) that leverage model-based Learn Prover system to assign psudo-label for majority responses. By leverage strong LLMs to parse rollouts and run multiple verification trials, JURY-RL achieves better reward F1 than LLM-as-a-Judge baseline. Moreover, the author propose a simple ResZero reward to assign valid learning signal for prompts which have non-conclusive majority response and found this simple reward improves both pass@1 and pass@k performance.

### Strengths
1. Evaluation results shows that JURY-RL gives better results than other label-free RL method baselines.
2. Using a Lean verifier to assign psudo-label gives good reward quality.
3. The proposed simple ResZero reward effectively improves pass@k and is easy to use.

### Weaknesses
1. Evaluation: Avg@k results are required on benchmarks like AIME24, AIME25, the reliability is questionable. For example, at least 16 trials are required for AIME24.
2. Lack important baseline like TTRL (NeurIPS 25').
3. Efficiency: The author adopt a Pass@K verification setting, might causing 400 sec overhead, while the LLM-as-a-Judge baseline might only introduce 10 sec if also only evaluate the majority. Even when the LLM-as-a-Judge method rewards each response, 400 sec overhead budge can afford much more larger models like Qwen3-235B-A22B. The author might have to give more analysis on this efficiency comparison to demonstrate the effectiveness.

### Questions
1. In Table 2, it is confusing that the Proof-Gate + MV setting  performs much worse on Qwen models compared with Majority-Voting baseline.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a type of hybrid reward that combines a verifiable reward with a label-free heuristic reward that is used when the output fails to be verified. Verifiable rewards (from a formal theorem prover) are highly reliable, but sparse. They can also be expensive to compute. Label-free heuristics (e.g., majority voting or LLM-as-a-judge) can give denser rewards, but are more unreliable. The "Votes Propose, Proofs Dispose" method proposed here attempts to leverage both of these by (a) only verifying the majority voted response, and then (b) if that fails, adding a fallback mechanism for assigning nonzero reward to the non-majority (unverified) responses based on a label free heuristic.

### Strengths
The problem the paper addresses is significant, and a key practical challenge for LLM post-training. The proposed idea of using a cheap heuristic to filter candidates before applying an expensive verifier is natural. The fallback reward is also simple + intuitive (encouraging exploration of the next-highest-voted outcomes) that is zero-mean with non-zero variance.

### Weaknesses
The main idea of hybridizing sparse rewards with denser, heuristic signals is natural and not particularly new. For example, process rewards achieve a similar goal, but are not discussed or compared to in the paper. Hybrid rewards in particular have also been explored before in Huang et al 2025. The empirical results are also not entirely compelling: in Table 1 in particular almost all of the results appear have confidence intervals that substantially overlap with other methods.

[1] Huang et al 2025. Pitfalls of Rule- and Model-based Verifiers – A Case Study on Mathematical Reasoning. https://arxiv.org/pdf/2505.22203v1

### Questions
- Not all of the baselines appear to be defined. What is CoReward referring to?
- The motivation for only validating the majority-voted answer is computational efficiency, but it's not clear exactly what is given up by this tradeoff. It would be nice to see how this method scales, for example, given a higher density of applied verified rewards (e.g. to top-k).

### Soundness
3

### Presentation
3

### Contribution
2
