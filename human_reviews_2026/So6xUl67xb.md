# Rethinking Entropy Regularization in Large Reasoning Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 6, 2

## Abstract
Reinforcement learning with verifiable rewards (RLVR) has shown great promise in enhancing the reasoning abilities of large reasoning models (LRMs). However,  it suffers from a critical issue: entropy collapse and premature convergence. Naive entropy regularization, a common approach for encouraging exploration in the traditional RL literature, fails to address this problem in the context of LRM. Our analysis reveals that this failure stems from the vast action space and long trajectories in LRMs, which easily trigger a global entropy explosion as the model indiscriminately explores all possible actions and states. To address this, we propose ***SIREN*** (**S**elect**I**ve ent**R**opy r**E**gularizatio**N**), a method that confines exploration to a meaningful subset of actions and states. ***SIREN*** achieves this through a two-step entropy masking mechanism, consisting of a top-p mask and a peak-entropy mask. In addition, regularization is transformed into a self-anchored form to stabilize training. Across five mathematical benchmarks, ***SIREN*** attains superior average performance over previous entropy-related RLVR approaches, exemplified by a +6.6 maj@k improvement on AIME24/25 with Qwen2.5-Math-7B. Further analysis confirms that ***SIREN*** promotes greater response diversity and maintains entropy at an appropriate level, which helps to preserve the validation pass@k throughout training. This effectively mitigates the premature convergence problem common in RLVR for LRMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces SIREN (SelectIve entRopy rEgularizatioN), an entropy-regularization variant for reinforcement learning with verifier rewards (RLVR). The key idea is to selectively apply entropy regularization via (1) a top-p mask restricting exploration to high-probability “policy nucleus” tokens, (2) a peak-entropy mask focusing on high-entropy positions along reasoning trajectories, and (3) a self-anchored regularization term keeping global entropy close to its initial value. Experiments on math reasoning benchmarks (AIME 24/25, AMC, OlympiadBench, MATH500) with Qwen2.5-Math-7B/1.5B and LLaMA 3.1-8B show moderate performance gains (+4.8 maj@k on average vs. Dr.GRPO) and more stable training dynamics.

### Strengths
- Timely topic – Addresses the practical issue of entropy collapse and premature convergence in RLVR for large reasoning models.

- Empirical clarity – Provides informative analyses (entropy distributions, trajectory-level entropy heatmaps) diagnosing “entropy explosion” with naive regularization.

### Weaknesses
## Limited Novelty

- Top-p masking: Using top-p sampling for language models is well-established (Holtzman et al., 2019, cited by authors). Applying it to entropy computation is a straightforward extension, not a fundamental insight.

- Self-anchored regularization: Computing regularization loss as MSE from an initial anchor is a relatively simple stabilization technique. The novelty lies mainly in the combination rather than individual components.

- Peak-entropy masking: Identifying important tokens via entropy is also explored by concurrent work (Wang et al., 2025; Cheng et al., 2025). The quantile-based selection (Eq. 9) is reasonable but incremental. These are all very recent papers (May-July 2025 based on arXiv dates), which explains why they're labeled "concurrent work"—they were likely being developed around the same time as SIREN.

1. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. 

2. Reasoning with exploration: An entropy perspective. 

3. The entropy mechanism of reinforcement learning for reasoning language models. 

 

## Lack of theoretical grounding

The paper repeatedly claims to “rethink entropy regularization”, yet provides no formal analysis or theoretical justification for why the proposed masking mechanism should mitigate explosion or collapse beyond intuition and empirical plots. There is no attempt to connect the approach to established RL theory (e.g., maximum-entropy RL, policy improvement bounds, or KL-controlled exploration). This weakens the “rethinking” claim.

## Framing and writing inflation

The title and abstract strongly over-sell the contribution (“rethinking entropy regularization”, “foundational insight”), whereas the actual contribution is heuristic and incremental. Much of Section 2–3 reiterates known observations about entropy explosion without introducing new analysis tools.

### Questions
- How sensitive are results to p (top-p cutoff) and τ (peak-entropy quantile)?

- Does SIREN generalize to domains beyond math, e.g., code generation or open-ended reasoning?

- Can the authors compare to methods like KL-controlled adaptive entropy (e.g., SAC-style) or dual-gradient regularization rather than hand-crafted masks?

- Can the authors investigate whether there are improvements over Pass@k (a larger k)? Several recent papers have studied the reasoning boundaries on math and other benchmarks. 

1. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? 

2. The invisible leash: Why rlvr may or may not escape its origin?

### Soundness
3

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
2

### Summary
This paper suggests a two-step entropy masking mechanism to enhance RLVR approaches. With this approach, semantically meaningless tokens disappeared, and the training process becomes stabilized.

### Strengths
- Main claim of the paper is backed by a wide range of experiments (math benchmarks) and ablation studies
- Implementation is provided in an anonymous repository.
- Concepts are clear: Top-p mask uses vocabulary subsets, and a peak-entropy mask selects them with quantile, and MSE is used for anchoring entropy

### Weaknesses
- The positioning of SIREN among entropy-regularized exploration methods was not sufficiently clear. I assume the extreme failure case in Fig.1 would not occur with Clip-Cov or Entropy Adv.
- Difficult to understand the flow of the recent approaches including Clip-Cov, Entropy Adv, ... (I have read Dr. GRPO, but wasn't aware of the details of the other baselines)

### Questions
- Figure 1 shows the side-effects of naive entropy regularization, which could be exaggerated. The extreme failure case illustrated in Figure 1 uses a naive approach with coefficient 0.005, but it's unclear whether more sophisticated baselines like Clip-Cov or Entropy Adv. also suffer from similar entropy explosions. Could you clarify if these recent methods exhibit comparable failure modes?
- The flow and positioning of recent entropy-based approaches is difficult to understand. While I've read about Dr.GRPO, the specific limitations of other baselines (Clip-Cov, Entropy Adv., RL on Forking Tokens) are not sufficiently explained. What are the concrete weaknesses of each method that SIREN addresses?
- It's challenging to fully grasp SIREN's positioning within the landscape of entropy-regularized exploration methods. The paper establishes that SIREN outperforms naive approaches, but the connection between SIREN's components (top-p mask, peak-entropy mask, self-anchored regularization) and the specific limitations of sophisticated baselines remains unclear. Could you provide explicit mappings, such as "top-p mask addresses Clip-Cov's limitation of X" or "peak-entropy mask improves upon Entropy Adv. by doing Y"?

### Soundness
2

### Presentation
2

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
This paper analyzes the limitations of naive entropy regularization in RLVR, demonstrating that the vast action space and long trajectories in large reasoning models (LRMs) tend to flatten probability distributions across most positions. This observation highlights the necessity of controlling the effective scope of regularization. To address this, the authors propose SIREN which selects exploration scopes at both the action and trajectory levels for more effective entropy regularization, and transforms naive regularization into a self-anchored form to stabilize training.

### Strengths
1.  The paper revisits the reasons why traditional entropy regularization fails to improve performance for LLM reasoning, which is an important research direction.
2.  The authors provide clear visualizations that support their claims.
3.  The analysis convincingly attributes the failure of naive entropy regularization to the vast action space and long trajectories in LRMs.
4.  The experimental evaluation is comprehensive and includes up-to-date baselines.

### Weaknesses
1.  The proposed method is mostly based on heuristics; more theoretical analysis would strengthen the work.
2.  Regarding the Policy Nucleus: "Since these tokens consistently occupy the top ranks in the original model’s probability distribution, we adopt the terminology of Top-p sampling and refer to this subset as the nucleus." The authors seem to constrain exploration to tokens commonly seen in the base model. I wonder whether this might limit the exploration capability, i.e., some useful exploration could fall outside such a subset.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper explores to improve RLVR, particularly entropy collapse and premature convergence. 
Authors argue that naive entropy regularization fails in LRMs due to their vast action spaces and long trajectories, often causing entropy explosions by flattening probability distributions across irrelevant tokens. The authors propose SIREN, a method that employs:
- a top-p mask to focus on semantically meaningful token subsets,
- a peak-entropy mask to target critical high-entropy positions in trajectories, and
- self-anchored regularization using mean squared error to stabilize entropy levels around an initial anchor. 

Built on Dr.GRPO, SIREN demonstrates superior performance on benchmarks like AIME24/25, AMC, OlympiadBench, and MATH500.

### Strengths
- Pinpoints the problem of traditional naive entropy regularization
- Significant performance improvement compared to naive Dr.GRPO with entropy regularization.

### Weaknesses
- All three proposed mechanisms of SIREN: top-p mask, peak-entropy mask, self-anchored regulrization, does seem to be heuristics, being an ad-hoc solutions to seek for a structured exploration.
- Top-p mask and peak-entropy mask introduce additional hyper-parameters that need to be tuned. While the paper is motivated from the difficulty of tuning entropy coefficients, the difficulty of tuning newly introduced hyper-parameters are not discussed. The difficulty of tuning naive entropy regularizatin coefficient and self-anchored regularization coefficient is also not discussed (only results on 1~2 hyperparameters are shown), which I think is very important for the paper.
- The limitations of current algorithm - when the proposed algorithm would not work well - is not discussed. For example, there are datasets where SIREN gives large performance boost (AIME), and small hboost (MATH 500 or Olympiad). What makes the difference?
-  While the focus of the paper is mostly on (top-p mask, peak-entropy mask), most of the performance improvement comes from (self-anchored regulrization), which is not really a new idea. For example, SAC also uses entropy constraints rather than entropy regularization, which can be seem as a principle version of anchored regularization. I believe that entropy constraints should also be compared with self-anchored regularization.
- Shown results are somewhat indirect, and there seems not enough "direct" evidence that proposed SIREN mechanisms (top-p mask, peak-entropy mask) help to have more structured explorations. There is no other algorithms that achieves SIREN(0.005) level of entropy without entropy explosion, probably due to the absence of anchored regularization, and the performance improvements might be only caused by the anchored regularization. Since we are adding hyper-parameters, the small amount of performance improvement suggested would have been possible with any modifications when appropriately tuned.

### Questions
- In figure 6, it is said that naive regularization used coefficient of 0.001 (and seems to be learned well), where in the hyperparameter setting in appendix it says 0.0001 is used for naive reg and any higher value leads to instability. Which is true?

### Soundness
2

### Presentation
3

### Contribution
2
