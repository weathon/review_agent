# BEYOND IMITATION: RECOVERING DENSE REWARDS FROM DEMONSTRATIONS

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Conventionally, supervised fine-tuning (SFT) is treated as a simple imitation learning process that only trains a policy to imitate expert behavior on demonstration datasets. In this work, we challenge this view by establishing a fundamental equivalence between SFT and Inverse Reinforcement Learning. We prove that the SFT objective is a special case of Inverse Q-Learning, which implies that the SFT process does not just learn a policy, but also an implicit, dense, token-level reward model that explains the expert demonstrations. We then show how to recover this dense reward signal directly from the SFT model by formulating a baseline-relative reward function. The availability of such a dense reward model offers numerous benefits, providing granular credit assignment for each token generated. We demonstrate one key application by using these recovered rewards to further improve the policy with reinforcement learning. Our method, Dense-Path REINFORCE, consistently outperforms the original SFT models on instruction-following benchmarks. This work reframes SFT not merely as policy imitation but as a powerful reward learning mechanism, opening new possibilities for leveraging expert demonstrations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes that supervised fine-tuning (SFT) can be viewed as a special case of inverse soft-Q learning under a deterministic token MDP with γ = 1 and a linear conjugate. Building on this, it defines a baseline-relative dense reward and introduces Dense-Path REINFORCE (DPR), a token-level REINFORCE update using log-probability differences between the SFT model and an earlier checkpoint as reward signals. Experiments on AlpacaEval, MT-Bench, LIMA, and Arena-Hard show modest but consistent gains over SFT and competitive results with SPIN and GSIL.

### Strengths
1. Theoretical link between SFT and inverse RL is clear and well-motivated.

2. The telescoping argument makes the equivalence intuitive under γ = 1.

3. DPR is simple, reproducible, and the dense reward idea is easy to understand.

4. Ablations on γ, checkpoint choice, and baseline removal are informative and thoughtfully designed.

### Weaknesses
1. Narrow validity: The equivalence holds only for γ = 1, deterministic token transitions, and a linear conjugate. The paper does test γ < 1, showing expected degradation, which confirms the limitation rather than extending the theory.

2. Inconsistent assumptions: The equivalence relies on a linear conjugate, while the later stability theorem assumes strong convexity. These are mathematically incompatible but presented as part of one framework.

3. Evaluation design: DPR is trained and tested on the same prompt set, so gains could reflect continued fine-tuning instead of genuine reward recovery.

4. Limited robustness: The temperature ablation is minimal and doesn’t analyze stochastic effects or variance.

5. Weak empirical evidence: All evaluations rely on GPT-4 judges without error bars, human checks, or multiple seeds; gains over SFT are small (a few percent).

6. Missing controls: No baseline comparing DPR to simply extending SFT training, making attribution unclear.

### Questions
1. Can the same ψ be both linear (for equivalence) and strongly convex (for contraction)?

2. Why does the halfway checkpoint consistently give the best result?

3. Did you test multiple seeds or new prompts to confirm robustness?

4. Could a continued-SFT baseline match the reported gains?

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
4

### Summary
The paper reframes SFT under a token-MDP view (γ≈1) and shows that the SFT objective is equivalent to an inverse soft-Q objective. This motivates extracting a dense proxy reward as a log-likelihood ratio between a final SFT model and a reference checkpoint, followed by a short, critic-free RL step (REINFORCE with KL to SFT). The method is closed-loop (no environment reward, no preference data, no reward model), simple to implement, and reports consistent gains over plain SFT across several backbones/evals.

### Strengths
Dense credit assignment: Per-token signals are actionable and address SFT’s plateau on long sequences and intermediate steps.

Low engineering overhead: Uses existing SFT artifacts (final SFT + ref checkpoint). No external judges or reward models.

Theory ↔ practice alignment: The SFT ↔ inverse-soft-Q connection and the safe-improvement style argument justify using the recovered dense signal for a small policy-gradient step.

Stable optimizer: REINFORCE + KL to SFT (no critic) keeps the pipeline robust and easy to reproduce.

### Weaknesses
Reference sensitivity: The approach assumes π_SFT is meaningfully better than π_ref. If ref is too weak (noisy signal) or too strong (vanishing signal), the log-ratio reward becomes brittle or tiny.

Log-ratio gaming: The policy may drift toward stylistic artifacts that inflate log π_SFT − log π_ref without improving task quality.

Domain narrowness: Because SFT and ref share training data, the proxy reward is intrinsically domain-tied; out-of-domain generalization of the dense signal is unclear.

Eval reliance: Gains are primarily shown via LLM-as-judge; stronger human or task-grounded metrics would strengthen the case.

minor
S1. Reference selection by evaluation, not step count:
– Choose π_ref via validation metrics (MT-Bench, small human slice, task-specific set) to target the “elbow” where the signal-to-noise of the log-ratio is highest.
– Alternatively use an EMA of SFT weights as π_ref to smooth noise.

S2. Multi-reference ensemble:
– Define log π̄_ref = logsumexp_i(log π_ref,i) − log k (geometric mean). Reward becomes r̂ = log π_SFT − log π̄_ref. This damps idiosyncrasies of any single checkpoint and makes “progress” less gameable.

S3. Dual-KL regularization and reward shaping to reduce gaming:
– Keep KL(π_θ || π_SFT) and add a small KL(π_θ || π_ref).
– Clip/normalize the per-token log-ratio (e.g., cap magnitude or z-score by position).
– Penalize tokens where both π_SFT and π_ref assign low probability (both uncertain) even if the difference is large.
– Add light style/fluency guards (repetition rate, perplexity bounds under a separate LM).

S4. Correlation-gated updates:
– On each batch, compute the correlation between r̂-improvement and a cheap proxy (exact-match on small QA, code unit tests, math verifier). If correlation drops below a threshold, reduce step size or increase KL. This is a simple “reward sanity check.”

S5. Leverage the re-forward pass to incorporate useful rewards (when available), without changing the core method:
– Hybrid reward: use R = α·r̂ + (1−α)·r_ext, where r_ext can be any lightweight verifier signal (unit tests for code, arithmetic checker, safety filter, format validator). α can be annealed from 1→0.8.
– Doubly-robust/token-aware AWR: advantage-weight the SFT tokens by r̂ (and r_ext if present), i.e., reweight the teacher-forced loss with w_t = exp(β·A_t) to unify imitation and RL in one pass.
– Counterfactual filtering: when the re-forward reveals contradictory beams (both low confidence), zero out r̂ for those tokens to avoid amplifying noise.

S6. Report a reference sweep and ablations:
– Show downstream metrics vs. ref placement (early/mid/late) to directly address “is π_SFT actually better than π_ref?”
– Include ablations for single-ref vs. multi-ref, with/without dual-KL, and with/without correlation gate.

### Questions
How is the reference checkpoint chosen? Is it purely by training step or by validation metrics? What is the sensitivity curve (early/mid/late)?
Q2. Can an ensemble of references reduce variance/bias? (Geometric mean of checkpoints as a smoother ref.)
Q3. How do you detect or mitigate reward-hacking (odd outputs that maximize the log-ratio)?
Q4. What is the robustness out of domain (math/code/safety) where SFT confidence calibration differs?
Q5. Since the method already performs fresh forward passes, can those passes be leveraged to incorporate additional rewards or verifiers when available (see Suggestions)?

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
3

### Summary
This paper introduces a novel idea that Supervised Fine-Tuning (SFT) can be viewed as a special case of Inverse Q-Learning, suggesting that SFT does not merely imitate expert policies but implicitly learns a dense, token-level reward model. The authors then recover this implicit reward from the SFT model and propose Dense-Path REINFORCE (DPR), which leverages the recovered reward to more efficiently optimize large language models (LLMs). Experimental results demonstrate that the proposed method outperforms standard SFT across multiple benchmarks.

### Strengths
1. The paper presents a novel idea that Supervised Fine-Tuning (SFT) can be viewed as a special case of Inverse Q-Learning, offering a new perspective for understanding SFT.

2. The authors provide a comprehensive theoretical analysis to support the proposed formulation.

3. Experimental results show considerable improvements over traditional large language model (LLM) training methods.

### Weaknesses
1. In the theoretical analysis, the authors make several strong assumptions about the setting—for example, assuming a deterministic token sequence and a fixed discount factor of $\gamma=1$, rather than a value smaller than 1 as typically used in RL.

2. In the proposed DPR method, the reference policy $\pi_\text{ref}$ is not formally defined. The authors state that it is an SFT checkpoint trained with half of the training samples; however, if the dataset is sufficiently large, wouldn’t this reference policy also become fully trained, thereby reducing the meaningful difference between $\pi_\text{ref}$ and $\pi_\text{SFT}$?

### Questions
1. The reviewer is not an expert in LLMs, but I question whether the current training paradigm of LLMs is primarily driven by SFT. Wouldn’t RLHF have a greater overall impact on model alignment and performance? I therefore have some concerns about the potential contribution and significance of this work to the broader LLM literature.

2. If SFT is theoretically equivalent to IQL, would it be possible to directly apply IQL methods to learn the reward function instead of recovering the reward from SFT?

3. The authors choose REINFORCE for policy optimization. Could the authors clarify why this choice was made instead of using more advanced RL algorithms, such as actor–critic or PPO-based methods?

### Soundness
3

### Presentation
3

### Contribution
3
