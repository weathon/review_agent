# On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning

- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Policy gradient algorithms have been successfully applied to enhance the reasoning capabilities of large language models (LLMs). KL regularization is ubiquitous, yet the design surface, choice of KL direction (forward vs. reverse), normalization (normalized vs. unnormalized), and estimator ($k_1/k_2/k_3$), is scattered across the literature and often intertwined with off-policy estimation. We ask a focused question: under the off-policy setting, what weighting is required for each KL variant so that the surrogate we optimize yields the exact gradient of the intended KL-regularized objective? We answer this with a compact, unified derivation we call the Regularized Policy Gradient (\textbf{RPG}) view. RPG (i) unifies normalized and unnormalized KL variants and shows that the widely-used $k_3$ penalty is exactly the unnormalized KL; (ii) specifies conditions under which REINFORCE-style losses with stop-gradient are gradient-equivalent to fully differentiable surrogates; (iii) identifies and corrects an off-policy importance-weighting mismatch in GRPO's KL term; and (iv) introduces RPG-Style Clip, a truncated-importance-sampling step within RPG-REINFORCE that enables stable, off-policy policy-gradient training at scale. On mathematical reasoning benchmarks (AIME24, AIME25), RPG-REINFORCE with RPG-Style Clip improves accuracy by up to $+6$ absolute percentage points over DAPO. Notably, RPG is a \emph{stable and scalable} RL algorithm for LLM reasoning, realized via (a) a KL-correct objective, (b) truncated importance sampling, and (c) an iterative reference-policy update scheme.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces RPG (Regularized Policy Gradient), a unified framework for KL-regularized policy-gradient training of LLMs under off-policy sampling. It systematizes design choices across: (i) KL direction (forward vs reverse), (ii) normalized vs unnormalized KL (UKL), and (iii) estimator/implementation (fully-differentiable vs REINFORCE-style with stop-gradient). Key theoretical results (e.g., Propositions for UFKL/URKL gradients and losses) show how to obtain exact gradients of the intended KL-regularized objective when training off-policy with importance weights, and establish gradient-equivalence between differentiable and REINFORCE-style surrogates. Practically, the paper (a) clarifies that the widely used k3 estimator corresponds exactly to the unnormalized KL, (b) identifies an importance-weighting mismatch in GRPO’s KL term under off-policy sampling and provides a corrected estimator, and (c) introduces RPG-Style Clip, a dual-clip truncated-importance step for stability. On AIME24/25 with Qwen3-4B, RPG-REINFORCE + RPG-Style Clip outperforms DAPO by up to +6pp absolute accuracy. The method also employs periodic reference-policy updates to maintain a practical KL trust region.

### Strengths
- It provides a compact derivation that covers Forward/Reverse × Normalized/Unnormalized KL, with explicit off-policy corrections and both differentiable and REINFORCE-style surrogates.
- It pinpoints the missing importance weight in GRPO’s KL when sampling off-policy, which could be an issue that many people overlook.
- The paper introduces RPG-Style Clip (dual-clip/truncation) and iterative reference updates, which empirically stabilize off-policy PG for LLM reasoning while requiring only a single active model on GPU (log-probs from $\pi_{\textrm{old}}$ cached).
- Empirically, it has strong performance with +6pp over DAPO in the best cases on AIME24/25, with stable entropy and controlled response length.

### Weaknesses
- All core results are on Qwen3-4B and math reasoning. Claims about “stable and scalable RL” would be stronger with larger backbones (e.g., 7B–14B) and diverse tasks or datasets.
- While GRPO/DAPO are relevant, head-to-head comparisons with varying KL directions (FKL/RKL vs UFKL/URKL) would solidify the empirical story. Current ablations on clip hyper-parameters and reference-update schedules are limited in the main text.
- RPG-Style Clip introduces controlled bias via truncation. The paper acknowledges this and suggests schedules, but there’s no systematic bias study (e.g., varying $\epsilon_1$, $\epsilon_2$) to quantify performance/variance trade-offs.

Overall, the theoretical foundations are solid and the proposed algorithm is well-motivated. The strengths outweigh the weaknesses, and I am leaning toward acceptance.

### Questions
Please refer to the weaknesses.

### Soundness
4

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
This paper presents RPG, a unified theoretical and practical framework for KL-regularized policy gradient algorithms. The authors systematically analyze different KL regularization variants and corresponding estimators, clarifying their gradient properties under off-policy sampling. Their derivation shows how to obtain the exact gradients of intended KL-regularized objectives. It identifies a weighting mismatch in GRPO, and introduces a truncated-importance estimator which stabilizes training. Experiments show RPG and RPG-REINFORCE variants outperform other baselines on mathematical reasoning benchmarks and have better stability.

### Strengths
1. The paper provides a unified framework connecting various KL regularization forms. The derivations offer clarity and formal grounding.
2. The authors propose RPG-Style Clip, which can effectively balance stability and bias under off-policy training and improve scalability for LLMs
3. The experiments demonstrate notable improvements in both accuracy and stability.

### Weaknesses
1. Some representative baselines are missing, e.g., RLOO, REINFORCE++, GPG. 
2. The experiments lack ablation studies isolating the contributions of RPG-Style Clip, reference-update frequency, and clipping thresholds.
3. The RPG-Style Clip introduces a bias-variance trade-off but it is not explored quantitatively.
4. Section 3 and 4 are heavy and hard to follow.

### Questions
1. Why the entropy grows up with training in DAPO and RPG while decreases in GRPO? I assume both DAPO and GRPO should both decrease but to different degrees. This is contradicted to the DAPO paper. 
2. More recent works have opted to remove the KL term from the RL objective and achieve good performance. In comparison, does a compromise of periodically updating the reference model offer significant and irreplaceable advantages? 
3. Do the empirical results show a significant performance difference between the corrected Normalized KL and Unnormalized KL?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The author studied the design of KL regularization in reinforcement learning. Specifically, the analysis shows that the commonly used k3 estimator is equivalent to the use of unnormalized KL. The author also shows that GRPO's original KL divergence implementation lacks an importance sampling term due to its off-policy attributes.

### Strengths
* Nice theoretical analysis between k3 estimator and UKL
* The tackled KL divergence issue is indeed a pitfall of GRPO

### Weaknesses
The experiments in the paper are quite poorly conducted:
* The model is tested only on math tasks, and even then, only two AIME benchmarks are used.
* Figure 2 is presented without any analysis. It seems that in Figure 2(c), DAPO is not converging — what would it look like if we extrapolated the training curve?
* (c) RL training often exhibits high variance; instead of showing a single curve, the authors should include a curve with a standard deviation region.
* (d) In Figure 2(d), why is only GRPO’s actor entropy decreasing? Out of the five curves, three increase and plateau, while the REINFORCEMENT curve continues to rise — why?
* (e) What is “mean@32”? Do you mean “pass@32”?

Also, there seem to be quite a few unanswered questions to be verified by the experiments:
* FKL and RKL seem to be comparable in performance. Is there any evidence that such two constraints result in different properties in the post-trained model?
* Some papers, like DAPO and Dr.GRPO, advise dropping KL divergence for RL tuning. Is there some analysis on how different designs and strengths of KL regularization change the training dynamic and the tuned model's properties?

### Questions
* In Figure 2, is the difference between GRPO and RPG-URKL in whether having the importance weight in the kl term?

### Soundness
3

### Presentation
2

### Contribution
3
