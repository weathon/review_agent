# Self-Guided Process Reward Optimization with Redefined Step-wise Advantage for Process Reinforcement Learning

- Decision: Reject
- Scores: 4, 6, 2, 8

## Abstract
Process Reinforcement Learning (PRL) has demonstrated considerable potential in enhancing the reasoning capabilities of Large Language Models (LLMs). 
However, introducing additional process reward models incurs substantial computational overhead, and there is no unified theoretical framework for process-level advantage estimation.
To bridge this gap, we propose \textbf{S}elf-Guided \textbf{P}rocess \textbf{R}eward \textbf{O}ptimization (\textbf{SPRO}), a novel framework that enables process-aware RL through two key innovations: (1) we show that process rewards can be derived intrinsically from the policy model itself, and (2) we redefine the step-wise advantage by introducing well-defined Cumulative Process Rewards (\textbf{CPR}) and \textbf{M}asked \textbf{S}tep \textbf{A}dvantage (\textbf{MSA}), which facilitates rigorous step-wise action advantage estimation within shared-prompt sampling groups.
Our experimental results demonstrate that SPRO outperforms vaniila GRPO with 3.4× higher training efficiency and a 17.5\% test accuracy improvement. 
Furthermore, SPRO maintains a stable and elevated policy entropy throughout training while reducing the average response length by approximately $1/3$, evidencing sufficient exploration and prevention of reward hacking.
Notably, SPRO incurs no additional computational overhead compared to outcome-supervised RL methods such as GRPO, which benefit industrial implementation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes to remove the costly process reward model and use the internal signal of the policy model itself for process supervision. It correspondingly proposes Cumulative Process Rewards (CPR) and Masked Step Advantage to calculate the advantage of each step. The empirical experiments validate its high efficiency.

### Strengths
- Empirically show that process signals intrinsically derived from the policy itself benefit the performance.

- Propose Cumulative Process Rewards (CPR) and Masked Step Advantage to obtain the step advantage.

- Empirical experiments validate the high efficiency of SRPO.

### Weaknesses
> **Overclaim the theoretical contribution** 
>---
>- Although authors claim they theoretically demonstrate that process rewards can be derived intrinsically from the policy model itself, the proposition 1 in section 2.1 is not novel. The similar proofs can be found in [1][2]
>
>- The derivations of Cumulative Process Reward (CPR) are also not novel, which is exactly the same as [3]
>
>[1] Zhong, Han, et al. "DPO Meets PPO: Reinforced Token Optimization for RLHF." Forty-second International Conference on Machine Learning.
>
>[2] Rafailov, Rafael, et al. "From $ r $ to $ Q^* $: Your Language Model is Secretly a Q-Function." First Conference on Language Modeling.
>
>[3] Xie, Tengyang, et al. "Exploratory Preference Optimization: Harnessing Implicit Q*-Approximation for Sample-Efficient RLHF." The Thirteenth International Conference on Learning Representations.


> **Unfair comparison**
> ---
> As shown in Fig.1, Fig.4, **the training appears not to have fully converged.** All the training has stopped early. Hence, although SPRO displays an early performance surge, it does not necessarily lead to final superior performance.


> **Doubtful claim about length bias**
> ---
>- The authors claim MSA does not induce the length bias. However, their argument is more like a step-number bias, i.e., MSA would not make the policy model prefer responses with more steps. Although there are definitely some correlations between the step number and response length. However, can they really be equivalent? Moreover, there is a length normalization $1/|y|$ in the SRPO loss, which is proven to be a factor inducing the length bias [4].
>- According to Table 2, the response length of SPRO across all datasets has a weirdly sharp decrease, which shows a contrary length bias towards shorter response length. 
>
>[4]Liu, Zichen, et al. "Understanding r1-zero-like training: A critical perspective." arXiv preprint arXiv:2503.20783 (2025).



> **Typos**
> ---
> - L23: vaniila -> vanilla
> - L96: advantages estimation -> advantage estimation
> - L248: implict -> implicit
> - L304: 3th -> third
> - L459: its -> their

### Questions
> Zero rewards for longest responses

As demonstrated in L304-308, if only the third response contains a valid step, the advantage of this step is zero. Therefore, for each group of responses, there must be some steps of the longest response unlearnt. Is it a waste of computation? Why does CPR and MSA can benefit this trade-off?

> Is outcome supervision always necessary for the training?

Current training signals combine the outcome and process supervision, which is understandable since when the policy model is not strong enough during the early training stage, the policy needs outcome rewards to improve its performance. However, in the later training stage, whether the pure process signals can make an effect is doubtful. Does CPR always necessarily combine the outcome rewards to work? 

> Contrary observation of entropy change

As shown in Fig.5(b), the entropy of vanilla GRPO remains stable, which is contrary to the previous empirical studies [5]

[5] Cui, Ganqu, et al. "The entropy mechanism of reinforcement learning for reasoning language models." arXiv preprint arXiv:2505.22617 (2025).

> Ablation of MSA

As shown in Fig.2(a), PRIME uses a different advantage calculation method, which aggregates all process rewards into a single group. Is grouping rewards at the same step (MSA) really superior to grouping rewards among all steps in the SPRO framework?  Does the advantage calculation of PRIME also work well in the SPRO framework?

### Soundness
1

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
2

### Summary
This paper proposes a new reward formulation called Cumulative Process Reward (CPR), which leverages the policy model’s own output as a source of reward.  
Based on this idea, the authors introduce a new advantage estimation method, MSA, demonstrating improvements in both reasoning performance and training efficiency.  
Theoretical analysis further shows that the method can mitigate reward hacking while preserving sufficient exploration and maintaining high policy entropy.

Parts of this review were discussed with a colleague to ensure clarity and accuracy.

### Strengths
1. The paper clearly articulates its contributions and presents a coherent logical progression. It first introduces the CPR concept, builds upon it to propose the MSA module, and finally develops the overall SPRO algorithmic framework. The overall structure is well-organized, and the presentation is smooth and easy to follow.  
2. The experimental design convincingly demonstrates that the proposed SPRO algorithm outperforms prior baselines in terms of both accuracy and training efficiency.  
3. The theoretical derivation of CPR is clear.

### Weaknesses
1. One of the main contributions of this paper is the introduction of the self-guided reward (CPR), which is theoretically formulated as a general reward mechanism that appears not to be constrained by reasoning tasks or step-wise processes. However, the experiments only evaluate CPR within reasoning-oriented benchmarks. It would significantly strengthen the paper to include results demonstrating CPR’s generality—specifically, whether it can serve as a replacement for reward models in other common RL fine-tuning scenarios beyond reasoning tasks.  
2. The quantitative experiments are conducted on a single base model (Eurus-2-7B-SFT). To better demonstrate the robustness and general applicability of the proposed method, it would be beneficial to include experiments on multiple model scales.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes SPRO, a process-aware RL method for reasoning LLMs that removes the extra PRM by deriving a self-guided token-level signal from the policy itself and redefines step-wise advantage via two constructs: Cumulative Process Reward and Masked Step Advantage (MSA). CPR is obtained by telescoping a policy–value identity to produce a per-prefix cumulative log-ratio against a reference model; MSA normalizes CPR across trajectories at the same step within a shared-prompt group, and the sum of (normalized) outcome reward and MSA is used in a PPO-style clipped objective (Algorithm 1). Experiments on math and code benchmarks report better accuracy and shorter outputs vs. GRPO and PRIME, with lower GPU hours.

### Strengths
SPRO avoids an auxiliary PRM and computes process feedback directly from the policy and a fixed reference model, preserving a dual-model training footprint.

MSA enforces per-step comparisons within a prompt’s rollouts, which directly targets the length-bias problem common in outcome-only grouping. The construction and “masked mean” baseline are clearly spelled out.

Simulation gains include shorter trajectories, higher accuracy, and better entropy than PRIME/GRPO under their setup, with tables/figures that are easy to read.

### Weaknesses
The key identity used to define CPR (Eq. 2–5) is derived by combining the max-entropy fixed-point and a Bellman-style relation that holds for the optimal policy/value. In practice, SPRO replaces $\pi^*$ with the current $\pi_\theta$​ to compute the log-ratio sum. The paper does not quantify the bias introduced by this substitution nor provide a bound that links CPR to true per-step advantage under model mismatch. This is central because CPR then drives the update.

Proposition 1 asserts that “any LLM is an optimal soft Q-function for some reward,” by scaling logits to define Q and then defining V accordingly so that \pi matches a soft-optimal policy (Eq. 1). This is a definition-level construction; it does not imply that the induced reward (or the resulting token-credit signal) is useful or less biased/noisier than an external PRM. The additional claim that “stronger LLMs provide more accurate credit assignment” is also not supported here.

The derivation of CPR and the log-ratio uses a KL-regularized, entropy-augmented setting with inverse temperature \beta and a KL term to the reference policy. But in experiments, the KL coefficient is set to 0, while CPR still sums log-ratios vs. \pi_ref. This disconnect raises questions: if the training does not penalize deviation from \pi_ref, why reference the right anchor for process signal, and what sets \beta numerically?


The paper attributes entropy rises and reduced length to SPRO’s fine-grained feedback, but these are correlational observations. No ablation shows that removing CPR or changing MSA grouping eliminates the effect; no statistical tests or confidence bands are reported.

### Questions
The definition of CPR relies on the fixed-point relation of the optimal entropy-regularized policy.
How valid is this identity when used with the current non-optimal policy \pi_\theta? Specifically, can you quantify or bound the bias introduced when replacing \pi^* with \pi_\theta? Without this, why should CPR correlate with the true per-step advantage?

Proposition 1 shows that any policy can be seen as optimal for some constructed reward.
What ensures that the “implied reward” derived from logits is meaningful for credit assignment rather than an arbitrary scaling?
Have you empirically verified that CPR correlates with external process rewards (e.g., from a PRM)?


Your derivation of CPR assumes a KL-regularized objective with temperature β and reference policy, but experiments disable the KL penalty. How do you justify computing log-ratios vs. \pi_ref. When the actual optimization omits that regularization?
Does \beta still have a numerical role during training? If not, is CPR just a scaled log-probability difference?

You claim SPRO enables “self-guided” credit assignment without a PRM. Is the guidance truly intrinsic, or is it implicitly dependent on the chosen reference \pi_ref? What happens if \pi_ref  is very weak or unrelated to  \pi_\theta?
Does CPR degenerate to pure likelihood training in that case?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces Self-Guided Process Reward Optimization (SPRO), a new reinforcement learning framework for LLM reasoning that removes the need for explicit Process Reward Models (PRMs).
Specifically, SPRO proposed two algorithm improvements, Cumulative Process Reward (CPR) and Masked Step Advantage (MSA), which is a self-derived process signal from the policy model’s own logits, serving as an intrinsic token-level reward and a step-wise advantage estimator computed within shared-prompt groups, enabling fair per-step comparisons without length bias.
Experiments on math and code benchmarks show 17.5% accuracy gain and 3.4× higher training efficiency than GRPO, while using no additional computational cost. SPRO also maintains high policy entropy and shorter reasoning traces, mitigating reward hacking and improving token efficiency.

### Strengths
1. The paper clearly identifies the computational inefficiency of PRM-based methods and proposes a conceptually elegant self-guided alternative grounded in theory (policy-as-Q-function perspective).
2. The derivation connecting token-level MDPs, policy logits, and process rewards (Eq. 1–5) is rigorous and builds well on prior implicit reward literature (e.g., DPO/PRIME).
3. The presentation of CPR + MSA, especially Fig. 2–3, offers an intuitive and well-structured comparison with GRPO/PRIME.
SPRO consistently outperforms strong baselines on diverse math/code tasks and demonstrates efficiency improvements of up to 6.7× in GPU hours.
4. Maintaining a dual-model (policy + reference) setup without an extra PRM makes deployment convenient and affordable for large-scale reasoning models.

### Weaknesses
1. Results are restricted to 7B-scale models; scalability claims to larger models reasoning are not empirically validated.
2. While the “policy-as-reward-model” proposition is appealing, it risks circular reasoning—reward quality depends on policy quality, which itself evolves via those rewards.
3. Baselinses are mainly include GRPO and PRIME; missing other RL methods (e.g., Rest-MCTS*, Reinforce++, DAPO) weakens generality claims.

### Questions
1. Could the method extend naturally to multi-turn dialogue or tool-use reasoning, where “steps” are semantically heterogeneous?

### Soundness
3

### Presentation
3

### Contribution
3
