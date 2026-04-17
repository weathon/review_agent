# DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Reinforcement learning (RL) with group relative policy optimization (GRPO) has become a widely adopted approach for enhancing the reasoning capabilities of multimodal large language models (MLLMs). While GRPO enables long-chain reasoning without a traditional critic model, it often suffers from sparse rewards, arising from the scarcity of positive feedback on difficult problems, and from advantage vanishing, which occurs when group-level rewards exhibit high consistency for problems that are too easy or too hard. Existing solutions fall into three categories: sample enhancement and expansion, which may aggravate vanishing advantage due to poor control of difficulty distribution; selective sample utilization, which fails to fully leverage the value of all data; and indirect reward design, which may introduce biased optimization directions due to misalignment between reasoning and the final outcome. However, these approaches overlook a fundamental question: for a given problem, how can we ensure that the within-group reward distribution of responses exhibits enough variance to yield clear optimization signals for each response? To address these issues, we propose DIVA-GRPO, a difficulty-adaptive variant augmentation advantage method that dynamically adjusts the difficulty distribution of variants for each problem from a global perspective. Our method dynamically assesses problem difficulty, samples variants with appropriate difficulty levels, and advantages are computed within both local and global(a problem and its variants) groups using difficulty-weighted and normalized scaling. This design alleviates reward sparsity and advantage vanishing, minimizes data waste, and improves training stability. Extensive experiments on six reasoning benchmarks demonstrate that DIVA-GRPO outperforms existing approaches in both training efficiency and reasoning performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes DIVA-GRPO, a novel reinforcement learning algorithm for multimodal large models designed based on the idea of dynamic difficulty adjustment. DIVA-GRPO dynamically evaluates the difficulty of each problem and generates query variants dynamically according to the difficulty to adjust the optimization signals in the training process. Additionally, this method puts forward Difficulty-weighted scaling and Reward-range-based Advantage Rescaling to optimize the calculation of advantage values during training.

### Strengths
Regulating the optimization signals in the training process based on dynamic difficulty adjustment is an intuitive, reasonable, and effective approach.
Detailed theoretical derivations support the motivation.
Leading results on multiple benchmarks demonstrate the effectiveness of the algorithm.

### Weaknesses
1. Possible missing related work: There are several studies closely related to the method in this paper and highly relevant. Noisy Rollout uses image noising to facilitate reinforcement learning; Hint GRPO introduces adaptive prompt augmentation; and Dr. GRPO removes the variance term in advantage estimation, which is similar to the RRB in this paper.
2. The comparison is conducted on only a single base model, Qwen2.5-VL-7B, which greatly limits the applicability of the proposed method. Evaluating across different model sizes and different model families (e.g., LLaVA) would help assess the generality and applicability of the approach.
3. Potentially misleading “training efficiency and stability” claim. In Fig. 3(c), the efficiency is argued based on the number of steps required to reach a certain performance, which can be misleading. First, DIVA-GRPO requires additional data preparation compared to GRPO. Second, training stability is not observed, and the provided curves suggest that performance gains have not yet converged; could you provide a longer training dynamic until the gains fully converge?

### Questions
In Section 4.3, the w/o Variant Generation setting (Table 2) performs lower than GRPO+Reward-Range (Table 3). Does this imply that the combination of Difficulty-Weighting, RRB, and G-L Balance underperforms using only RRB?

If the authors’ response sufficiently addresses my concerns, I will consider increasing my score.

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
4

### Summary
The paper proposes a difficulty-adaptive variant advantage framework that (i) dynamically assesses per-problem difficulty from rollout accuracy each epoch; (ii) generates difficulty-matched variants that preserve the gold answer; and (iii) computes both local and global advantages, then stabilizes them via batch z-score normalization, difficulty-weighted scaling, and a Reward-Range-Based (RRB) rescaling that down-weights spurious advantages when reward spread is tiny. Together these steps keep within-group reward variance informative and training signals balanced across difficulties.

### Strengths
1. The paper is generally well-written and easy to follow, with a clear description of the method.
2. The paper provides intuitive visual demonstrations to help better understand the paper.

### Weaknesses
1. **Clarify local vs. global advantage magnitudes.** In `Line 263-264`, the paper argues that because global advantages are computed over `m×k` samples, their magnitudes differ from local advantages. However, in `Line 250-252` the global advantage is also normalized, which should mitigate raw magnitude discrepancies. Moreover, the subsequent difficulty-weighted scaling implies the effective magnitude of the global term should also depend on the direction and strength of the difficulty adjustment. Could the authors provide a more detailed and intuitive explanation on this claim and how the proposed pipeline balances the two streams in practice.

2. **Difficulty range and initialization.** The difficulty range is preset to `[1, 9]` with an initial value of `5`. Is there empirical or theoretical justification for this specific choice? Please consider reporting a sensitivity analysis over alternative ranges (e.g., `[0, 1]`, `[0, 5]`, `[1, 7]`) and different initializations to assess stability, convergence rate, and final accuracy. If the choice is largely heuristic, a brief rationale (e.g., numerical stability) would be helpful.

3. **Fairness of efficiency claims.** `Figure 3(c)` reports a `3.17×` faster training speed in terms of *steps*. However, DIVA-GRPO performs **difficulty-adaptive variant generation** at each step, which adds per-step overhead (both compute and memory). Comparing steps alone may therefore be unfair to the GRPO baseline. Please report **wall-clock training time** and **peak memory usage** across runs, and (ideally) throughput metrics (e.g., tokens/sec) to substantiate the claim and isolate where the efficiency gains come from (faster convergence vs. higher per-step cost).

4. **Rollout parity across methods.** In `Line 368-369`, the rollout count is set to `k=5`. If each step also includes `m` difficulty-adaptive variants, then the *effective* number of rollouts per original prompt is larger than `k`. To ensure a fair comparison, could the baseline GRPO be configured with an equivalent total number of rollouts (e.g., set its rollout count to match the sum of local and global samples), or otherwise report an ablation that aligns the **total rollout budget** per step across methods? This will help attribute gains to the algorithm rather than additional sampling.

5. **Relative vs. absolute difficulty weighting.** As noted in `Line 301-302`, the difficulty-weighted scaling uses **relative** difficulty within a group to reweight advantages, whereas many prior works use **absolute** difficulty. It would strengthen the paper to include an ablation comparing relative vs. absolute difficulty weighting (keeping other factors fixed), reporting effects on stability (variance of advantages/gradients), convergence speed, and final performance.

6. **Answer preservation and broader augmentation.** When modifying the text description of the original problem, how do the authors **guarantee** that the final answer remains unchanged? In addition, the current **difficulty-adaptive variant generation** constrains variants to preserve the original answer. Are there plans (or results) for generating **new problems with new answers** under verifiable supervision (e.g., programmatic solvers or formal checkers)? Such a setting could better align with the paper’s stated goal in `Line 232-233` of “improving generalization to unseen expressions”.

### Questions
See the `Weaknesses` part.

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
The paper proposes DIVA-GRPO, a GRPO-based RL framework for MLLMs that (i) dynamically assesses problem difficulty and expands each item into semantically consistent variants with controlled difficulty; (ii) computes local (per-problem) and global (across variants) advantages with batch z-score normalization and difficulty-weighted scaling; and (iii) introduces Reward-Range-Based rescaling (RRB) to avoid exaggerated advantages when reward variance is small.

### Strengths
Method is clearly presented and well-motivated: The paper crisply identifies advantage vanishing and reward sparsity in GRPO, then designs a pipeline that directly targets these failure modes.

Thorough experiments and diagnostics: Results span six benchmarks with ablations and a neat speedup study vs GRPO; figures/tables are informative.

Generalizable component (RRB): The RRB trick improves vanilla GRPO too, suggesting portability beyond this specific framework.

### Weaknesses
Unclear significance under matched baselines: Gains over strong GRPO variants are modest or inconsistent, and baseline setups do not appear strictly aligned.

Scope and transfer are not well positioned: It is unclear whether the difficulty-adaptive variant + local/global advantage scheme is truly first for MLLMs, how it relates to similar ideas, and whether the recipe generalizes to text-only GRPO without images.

Figure interpretation ambiguity: The GRPO training curves in Fig. 3(b) vs. Fig. 3(c) differ in shape, but the exact experimental differences are not fully specified.

### Questions
Baselines (GRPO family): Do you have quantitative head-to-head results against other GRPO or GRPO-improvement methods (such as DAPO or other relevant methods referenced in your Related Work), under identical settings (same backbone, datasets, steps)?

Scope & transfer: Is this the first time the proposed difficulty-adaptive variant + local/global advantage scheme appears in MLLMs? Could the same recipe be used in text-only GRPO training without images?

Figure details: Why do GRPO curves in Fig. 3(b) vs 3(c) look different?

### Soundness
3

### Presentation
4

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
This paper introduces DIVA-GRPO, a difficulty-adaptive variant of Group Relative Policy Optimization (GRPO), designed to improve multimodal reasoning ability of MLLMs. The key innovation lies in dynamically adjusting the difficulty of training problems and their variants to maintain informative reward signals, thereby mitigating reward sparsity and advantage vanishing in reinforcement learning. Also, DIVA-GRPO computes both local and global advantages using difficulty-aware normalization and reward-range-based rescaling. Extensive experiments demonstrate the effectiveness of the proposed DIVA-GRPO

### Strengths
1. The method is thoroughly developed, with detailed algorithms, theoretical analysis (e.g., variance reduction and convergence), and extensive empirical validation across multiple benchmarks.
2. The idea of dynamically adjusting problem difficulty and generating semantically consistent variants is well-motivated.

### Weaknesses
1. The method introduces new hyperparameters (e.g., difficulty scaling factor k, learning rate η) that lack robust ablation or automatic tuning. This may hinder out-of-the-box usability and necessitate per-task calibration.
2. The training flow illustrated in Figure 2 is somewhat confusing. A clearer depiction of how variants are sampled, advantages computed, and the policy updated would improve understanding.
3. The difficulty of each query is updated after every epoch, but the total number of training epochs is not specified. Clarifying this and analyzing its impact on performance would be helpful.
4. In Figure 3, training steps range from 0 to 150. It is unclear whether this covers more than one epoch. If not, the difficulty of each query remains near its initial moderate value—raising the question of where the observed gains originate.
5. Compared to vanilla GRPO, DIVA-GRPO requires additional rollouts on variants. A detailed analysis of the resulting computational cost and trade-offs should be provided.
6. When the paper reviews the original GRPO, it states that the reward is “rule-based”; however, GRPO itself is not tied to rule-based rewards and can equally be used with a reward model.

### Questions
1. How do you ensure that the generated variants (especially for hard problems) are semantically equivalent and answer-preserving? Moreover, how can we verify that the proposed recipe for “easy” and “hard” variants really behaves as intended—i.e., after the perturbations the model’s accuracy on the easy set effectively goes up while that on the hard set goes down?
2. Since reasoning variants are generated using external models, how do you ensure their quality and consistency? Have you observed any degradation in performance when using different external models or prompts?
3. Could you provide a sensitivity analysis for the difficulty-weighted scaling parameter k and the difficulty update rate η? How sensitive is the model to these choices, and are there general guidelines for setting them?
4. What is the computational cost of generating and training with variants compared to standard GRPO? Is the speedup in convergence enough to offset the additional overhead?
5. Are there specific types of problems or difficulty levels where DIVA-GRPO still struggles? It would be helpful to include examples or failure analysis to understand the method’s limitations.
6. Do you plan to release the generated variants or the code for variant generation? This would help the community reproduce and build upon your work.

### Soundness
3

### Presentation
2

### Contribution
2
