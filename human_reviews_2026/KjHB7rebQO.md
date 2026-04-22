# RiskPO: Risk-based Policy Optimization with Verifiable Reward for LLM Post-Training

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
Reinforcement learning with verifiable reward has recently emerged as a central paradigm for post-training large language models (LLMs); however, prevailing mean-based methods, such as Group Relative Policy Optimization (GRPO), suffer from entropy collapse and limited reasoning gains. We argue that these issues stem from overemphasizing high-probability output sequences while neglecting rare but informative reasoning paths. To address these challenges, we propose Risk-based Policy Optimization (RiskPO), which substitutes classical mean-based objectives with principled risk measures. Specifically, we introduce a Mixed Value-at-Risk objective that integrates weighted attention over multiple regions of the reward distribution, thereby amplifying gradient signals on challenging instances and preventing overconfident convergence. We further design a bundling scheme that aggregates multiple questions into bundles, thus enriching the feedback signal and yielding more stable and informative training dynamics. Theoretically, we prove that the risk-averse update alleviates entropy collapse and promotes exploration. Numerically, RiskPO achieves consistent and significant improvements in mathematical reasoning, multi-modal reasoning, and code generation benchmarks, surpassing GRPO and its variants on both Pass@1 and Pass@k metrics. Our results demonstrate that risk-based optimization provides a rigorous and effective paradigm for enhancing LLM reasoning capabilities. The implementation is available at https://github.com/RTkenny/RiskPO.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Risk-based Policy Optimization (RiskPO), a post-training reinforcement learning (RL) framework for large language models (LLMs) that replaces the standard mean-based GRPO objective with a risk-sensitive objective called Mixed Value-at-Risk (MVaR).
The motivation is that GRPO’s mean-based objective focuses on high-probability “easy” reasoning trajectories, causing entropy collapse and limited exploration. RiskPO instead amplifies gradient signals from low-reward or rare reasoning paths, encouraging exploration of harder problems.
The authors bundle multiple questions to enrich the binary reward signal, analyze entropy theoretically, and show that the risk-averse formulation mitigates overconfidence and improves reasoning diversity.

Extensive experiments on over ten benchmarks—including AIME24/25, Minerva, Olympiad, MATH500, and AMC—show that RiskPO outperforms GRPO, DAPO, and Dr.GRPO, achieving higher Pass@k and Avg@k metrics. The improvement is more pronounced at higher k, indicating broader coverage of valid reasoning trajectories.

### Strengths
- Clearly identifies the limitations of mean-based GRPO objectives and motivates a principled alternative.  
- Proposes a novel risk-sensitive objective (MVaR) that enhances exploration and mitigates entropy collapse.  
- Shows consistent performance improvements across reasoning benchmarks, especially on harder problems.

### Weaknesses
- Experiments are mostly limited to mathematical reasoning tasks, with unclear generalization to other domains.  
- The empirical analysis of how risk sensitivity affects exploration or diversity is insufficient.

### Questions
See weaknesses.

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
This paper introduces RiskPO, a reinforcement learning algorithm method for LLM post-training. RiskPO replaces the original objective in GRPO with a risk-sensitive objective called Mixed Value-at-Risk (MVaR). The key motivation is that methods like GRPO suffer from entropy collapse and poor exploration. RiskPO uses bundled questions and tail-focused optimization to amplify gradient signals from rare but informative low-reward cases, thereby promoting exploration and better reasoning. Experiments show that RiskPO outperforms other baselines on several benchmarks and mitigates the overconfidence issue.

### Strengths
- This paper investigates a question of significant concern to the community: the entropy collapse of GRPO during training and the sparse reward issue on difficult problems.
- It’s very interesting to incorporate a risk-sensitive objective based on the Mixed Value-at-Risk into the advantage estimation. The detailed theoretical explanation makes it clearer.
- RiskPO shows promising performance gain compared to other baselines. The ablation study is thorough.

### Weaknesses
- The motivation is not convincing enough. The authors argue that GRPO suffers from entropy collapse, which is indeed a consensus in the community. However, they attribute this to overemphasizing high-probability output without any experimental or theoretical verification. Due to the design of the GRPO advantage, the model's gradient is 0 on both all-correct and all-incorrect problems, only updating samples with performance differences within the same group. I'd like to know how the authors correlate these non-zero advantage samples with high probability; is there any experimental verification or theoretical derivation?
- I think considering GRPO as a mean-based method is debatable, since there is a significant difference between directly using reward signals for gradient optimization and using advantages. Therefore, I do not consider GRPO to be truly maximizing the expected average reward. In my view, RiskPO essentially extends the original group definition in GRPO, using different baseline definitions to mitigate the sparse reward problem in difficult tasks. In this context, RiskPO and GRPO should be classified similarly in terms of their objectives.
- Although the proposed bundling strategy is intended to approximate a continuous reward distribution, the underlying feedback remains binary in nature. This discreteness introduces a potential mismatch between the analytical assumptions in Section 4 (which rely on smooth reward distributions) and the practical training setup. Moreover, the quality of the MVaR-based advantage estimation may depend sensitively on the bundle and generation sizes (B and G). Given that both are relatively small in the experiments, it remains unclear whether the estimated quantiles accurately reflect the model’s true reward distribution at each training step. Additional evidence or analysis would strengthen the authors’ claim that MVaR optimization remains valid under such discrete and limited sampling conditions.
- The experimental results should be expanded, e.g., on larger models. Additionally, the settings of the current experiments should be described more clearly. For Tables 1 and 2, how many samples were used to compute Pass@1? Some baselines perform worse than vanilla GRPO, and I wonder why. Disclosing this information would make the results more convincing.

### Questions
- Can you further explain why the bundle grouping makes sense? 
- What does “papers” refer to in lines 251 and 252?
- I think the connection between Sec. 5 and Secs. 3,4 is somewhat weak. Could you further explain why strengthening the gradients on hard questions can address the issue of overconfident convergence?
- Please see Weaknesses for the other questions mentioned.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a new RL algorithm with verifiable rewards. The authors argue that GRPO mainly optimize mean rewards, leading to model overfitting to easy prompts and entropy collapse. The authors propose to replace mean-based objectives with risk-sensitive measures that emphasizing the lower-tail of the reward distribution. To obtain distributional information on the reward, authors propose to group multiple prompts and compute a shared reward distribution. With extensive experiments, the authors demonstrate that the proposed method improves performance across reasoning tasks compared to GRPO.

### Strengths
* Novel design: This work proposes a novel RL objective that leverages distributional information of reward to effectively enhance model's capability in solving difficult tasks. 
* The work is very well written, presenting complex ideas with clarity that makes both the theoretical analysis and experimental results easy to follow.

### Weaknesses
* The proposed framework adds complexity to RL training with additional hyper-parameters to tune.

### Questions
NA

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
3

### Summary
The paper proposes a risk-sensitive alternative to mean-based RLVR (Reinforcement Learning with Verifiable Reward) objectives. It introduces a Mixed Value-at-Risk (MVaR) loss that emphasizes the lower tail of the reward distribution and a “bundling” trick that aggregates multiple questions to avoid vanishing advantages when all samples for a question are wrong. The theory links advantage–logprob covariance to entropy changes and argues MVaR mitigates entropy collapse. Empirically, the paper reports Pass@k gains over GRPO and recent variants on several math benchmarks, plus smaller gains on coding and multimodal tasks.

### Strengths
Motivation is reasonable: mean objectives over-weight easy, frequent modes; a lower-tail-aware loss could push exploration of harder cases.

Simple to implement: the method is a drop-in change atop GRPO-like training (quantile tracking + bundle-level advantage).

Some positive results: the paper reports consistent improvements on hard math suites and provides ablations (quantiles, bundle size, risk-seeking vs risk-averse).

### Weaknesses
Novelty feels incremental.
Applying standard risk measures (CVaR/RVaR/MVaR) to RLVR is a natural extension, and the “bundling” device is an engineering fix to sparse binary rewards. The paper would read more convincingly if it isolated how much of the gain comes from risk weighting versus bundling. A missing baseline is Mean + Bundling (i.e., GRPO-style mean objective but with bundles). A direct CVaR-only baseline adapted to RLVR is also needed to understand if “mixed” VaR really matters.

Benchmark choice is dated and misses stronger generalization tests.
MATH500 and GSM8K are saturated; GSM8K-Platinum exists, and Putnam-AXIOM (arXiv:2508.08292) directly targets extrapolative generalization with functionally novel problems. If the claim is “expands the reasoning frontier,” I would expect evaluation on Putnam-AXIOM and related unseen-variation splits. As written, improvements on older suites are less persuasive.

Catastrophic forgetting is not ruled out.
RL-trained reasoning models are known to regress on distributional pockets they previously handled well. The paper attributes entropy dynamics to the objective, but the same curves could reflect forgetting. Please include a retention analysis: pre/post accuracy on held-out skills or difficulty bins (and on easier sets that should not degrade), plus calibration/entropy before/after. Without this, the claimed mechanism is under-identified.

Prompt-level diversity controls are a missing baseline.
There is emerging evidence that you can preserve output diversity/entropy via prompted diversity instructions (e.g., “enumerate diverse solution paths; assign probabilities”) and light sampling changes—no RL required. Please include a strong prompt-only diversity baseline, matched on compute, to test whether RiskPO’s benefits exceed careful prompting.

Figure 1 is not convincing; statistics are thin elsewhere.
The AIME learning curves (Pass@32/Avg@32) are noisy and, as plotted, do not clearly outperform GRPO in a way that rules out randomness; the panel lacks confidence intervals. AIME’s n=30 also makes variance high. Later figures (e.g., Figure 4) are more favorable but still fairly close on some panels. Please add multi-seed CIs, paired bootstrap on Pass@k, and exact tests (e.g., permutation) for headline numbers across all datasets, and report standardized effect sizes. For small suites (AIME), de-emphasize them in the main paper or aggregate multiple seeds/splits.

Theory–practice gap.
The entropy analysis uses a tabular softmax and a natural-gradient step; actual training is transformer + Adam with sequence-level clipping and IS. While this is common in theory, please add practical diagnostics (token-level entropy, solution-path diversity, coverage metrics) to demonstrate that the claimed mechanism holds in the real training dynamics observed here.

Verifier details and noise sensitivity.
Because RLVR relies on automatic verifiers, please specify the exact checkers per dataset (math normalization rules, code execution policy, multimodal grading) and include a robustness check to verifier noise. Tail-emphasizing objectives can amplify reward mislabels.

### Questions
Specific requests / experiments that would change my mind

Add Putnam-AXIOM (and other unseen-variation splits) with 95% CIs and p-values; show RiskPO beats strong GRPO/DAPO/GSPO baselines and a prompt-diversity baseline at matched compute.

Ablate attribution: (a) Mean + Bundling, (b) CVaR-only + Bundling, (c) MVaR without bundling.

Forgetfulness audit: pre/post retention on easy skills and previously-mastered bins; report calibration changes.

Statistics: ≥5 seeds, paired bootstrap CIs for Pass@k and Avg@k, and a preregistered evaluation script.

Compute parity: confirm identical budgets, temperatures, decoding, and training steps for all baselines.

Prompt baseline: add a well-tuned diversity-prompting method that explicitly enumerates multiple reasoning paths with probability assignment, to test whether RL is necessary for the claimed effects.

Minor comments

Clearly define Avg@k in the main text; it is referenced but not formalized.

Explain the length-normalized sequence IS ratio exponent (the 1/|y| in Eq. (3)): why this choice? Any stability issues without it?

Acknowledge AIME’s small-n variance in-paper (not just in the appendix) and report CIs in all main figures.

Tighten text and fix minor wording (“catalyse”, “papers” vs “bundles” in Algorithm 1 Step 6).


Note: all writing was assisted by LLMs, but all the thoughts were my own.

### Soundness
2

### Presentation
3

### Contribution
2
