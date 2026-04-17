# No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) is a powerful framework for improving the reasoning abilities of Large Language Models (LLMs). However, current methods such as GRPO rely only on problems where the model responses to the same input differ in correctness, while ignoring those where all responses receive the same reward — so-called zero-variance prompts. In this work, we argue that such prompts are not useless but can, in fact, provide meaningful feedback for policy optimization. To this end, we introduce Reinforcement Learning with Zero-Variance Prompts (RL-ZVP), a novel algorithm that extract learning signals from zero-variance prompts. RL-ZVP directly rewards correctness and penalizes errors even without contrasting responses, modulating feedback with token-level characteristics to preserve informative, nuanced signals. Across six math reasoning benchmarks, RL-ZVP achieves significant improvements of up to 8.61 points in accuracy and 7.77 points in pass rate over GRPO, while consistently outperforming other baselines that filter out zero-variance prompts. These results highlight the untapped potential of learning from zero-variance prompts in RLVR.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses a critical inefficiency in Reinforcement Learning with Verifiable Rewards (RLVR), known as the "advantage vanishing" problem. Standard methods, such as GRPO, fail to learn from "zero-variance prompts"—instances where all sampled responses are either uniformly correct or incorrect—because their advantage calculation collapses to zero. The authors compellingly argue that this constitutes a significant source of inefficiency, as these prompts are both computationally expensive (rollouts consume approximately 50% of training time) and highly prevalent (comprising 30-99% of the data). The proposed method, RL-ZVP, leverages these prompts rather than filtering them. It achieves this by defining a new advantage signal: the direction is set to positive for all-correct groups and negative for all-incorrect groups, while the magnitude is modulated by token-level entropy. This "entropy-guided" shaping mechanism rewards high-entropy (uncertain, complex) tokens in correct responses and penalizes low-entropy (confident) tokens in incorrect responses. Rigorous experiments demonstrate that RL-ZVP significantly outperforms GRPO. Notably, it also surpasses modern filtering-based baselines (e.g., GRPO-DS, GRESO), even when those baselines are allocated 3–5 times more computational budget for rollouts.

### Strengths
The paper's primary strength lies in its novel and intuitive solution to a practical and well-defined problem. The entropy-guided advantage shaping mechanism is a well-motivated and insightful design that effectively rewards productive, uncertain reasoning steps while penalizing confident errors. This strong methodology is validated by a particularly rigorous experimental setup. The authors compare RL-ZVP not only to the standard GRPO but also to two strong baselines (GRPO-DS, GRESO) specifically designed to filter the problematic prompts. By testing under both equal-rollout (fair) and equal-gradient-step (baseline-favored) conditions, the paper offers compelling evidence for RL-ZVP's superior performance and efficiency. This quantitative strength is substantiated by in-depth analysis, including an examination of training dynamics that explains its enhanced stability (leveraging signals from both easy and hard prompts at different training stages) and qualitative examples (Appendix D) that demonstrate a tangible improvement in reasoning correctness, not just stylistic complexity.

### Weaknesses
1）The method proposed is highly sensitive to the $\alpha$ hyper-parameter. The experiments results in Table 3 demonstrate that the optimal performance window for the scaling factor $\alpha$ is very narrow. This raises concerns that the method may require extensive and precise fine-tuning when applied to different datasets or models.

2) The description on negative prompts seems confusing. The logic for advantage shaping (Section 3.2) is asymmetrical. Positive prompts reward high-entropy tokens far more than  negative prompts penalize high-entropy tokens. Could the author explain why use this setting?

3 I think it will be better to provide more sufficient ablation Studies. The ablation study Table 2, while strong, omits a key comparison. It does not test a symmetrical penalty formula for negative prompts (e.g., $\hat{A} = -\alpha H_{i,t}$, which would penalize high-entropy error tokens more to see how it would affect performance.

### Questions
See weakness

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
The paper aims to leverage Zero-Variance Prompts (consisting of fully correct or fully incorrect responses) during RL to enable more efficient learning. The authors propose using token-level entropy-based advantage updates for these responses. Extensive experiments are conducted to demonstrate the effectiveness of the method on reasoning tasks.

### Strengths
- The paper is well written and easy to follow.

- The proposed method of assigning advantages to tokens in responses to zero-variance prompts is intuitive and well motivated.

- Extensive experiments are conducted to demonstrate the effectiveness of the method on reasoning tasks.

### Weaknesses
- Significance of the problem: Are there any empirical statistics on the proportion of zero-variance prompts? I notice that the authors only train 8B models on the harder DAPO-Math-17k dataset. It would be helpful to also show empirical results on Math with 8B models (scenarios with fewer zero-variance prompts) to evaluate the effect of the newly designed advantage. We might expect that it does not hurt performance in scenarios with fewer zero-variance prompts, while providing greater benefits when the number of zero-variance prompts is larger.

- How is it ensured that the advantages assigned to tokens in zero-variance prompts are on the same scale as normal advantages? Could differences in scale affect training stability?

- How should an appropriate value of α be chosen? Are there any guidelines or heuristics for selecting it?

### Questions
Please refer to the weakness part.

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
The paper targets a failure mode in RLVR training (e.g., GRPO) where prompts whose rollouts are uniformly correct or uniformly incorrect yield zero reward variance, causing normalized advantages to collapse and producing no gradient signal. The authors propose RL-ZVP, a simple modification that preserves the standard GRPO update when variance is nonzero and substitutes an entropy-guided, sign-corrected token-level advantage when variance is zero. Intuitively, positive zero-variance prompts should reinforce the behavior, while negative ones should be discouraged; using token entropy modulates the update to emphasize informative positions. Experiments on multiple math benchmarks with Qwen backbones report consistent improvements over GRPO under comparable training setups and show smoother training dynamics.

### Strengths
- **Well-motivated objective:** The paper clearly identifies a practical inefficiency in GRPO, zero-variance prompts producing zero gradients, then directly targets it with a minimal change to the loss. Conceptually, reclaiming learning signal from these otherwise “wasted” batches is appealing because it improves sample/compute efficiency without redesigning the overall RLVR pipeline. The proposed branch only activates when the variance is zero and leaves GRPO behavior untouched elsewhere, so the method respects the existing training regime rather than replacing it.
- **Solid empirical results:** Across several math datasets and multiple model sizes, the method reports consistent gains over GRPO under comparable setups. The improvements appear meaningful rather than marginal, and the training dynamics presented suggest more stable progress rather than spiky or brittle behavior.

### Weaknesses
- **Checkpoint selection leans to optimistic:** The evaluation mixes “best checkpoint during training” with other reporting choices, which can yield optimistic numbers and makes comparisons harder to interpret. It would be beneficial to follow the ML practice where checkpoints are chosen based on a held-out validation set, and results are reported from a single, consistently selected checkpoint (not a mixture).
- **Limited training-curve comparisons:** Training curves (Fig. 4) are shown only for GRPO and the proposed method. Including additional baselines would clarify whether RL-ZVP’s dynamics are uniquely beneficial or simply reflect broader trends across methods, which helps to better assess the proposal approach.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
