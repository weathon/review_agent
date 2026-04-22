# LSPO: Length-aware Dynamic Sampling for Policy Optimization in LLM Reasoning

- Avg Score: 2.50
- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
Since the release of Deepseek-R1, reinforcement learning with verifiable rewards (RLVR) has become a central approach for training large language models (LLMs) on reasoning tasks. Recent work has largely focused on modifying loss functions to make RLVR more efficient and effective. In this paper, motivated by studies of overthinking in LLMs, we propose Length-aware Sampling for Policy Optimization (LSPO), a novel meta-RLVR algorithm that dynamically selects training data at each step based on the average response length. We evaluate LSPO across multiple base models and datasets, demonstrating that it consistently improves learning effectiveness. In addition, we conduct a detailed ablation study to examine alternative ways of incorporating length signals into dynamic sampling, offering further insights and highlighting promising directions for future research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces LSPO, a reinforcement learning algorithm aimed at enhancing reasoning in LLMs. Motivated by the overthinking issue in LLMs, LSPO uses dynamic sampling based on response length to improve model effectiveness. It selectively retains prompts with the shortest and longest responses, focusing on data likely to be uncertain or overlength. The paper demonstrates that LSPO consistently improves the effectiveness of LLMs trained on reasoning tasks, including various challenging datasets and models. An ablation study is also provided to examine different length-based sampling strategies.

### Strengths
1. **Originality:** The perspective is fresh on dynamic data sampling in RLVR, using response length as a heuristic to guide training.
2. **Clarity:** The explanations of the algorithm and its components are clear, with well-documented experiment settings and discussions.
3. **Potential:** This work has the potential to influence future RLVR strategies and could lead to more efficient RL on LLMs.

### Weaknesses
1. **Training efficiency:** While LSPO improves the effectiveness of the model, its sampling process adds significant extra computation time per step, which could be a drawback in large-scale training scenarios.
2. **Response length limitation:** The filtering mechanism based solely on length may not generalize well if the model were required to generate fixed-length responses (e.g., for reasoning effort manipulation, creative writing, and other length-related tasks). Further work is needed to adapt LSPO to such constraints. 
3. **Applicability to broader scopes:** The current formulation may not extend well to other types of RL tasks or environments where response length is not primarily correlated with performance. For instance, in coding tasks, writing Rust/C++ snippets inherently requires longer responses during reasoning compared to Python/JavaScript and other script-like languages. 
4. **Limited interpretability:** The rationale behind why length-based sampling improves RL dynamics is not deeply explored. More analysis on the relationship between response length sampling and RL dynamics would strengthen the paper's contributions.

### Questions
1. What does $T_{min}$ stand for in Line 417?
2. How does RL dynamics change under the same training steps in Figure 2? I understand that controlling the training time is valid but not informative enough.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes LSPO, a meta dynamic sampling algorithm for RLVR to improve reasoning performance of large language models. LSPO leverages response length as a heuristic signal and dynamically filters training examples, retaining prompts with either the shortest or longest sampled responses and discarding those in the middle.

### Strengths
- The algorithm is practical and compatible with any RLVR pipeline.
- Good discussion of limitations and future research.

### Weaknesses
- The approach is primarily heuristic, and the paper lacks a deeper theoretical explanation of why focusing on the shortest and longest responses drives stronger policy updates. More evidence or analysis is needed to justify this selection mechanism beyond empirical observation.
- The algorithm filters out prompts with medium length which could lead to under-training on those prompts.
- GSPO is used as a baseline throughout the paper, but the method is not introduced or sufficiently explained.
- For Llama-3.2-4B, only training curves are shown. No test-set evaluation results are provided, making the claim of cross-model generality incomplete.
- The experimental scale is relatively limited. Qwen models are capped at 2048 tokens and Llama at 3072 tokens, which is short for reasoning-focused RLVR. Since the core contribution relies on response length filtering, the short context limits the generalization of these findings to more realistic long-reasoning settings.
- Performance improvements are small (<2% in many cases). Additional trials with multiple seeds and reporting mean and std would help validate whether improvements are statistically significant rather than noise.
- Typos:
    - Equation 7 missing a summation sign.
    - Equation 10 missing ']'

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes LSPO (Length-aware Sampling for Policy Optimization), a dynamic data sampling strategy for reinforcement learning with verifiable rewards (RLVR) in LLM reasoning. LSPO filters training prompts based on the average length of generated responses, retaining only those with the shortest or longest outputs under the hypothesis that these extremes are most informative for policy improvement. The method is evaluated across multiple base models (Qwen, Llama) and RLVR algorithms (GRPO, DAPO, GSPO), reporting modest but consistent gains in final accuracy on math reasoning benchmarks.

### Strengths
- The core idea—leveraging response length as a signal for dynamic data selection—is intuitive and grounded in prior observations about “overthinking” in LLMs.
- LSPO is designed as a meta-algorithm, making it compatible with various RLVR base methods, which enhances its practical applicability.
- The ablation studies systematically explore design choices (e.g., percentile vs. value thresholds, retained length ranges), providing useful empirical insights into the role of response length in RLVR training dynamics.

### Weaknesses
- Unclear problem formulation and unsubstantiated core assumption: The paper’s central motivation rests on the claim that “intermediate-length responses are less informative” and that filtering them improves final model effectiveness. However, this key assumption is never empirically validated. The ablation study shows that training only on intermediate-length responses yields lower performance, but this does not imply that including them in a full-data setting harms learning—yet the filtering strategy is justified precisely on that implicit premise. Without evidence that intermediate-length samples actively degrade training (e.g., via gradient conflict analysis, noise estimation, or ablation with full-data vs. filtered-data under matched compute), the rationale for discarding them remains speculative. Moreover, the paper does not clearly articulate what concrete learning problem LSPO solves: Is it combating overthinking? Improving sample efficiency? Enhancing generalization on hard examples? The lack of a well-defined objective makes it difficult to assess whether LSPO meaningfully advances the field or merely implements a heuristic with marginal gains.
- Insufficient comparison with prior dynamic sampling methods: The paper claims that existing dynamic sampling approaches (e.g., GRESO, PODS) only improve efficiency, not final model effectiveness, yet provides no experimental evidence to support this assertion. A direct comparison of LSPO against these methods in terms of final accuracy is missing.
- Marginal gains relative to overhead: The reported improvements are small, while LSPO incurs nontrivial rollout overhead. The cost-benefit trade-off is not convincingly justified.
- Incomplete empirical validation: Training curves are only shown for GSPO + LSPO vs. GSPO. Curves for GRPO and DAPO (the other two base algorithms) are absent, making it difficult to assess whether LSPO consistently accelerates learning or merely shifts the final performance slightly.
- Incomplete training dynamics analysis: The paper presents training curves for only two specific configurations: (1) Qwen-2.5-Math-7B with GSPO on Olympiad-bench, and (2) Llama-3.2-4B-Instruct with DAPO on MATH. However, the main results in Table 1 span two models, three base algorithms (GRPO, DAPO, GSPO), and three evaluation benchmarks (AIME25, Olympiad, Minerva). Without training curves for other combinations—e.g., GRPO+LSPO on AIME25 or DAPO+LSPO on Minerva—it is unclear whether LSPO consistently accelerates learning or merely yields small final gains under favorable conditions. Given the marginal absolute improvements (often ≤1%), this omission weakens the evidence for LSPO’s general effectiveness.
- Lack of theoretical justification: There is no analysis or proof suggesting that filtering by response length should improve convergence or generalization. The heuristic is plausible but not grounded in optimization or learning theory.
- Technical error in formulation: Equation (3) defining average response length $L(q) := \frac{1}{G} |o_i|$ lacks a summation which is ill-formed.

### Questions
- Problem clarity: The paper claims LSPO improves “final effectiveness” unlike prior dynamic sampling methods, but never specifies what concrete learning problem it solves—e.g., mitigating overthinking, improving hard-example generalization, or accelerating convergence. What is the precise objective that justifies filtering by length?
- Core assumption: The method assumes intermediate-length responses are “less informative,” yet the ablation only shows that training exclusively on them performs poorly. This does not prove they harm full-data training. Can the authors provide evidence (e.g., with matched compute) that excluding them actually helps?
- Gap Justification: Could the authors clarify what specific prior works in RLVR they consider as “dynamic sampling for efficiency only”? Are there no existing methods that use data selection to improve final accuracy? If such methods exist, how does LSPO differ in objective or mechanism?

- Theoretical Grounding: Is there any theoretical rationale (e.g., based on gradient variance, signal-to-noise ratio, or curriculum learning) that explains why filtering by response length extremes should improve policy optimization?

- Experimental Results: 
  - Baseline comparison: The claim that methods like GRESO “do not improve effectiveness” is unsupported by experiments. Given LSPO’s marginal gains (often ≤1%), a direct comparison against GRESO/PODS in final accuracy is essential to validate its novelty.
  - The paper states that “other dynamic sampling methods such as GRESO do not improve effectiveness.” Is this claim supported by experiments? If not, could the authors include a direct comparison of LSPO against GRESO, PODS, or similar methods in terms of final test accuracy?
  - Training curve coverage: Training curves are shown only for two specific (model, algorithm, dataset) combinations. Can the authors provide curves for other main settings in Table 1 (e.g., GRPO+LSPO on AIME25) to support the claim of consistent improvement?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces LSPO (Length-aware Sampling for Policy Optimisation) — a meta-algorithm for reinforcement learning with verifiable rewards (RLVR) in reasoning-focused LLMs. The method dynamically filters prompts during RL training based on response length, keeping the shortest and longest ones (based on percentile thresholds), motivated by empirical observations that overthinking correlates with longer, incorrect reasoning chains. LSPO aims to improve training effectiveness (final accuracy) rather than efficiency. Experiments and reported results show modest accuracy improvements over baselines. 

1. The core idea of filtering samples based on response length is heuristic rather than theoretically grounded. Can the authors provide more intuition on why this is important? 
2. Prior works such already explored length-aware optimisation, including direct regularisation of response length. LSPO’s novelty lies mainly in applying this as a data sampling heuristic, not as a new RL algorithm. The contribution feels incremental, more like a training trick than a principled new method.
3. Reported improvements are marginal. This makes me ask if such gains are within normal RL variance and could be due to stochasticity rather than the method itself. Can the authors conduct statistical significance tests to understand if those are actual gains? 
4. Can the authors report results beyond math reasoning datasets? There are other reasoning domains; would we see such gains there as well? 
5. I am not sure if 24 hrs is actually enough time for RL convergence. Can the authors please elaborate on the choice of this number? Why 24 hrs? Have you seen stable conclusions across the RL experiments? 
6. From my understanding, the proposed method changes the batch composition per step? Is this true? If so, are the comparisons made under fixed time or batch settings? If so, it might be worth controlling those for the number of updates per unique sample.
7. To me, the claims are overstated. I wouldn't agree that this is the first paper to study the length-aware setup. I believe other algorithms have considered this already during optimisation. 
8. The plots need to include variances. What stops me from saying those results in the figure are hand-picked? I think it is important to report variances.

### Strengths
Please see above

### Weaknesses
Please see above

### Questions
Please see above

### Soundness
2

### Presentation
2

### Contribution
1
