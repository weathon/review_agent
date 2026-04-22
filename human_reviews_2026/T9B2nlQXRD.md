# Clip Your Sequences Fairly: Enforcing Length Fairness for Sequence‑Level RL

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
We propose FSPO (Fair Sequence Policy Optimization), a sequence-level reinforcement learning method for LLMs that enforces length-fair clipping on the importance-sampling (IS) weight. We study RL methods with sequence-level IS and identify a mismatch when PPO/GRPO-style clipping is transplanted to sequences: a fixed clip range systematically reweights short vs. long responses, distorting the optimization direction. FSPO introduces a simple remedy: we clip the sequence log-IS ratio with a band that scales as $\sqrt{L}$. Theoretically, we formalize length fairness via a Length Reweighting Error (LRE) and prove that small LRE yields a cosine directional guarantee between the clipped and true updates. Empirically, FSPO flattens clip rates across length bins, stabilizes training, and outperforms baselines across model sizes and evaluation datasets, with the largest gains on the Qwen3‑8B‑Base model.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a sequence-level reinforcement RL method designed for Large Language Models. The authors identify a mismatch in current methods (like GRPO and RLOO) that apply PPO-style fixed-range clipping to sequence-level importance-sampling rations. They argue that longer responses are clipped more frequently distorting the optimization direction. FSPO addresses this by clipping the sequence log-IS ratio using a dynamic band that scales with length.  FSPO is evaluated on math reasoning tasks using Qwen3 base models (1.7B and 8B), showing improved stability, flatter acceptance rates across length bins, and increased performance compared to the baselines.

### Strengths
FSPO preserves IS semantics while restoring fairness via the length scaled band. Another strenght is the theoretical contribution - the paper formalizes the problem via lenght reweighting error and provides a theorem linking this to update direction fidelity. It grounds its solution in the asymptotic Gaussian law of sequence log-IS ratios. The method shows improvement over strong baselines across multiple benchmarks and model scales, with the most significant improvements on harder tasks (AIME24/25) and larger models (8B)

### Weaknesses
The evaluations domain is a bit limited as it only considers math reasoning tasks. And while there are some gains in performance they are not huge. However the principled approach is nice.  Another limitation might be the drift assumption as this could change in different experimental settings.

### Questions
how do you expect the drift assumption to hold in significantly different experiment settings?

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
This paper introduces FSPO which  propose a length-scaled clipping band in log-IS space to equalize acceptance rates across sequence lengths. Empirical results on math benchmarks (MATH500, AIME24/25) show FSPO outperforms existing methods.

### Strengths
1. this paper address a critical issue in RLVR.
2. method is intuitive and simple to implement.

### Weaknesses
1. Author did not thoroughly study the clipping hyperparameters and their effect. This is crucial because the choice of clipping band can significantly influence variance reduction and fairness.

2. Some assumptions may be too strict. For assumption 3.1, clipping may affect sequences with high variance and especially when the data is limited.  In such cases, clipping might distort the distribution and negatively impact learning.

### Questions
1. As Figure 2 "The scale gap between the theoretical and empirical curves is expected due to the asymmetry between the upper and lower clip ranges in implementation." Can adaptively adjust the clip band further improve performance e.g. reduce the gap? An adaptive strategy might better handle varying sequence lengths and variance. 

2. Figure 4 has very large variance which makes hard to make reasonable conclusion, can author consider plot it with confidence interval or std? This would help clarify the statistical significance of the observed patterns and better support the claims regarding length fairness.

3. In Table 3, author list upper clip, lower clip and dual clip. Can author give more explanation on how this dual clip affect the fairness or stability?

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
This paper proposes FSPO (Fair Sequence Policy Optimization), a novel sequence-level reinforcement learning method for large language models (LLMs) that addresses length bias in importance sampling (IS) weight clipping. The authors identify that fixed clipping ranges in existing methods like PPO/GRPO lead to systematic reweighting of short versus long responses, distorting optimization. FSPO introduces a √L-scaled clipping band on the sequence log-IS ratio to enforce length fairness, formalized through a Length Reweighting Error (LRE) metric. Theoretically, small LRE ensures a cosine directional guarantee between clipped and true updates. Empirically, FSPO is evaluated on mathematical reasoning tasks using Qwen3 models, showing improved performance over baselines on benchmarks like MATH500 and AIME24/25, with flattened clip rates across length bins and stabilized training.

### Strengths
- Novelty: FSPO is the first method to explicitly address length bias in sequence-level IS clipping, filling a critical gap in RLVR literature.
- Theoretical Foundation: The LRE metric and cosine guarantee provide a rigorous basis for length fairness, supported by Markov chain CLT.
- Empirical Validation: Comprehensive experiments on math benchmarks show consistent improvements, with strong ablation studies and diagnostics.
- Practicality: FSPO is easy to implement and compatible with existing RL components, making it accessible for real-world applications.
- Reproducibility: Detailed configurations and open-source references facilitate replication.

### Weaknesses
- Limited Scope: Experiments focus solely on mathematical reasoning tasks; generalization to other domains (e.g., code generation or tool-use) is not verified.
- Compute Constraints: Hyperparameter tuning is limited due to resource costs, potentially affecting optimality across diverse settings.
- Assumption Dependency: Theoretical guarantees rely on assumptions like bounded stratification, which may not hold in all scenarios.
- Empirical Drift Simplification: Setting μ≈0 for drift terms is justified empirically but might not generalize to policies with large KL divergence.

### Questions
- How does FSPO perform on non-mathematical tasks, such as code generation or dialogue, where length distributions may differ?
- Could the √L scaling be adapted dynamically based on task-specific length variance, rather than using a fixed σ estimate?
- What are the implications of the bounded correlation assumption (Assumption 3.2) in practice? Are there cases where it might fail?
- How sensitive is FSPO to the choice of c (clip scale) across different model architectures or reward functions?
- Have the authors considered combining FSPO with advanced advantage estimators for further gains?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes FSPO (Fair Sequence Policy Optimization), a sequence-level reinforcement learning method for large language models (LLMs) that addresses the issue of length bias in importance sampling (IS) weight clipping. The authors identify that fixed clipping ranges in existing sequence-level RL methods (e.g., RLOO, GSPO) disproportionately affect sequences of different lengths, leading to unstable training and suboptimal performance. FSPO introduces a length-scaled clipping mechanism. Theoretical analysis formalizes this via Length Reweighting Error (LRE), linking small LRE to directional fidelity in policy updates. Empirical results on mathematical reasoning tasks (MATH500, AIME24/25) demonstrate that FSPO stabilizes training, flattens clip rates across lengths, and outperforms baselines, especially on larger models (e.g., Qwen3-8B).

### Strengths
1. The paper clearly articulates a critical underexplored issue—length-dependent bias in sequence-level RL clipping—and formalizes it through LRE, providing theoretical grounding.

2. FSPO is a simple yet effective modification to existing methods, requiring minimal changes (e.g., plug-in log-space clipping) while maintaining compatibility with RL frameworks like GRPO.

3. Experiments are comprehensive, covering multiple model sizes (1.7B/8B), benchmarks, and baselines. Diagnostic plots (e.g., clip fraction vs. length) convincingly validate the method’s fairness claims.

4. FSPO achieves consistent gains, with notable improvements on harder tasks (AIME) and larger models, suggesting scalability and practical utility.

### Weaknesses
1. The method relies on a tuned scaling factor, which may require pilot runs for new settings. The paper notes compute constraints limited hyperparameter search, raising questions about robustness.

2. Baseline Comparisons: While FSPO outperforms RLOO/GSPO, ablations show that simply widening the clip range fails, but more analysis on why FSPO's scaling is optimal is needed.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3
