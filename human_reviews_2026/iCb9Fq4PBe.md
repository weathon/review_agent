# Beyond Pass@k: Breadth-Depth Metrics for Reasoning Boundaries

- Decision: Reject
- Scores: 4, 0, 4, 4

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful paradigm to improve Large Language Models on reasoning tasks such as coding, math or logic. To assess the reasoning boundary (the fraction of problems a model can solve) researchers often report pass@k at large sampling budgets. Recent results reveal a crossover phenomenon: while RLVR models outperform the base model at small k values, the base model usually outperforms them when sampling a very large number of completions. This has been interpreted as evidence that base models have a larger reasoning boundary. We argue that on tasks with discrete answer spaces, such as math with numeric outputs, pass@k at large k reflects the increasingly higher chance of success in the limit of the number of trials rather than genuine reasoning, and can therefore be misleading. We propose cover@tau, which measures the fraction of problems that a model can solve for which at least a tau proportion of completions are correct. Unlike pass@k, cover@tau captures reasoning under an explicit reliability threshold: models that rely on random guessing degrade rapidly as tau increases. We evaluate several RLVR models using cover@tau-based metrics and illustrate how the relative rankings of popular algorithms change compared to pass@1, offering a different perspective on reasoning boundaries.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Cover@τ, a new evaluation metric intended to complement or replace Pass@k when assessing the reasoning performance of Reinforcement Learning with Verifiable Rewards (RLVR) models. The authors argue that Pass@k can be misleading for tasks with small discrete answer spaces, such as mathematical reasoning, because increasing k artificially raises success rates through random chance rather than genuine reasoning ability. They formally define Cover@τ as the fraction of tasks for which at least a τ fraction of completions are correct, show that Pass@k is a Beta-weighted integral of Cover@τ, and empirically evaluate various RLVR models (GRPO, GSPO, PPO-GAE, KL-Cov, GRPO-Unlikeliness) on OMEGA and Reasoning Gym datasets. The results suggest that Cover@τ provides a more nuanced view of “breadth” versus “depth” of reasoning capabilities, favoring exploration-preserving algorithms such as KL-Cov.

### Strengths
###  Clear problem statement and motivation:

The paper articulates a genuine issue with Pass@k saturation for small answer spaces and provides both intuitive and formal justification.

### Elegant theoretical connection:

The derivation showing Pass@k as a weighted average over Cover@τ via a Beta(1, k) distribution is mathematically neat and offers interpretability.

### Visualization clarity:

The Pass@k vs. Cover@τ plots (Figures 1–3) effectively demonstrate the saturation and trade-off phenomena, supporting the paper’s main claims.

### Weaknesses
### Limited novelty beyond metric reformulation:

While Cover@τ is conceptually appealing, it is essentially a restatement of existing majority or consistency metrics (maj@k, cons@k) with a continuous threshold. The main novelty lies in the interpretation, not in the metric itself. This weakens the contribution for a top-tier venue.

### Overstated theoretical contribution:

The mathematical results (Proposition 1–2 and corollaries) are straightforward consequences of calculus and probability theory. They provide insight but do not represent new theory on reasoning evaluation or RLVR optimization.

### ncomplete problem diagnosis: 
The paper identifies a real issue with Pass@k but the solution doesn't fully address it:

- The problem occurs specifically when answer spaces are small and uniformly distributed. The paper doesn't systematically characterize when Pass@k is problematic vs. when it's fine.

- For many reasoning tasks (especially open-ended or continuous domains), the random guessing problem is minimal, making the motivation less universal than presented.

- No analysis of task properties that determine whether random guessing is a concern.

### Practical utility unclear:

- The paper shows that rankings can differ across τ values, but doesn't establish when practitioners should care about different τ levels.

- No clear recommendation for which τ to use. Should we report a τ profile or a single number? If a single number, how do we choose it?

- The claim that Cover@τ is "more informative" is not well-supported. Information ≠ utility. Practitioners need guidance on decision-making, not just more curves.

### Questions
I know my review is kind of critical. But would be pleased to see your response over:

- In practical applications (e.g., code generation, math competition problems), how often is random guessing actually a problem?

- How should practitioners choose τ for their use case?

- Can you provide guidance on when Cover@τ should replace Pass@k vs. complement it?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper introduces Cover@τ, a reliability-thresholded extension of Pass@k that measures the fraction of problems solved with at least a τ proportion of correct completions. The metric is used to analyze reasoning boundaries of RLVR models and slove limitations of Pass@k at large sampling numbers.

### Strengths
1. The paper is clearly written, with well-organized sections and consistent notation.
2. The topic of evaluation metrics for reasoning stability is important for RLVR research community.

### Weaknesses
The core idea of thresholded generalization of Pass@k is not novel. The recently published ACL 2025 paper “Are Your LLMs Capable of Stable Reasoning?” introduced an almost identical formulation, G-Pass@k and mG-Pass@k, with the same motivation and nearly equivalent equations. That paper also showed that Pass@k is a special case as τ→0 and provided extensive stability analyses.

The current submission does not cite or discuss that prior work, which gives a misleading impression of originality. Mathematically, Cover@τ and G-Pass@kτ differ only in notation. Hence, the contribution appears incremental without proper acknowledgment or comparison.

### Questions
As mentioned in weaknesses.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Cover@τ, a new evaluation metric for assessing reasoning abilities of LLMs that accounts for reliability thresholds. The authors argue that Pass@k at large k is misleading for tasks with discrete answer spaces, as it conflates lucky guesses with genuine reasoning. They demonstrate that Pass@k is a Beta(1,k)-weighted average of Cover@τ, introduce the OMEGA and Reasoning Gym benchmarks, and evaluate several RLVR methods showing different breadth-depth trade-offs.

### Strengths
1. **Clear problem formulation**: Effectively articulates the degeneracy of Pass@k 
   at large k with concrete examples (Figure 1), showing how models can achieve 
   Pass@k=1 through random guessing on tasks with small answer spaces. The Beta(1,k)-weighted interpretation of Pass@k 
   (Proposition 1) provides valuable insight into why Pass@k is biased toward low-τ 
   regions, emphasizing "lucky hits" over reliability. Proposition 2 shows that 
   Cover@τ dominance implies Pass@k dominance but not vice versa.

2. **Breadth-depth framework**: The explicit trade-off between coverage (low τ) and 
   reliability (high τ) is conceptually clean and practically interpretable.

3. **Systematic RLVR evaluation**: First comprehensive comparison of RLVR methods 
   (GRPO, PPO, GSPO, KL-Cov, Unlikeliness) using reliability-aware metrics, revealing 
   that entropy-preserving methods (KL-Cov) achieve better stability.

### Weaknesses
1. **Insufficient acknowledgment of prior work**: The recent work "Are Your LLMs 
Capable of Stable Reasoning?" (Liu et al., arXiv:2412.13147, Dec 2024) proposed 
G-Pass@k, which measures the same concept—coverage at reliability threshold τ. 
While your Cover@τ provides cleaner theoretical formulation and focuses on RLVR 
methods, the submission should:
- Discuss the relationship between Cover@τ and G-Pass@k
- Clarify whether this is concurrent/independent work
- Compare the estimation approaches (your implicit empirical frequency vs. their 
  explicit hypergeometric distribution)
). 
2. **Shallow RLVR analysis**:
Why does preventing entropy collapse (KL-Cov) improve Cover@τ specifically, how do training hyperparameters affect the breadth-depth trade-off, what Cover@τ profile should practitioners target during training? 
3. **Missing critical experiments**: Data contamination/overfitting effects on Cover@τ (present in G-Pass@k)
Sample complexity analysis: how does estimation variance change with n and k?
Ablation on AUC+ vs. alternative aggregation methods
4. **Limited practical guidance**: The paper shows Cover@τ reveals different rankings but doesn't provide actionable recommendations for τ selection based on application requirements.

### Questions
See the questions in **Weaknesses** part

### Soundness
2

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
4

### Summary
The paper argues that Pass@k at large k can be misleading for reasoning tasks with discrete answer spaces, because success can arise from random guessing without reliable reasoning. The authors propose Cover@τ, the fraction of problems for which at least a τ proportion of sampled completions are correct. They prove that Pass@k is a Beta weighted average of the Cover@τ curve, so Pass@k emphasizes very low τ and ignores reliability, and they also introduce a pairwise comparison measure AUC⁺_cover. Experiments on OMEGA and Reasoning Gym show that rankings of RLVR methods change when we look at coverage under reliability thresholds, with methods that preserve exploration and entropy doing better at higher τ. The crossover plots and the tables in this paper support the claim that Cover@τ reveals different capability trade-offs than Pass@k.

### Strengths
1. Cover@τ provides a simple but powerful reliability controlled view. The identity that Pass@k is a Beta weighted integral over τ makes the bias of large k very clear and easy to communicate to practitioners. This theory part is clean and correct from first principles.
2. On OMEGA Probability, the figure on page 2 shows Pass@k saturates as k grows because the answer support is small, while the Cover@τ curve decreases smoothly and separates models by reliability.
3. The work addresses the widely reported crossover between RLVR and base models under Pass@k and provides a principled reason why this happens.

### Weaknesses
1. To estimate per-problem success rates, the paper uses very large K, up to 8196 samples, but the statistical uncertainty of the empirical proportions is not analyzed. Confidence bands for Cover@τ and sensitivity to K and temperature are important for fair comparisons, especially when τ is small.
2. All main results are on math with numeric answers. Code tasks with test suites and other verifiable domains would strengthen the claim that the metric generalizes across RLVR use cases. The ongoing literature already evaluates both math and code when discussing reasoning boundaries.
3. Experiments start from a single family and size, Qwen 2.5 7B Instruct, so it is hard to know if the findings hold for other families or larger models. For example, due to contaminations, Qwen2.5 7B may perform very differently from Llama. Several reports show that behavior under RLVR depends on the base model and training recipe.

### Questions
1. How sensitive are your Cover@τ curves to the sample size K, decoding temperature, and nucleus top p. Can you add bootstrap confidence intervals or a Bayesian treatment that reports uncertainty on coverage across τ.
2. Can you include a code generation study, for example HumanEval or other verifiable suites, and compare Cover@τ with CoT Pass@k and maj@k there. This would help show whether the metric behaves similarly beyond numeric answers.
3. Could you report results on at least one more base model family or size. Some recent papers suggest that the interplay between RLVR and base distribution varies across models.

My final rating depends on the rebuttal.

### Soundness
3

### Presentation
3

### Contribution
3
