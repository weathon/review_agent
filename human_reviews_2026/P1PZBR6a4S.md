# SONATA: Synergistic Coreset Informed Adaptive Temporal Tensor Factorization

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Analyzing dynamic tensor streams is fundamentally challenged by complex, evolving temporal dynamics and the need to identify informative data from high-velocity streams. Existing methods often lack the expressiveness to model multi-scale temporal dependencies, limiting their ability to capture evolving patterns. We propose SONATA, a novel framework that unifies expressive dynamic embedding modeling with adaptive coreset selection. SONATA leverages principled machine learning techniques for efficient evaluation of each observation for uncertainty, novelty, influence, and information gain, and dynamically prioritizes learning from the most valuable data using Bellman-inspired optimization. Entity dynamics are modeled with Linear Dynamical Systems and expressive temporal kernels for fine-grained temporal representation. Experiments on synthetic and real-world datasets show that SONATA consistently outperforms state-of-the-art methods in modeling complex temporal patterns and improving predictive accuracy for dynamic tensor streams.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces SONATA, a novel framework that integrates LDS-based dynamics, multi-faceted coreset selection, and online Bayesian inference for dynamic tensor stream analysis. It combines expressive temporal modeling with adaptive coreset selection to address two fundamental challenges: capturing complex, multi-scale temporal dependencies and achieving computational efficiency in high-velocity streaming settings. SONATA models entity embeddings using Linear Dynamical Systems with Matérn kernels  to represent multi-scale dynamics. Each entity’s latent state evolves via a stochastic differential equation, with embeddings projected linearly for tensor entry prediction under a CP decomposition structure. The proposed synergistic coreset selection strategy  frames selection as a sequential decision-making problem with four criterions that maintains a compact and informative subset of the stream. For inference, SONATA employs an online Expectation Propagation algorithm, where the coreset guides the update process by emphasizing high-value observations, enabling efficient and adaptive learning.

Extensive experiments on synthetic and real-world datasets  demonstrate that SONATA consistently outperforms many static and streaming baselines in predictive accuracy (RMSE and MAE) while maintaining computational efficiency. Ablation studies further confirm the necessity of each coreset criterion and the framework’s robustness to hyperparameter variations. Overall, the paper offers an interesting solution for temporal tensor decomposition.

### Strengths
1.The paper is well-written, clearly outlining the problem and the limitations of existing methods, proposing a solution, and demonstrating robust performance.

2.The idea of a synergistic coreset selection strategy, specifically designed to capture evolving dynamics and ensure long-term utility, is  novel and well-motivated. It aims to enhance computational efficiency while  improving model performance.

3.Experiments demonstrate strong performance against various baselines and robustness across a variety of datasets.

4.The ablation study of synergistic coreset selection strategy is convincing and reasonable.

### Weaknesses
1. The use of Linear Dynamical Systems: the evolving parameters F,H,L are time-invariant, which limits the model’s expressiveness in capturing complex dynamics.

2. Previous work has already employed SDE-represented Gaussian Processes to model the entities and enable streaming inference (e.g., SFTL; Fang et al., 2023), so the modeling is of fairly limited novelty.

3. The scalibity of the proposed method over data size has not been explored in experiments.

### Questions
1. Can this method be directly applied to data with irregularly sampled timesteps? If not, how could it be adapted to handle this more realistic scenario?

2. The authors emphasize the computational efficiency of the proposed method but only report computational time. How does the method scale with respect to the number of data points?

3. Why SONATA is able to capture multi-scale temporal dynamics, is there any special design?

4. I am curious about the motivation for designing the importance score using four criteria: uncertainty, influence, novelty, and information increment. Could the authors provide further explanation? Conducting ablation studies on different combinations of these four criteria may better illustrate the design rationale.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose SONATA, a novel framework that integrates expressive dynamic embedding modeling with adaptive coreset selection. SONATA employs principled machine learning techniques to efficiently evaluate each observation for uncertainty, novelty, influence, and information gain. It dynamically prioritizes learning from the most valuable data using Bellman-inspired optimization.

### Strengths
This paper integrates Linear Dynamical Systems (LDS) with temporal kernels to capture fine-grained and hierarchical temporal dependencies, addressing a key limitation of existing methods.

It enables adaptive representation of evolving patterns across different time scales (short-term vs. long-term dynamics).

Bellman-inspired optimization ensures optimal data selection, improving scalability for high-velocity streams.

### Weaknesses
While coreset selection improves efficiency, the evaluation of uncertainty, novelty, and influence for each observation may introduce latency in real-time systems.

The performance of temporal kernels and LDS models may depend heavily on hyperparameter tuning.

Performance may degrade with noisy or incomplete streams, as coreset selection relies on accurate uncertainty estimation.

### Questions
The representation of SDE in Figure 1 is different from its descriptioin in this paper.

I can not find the details of multi-scale feature extraction in this paper.

### Soundness
3

### Presentation
3

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
The paper presents SONATA, a streaming tensor factorization framework that models each factor trajectory with a linear dynamical system (LDS) and augments learning with a multi-criteria, martingale-guided coreset. The work integrates (i) continuous-time state-space priors for temporal dynamics and (ii) an online coreset selector that combines uncertainty, novelty, influence, and a martingale increment into a Bellman-optimized score under a fixed budget. Empirically, SONATA is evaluated on several real-world and synthetic datasets and generally attains lower RMSE/MAE than static and streaming baselines; the paper also reports per-iteration/epoch runtime, kernel/length-scale sensitivity, and additional experiments.

### Strengths
Well-motivated architecture. The combination of continuous-time LDS factors with an online coreset mechanism is coherent and addresses a meaningful streaming setting. The design is carefully tied to spatiotemporal structure via Matérn priors and a principled scoring rule. 

Consistent accuracy gains. Experiments report SONATA achieving the best or second-best error across multiple datasets compared to strong static and streaming baselines. 

Useful diagnostics of the coreset. Visual comparisons of coreset vs. non-coreset factor trajectories help understand what the selector retains.

### Weaknesses
Computational fairness. The paper reports per-iteration/epoch runtime for several methods (e.g., SONATA 0.338s, CT-CP 0.018s, CT-GP 0.105s, THIS-ODE 7.190s), but does not equalize wall-clock time across methods when comparing accuracy; faster methods could take many more iterations within the same time budget. 

Compute/memory analysis. The coreset budget directly affects memory footprint and update cost, yet the paper does not provide memory-usage vs. budget or throughput measurements to quantify the trade-off between retained samples and efficiency.

Ablations. The coreset analyses (full-data vs. coreset, random sampling) appear focused on a single dataset in the appendix; extending them to more datasets or reporting summary trends would better calibrate the benefits of the martingale-guided selector over simpler heuristics. 

Hyperparameter. The method exposes several hyperparameters, with dataset-dependent choices. While the paper includes some sensitivity tables, systematic guidance for robust settings in realistic streaming scenarios is limited. 

Readability and consistency. The manuscript’s readability suffers in several places due to redundant wording and inconsistent cross-referencing. For example, equation mentions sometimes read like “Eq. equation” and there is mixing of styles (e.g., “Figure” vs. “Fig.”). Please standardize to a single reference style.

### Questions
Time-normalized comparisons. Please provide convergence curves for all methods and report each baseline’s accuracy at the same total runtime as SONATA. This would directly address efficiency/accuracy trade-offs. 

Memory and budget trade-offs. Please report memory usage and update time as functions of the coreset budget and include accuracy vs. budget curves on multiple datasets. As of now, no budget-sweep experiments are included, which makes it difficult to assess how performance scales with budget.

Ablations vs. simple heuristics. The appendix contrasts coreset selection to random sampling on one dataset. Could you add the same comparison across other datasets and summarize whether martingale-guided selection consistently outperforms random sampling at matched budgets? 

Robust defaults. The current sensitivity tables are promising but seem dataset-specific. Can the authors propose and validate robust default settings?

### Soundness
2

### Presentation
2

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
The paper introduces SONATA, a framework for streaming temporal tensor factorization that integrates: (1) Matérn-kernel-based linear dynamical systems (LDS) for multi-scale temporal modeling, (2) a synergistic coreset selection mechanism combining four criteria (uncertainty, influence, novelty, information gain), (3) Bellman equation inspired long-term value optimization, and (4) expectation propagation (EP) for online Bayesian inference. The approach demonstrates strong empirical performance and computational efficiency compared to existing streaming tensor methods.

### Strengths
1. Well-Motivated Integration with Clear Value Proposition

The paper addresses a problem of balancing expressiveness and efficiency in high-velocity tensor streams.
The 4-criterion coreset selection is principled and goes beyond simple heuristics.
Empirical gains are substantial: 61.5% RMSE reduction vs. SFTL-CP on CA Traffic (Table 1).

2. Strong Empirical Performance Across Multiple Dimensions

The authors reprorted statistically significant improvements (p<0.05) across 4 diverse datasets.
The proposed method is faster than processing all data, ~21× faster than THIS-ODE (Fig. 3c) while maintaining superior accuracy.
Bayesian uncertainty quantification (Fig. 2) and smooth factor trajectories provide domain-interpretable representations unavailable in deep learning alternatives.

3. Solid Technical Execution

The Bellman-equation-based long-term coreset optimization (Eq. 17) is conceptually novel in streaming tensor factorization.
Discount factor analysis (Table 2) shows data-dependent optimal strategies, suggesting principled adaptability.
Code availability enhances reproducibility.

### Weaknesses
1. Possibliity of Insufficient Related Work and Missing Key Comparisons
The paper seems to omit important recent work that directly challenges its novelty claims.
- OnlineGCP (SIGMOD'23): Generalizes streaming CP to exponential family distributions (Poisson, Bernoulli), directly addressing SONATA's Gaussian limitation. Authors should discuss why Gaussian assumptions are sufficient or provide non-Gaussian extensions

- SOFIA (ICDE'21): Incorporates seasonality + outlier/missing data robustness
- OR-MSTC (IJCAI'19): Multi-aspect streaming with outlier separation via ADMM
SONATA's robustness claims are unsupported without comparison to these methods

- SBDT (ICML'21): Streaming Bayesian deep tensor factorization with spike-and-slab priors
The "without deep neural networks" claim needs explicit comparison to justify EP over deep Bayesian approaches

2. Insufficient Ablation Studies to Support Synergistic
The paper does not demonstrate that all components are essential.
To support their claims, authors should conduct these experiment:
-  Individually remove each coreset criterion (uncertainty/influence/novelty/martingale)
-  Bellman optimization vs. greedy myopic policy (γ=0)
-  EP vs. simpler online filters 
-  Matérn vs. simpler kernels (RBF, linear)

3. Minor Issues
- Coreset budget selection: Mmax seems manually tuned per dataset (100-3000 in Appendix B.1). they provide no principled guidance.
- How does runtime scale with coreset size Mmax?

### Questions
- Could you please clarify your contribution to non-Gaussian and robust streaming methods?
- Is there a possibility of lacking ablation studies?
- Can you explain your procedure for selecting hyperparameters?

### Soundness
2

### Presentation
2

### Contribution
2
