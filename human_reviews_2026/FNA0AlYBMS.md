# Top-K Structure Search with Solution Path

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Structure learning algorithms often output a single estimated graph without offering alternative candidates or a way to capture model uncertainty. This is limiting in finite-sample settings with weak signals or noise, where multiple structures can explain the data equally well. In this work, we propose Top-K Structure Search with Solution Path, an algorithm that systematically tracks the evolution of edge weights across a range of values of the $\ell_1$ sparsity regularization parameter $\lambda$. By scoring candidate structures with the Bayesian Information Criterion (BIC), our method ranks and returns the Top-K most plausible structures. Unlike traditional approaches that yield a single solution, our framework provides a ranked set of candidates, enabling better uncertainty assessment. Experiments on synthetic and real-world datasets demonstrate the effectiveness of our approach in capturing structural variability. This highlights the advantage of leveraging solution paths for structure learning, especially in scenarios where committing to a single graph is unreliable. Our framework offers a complementary perspective on structure learning by considering multiple candidate solutions, thereby mitigating the practical instability of solely relying on a single result.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This study proposes the "Top-K Structure Search with Solution Path" algorithm to fix the issue that most Bayesian structure learning algorithms only output one "optimal" graph and fail to capture model uncertainty. It tracks edge weight changes via the L1 sparsity regularization parameter (lambda) to find structural critical points, scores candidates with BIC, and selects top K plausible graphs. With a gradient-optimized objective function (likelihood, L1 penalty, soft acyclicity constraint) and temperature-scaled BIC-based uncertainty quantification, experiments on synthetic (varying samples, variables, density) and Sachs datasets show it outperforms GES/PC/BOSS/Top-K A* in F1/recall for small samples/weak signals/high dimensions, with better scalability for medium-high dimensional networks.

### Strengths
1. Combines L1 regularization solution paths with Top-K selection, avoiding exhaustive graph search and super-exponential complexity of dynamic programming/A*.
2. Provides dual uncertainty quantification (graph-level probability, edge occurrence probability) with temperature scaling for reasonable entropy, boosting interpretability.
3. Systematic experiments (synthetic scenarios, Sachs data) confirm it captures weak edges missed by traditional methods, verifying robustness.
4. Scalable computationally (complexity unrelated to K), handling 60-variable networks where Top-K A* fails.

### Weaknesses
1. This motivation is open to debate, as the core idea of generating multiple candidate graphs seems somewhat forced. In reality, many existing methods can also generate multiple candidate graphs, though they have not been applied in this specific context. From this perspective, the work has certain limitations.
2. Hyperparameter selection lacks data-driven automation. Key hyperparameters like lambda grid granularity (epsilon), Top-K value (K), temperature parameter (T), and edge weight threshold (tau) rely on experience or heuristics. For example, T is set to ensure the probability of the K-th graph equals 1/(2K), and epsilon is required to be "small enough to capture all critical points"—but there is no clear data-driven method to determine these values.
3. Insufficient discussion on the soft acyclicity constraint. The objective function uses the soft acyclicity constraint expressed as "trace of the matrix exponential of the Hadamard product of the adjacency matrix (B) with itself minus the number of variables (d)". While referencing prior work (Zheng et al., 2018), the study does not analyze its applicability in special scenarios: for instance, whether it introduces bias in extremely high-density graphs (e.g., density 0.8) or when there are pseudo-cycles from unobserved confounders; nor does it discuss if combining with hard acyclicity constraints (e.g., node ordering) could improve performance.

### Questions
1. The study states epsilon needs to be "small enough to capture all critical points", but there is no quantitative standard for "small enough"—for example, how does epsilon relate to the number of variables (d) and graph density? Can supplementary experiments compare the impact of different epsilon values (e.g., epsilon = maximum lambda/50, epsilon = maximum lambda/200) on the number of identified critical points and Top-K structure quality (e.g., BIC score stability)? For K selection, only K=1,5,10 are tested; when K exceeds 20, will performance saturate (F1 score no longer improves) or decline (introduce too many low-quality structures)? Is there a way to determine the optimal K based on dataset characteristics?
2. Edge-level uncertainty (probability of edges appearing in Top-K graphs) can be a "soft confidence score", but its practical utility is unproven. For example, on the Sachs dataset, if edges with a confidence score >0.3 are considered high-confidence, can this improve downstream task accuracy (e.g., protein interaction prediction)? How does this uncertainty quantification method compare to existing ones like Bootstrap resampling in computational efficiency and accuracy? Is there a way to validate the reliability of the uncertainty results?

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
This paper introduces a new structure learning framework for linear causal models, named Top-K Structure Search with Solution Path. The method explores multiple candidate DAGs by following the evolution of edge weights along the ℓ₁ regularization path and selecting the top-K graphs based on their BIC scores. The goal is to quantify model uncertainty and capture alternative plausible structures rather than committing to a single learned DAG. Experiments are conducted on synthetic datasets and the Sachs protein network, with comparisons against PC, GES, and BOSS.

### Strengths
1- The paper is mathematically sound and clearly written, with consistent notation and reasonable theoretical grounding.


2- The idea of tracing the solution path for structure learning and selecting the Top-K BIC-scored DAGs is well-motivated for exploring structural uncertainty.


3- The inclusion of uncertainty quantification using temperature-scaled probabilities adds an interesting perspective to the interpretation of multiple candidate structures.


4- The paper is well-organized and technically detailed.

### Weaknesses
1-  The proposed approach extends existing linear DAG learning formulations (notably GOLEM and NOTEARS) by varying the regularization parameter λ and collecting multiple solutions along the path. While the idea of leveraging the Lasso solution path for DAG estimation is interesting, it is a relatively small methodological step beyond prior work. The novelty is modest compared to established continuous optimization frameworks for causal structure learning.

2- The experimental section compares the proposed method only with PC, GES, and BOSS—methods that handle both linear and nonlinear models but use very different principles (constraint- and order-based search).
 Crucially, the paper does not include comparisons with NOTEARS, GOLEM, or other recent continuous-optimization DAG-learning methods, which are the most relevant baselines given that the proposed model is also linear and differentiable. Without these comparisons, it is difficult to assess the relative performance or real improvement of the proposed method.

3- The method is limited to linear structural equation models (SEMs). This restriction significantly limits its generality, as many real-world datasets exhibit nonlinear dependencies.

4- While the Top-K framework for structure learning is conceptually appealing, the practical impact of ranking multiple linear DAGs is unclear. The paper does not convincingly show that the Top-K set provides substantial benefits beyond what standard resampling or Bayesian averaging techniques could achieve.

5- Evaluation focuses on small to moderately sized networks (≤60 variables) and does not demonstrate scalability beyond what has already been achieved by existing gradient-based methods.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a solution-path approach to structure learning for linear Gaussian SEMs that returns a ranked set of K candidate graphs rather than a single estimate. The method traces the evolution of edge weights along a sparsity path (varying the ℓ1 penalty λ), identifies “critical points” where supports change, and selects the Top-K graphs based on BIC after an OLS refit on the selected supports. It also provides graph- and edge-level uncertainty via a temperature-scaled softmax over BIC scores. Experiments on synthetic data and the Sachs protein signaling dataset suggest that the approach improves recall and F1—especially in low-sample, low-SNR regimes. The submission is clear and well framed, and the empirical results indicate measurable benefits in regimes where committing to a single structure is unreliable.

### Strengths
OLS refit on supports is a sensible correction for Lasso shrinkage before BIC scoring.
Synthetic stress tests in low‑n and weak‑edge regimes are well chosen; you vary n, K, density ρ, and d. Gains in recall/F1—and often accuracy—are consistent with the method’s design. Sachs analysis is illustrative: Top‑2 matches baseline skeleton; Top‑7 yields best F1/accuracy, showing the value of exploring K>1.

### Weaknesses
The paper states the objective is differentiable, but the ℓ1 norm is non-differentiable at zero. While the active-set approach and subgradients are often used in practice, you should explicitly acknowledge non-differentiability and clarify whether you use subgradients or a proximal step/soft-thresholding. As written, there is an inconsistency.
Add a concise algorithm box with all steps.
Add continuous‑optimization baselines (NOTEARS, GOLEM, and possibly DAGMA/GraN‑DAG) which are most methodologically comparable.
Since observational data typically identify only MECs, report SHD to CPDAG, orientation precision/recall (when meaningful), and possibly F1 on CPDAG edges. Skeleton‑only is incomplete for many readers.
If GIES appears in figures, clarify whether interventional data in Sachs were used; if not, remove GIES or explain its role.

### Questions
How sensitive are the results to ε, δ, τ, and α?On Sachs: Did you use interventional data, and if so, how was scoring adjusted? If not, why is GIES referenced in figures?Can you provide CPDAG-oriented metrics and SHD to evaluate orientations more fully?On Sachs: Did you use interventional data, and if so, how was scoring adjusted? If not, why is GIES referenced in figures?

### Soundness
3

### Presentation
3

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
The paper introduces Top-K Structure Search with Solution Path, a method for Bayesian structure learning that goes beyond predicting a single graph. Instead, it traces how edges evolve as the L-1 sparsity parameter varies and identifies critical points where structural changes occur. These candidate graphs are then scored using BIC, and the Top-K most plausible structures are returned, allowing better modeling of uncertainty in noisy or limited-sample scenarios. The algorithm uses gradient-based optimization with soft DAG constraints and re-estimates weights to correct Lasso shrinkage. It also provides graph- and edge-level uncertainty estimates. Experiments on synthetic and real-world datasets show improved robustness and accuracy compared to PC, GES, BOSS, and Top-K A*, especially in cases where multiple structures fit the data similarly.

### Strengths
1. Provides a systematic method to generate and rank multiple plausible graph structures instead of relying on a single estimate.
2. Efficiently identifies candidate structures by tracking edge support changes along the $\ell_1$ regularization path with detailed mathematical backing.
3. Demonstrates robustness across synthetic and real-world datasets with comprehensive evaluations over sample size, dimensionality, graph density, and $K$, using clear performance metrics.

### Weaknesses
1. The paper does not engage with Top-K search work beyond Bayesian graphs, limiting its relevance framing.
2. Unclear Hyperparameter Design: Thresholds such as $\delta$ (active set) and $\tau$ (binarization) lack principled justification or sensitivity analysis, making results potentially hyperparameter-dependent.
3. The soft acyclicity penalty may fail to ensure strict DAGs, yet failure cases and practical violations are not analyzed.
4. Using a uniform $\lambda$ grid instead of recovering the full continuous LASSO path can miss critical support transitions.
5. Limited Real-World Evaluation: Only the small Sachs dataset is used, leaving scalability to large real-world graphs untested.
6. No Comparison for Uncertainty Estimation: The uncertainty framework is not compared against Bayesian model averaging, bootstrap, or sampling-based alternatives.
7. The method lacks guarantees that the grid-based candidate set captures all key structures or modes.
8. No Ablation Studies: Important design choices (temperature scaling, $\epsilon$, and $K$) are not ablated despite noted importance.

### Questions
Refer weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
