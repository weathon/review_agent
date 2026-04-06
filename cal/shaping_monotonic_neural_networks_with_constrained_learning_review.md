=== CALIBRATION EXAMPLE 31 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**
The title accurately reflects the contribution. The abstract clearly states the problem, the proposed constrained learning framework, and key advantages (flexibility via chance constraint, no architectural restrictions, no regularization tuning). The claims of "competitive performance" are supported by the experiments section. However, the claim of "needs only small extra computations" is not directly quantified or compared to baselines (e.g., per-iteration overhead for gradient computations). The claim of no pre-processing overlooks the need to set hyperparameters like the chance constraint level α, dual learning rate γ_μ, and sampling size N.

**Introduction & Motivation**
The problem is well-motivated with relevant applications in fairness and safety-critical systems. The categorization of existing methods (architecture vs. regularization) is appropriate. The stated contributions (flexibility via chance constraint, architectural freedom, adaptability) are clear and address gaps in prior work. A minor issue: the introduction could more sharply differentiate the novelty of applying primal-dual methods with a *chance constraint* to monotonicity, versus standard Lagrangian approaches for hard constraints.

**Method / Approach**
This is the core of the paper and has several strengths and critical weaknesses.

*   **Formulation & Claim 1:** The reformulation using a chance constraint is a key idea. Claim 1 provides a sufficient condition for the chance constraint using a continuously differentiable surrogate. The reasoning is intuitive but presented informally; a more formal statement and proof (even in an appendix) would strengthen rigor. More importantly, the tightness of this surrogate approximation is not discussed. How conservative is it? What is the effect on the achieved probability of monotonicity violation? This is a significant gap, as a very conservative surrogate could unnecessarily harm predictive performance.

*   **Algorithm & Practical Implementation:** Algorithm 1 is a standard application of a stochastic primal-dual (Lagrangian) method to the surrogate-constrained problem. The claim of "small extra computations" is questionable: computing the constraint term requires the gradient of the network output w.r.t. specific inputs (∂f_θ(z)/∂z_m). This involves a backward pass through the network for each sampled z (or a specialized gradient computation), effectively doubling the computational graph cost per iteration compared to standard training. This overhead should be acknowledged and, ideally, measured.

*   **Differentiability Assumption:** The method relies on the gradient ∂f_θ(x)/∂x_m existing and being usable in updates. The paper states it works with "commonly used activation functions (e.g., ReLU)". However, ReLU networks are piecewise linear and have undefined second derivatives at knot points. In practice, automatic differentiation will return a subgradient (e.g., 0 at the origin), but the theoretical validity and practical stability of the primal-dual updates in this non-smooth setting are not addressed. This is a notable oversight for a method centered on gradient constraints.

*   **Enforcing Monotonicity on 𝓧:** The paper rightly notes the importance of generalizing monotonicity beyond the dataset. The approach of sampling z ∼ Uni(𝓧) is pragmatic but problematic in high dimensions: uniform sampling becomes inefficient, and the volume of relevant regions near the data manifold is tiny. The paper uses N=128 samples regardless of input dimension; this is likely insufficient for meaningful coverage or enforcement in high-dimensional spaces (e.g., Blog Feedback has 276 features). The method's effectiveness in truly guaranteeing monotonicity over 𝓧 is thus not convincingly established.

**Experiments & Results**
The experimental evaluation is broad, covering synthetic, tabular, and control tasks. However, several concerns affect the strength of the evidence.

*   **Evaluation Protocol:** Reporting the mean and standard deviation of the *best five out of ten runs* is non-standard and introduces a positive bias. Standard practice is to report results over all independent runs or fixed train/validation/test splits. This makes direct comparison with cited results (which may use different protocols) less reliable.

*   **Baseline Comparison & Fairness:** The results in Tables 1 and 2 show the proposed method is often best or competitive. However, the comparison is partially confounded by model size. The authors highlight achieving good performance with fewer parameters, which is a valid point for efficiency. Yet, it's unclear if the performance gains (e.g., in Auto MPG) stem from the constrained optimization method or simply from using a standard MLP architecture that is more expressive than the restricted architectures of some baselines (e.g., Min-Max Net, DLN). A more controlled ablation comparing the proposed training method against a standard MLP trained with a heuristic gradient penalty on the same architecture would better isolate the benefit of the primal-dual chance-constraint framework.

*   **Statistical Significance:** Standard deviations are reported, but no statistical significance tests are performed. Given the often-small differences (e.g., 69.4% vs. 69.3% accuracy on COMPAS), it's unclear if improvements are statistically meaningful.

*   **Control Experiment:** The frequency control experiment is a valuable extension to an unsupervised task. The reported lower objective cost is promising. However, evaluation appears to be based on a single power disturbance scenario (Figure 2). To robustly claim improved performance, results should be aggregated over multiple disturbance scenarios or a held-out test set of disturbances. The observation that SMNN controllers have truncated output ranges (Figure 3) is interesting but highlights an architectural limitation of SMNN rather than a direct flaw the authors' method fixes.

**Writing & Clarity**
The paper is generally well-written and logically structured. The method derivation is clear. Some minor issues: The notation in Section 3.1 becomes slightly tangled (e.g., the introduction of **t** and the reformulation from (4) to (6) could be streamlined). Figure 1's caption references "the number in the parenthesis" but the parentheses are not easily visible in the provided text. The parser artifacts in the later parts (e.g., garbled tables) are distracting but not the authors' fault.

**Limitations & Broader Impact**
The paper lacks a dedicated limitations section. Key limitations that should be explicitly acknowledged include: 1) The computational overhead of computing input gradients, 2) The potential conservativeness of the surrogate constraint, 3) The practical challenge of enforcing monotonicity over high-dimensional input spaces via uniform sampling, 4) The assumption of differentiability and handling of non-smooth activations. The broader impact statement is appropriate, noting the defensive/safety-aware nature of the work.

### Overall Assessment

This paper presents a novel formulation for monotonic neural networks using a chance constraint and a primal-dual training algorithm. The core idea is interesting and the approach is flexible, working with general architectures. However, the submission in its current form has significant methodological gaps that must be addressed for ICLR. The most critical issues are the lack of analysis on the tightness of the key surrogate constraint (Claim 1), insufficient discussion of the computational costs and differentiability assumptions, and a potentially biased experimental evaluation protocol. The experiments, while extensive, require more rigorous statistical analysis and controlled ablations to convincingly demonstrate the advantages of the proposed *training framework* over simpler regularization baselines. If these concerns can be thoroughly addressed, the contribution could be suitable for ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes a novel constrained learning framework for training monotonic neural networks. The method transforms the monotonicity requirement into a chance constraint, which is then relaxed into a continuously differentiable form, enabling the use of a stochastic primal-dual gradient (SPDG) algorithm. The framework is architecture-agnostic, requires no manual regularization tuning, and allows a flexible trade-off between monotonicity satisfaction and predictive performance via a parameter α. Experiments on classification, regression, and a control task demonstrate competitive performance against state-of-the-art monotonic network methods.

### Strengths
1.  **Novel Formulation:** The paper provides a fresh, optimization-centric perspective on enforcing monotonicity by framing it as a chance-constrained optimization problem and solving it with a primal-dual algorithm. The use of the auxiliary variable **t** and the relaxation via Claim 1 is a clever technical contribution that makes the constraint tractable for gradient-based learning.
2.  **Architectural Flexibility and Competitive Performance:** A core claimed strength is validated experimentally: the method imposes no architectural constraints, allowing the use of standard MLPs. Tables 1 and 2 show it achieves top or competitive accuracy/RMSE on several public datasets (COMPAS, Auto MPG, Heart Disease) while often using fewer parameters than specialized architectures like SMNN or LMN.
3.  **Adaptive Constraint Handling:** The dual variable **µ** automatically adjusts the penalty strength based on constraint violation during training (Eq. 9c), eliminating the need for the case-by-case regularization coefficient scheduling required by methods like Certified MNN. This is a practical advantage for ease of use.
4.  **Extension to Unsupervised/RL Task:** The successful application to the optimal frequency control problem (an RL-based control task) in Section 4.2 demonstrates the framework's generality beyond supervised learning. The results show improved performance over monotonic SNN and SMNN, highlighting the benefit of using general networks.

### Weaknesses
1.  **Lack of Theoretical Convergence Guarantees:** While the SPDG algorithm is presented, the paper provides no convergence analysis or guarantees for the non-convex setting of neural network training. For ICLR, a discussion on the convergence properties (even if under standard assumptions) or the quality of the obtained saddle point would strengthen the theoretical contribution.
2.  **Computational Overhead and Scalability Concerns:** The algorithm requires computing gradients with respect to inputs for points **z** sampled uniformly from the input domain **X** (Eq. 10). For high-dimensional input spaces (e.g., Blog Feedback with 276 features), uniformly sampling to adequately cover **X** could be expensive or inefficient, and the paper does not analyze this cost or scalability. The comparison of wall-clock training time against baselines is missing.
3.  **Incomplete Empirical Analysis:** (a) The choice of the key hyperparameter **α** and its impact on the monotonicity-proficiency trade-off is not studied. (b) The auxiliary variable **t** is fixed to a small constant in experiments, so the benefit of optimizing it (as in the full algorithm) is not demonstrated. (c) There is no reported quantitative measure of the *final* monotonicity satisfaction rate (e.g., % of test points where monotonicity holds) to verify the constraint was met for the chosen **α**.
4.  **Clarity and Presentation Issues:** The reformulation from Eq. (4) to Eq. (6) via Claim 1 is technically correct but could be explained with more intuition. The connection between the chance constraint and the practical enforcement over `Uni(X)` needs clearer justification. Several figures (e.g., Fig 1, parts of Tables 1,2) suffer from obvious parser artifacts (garbled text/numbers), which, while not the authors' fault, significantly hinder interpretation of the results.

### Novelty & Significance
**Novelty:** The approach is novel within the monotonic NN literature. It distinctively combines ideas from chance-constrained optimization and primal-dual learning, differing from both architectural redesigns (e.g., DLN, SMNN) and heuristic regularization or counterexample-guided methods (e.g., Certified MNN, COMET). The relaxation in Claim 1 and the resulting SPDG algorithm appear to be new for this problem.
**Significance:** The work is significant as it provides a flexible, general-purpose tool for incorporating monotonicity, a crucial inductive bias for safety, interpretability, and fairness. Its architecture-agnostic nature is a key practical advantage, allowing users to leverage modern networks. The results are strong, often outperforming recent SOTA, suggesting the method could become a standard baseline. It meets ICLR's emphasis on novel, impactful machine learning methods.

### Suggestions for Improvement
1.  **Strengthen Theoretical Foundation:** Provide a convergence analysis of the SPDG algorithm, even if under simplifying assumptions (e.g., convex loss, compact domain). Discuss the relationship between the relaxed constraint (6b) and the original chance constraint, perhaps bounding the approximation gap.
2.  **Enhance Empirical Evaluation:** (a) Include an ablation study on the effect of **α** and the choice to optimize vs. fix **t**. (b) Report a quantitative metric for monotonicity satisfaction on test/hold-out data. (c) Compare training time and memory footprint against key baselines to address scalability. (d) Visually demonstrate the effect of **α** on the learned function (e.g., on the 2D example).
3.  **Improve Clarity and Exposition:** Rewrite Section 3.1 to build more intuition. Clearly state why enforcing the constraint over `Uni(X)` is a sufficient (or necessary) condition for generalizability, and discuss the practical implications and potential limitations of this uniform sampling approach for high-dimensional spaces.
4.  **Address Computational Efficiency:** Propose and evaluate more efficient sampling strategies for high-dimensional **X** (e.g., focused sampling near decision boundaries) or discuss how many samples **N** are needed in practice. This is critical for convincing applications to large-scale problems.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation on the chance constraint parameter α and the auxiliary variable t.** The paper claims a key feature is trading off monotonicity satisfaction and prediction performance via α, and that t can be fixed to ease training. However, no experiment shows the effect of varying α on this trade-off or the sensitivity to the choice of t (fixed at 1e-4). Without this, the claim of flexibility is unsupported.
2. **Comparison against a strong, simple baseline: post-hoc monotonicity projection or regularization.** The paper dismisses regularization methods as requiring manual tuning, but does not compare against a well-tuned gradient penalty or a method that projects weights to ensure monotonicity. The claimed advantage of "no tuning" is hollow without showing their method outperforms a properly tuned baseline.
3. **Scalability to high-dimensional monotonic features.** Experiments are on datasets with few (≤8) monotonic features. The method samples uniformly from the input domain (Uni(X)); its efficiency and effectiveness with dozens or hundreds of monotonic features are unknown, undermining the claim of "advanced capability" for general architectures.
4. **Validation with non-ReLU activation functions.** The paper claims the framework works for "general architectures" and "commonly used activation functions," but all experiments use ReLU networks. Evidence is needed for sigmoid, tanh, etc., to substantiate this generality claim.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of the dual variable's adaptive behavior.** The paper claims the dual variable µ automatically adjusts penalty strength, eliminating manual tuning. However, there is no analysis showing how µ evolves during training compared to a fixed regularization weight, or how it correlates with constraint violation. This is crucial to trust the "adaptability" claim.
2. **Sample complexity analysis for the uniform sampling (Uni(X)).** Constraint satisfaction is estimated using N=128 uniform samples. The paper needs an analysis of how many samples are needed for reliable enforcement across the domain, especially for high-dimensional inputs, and how the choice of N impacts final monotonicity guarantees and performance.
3. **Theoretical justification for Claim 1.** The reformulation hinges on Claim 1, which is only informally reasoned. For ICLR, a formal proof (or citation to one) is required to establish the correctness of the inner approximation and its tightness, especially regarding the introduction of the auxiliary variable **t**.
4. **Quantification of monotonicity violation.** The paper reports accuracy/MSE but not the degree of monotonicity violation (e.g., percentage of violated pairs, maximum negative gradient) on test data or over the entire domain Uni(X). Without this, the method's success in enforcing the constraint is not measurable.

### Visualizations & Case Studies
1. **Visualization of constraint violation during training.** Plot the constraint violation (e.g., E[[...]+]) and the dual variable µ over training epochs for a representative task. This would visually demonstrate the adaptive enforcement mechanism and whether constraints are truly satisfied.
2. **Case studies highlighting failure modes of baselines and success of proposed method.** For a specific dataset, show input points where baseline models (e.g., SMNN, regularization) produce non-monotonic predictions while the proposed method does not, and analyze the impact on performance.
3. **Sensitivity visualization for the 2D example.** Extend Figure 1 to show how the learned function changes with different α values, illustrating the claimed trade-off between monotonicity and fit accuracy.

### Obvious Next Steps
1. **Formal proof of Claim 1** and discussion of the conservativeness of the approximation. This is a theoretical gap that should be addressed in the current paper.
2. **Experiments on the effect of the number of uniform samples N.** A simple sweep of N (e.g., 32, 128, 512, 1024) showing its impact on final model monotonicity and performance is necessary to justify the chosen setting and discuss computational trade-offs.
3. **Include an unconstrained NN baseline in the control experiment.** Figure 2 compares three monotonic methods, but it's unclear if monotonicity is even necessary for good performance here. An unconstrained NN baseline would contextualize the performance gains/losses due to the monotonicity constraint.
4. **Clarify experimental results in tables.** The garbled entries (e.g., "0. _±_ 0.501" in Table 1) need correction and explanation to ensure the reported superiority is trustworthy. The number of parameters for "Ours" in each dataset should be clearly stated and compared fairly.

# Final Consolidated Review
## Summary
This paper proposes a constrained learning framework to enforce monotonicity in neural networks by formulating it as a chance-constrained optimization problem, solved via a stochastic primal-dual algorithm. The method is architecture-agnostic, requires no manual regularization tuning, and allows a flexible trade-off between monotonicity satisfaction and prediction performance through a parameter α. Experiments on tabular datasets and a control task show competitive performance against state-of-the-art monotonic networks.

## Strengths
- **Novel optimization-centric formulation:** The paper introduces a fresh approach by framing monotonicity as a chance constraint, relaxed via Claim 1 into a differentiable surrogate, and solved with a primal-dual algorithm. This differs from prior architectural redesigns or heuristic regularization methods.
- **Architectural flexibility with strong empirical performance:** The method imposes no architectural constraints, enabling the use of standard MLPs. Experiments demonstrate top or competitive accuracy/RMSE on datasets like COMPAS and Auto MPG, often with fewer parameters than specialized monotonic architectures (e.g., SMNN, LMN).
- **Adaptive constraint handling:** The dual variable automatically adjusts penalty strength based on constraint violation during training, eliminating the case-by-case regularization tuning required by methods like Certified MNN.
- **Generality beyond supervised learning:** The framework is successfully applied to an unsupervised reinforcement learning task (frequency control), outperforming specialized monotonic networks and highlighting its versatility.

## Weaknesses
- **Lack of analysis on surrogate constraint tightness:** Claim 1, which underpins the differentiable relaxation, is only informally reasoned. Without a formal proof or discussion of how conservative the approximation is, it is unclear how well the enforced trade-off aligns with the intended chance constraint.
- **Computational overhead and scalability concerns:** The method requires computing input gradients for points uniformly sampled from the entire input domain, incurring additional per-iteration cost that is not quantified. Uniform sampling may be ineffective in high-dimensional spaces (e.g., Blog Feedback with 276 features), undermining claims of general applicability.
- **Incomplete empirical validation:** Key claims are not substantiated experimentally: the effect of the chance constraint parameter α on the monotonicity-performance trade-off is not studied, quantitative monotonicity satisfaction rates are not reported, and only ReLU networks are tested despite claiming compatibility with general activation functions.
- **Potential experimental bias:** Reporting the mean of the best five out of ten runs, while noted as aligned with some prior work, may inflate performance metrics and reduce reproducibility compared to standard full-run reporting.

## Nice-to-Haves
- Theoretical convergence guarantees for the primal-dual algorithm in the non-convex setting.
- Ablation study comparing against a well-tuned gradient penalty baseline to isolate the benefit of the constrained framework.
- Analysis of the dual variable's evolution during training to visually demonstrate adaptive enforcement.
- Investigation of more efficient sampling strategies (e.g., near decision boundaries) for high-dimensional input domains.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Include a formal proof or detailed analysis of Claim 1 in the appendix to clarify the approximation quality.
- Conduct experiments varying α and report monotonicity violation metrics (e.g., percentage of test points satisfying monotonicity) to validate the flexibility claim.
- Compare training time and memory usage against key baselines to quantify computational overhead.
- Test the method with non-ReLU activation functions (e.g., sigmoid, tanh) to substantiate the claim of architectural generality.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
