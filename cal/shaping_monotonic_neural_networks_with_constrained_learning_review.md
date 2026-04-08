=== CALIBRATION EXAMPLE 12 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title is clear and accurately reflects the paper's content. The abstract's claim that the framework "does not impose any constraints on the neural network architectures and needs no pre-processing such as tuning of the regularization" is partially misleading: the method introduces its own hyperparameters (α, γ_µ, **t**, N), which require selection. While α is framed as a design knob rather than a regularization weight, the claim of being "tuning-free" is not fully supported when the final algorithm always uses α = 0.1 across all experiments. The claim of "competitive performance" is modest and appropriate, though the experimental methodology used to generate those numbers is problematic (see Experiments).

---

### Introduction & Motivation (Sections 1–2)

The motivation is sound and the two-category taxonomy of existing approaches (architecture-based vs. regularization-based) is accurate and useful. The three advertised features (flexibility, capability, adaptability) are reasonable positioning statements.

However, the "Strong Adaptability" claim—that the method "does not require empirical tuning of the regularization terms"—is somewhat disingenuous. The primal-dual approach replaces manual regularization tuning with the choice of α, γ_µ, **t**, and N. These are not less demanding choices; they are simply different ones. The paper never provides guidance on how to set these in general, which undermines the claim.

---

### Method (Section 3) — **Most Critical Section**

**Claim 1 and the inner approximation.** Claim 1 provides a *sufficient condition* (an inner approximation), not an equivalence. The feasible set of (6) is strictly contained within that of (4). This conservatism is never quantified. How tight is the bound in practice? The choice of **t** critically controls the tightness: a very small **t** makes the surrogate constraint very tight and potentially infeasible, while a large **t** makes it very loose. Since **t** = 10⁻⁴ is fixed throughout all experiments, the authors have effectively made this a regularization-like design choice while claiming otherwise.

**The auxiliary variable t is always fixed, yet introduced as an optimization variable.** Algorithm 1 lists **t** as an optimization variable updated by (9b), but Section 4 (and Appendix A) state "we fix the auxiliary variable **t** = 1 × 10⁻⁴" for every single experiment. This contradiction is never reconciled. If **t** is always fixed, why include it as a free variable in the theoretical formulation? At minimum, an ablation over **t** is needed; at most, this suggests the theoretical treatment is overcomplicated for what is actually implemented.

**Computational cost of ∇_θ L.** The constraint (6b) involves ∂f_θ(**z**)/∂**z**_m (first-order partial derivatives of the network output w.r.t. inputs). The primal gradient ∇_θ L in (10) requires ∇_θ[∂f_θ/∂z_i], which is a second-order (Hessian-vector) computation involving mixed partials. The paper asserts "only small extra computations" but provides no timing comparisons, no FLOPs analysis, and no empirical runtime benchmarks. For deep networks, second-order computations are decidedly not free. This claim must be substantiated or retracted.

**Monotonicity over the full input space.** The paper argues (Section 3.1) that by sampling {**z**^n} from Uni(**X**), monotonicity is enforced over the full domain. However, N = 128 points are drawn per batch. For Blog Feedback with 276 input dimensions (even only 8 monotonic), the coverage of the relevant input hypercube is astronomically sparse. No discussion of how N should scale with dimension is provided, and no experiment examines the effect of N on either performance or monotonicity satisfaction.

**Absence of theoretical analysis.** The paper uses the language of constrained optimization and primal-dual methods, which naturally invites convergence analysis. No convergence result of any kind is provided — not even a convergence to a stationary point of the non-convex Lagrangian, which is typically achievable. For a methods paper at ICLR, this gap is significant, especially since convergence of primal-dual methods for non-convex neural networks is nontrivial and requires careful treatment.

**The α = 0 case.** The claim "when α = 0, problem (6) exactly returns to the original problem (3)" is only correct at the level of problem formulations. In the algorithmic implementation, α = 0 would mean no violations are allowed in the expectation over the N uniformly sampled points — which cannot guarantee strict monotonicity over the continuous domain **X** with finite-sample Monte Carlo estimation. This conflation between the continuous formulation and the finite-sample algorithm is never acknowledged.

---

### Experiments & Results (Section 4) — **Second Critical Section**

**Monotonicity satisfaction is never reported.** The central claim of the paper is that it enforces monotonicity. Tables 1 and 2 report only predictive accuracy and MSE. The fraction of the input domain satisfying the monotonicity constraint (e.g., the percentage of sampled test points with ∂f/∂x_i ≥ 0) is never measured or reported for any method. This is an extraordinary omission for a paper about enforcing monotonicity: the reader cannot evaluate whether the proposed method actually achieves its stated goal, or whether it merely happens to produce good predictions while possibly violating monotonicity frequently.

**Flawed evaluation methodology.** Section 4.1 states: "We run the experiments ten times per dataset after finding the optimal hyperparameters and report the mean and standard deviation of the **best five** results." Selecting the best five of ten runs is a form of oracle selection that systematically inflates reported performance. Standard practice at ICLR is to report statistics over all runs. If baselines are taken from published papers that did not use this selection protocol, the comparison is unfair.

**Single-run cherry-picking in the control experiment.** Section 4.2 states "We conduct five independent runs on each model and plot the **best results** in Figure 2." Reporting only the best single run with no standard deviation is not an acceptable scientific evaluation. The control experiment's central quantitative claim (25% improvement over SMNN, 5.3% over SNN) comes from this single best run and is not statistically validated.

**Chance constraint tolerance vs. strict monotonicity.** α = 0.1 means up to 10% of the input domain can violate monotonicity. Architecture-based methods (LMN, SMNN, Constrained MNN) guarantee strict monotonicity everywhere. Regularization-based methods (Certified MNN, COMET) use post-hoc certification to verify strict monotonicity. Comparing the proposed method's accuracy against these while allowing 10% violations is not a like-for-like comparison. The performance advantage may simply be the result of relaxing the constraint, not an algorithmic advance.

**Missing ablations.** Despite the paper's explicit claim that α provides "flexibility," α = 0.1 is used universally and no ablation over α is provided anywhere. Similarly, no ablation is provided for N (number of domain samples), for γ_µ, or for fixed vs. learned **t**. The three "key features" of the method are asserted but not experimentally demonstrated through controlled comparisons.

**Concerning variance on Heart Disease.** Table 2 reports 0.92 ± 0.14 for the proposed method on Heart Disease (61 test samples), versus 0.90 ± 0.02 for LMN and 0.89 ± 0.00 for Constrained MNN. The standard deviation of 0.14 is anomalously large and suggests high instability. A ±0.14 range over test accuracy on 61 samples implies results ranging from 0.78 to 1.0 across runs. This "best five of ten" selection strongly amplifies apparent performance on this highly variable dataset.

**No runtime comparison.** Given the central claim of "small extra computations," a table reporting training time per epoch for the proposed method vs. baselines is conspicuously absent.

---

### Writing & Clarity

The technical presentation suffers from structural issues that impede understanding. Equation (6) and the paragraph following it (page 4) appear *before* its derivation is complete — a block of text introducing the Lagrangian is followed mid-sentence by the constraint (4b) on the next rendered page, as if two sections were accidentally interleaved. This is likely a PDF extraction artifact, but it reflects a fragmented presentation. Algorithm 1 appears displaced from its surrounding derivation.

The three bullet-pointed features of Section 1.1 are never revisited in the experiments. Flexibility (α) is never ablated. Advanced Capability (arbitrary architectures) is only demonstrated on MLPs — no experiment uses a ResNet, Transformer, or other architecture that would be genuinely incompatible with existing methods. Adaptability (no manual tuning) is asserted but α, **t**, γ_µ, and N all require manual selection.

---

### Limitations & Broader Impact

The paper has no explicit limitations section. Key limitations unacknowledged by the authors include:

1. **No post-hoc monotonicity guarantee**: Unlike Certified MNN or COMET, there is no mechanism to verify that the trained network is actually monotonic. For safety-critical applications (the primary motivation), this is a fundamental gap.
2. **Input domain requirement**: The method requires a known bounded input domain **X** from which to sample uniformly. For distributions with unknown support or highly non-uniform density, this is non-trivial.
3. **Scalability of second-order gradients**: As noted, computing ∇_θ[∂f/∂x_i] scales poorly for large networks. The Blog Feedback network, while small here, suggests practical limitations.
4. **Noise in the data**: If the ground-truth function is not actually monotonic (i.e., the monotonicity requirement is imposed as a prior rather than a fact), the method's behavior as α → 0 is unclear.

---

### Overall Assessment

The paper addresses a well-motivated and practically relevant problem, and the core idea — recasting monotonicity as a chance constraint and applying primal-dual optimization — is principled and applicable to arbitrary network architectures. However, several serious weaknesses prevent acceptance at the ICLR bar. Most critically: *monotonicity satisfaction is never measured*, which is fatal for a paper about enforcing a constraint; the experimental evaluation methodology (best-5-of-10 selection, single-best-run for the control experiment) is non-standard and inflates performance claims; and the comparison against strictly monotonic baselines while permitting 10% violations is unfair. On the theory side, the absence of any convergence analysis is a notable gap, and the contradiction between the theoretical treatment of **t** as an optimization variable and its invariant fixation at 10⁻⁴ in all experiments weakens the paper's claimed framework. Addressing these issues — particularly adding monotonicity satisfaction rates, using standard evaluation protocols, ablating α and N, and providing at least a basic theoretical convergence discussion — would substantially strengthen the submission.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes a primal-dual constrained learning framework to enforce input-output monotonicity in neural networks with arbitrary architectures. By reformulating the monotonicity requirement as a differentiable chance constraint and introducing adaptive dual variable updates, the method avoids restrictive architectural modifications and eliminates manual regularization tuning. Experiments across classification, regression, and a reinforcement learning-based frequency control task demonstrate competitive predictive performance with fewer parameters compared to recent state-of-the-art monotonic network baselines.

### Strengths
1. **Architecture-Agnostic Formulation:** The method preserves standard deep learning backpropagation without constraining weights or enforcing specialized layers. This allows immediate integration with modern architectures (e.g., ResNets, dense MLPs), as evidenced by the direct use of gradient descent updates in Eq. (10) and standard MLP structures in Appendix A.2.
2. **Adaptive Penalty via Primal-Dual Updates:** Unlike prior regularization-based methods that require tedious manual scheduling of penalty coefficients, the framework leverages dual ascent (Eq. 9c) to dynamically increase the penalty only when constraints are violated. Section 3.2 correctly identifies this as an "adaptive regularization" mechanism, reducing heuristic tuning.
3. **Broad Empirical Validation:** The paper evaluates the method on five diverse supervised benchmarks (Table 1 & 2) and extends to a safety-critical, RL-based power system control task (Section 4.2). Results show the method achieves superior or comparable accuracy/RMSE with significantly fewer parameters (e.g., 847 params vs. 1421+ for SMNN on Blog Feedback), indicating strong parameter efficiency.
4. **Tractable Surrogate for Hard Constraints:** The use of Claim 1 to transform a non-differentiable indicator in the chance constraint into a smooth expectation over $(t - \partial f/\partial x_m)_+$ is mathematically elegant. It successfully bridges rigorous constraint satisfaction with practical SGD training.

### Weaknesses
1. **Missing Monotonicity Violation Metrics:** A critical evaluation gap is the absence of quantitative monotonicity violation rates or post-training certification results in Tables 1 and 2. While prediction accuracy is reported, readers cannot verify if the proposed method actually achieves the claimed constraint satisfaction or how it compares to methods with formal guarantees (e.g., Certified MNN).
2. **Biased Evaluation Protocol:** Section 4.1 states the authors "report the mean and standard deviation of the best five results" out of ten runs. This selection bias artificially inflates performance and increases variance in the reported mean. ICLR standards typically require reporting statistics over all independent seeds or employing formal statistical significance testing.
3. **Under-Explored Hyperparameter Trade-offs:** The paper claims to avoid "pre-processing such as tuning of regularization," yet introduces new hyperparameters: the chance-level $\alpha$, dual learning rate $\gamma_\mu$, and auxiliary variable $t$. All are fixed to single values (e.g., $\alpha=0.1$) across datasets without sensitivity analysis or ablation. There is no empirical demonstration of the claimed accuracy-monotonicity trade-off curve.
4. **Scalability of Uniform Domain Sampling:** The framework enforces monotonicity by sampling uniformly from $\text{Uni}(\mathbf{X})$. This approach suffers severely from the curse of dimensionality as the number of monotonic features $|m|$ grows. The paper does not discuss sampling strategies for higher-dimensional monotonic subspaces or the resulting sample complexity.

### Novelty & Significance
**Novelty:** Moderate. The core ideas—primal-dual constrained optimization, chance constraints, and smooth surrogate relaxations—are well-established in constrained machine learning literature (e.g., Eisen et al. 2019, Cotter et al. 2019). The novelty lies in carefully adapting these tools specifically to neural monotonicity, designing the differentiable surrogate in Eq. (6), and validating it across both supervised and RL control settings. It represents a solid incremental contribution rather than a fundamental theoretical breakthrough.
**Significance:** High. Enforcing monotonicity efficiently without architectural surgery or expensive MILP/SMT verification is highly valuable for trustworthy AI, particularly in safety-critical domains. The architecture-agnostic nature and low parameter overhead make it practically appealing to the ICLR community.
**Clarity:** High. The mathematical derivation flows logically from problem formulation to algorithm design. The distinction between training data distribution and uniform input domain sampling is clearly motivated.
**Reproducibility:** High. The paper provides explicit network architectures, learning rates, batch sizes, $\alpha$, $\gamma_\mu$, and sampling budgets $N$ in the appendices. Algorithm 1 is fully specified, and open-source code is promised, meeting ICLR's reproducibility bar.

### Suggestions for Improvement
1. **Add Monotonicity Certification/Violation Metrics:** Augment Tables 1 and 2 with a "Violation Rate %" or post-hoc certification results (e.g., using interval bound propagation or a subset of MILP checks) to empirically prove that the learned models satisfy monotonicity constraints at the specified $\alpha$ level.
2. **Adopt Standard Evaluation Protocols:** Report mean $\pm$ standard deviation over *all* random seeds (or a clearly justified subset) rather than the best five. Include statistical comparison tests (e.g., paired t-test or Wilcoxon) against the strongest baselines (LMN, SMNN) to confirm that performance gains are statistically significant.
3. **Provide Hyperparameter Ablation & Trade-off Curves:** Include an ablation study varying $\alpha$ (e.g., 0.0, 0.05, 0.1, 0.2) and plot the resulting test accuracy vs. monotonicity violation rate. This would quantitatively validate the "flexibility" claim and guide practitioners on parameter selection.
4. **Address High-Dimensional Sampling Challenges:** Discuss how the method behaves when the number of monotonic features is large. If $\text{Uni}(\mathbf{X})$ suffers from sparsity, consider proposing or citing adaptive sampling heuristics (e.g., focusing on boundaries or regions of high gradient uncertainty) to maintain constraint enforcement efficiency.
5. **Theoretical Grounding:** Briefly discuss the tightness/bias of the relaxation in Claim 1 (it effectively acts as a Conditional Value-at-Risk or hinge relaxation). Additionally, cite recent work on convergence of stochastic primal-dual algorithms for non-convex deep learning, or acknowledge that convergence guarantees are heuristic in this setting, as required by ICLR's standards for rigorous theoretical framing.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Monotonicity Violation Metrics:** Report the percentage of test samples and uniform samples that violate monotonicity constraints for every dataset. Without explicit violation rates, the core claim that the method "enforces monotonicity" is unverifiable despite accuracy tables.
2. **Hyperparameter Sensitivity ($\alpha$):** Provide a sensitivity analysis varying the chance constraint coefficient $\alpha$ (e.g., 0.01 to 0.5). A single fixed value ($\alpha=0.1$) fails to demonstrate the claimed flexibility to "trade off between probability of satisfaction and performance."
3. **Computational Overhead:** Report training time per epoch and wall-clock time compared to standard unconstrained training and regularization baselines. The claim of "only small extra computations" is dubious given the need for input-gradient computations ($\partial f / \partial x$) and uniform sampling per batch.

### Deeper Analysis Needed (top 3-5 only)
1. **High-Dimensional Sampling Validity:** Analyze how uniform sampling from $\text{Uni}(\mathbf{X})$ ensures monotonicity in the 276-dimensional Blog Feedback dataset. Sparse sampling in high dimensions mathematically undermines the guarantee of enforcing constraints "across the entire input domain."
2. **Dual Variable Convergence:** Include traces of the dual variable $\mu$ over training epochs to verify adaptive behavior. This is necessary to substantiate the claim that the method "continuously and adaptively enforce[s] the monotonicity" without manual tuning.
3. **Surrogate Constraint Tightness:** Quantify the gap between the differentiable surrogate (Claim 1) and the actual chance constraint. If the surrogate is too loose, the theoretical guarantee linking the optimization objective to actual monotonicity satisfaction is invalid.

### Visualizations & Case Studies
1. **Pareto Frontiers:** Plot Performance (Accuracy/MSE) vs. Monotonicity Violation Rate for your method versus tuned regularization baselines. This visualization is required to prove the "adaptive" advantage over standard penalty methods rather than just showing single-point comparisons.
2. **Partial Dependence Plots:** Show partial dependence plots for monotonic features on high-dimensional datasets (e.g., Blog Feedback). This would visually expose whether the method fails to enforce monotonicity in regions where uniform sampling was insufficient.

### Obvious Next Steps
1. **Formal Certification:** Apply a formal monotonicity verifier (e.g., MILP or SMT used in baselines like Liu et al., 2020) to the trained models. Relying on sampled estimates is insufficient for a paper claiming robust constraint satisfaction for safety-critical systems.
2. **Ablation on Dual Update:** Compare the primal-dual update against a fixed Lagrange multiplier to isolate the benefit of the adaptive dual learning mechanism. This determines if the performance gain comes from the constraint formulation or the adaptive optimization strategy.

# Final Consolidated Review
## Summary

This paper proposes a primal-dual constrained learning framework for enforcing partial monotonicity constraints in neural networks. By reformulating monotonicity as a chance constraint and using a differentiable surrogate, the method can be applied to arbitrary neural network architectures without weight constraints or specialized layers. A stochastic primal-dual gradient algorithm adaptively adjusts the penalty strength during training, eliminating manual regularization tuning.

## Strengths

- **Architecture-agnostic formulation:** Unlike lattice-based methods (DLN, Min-Max Net) or constrained-architecture approaches (LMN, SMNN), this method applies to any differentiable network architecture through standard backpropagation with gradient modification (Eq. 10), preserving modern deep learning capabilities such as residual connections and diverse activations.
- **Adaptive dual mechanism:** The primal-dual formulation (Eq. 8-9) replaces manual regularization tuning with an automated Lagrange multiplier update, where dual variables increase only when constraints are violated. This addresses a practical pain point in regularization-based approaches that require iterative penalty adjustment.
- **Strong parameter efficiency:** Tables 1 and 2 show competitive or superior accuracy with significantly fewer parameters than most baselines (e.g., 847 vs. 1421 for SMNN on Blog Feedback; 65.4% accuracy with 1,353 parameters vs. 65.4% with 2,225 for LMN on Heart Disease).
- **Diverse empirical validation:** The method is evaluated on five supervised benchmarks (classification and regression) plus an unsupervised RL-based frequency control task, demonstrating broader applicability than most prior monotonicity work focused solely on supervised learning.

## Weaknesses

- **Monotonicity satisfaction is never measured or reported.** This is the most critical weakness. Tables 1 and 2 report only predictive accuracy/MSE, with zero quantification of actual monotonicity violation rates. The central claim is that the method "enforces monotonicity," yet readers cannot verify whether trained models satisfy constraints or compare fairly to baselines with formal guarantees (Certified MNN, COMET). At minimum, the paper should report the percentage of uniformly sampled test points where ∂f/∂x_i ≥ 0 fails to hold.

- **Non-standard evaluation protocol inflates performance.** Section 4.1 states: "We run the experiments ten times per dataset after finding the optimal hyperparameters and report the mean and standard deviation of the best five results." Selecting the best 5 of 10 runs is a form of oracle selection that systematically biases upward reported means and distorts variance estimates. Standard practice is to report statistics over all runs.

- **Contradiction between theoretical formulation and implementation.** Algorithm 1 presents the auxiliary variable **t** as an optimization variable updated via (9b), yet Appendix A.2 states "we fix the auxiliary variable **t** = 1 × 10⁻⁴" for all experiments. If **t** is always fixed, its treatment as a learned variable in the theoretical development is unnecessary. The paper should either implement the algorithm as specified or revise the theoretical presentation.

- **No ablation over α despite claiming "flexibility" as a key feature.** The α parameter purportedly enables trading monotonicity satisfaction for predictive performance, but α = 0.1 is used universally across all experiments. Without demonstrating how varying α affects the monotonicity-performance tradeoff, this claimed advantage remains unsubstantiated.

- **Computational overhead of second-order gradients is not quantified.** The gradient computation ∇_θ[∂f_θ/∂z_i] in Eq. (10) requires mixed partial derivatives (Hessian-vector products). The claim of "only small extra computations" lacks empirical support—no timing comparisons, FLOPs analysis, or training wall-clock times are provided.

- **Comparison asymmetry with strict monotonicity methods.** Architecture-based methods (LMN, SMNN) and certification-based methods (Certified MNN, COMET) guarantee strict monotonicity everywhere, while this method allows up to 10% violations (α = 0.1). Any performance gains may derive from this relaxed constraint rather than algorithmic superiority. The paper should acknowledge this and ideally compare at equivalent violation levels.

## Nice-to-Haves

- **Convergence analysis:** As a methods paper employing primal-dual optimization for non-convex neural networks, even a basic discussion of convergence to stationary points would strengthen the theoretical foundation.
- **Sensitivity analysis for uniform sampling:** The method samples N = 128 points from Uni(X) to estimate constraint violations. For high-dimensional inputs (Blog Feedback has 276 features), this coverage is sparse. Ablation over N or discussion of sample complexity would clarify scalability.

## Removed Points

These points are flagged to be removed, treat them with caution:
- "Presentation suffers from structural issues where equations appear before derivation is complete" — This appears to be a PDF extraction artifact in the review copy, not an actual paper flaw.
- "No explicit limitations section" — Not a standard requirement; most papers do not include dedicated limitations sections.
- "The title is misleading" — The title accurately reflects the paper's constrained learning approach to monotonicity.
- "Absence of comparison to neural network fairness methods" — Scope creep; the paper addresses monotonicity, not general fairness.

## Novel Insights

The primal-dual formulation elegantly transforms monotonicity enforcement from a discrete search over architectures or an iterative regularization-tuning procedure into a single optimization loop with adaptive constraint handling. The insight that dual variables naturally encode "how much to penalize violations" replaces the opaque hyperparameter scheduling in prior work with a principled mechanism derived from constrained optimization theory. However, this comes at a cost: the uniform sampling strategy (sampling from Uni(X) rather than the data distribution) means the method enforces monotonicity over regions that may never appear in practice, potentially wasting capacity on irrelevant input regions. The chance constraint formulation (α < 1) implicitly acknowledges this tradeoff but the paper does not exploit or analyze it.

## Suggestions

- Add a column to Tables 1 and 2 reporting monotonicity violation rate (percentage of sampled test points where ∂f/∂x_i < 0 for any monotonic feature i). This enables fair comparison to baselines with guarantees.
- Rerun experiments using standard evaluation protocol: report mean ± std over all 10 seeds, not best 5 of 10.
- Provide an ablation study varying α ∈ {0.0, 0.05, 0.1, 0.2, 0.5} to demonstrate the claimed flexibility-performance tradeoff.
- Clarify the role of **t**: either justify fixing it at 10⁻⁴ with theoretical analysis of why this is a reasonable default, or implement the full algorithm with learned **t**.
- Report training time per epoch and total training time relative to unconstrained baselines to substantiate the "small extra computations" claim.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
