=== CALIBRATION EXAMPLE 21 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The abstract accurately characterizes the approach: a primal-dual, chance-constrained framework that imposes monotonicity without restricting architecture or requiring manual regularization tuning. Three key claims are made: (1) architecture-agnostic, (2) no manual regularization tuning, and (3) competitive empirical performance. All three are at least partially borne out in the body, though claims (1) and (2) require some nuance (see Method section below). The positioning as "novel" and "adaptive" is reasonable but slightly overstated given the closeness to existing primal-dual constrained learning work (e.g., Eisen et al., 2019, which the authors themselves cite).

---

### Introduction & Motivation (Sections 1, 1.1, 1.2)

The motivation is well-articulated. The two-category taxonomy (architecture-based vs. regularization-based) is a useful organizing frame. The three claimed contributions—flexibility, advanced capability, strong adaptability—are reasonable but largely qualitative. More concerning:

- **Missing connection to chance-constrained optimization literature.** The idea of chance constraints with a CVaR-type inner approximation (Claim 1 below) is well-established in stochastic optimization (Rockafellar & Uryasev, Nemirovski & Shapiro, etc.). The paper does not cite or contextualize this body of work, which makes the "novel" framing somewhat misleading.

- **Contribution (2) ("no constraints on architectures")** is somewhat weakened in practice: all experiments exclusively use simple ReLU MLPs, which is the very architecture for which prior certification methods like Liu et al. (2020) were designed. The claim that the framework enables, say, ResNets or Transformers to be monotone-trained is entirely untested.

---

### Method / Approach (Sections 2–3)

This is the paper's central technical contribution. Several important issues arise.

**Claim 1 is not novel.** The derivation in Claim 1 applies the inequality `1(g(x) ≥ 0) ≤ [1 + g(x)/t]₊` to convert a probability constraint into an expectation of a hinge loss. This is precisely the CVaR/hinge-loss relaxation of chance constraints (sometimes called the "Bernstein" or "sample average approximation" upper bound). It is a textbook technique in chance-constrained optimization and does not constitute a new theoretical contribution. The paper presents it as if it were derived from scratch without any citation to this prior art.

**Fixing t undermines a claimed advantage.** Equation (6b) introduces an auxiliary variable **t** that shapes the tightness of the inner approximation. The authors then state, "In practice, one may also consider to fix the auxiliary variable **t** at a small positive constant vector," and in every single experiment, **t** = 1×10⁻⁴ is indeed hardcoded. This raises two issues: (i) the tightness of the inner approximation as a function of t is never analyzed, and (ii) t itself is a hyperparameter that must be tuned, partially contradicting the claim that the framework "needs no pre-processing such as tuning of the regularization." The same applies to α (fixed at 0.1 universally) and γ_µ (set to 10).

**No convergence guarantees.** Algorithm 1 is a straightforward stochastic primal-dual gradient (SPDG) method, and the paper provides no convergence theorem—not even a brief sketch of conditions under which the algorithm converges to a KKT point or a constrained stationary point. The non-convexity of the neural network objective is non-trivial. For ICLR, even a citation of applicable convergence results from the constrained non-convex optimization literature would be expected.

**Monotonicity over the full input space X.** The authors switch from computing the expectation over the training dataset **D** to Uni(**X**) (Section 3.1). This is a meaningful design choice but raises important questions. For datasets like Blog Feedback with 276 features (most of which are non-monotonic), the input domain **X** = ×[lᵢ, uᵢ] is defined by per-feature bounds. Uniform sampling in 276 dimensions is extremely sparse relative to the manifold where data actually lives. Sampling N = 128 points from this 276-dimensional hypercube covers a vanishingly small fraction of the space. The paper does not discuss whether the computed gradient correction is meaningful under such conditions.

**The α=0 claim.** The paper repeatedly states "when α=0, problem (6) is exactly equivalent to the original problem (3)." This requires more care: the equivalence holds only if the hinge-loss upper bound is tight at α=0, which in general it is not (it is a sufficient condition, not a necessary one). Under the inner approximation of Claim 1, α=0 enforces E[(**t** − ∂f/∂xₘ)₊] ≤ 0 (after noting that -αt = 0), which does imply pointwise monotonicity at all z ∈ Uni(**X**) samples, but not strict equivalence to (3b).

---

### Experiments & Results (Section 4)

**Reporting methodology is non-standard and unfavorable to reproducibility.** Section 4.1 states: "We run the experiments ten times per dataset after finding the optimal hyperparameters and report the mean and standard deviation of the best five results." Selecting the best 5 out of 10 runs is cherry-picking and is not standard practice. Standard reporting would use the mean over all runs (or median). This is particularly concerning because the paper is comparing against prior work that did not use this same selection procedure—making comparisons potentially unfair.

**Monotonicity satisfaction is never measured.** This is the most significant experimental gap in the paper. The entire framework is designed to enforce monotonicity, yet not a single table or figure reports the fraction of monotonicity violations in the trained models. How monotonic are the resulting networks on held-out test points? Do all methods (including the unconstrained baseline) happen to be approximately monotone on these datasets? Without this measurement, it is impossible to assess whether the constrained learning framework is actually achieving its stated purpose better or worse than alternatives.

**Figure 2 shows only the best of 5 runs.** This is explicitly stated ("we conduct five independent runs on each model and plot the best results"). The frequency control comparison—showing 25.0% improvement over SMNN—is therefore based on a single cherry-picked run. Reporting mean ± std over all 5 runs is required for a fair comparison.

**Table 1 comparison fairness.** The Blog Feedback RMSE of our method (0.151) is compared against SMNN, but the SMNN row in Table 1 appears to have a garbled value. More importantly, the paper does not compare against regularization-based baselines (e.g., the negative gradient penalty from Gupta et al. 2019) in any dataset—one of the most direct baselines for the proposed approach.

**COMET (Sivaraman et al., 2020) is absent from Table 1.** The authors include COMET in Table 2 but not Table 1. The reason is not explained. COMET is one of the most relevant baselines and should appear consistently.

**No ablation studies.** The paper provides no ablation on: (i) the effect of α (other than the single fixed value of 0.1); (ii) fixed t versus jointly optimized t; (iii) the number of uniform samples N; or (iv) the sensitivity to γ_µ. These are all design choices that affect performance, and their impact is entirely unexplored.

**Generalizability of "architecture-agnostic" claim.** All five public-dataset experiments use simple ReLU MLPs (3–4 layers). No experiment demonstrates the framework working with a modern architecture (e.g., with skip connections, attention, or batch normalization). The primary architectural advantage claimed in the paper—that existing methods are incompatible with residual connections—is never demonstrated as a problem solved by this approach.

---

### Writing & Clarity (All Sections)

The paper is generally readable, but the presentation of the mathematical formulation is severely disrupted by the PDF parser (equations are fragmented across line numbers 230–400 in an almost unreadable way). Setting aside parser artifacts, the logical flow from problem (3) → (4) → (5) → (6) is clear. However, the algorithm box (Algorithm 1) references initialization `**t** = 0` while the text says `**t** = 1×10⁻⁴`—a minor but confusing inconsistency.

---

### Limitations & Broader Impact (Section 5)

The conclusions section is brief and does not acknowledge key limitations: (1) the lack of any convergence guarantee for the SPDG algorithm, (2) the potential failure of uniform sampling over high-dimensional input domains, (3) the inner approximation's conservatism, or (4) the fact that monotonicity satisfaction is never empirically verified. The broader impact statement is perfunctory ("our work does not discover any new threat"). The COMPAS dataset (used for criminal recidivism prediction) deserves more substantive ethical reflection given its well-documented fairness controversies.

---

### Overall Assessment

This paper proposes applying a chance-constrained primal-dual training paradigm to enforce monotonicity in neural networks. The framework is principled, architecture-agnostic, and reasonably well-motivated. However, the contribution is weaker than presented on several fronts. Theoretically, the core technical device (Claim 1) is a standard CVaR/hinge-loss relaxation of chance constraints that is not novel, and the paper offers no convergence analysis. Empirically, the most critical measurement—whether the trained networks are actually monotone—is entirely absent, the cherry-picking reporting methodology (best 5/10 runs) undermines the comparisons, and the "architecture-agnostic" advantage is claimed but never demonstrated on any non-MLP architecture. The auxiliary variable t and parameter α remain hyperparameters, partially contradicting the "no tuning required" narrative. In its current form, the paper is unlikely to clear the ICLR bar: the theoretical depth is insufficient for a venue that prizes formal analysis, and the experimental rigor falls short of what is needed to establish credibility for the empirical claims. Substantial revisions would be needed—specifically: (i) adding monotonicity violation metrics to all experiments, (ii) adopting standard reporting (mean over all runs), (iii) providing at least an informal convergence argument and citing the chance-constrained optimization literature, and (iv) testing on at least one non-MLP architecture to validate the core architectural claim.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes a novel learning framework to enforce monotonicity constraints on neural networks with general architectures using a chance-constrained optimization approach. The method utilizes a stochastic primal-dual algorithm to adaptively penalize monotonicity violations, allowing a trade-off between prediction performance and constraint satisfaction controlled by a parameter $\alpha$. Extensive experiments on public datasets and a safety-critical frequency control task demonstrate competitive performance compared to state-of-the-art methods, often achieving better results with fewer parameters.

### Strengths
1.  **General Architecture Compatibility:** Unlike many monotonic NN methods that require specific architectural modifications (e.g., monotone lattice layers, sign-constrained weights), this framework applies to standard MLPs without structural restrictions. This preserves the expressive power and ease of training associated with general deep learning architectures.
2.  **Adaptive Regularization:** The method eliminates the need for manual tuning of regularization strength, which is a common bottleneck in monotonicity-by-regularization approaches. The dual variable $\mu$ automatically adjusts the penalty based on constraint violations, as evidenced by the consistent convergence across different tasks without task-specific hyperparameter searches for the constraint penalty.
3.  **Comprehensive Evaluation:** The empirical validation is robust, covering both supervised tasks (classification/regression on 5 datasets) and reinforcement learning in a safety-critical physical system (power grid frequency control). The inclusion of recent SOTA baselines (e.g., SMNN, Certified MNN, LMN) and the demonstration of performance gains in the control domain add significant practical relevance, aligning well with ICLR's interest in robust and safe learning.

### Weaknesses
1.  **Theoretical Guarantees:** While the paper reformulates the problem as a chance-constrained optimization and applies primal-dual updates, there is no formal convergence analysis provided for the stochastic primal-dual algorithm in the context of non-convex neural network landscapes. Given ICLR's emphasis on theoretical rigor, the lack of a convergence rate proof or stability analysis for the non-convex max-min problem is a notable gap.
2.  **Safety Implications of Chance Constraints:** The method explicitly allows for constraint violations up to a probability $\alpha$ (set to 0.1 in experiments). In the context of the safety-critical power system experiment, stating that 10% of inputs may violate monotonicity raises safety concerns. The paper claims stability is maintained, but the mechanism ensuring safety despite probabilistic constraint violation is not rigorously explained or proven.
3.  **Computational Overhead:** The method requires sampling points from the entire input domain $\mathcal{X}$ (Uniform distribution) to evaluate the constraint gradient, rather than relying solely on the training dataset $\mathcal{D}$. While the paper claims "small extra computations," there is no quantitative comparison of training time or memory overhead against baseline methods, which rely on standard SGD or simpler regularization terms.

### Novelty & Significance
The novelty lies in the specific application of chance-constrained optimization with a primal-dual solver to the monotonicity problem, offering a "soft-strict" trade-off that bridges the gap between rigorous architectural constraints and loose regularization. This is significant for applications in fair ML and safety-critical systems where strict monotonicity is desired but might conflict with model accuracy. However, the foundational use of primal-dual methods for NN constraints is not entirely new, so the contribution is primarily in the specific formulation for monotonicity and its empirical efficacy. The significance is high for practitioners needing interpretability without sacrificing model capacity, though the theoretical underpinnings need strengthening for full ICLR acceptance.

### Suggestions for Improvement
1.  **Address Safety Guarantees:** For the power grid experiment, provide a clearer analysis or theoretical justification on how the system remains stable despite the 10% violation allowance ($\alpha=0.1$). If strict safety is required, a discussion on setting $\alpha \to 0$ or a verification step would be warranted.
2.  **Clarify Convergence:** Expand the theoretical section to discuss the convergence properties of the stochastic primal-dual updates. Acknowledging potential issues with non-convex saddle points and how the specific update rules mitigate local minima would strengthen the methodological contribution.
3.  **Report Computational Costs:** Include a wall-clock time benchmark or FLOPs comparison against the selected baselines (especially SMNN and LMN). Proving that the extra sampling over $\mathcal{X}$ and dual variable updates do not significantly hinder training efficiency is crucial for adoption claims.
4.  **Hyperparameter Sensitivity Analysis:** Provide a sensitivity analysis on the chance constraint parameter $\alpha$. How does performance vary if $\alpha$ changes from 0 to 0.5? This would demonstrate the "flexibility" claimed in the abstract and help users tune the trade-off effectively.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Quantitative Monotonicity Verification:** Report violation rates (e.g., percentage of test pairs violating monotonicity) for all datasets. Accuracy metrics alone do not verify the core claim that monotonicity is actually enforced.
2. **Computational Overhead:** Measure wall-clock training time and FLOPs compared to baselines. Computing input gradients for constraints adds significant cost not quantified in the "small extra computations" claim.
3. **Hyperparameter Sensitivity:** Sweep $\alpha$ and dual learning rate $\gamma_\mu$ to test the "no tuning" claim. Performance likely degrades if these fixed values are not tailored to specific datasets.
4. **High-Dimensional Constraint Satisfaction:** Evaluate violation rates specifically on the 276-dimensional Blog dataset. Uniform sampling becomes ineffective in high dimensions, potentially leaving constraints unenforced in sparse regions.

### Deeper Analysis Needed (top 3-5 only)
1. **Surrogate Constraint Tightness:** Analyze the gap between the smooth approximation (Claim 1) and the true chance constraint. A loose bound undermines the theoretical guarantee of monotonicity satisfaction.
2. **Performance-Monotonicity Trade-off:** Plot Pareto frontiers showing accuracy vs. violation rate. This validates the claimed flexibility of the chance constraint mechanism.
3. **Dual Variable Dynamics:** Analyze the convergence behavior of $\mu$ across epochs. Oscillating dual variables would indicate unstable constraint enforcement despite the "adaptive" claim.

### Visualizations & Case Studies
1. **Partial Dependence Plots:** Visualize output vs. monotonic features for tabular datasets. This exposes non-monotonic behavior that aggregate metrics hide.
2. **Violation Heatmaps:** For the 2D example, highlight regions where $\partial f/\partial x < 0$. This shows exactly where the method fails compared to architectural baselines.
3. **Constraint Loss Trajectory:** Plot the constraint violation term over training epochs. This reveals if monotonicity is achieved early or only at convergence.

### Obvious Next Steps
1. **Formal Verification:** Apply MILP or SMT verification to the final trained models. Sampling cannot guarantee global monotonicity, which is expected for safety-critical claims.
2. **Sampling Strategy Ablation:** Compare Uniform sampling vs. training data sampling for constraints. Uniform sampling in high dimensions is likely inefficient and should be benchmarked.
3. **Ablate Auxiliary Variable $t$:** Verify if learning $t$ provides benefit over fixing it, as the text suggests fixing it eases training. This validates the necessity of the proposed formulation.

# Final Consolidated Review
## Summary

This paper proposes a constrained learning framework for training monotonic neural networks using a chance-constrained optimization formulation solved via stochastic primal-dual gradient descent. The key idea is to reformulate hard monotonicity constraints as probabilistic constraints that can be traded off against prediction performance via a parameter α, and to enforce them adaptively through dual variable updates rather than manually tuned regularization. The authors evaluate the method on five public datasets and a power system frequency control task, comparing against recent monotonic neural network methods.

## Strengths

- **Architecture-agnostic formulation:** Unlike architectural approaches (Min-Max Net, Deep Lattice Networks, SMNN), this framework imposes monotonicity as a training constraint on standard neural networks without structural restrictions. This preserves the expressive capacity and ease-of-training of conventional architectures, which is a meaningful practical advantage.

- **Adaptive constraint enforcement:** The primal-dual formulation automatically adjusts penalty strength through dual variable μ updates (equations 9c), eliminating the need for manual regularization tuning that methods like Certified MNN require. The paper demonstrates this by using consistent hyperparameters (α=0.1, γ_μ=10) across all experiments.

- **Competitive empirical performance with fewer parameters:** Tables 1 and 2 show the proposed method achieves top or near-top accuracy on COMPAS, Auto MPG, and Heart Disease datasets while using fewer parameters than most baselines. The frequency control experiment (Figure 2) demonstrates meaningful improvement (25% over SMNN, 5.3% over monotonic SNN) on a real-world safety-critical application.

## Weaknesses

- **Critical gap: Monotonicity satisfaction is never verified.** The entire framework is designed to enforce monotonicity, yet no experiment reports the fraction of monotonicity violations in trained models. Without measuring whether the constrained networks are actually monotone—on test data or the full input domain—the core purpose of the method cannot be evaluated. This is especially concerning for the safety-critical control application.

- **Claim 1 is a known technique without attribution.** The derivation in Claim 1 applies the inequality 1(g(x) ≥ 0) ≤ [1 + g(x)/t]₊ to convert probability constraints to hinge-loss form. This is precisely the CVaR/conditional value-at-risk relaxation of chance constraints, a textbook technique in stochastic optimization (Rockafellar & Uryasev, Nemirovski & Shapiro). The paper presents this as novel without citing this literature.

- **No convergence analysis for non-convex settings.** Algorithm 1 is applied to neural networks with non-convex loss landscapes, yet the paper provides no convergence theorem, rate analysis, or even informal discussion of saddle-point conditions. For ICLR, some theoretical grounding for the stochastic primal-dual updates in non-convex settings is expected.

- **Non-standard reporting methodology.** Section 4.1 states: "We run the experiments ten times per dataset after finding the optimal hyperparameters and report the mean and standard deviation of the best five results." Selecting the best 5 of 10 runs is cherry-picking that inflates reported performance and undermines fair comparison with baselines that use standard reporting.

- **"Architecture-agnostic" claim remains untested on modern architectures.** All experiments use 3–4 layer ReLU MLPs. The claimed advantage over architectural methods—that they cannot use residual connections, attention, etc.—is never demonstrated with an actual experiment on non-MLP architectures.

- **Uniform sampling in high-dimensional input domains.** For Blog Feedback (276 features), the method samples N=128 points from a 276-dimensional hypercube to compute constraint gradients. The paper does not discuss whether such sparse sampling meaningfully enforces constraints across the input space.

## Nice-to-Haves

- Wall-clock training time or FLOPs comparison against baselines to substantiate the "small extra computations" claim.

- Sensitivity analysis for α beyond the single fixed value of 0.1—how does the accuracy-violation trade-off vary?

- Ablation comparing uniform sampling vs. training-data sampling for constraint evaluation.

- Verification of monotonicity on held-out test data using pair-wise violation checks or formal verification tools.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **PDF formatting artifacts criticism:** The critic noted fragmented equations, but this is clearly a parser issue and not a paper problem.

- **Claim that α and t hyperparameters contradict "no tuning":** While true that α and t are hyperparameters, the authors use identical values (α=0.1, t=10⁻⁴) across all experiments without dataset-specific tuning, which partially supports their "adaptive" claim. The criticism is valid but should be weakened—not about hyperparameters existing, but about their sensitivity being unexplored.

- **t=0 initialization inconsistency:** The algorithm box shows t=0 initialization while text mentions t=10⁻⁴—this is a minor notation inconsistency but not a substantive flaw.

## Novel Insights

The paper reveals an interesting tension in constrained deep learning: methods that guarantee monotonicity through architecture (SMNN, lattice networks) sacrifice expressive capacity, while regularization-based methods provide no guarantees. The chance-constrained formulation offers a middle ground—trading strict guarantees for retained expressivity—but this trade-off is precisely what remains unmeasured. The dual variable dynamics also present an understudied angle: μ automatically scales the penalty based on observed violations, functioning as a learned regularization strength. Analyzing μ's trajectory across training could reveal whether convergence is stable or oscillatory, which would illuminate the practical behavior of constrained learning algorithms.

## Suggestions

- Add a monotonicity verification metric to all experiments: report the percentage of randomly sampled input pairs (x₁ ≤ x₂) where f(x₁) ≤ f(x₂) fails to hold, or use formal verification (MILP/SMT) for ReLU networks.

- Report mean ± std over all 10 runs, not the best 5 of 10. This is the standard practice and enables fair comparison.

- Cite the CVaR and chance-constrained optimization literature for the hinge-loss relaxation technique, clarifying that Claim 1 applies established methods to this specific problem.

- Provide at least one experiment on a non-MLP architecture (e.g., ResNet with skip connections) to validate the core architecture-agnostic claim.

- Discuss the limitations of uniform sampling in high dimensions and propose alternatives (e.g., sampling near decision boundaries, using training data distribution).

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
