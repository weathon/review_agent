# Joint Distribution–Informed Shapley Values for Sparse Counterfactual Explanations

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 8

## Abstract
Counterfactual explanations (CE) aim to reveal how small input changes flip a model’s prediction, yet many methods modify more features than necessary, reducing clarity and actionability. We introduce COLA, a model- and generator-agnostic post-hoc framework that refines any given CE by computing a coupling via optimal transport (OT) between factual and counterfactual sets and using it to drive a Shapley-based attribution p-SHAP that selects a minimal set of edits while preserving the target effect. Theoretically, OT minimizes an upper bound on the $W_1$ divergence between factual and counterfactual outcomes and that, under mild conditions, refined counterfactuals are guaranteed not to move farther from the factuals than the originals. Empirically, across four datasets, twelve models, and five CE generators, COLA achieves the same target effects with only 26–45% of the original feature edits. On a small-scale benchmark, COLA shows near-optimality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces COLA (COunterfactuals with Limited Actions), a post-hoc framework that refines existing counterfactual explanations to use fewer feature modifications while maintaining the desired outcome. The key innovation is p-SHAP, a generalized Shapley value method that uses optimal transport (OT) to compute a coupling between factual and counterfactual instances, then leverages this coupling to identify which features are most critical to modify. Existing counterfactual explanation methods often modify more features than necessary to flip a model's prediction.

### Strengths
The paper provides a rigorous theoretical foundation for its approach with three well-formulated theorems. Theorem 4.1 establishes an important connection between optimal transport and counterfactual effect, proving that the OT-derived coupling minimizes an upper bound on the W₁ divergence between model outputs. This provides principled justification for why OT is the right tool for aligning factual and counterfactual instances. Theorem 4.2 demonstrates that p-SHAP captures genuine interventional effects, showing that the value function represents the causal impact of intervening on feature subsets. Theorem 5.1 offers proximity guarantees, proving that under certain conditions, the refined counterfactuals stay at least as close to the factuals as the original counterfactuals. These theoretical contributions are properly formalized with complete proofs in the appendices, demonstrating mathematical rigor and providing confidence in the method's foundations beyond purely empirical validation.

### Weaknesses
Weaknesses:

Theory-practice gap: The main theoretical results assume Lipschitz continuity (Theorem 4.1) and deterministic OT matching with n=m (Theorem 5.1). These conditions are restrictive and don't hold generally. The paper doesn't adequately discuss when these assumptions are violated in practice.
NP-hardness result (Theorem B.1) undermines the approach: If the problem is NP-hard even for d=1 linear models, why should we expect good approximations from the proposed heuristic? No approximation guarantees are provided.
Limited theoretical justification for OT choice: While Theorem 4.1 shows OT minimizes an upper bound, it doesn't prove this leads to optimal feature selection for the discrete action problem in Eq. 1.
Experimental methodology concerns:
- Standard deviations in Table 3 are very small (±0.02-0.09), suggesting possibly limited diversity in runs
- The "near-optimal" claim in Result III is based only on German Credit dataset - too limited to generalize
- Figure 4 shows significant gaps between CF-pOT/CF-pEct and optimal, contradicting "near-optimal" claims

### Questions
The paper's theoretical contributions rest on restrictive assumptions that may not hold in practice. Theorem 4.1 requires Lipschitz continuity of the model f, yet many of the 12 tested models (particularly DNNs with ReLU activations) are not globally Lipschitz continuous. The authors do not verify which models satisfy this assumption or report how tight the theoretical bound is empirically. Theorem 5.1 only applies when n=m and ε=0 (deterministic matching), yet most experimental scenarios violate these conditions. Furthermore, while Theorem B.1 proves the problem is NP-hard even for simple cases, the paper provides no approximation guarantees or worst-case bounds for the proposed heuristic solution, leaving a significant gap between the hardness result and the practical algorithm's performance guarantees.

Several core design choices in Algorithm 1 lack adequate justification. The probabilistic sampling mechanism in Line 7 introduces unnecessary randomness when deterministic top-C selection would be simpler and potentially more stable—no ablation study compares these approaches. The specific formulation of p-SHAP in Equation 7 is presented without explaining why this particular way of incorporating the coupling p into Shapley values is optimal compared to alternatives like weighted Shapley methods. Most puzzling is the observation in Figure 4 where CF-pOT sometimes outperforms CF-pEct despite the latter using the "true" alignment from the CE generator—this counterintuitive result demands deeper investigation as it questions fundamental assumptions about what constitutes the "correct" factual-counterfactual pairing. Additionally, the choice between A^max_Value and A^avg_Value appears arbitrary across datasets without clear selection criteria.

The experimental evaluation, while broad in scope (4 datasets, 12 models, 5 CE methods), lacks critical depth in several dimensions. Most notably, there is no computational cost analysis despite complexity bounds being provided in Appendix F—readers cannot assess whether COLA's sparsity improvements justify its computational overhead. The "near-optimal" claim relies solely on the German Credit dataset, where Figure 4 actually shows 20-40% gaps to optimality undermining this assertion. No statistical significance tests are provided, making it unclear whether observed improvements are meaningful or due to random variation. The extremely small standard deviations in Table 3 (±0.02-0.09) are suspicious given the sampling process and warrant verification. Crucially, the paper lacks comparisons to methods explicitly designed for sparse counterfactual generation, such as L1-regularized CE methods, sparse regression approaches, or greedy feature selection baselines—the current baselines only compare different Shapley variants within COLA.

### Soundness
2

### Presentation
2

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
This paper addresses the problem of sparse counterfactual explanations: how to achieve a desired model prediction with minimal feature modifications. The authors propose COLA, a general post-hoc framework applicable to any model and counterfactual generator. The method introduces a novel joint distribution–informed Shapley attribution (p-SHAP), which uses an optimal transport coupling between factual and counterfactual samples to guide feature selection. Theoretical results show that OT minimizes an upper bound on the Wasserstein distance between model outputs and targets.

### Strengths
1. The paper tackles a meaningful and practical issue in explainable AI: generating concise and actionable counterfactual explanations.
2. The problem and algorithmic design are mathematically well-structured.
3. Experiments across several datasets and models show consistent improvements in sparsity and performance.
4. The writing is generally clear and organized; the figures and tables support understanding of the method.

### Weaknesses
1. The proposed p-SHAP is largely a recombination of existing ideas (Optimal Transport + Shapley values). The conceptual contribution beyond combining these two paradigms remains limited. It is unclear what fundamentally new insight p-SHAP provides.
2. The theorems mainly show that OT minimizes a Wasserstein upper bound, but this does not directly imply minimal feature edits in the discrete action space. The theoretical link between Theorem 4.1/5.1 and the claimed “minimal action” property is not convincingly established.
3. All experiments are on small-scale tabular datasets. There is no evidence that the approach scales to more complex or high-dimensional data, nor comparison to recent counterfactual sparsity benchmarks.
4. Since COLA may select arbitrary feature edits, how does it ensure the modifications are feasible or ethically valid (e.g., immutable features like age or gender)? The authors briefly mention this in the ethics statement but do not model it in the algorithm.

### Questions
1. Can the authors clearly articulate the new theoretical or methodological insight that distinguishes p-SHAP from prior Shapley-based counterfactual attribution methods?
2. How does Theorem 4.1 translate into sparse modification guarantees in practice? Please provide an intuitive or empirical justification of this link.
3. What is the computational complexity and runtime overhead of COLA compared to direct CE methods？
4. Can COLA handle mixed data types (categorical + continuous) and feasibility constraints in real applications？

### Soundness
2

### Presentation
3

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
This paper introduces COLA, a post-hoc framework for refining counterfactual explanations by leveraging optimal transport (OT)-induced joint distributions to guide a novel p-SHAP attribution method. The key innovation lies in using OT to model the alignment between factual and counterfactual instances, which then informs Shapley-based feature attribution to select the minimal set of actionable edits that preserve the desired counterfactual outcome. Theoretically, the authors prove that OT minimizes an upper bound on the W₁ divergence between factual and counterfactual predictions under Lipschitz continuity, and that refined counterfactuals remain no farther from the original facts than the original CEs. Empirically, COLA achieves 26–45% fewer feature edits than baseline CEs across 4 datasets, 12 models, and 5 CE generators, with p-SHAP consistently outperforming other Shapley variants.

### Strengths
## Novelty and Theoretical Depth:
The integration of OT as a coupling mechanism to inform Shapley values is a fresh and well-justified approach. It moves beyond ad-hoc or model-specific alignments and provides a principled way to define meaningful contrastive references.
The p-SHAP framework generalizes B-SHAP, RB-SHAP, and CF-SHAP under a unified probabilistic coupling, offering a modular and flexible interface for attribution.

## Strong Theoretical Guarantees:
Theorem 4.1 is compelling: it establishes that OT minimizes an upper bound on the $W_1$ divergence between $f(x)$ and $y*$, linking the cost of feature modification directly to prediction fidelity. This is a non-trivial theoretical contribution.
Theorem 5.1 provides a provable guarantee that refined CEs do not move farther from the factuals than the originals (in Frobenius norm).

## Comprehensive Empirical Evaluation:
The experiments are rigorous and well-designed: 4 datasets, 12 models, 5 CE generators, multiple divergence metrics (OT, MMD, MeanD, MedianD), and ablations across different Shapley methods. The use of CF-$p_{OT}$ as the proposed method is clearly contrasted with baselines, demonstrating that alignment matters more than mere use of counterfactuals. The MILP-based optimality benchmark adds significant credibility, showing COLA’s near-optimality in a controlled setting.

## Clear Motivation and Problem Framing:
The paper correctly identifies the de-coupling problem: standard FA methods (like Shapley) ignore the specific path to counterfactual outcomes, leading to suboptimal or even counterproductive edits. COLA addresses this by grounding attribution in the factual-counterfactual alignment.

### Weaknesses
## Ambiguity in the Role of OT in p-SHAP:
The paper frames OT as a means to define a joint distribution p, which is then used in p-SHAP. However, OT is not directly used in the Shapley computation.

## Assumption of Known Counterfactuals:
The framework assumes that a counterfactual set r is already available via some CE method. While this is standard in post-hoc refinement, the paper does not discuss how errors in the initial CE propagate into the final refinement. A sensitivity analysis to noisy or suboptimal CEs would strengthen the claim.

## Limited Discussion on Scalability:
While computational complexity is analyzed, authors do not discuss practical bottlenecks: OT with entropic regularization is expensive for high dimensionality. Solving OT on 1,000 × 1,000 matrices might be feasible but slow. Authors assume OT is "cheap" via Sinkhorn, but does not report scalability limits.

## Averaging Across Datasets/Models May Mask Heterogeneity:
The results are averaged across scenarios. While this is standard, it could obscure cases where COLA fails (like with highly non-linear models or sparse features). A per-scenario analysis (e.g., worst-case performance) would help assess robustness.

### Questions
Is the OT coupling used only to define the joint distribution p for p-SHAP, or does it also influence the actual value of the Shapley contributions beyond the reference distribution? Could you provide a small example where $p_{OT}$ leads to different attribution than $p_{Uni}$ or $p_{Rnd}$?

How does COLA perform when the initial CE is poor (high divergence from factuals, or incorrect predictions)? Could the refinement process amplify errors?

What are the practical limits of the OT step in terms of n (number of instances) and d (features)? Have you tested COLA on larger datasets (>10k instances)?

Why not use the OT plan directly for edit selection? Instead of using $p_{OT}$ to compute p-SHAP, could one directly use the OT plan to select which features to modify by summing $p_{ij}$ over $j$)? How does this compare to p-SHAP?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a method for obtaining sparse counterfactual explanations titled COunterfactuals with Limited Actions (COLA). The main novelty resides on the one side of it being a post-hoc method that refines existing counterfactuals to reduce their cardinality while minimizing the impact on the counterfactual's performance, and secondly to apply an Optimal Transport setting for guiding the refinement of the counterfactuals using Shapley attributions.

### Strengths
The paper's main strength is its combinaton of theory-guided framework with its extensive experimental setup and computational experiments. Another major strength of the work is the source code provided by the authors, which allows for reproducibility of the results end-to-end. The appendices are not used merely as extension of the paper but provide extensive theoretical and practical background about the study.

### Weaknesses
At times, the theoretical exposition remains a bit obscure, mainly due to the fact that the authors frame the problem around the group setting (generating a group of counterfactuals for a given group of factual instances) instead of in my opinion much common framing of the problem where a set of counterfactuals is generated for a specific factual instance. This makes the exposition harder for non-specialists. It is worth asking if the benefit (more generalizability) overweights the cost (clarity of exposition).

### Questions
- In the problem formulation (1a)-(1e), the authors state their model without but decide not to include the flipping of the target as an additional constraint. In fact, model (1a)-(1e) would by itself not produce any counterfactuals but merely minimum-distance synthetic samples. I think that should be included in the model.
- In Section 4, it remains unclear to me what the authors mean with an i <-> j alignment for any x_i, r_j (line 145). I guess it refers to the matching step in Algorithm 1, but the paper would benefit from a clear definition of what alignment means in this case. In the subsequent line, it is also unclear what it means for an algorithm to be "independent of CE". Can the authors clarify this?

### Soundness
3

### Presentation
3

### Contribution
3
