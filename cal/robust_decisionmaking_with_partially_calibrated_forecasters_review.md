=== CALIBRATION EXAMPLE 73 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly reflects the paper’s contribution. The abstract succinctly states the problem, approach, and main results. The claim that decision calibration recovers best-response optimality in a minimax sense is a central and well-supported result.

### Introduction & Motivation
The introduction effectively motivates the problem: full calibration is intractable in high dimensions, while weaker calibration notions lack clear decision-theoretic guarantees. The paper’s central question—how to derive optimal decision policies under partial calibration—is well-posed. Contributions are clearly listed in Section 1.1.

### Method / Approach (Sections 2–4)
**Section 2** clearly defines \(H\)-calibration and the robust decision-making framework. Assumption 2.1 (linear utility) is standard but a notable limitation; the paper acknowledges this and suggests future work.

**Section 3 (Theorem 3.1)** characterizes the optimal robust policy via duality. The proof (Appendix A) appears correct, but there is a subtle technical issue: the use of Sion’s minimax theorem requires compactness of the set \(Q\). The authors state \(Q\) is “compact,” but they do not specify the topology. Since \(Q\) consists of measurable functions from \([0,1]^d\) to \([0,1]^d\), compactness is non-trivial. Typically, one works with a weakly compact set (e.g., by considering \(L_2\) constraints), but this should be clarified or justified. This does not invalidate the result but is a gap in the rigorous presentation.

**Section 4** provides insightful specializations. Theorems 4.1 and 4.2 show that decision calibration (or any stronger notion) collapses the robust policy to plug-in best response. This is a sharp and important finding. The proofs rely on the invariance property (equation 9), which is correct under the linear utility assumption. Propositions 4.4 and 4.5 derive practical robust policies for self-orthogonality and bin-wise calibration. These are valuable for applications.

**Overall**, the method is reproducible and the logical flow is sound, modulo the compactness concern.

### Experiments & Results (Section 5)
Experiments on two regression datasets (Bike Sharing, California Housing) validate the theory. The robust policy (based on self-orthogonality) outperforms the plug-in rule under adversarial distribution shifts, with a small cost under i.i.d. conditions. The experiments are adequate but limited in scope:
- Only two datasets and one utility function are used.
- The adversarial distributions are constructed theoretically, but real-world distribution shifts are not tested.
- The choice of \(H = \{h(v)=v\}\) (self-orthogonality) is natural but more calibration notions could be compared.
While the experiments support the theory, broader empirical evaluation would strengthen the paper. Nonetheless, given the theoretical nature of the work, the experiments are sufficient.

### Writing & Clarity
The paper is exceptionally well-written. The narrative is clear, figures are helpful, and the appendix is thorough. No major clarity issues impede understanding.

### Limitations & Broader Impact
The conclusion discusses key limitations: risk-neutral (linear) utilities, finite action sets, and the challenges of non-linear utilities. The broader impact is not explicitly discussed; given the focus on trustworthy decision-making, a brief discussion of societal implications (e.g., in healthcare or finance) would be appropriate, though not required.

### Appendix
Appendix A contains complete proofs, which appear correct and detailed. Appendix B extends the results to approximate calibration, adding value. The proofs there are also rigorous.

### Overall Assessment
This paper makes a significant theoretical contribution by linking partial calibration to robust decision making. The characterization of minimax optimal policies and the collapse to best response under decision calibration are novel and impactful results. The technical execution is strong, though the compactness assumption in Theorem 3.1 should be clarified. The experiments, while limited, adequately support the theory. The paper meets ICLR’s standards for novelty, significance, and clarity. It is likely above the acceptance bar, pending minor revisions to address the technical gap.

# Neutral Reviewer
## Balanced Review

### Summary
This paper studies robust decision-making when forecasts satisfy only partial (weaker) calibration guarantees, formalized as H-calibration. The authors frame the problem as a minimax optimization: choose a decision policy that maximizes worst-case expected utility over all outcome distributions consistent with the H-calibration constraints. They characterize the optimal policy via duality and show that for the tractable notion of decision calibration (and any stronger calibration), the minimax-optimal policy collapses to simply best-responding to the forecasts. For weaker calibration notions (e.g., those arising from squared-loss regression), they derive efficiently computable robust policies and demonstrate their performance empirically.

### Strengths
1. **Novel and well-motivated problem**: The paper addresses a critical gap between calibration theory and practical decision-making, providing a principled framework for acting on forecasts that are not fully calibrated. This is highly relevant for high-stakes applications where full calibration is intractable.
2. **Theoretical elegance and insight**: The duality-based characterization of the minimax optimal policy is clean and general. The “sharp transition” result—that decision calibration suffices for plug-in best response to be optimal—is both surprising and significant, as it justifies a tractable calibration notion with strong decision-theoretic guarantees.
3. **Practical relevance and empirical validation**: The paper includes experiments on real-world datasets (Bike Sharing and California Housing) that validate the theory: the robust policy outperforms the plug-in rule under adversarial distribution shifts consistent with the calibration constraints, with only a minor performance drop under i.i.d. conditions. This demonstrates the framework’s practical utility.
4. **Clear exposition**: Despite the technical nature, the paper is well-structured, with intuitive explanations of key ideas (e.g., the interpolating property between aggressive and conservative extremes) and a thorough discussion of related work.

### Weaknesses
1. **Restrictive utility assumption**: Assumption 2.1 requires linearity of the utility in the outcome probabilities. While common in the calibration literature, this excludes risk-averse or non-linear decision makers, limiting the scope of applications. The authors acknowledge this but do not offer a workaround.
2. **Limited empirical evaluation**: Experiments are confined to two regression datasets with simple utility functions and a specific H-calibration derived from squared loss. Broader evaluation—including classification tasks, more complex neural networks, and diverse utility structures—would strengthen the empirical claims.
3. **Computational considerations under-explored**: Although the policies are claimed to be efficiently computable for finite H and finite actions, no detailed complexity analysis or scalable implementations are provided. For large action sets or high-dimensional forecasts, the pointwise optimization might become burdensome.
4. **Narrow focus on finite dimensions**: The theory assumes finite-dimensional H and finite action sets. While this covers many cases, extensions to continuous or infinite action spaces and nonparametric H are left as future work, which may limit applicability in some settings.

### Novelty & Significance
The paper makes a novel contribution by bridging calibration guarantees with minimax robust decision-making. The key insight—that decision calibration recovers the “trust the forecast” semantics in a minimax sense—is significant because decision calibration is far more tractable than full calibration, especially in high dimensions. This provides a strong justification for using decision calibration in practice. The work also unifies several lines of research (calibration, robust optimization, and decision theory) and offers a practical framework for robust decision-making under partial trustworthiness. The results meet ICLR’s standards for novelty and potential impact.

### Suggestions for Improvement
1. **Generalize the utility assumption**: Explore extensions to non-linear utilities, perhaps via linearization over bases (as hinted in the conclusion) or by deriving robust policies for specific classes of convex/concave utilities. This would greatly broaden the applicability.
2. **Expand empirical evaluation**: Include experiments on multi-class classification datasets, vary the utility functions (e.g., risk-sensitive utilities), and test with more complex models (e.g., deep neural nets) to demonstrate robustness across settings.
3. **Provide algorithmic details and complexity analysis**: Add pseudocode for computing the robust policy, discuss computational bottlenecks, and suggest approximations or scalable implementations for large-scale problems.
4. **Clarify connections to distributionally robust optimization (DRO)**: The ambiguity set Q defined by moment constraints is reminiscent of DRO. A more explicit discussion of this relationship could help situate the work within the broader robust optimization literature.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **No validation for the core claim about decision calibration.** The paper proves that decision calibration makes the plug-in rule minimax optimal, but provides no empirical demonstration. To validate this key theoretical result, an experiment should train a decision-calibrated forecaster (using existing algorithms) and show the robust policy indeed collapses to the plug-in rule.
2. **Only one calibration notion (self-orthogonality) is tested.** The general framework is claimed to work for any H, but experiments do not test other natural choices (e.g., bin-wise calibration or top-label calibration). Without this, the practical breadth of the method is unsubstantiated.
3. **Adversarial evaluations are synthetic and not representative of real shifts.** The constructed worst-case distributions are theoretical tools; the paper lacks evaluation on realistic distribution shifts (e.g., temporal, geographic, or subpopulation shifts) that are common in practice. This undermines the claimed practical utility.
4. **No comparison to existing robust decision-making baselines.** Methods like conformal prediction or Bayesian uncertainty quantification also offer ways to act conservatively. Without comparisons, it is unclear whether the proposed robust policy offers any advantage.

### Deeper Analysis Needed (top 3-5 only)
1. **The linear utility assumption is crucial but unexplored.** The theory requires utilities linear in the outcome probabilities. The paper should discuss how restrictive this is for real decision problems (e.g., risk-averse or safety-critical settings) and whether approximations (like basis expansions) are viable.
2. **Computational scalability for large or infinite H is not addressed.** The dual optimization dimension scales with |H|. For decision calibration with many actions, or for combining many decision problems, solving the dual may become costly. An analysis of computational complexity and scalable approximations is missing.
3. **The relationship to swap regret guarantees is not sufficiently clarified.** The paper claims decision calibration yields minimax optimality, which is stronger than swap regret. A direct comparison showing how minimax optimality subsumes or differs from swap regret would strengthen the theoretical contribution.

### Visualizations & Case Studies
1. **Visualize how the robust policy deviates from the plug-in rule.** For a simple 2D forecast space (e.g., two-class probabilities), plot the action regions of the robust policy versus the plug-in policy under a specific H (like self-orthogonality). This would concretely show how robustness alters decisions.
2. **Case studies illustrating failure modes.** Show concrete examples (e.g., with specific utility functions and forecasts) where the robust policy incurs significant utility loss under i.i.d. data, quantifying the trade-off between robustness and nominal performance.

### Obvious Next Steps
1. **Implement and test decision calibration.** The most immediate next step is to empirically verify the collapse to plug-in best response when decision calibration holds, using existing calibration algorithms.
2. **Release a general-purpose solver for the robust policy.** Provide open-source code that computes the optimal robust policy for any user-specified H and utility function, enabling adoption and further research.
3. **Extend experiments to classification tasks and larger action spaces.** The current experiments are on regression with tiny action sets. Testing on multi-class classification with more complex decision problems would demonstrate broader applicability.

# Final Consolidated Review
## Summary
This paper studies robust decision-making when forecasts satisfy only partial calibration guarantees. It characterizes the minimax-optimal decision policy via a duality argument and shows that, surprisingly, under the tractable notion of decision calibration, the optimal policy collapses to simply best-responding to the forecasts. Experiments on regression datasets demonstrate the practical value of the robust policy under adversarial distribution shifts.

## Strengths
- The paper addresses a critical gap between calibration theory and practical decision-making, providing a principled framework for acting on forecasts that are not fully calibrated, which is especially relevant in high-dimensional settings where full calibration is intractable.
- The theoretical analysis yields a sharp and insightful result: decision calibration—a tractable notion—suffices for the plug-in best response to be minimax optimal, effectively recovering the "trust the forecast" semantics of full calibration. This provides a strong justification for using decision calibration in practice.
- The framework is empirically validated on real-world regression datasets, showing that the proposed robust policy outperforms the plug-in rule under adversarial shifts consistent with the calibration guarantees, with only a minor performance drop under i.i.d. conditions.

## Weaknesses
- The linear utility assumption (Assumption 2.1) restricts the framework to risk-neutral decision makers, excluding important classes of problems with risk-averse or non-linear utilities. While acknowledged, this limitation narrows the applicability of the results.
- The empirical evaluation is limited in scope: only two regression datasets and one calibration notion (self-orthogonality) are tested. The key theoretical claim about decision calibration is not empirically validated, and the experiments do not cover classification tasks or more diverse utility structures.
- The proof of Theorem 3.1 relies on the compactness of the set \(Q\) without explicitly justifying the topology or providing a rigorous compactness argument. While this does not invalidate the results, it leaves a technical gap in an otherwise rigorous presentation.

## Nice-to-Haves
- A more extensive empirical evaluation, including validation of the decision calibration collapse and comparisons to other robust decision-making baselines (e.g., conformal prediction), would strengthen the practical claims.
- Discussion of computational complexity and scalable approximations for large action sets or high-dimensional \(H\) would aid implementation.
- Visualizations illustrating how the robust policy deviates from the plug-in policy in simple settings could enhance intuition.

## Novel Insights
The paper provides a novel connection between partial calibration and minimax robust decision-making. Its core insight is that decision calibration—a tractable and much weaker condition than full calibration—recovers the same "trust the forecast" semantics when viewed through a minimax lens. This collapse is surprising because it shows that a small set of calibration constraints (one per action) is sufficient to guarantee that best-responding is the optimal robust policy, effectively bridging the gap between theoretical intractability and practical trustworthiness.

## Suggestions
- Clarify the compactness argument in Theorem 3.1 (e.g., by specifying the topology or adding a brief justification) to strengthen the mathematical presentation.
- In the experimental section, include at least a small-scale validation of the decision calibration result (e.g., using a synthetic dataset or an existing decision calibration algorithm) to empirically confirm the theoretical collapse.
- Expand the related work discussion to explicitly connect the ambiguity set \(Q\) to distributionally robust optimization, highlighting how the calibration constraints lead to a novel and structured uncertainty set.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0]
Average score: 7.3
Binary outcome: Accept
