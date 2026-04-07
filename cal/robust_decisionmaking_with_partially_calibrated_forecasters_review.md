=== CALIBRATION EXAMPLE 73 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**
The title accurately reflects the core contribution (robust decision-making with partially calibrated forecasts). The abstract clearly states the problem, approach, and main results (minimax optimal decision rule, collapse to best response under decision calibration). The claim that decision calibration recovers the "trustworthiness" semantics of full calibration is a strong and compelling summary. No substantive issues.

**Introduction & Motivation**
The problem is exceptionally well-motivated, bridging the gap between the appealing decision-theoretic guarantees of full calibration and its practical intractability in high dimensions. The framing of "On the Model Side" vs. "On the Decision Making Side" is effective. The contributions (listed in Section 1.1) are clearly and precisely stated. The introduction successfully argues that the paper's minimax lens is a novel and needed perspective in this literature.

**Method / Approach**
*Section 2 (Problem Setup)*: The definition of H-calibration and the robust decision-making problem (Eq. 5) are clear. **A major assumption (2.1) is that the utility is linear in the forecast vector `v`.** This is explicitly stated and justified for risk-neutral expected utility settings but is a significant restriction. The paper correctly notes it as a direction for future work, but reviewers may question the practical scope of this linearity assumption (e.g., it excludes any consideration of risk or variance).
*Section 3 (General Characterization)*: Theorem 3.1 is the core technical result. The proof (in Appendix A) appears correct and leverages standard tools (Sion's minimax theorem, Lagrangian duality). The characterization of the optimal policy and worst-case belief via a dual multiplier `λ*` is elegant and provides a computational pathway. A minor point: the theorem states the policy is a best response to `q*(v)`, but the computation of `λ*` itself requires solving an optimization over the distribution of `f(X)`. The text mentions this can be done via standard methods, which is acceptable, but the complexity of this outer-loop optimization could be noted as a practical consideration.
*Section 4 (Special Cases)*:
    *Theorems 4.1 & 4.2 (Decision Calibration)*: These are the most striking results. The proof that the plug-in best response is minimax optimal under (or with) decision calibration is clever, using the invariance property (Eq. 9 in the proof). This is a significant conceptual contribution, elevating decision calibration from a swap-regret guarantee to a full minimax-optimality guarantee. The "sharp transition" point is well-illustrated in Figure 2.
    *Proposition 4.4 (Self-orthogonality)*: This is a useful and practical insight, connecting standard squared-error regression training to a specific H-calibration guarantee. It grounds the theory in common practice.
    *Proposition 4.5 (Bin-wise Calibration)*: Provides a simple, interpretable robust policy. This is a good example of instantiating the general framework.
*Overall Methodological Assessment*: The theoretical development is rigorous, novel, and well-structured. The assumptions are clearly flagged. The transition from a general characterization to interpretable special cases is logical. The reproducibility is high given the detailed proofs in the appendix.

**Experiments & Results**
This is the section most likely to be critiqued at a conference like ICLR, which often expects empirical validation even for theory papers.
*Strength*: The experiments directly test a theoretical prediction: the robust policy (`a_robust`) should outperform the plug-in policy (`a_BR`) under adversarial distributions consistent with the H-calibration guarantee, while potentially paying a mild cost under i.i.d. conditions. The results in Table 1 confirm this pattern.
*Weaknesses/Limitations*:
    1. **Scale and Scope**: The experiments are limited to two standard regression datasets with small, discrete action sets (3 actions). While sufficient for proof-of-concept, they do not demonstrate the framework's utility in more complex, high-dimensional decision problems (e.g., large action spaces, structured outputs) which the theory is designed to address.
    2. **Adversary Construction**: The description of how the "worst-case" adversarial test distributions are generated is vague. The text says they are "tailored to the plug-in policy" and "induced by the robust dual," but no algorithmic details are provided in the main text or appendix. This makes the experimental results difficult to reproduce or fully evaluate.
    3. **Missing Baseline**: A natural conservative baseline is the constant minimax strategy (`argmax_a min_y u(a,y)`), which is the optimal policy under an empty `H`. Comparing to this would better illustrate the benefit of incorporating even weak calibration information (like self-orthogonality).
    4. **Statistical Significance**: No measures of variance (e.g., standard errors over multiple splits/seeds) are reported. The differences, while consistent with theory, are small.
*Conclusion on Experiments*: They serve their primary purpose of illustrating the theoretical concepts but are not a comprehensive empirical study. The lack of detail on adversary generation is a notable omission for reproducibility.

**Writing & Clarity**
Despite OCR/parser artifacts (e.g., "CALI### BRATED", broken equation formatting), the paper is generally well-written and logically organized. The figures (1 and 2) are helpful schematics. The flow from general problem to general solution to special cases is clear. A few points:
* The notation `q` is used both for the conditional expectation map `q(v) = E[Y|f(X)=v]` and for specific values `q(v)`; this is standard but requires careful reading.
* The switch between `v` and `ν` for forecast values is slightly inconsistent.
* Section 3 would benefit from a brief, high-level intuition for Theorem 3.1 before diving into the formal statement, explaining the role of the dual multiplier `λ*` as an "adversarial tilt."
These are minor clarity issues, not fundamental obstacles to understanding.

**Limitations & Broader Impact**
The paper explicitly discusses its main technical limitation: the linearity assumption for utility (Assumption 2.1). It also briefly mentions the intractability of full calibration in high dimensions as motivation. The societal impact discussion is absent, which is reasonable for a paper of this theoretical nature; the framework itself is neutral, aiming to improve the reliability of decisions based on ML forecasts.

**Appendix B (Approximate Calibration)**
This is a substantial and valuable extension, showing stability under `ε`-slack in the calibration constraints. Theorems B.1 and B.2 effectively generalize the main results to the approximate case, providing `O(ε)` degradation bounds. This strengthens the paper's relevance to practical settings where calibration is only approximate. This appendix significantly bolsters the paper's completeness.

### Overall Assessment
This is a strong, theoretically novel paper that makes a clear contribution. It provides a principled, minimax framework for decision-making with partially calibrated forecasts and derives a sharp, insightful result: decision calibration—a tractable condition—suffices to make the simple plug-in best response minimax optimal. The technical execution is rigorous, with complete proofs. The main weaknesses are the limited scale and somewhat underspecified experimental validation, which is common for theory-focused submissions. The linearity assumption is a genuine but acknowledged limitation. The paper meets the high bar for ICLR in terms of novelty, significance, and technical soundness. Addressing the experimental reproducibility concern (detailing the adversary construction) is the most critical point for the authors to clarify in a revision.

# Neutral Reviewer
## Balanced Review

### Summary
This paper addresses how a decision maker should optimally act when given forecasts that satisfy only partial (weaker) calibration guarantees, rather than full calibration. The authors formulate a minimax robust decision problem where the goal is to maximize worst-case utility over all distributions consistent with the promised calibration constraints. They characterize the optimal decision rule via a duality argument and show, surprisingly, that for the tractable notion of decision calibration (and any stronger calibration), the minimax optimal rule collapses to the simple plug-in best response—effectively restoring the "trust the forecast" principle. For weaker calibration notions, they derive efficiently computable robust rules and provide empirical validation on regression datasets.

### Strengths
1. **Novel and well-motivated framework**: The paper introduces a novel minimax perspective for decision making under partial calibration, bridging calibration theory and robust optimization. This addresses a practical gap: full calibration is intractable in high dimensions, but weaker forms are often achieved, and the paper provides a principled way to exploit them.
2. **Theoretically sound and insightful results**: The characterization of the optimal robust policy (Theorem 3.1) is clean and general. The collapse to plug-in best response under decision calibration (Theorems 4.1 and 4.2) is a strong and surprising result, showing that a tractable calibration notion suffices for optimal decision making. Proofs are provided and appear correct.
3. **Practical relevance and algorithmic implications**: The paper connects to practical scenarios by deriving robust rules for calibration guarantees that arise naturally from standard training (e.g., self-orthogonality from squared loss) and post-hoc methods (e.g., bin-wise calibration). The proposed policies are efficiently computable for finite action sets.
4. **Clear exposition**: The paper is well-structured, with a clear problem statement, intuitive explanations, and a good discussion of related work. The interpolation property between aggressive and conservative extremes is nicely illustrated.

### Weaknesses
1. **Limited empirical evaluation**: Experiments are conducted on only two regression datasets with simple, discrete action spaces (3 actions) and synthetic linear utilities. The evaluation under adversarial distribution shift is somewhat contrived. More diverse benchmarks (e.g., multiclass classification, larger action sets, real-world decision tasks) would strengthen the claims of practical applicability.
2. **Restrictive assumptions**: The framework assumes risk-neutral linear utilities and finite action spaces. While common in the literature, this limits the scope. Nonlinear utilities (e.g., risk-averse) and continuous actions are important for many real-world decisions but are left as future work with only brief discussion.
3. **Lack of comparison to alternative robust baselines**: The experiments compare only the proposed robust rule against the plug-in rule. It would be valuable to compare with other robust decision-making approaches or calibration methods to better contextualize the performance gains.
4. **Computational considerations understated**: Although the paper claims efficient computability, the pointwise optimization required for the robust rule may become expensive for very large action sets or high-dimensional forecasts. No discussion of approximation methods or scalability is provided.

### Novelty & Significance
The paper makes a significant contribution by linking partial calibration guarantees to minimax optimal decision rules. The key insight—that decision calibration (a tractable condition) is sufficient for the plug-in best response to be minimax optimal—is both novel and impactful. It provides a rigorous decision-theoretic foundation for using tractable calibration notions in high-stakes applications. The work is timely and aligns well with ICLR's focus on trustworthy and reliable machine learning.

### Suggestions for Improvement
1. **Expand the experimental section**: Include more datasets (especially multiclass classification) and more complex decision problems (e.g., larger action sets, non-linear utilities). Evaluate under a wider range of distribution shifts, including real-world shifts, to better demonstrate robustness.
2. **Discuss computational scalability**: Provide an analysis of the computational cost of the robust rule as a function of the number of actions and forecast dimension. Suggest approximation techniques (e.g., using convex solvers, sampling) for large-scale settings.
3. **Compare with additional baselines**: Incorporate comparisons to other robust decision-making methods (e.g., distributionally robust optimization) and calibration techniques (e.g., temperature scaling, conformal prediction) to better position the proposed approach.
4. **Clarify the adversarial construction in experiments**: In Section 5, detail exactly how the worst-case distributions for the plug-in and robust rules are generated, ensuring reproducibility. Consider including a sensitivity analysis on the slack parameter ε for approximate calibration.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Validate the collapse to plug-in under decision calibration.** The paper's central theoretical result (Theorems 4.1-4.2) states that if the forecaster is decision-calibrated, the minimax optimal rule is the plug-in best response. However, the experiments only test self-orthogonality (H={v}). To substantiate this claim, the authors must train or post-process a model to be decision-calibrated (e.g., using multicalibration algorithms) and demonstrate that the plug-in rule indeed cannot be outperformed under adversarial shifts consistent with decision calibration.

2. **Assess performance under approximate calibration.** The theory assumes exact H-calibration, but practical models only satisfy it approximately. The paper should empirically evaluate how the robust rule's performance degrades with increasing calibration error (ε), and verify the bounds in Appendix B. Without this, the practical utility of the theory is unclear.

3. **Compare to alternative robust decision-making baselines.** The paper lacks comparisons to established methods like distributionally robust optimization (DRO) with moment constraints or Bayesian decision rules. Such comparisons are necessary to demonstrate that the proposed minimax approach offers tangible advantages over existing techniques for handling uncertainty.

4. **Test on diverse decision problems and data modalities.** Experiments are limited to two regression datasets with synthetic three-action linear utilities. To establish generality, the authors should evaluate on classification tasks (multiclass outcomes) and with non-linear utility functions, which are common in real-world decisions.

### Deeper Analysis Needed (top 3-5 only)
1. **Analyze computational scalability.** The paper claims the robust rule is "efficiently computable" for finite H but provides no analysis of runtime, memory, or scalability as |H| grows (e.g., for multicalibration with many groups). Without this, practitioners cannot assess feasibility for complex H.

2. **Discuss the impact of the linear utility assumption.** The entire theoretical framework requires utilities linear in outcome probabilities, excluding risk-averse or other non-linear utilities common in economics and healthcare. The paper should explicitly discuss this limitation and the practicality of linearization techniques mentioned.

3. **Explore sensitivity to the choice of H.** The robust rule's behavior depends critically on the set H of calibration tests. The paper does not guide how to select H when decision calibration is infeasible. An analysis comparing performance across different H (e.g., self-orthogonality vs. bin-wise) on the same problem would provide crucial insight.

4. **Clarify novelty relative to robust optimization with moment constraints.** The paper claims to be the first to apply a minimax lens to partially calibrated forecasts, but robust optimization with moment constraints is well-studied. A deeper discussion situating the work within that literature is needed to articulate the specific contribution.

### Visualizations & Case Studies
1. **Visualize action regions in forecast space.** For a simple 2D forecast (e.g., two-class probabilities), plot the action regions chosen by the plug-in rule versus the robust rule under different H-calibration guarantees. This would concretely show how robustness alters decisions and where conservatism increases.

2. **Case study on a real-world decision problem.** Apply the framework to a high-stakes domain (e.g., medical diagnosis or loan approval) with a realistic utility function and domain-specific calibration guarantees. Demonstrate that the robust rule improves decisions under distribution shift compared to naive plug-in, highlighting practical impact.

3. **Illustrate constructed adversarial distributions.** In experiments, adversarial distributions are tailored to hurt specific policies. Visualizing how these adversaries shift outcomes conditional on forecasts (e.g., via histograms or scatter plots) would build intuition about the nature of the worst-case and how the robust rule counters it.

### Obvious Next Steps
1. **Implement decision calibration and verify theorem.** As a direct validation, implement decision calibration (using existing multicalibration algorithms) on a forecaster and empirically confirm that no policy can outperform plug-in under admissible distribution shifts.

2. **Extend experiments to classification and non-linear utilities.** Test the framework on standard classification datasets (e.g., CIFAR-10) with multiclass forecasts and non-linear utility functions to demonstrate broader applicability beyond regression.

3. **Investigate adaptive or data-driven selection of H.** Propose and evaluate a method to choose H from data (e.g., based on validation performance or complexity) to balance robustness and tractability, addressing a key practical question.

4. **Release open-source code for the robust rule.** To facilitate adoption and reproducibility, provide a well-documented implementation that computes the robust decision rule given a calibrated forecaster, utility function, and calibration set H.

# Final Consolidated Review
## Summary
This paper introduces a minimax framework for decision-making with forecasts that satisfy partial calibration guarantees (H-calibration). It characterizes the optimal robust decision rule via duality and shows that under the tractable notion of decision calibration, the rule collapses to the simple plug-in best response—effectively restoring "trust the forecast" semantics. For weaker calibration, efficient computations are provided, with empirical validation on regression tasks.

## Strengths
- **Novel and well-motivated framework**: Bridges calibration theory and robust optimization, addressing the practical gap between intractable full calibration and weaker, achievable guarantees.
- **Theoretically insightful results**: Theorem 3.1 gives a general characterization via duality, and Theorems 4.1–4.2 show that decision calibration suffices for plug-in best response to be minimax optimal—a sharp and surprising collapse that elevates decision calibration's decision-theoretic status.
- **Practical instantiations**: Connects to common scenarios like self-orthogonality from squared-loss training (Proposition 4.4) and bin-wise calibration (Proposition 4.5), yielding efficiently computable rules for finite action sets.

## Weaknesses
- **Linear utility assumption**: The framework requires utilities linear in outcome probabilities (Assumption 2.1), excluding risk-averse or non-linear utilities common in real-world decisions. While acknowledged as a limitation, this restricts applicability.
- **Limited empirical validation**: Experiments are confined to two regression datasets with small, discrete action sets and do not test the central collapse under decision calibration—only self-orthogonality is evaluated. Adversarial distribution generation is underspecified (Section 5), hindering reproducibility.
- **Missing comparisons to robust baselines**: The experiments compare only plug-in and proposed robust rules; omitting the constant minimax strategy (mentioned as an extreme in the introduction) or other distributionally robust methods makes it hard to gauge the value of incorporating calibration information.
- **Computational scalability not analyzed**: Although claimed "efficiently computable" for finite H, no analysis of runtime, memory, or scalability with |H| or action set size is provided, which is important for practical deployment.

## Nice-to-Haves
- Empirical validation of the collapse theorem using decision-calibrated forecasters (e.g., via multicalibration algorithms).
- More diverse experimental settings, such as multiclass classification or non-linear utility functions, to demonstrate broader applicability.
- Guidance on selecting H when decision calibration is infeasible, perhaps with sensitivity analysis across different H.
- Visualizations of action regions in forecast space to illustrate how robustness alters decisions.

## Removed Points
*These points are flagged to be removed, treat them with caution:*
- **Criticism about outer-loop optimization complexity**: The paper notes standard methods can be used for the dual optimization, and this is a minor implementation detail rather than a core flaw.
- **Request for deeper discussion on novelty relative to robust optimization with moment constraints**: The paper situates itself in related work (Section 1.2), and this is not a substantive weakness.

## Novel Insights
The key insight is that decision calibration—a tractable condition requiring only |A| tests—ensures the plug-in best response is minimax optimal, effectively restoring the "trust the forecast" principle without needing full calibration. This collapses a potential hierarchy of robust rules and provides a rigorous decision-theoretic foundation for using tractable calibration in high-stakes applications, bridging the gap between theory and practice.

## Suggestions
- Detail the algorithm for generating adversarial distributions in experiments (Section 5) to ensure reproducibility.
- Include the constant minimax baseline in comparisons to demonstrate the value of even weak calibration information.
- Report statistical significance or variance measures (e.g., standard errors over multiple splits) in experimental results.
- Consider a brief case study with a real-world decision problem (e.g., medical diagnosis) to highlight practical impact.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0]
Average score: 7.3
Binary outcome: Accept
