=== CALIBRATION EXAMPLE 47 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Title Relevance**: The title accurately reflects the paper's core focus on evaluating and improving the calibration of conformal prediction set sizes.
- **Abstract Clarity**: The abstract clearly outlines the problem (alignment of PSS with predictive correctness), the proposed methodology (definition of CP calibration, $1/k^\tau$ target, CPAC bi-level optimization), and the empirical scope.
- **Unsupported/Overstated Claims**: The abstract states the calibration target is proposed "based on a theoretical analysis of the predictive distributions." However, Theorem 4.2 derives a relationship involving the total number of classes $K$, while the proposed target scales with the set size $k$. This transition is heuristic rather than a direct derivation. The abstract would benefit from toning down the claim of a strictly "theoretically derived" target or clarifying that the theory motivates the power-law form empirically adapted to $k$.

### Introduction & Motivation
- **Problem Motivation & Gap**: The motivation is strong and well-contextualized. The distinction between confidence-based calibration and PSS calibration in CP is clearly articulated, and the separation from conditional coverage is an important clarification. Citing the lack of systematic classification-focused work (contrasting with regression studies) establishes a clear gap.
- **Contributions**: The three contributions are stated clearly and align with the paper's content.
- **Overclaim/Undersell**: The introduction adequately scopes the problem. However, the claim of "first systematic investigation" should be carefully bounded, as prior works on CP efficiency adaptivity and rank calibration implicitly touch on set-size behavior. Positioning the work as focusing specifically on *explicit PSS-to-accuracy calibration curves and post-hoc optimization* would be more precise. The introduction does a good job motivating the three challenges without over-selling.

### Method / Approach
- **Clarity & Reproducibility**: The definition of CP calibration via multinomial sampling (Eq 4-5) is intuitive and reproducible. The CPAC algorithm structure (Alg 1) is clear at a high level.
- **Assumptions & Theoretical Gaps**:
    1. **Theorem 4.2 to Target Mismatch**: Theorem 4.2 proves $E[\mathbf{q}\cdot\mathbf{p}] = K^{-\tau}$ under Dirichlet assumptions, where $K$ is the total class count. However, the calibration target used in optimization is $f(k) = 1/k^\tau$, where $k=|S_i|$ is the *variable prediction set size*. The manuscript transitions from $K$ to $k$ based on the intuition that effective set size decreases, but does not formally justify why a theorem about the full simplex applies conditionally to subsets of size $k$. This is a logical gap; the power-law is empirically plausible but theoretically unsupported for variable $k$.
    2. **Non-Differentiable Optimization**: Equation 9 optimizes over $\sum_{j \in S_i}$. The set $S_i$ is determined by a rank-order cut-off at quantile $\nu$. As $W, b$ change, elements can jump in/out of $S_i$, making the objective non-differentiable. The method does not specify how gradients are computed through the discrete set boundary (e.g., straight-through estimator, continuous relaxation, or freezing $S_i$ during backpropagation). This omission makes the optimization procedure incomplete.
    3. **Bi-Level Approximation**: The assumption that $\nu$ does not change "drastically" during the $(W, b)$ update is not validated. In split CP, $\nu$ is an order statistic of the scores; small logit shifts near the quantile boundary can cause discrete jumps in $\nu$ and $S_i$.
- **Coverage Guarantee Violation Risk**: Conformal prediction guarantees rely on the score function being fixed before exposure to $D_{cal}$. By optimizing $W, b$ directly on $D_{cal}$ to minimize CP-ECE, the model adapts to the calibration data, potentially violating exchangeability conditions and invalidating the theoretical $1-\alpha$ coverage guarantee. While empirical coverage is reported near target, the theoretical breach should be addressed (e.g., by using a nested split or proving bounded violation).

### Experiments & Results
- **Testing Claims**: The experiments directly evaluate the proposed CP-ECE metrics and demonstrate improvement via CPAC. The use of perturbations to test robustness is appropriate.
- **Baselines & Ablations**: Baselines (PS, PS-Full) are reasonable comparisons to standard Platt scaling. However, missing material ablations weaken the conclusions:
    1. **Gradient Handling**: No ablation compares different strategies for handling the non-differentiability of $S_i$ in Eq 9.
    2. **Selection Bias**: Sec 5.1 notes CPAC is only applied to "low PSS samples" (PSS < 400). Since Uniform CP-ECE weighs all $k$ equally, optimizing only low-$k$ samples could artificially lower the reported metric without improving tail calibration. An ablation restricting the metric computation to the optimized region, or showing performance on high-PSS bins separately, is needed to validate generalization.
    3. **Hyperparameter Sensitivity**: The impact of the temperature $t$ for sampling on optimization stability and final calibration is not analyzed.
- **Error Bars/Significance**: Standard deviations over 5 seeds are provided in tables, which is good practice. Formal statistical testing is absent.
- **Results Support Claims**: Results generally support the claim that CPAC reduces calibration error. However, the claim of "decreasing PSS" is context-dependent; Table 4 shows PSS *increases* when coverage is strictly fixed to 90%, which contradicts the efficiency narrative. The authors note this but dismiss it as impractical. A trade-off curve (Pareto frontier of PSS vs CP-ECE) would present a more honest view.

### Writing & Clarity
- **Clarity of Explanations**: The core definitions are understandable. The discussion on Dirichlet vs logistic-normal assumptions in Appendix A is rigorous.
- **Figures & Tables**: Reliability diagrams effectively visualize the calibration gaps. Table parsing artifacts are ignored per instructions, but the column headers for CP-ECE vs Accuracy vs Cov vs PSS could be grouped more clearly in the final version. The link between Figure 6 entropy analysis and the $\tau$ evolution is clear and insightful.

### Limitations & Broader Impact
- **Acknowledged Limitations**: The authors correctly identify the lack of theoretical convergence guarantees for the bi-level optimization as a weakness.
- **Missed Fundamental Limitations**:
    1. **Computational Cost**: Bi-level optimization requiring conformity score computation and quantile search per training step is $O(N_{cal} \log N_{cal})$ per iteration. Compared to closed-form or iterative temperature scaling, this is expensive. The paper should discuss runtime trade-offs.
    2. **Method Applicability**: The method is demonstrated on APS. It is unclear how CPAC extends to other CP schemes like RAPS (which penalizes size) or LAC.
    3. **Calibration Set Overfitting**: As noted in Method, adapting weights to $D_{cal}$ risks overfitting the calibration split, which is not discussed in the limitations.
- **Societal Impact**: Calibrated uncertainty is critical for safety-critical deployments. The work has positive potential impact. No negative impacts are anticipated, though reliance on post-hoc tuning could introduce complexity barriers for adoption in resource-constrained settings.

### Overall Assessment
This paper addresses a meaningful and underexplored gap in uncertainty quantification: the calibration of conformal prediction set sizes versus prediction accuracy. The empirical study reveals that pre-trained models often suffer from poor PSS calibration, which is a valuable insight for practitioners. The proposed CPAC algorithm offers a practical approach to improve alignment and demonstrates solid empirical gains across vision and language models. 

However, the contribution faces significant methodological hurdles that must be resolved for ICLR standards. The theoretical justification for the $1/k^\tau$ target does not rigorously follow from Theorem 4.2 (which concerns total class count $K$, not set size $k$). The bi-level optimization formulation glosses over the non-differentiability of set inclusion and the risk of violating CP exchangeability guarantees by training on the calibration split. Additionally, the practice of optimizing only low-PSS samples introduces potential bias in the Uniform CP-ECE metric that requires deeper analysis. **Recommendation: The paper contains a novel concept and strong empirical scope, but requires theoretical clarification on the target function derivation, rigorous handling of the discrete optimization landscape, and a transparent discussion of coverage preservation and computational costs.**

# Neutral Reviewer
## Balanced Review

### Summary
This paper investigates whether the prediction set size (PSS) in conformal prediction (CP) is well-calibrated with respect to actual predictive accuracy, addressing an underexplored gap in uncertainty quantification. The authors define a CP calibration metric via multinomial sampling over the prediction set, propose a power-law target function motivated by Dirichlet distribution assumptions, and introduce CPAC, an alternating optimization algorithm that calibrates model logits prior to quantile computation. Extensive experiments across vision and language models demonstrate that CPAC significantly reduces calibration error (Uni CP-ECE) under both clean and perturbed conditions.

### Strengths
1. **Addresses a Meaningful UQ Gap:** The paper correctly identifies that while CP guarantees marginal coverage, it lacks guarantees on whether set size reliably reflects predictive correctness. This is highly relevant for risk-aware decision-making where users rely on set size as a proxy for uncertainty (Sec 1).
2. **Comprehensive Empirical Evaluation:** The experimental design is robust, covering multiple architectures (ResNet, ViT, GPT-2), datasets (CIFAR100, ImageNet, Topic Classification), and diverse perturbation regimes (Gaussian noise, blur, dropout, typos) (Sec 5.1). Reporting both standard and uniform CP-ECE provides a thorough view of calibration behavior.
3. **Clear Algorithmic Contribution:** CPAC (Algorithm 1) offers a practical, post-hoc calibration procedure that directly optimizes a defined calibration objective. Empirical results consistently show reductions in Uni CP-ECE compared to standard Platt scaling baselines while maintaining valid coverage (Tables 1, 2, 5).

### Weaknesses
1. **Mischaracterization of the Optimization Scheme:** Section 4.3 frames the problem as "bi-level optimization," but Algorithm 1 implements a simple alternating update that assumes the quantile $\nu$ remains fixed during inner gradient steps. True bi-level optimization requires handling the implicit dependence of $\nu$ on $(\mathbf{W}, \mathbf{b})$. The paper sidesteps this without discussing approximation error or convergence guarantees.
2. **Questionable Subset Training Strategy:** Section 5.1 states optimization is restricted to low-PSS samples ($<400$ on ImageNet, $<70$ for Topic Cls.) because "we only need to cover $(1-\alpha)$ of all samples in CP." This reasoning is statistically misaligned. The CP quantile $\nu$ depends on the empirical distribution of *all* calibration scores. Optimizing the loss on a subset changes that subset's scores, which alters $\nu$ for the entire set. The interaction between subset gradients and the full-set quantile in Line 2 of Alg. 1 is not justified.
3. **Under-discussed Efficiency Trade-off:** The main text emphasizes decreased PSS, but Appendix C (Table 4) shows that when coverage is strictly controlled at 90%, CPAC consistently *increases* PSS. The authors dismiss this as "not doable in practice," yet fixed-coverage comparison is the standard metric for CP efficiency. This trade-off is crucial and warrants rigorous analysis rather than dismissal.
4. **Strong Distributional Assumptions for Theoretical Target:** Theorem 4.2 derives the $1/k^\tau$ target under the assumption that predictive probabilities and sampling weights follow Dirichlet distributions. The paper acknowledges in Appendix A that this is merely an "illustrative instantiation." Modern DNN logits rarely satisfy these assumptions, making the theoretical derivation more heuristic than rigorous for the chosen target function.

### Novelty & Significance
The paper presents a highly novel perspective on conformal prediction by shifting focus from marginal coverage to set-size calibration, a property largely overlooked in CP literature. The formulation of a calibration target function and the CPAC algorithm are original contributions that meaningfully advance how CP is evaluated and refined. In terms of significance, the work aligns well with ICLR’s emphasis on trustworthy ML and uncertainty quantification. However, its practical impact is currently tempered by methodological ambiguities in the optimization scheme, the lack of comparison against adaptive CP baselines (e.g., RAPS, SAPS), and the noted increase in set size under fixed coverage. If these concerns are resolved, the work would reach ICLR's acceptance bar.

### Suggestions for Improvement
1. **Accurately Describe and Strengthen the Optimization:** Rename "bi-level optimization" to "alternating optimization" in Sec 4.3 and Alg 1. Explain why fixing $\nu$ during gradient steps is empirically sufficient, or implement a proper bi-level solver with implicit gradients to ensure theoretical consistency.
2. **Justify or Modify the Subset Training Protocol:** Clarify how restricting loss computation to low-PSS samples aligns with full-set quantile estimation. Either optimize over the complete calibration distribution to ensure gradient consistency, or provide a formal argument showing that subset optimization does not degrade coverage validity or quantile estimation.
3. **Analyze the Coverage-Efficiency Trade-off:** Directly address the PSS increase observed in Table 4 when coverage is controlled. Explain why CPAC expands sets (e.g., logit reshaping alters the score distribution) and explore adding an explicit efficiency regularizer to Equation 9 to recover smaller sets while maintaining calibration.
4. **Include State-of-the-Art CP Baselines:** Compare CPAC against modern adaptive conformal methods known for improved set calibration and efficiency, such as RAPS and SAPS. This will better contextualize CPAC's contributions beyond standard temperature scaling.
5. **Clarify the Multinomial Sampling Metric:** Provide a stronger rationale for using stochastic sampling (Eqs 2-4) to define expected accuracy instead of deterministic proxies (e.g., average confidence, entropy). Discuss how sampling variance at different temperatures affects ECE computation, and consider ablation experiments showing that deterministic metrics yield similar calibration conclusions.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Strict exchangeability validation for coverage guarantees:** CPAC optimizes logits and estimates the quantile $\nu$ on the *same* calibration split, breaking split-CP exchangeability. Report empirical coverage on a strictly held-out calibration set or use nested V-fold CP; otherwise, the distribution-free guarantee is empirically compromised and the coverage numbers are unreliable.
2. **Controlled ablation of full matrix vs. scalar temperature:** The paper claims optimizing a full $(\mathbf{W}, \mathbf{b})$ outperforms scalar temperature scaling (PS) without an isolated ablation. Run the identical bi-level optimization using only a scalar/vector temperature to prove the full matrix is necessary for performance gains rather than overfitting high-dimensional logits.
3. **Direct comparison to advanced CP baselines:** Compare against RAPS (Randomized Adaptive Prediction Sets) and THR, which are standard efficiency-calibration methods. Without this, it is impossible to attribute Uni CP-ECE reductions to the proposed calibration target rather than generic CP refinements already present in state-of-the-art set selection.

### Deeper Analysis Needed (top 3-5 only)
1. **Metric validity for uniform sampling ($t=0$):** Under uniform multinomial sampling from a normalized prediction set, expected accuracy is mathematically guaranteed to be exactly $1/k$. The Uni CP-ECE deviation from $1/k^\tau$ therefore measures the choice of exponent $\tau$, not actual model miscalibration. Prove analytically or empirically that the metric isolates predictive uncertainty bias rather than trivial set-construction artifacts.
2. **Pareto trade-off between PSS and calibration error:** CPAC frequently increases PSS significantly while reporting lower Uni CP-ECE. Because the target function is monotonically decreasing, arbitrarily inflating PSS will mechanically reduce the error. Provide an explicit efficiency frontier showing that CPAC improves calibration *per unit of set size*; otherwise, the method simply trades away CP's core advantage (tight sets) for a better curve-fit score.
3. **Robustness of Dirichlet/Logistic-Normal assumptions:** Theorem 4.2 assumes predictive distributions follow Dirichlet laws. Provide empirical density tests or sensitivity bounds showing how CPAC performance degrades when real softmax outputs violate these concentration assumptions, as deep classifier logits rarely conform to Dirichlet families.

### Visualizations & Case Studies
1. **Failure mode visualization of inflated sets:** Plot concrete prediction sets pre- and post-CPAC for instances where PSS drastically increases. This reveals whether the method correctly expands sets only for genuine ambiguities or blindly adds low-probability classes to satisfy the target curve, which would indicate metric-gaming rather than uncertainty improvement.
2. **Stratified reliability diagrams by input difficulty/class:** Plot Uni CP-ECE conditioned on class frequency and noise severity. Aggregate error metrics obscure severe subgroup misalignment; demonstrating that calibration holds across easy/hard slices is essential to claim CP is "well-calibrated" across distributions.

### Obvious Next Steps
1. **Resolve quantile estimation data leakage:** The current algorithm computes $\nu$ on $D_{cal}$ and optimizes parameters on the same $D_{cal}$. Implement a three-way split (train / quantize-calibrate / optimize) or jackknife-aggregation to prevent the optimizer from memorizing an over-optimistic threshold.
2. **Report computational overhead and scaling behavior:** Bi-level optimization over full logit dimensions for ViT-L and GPT-2 is likely expensive. Provide exact wall-clock training times and FLOP counts per dataset to prove the method is practically feasible for the large-scale models it claims to address.

# Final Consolidated Review
## Summary
This paper investigates whether the prediction set size (PSS) in conformal prediction (CP) is well-calibrated with respect to predictive accuracy, defining a CP-specific calibration metric via multinomial sampling and proposing a power-law target curve $1/k^\tau$. The authors introduce CPAC, an optimization routine that adjusts model logits on a calibration split prior to quantile computation to reduce calibration error. Empirical results across vision and language models demonstrate that CPAC reduces curve-fitting calibration error under various perturbations, though it introduces methodological compromises regarding theoretical guarantees and optimization rigor.

## Strengths
- **Identifies an Underexplored UQ Property:** The paper cleanly separates marginal coverage (a CP guarantee) from PSS-to-accuracy alignment, providing a clear definition and metric (Standard/Uniform CP-ECE) that captures how well set sizes reflect per-instance reliability.
- **Comprehensive & Diagnostic Empirical Study:** Experiments span multiple architectures (ResNet, ViT, GPT-2), datasets, and realistic perturbation regimes (noise, blur, dropout, typos). The reliability diagrams effectively reveal that high-accuracy pre-trained models can exhibit poorer PSS calibration than randomly initialized ones, a valuable insight for practitioners deploying CP in the wild.
- **Practical Post-Hoc Calibration Procedure:** CPAC empirically reduces calibration error relative to standard Platt scaling while maintaining empirical coverage near the nominal $1-\alpha$ level, demonstrating that logit reshaping can improve the interpretability of CP uncertainty.

## Weaknesses
- **Violates the Exchangeability Assumption Undermining CP Guarantees:** CPAC optimizes logits on the calibration set $D_{cal}$ *before* computing the empirical quantile $\hat{\nu}$ on the same split. This breaks the fundamental exchangeability assumption of split conformal prediction, as the optimization process adapts the model to the very scores used to determine the coverage threshold. Consequently, the theoretical $1-\alpha$ guarantee is invalidated; the reported near-90% coverage is entirely data-dependent and cannot be claimed as a distribution-free property.
- **Theoretical Derivation Does Not Support the Proposed Target Function:** Theorem 4.2 rigorously derives $E[\mathbf{q} \cdot \mathbf{p}] = K^{-\tau}$ under Dirichlet assumptions, where $K$ is the *total number of classes*. The calibration target, however, is applied to the *variable prediction set size* $k = |S_i|$. The manuscript acknowledges this gap but frames the target as theoretically motivated; in reality, replacing the fixed $K$ with the stochastic $k$ is an unproven heuristic that lacks formal justification for conditional subsets.
- **Misleading Efficiency Claims & Absence of Adaptive CP Baselines:** The abstract and introduction emphasize set tightness, yet Appendix C (Table 4) shows CPAC consistently *increases* PSS when coverage is strictly controlled at 90%. Dismissing fixed-coverage comparisons as "not doable in practice" ignores a standard CP benchmark. Furthermore, the paper compares only to Platt scaling. Without evaluation against adaptive conformal methods like RAPS or SAPS, which explicitly regularize set size and improve efficiency-calibration trade-offs, it is unclear whether gains stem from the novel target or generic logit adjustments already captured by existing CP frameworks.
- **Optimization Heuristics Lack Rigorous Formulation:** The method is framed as "bi-level optimization," but Algorithm 1 implements a basic alternating update that freezes the quantile $\nu$ during gradient steps, ignoring the implicit dependence of $\nu$ on the logits. Crucially, the objective sums over $j \in S_i$, where $S_i$ changes discretely as logits shift, introducing non-differentiability that is neither addressed nor relaxed. Restricting optimization to "low-PSS samples" (e.g., $k < 400$ on ImageNet) further compounds this: it optimizes a subset while reporting Uniform CP-ECE (which weights all $k$ bins equally), creating an evaluation mismatch that risks artificially suppressing the reported error without ensuring tail calibration.

## Nice-to-Haves
- Provide a Pareto frontier analysis explicitly trading off CP-ECE reduction against PSS growth to demonstrate whether calibration improvements come at an acceptable efficiency cost.
- Report wall-clock training time and FLOP overhead of CPAC compared to standard temperature scaling to assess practical feasibility on large models like ViT-L and GPT-2.
- Ablate the full $(\mathbf{W}, \mathbf{b})$ optimization against a scalar/vector temperature scaling version of the same objective to isolate the contribution of high-dimensional logit reshaping vs. simpler scaling.

## Novel Insights
The systematic finding that modern, high-accuracy pre-trained classifiers frequently exhibit worse PSS calibration than smaller or randomly initialized counterparts is counter-intuitive and methodologically significant. It suggests that representational learning and confidence concentration in deep networks may inadvertently decorrelate from the rank-order structure that CP relies upon for set construction. The paper's observation that input perturbations compress the entropy within prediction sets while flattening the global probability landscape (Fig. 6) provides a compelling diagnostic of how model robustness and conformal uncertainty interact under distribution shift, highlighting a previously overlooked tension in UQ design.

## Potentially Missed Related Work
- **Angelopoulos et al. (2020/2021) - RAPS & SAPS** — These methods introduce size-regularization and adaptive score penalization directly into the CP framework, offering established baselines for the tension between set tightness and calibration/coverage that this paper investigates.
- **Stutz et al. (2023/2024) - Conditional Conformal Predictions / Rank-Calibration** — Works exploring the relationship between model confidence rankings and conformal set validity, which directly intersect with the PSS-accuracy alignment problem.

## Suggestions
- **Isolate CP Quantile Estimation:** To preserve the distribution-free guarantee, implement a strict three-way split: use one subset to compute the initial quantile $\hat{\nu}$, and a disjoint subset to run CPAC optimization. Alternatively, reframe CPAC as a pre-training or early-stage calibration step that precedes any CP score computation.
- **Reframe Theoretical Motivation & Target Function:** Clearly state that $f(k) = 1/k^\tau$ is an empirically motivated proxy inspired by the $K^{-\tau}$ derivation, not a direct theoretical consequence. Provide an ablation or theoretical bound showing how well the $1/k^\tau$ curve fits real softmax distributions vs. the Dirichlet assumption.
- **Formalize the Optimization Landscape:** Drop the "bi-level" terminology and accurately describe the alternating scheme. Explicitly detail how gradients are computed through the discrete $S_i$ boundary (e.g., straight-through estimator, frozen set membership during backprop, or continuous relaxation). Justify mathematically or empirically why fixing $\nu$ during updates does not degrade convergence.
- **Expand Baselines:** Include RAPS and/or SAPS as comparative methods. Even if CPAC targets a different metric, showing how it performs against adaptive CP methods that natively handle set-size regularization will contextualize the contribution within modern UQ literature.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 2.0, 2.0]
Average score: 3.5
Binary outcome: Reject
