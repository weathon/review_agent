=== CALIBRATION EXAMPLE 51 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**  
The title clearly reflects the core contribution: investigating whether prediction set size (PSS) in conformal prediction (CP) is well-calibrated. The abstract succinctly states the problem, defines CP calibration, proposes a target function and a calibration method (CPAC), and summarizes experimental results. The claims are supported by the paper.

**Introduction & Motivation**  
The introduction effectively motivates the problem: while CP provides coverage guarantees, the uncertainty signaled by PSS may not align with predictive correctness, which is critical for risk-aware decisions. The contributions are clearly listed (connection between PSS and accuracy, calibration target, CPAC algorithm). The distinction from prior work (e.g., van der Laan & Alaa 2024 on regression, conditional coverage) is well-drawn.

**Preliminaries**  
Standard notation and background on split conformal prediction (using APS) are provided. No major issues.

**4. Calibration of Conformal Prediction**  

*4.1 Calibration of a Conformalized Model*  
Definition 4.1 introduces CP calibration via multinomial sampling to link PSS to accuracy. This is a creative and reasonable way to obtain a point prediction from a set. The two calibration error metrics (standard and uniform CP-ECE) are defined, with uniform CP-ECE justified as giving equal weight to each PSS bin (important for fairness and curve-fitting). A minor note: the definition depends on the sampling temperature \(t\), which is later fixed to Top-1 (\(t \to \infty\)) in experiments. This is acceptable but should be clarified as a choice.

*4.2 Calibration Target Function*  
The proposed target \(f(k) = 1/k^{\tau}\) is motivated empirically (uniform sampling gives \(1/k\)) and theoretically via Theorem 4.2 (Dirichlet assumption). However, the theorem assumes \(\mathbf{p}\) and \(\mathbf{q}\) are drawn from Dirichlet distributions over all classes, not the subset in a prediction set. The connection to dynamic set sizes is not directly established; the theorem serves more as intuition than a rigorous derivation. The logistic-normal derivation in Appendix A similarly relies on strong assumptions. In practice, \(\tau\) is selected via grid search, which is empirical but works. Table 3 shows the power function is reasonable compared to exponential/logarithmic alternatives. Overall, the target is plausible but the theoretical justification is not tight.

*4.3 Conformal-Prediction-Calibration with Bi-Level Optimization*  
CPAC is a bi-level optimization that adjusts logits (via matrix \(\mathbf{W}\) and bias \(\mathbf{b}\)) to minimize calibration error while maintaining coverage. Algorithm 1 uses alternating updates. Several concerns arise:
1. **Convergence & stability**: No theoretical analysis of the bi-level optimization is provided; the assumption that \(\nu\) changes slowly is not justified.
2. **Generalization & overfitting**: Optimizing a full \(K \times K\) matrix on the calibration set risks overfitting, despite regularization. The paper does not analyze generalization to new test sets.
3. **Effect on coverage**: Since the calibration set is used for both optimizing parameters and computing the conformal threshold, the exchangeability condition for split CP must hold conditional on the optimization. This should be discussed (similar to temperature scaling in APS).
4. **Scalability**: For large \(K\) (e.g., 1000 classes), optimizing a million parameters may be computationally heavy, conflicting with CP’s low-computation appeal.
5. **Trade-offs**: The method may affect accuracy, coverage, or efficiency (PSS), as seen in the experiments.

**5. Experimental Results**  

*5.1 Settings*  
Datasets (CIFAR100, ImageNet, Topic Classification), models (ResNet, ViT, GPT-2), and perturbations are well-described. Using 20% of the test set for calibration is standard.

*5.2 Target Calibration Function*  
Table 3 supports the choice of the power function. No major issues.

*5.3 Factors Affecting CP Calibration*  
Interesting findings: pre-trained models often have worse calibration than randomly initialized ones; more data does not improve calibration; noise affects the distribution within the prediction set. These insights are valuable.

*5.4 Performance of CPAC*  
Tables 1, 2, 5 show CPAC often reduces uniform CP-ECE, but with mixed effects on other metrics. For example, in Table 1 (ViT-L/clean), CPAC improves uniform CP-ECE (11.02 → 6.74) but slightly decreases accuracy (81.98 → 80.17) and coverage (94.52 → 92.39). In Table 5 (ResNet101/clean), CPAC improves uniform CP-ECE (11.33 → 7.82) but increases standard CP-ECE (7.01 → 9.16) and PSS (16.04 → 29.57). The claim that CPAC “decreases the PSS” is not universally true. The fixed-coverage experiment (Table 4) shows CPAC sometimes increases PSS. The trade-offs between calibration, accuracy, coverage, and efficiency need explicit discussion. The overall effectiveness of CPAC is demonstrated, but the costs should be acknowledged.

**Writing & Clarity**  
The paper is generally well-written, though some sections could be clearer. The explanation of Theorem 4.2’s relevance to dynamic set sizes is lacking. The description of how \(\tau\) is selected during CPAC (grid search) should be more explicit. Minor formatting artifacts from the PDF parser do not hinder understanding.

**Limitations & Broader Impact**  
The paper notes that convergence/generalization of the bi-level optimization is only empirical—a valid limitation. Broader impact is not discussed; improved calibration could enhance reliability in safety-critical applications, but potential negative societal impacts (e.g., if calibration fails in certain subgroups) are not considered.

### Overall Assessment
The paper makes a novel and important contribution by formalizing and addressing the calibration of prediction set size in conformal prediction—a topic largely overlooked. The definition, metrics, target function, and CPAC method are thoughtfully designed. Experiments are extensive and reveal interesting properties of CP calibration.

However, significant concerns remain: the theoretical justification for the target function is weak; the bi-level optimization lacks theoretical grounding and may overfit; experimental results show trade-offs that are not fully acknowledged; and the method’s computational cost for large \(K\) is unexamined. With substantial revisions—strengthening the theory, analyzing optimization convergence, discussing trade-offs, and clarifying limitations—the paper could meet ICLR’s standards. In its current form, it is not yet ready for acceptance but has strong potential.

# Neutral Reviewer
## Balanced Review

### Summary
This paper investigates the calibration of prediction set sizes (PSS) in conformal prediction (CP) for classification, asking whether smaller sets reliably indicate higher accuracy. The authors define CP calibration via multinomial sampling, propose a power-law calibration target based on theoretical analysis, and introduce CP-aware calibration (CPAC), a bi-level optimization method to improve calibration. Experiments on image and text datasets with models like ResNet, ViT, and GPT-2 show standard CP is poorly calibrated and CPAC reduces calibration error while preserving coverage.

### Strengths
1. **Novel Problem Formulation:** The paper systematically formalizes and studies the calibration of prediction set sizes in conformal prediction, an underexplored but important aspect of uncertainty quantification. This addresses a clear gap, as CP's coverage guarantee does not ensure that set size aligns with per-instance reliability.
2. **Theoretical Grounding:** Theoretical analysis (Theorem 4.2, Appendix A) links expected accuracy to set size under Dirichlet/logistic-normal assumptions, motivating the power-law calibration target. This provides a principled foundation beyond heuristic choices.
3. **Comprehensive Empirical Evaluation:** Extensive experiments across three datasets (CIFAR100, ImageNet, topic classification), multiple model architectures (ResNet, ViT, GPT-2), and various perturbations (noise, blur, dropout, typos) consistently demonstrate weak calibration of standard CP and the effectiveness of CPAC. Tables and reliability diagrams provide strong evidence.
4. **Practical Algorithm:** CPAC is a practical post-hoc calibration method that integrates with existing CP frameworks via bi-level optimization. It maintains coverage guarantees while improving calibration, as shown in the results.
5. **Clarity and Structure:** The paper is well-organized, with clear definitions, methodology, and experimental sections. Figures effectively illustrate calibration issues and improvements.

### Weaknesses
1. **Limited Theoretical Guarantees for CPAC:** While the calibration target is theoretically motivated, the convergence and generalization of the bi-level optimization algorithm are not analyzed. The paper acknowledges this in the conclusion but offers no theoretical assurance.
2. **Incomplete Ablation and Sensitivity Analysis:** The paper lacks ablation studies on key components of CPAC (e.g., impact of regularization, optimization rounds, batch size) and sensitivity to hyperparameters (τ, sampling temperature t). This makes it hard to assess robustness.
3. **Narrow Baseline Comparison:** Comparisons are limited to standard Platt scaling (PS) and full-matrix Platt scaling (PS-Full). Other CP-specific calibration methods (e.g., temperature scaling variants from Xi et al. 2024 or Dabah & Tirer 2024) are mentioned but not directly compared.
4. **Ambiguity in Calibration Metrics:** The paper introduces two CP-ECE metrics (standard and uniform) but does not deeply discuss their practical implications. For instance, uniform CP-ECE might overweight rare set sizes, potentially misleading practitioners.
5. **Computational Overhead Unexplored:** CPAC involves bi-level optimization, which likely adds computational cost compared to standard CP. This overhead is not quantified (e.g., training time increase), which is important for practical deployment.
6. **Trade-offs Insufficiently Analyzed:** Results show CPAC sometimes increases PSS (Table 1) or slightly reduces coverage. The trade-off between calibration error, coverage, and set size efficiency could be analyzed more explicitly, perhaps via Pareto curves.

### Novelty & Significance
The paper is novel in systematically studying calibration of prediction set sizes in conformal prediction for classification. While confidence calibration is well-studied, this work shifts focus to set-based uncertainty. The proposed CPAC method is a significant contribution, improving calibration without breaking coverage guarantees. The work is timely given CP's growing use in safety-critical applications, and it meets ICLR's bar for novelty and potential impact.

### Suggestions for Improvement
1. **Theoretical Analysis of CPAC:** Provide convergence guarantees for the bi-level optimization or at least discuss conditions for stability. Analyze the effect of regularization on generalization.
2. **Ablation and Sensitivity Studies:** Include experiments ablating key components of CPAC (e.g., regularization λ, optimization rounds) and testing sensitivity to τ and t. Compare using a single temperature scalar versus full matrix W.
3. **Broader Baseline Comparison:** Compare CPAC with other CP-specific calibration methods (e.g., from Xi et al. 2024, Dabah & Tirer 2024) to better situate its contributions.
4. **Guidance on Metrics:** Discuss practical scenarios where standard vs. uniform CP-ECE is more appropriate. Provide guidance on selecting τ based on data/model characteristics.
5. **Computational Efficiency:** Report training/time overhead of CPAC versus standard CP. Discuss whether improved calibration justifies the cost.
6. **Trade-off Analysis:** Explicitly analyze the trade-offs between calibration error, coverage, and set size. Could CPAC be tuned for different balances? Consider including a Pareto analysis.
7. **Clarity on Sampling:** Justify the choice of sampling temperature t=3; show how calibration error varies with t. Discuss the implications of using multinomial sampling versus other strategies.
8. **Reproducibility:** Provide code or more implementation details (e.g., optimization algorithm, initialization, hyperparameter ranges) to ensure reproducibility.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation on the necessity of bi-level optimization versus a simpler joint optimization.** The paper proposes a complex bi-level optimization without showing that a simpler one-step or alternating method fails. Without this, the added complexity is not justified.
2. **Comparison to alternative calibration targets beyond the proposed power function.** Only the power function is seriously evaluated; the exponential and logarithmic alternatives are dismissed with limited data. A systematic comparison is needed to validate the choice.
3. **Experiments with other conformal score functions beyond APS.** The paper only uses APS (Adaptive Prediction Sets). The findings and CPAC’s effectiveness must be verified with other common scores (e.g., RAPS, regularized) to ensure generality.
4. **Evaluation under distribution shift or out-of-distribution settings.** The paper tests with synthetic noise but not with natural distribution shifts (e.g., ImageNet-C). Calibration is critical under shift, and the method’s robustness is unknown.

### Deeper Analysis Needed (top 3-5 only)
1. **Theoretical or empirical analysis of CPAC’s effect on coverage guarantee.** The bi-level optimization could distort the quantile estimation, potentially breaking the coverage guarantee. The paper should analyze how CPAC affects coverage theoretically or with held-out tests.
2. **Analysis of the optimization stability and hyperparameter sensitivity.** The method has several hyperparameters (τ, λ, t, learning rate) chosen by grid search. A sensitivity analysis is missing, making it unclear if CPAC is robust or requires extensive tuning.
3. **Investigation of why CPAC sometimes increases prediction set size.** Table 1 shows CPAC often reduces PSS, but sometimes it increases (e.g., under strong noise). The paper should analyze when and why this happens, as it impacts practical utility.
4. **Explanation of the relationship between the Dirichlet assumption and real predictive distributions.** Theorem 4.2 relies on a Dirichlet assumption, but no evidence is provided that real model probabilities match this. The paper should check the fit empirically.

### Visualizations & Case Studies
1. **Visual examples of prediction sets before and after CPAC for specific instances.** Show concrete images/text where CPAC changes the set size and whether that change better reflects correctness. This would demonstrate if the method works intuitively.
2. **Plot of the learned weight matrix W to interpret its effect.** CPAC learns a full matrix W; visualizing its structure (e.g., heatmap) could reveal if it performs meaningful re-ranking or is just noise.
3. **Reliability diagrams for both uniform and standard CP-ECE on the same plots.** The paper reports two metrics but shows diagrams for only one. Overlaying both would clarify the trade-offs and which metric CPAC actually improves.

### Obvious Next Steps
1. **Incorporate calibration error directly into the conformal score.** Instead of post-hoc bi-level optimization, design a conformity score that inherently encourages calibration, which could be more efficient and principled.
2. **Extend to regression tasks.** The paper focuses on classification, but the core idea of calibrating set size to accuracy could apply to regression (calibrating interval width to error). This is a natural extension.
3. **Provide an open-source implementation.** Given the complexity of bi-level optimization and the many hyperparameters, releasing code is essential for reproducibility and adoption.

# Final Consolidated Review
## Summary
This paper systematically investigates whether the prediction set size (PSS) from conformal prediction (CP) is a well-calibrated measure of uncertainty, i.e., if smaller sets reliably indicate higher accuracy. It defines CP calibration via multinomial sampling, proposes a power-law calibration target motivated by theoretical analysis, and introduces CPAC, a bi-level optimization algorithm to improve calibration. Extensive experiments on image and text datasets with models like ResNet, ViT, and GPT-2 demonstrate that standard CP is poorly calibrated and that CPAC can reduce calibration error.

## Strengths
- **Novel and Important Problem Formulation:** The paper is the first to systematically formalize and study the calibration of prediction set sizes in conformal prediction for classification. This addresses a significant gap, as CP's coverage guarantee does not ensure that set size aligns with per-instance reliability, which is critical for risk-aware decision-making.
- **Comprehensive Empirical Evidence:** The experimental evaluation is extensive, covering three datasets (CIFAR100, ImageNet, topic classification), multiple model architectures (ResNet, ViT, GPT-2), and various input perturbations (noise, blur, dropout, typos). The results consistently show weak calibration in standard CP and demonstrate the effectiveness of the proposed CPAC method in reducing calibration error (e.g., uniform CP-ECE in Table 1 for ViT-L drops from 11.02 to 6.74 on clean ImageNet).
- **Practical Algorithm with Preserved Coverage:** CPAC is a practical, post-hoc calibration method that integrates with the existing CP framework via bi-level optimization. The experiments confirm that it maintains the core coverage guarantee of CP while improving calibration, making it a deployable solution.

## Weaknesses
- **Theoretical Justification for Calibration Target is Loose:** The proposed power-law target \(f(k) = 1/k^\tau\) is motivated by Theorem 4.2 and Appendix A, which assume Dirichlet or logistic-normal distributions over all classes. The connection to the dynamic, instance-specific prediction set is not rigorously established, making the theoretical support more intuitive than conclusive.
- **Lack of Analysis for the Bi-Level Optimization:** The convergence, stability, and generalization of the CPAC optimization algorithm are only validated empirically. There is no theoretical analysis of the bi-level formulation (e.g., conditions for convergence, effect on the exchangeability assumption), which is important given the risk of overfitting when optimizing a full weight matrix on the calibration set.
- **Incomplete Experimental Analysis of Trade-offs and Baselines:** While CPAC improves calibration error, the results show trade-offs with other metrics (e.g., sometimes reduced accuracy or increased PSS, as seen in Table 1 and 5). These trade-offs are not deeply analyzed. Furthermore, comparisons are limited to standard Platt scaling; other CP-specific calibration methods (e.g., temperature scaling variants from cited works) are mentioned but not empirically compared, making it hard to gauge CPAC's relative advantage.

## Nice-to-Haves
- A sensitivity/ablation study on CPAC's hyperparameters (e.g., regularization strength \(\lambda\), optimization rounds, sampling temperature \(t\)) would help users understand its robustness.
- Visual examples of how CPAC changes prediction sets for specific instances could provide more intuitive insight into the method's behavior.
- A discussion on the practical implications of choosing between standard and uniform CP-ECE would guide practitioners in selecting the appropriate metric.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness about computational overhead being unexamined:** The paper's focus is on establishing the calibration concept and method; detailed runtime analysis, while useful, is not a core requirement for evaluating the contribution.
- **Weakness about missing distribution shift experiments:** The paper already evaluates under synthetic perturbations (noise, blur, typos). While testing on natural distribution shifts (e.g., ImageNet-C) is a valuable extension, it is outside the stated scope of establishing the calibration problem and a first solution.
- **Weakness about the need for visualizations of the weight matrix \(W\):** Interpreting the learned matrix is an interesting analysis but not necessary to validate the method's core efficacy.
- **Criticism that CPAC "decreases the PSS" is not universally true:** The paper acknowledges this in Section 5.4, noting that PSS can increase and linking it to the focus on low-PSS samples. This is a discussed trade-off, not a hidden flaw.
- **Criticism about scalability for large \(K\):** The method uses a regularized full matrix, and the paper demonstrates successful experiments on ImageNet (K=1000). While computational cost is a practical consideration, it does not invalidate the methodological contribution.

## Novel Insights
The paper provides the novel insight that the uncertainty conveyed by conformal prediction—through prediction set size—is often poorly calibrated, meaning smaller sets do not reliably correspond to higher accuracy. This misalignment is shown to be an independent issue from coverage guarantees and is systematically characterized across models and datasets. Furthermore, the work reveals that factors like pre-training and data quantity can negatively affect this calibration, highlighting a previously overlooked dimension of uncertainty quality in a widely used method.

## Suggestions
- Strengthen the theoretical motivation for the calibration target by more directly linking the Dirichlet/logistic-normal analysis to the setting of a *subset* of classes forming the prediction set.
- Include a discussion or simple empirical check on how the bi-level optimization might affect the exchangeability condition required for valid coverage in split conformal prediction.
- Expand the baseline comparisons in experiments to include at least one other recent CP-specific calibration method (e.g., from Xi et al. 2024 or Dabah & Tirer 2024) to better position CPAC's contribution.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 2.0, 2.0]
Average score: 3.5
Binary outcome: Reject
