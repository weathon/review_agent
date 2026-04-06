=== CALIBRATION EXAMPLE 43 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title clearly states the core research question. The abstract effectively summarizes the problem (unknown alignment of CP's uncertainty with correctness), the proposed definition, the empirical finding of weak calibration, and the proposed CPAC method. Claims are supported by the paper's content.

**Introduction & Motivation:** The introduction successfully motivates the problem by contrasting CP's coverage guarantee with the under-explored issue of whether its set size (the conveyed uncertainty) is calibrated to per-instance reliability. It clearly differentiates the work from point-prediction calibration and conditional coverage. The three stated challenges and contributions are precise and map well to the paper's structure.

**Method / Approach:**
*   **Definition 4.1 & Connection to Accuracy:** The use of multinomial sampling (Eq. 2, 4) to connect PSS to a measurable accuracy is creative and sensible for defining calibration. However, a significant conceptual issue arises: **the sampling procedure and the definition of `Acct(x)` are decoupled from the actual decision-making process that would use a CP set.** The metric `Acct(x)` (Eq. 7) is the expected accuracy *if* one were to sample a label from the set according to distribution `q(t)`. But in practice, a user presented with a set of size k must make a decision—they might guess uniformly, use the top-1 label, or use some other strategy. The paper's calibration definition and subsequent optimization (CPAC) calibrate towards the accuracy of this *specific, arbitrary sampling strategy*. The practical utility of calibrating to this particular expected accuracy, rather than to the accuracy of a more natural decision rule (e.g., predicting the class with the highest probability within the set), is not sufficiently justified. This weakens the claim that CPAC makes CP "more reliable" for risk-aware decisions.
*   **Calibration Target Function (Sec. 4.2):** The power-law target \( f(k) = 1/k^\tau \) is motivated empirically (Fig. 4) and theoretically (Theorem 4.2). However, **the theoretical justification has critical limitations.** Theorem 4.2 assumes `p` and `q` are drawn from *independent* Dirichlet distributions with the same underlying mean vector `a`. This independence assumption is problematic: in the proposed framework, `q(t)` is a deterministic function of the renormalized probabilities `p̃` (Eq. 2). They are not independent draws. The theorem, therefore, does not directly justify the target for the proposed sampling scheme. The logistic-normal analysis in Appendix A faces a similar issue of independence. The empirical fit in Fig. 4 is the stronger justification, but the theoretical motivation needs revision or clearer presentation as a heuristic analogy rather than a derivation.
*   **CPAC Algorithm (Sec. 4.3):** The bi-level optimization formulation is a natural approach to the problem. The practical implementation (Alg. 1) seems clear. Major concerns are:
    1.  **Justification for Full Matrix `W`:** The claim that a single temperature scalar "does not affect the calibration significantly" is stated without supporting evidence (e.g., an ablation or citation). Given that Platt scaling (a single parameter) is standard and effective for confidence calibration, this strong claim needs validation.
    2.  **Optimization Stability and Generalization:** The algorithm alternates between finding the quantile `ν` and taking gradient steps. The assumption that `ν` does not change drastically is stated but not justified. There is no analysis of convergence or the risk of overfitting to the calibration set, especially with a full matrix `W`. The regularization term is a good inclusion but its efficacy is only shown empirically.
    3.  **Loss for PSS=1:** The switch to cross-entropy loss for singleton sets is mentioned but not explained. Why is this necessary? Does it create an inconsistency in the objective?
    4.  **Focus on Low-PSS Samples:** The choice to optimize only samples with PSS below a threshold is pragmatic but introduces a bias. The algorithm explicitly optimizes calibration for "easier" samples (smaller sets) at the potential expense of the high-PSS region. This is a significant limitation that should be highlighted and discussed.

**Experiments & Results:**
*   **Scope and Baselines:** The empirical study is extensive, covering multiple datasets, models (ResNet, ViT, GPT-2), and perturbation types. The comparison between standard Platt Scaling (PS) and the proposed CPAC is appropriate.
*   **Evaluation of Calibration:** The introduction of Uniform CP-ECE is well-motivated (avoiding discrimination against minority PSS groups). Results in Tables 1, 2, and 5 show CPAC consistently reduces Uniform CP-ECE compared to PS, which is the core claim. The fact that Standard CP-ECE sometimes increases is noted and acceptable given the focus on uniform weighting.
*   **Key Omission - Coverage-Preserving Comparison:** A critical flaw in the presentation of results (Tables 1, 2, 5) is that **coverage is not fixed across methods.** The `Cov.` column shows PS and CPAC achieve different empirical coverage (often CPAC is ~1-2% lower). Since CP provides a coverage guarantee, a fair comparison of efficiency (PSS) and calibration must be done *while ensuring the same empirical coverage*. The experiment in Appendix C/Table 4 fixes coverage at 90% and shows CPAC often leads to *larger* prediction sets than PS. This contradicts the main tables' narrative that CPAC improves calibration *and* reduces set size. This discrepancy must be centrally addressed and explained. The increase in set size for fixed coverage is a major practical downside of CPAC.
*   **Ablations Missing:** Crucial ablations are absent: 1) Comparison to using a simple temperature scalar (`t` in PS) instead of the full matrix `W`. 2) An evaluation of how the choice of sampling temperature `t` (used in CPAC optimization) affects the final calibration for different decision rules (e.g., if a user takes the top-1 label, does calibrating with `t=3` help?).
*   **Statistical Reporting:** Reporting standard deviations over seeds is good practice.

**Writing & Clarity:** The paper is generally well-written. Figures are informative. Some sections are dense (e.g., the theoretical analysis) but understandable. The flow from problem definition to method to experiments is logical.

**Limitations & Broader Impact:** The conclusion briefly mentions the lack of theoretical analysis for the bi-level optimization as a weakness. However, the review identifies several more fundamental limitations that are not acknowledged:
1.  The calibration definition is tied to a specific, potentially non-standard decision rule (multinomial sampling).
2.  The CPAC method, when evaluated under a fixed coverage guarantee, appears to trade off better uniform calibration for larger set sizes (reduced efficiency).
3.  The algorithm's focus on low-PSS samples and its stability/generalization are not discussed as limitations.
4.  The societal impact section is missing. While the topic is methodological, a brief discussion on the implications of better/worse uncertainty calibration for high-stakes applications would be appropriate for ICLR.

### Overall Assessment
This paper addresses a novel and important question about the trustworthiness of conformal prediction's uncertainty representation. The core idea—defining and improving the calibration of prediction set size—is meaningful and timely. The empirical study is substantial and reveals interesting phenomena (e.g., pre-training can hurt PSS calibration). However, the work is significantly marred by **a disconnect between the proposed calibration metric and practical utility**, and by **a critical oversight in the experimental evaluation regarding fixed coverage**. The theoretical motivation for the target function is also flawed. While the CPAC algorithm demonstrates an ability to reduce the proposed uniform calibration error, its practical value is undermined by the apparent cost in set size efficiency when coverage is properly controlled. For ICLR, these issues are substantial and would require major revisions, including a more compelling justification for the calibration metric, a thorough re-evaluation of results under fixed coverage, and a discussion of the observed trade-offs. The contribution is promising but not yet complete.

# Neutral Reviewer
## Balanced Review

### Summary
This paper investigates whether the prediction set size (PSS) in conformal prediction (CP) is well-calibrated with respect to predictive accuracy—i.e., whether smaller sets reliably indicate higher per-instance correctness. It defines a notion of CP calibration, proposes a theoretical target function linking PSS to expected accuracy, and introduces CP-aware calibration (CPAC), a bi-level optimization method to improve calibration. Extensive experiments on image and text classification tasks demonstrate that standard CP is poorly calibrated and that CPAC reduces calibration error while maintaining coverage.

### Strengths
1. **Identifies and formalizes a novel, important problem.** The paper convincingly argues that while CP provides marginal coverage guarantees, the uncertainty conveyed by PSS is not necessarily calibrated with accuracy, which is crucial for risk-aware decision-making. This is an underexplored angle in the CP literature (Sec. 1, 2).
2. **Comprehensive empirical analysis.** The study systematically evaluates CP calibration across multiple factors (pre-training, data size, noise) using diverse models (ResNet, ViT, GPT-2) and datasets (CIFAR100, ImageNet, topic classification). The results consistently show weak calibration, strengthening the paper's core motivation (Sec. 5.3, Figs. 2-5).
3. **Effective proposed method (CPAC).** The CPAC algorithm, formulated as a bi-level optimization, demonstrably reduces both uniform and standard CP-ECE across most experimental settings without severely compromising accuracy or coverage (Sec. 5.4, Tabs. 1, 2, 5). The method is presented as a practical pre-processing step.

### Weaknesses
1. **Limited theoretical grounding for the optimization.** The paper acknowledges that "the convergence and generalization of the bi-level optimization problem are only validated empirically but not analyzed in theory" (Sec. 6). This is a significant gap, as the stability and guarantees of CPAC are not formally established.
2. **Trade-off between calibration and efficiency is underexplored.** While CPAC improves calibration, it sometimes leads to an increase in the average prediction set size (PSS) compared to baselines (e.g., Tab. 1, ViT-L, Clean setting: PSS increases from ~13.7 to ~18.5). Since set size efficiency is a key metric in CP, this trade-off deserves more critical discussion. The coverage-fixed experiment in Appendix C shows CPAC can inflate PSS significantly (Tab. 4).
3. **Theoretical assumptions may be restrictive.** Theorem 4.2 and the target function derivation rely on the assumption that the predictive probabilities within a set follow a Dirichlet distribution with a specific structure. While the authors note this is an illustrative instantiation (Remark after Thm. 4.2), the practical validity of this assumption across diverse models and data perturbations is not thoroughly validated.

### Novelty & Significance
**Novelty:** This is the first work to systematically define and study the calibration of prediction set size in conformal prediction for classification tasks. It clearly distinguishes this problem from conditional coverage and traditional confidence calibration. The proposed calibration target function and the CPAC algorithm are novel contributions.
**Significance:** The work addresses a fundamental gap in uncertainty quantification with CP. If CP's set sizes are poorly calibrated, users might misinterpret the uncertainty. The paper raises important awareness and provides an initial solution. Given CP's popularity in safety-critical applications, this research direction is highly significant.

### Suggestions for Improvement
1. **Strengthen the theoretical analysis.** Provide a convergence analysis for the bi-level optimization (even under simplifying assumptions) or discuss the conditions under which the algorithm is stable. Additionally, empirically validate the Dirichlet/logistic-normal assumptions (e.g., by testing goodness-of-fit on the computed probabilities).
2. **Deeper analysis of the calibration-efficiency trade-off.** Include a dedicated discussion section on this trade-off. Could CPAC be modified with a regularization term to control PSS inflation? Analyze whether the increase in PSS is concentrated in high-uncertainty samples and its practical implications.
3. **Compare against more relevant baselines.** The main comparison is with standard Platt Scaling (PS). It would be valuable to compare against other post-hoc calibration methods adapted for CP (beyond full-matrix Platt scaling) and recent CP variants that might implicitly affect calibration, providing a clearer view of the state-of-the-art.
4. **Improve the presentation of the calibration target.** Section 4.2 could be clearer. The transition from the uniform sampling case (f(k)=1/k) to the general power-law target (f(k)=1/k^τ) is motivated empirically but could benefit from a more intuitive bridge. Explicitly stating that τ acts as a "discount factor" for non-uniform distributions would help.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation study on the bi-level optimization components and hyperparameters (e.g., regularization λ, temperature t).** Without this, it is unclear which parts of CPAC are necessary and whether the gains are robust to hyperparameter choices, undermining its reliability.
2. **Direct comparison to alternative CP calibration baselines beyond simple Platt scaling.** The paper lacks comparisons to recent methods like conformalizing Venn-Abers (for classification) or random-set approaches, making its claimed novelty and effectiveness unconvincing.
3. **Experiment verifying coverage guarantee preservation after CPAC.** The method alters logits before quantile computation; a statistical test showing coverage still holds at the promised (1−α) level on new test data is essential to trust its conformal validity.
4. **Evaluation on a wider range of CP algorithms (e.g., RAPS, HPS) and model architectures.** The paper only uses APS; showing generalizability to other popular CP variants is needed to support the core claim of improving CP calibration broadly.

### Deeper Analysis Needed (top 3-5 only)
1. **Theoretical or empirical analysis of the bi-level optimization’s convergence and generalization.** The paper admits this is only empirically validated; without analysis, it is unclear if CPAC reliably converges or overfits the calibration set, jeopardizing its practical use.
2. **Investigation into why pre-trained models exhibit worse CP calibration than randomly initialized ones.** The observation is noted but not explained; understanding the cause (e.g., overconfidence, distribution shift) is critical for diagnosing and addressing the issue.
3. **Analysis of the trade-off between calibration error (CP-ECE) and prediction set size (efficiency).** The paper reports PSS changes but does not systematically analyze whether calibration gains come at the cost of larger sets, which is central to CP’s utility.

### Visualizations & Case Studies
1. **Per-instance case studies showing where CPAC succeeds/fails.** Visualizing specific test samples where CPAC significantly improves or worsens calibration (e.g., showing prediction sets and true class) would reveal whether the method works as intended or has systematic failure modes.
2. **Reliability diagrams for all compared methods (PS, PS-Full, CPAC) across all datasets in the main paper or appendix.** Currently only selected diagrams are shown; full visual evidence is needed to assess calibration claims comprehensively.
3. **Visualization of how CPAC changes the predictive distributions (e.g., entropy, confidence histograms) before and after calibration.** This would clarify whether CPAC meaningfully reshapes uncertainty or merely shifts scores arbitrarily.

### Obvious Next Steps
1. **Include a simpler baseline that directly minimizes the proposed calibration error via post-hoc scaling (without bi-level optimization).** This would test whether the complex bi-level setup is necessary or if a simpler approach could achieve similar gains.
2. **Evaluate on more challenging distribution shifts (e.g., ImageNet-C, adversarial perturbations) and larger models (e.g., LLMs).** The current perturbations are relatively mild; stronger out-of-distribution tests are needed to claim robustness.
3. **Provide code and reproducible scripts for all experiments.** ICLR emphasizes reproducibility; without public code, the results and implementation details of CPAC cannot be verified or built upon.

# Final Consolidated Review
## Summary
This paper formalizes and addresses the calibration of prediction set size (PSS) in conformal prediction (CP). It defines CP calibration by linking PSS to expected accuracy via multinomial sampling, proposes a power-law target function motivated by empirical and theoretical analysis, and introduces CP-aware calibration (CPAC), a bi-level optimization method that improves calibration across diverse image and text classification tasks.

## Strengths
- **Identifies and formalizes a novel, important problem.** The paper compellingly argues that while CP guarantees coverage, the uncertainty conveyed by prediction set size is not necessarily calibrated to per-instance reliability—a critical gap for risk-aware decision-making that has been largely overlooked in the literature.
- **Extensive and systematic empirical analysis.** The study thoroughly evaluates CP calibration across multiple factors (pre-training, data size, input perturbations) using a variety of models (ResNet, ViT, GPT-2) and datasets (CIFAR100, ImageNet, topic classification), consistently demonstrating weak calibration and thereby solidly motivating the work.
- **Effective proposed method with demonstrated utility.** The CPAC algorithm, formulated as a practical pre-processing bi-level optimization, reliably reduces the proposed uniform calibration error (CP-ECE) across most experimental settings without severely compromising accuracy or marginal coverage, providing a concrete tool for improving CP reliability.

## Weaknesses
- **Underexplored trade-off between calibration and efficiency under fixed coverage.** When empirical coverage is controlled to be identical (as shown in Appendix C, Table 4), CPAC often leads to larger prediction sets compared to standard Platt scaling. Since set size efficiency is a core metric in CP, this calibration-efficiency trade-off is a significant practical limitation that is not sufficiently analyzed or discussed in the main narrative.
- **Theoretical justification for the calibration target has simplifying assumptions.** Theorem 4.2 derives the power-law target under the assumption that the sampling probability **q** and the correctness probability **p̃** are independent draws from Dirichlet distributions. While the authors note this is an illustrative instantiation, the independence assumption does not strictly hold in the proposed sampling scheme (**q** is a deterministic function of **p̃**). The empirical fit remains the primary justification, but the theoretical motivation could be clearer.

## Nice-to-Haves
- An ablation study comparing the full-matrix transformation in CPAC to a simple temperature scaling baseline, to better justify the added complexity.
- Analysis of the bi-level optimization's convergence properties or sensitivity to hyperparameters (e.g., regularization λ, sampling temperature t), beyond the empirical validation provided.

## Novel Insights
The paper provides the first systematic investigation into whether the prediction set size from conformal prediction is calibrated to predictive accuracy. It introduces a formal definition and metric for this calibration, empirically reveals that standard CP is often poorly calibrated (with notable findings such as pre-training sometimes worsening calibration), and derives a power-law target function that well-models the relationship between set size and accuracy across sampling strategies. The proposed CPAC method demonstrates that this calibration can be improved via a tailored optimization, establishing a new direction for enhancing the interpretability and reliability of CP uncertainty.

## Suggestions
- Include a dedicated discussion section analyzing the observed trade-off between improved calibration and increased prediction set size when coverage is fixed, with potential suggestions for controlling this trade-off (e.g., via a weighted objective).
- Clarify the theoretical motivation in Section 4.2 by more explicitly framing Theorem 4.2 and the logistic-normal analysis as illustrative instantiations of the target behavior, rather than strict derivations, to avoid potential misinterpretation.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 2.0, 2.0]
Average score: 3.5
Binary outcome: Reject
