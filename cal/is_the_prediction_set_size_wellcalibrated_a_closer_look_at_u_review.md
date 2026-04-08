=== CALIBRATION EXAMPLE 32 ===

# Final Consolidated Review
## Summary

This paper investigates whether the prediction set size (PSS) produced by conformal prediction (CP) in classification tasks is well-calibrated with respect to prediction accuracy—a property distinct from coverage guarantees. The authors formalize "CP calibration" via a multinomial sampling strategy connecting PSS to expected accuracy, propose a power-function calibration target grounded in a Dirichlet distribution analysis (Theorem 4.2), and introduce CP-Aware Calibration (CPAC), a bi-level optimization algorithm that adjusts model logits as a pre-processing step before conformal quantile computation. Experiments across CIFAR-100, ImageNet-1k, and a topic classification dataset with models including ResNet, ViT, and GPT-2 reveal weak calibration in standard CP and demonstrate CPAC's effectiveness at reducing CP-ECE.

## Strengths

- **Novel and well-motivated problem formulation.** The distinction between PSS calibration and conditional coverage (Gibbs et al., 2025) is clearly articulated, and the paper convincingly demonstrates that smaller prediction sets do not always correspond to higher per-instance reliability—a finding with direct implications for risk-aware decision-making. This shifts focus from the well-studied coverage/efficiency axes to an underexplored calibration dimension specific to CP.

- **Theoretical grounding for the calibration target.** The derivation of $f(k) = 1/k^\tau$ from Dirichlet distribution assumptions (Theorem 4.2), supplemented by a logistic-normal extension (Theorem A.1), provides principled motivation for the target function rather than relying purely on heuristic curve fitting. This is a meaningful analytical contribution even if the assumptions are restrictive.

- **Broad empirical evaluation across architectures and domains.** The paper evaluates on three datasets with seven models spanning CNNs, Vision Transformers, and an LLM, under diverse perturbations (Gaussian noise, blur, dropout, typos). This breadth provides substantial evidence that the calibration problem is real and pervasive.

## Weaknesses

### Major:

- **Potential exchangeability violation from reuse of the calibration set.** Algorithm 1 optimizes $(W, b)$ on $D_{cal}$, then uses the *same* $D_{cal}$ to compute the conformal quantile $\nu$. Split conformal prediction requires that the calibration data be exchangeable with test data *and not used for any prior fitting step*. By optimizing model parameters on $D_{cal}$ before quantile computation, the exchangeability assumption that underpins CP's distribution-free coverage guarantee is compromised. The paper does not address this foundational concern—no nested split (e.g., $D_{cal\_opt}$ and $D_{cal\_quantile}$) is used, and no theoretical bound on potential coverage violation is derived. This is a serious gap for any paper proposing a modification to CP that must preserve coverage validity.

- **The coverage–calibration–efficiency trade-off is insufficiently discussed in the main text.** Appendix C reveals that when coverage is controlled at exactly 90%, CPAC *increases* PSS relative to the baseline (e.g., ViT-B Clean: PSS goes from 6.20 to 7.81 in Table 4). Yet Section 5.4 claims CPAC "maintains the accuracy and decreases the PSS." The tables with uncontrolled coverage (Tables 1–2) show mixed results on PSS—some settings show decreases, others show increases or near-parity (e.g., ViT-L Norm-0.8: 239.18 → 239.63). The paper needs an honest, prominent discussion of when PSS calibration comes at the cost of set efficiency, as this trade-off directly affects practical deployment.

- **Limited baseline comparisons.** Only Platt Scaling (PS) and its full-matrix variant (PS-Full) are compared. The paper does not compare against efficiency-focused CP methods such as RAPS (Angelopoulos et al., 2021) or other recent regularization approaches (Ghosh et al., 2023a;b; Xi et al., 2024) that also modify the score function or temperature to influence PSS. Without such comparisons, it is unclear whether CPAC's calibration gains could be achieved more simply by tuning existing CP efficiency methods, or whether CPAC provides genuinely orthogonal benefits.

### Minor:

- **Accuracy degradation not adequately acknowledged.** CPAC sometimes reduces top-1 accuracy (e.g., ViT-L Clean: 81.98% → 80.17%; ViT-B Clean at fixed coverage: 80.35% → 77.68%). The paper states CPAC "maintains the accuracy," but drops of 1–3 percentage points are non-trivial, especially in high-stakes settings. The text should quantify and discuss these trade-offs.

- **Convergence and computational cost of bi-level optimization.** The paper itself acknowledges the lack of theoretical convergence analysis. Additionally, no wall-clock time or computational cost comparison is reported. CPAC optimizes a full weight matrix $W \in \mathbb{R}^{K \times K}$ over $M_{opt}$ rounds with batch processing—for ImageNet ($K=1000$), this is non-trivial. Practitioners need cost-benefit data.

- **$\tau$ selection protocol lacks clarity and may involve data leakage.** Section 5.2 states that "During the test stage, we use grid search to find the optimal $\tau$." If $\tau$ is selected based on test-set performance metrics, this constitutes data leakage and inflates results. The paper needs to clarify whether $\tau$ is chosen on the calibration set or the test set, and provide guidance for principled selection without test-set access.

- **The Dirichlet assumption is restrictive.** Theorem 4.2 assumes the predictive distribution and sampling distribution follow Dirichlet distributions with shared mass vector $\mathbf{a}$. While the logistic-normal extension (Appendix A) provides partial relief, neither assumption has been empirically validated on neural network outputs. The paper would benefit from an empirical assessment of how well these distributional approximations fit real softmax outputs.

- **GPT-2 improvements are modest.** On the topic classification task (Table 2), CPAC's Uniform CP-ECE improvements over PS-Full are often within one standard deviation (e.g., Clean: 6.01±0.53 vs. 5.50±0.46). No statistical significance testing is provided, making it difficult to assess whether the gains are meaningful for NLP tasks.

### Trivial:

- The paper's claim of being "the first attempt to systematically investigate the calibration of CP on classification tasks" slightly overstates novelty given van der Laan & Alaa (2024) and Lu et al. (2023), though the distinction (classification vs. regression, joint vs. separate calibration) is defensible.

## Nice-to-Haves

- **Nested calibration split.** Restructure CPAC to use separate data for optimization ($D_{opt}$) and quantile computation ($D_{cal}$) to restore theoretical coverage guarantees—this would significantly strengthen the contribution.

- **Ablation studies** on key hyperparameters ($\lambda$, $t$, $M_{opt}$, matrix vs. scalar temperature) and on the calibration set size sensitivity, to demonstrate robustness and guide practitioners.

- **Comparison with RAPS or other efficiency-regularized CP methods** to establish that CPAC's calibration benefits are not achievable via simpler set-size penalties.

- **Downstream utility demonstration**, e.g., selective classification or human-in-the-loop deferral, to show that lower CP-ECE translates to better real-world decisions.

- **Convergence plots** across seeds showing the bi-level optimization is stable and not overfitting $D_{cal}$.

## Removed Points

These points were flagged for removal—treat them with caution as they may reflect reviewer knowledge gaps or misunderstandings:

- **"GPT-2 fine-tuning details not fully specified, affecting reproducibility."** Removed per the rule against nitpicking about trivial reproducibility details. The paper provides sufficient detail for a 137M GPT-2 model on a well-defined task.

- **"Figure 1 content is garbled / point prediction coverage claim is well-known."** The garbling is a parser artifact (removed per formatting rule). The claim about point prediction lacking coverage guarantees is a standard framing device, not a weakness.

- **"Demand comparison with Mossina et al. (2024) for conformal segmentation."** Removed—this is a different task domain (segmentation vs. classification) and constitutes requesting comparison outside the paper's stated scope.

- **"Why can't van der Laan & Alaa's regression method be adapted to classification?"** Removed—the paper adequately explains that their method produces calibration multi-prediction and prediction interval separately, while this paper aims to calibrate the prediction set directly, which is a fundamental difference in objective.

- **"Demand for user study / human evaluation."** Removed per the soft rule: requesting user studies for a purely algorithmic contribution is outside standard practice.

- **"The claim that point prediction 'cannot guarantee coverage' is well-known and doesn't need illustration."** Removed—this is a framing observation, not a substantive weakness.

## Novel Insights

The most striking insight emerging from the three reviews is the **exchangeability tension at the core of CPAC**: the algorithm uses the same calibration set for both logit-space optimization and quantile computation, which may silently violate the very distribution-free guarantee that makes CP attractive. This is not merely a theoretical nicety—it is the foundation of CP's appeal, and modifying logits on the calibration data before computing conformity scores could introduce a systematic dependency that inflates coverage on the calibration set while degrading it on held-out data. The paper's Appendix C results, where CPAC under fixed coverage produces larger sets, may be a symptom of this issue. Resolving this through a nested split would both preserve guarantees and potentially explain the accuracy/PSS trade-offs observed empirically. A second insight is that **the Uniform CP-ECE metric, while motivated by fairness, may overemphasize rare PSS bins**, making the calibration improvement appear larger than it is for typical inputs—correlation analysis between Standard and Uniform CP-ECE could clarify whether the two metrics diverge in practice.

## Suggestions

- Implement a **nested calibration protocol**: split $D_{cal}$ into $D_{opt}$ (for learning $W, b$) and $D_{quant}$ (for computing $\hat{\nu}_{cal}$), and verify that coverage guarantees hold. This is the single most important change for theoretical soundness.

- Move the **coverage-controlled comparison** (currently in Appendix C) to the main text, and honestly discuss the PSS inflation trade-off. A Pareto-style plot of CP-ECE vs. PSS at fixed coverage would make this concrete.

- Add **confidence intervals or significance tests** (e.g., bootstrap) to CP-ECE comparisons, especially where improvements over baselines are within one standard deviation.

- Report **wall-clock time** for CPAC vs. standard APS on each dataset, and include an **ablation** comparing full-matrix $W$ optimization against scalar temperature scaling with the CP-specific loss—this would clarify whether the added complexity is necessary.

- Clarify the **$\tau$ selection protocol**: specify whether $\tau$ is chosen on $D_{cal}$ or $D_{te}$, and ideally propose a validation-based selection procedure that avoids test-set access.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 2.0, 2.0]
Average score: 3.5
Binary outcome: Reject
