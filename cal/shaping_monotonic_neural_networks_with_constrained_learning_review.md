=== CALIBRATION EXAMPLE 5 ===

# Final Consolidated Review
## Summary

This paper proposes a stochastic primal-dual learning framework to enforce partial monotonicity in neural networks with general architectures. The monotonicity constraint is reformulated as a chance-constrained optimization problem using a smooth inner approximation (Claim 1), and a Lagrangian-based primal-dual algorithm adaptively adjusts the penalty strength during training. A chance constraint parameter $\alpha$ provides an interface to trade off monotonicity satisfaction probability against prediction accuracy.

## Strengths

- **Architecture-agnostic constraint enforcement:** Unlike prior work that requires specialized architectures (Min-Max Net, DLN, SMNN, LMN), this framework imposes no architectural restrictions, allowing the use of standard networks with their full expressivity. This is a genuine and important advantage, as architecture-constrained methods explicitly limit the hypothesis space and can degrade performance (Section 1, Section 3.2).

- **Adaptive penalty via dual variables:** The dual variable $\mu$ automatically modulates the constraint penalty strength based on the degree of monotonicity violation (Eq. 9c, 11), eliminating the need for manual schedule-based regularization tuning that methods like Certified MNN require. The paper explicitly demonstrates that Certified MNN suffers training failures from excessive penalties while the proposed method does not (Section 4.1).

- **Strong empirical performance with compact models:** Tables 1 and 2 show the method achieves top or near-top accuracy across five datasets while using substantially fewer parameters than most baselines (e.g., 2069 vs. 23112 for Certified MNN on COMPAS). The frequency control experiment (Figure 2) demonstrates 25% improvement over SMNN and 5.3% over monotonic SNN in objective cost, with the method learning more aggressive and effective control actions.

## Weaknesses

### Major:

- **Theory-practice gap with auxiliary variable $\mathbf{t}$:** The theoretical formulation (Eq. 6, 9b) treats $\mathbf{t}$ as a learnable primal variable, while all experiments fix $\mathbf{t} = 1 \times 10^{-4}$ (Section 4.1, Appendix A.2). The paper briefly notes this practical choice ("one may also consider to fix the auxiliary variable $\mathbf{t}$ at a small positive constant vector to further ease the training"), but does not justify why fixing $\mathbf{t}$ does not degrade optimization or how sensitive results are to this choice. Since $\mathbf{t}$ controls the tightness of the smooth approximation in Claim 1, fixing it means the method cannot adaptively tighten the bound during training, which partially undermines the claimed "Strong Adaptability."

- **Core "flexibility" claim ($\alpha$ trade-off) is empirically unsubstantiated:** The paper's first listed contribution is "High Flexibility" to trade off monotonicity satisfaction probability and prediction performance via the chance constraint coefficient $\alpha$ (Section 1.1). Yet $\alpha = 0.1$ is used for all experiments with no ablation. Without showing how varying $\alpha$ (e.g., $\alpha = 0$, $0.01$, $0.1$, $0.5$) affects both accuracy and monotonicity violation rates, this central claim remains unverified. The reader cannot determine whether the chance constraint provides a meaningful control knob or whether performance collapses as $\alpha \to 0$ (strict monotonicity).

- **Sparse sampling undermines "whole input space" enforcement claim:** The method enforces monotonicity by sampling from $\text{Uni}(\mathbf{X})$ (Section 3.1), with $N = 128$ samples per batch step (Appendix A.2). In high-dimensional settings like Blog Feedback ($d = 276$), 128 uniform samples provide exponentially negligible coverage of the input hyperrectangle. The paper claims to "enforce the monotonicity requirement across the entire input domain $\mathbf{X}$" but the stochastic guarantee is only as good as the sampling density, which degrades catastrophically with dimension. No discussion of this fundamental limitation is provided, and no quantitative measurement of actual monotonicity violation rates (e.g., via dense grid evaluation or formal certification) is reported.

- **"Best 5 of 10" reporting introduces selection bias:** Section 4.1 states "We run the experiments ten times per dataset after finding the optimal hyperparameters and report the mean and standard deviation of the best five results." Selecting the top 50% of runs inflates reported performance and understates variance. While the paper notes this aligns with prior work (Runje & Shankaranarayana, 2023; Kim & Lee, 2024), it remains a methodological concern that makes it difficult to assess the true robustness of the method.

### Minor:

- **"Small extra computations" claim is unquantified:** The abstract states the method "needs only small extra computations," but no wall-clock time, FLOPs comparison, or per-epoch overhead measurement is provided. The method requires computing input-gradient derivatives at $N$ sampled points per step, which involves additional backward passes. Whether this overhead is truly "small" relative to baselines (especially those requiring expensive MILP/SMT certification) is not demonstrated.

- **Tightness of the Claim 1 approximation is unanalyzed:** Claim 1 provides a sufficient condition—the inner approximation $E[\mathbf{t} + (\mathbf{0} - \nabla_{\mathbf{z}_m} f_\theta)_+] \leq \alpha \mathbf{t}$ implies the chance constraint holds, but the converse is not true. The paper does not discuss how tight this approximation is or what fraction of the feasible set is lost, making it unclear how conservative the method is in practice.

### Trivial:

- The paper claims "no pre-processing such as tuning of the regularization" (Abstract, Section 1.1), but the method introduces new hyperparameters ($\alpha$, $\gamma_\mu$, the fixed $\mathbf{t}$ value, $N$). While the dual variable does adapt automatically, the claim is somewhat overstated since $\alpha$ and $\gamma_\mu$ still require selection.

## Nice-to-Haves

- Post-hoc formal certification of monotonicity (e.g., via MILP as in Liu et al., 2020) on the trained models, to quantitatively measure actual violation rates and validate the chance constraint approximation.
- Convergence or stability analysis for the primal-dual algorithm in the non-convex neural network setting.
- Ablation comparing uniform sampling from $\text{Uni}(\mathbf{X})$ against adversarial sampling that actively seeks monotonicity violations, to justify the uniform sampling strategy in high dimensions.
- Evaluation on modern deep architectures (e.g., ResNets) beyond simple MLPs, to substantiate the "general architectures" claim for deeper networks.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Missing related works / baselines:** Multiple reviews requested additional baselines or cited missing related work. Per the hard rules, I cannot confirm existence of uncited works and remove this category of criticism.

- **Parameter count asymmetry with LMN (37 params vs. 2069):** While the reviewer flagged this as an unfair comparison, LMN's small parameter count is a feature of its specialized architecture. The proposed method still achieves only marginally better accuracy (69.4% vs. 69.3% on COMPAS) despite vastly more parameters, which actually highlights LMN's parameter efficiency rather than unfairly favoring the proposed method. This is noted but not a weakness of the experimental design.

- **SMNN Blog Feedback RMSE parsing artifact:** The "0. ± 0.501" value in the parsed table is clearly a PDF extraction artifact. The original paper presumably contains the correct value.

- **Formatting and notation consistency nitpicks:** Minor notational issues (e.g., $\mathbf{t}$ optimized in theory but fixed in practice) are already captured substantively above; pure style complaints are removed per hard rules.

- **Reproducibility concerns about undisclosed hyperparameters:** Detailed hyperparameters are provided in Appendix A.2 (Table 4). Per hard rules, trivial reproducibility nitpicks are removed.

## Novel Insights

The primal-dual formulation reframes monotonicity enforcement not as a hard constraint or fixed penalty, but as a resource allocation problem where the dual variable $\mu$ acts as a dynamically priced "violation tax." This perspective reveals a fundamental tension in the method: the chance constraint framework provides probabilistic guarantees that are appropriate for preference-style monotonicity (e.g., loan approval fairness), but the paper also markets the method for safety-critical systems (e.g., power system control) where probabilistic violations may be unacceptable. The uniform sampling strategy is the weakest link connecting these two domains—while it suffices for low-dimensional supervised tasks, it cannot provide the guarantees needed for safety-critical applications without formal post-hoc certification. The method's greatest untapped potential may lie not in the $\alpha$ trade-off (which the experiments never explore) but in the adaptive dual mechanism itself, which could potentially be combined with adversarial violation-seeking samplers to create a more robust enforcement strategy.

## Suggestions

- Run a systematic ablation varying $\alpha \in \{0, 0.01, 0.05, 0.1, 0.2, 0.5\}$ on at least two datasets, reporting both prediction performance and the measured monotonicity violation rate on a held-out test set or dense grid. This directly validates the paper's primary claimed contribution.
- Report results over all 10 random seeds (not best 5 of 10), or at minimum include both statistics side-by-side, so readers can assess the method's true variance.
- Add a brief subsection discussing the curse of dimensionality for uniform sampling and the resulting gap between the theoretical "whole input space" guarantee and what $N=128$ samples can practically certify, along with practical guidance on scaling $N$ with dimensionality.

# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
