## Summary

This paper introduces a contrastive meta-learning framework for dynamical system forecasting. It proposes an Element-wise Square Ratio Loss (ESRL) with a covariance regularizer to learn system-specific embeddings from trajectory data without labeled physical coefficients, a Local Linear Least Square (LLLS) feature extractor for vector-based systems, and a Spatially Adaptive LinEar Modulation (SALEM) layer for grid-based PDE systems. The framework enables embedding-augmented forecasting of unseen systems via a short initial trajectory segment, demonstrating consistent MSE improvements over standard and meta-learning baselines across ODE and PDE benchmarks.

---

## Strengths

- **Unsupervised coefficient discovery without any label supervision.** The paper's core contribution — learning system-specific embeddings purely from trajectory observations, without labeled physical parameters — addresses a genuine and underexplored gap. Prior meta-learning works (LEADS, CAMEL, DyAd) all require either labeled coefficients or few-shot fine-tuning; this work is the first to demonstrate coefficient-agnostic, adaptation-free embedding for dynamical systems.

- **Well-motivated ESRL design with supporting ablations.** The element-wise ratio loss is specifically designed to prevent dimensional collapse in embedding dimensions (both constant-dimension collapse and linear-correlation collapse), a failure mode not addressed by Info-NCE or Triplet loss in multi-system settings. Table 4 provides clear evidence: removing the covariance regularizer degrades performance substantially (e.g., LV-4D: 10.7e-2 vs. 8.31e-2), and replacing ESRL with Info-NCE or Triplet loss produces severe degradation on harder tasks (22.4e-2 and 12.2e-2 vs. 8.31e-2 on LV-4D).

- **SALEM outperforms DyAN despite using no labeled information.** On incompressible flow (Table 2), SALEM achieves 9.06e-2 / 3.12e-2 on buoyancy/supply-rate tasks, beating DyAN (9.51e-2 / 3.86e-2) which is explicitly given pre-calculated vorticity as privileged information. That an unsupervised embedding method surpasses a supervised-coefficient method is a strong empirical result.

- **Qualitative embedding interpretability in the spring-mass system.** Figure 3 shows that the learned 2D embeddings for the dual spring-mass system form a rotated-and-scaled version of the true coefficient grid, without any supervision — an emergent property that is physically meaningful and consistent with the fact that LLLS directly extracts the linear coefficient matrix.

---

## Weaknesses

### Fatal
None.

### Major

- **Absence of comparison baselines for vector-based systems.** Table 1 compares only against "Standard training" (a plain MLP). The paper justifies this by noting all prior works (LEADS, CAMEL, DyAd, Kirchmeyer et al.) require labeled coefficients or fine-tuning. While the strict zero-label, zero-adaptation setup is genuinely different, the paper should at minimum adapt *one* prior method (e.g., by replacing its supervised coefficient learning with any unsupervised embedding) to demonstrate that the performance gap is not primarily due to the proposed loss or architecture, rather than simply augmenting the baseline. As written, reviewers cannot distinguish whether the gain in Table 1 comes from the meta-learning formulation or from simply providing embedding information.

- **All experiments are exclusively on synthetic data.** Spring-mass, Lotka-Volterra, and PhiFlow-generated PDEs are all noise-free simulations. The paper claims broad applicability and "a new benchmark in the field," but the absence of any real-world physical dataset — with noise, partial observability, or irregular sampling — makes it impossible to assess generalization beyond idealized conditions. This is a significant gap for an ICLR submission in scientific ML.

- **"Zero-shot" is a misnomer that inflates the novelty claim.** The paper repeatedly uses "zero-shot meta-learning" to describe the framework, but Section 3.2 explicitly states: "given a new system, we would like to leverage a short observed trajectory to infer its system embedding." Section 4.3 further confirms: "the initial segment is utilized to calculate the trajectory embedding." This is amortized inference (or unsupervised online embedding), not zero-shot by any standard definition. Zero-shot implies no examples from the new system at inference time. This mislabeling risks misleading readers about what the method actually does.

### Minor

- **No ablation for grid-based systems.** Table 4 ablates only vector-based systems. For PDE systems (Tables 2–3), there is no controlled experiment decomposing the contribution of (a) the contrastive embedding vs. (b) the SALEM architecture. A simple variant of SALEM with a random or constant embedding would isolate the effect of embedding quality on PDE forecasting.

- **Unexplained LLLS inconsistency in LV-4D (Table 4).** The row "ESR + λ=0.5 + no local feature extractor" achieves **7.73e-2** on LV-4D, which is *better* than the full model with LLLS (**8.31e-2**). The paper states in Section 6.1 that "performance is slightly worse" without LLLS, which directly contradicts Table 4 for this case. This unexplained result raises doubts about whether LLLS is consistently beneficial and should be addressed.

- **DyAN catastrophic failure on Gray-Scott is unexplained.** DyAN produces 31.1e-3 on Gray-Scott (feed rate), which is ~9× worse than standard ResNet (3.49e-3), despite DyAN using privileged labeled coefficients. This anomaly is not discussed anywhere in the paper. It undermines DyAN as a credible reference point in Table 3 and deserves an explanatory footnote at minimum.

- **Interpretability claim overstated for nonlinear systems.** The paper claims "explicit physical significance" for its embeddings as a notable byproduct, but in the Lotka-Volterra (2D) case (Figure 3), the paper itself describes only "a roughly rotated shape" that "loosely correlates" with the true coefficients. No quantitative measure (R², mutual information, or disentanglement score) of embedding–coefficient alignment is provided. The spring-mass result is strong, but extending the "interpretability" claim to cover the nonlinear case without quantification is unjustified.

### Tiny

- **Covariance regularizer scaling (Eqs. 2–3).** The features z̃ are normalized to zero mean and unit variance across the batch, so the diagonal of C(Z̃) = Z̃ᵀZ̃ equals N (batch size), not 1. The off-diagonal elements of C² therefore scale with N², making λ effectively batch-size-dependent. The paper fixes λ=0.5 "across all standard experiments," implying a fixed batch size — but this should be noted for reproducibility.

- **Sensitivity of segment length r not analyzed.** The LLLS extractor introduces hyperparameter r (segment length) but no ablation or analysis is provided. For chaotic or fast-varying systems, local linearity breaks down at short timescales, and the choice of r directly affects the quality of the A_k matrices fed into the RNN.

- **SALEM spatial coordinate specification is ambiguous.** Section 4.4 describes "an arbitrary spatial coordinate system for the domain (x-y)" without specifying the coordinate range (normalized [0,1]? pixel indices?). This affects reproducibility.

---

## Nice-to-Haves

- **O(D²) complexity of LLLS should be discussed in the methods section**, not deferred to the conclusion. Practitioners working with high-dimensional state spaces need this information when deciding whether to use the method.

- **Quantitative embedding–coefficient alignment metrics.** Adding R² correlation or mutual information between learned embeddings and true physical coefficients across all systems would turn Figure 3 from a qualitative observation into a solid, citable result supporting the interpretability claim.

- **Lookback window length s sensitivity.** Equation (6) uses a sliding window of length s for embedding inference, but no analysis of how performance varies with s is provided. A brief study would help practitioners choose s.

- **Embedding utility beyond forecasting.** If the embeddings genuinely capture physical coefficients, demonstrating their use for system identification or anomaly detection (even briefly) would significantly strengthen the representation learning contribution.

- **Convergence / training stability analysis.** FiLM and DyAN both produce NaN in several experimental conditions. Reporting why SALEM is more stable (gradient behavior, architecture choices) would help practitioners and strengthen the paper's practical contributions.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"Equation (2) is not a correlation matrix" (harsh critic).** The features z̃ are explicitly z-score normalized (mean 0, unit variance) before computing C = Z̃ᵀZ̃, consistent with VICReg (Bardes et al., 2021), which inspired this formulation. The regularizer correctly targets correlation collapse. The normalization inconsistency is a mild scale issue absorbed by λ, not a "non-trivial technical error."

- **"DyAN comparison is unfair" (harsh critic).** The paper *explicitly discloses* that DyAN uses labeled pre-calculated vorticity while the proposed method does not (Section 5.2, footnote 3). The asymmetry in information favors DyAN, not the authors. The proposed method outperforming a supervised-coefficient baseline is a *stronger* result, not a methodological flaw. Per review guidelines, this removal is appropriate.

- **"Statistical significance is marginal — Spring-Mass: 2.58±0.79 vs 4.86±0.60" (harsh critic).** The difference (~2.28e-4) is approximately 2.9σ of the proposed method's standard deviation and 3.8σ of the baseline's — this is statistically meaningful, not marginal.

- **Missing related works criticisms.** Dismissed per instructions — no external sources to verify.

---

## Novel Insights

The most genuinely novel observation in these reviews is the implicit tension between ESRL's element-wise design and the LLLS preprocessing: for linear systems, LLLS directly extracts the A matrix, making the embedding task easy and interpretable; for nonlinear systems (e.g., Lotka-Volterra), the locally-linear A_k matrices are time-varying and noisy, potentially making the contrastive embedding problem harder. The fact that LLLS actually *hurts* LV-4D performance (Table 4: 7.73e-2 without vs. 8.31e-2 with LLLS) suggests an unresolved interaction between the feature extractor's linearity assumption and the embedding network's ability to aggregate varying A_k snapshots. This is not discussed in the paper but is important for understanding the method's limits.

---

## Suggestions

1. **Replace "zero-shot" throughout with "adaptation-free" or "unsupervised amortized embedding."** Add a paragraph in Section 2 clarifying the distinction from standard zero-shot learning (no new-system observations) and few-shot fine-tuning.

2. **Add at least one adapted baseline for vector systems.** For example, adapt CAMEL or LEADS to use the same ESRL-based embedding instead of labeled coefficients, and compare. Even a weak adapted baseline is more informative than a plain-MLP-only comparison.

3. **Explain the LLLS result in LV-4D (Table 4).** Investigate whether LLLS hurts because the locally-linear approximation is poor for high-dimensional nonlinear systems, and add a sentence or footnote in the ablation section.

4. **Explain or remove DyAN from Gray-Scott Table 3.** A brief footnote explaining why DyAN fails catastrophically on this dataset (9× worse than ResNet despite using labeled coefficients) is necessary for the table to be interpretable.

5. **Include at least one quantitative embedding–coefficient metric for Figure 3.** Pearson R or Spearman ρ between embedding dimensions and true coefficients would take minutes to compute and would directly validate the interpretability claim.

6. **Specify the SALEM coordinate system precisely.** State whether coordinates are normalized to [0,1], [-1,1], or pixel indices, and include this in the reproducibility section.

---

**Summary assessment:** The paper addresses a genuine and underexplored problem — unsupervised, adaptation-free embedding for dynamical systems — and the ESRL + LLLS + SALEM combination is a technically motivated and competently executed contribution. The empirical results on PDE systems are particularly compelling (outperforming a privileged-label baseline without any coefficient supervision). However, the paper is held back by: the inflated "zero-shot" framing, exclusively synthetic experiments, a single weak baseline for vector systems, and several unexplained anomalies (LLLS hurting LV-4D, DyAN collapsing on Gray-Scott). In its current form, the work is interesting but insufficiently validated for confident acceptance; addressing the major weaknesses above would meaningfully strengthen its case.