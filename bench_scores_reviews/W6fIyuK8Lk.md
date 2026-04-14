## Summary
This paper proposes a framework for detecting data drift, identifying drift type, and estimating drift magnitude in image classification neural networks. The central technical contribution is a quantification-based approach that extends the threshold-based prediction-probability method of Senarathna et al. (2023) to handle varying class distributions, using an overdetermined linear system combining raw and thresholded predicted-class counts. The framework is validated on MNIST, CIFAR10, and CIFAR100 across six synthetic corruption types (three noise and three weather effects).

---

## Strengths

- **Concrete handling of varying class distributions.** The paper pinpoints a real limitation of Senarathna et al. (2023): its reliance on a fixed class prior. The proposed quantification layer—constructing an overdetermined linear system via both the raw confusion-derived matrix A₁ and the threshold-conditioned matrix A₂—directly resolves this. Table 2 shows consistent improvement under high-skew distributions, with the baseline's average normalized error often reaching 2–5 compared to ≤1.5 for the proposed method, demonstrating that the added complexity is empirically justified.

- **Monotonicity of magnitude estimates.** Figure 2 shows that even when point estimates are off by one quantization level, they increase monotonically with actual magnitude. This is a practically significant property: it means the method avoids dangerous large underestimations of severity, which matters for safety-critical triggering.

- **Use of both raw and thresholded count vectors in a single overdetermined system.** Combining two distinct linear constraints (Equations 3 and 4) to build an overdetermined system is a non-trivial design decision that adds redundancy and improves conditioning relative to using either alone. This is a technically sound contribution not highlighted enough by the reviewers.

- **Breadth of experimental sweep.** Evaluating six drift types across three datasets, two class-distribution skew levels, and twenty random class distributions per magnitude—totalling a very large number of evaluation conditions—is well beyond a typical single-dataset ablation. Tables 2 and 3 present this comprehensively.

---

## Weaknesses

### Fatal
None.

### Major

- **Magnitude estimation results are reported conditional on correct type detection.** The paper explicitly states (Section 4): *"Note that the magnitude estimation results are presented considering the estimated magnitude by the method correspond to the correct drift type."* This means Tables 2 and 3 and Figures 2–3 measure only one component of the pipeline in isolation. In deployment, a type error propagates into a magnitude error. Without an end-to-end evaluation (i.e., magnitude error computed using the pipeline's predicted type, not the oracle type), the practical accuracy of the full framework is unsubstantiated.

- **No non-negativity or simplex constraints in the least-squares solve.** Class counts x_j must be non-negative integers summing to the batch size. The paper uses unconstrained least squares to solve Equation (5). Unconstrained solutions can produce negative entries, which are physically meaningless and would corrupt the residual comparison used for magnitude selection in Equation (7). Constrained least squares (non-negative least squares or simplex projection) is straightforward and should be the default here.

- **Baseline comparison is limited to a single method (Senarathna et al. 2023).** The paper cites Suprem et al. (2020), Dube & Farchi (2020), and Ackerman et al. (2021) as related methods in the same problem area but provides no quantitative comparison against them for drift detection accuracy. Positioning the framework as "comprehensive" requires at least one comparison to an independent competing method, even if only for the detection sub-task.

- **CIFAR100 requires a shadow network due to insufficient per-class validation samples.** The paper acknowledges (Section 4) that with only 100 validation images per class, the linear system becomes unstable, necessitating a shadow classifier trained on 20 CIFAR100 superclasses. This is a real scalability limitation: the framework requires sufficient per-class calibration data, making it fragile for high-cardinality classification tasks without this ad hoc workaround. The sensitivity of the linear system to calibration set size is not analyzed, and the degradation when moving from 100 classes to 20 superclasses changes the problem granularity in a way that weakens comparability with the MNIST and CIFAR10 results.

### Minor

- **Residual notation inconsistency in Equations (6) and (7).** Equation (6) defines r_M = Y − AX̂ as a vector, but Equation (7) applies argmin over (r₀, …, r_m), which requires scalar values. The text clarifies that the Euclidean norm is intended, but the equations themselves are inconsistent. This is a substantive notation error, not a style issue, because it creates ambiguity about what is actually minimized.

- **The α parameter is undefined.** The drift detection criterion is "more than α types indicate a non-zero magnitude," but α is never given a concrete value, and no analysis of false-positive/false-negative tradeoffs as a function of α is provided. This is a key operational hyperparameter.

- **Type detection network training details are underspecified.** The paper states the type detection network is trained using only drifted images "regardless of drift magnitude," but does not explain how magnitude invariance is enforced, whether a single multi-class label per corruption type is used across all magnitudes, or how the network is prevented from exploiting magnitude-specific artifacts. These details affect reproducibility and reliability of type detection claims.

- **Threshold heuristic (maximum gradient over neighboring magnitude CDFs) lacks principled justification.** Equation (2) uses a finite-difference approximation of the CDF gradient across magnitude levels to choose thresholds. While intuitive, there is no argument that maximizing pointwise separation between adjacent magnitudes is optimal for the downstream argmin in Equation (7), nor is there sensitivity analysis showing threshold stability under resampling.

- **No limitations section.** The conclusion does not discuss the assumptions and failure modes of the method. ICLR papers are expected to discuss limitations explicitly, and several substantive ones exist (see Weaknesses and Nice-to-Haves).

### Tiny

- Minor grammatical errors in the conclusion: "A novel framework is proposed… data drifts *occur*" and "It *is consisting* of…" weaken professional presentation.
- Notation overload between distributions and scalar percentages (P_i, P_{i,j}, P_{C,M}, P̂_{C,M}) increases reading burden in Sections 2–3.

---

## Nice-to-Haves

- **Unknown drift type handling.** The type detection network is closed-set. Discussing how the residual scores s_{r,T} could signal an outlier type not in the training set—or adding an "unknown" class—would materially increase practical relevance.
- **Compound/simultaneous drift evaluation.** The framework assumes exactly one drift type per batch. Testing graceful degradation under two co-occurring effects is the most realistic single extension.
- **Conditioning analysis of the linear system.** A brief analysis of when A becomes ill-conditioned as a function of batch size, number of classes, or classifier accuracy would inform practitioners on minimum reliable operating conditions.
- **Visualization of residuals across types.** Showing r_M curves for correct vs. incorrect drift types on the same plot (for one representative batch) would directly demonstrate whether type discrimination is robust or marginal.
- **Comparison of end-to-end type + magnitude error to the oracle-type magnitude error.** This single experiment would clarify how much the type classification step actually costs in overall pipeline accuracy.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **"CDF shifts to the right" is suspicious.** Reviewer 1 questioned whether CDF curves shifting right under increasing noise is physically plausible. Since the figure itself is not viewable in text rendering and the paper provides explicit numeric support for the threshold idea (e.g., 99.3% → 71.1% at τ=0.9 across noise levels), this concern cannot be confirmed and should not be penalized.
- **Equation (1) is "mathematically informal."** The formulation of P_i as a weighted mixture of P_{i,j} distributions is standard mixture-model notation and is mathematically sound. The notation is dense but not incorrect.
- **Missing broader related directions (OOD detection, conformal prediction, two-sample tests, label-shift estimation).** The paper is explicitly scoped to covariate drift detection in image classification using prediction probabilities. Criticizing the absence of literature from adjacent but distinct fields constitutes scope creep given the paper's stated contribution.
- **No online/streaming detection.** The paper explicitly frames batch processing as its operating mode. Criticizing the absence of online streaming capability is out of scope.
- **No theoretical guarantees.** This is an empirical systems paper. Demanding theoretical proofs for an applied framework is not standard for this community.
- **"Acceptable range" is subjective.** While true, defining operational thresholds for quantization error is an application-specific engineering decision, not a methodological flaw.
- **Requesting confidence intervals or multiple-run statistics.** The paper already evaluates over 20 random class distributions per condition and reports average, max, and standard deviation of quantization error. Additional statistical reporting is not needed.
- **Larger dataset size / more architectures.** Three datasets and multiple architectures are adequate for this type of work.

---

## Novel Insights

The most underappreciated insight in this paper is the use of *two distinct coefficient matrices*—one derived from the raw confusion matrix (A₁) and one from the thresholded confusion matrix (A₂)—to form a single overdetermined system. This is a non-obvious design: the thresholding operation creates a second, structurally different set of linear equations from the same input data, adding information about the prediction-probability distribution shape beyond class-level counts. The spark-finder review correctly identifies this as a novel construction, but the paper itself does not clearly articulate why the overdetermined system is strictly better than either matrix alone, leaving the key insight implicit. An ablation comparing the full overdetermined system to A₁-only or A₂-only estimation would be the single most valuable addition to make this insight actionable.

---

## Suggestions

1. **Add an end-to-end evaluation**: Report magnitude estimation error using the pipeline's predicted drift type (not the oracle type) across all datasets and drift types. This is essential for assessing deployed performance.
2. **Replace unconstrained least squares with non-negative least squares** (e.g., scipy's `nnls`) to enforce physically meaningful solutions.
3. **Specify α explicitly** and include a brief ROC-style curve or at minimum a table showing false positive rate under clean batches vs. true positive rate under drifted batches as α varies.
4. **Add an ablation for A₁-only vs. A₂-only vs. combined [A₁; A₂]** to justify the overdetermined system design quantitatively.
5. **Correct the residual notation** in Equations (6)–(7) to use ||Y − AX̂||₂ consistently.
6. **Add a limitations section** covering at minimum: closed-set drift types, batch-level operation, and the calibration-data-size requirement for high-cardinality tasks.
7. **For CIFAR100, report performance degradation** when varying the shadow network's number of superclasses (e.g., 5, 10, 20) to characterize the method's sensitivity to class granularity.