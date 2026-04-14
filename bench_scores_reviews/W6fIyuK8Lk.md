## Summary
This paper proposes a framework for diagnosing data drift in image classification neural networks, covering three aspects: drift detection, drift type identification, and drift magnitude estimation. The core novelty over prior work (Senarathna et al., 2023) is a quantification-based extension that handles varying class distributions via a linear system combining confusion-matrix-derived coefficient matrices, making magnitude estimation robust to class distribution skew in the input batch. Drift type is identified using a secondary neural network combined with a residual-based score. Experiments on MNIST, CIFAR10, and CIFAR100 under six synthetic drift types show high detection/type accuracy and low quantization error.

---

## Strengths

- **Concrete improvement over Senarathna et al. (2023) under distribution shift.** Table 2 clearly shows that under high-skew class distributions, the proposed method consistently achieves lower average, maximum, and standard deviation of normalized quantization error compared to the baseline. For CIFAR10 Gaussian, the baseline average jumps to 2.75 under high-skew while the proposed method stays at 0.54. This is a genuine, measurable improvement on the paper's targeted limitation.

- **Monotonic magnitude estimation with practical value.** Figure 2 demonstrates that estimated magnitude increases monotonically with actual magnitude. Even when exact estimation fails at the lowest levels, the system never severely underestimates higher drifts, which is precisely the property needed to trigger remedial actions in deployed systems. This is a non-trivial property of the design that most drift detection papers do not explicitly verify.

- **Multi-task diagnostic capability.** Most prior methods only detect *whether* drift occurs. This paper is one of the few to jointly estimate drift type and magnitude in a single framework, which is substantially more informative for operational decision-making (e.g., deciding on targeted retraining vs. hardware check vs. sensor replacement).

- **Reasonable breadth of evaluation.** The evaluation spans three datasets, six drift types (three noise, three weather), two class skew regimes (low and high), and two quantization granularities. Twenty random class distributions per magnitude provide distributional estimates shown in violin plots, giving reasonable empirical coverage for the stated problem scope.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Magnitude estimation results are conditioned on correct type detection.** Section 4, line 366 states: "the magnitude estimation results are presented considering the estimated magnitude by the method correspond to the correct drift type." This means the magnitude estimation tables (Tables 2–3) and figures decouple the magnitude estimation from the full end-to-end pipeline. In deployment, type errors propagate to magnitude errors. The paper never reports unconditional end-to-end magnitude estimation performance. This significantly inflates the apparent usefulness of the magnitude estimation component.

- **Mathematical inconsistency in residual definition (Equations 6–7).** The text says "the residual is obtained by computing the **Euclidean norm** of the difference between Y and AX̂," but Equation (6) defines *r_M = Y − AX̂*, which is a **vector**. Equation (7) then uses argmin(r₀, …, r_m), which is only valid if r_M is a scalar. The scalar norm and the vector are conflated across two adjacent equations. This is not a formatting artifact — it creates a genuine ambiguity about what quantity is actually minimized and undermines reproducibility.

- **No false positive rate on clean batches.** Drift detection accuracy (Table 1) is reported only on drifted batches. There is no evaluation of specificity — i.e., what fraction of clean batches are incorrectly flagged as drifted. For a deployment-oriented monitoring system, false positive rate on in-distribution data is at least as important as true positive rate, yet it is entirely absent from the evaluation.

- **Batch size sensitivity is unanalyzed.** All experiments use batches derived from ~400 images per class (effectively ~4,000–40,000 images per batch depending on dataset). The entire method is statistical and batch-based; its performance will degrade with smaller batches. The paper provides no guidance on the minimum viable batch size, which is a critical parameter for real-world deployability. This omission prevents meaningful assessment of practical applicability.

- **No non-negativity constraint on the least-squares solution.** The linear system Y = AX is solved with unconstrained least squares, which can produce negative class counts (X̂ with negative entries). Class counts are inherently non-negative, and unconstrained LS solutions under skewed or noisy conditions are likely to violate this. Non-negative least squares (NNLS) is a standard alternative that would make the solution physically meaningful. No justification is given for using unconstrained LS, and no evaluation shows whether negative solutions occur in practice.

### Minor

- **The threshold α for drift detection is underdefined.** The framework uses "more than α types indicate non-zero magnitude" as the drift detection criterion, but the paper never specifies how α is chosen, whether it is dataset- or type-dependent, or how sensitive results are to its value.

- **Score combination s_T = s_{i,T} + s_{r,T} lacks justification.** The type detection score adds the type-detection network percentage with a normalized residual score. These quantities have different scales and statistical interpretations, and equal additive weighting is assumed without discussion. Why not product, max, or a learned combination?

- **Equation (5) does not explicitly write the linear model.** The equation defines Y, X, and A, then says "Then," followed by the next sentence describing the least-squares solution. The equation Y = AX is never explicitly written, which is an unusual omission for the central equation of the method.

- **Abstract overclaims generality.** The abstract states the method "applies to any type of drift that occurs in images due to various factors." The method explicitly requires a predefined catalog of drift types, discrete candidate magnitudes, and labeled calibration data per type and magnitude. These constraints should appear in the abstract.

### Tiny

- **Notation in Equation (1) treats distributions as scalars.** P_i and P_{i,j} are described as probability distributions, but the equation writes them in a weighted-average formula that treats them as scalars. The mixture interpretation is conceptually correct but should be written as a mixture density: P_i(p) = Σ_j w_{i,j} P_{i,j}(p), to be mathematically precise.

- **"Acceptable range" language is used without definition.** Phrases like "within an acceptable range" appear multiple times without specifying application-level tolerances or reference criteria.

---

## Nice-to-Haves

- **Explicit ablation of the quantification contribution.** Table 2 compares the proposed method with Senarathna et al. under both skew conditions, which functions as an implicit ablation. Framing one row explicitly as "proposed method without quantification (static distribution)" would make the contribution cleaner and is relatively easy to add.

- **Batch size sensitivity analysis.** Even a simple curve showing detection/estimation accuracy vs. batch size (e.g., 50, 100, 500, 1000, 5000 images) would dramatically improve practical relevance.

- **Evaluation on natural/real-world distribution shifts.** Testing on benchmarks like CIFAR-10→CIFAR-10.1 or domain shift datasets would go a long way toward validating real-world utility, even if the framework's predefined-type assumption limits scope. The paper could acknowledge it requires known types while showing it correctly identifies the closest known type.

- **Behavior on unknown drift types.** Showing what residuals and type-network outputs look like when an out-of-catalog drift is injected would help practitioners understand failure modes and potentially design an "unknown drift" alarm using residual magnitude.

- **Uncertainty quantification on individual magnitude estimates.** Reporting a confidence interval or reliability flag for individual batch estimates (beyond aggregate tables) would be valuable for safety-critical applications.

- **Discussion of and comparison with NNLS.** A brief empirical comparison of unconstrained LS vs. non-negative LS would settle the concern about negative class count solutions.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Full concept drift" assumption as a major weakness.** Multiple reviewers criticize the assumption that all images in a batch share one drift type and magnitude. However, the paper explicitly scopes this in Section 2: "we consider 'full concept drifts' where every input in the input data stream is equally affected with the same drift magnitude." Criticizing the paper for not handling mixed or partial drifts is scope creep for this contribution.

- **Comparison with feature-based drift detectors (MMD, KS on features).** The paper's stated contribution is magnitude estimation via prediction probabilities, extending Senarathna et al. Feature-based methods (Suprem et al., Dube & Farchi, Ackerman et al.) are cited and acknowledged in the introduction as not providing magnitude estimation. Requiring the paper to beat them on detection-only metrics is outside its stated scope, though they were cited as context.

- **CIFAR100 shadow network as a "scalability failure."** The paper explicitly acknowledges (Section 4, lines 334–337) that CIFAR100's small per-class validation set caused instability and justified the shadow network. This is a disclosed limitation, and the shadow network is a reasonable engineering workaround rather than a hidden flaw.

- **Terminology: "concept drift" vs. "covariate drift" inconsistency.** Section 2 carefully taxonomizes these terms and explicitly clarifies the paper focuses on covariate drifts. The opening sentence is loosely worded but corrected immediately. This is too minor to raise as a substantive weakness.

- **Demand for multiple-seed statistics / confidence intervals.** The method evaluates 20 random class distributions per magnitude and reports average, max, and std in tables. Demanding formal significance tests across random seeds is not standard practice for this type of systems evaluation. The existing variance reporting is adequate.

- **Demand for theoretical proofs for threshold criterion.** The criterion in Equation (2) is heuristic, but the paper is an empirical systems paper, and demanding theoretical optimality proofs for a finite-difference gradient heuristic on CDFs is an inappropriate standard for this type of work.

---

## Novel Insights

The synthesis of the three reviews points to one genuinely underappreciated design issue: the paper's quantification-based formulation inadvertently reveals a structural tension. The linear system Y = AX is solved separately for each candidate drift type and magnitude, and magnitude is selected by argmin of residuals. This means the method is fundamentally a **nearest-template matching** scheme over a pre-computed corruption library, not a continuous or generative model of drift. This framing makes it clearer why the method degrades gracefully with coarser quantization (Table 3): with fewer templates, the correct template is better separated from neighbors. It also suggests that the method's "magnitude estimation" is really a **discrete matching** step, and accuracy depends critically on the discriminability of the corruption templates — a quantity the paper never characterizes. A conditioning analysis of the stacked coefficient matrix A as a function of drift type and magnitude would directly predict where the method fails, which could guide both evaluation design and calibration data collection.

---

## Suggestions

1. **Report end-to-end magnitude estimation accuracy** (unconditional on correct type detection) alongside the currently reported conditional results, to give an honest picture of the full pipeline's performance.

2. **Fix the residual definition**: either define r_M as the Euclidean norm (scalar) in Equation (6), or state explicitly in Equation (7) that argmin is taken over ||r_M||₂.

3. **Write Equation (5) explicitly** as Y = AX to remove ambiguity about the linear model.

4. **Add a clean-batch false positive rate** to Table 1 (or a separate row/table). This is necessary to evaluate the detection criterion in any deployment context.

5. **Conduct and report a batch size sensitivity experiment**, even briefly, to establish minimum viable batch size.

6. **Specify how α is chosen** and provide a sensitivity analysis or at least report the value used.

7. **Add a non-negative least squares variant** as a comparison or explain why unconstrained LS is preferred given the non-negativity requirement of class counts.

8. **Revise the abstract** to be explicit that the method requires: (a) a predefined catalog of drift types and magnitudes, (b) labeled calibration data per type/magnitude pair, and (c) a full-concept-drift assumption (every image in the batch affected uniformly).

---

**Evaluation summary:** The paper addresses a practical and underexplored problem with a technically coherent solution. Its novelty is **moderate** — the quantification extension is a meaningful and well-executed contribution over a known baseline, but it is incremental rather than conceptually transformative. Technical soundness is **adequate at the design level but has real notation errors** that would block reproduction. Empirical support is **sufficient within the paper's stated synthetic scope** but is undermined by the conditional reporting of magnitude results and the absence of false positive evaluation. Significance is **high for the MLOps/deployment community** and limited for fundamental ML research. Clarity is **good at the high level but breaks down in the method's mathematical core**. Addressing the major weaknesses — particularly the conditional magnitude results, the residual inconsistency, and the missing false positive rate — is necessary before this work meets venue standards.