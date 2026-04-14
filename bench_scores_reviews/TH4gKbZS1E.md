## Summary

This paper presents a controlled empirical benchmark comparing Kolmogorov-Arnold Networks (KAN) and Multi-Layer Perceptrons (MLP) on ten hand-curated 1D functions spanning six regularity classes: regular/smooth, continuous-but-non-differentiable, jump-discontinuous, singular, coherently-oscillatory, and noisy variants thereof. The authors match parameter counts between architectures and study the effects of training sample size and optimizer choice (Adam vs. L-BFGS). The central finding is that KAN does not uniformly outperform MLP: KAN appears superior on regular and singular/oscillatory functions, while MLP prevails on cusp and jump-discontinuity functions. The paper is an explicit extension of the authors' prior work (Shen et al., 2024) on KAN noise sensitivity.

---

## Strengths

- **Structured function taxonomy with controlled parameter matching:** The six-category classification is practically motivated, and Table 2 shows near-exact parameter parity (118 vs. 120, and 238 vs. 240 parameters). Matching parameters rather than relying on ad hoc choices is a deliberate and methodologically sound choice that is missing from many KAN vs. MLP comparisons in the literature.

- **Concrete optimizer sensitivity findings with runtime data:** Tables 3 and 4 provide quantitative evidence that KAN with L-BFGS incurs 30–70× wall-clock overhead versus MLP (e.g., 588s vs. 8.3s for f₇). This is a specific, practically consequential finding that practitioners comparing architectures need but seldom find explicitly reported.

- **Non-trivial asymmetric performance pattern:** The empirical result that KAN is worse on cusp/jump functions (f₃–f₆) but better on singular/oscillatory functions (f₇–f₁₀) is not an obvious or foregone conclusion. It distinguishes between function types that other comparative studies treat uniformly and offers a structurally interesting—if underanalyzed—observation about spline-based inductive biases.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Figure 3 caption directly contradicts Section 3.3 text.** The text (Section 3.3) states: "Results show that the MLP outperforms the KAN" for jump functions f₅ and f₆. The alt-text/caption of Figure 3 states: "In all cases, KAN (red dashed line) fits the target function (green squares) much better than MLP (blue dashed line)." These claims are logically incompatible. This contradiction is not a parser artifact—both appear explicitly in the extracted document. It creates fundamental ambiguity about what the actual experimental result is for one of the paper's most important function categories, and directly undermines the reliability of the reported findings.

- **Optimizer mismatch confounds the architectural comparison for f₉.** Figure 8's caption confirms that for f₉, the comparison is MLP (Adam) vs. KAN (L-BFGS). The paper then concludes "KAN consistently surpasses MLP" on coherent oscillations. Since L-BFGS is the better-performing optimizer for KAN on f₉ (per Figure 7), this comparison attributes to *architecture* an advantage that is at least partly attributable to *optimizer*. The same mismatch appears in Figure 11(g) for the noisy f₉ comparison. Claims about architectural superiority on oscillatory functions are therefore not cleanly supported.

- **No statistical reliability analysis.** All results appear to be single training runs with no reporting of variance across seeds, initializations, or data subsamples. Neural network training is stochastic, and the performance gaps in many of the figures are moderate. Without multiple runs and summary statistics, it is not possible to distinguish genuine architectural differences from initialization noise. This is a baseline methodological requirement for empirical comparisons at ICLR.

- **Limited novelty relative to stated prior work.** The paper explicitly states: "This research continues directly and naturally from our recent study on the efficacy of KANs in fitting noisy functions (Shen et al., 2024)." The additional contribution—extending from noisy regular functions to a broader taxonomy of irregular functions—is incremental, and the paper does not articulate a clear mechanistic advance or conceptual insight that goes beyond cataloging performance across function types. The absence of any explanatory analysis (see below) makes this feel like a dataset-of-experiments extension rather than a standalone contribution.

- **No mechanistic explanation for observed performance patterns.** The paper documents that KAN underperforms on cusps and jumps but does not explain why. Is it the smooth B-spline basis that cannot represent discontinuities efficiently? Grid resolution? Optimization landscape geometry near non-smooth targets? Without any mechanistic analysis—even informal—the findings remain a list of empirical observations rather than actionable insights. ICLR expects understanding, not just measurements.

### Minor

- **Shallow and narrow architecture search.** Only single-hidden-layer MLPs ([1,39,1] and [1,79,1]) are evaluated. Multi-layer MLPs are the dominant baseline in modern deep learning and are known to behave differently on approximation tasks. The results may not generalize to the architectures practitioners actually use.

- **Computational cost is measured but underemphasized in conclusions.** Tables 3 and 4 clearly show that KAN with L-BFGS is orders of magnitude slower than MLP. The conclusion's summary of findings does not foreground this tradeoff. For a paper whose audience includes practitioners deciding between architectures, the efficiency cost is as important as the accuracy result.

- **Noise model is insufficiently specified.** Section 4 introduces noise levels of 0, 2, 4, and 10, and Figure 11 uses SNR=0, 4, 10, but the main text never formally defines the noise distribution (Gaussian? uniform?), how SNR is computed relative to the signal, or whether noise is applied to inputs or outputs only. This makes the noisy-function experiments difficult to reproduce or interpret precisely.

- **"Severe discontinuities" is a misleading category label.** In Section 4, singularities (f₇, f₈) and coherently oscillatory functions (f₉, f₁₀) are grouped under "severe discontinuities." These are qualitatively distinct: singularities have well-defined limits that diverge, while coherent oscillations have no limit. Conflating them under one label may obscure differences in behavior.

### Tiny

- The section header "KOMOGOROV-ARNOLD THEOREM" in Section 2 contains a typo ("Komogorov" missing one 'l').

---

## Nice-to-Haves

- **Hyperparameter ablation for KAN (grid size, spline order k).** Only grid=3, k=3 is tested throughout. Since grid resolution directly controls the B-spline's ability to represent sharp features, ablating this setting would clarify whether the observed weaknesses of KAN on jumps and cusps are fundamental to the architecture or an artifact of the chosen configuration.

- **Compute-budget-matched comparisons.** Parameter count matching is a first step, but given the 10–70× runtime gap shown in Tables 3–4, a comparison at equal wall-clock time or equal number of function evaluations would give a more practically relevant picture of the accuracy–efficiency tradeoff.

- **Extension to multivariate functions.** The Kolmogorov-Arnold theorem and KAN's theoretical motivation are specifically about multivariate compositional structure. Including even a few bivariate test cases (e.g., f(x₁,x₂) = sin(x₁)/x₂ with a singularity at x₂=0) would strengthen the paper's connection to KAN's theoretical basis and expand the scope of conclusions.

- **Zoomed visualizations near irregular points.** Global function-fitting plots make it difficult to assess whether either architecture is capturing local behavior near cusps, jump locations, or singularities. Inset zoomed views near x=0 (for f₃, f₇, f₉) and x=±0.5 (for f₅, f₆) would provide clearer diagnostic evidence.

- **Visualization of learned KAN activation functions.** A claimed advantage of KAN is interpretability. Showing the learned univariate activations for a few cases (e.g., whether KAN learns a 1/x-shaped activation for f₇) would both validate or challenge the interpretability claim and provide mechanistic insight into how KAN represents different function types.

---

## Removed Points

*These points were raised by sub-reviewers but are removed or substantially weakened here for the stated reasons — treat them with caution.*

- **Criticism: The paper should evaluate on high-dimensional or real-world datasets.** Removed as scope creep. The paper explicitly scopes its contribution to controlled function regularity benchmarks. Evaluating whether KAN does X well should not be penalized for not also doing Y. Multivariate extension is listed as a nice-to-have but is not a core flaw given the stated scope.

- **Criticism: Formal contributions are not enumerated in the introduction.** Removed as a pure presentation nitpick. The paper's structure makes the contributions inferrable even without a bullet-pointed list.

- **Criticism: The regularity taxonomy is not grounded in formal approximation theory (Sobolev, Hölder, BV).** Removed as an unjustified rigor demand for an empirical paper. The informal categorization is serviceable for the experiments and is pedagogically clear. A formal measure-theoretic taxonomy is not standard practice in this subfield.

- **Criticism: Comparisons where the worse optimizer is used for MLP are "unfair."** Removed — using the weaker optimizer for the *stronger-performing* architecture (MLP) would only strengthen the paper's claims where MLP wins. The cases where the concern is real (optimizer mismatch favoring KAN) are already captured as a genuine Major weakness above.

- **Criticism: References to Shen et al. 2024 and other cited works may not exist.** Removed per instruction — if cited, assumed to exist.

- **Criticism: The title is too narrow.** Removed as a formatting/style concern.

- **Weakness claimed: "KAN exhibits faster convergence across all tested functions" is unsupported.** Partially mitigated — the paper text in Section 5 does make this claim, and the figure descriptions are largely consistent with it (Figure 4: "KAN consistently achieves lower loss than MLP"). The convergence advantage appears consistently observed, even if it must be tempered by the lack of statistical confidence.

---

## Novel Insights

The most interesting and underexploited observation in this paper is the asymmetry between the two main failure modes: KAN underperforms on functions with localized, bounded irregularities (cusps and jumps), yet outperforms on functions with globally extreme behavior (singularities and densely oscillatory near unreachable points). This asymmetry is not explained anywhere in the paper but hints at a genuine inductive bias story: spline-based activations may be well-suited for capturing globally steep or monotone-local behavior but ill-suited for capturing bounded, localized transitions. If the paper could articulate and test this mechanistic hypothesis — e.g., by examining whether increasing grid density helps on cusps but not jumps, or whether the spline coefficients exhibit pathological behavior near discontinuities — it would transform a catalog of observations into an explanatory contribution.

---

## Suggestions

1. **Immediately resolve the Figure 3 vs. Section 3.3 contradiction.** Check whether the figures show KAN or MLP winning on jump functions and align caption, figure, and text to reflect the actual data. This is the single most urgent correction.

2. **Fix the optimizer mismatch for f₉.** Either run MLP and KAN under the same best optimizer for each (with clear reporting) or present a full 2×2 optimizer × architecture factorial design for the affected functions, so architectural and optimizer effects can be disentangled.

3. **Run each experiment with at least 5 random seeds and report mean ± std.** This applies to all learning curves and final test loss values. Even condensed to a supplementary table, this would substantially increase the evidential value of the comparisons.

4. **Add a mechanistic analysis section.** For the most striking results (KAN's failure on jump functions and success on singularities), provide at least an ablation: vary grid size for KAN on f₅/f₆ and f₇/f₈, and report whether increasing grid resolution closes the gap on jumps. This would directly test the hypothesis that spline resolution (rather than architecture fundamentals) drives the observed differences.

5. **Define the noise model precisely** in Section 4, specifying distribution, parameterization of noise levels, and whether test loss is evaluated against clean or noisy labels.

6. **Foreground the computational cost tradeoff** in the abstract and conclusion. The 10–70× runtime overhead of KAN is a practically critical result that deserves equal prominence to the accuracy comparisons.

---

## Evaluation

| Axis | Assessment |
|------|------------|
| **Originality** | Low-to-moderate. The function taxonomy is useful, but the paper is explicitly incremental relative to the authors' prior work (Shen et al., 2024), and the empirical KAN-vs-MLP comparison space is now crowded. No new methodology, theoretical insight, or analytical framework is introduced. |
| **Importance of research question** | Moderate. Knowing when KAN helps vs. hurts relative to MLP is practically relevant for the many researchers now considering KAN as an alternative. |
| **Claim support** | Weak. The Figure 3 contradiction, optimizer mismatch for f₉, and absence of multi-seed statistics mean several headline claims cannot be taken at face value without correction. |
| **Soundness of experiments** | Weak. Parameter matching is careful, but the absence of statistical analysis, the shallow-architecture restriction, and the optimizer mismatch undermine the controlled-comparison framing. |
| **Clarity of writing** | Adequate. The structure is easy to follow, but the Figure 3 contradiction, imprecise noise model, and misleading category label ("severe discontinuities") are genuine clarity failures. |
| **Value to research community** | Limited in current form. The asymmetric performance pattern across regularity classes is potentially useful, but the methodological issues mean practitioners cannot fully trust the results without replication. |
| **Contextualization relative to prior work** | Adequate acknowledgment of the negative KAN literature; the explicit framing as an extension of Shen et al. 2024 is honest, though it weakens the novelty claim. |