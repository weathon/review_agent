## Summary

This paper empirically compares Kolmogorov-Arnold Networks (KAN) and Multi-Layer Perceptrons (MLP) on ten synthetic 1D functions spanning six regularity categories: regular (smooth), continuous but non-differentiable, jump-discontinuous, singular, coherently oscillatory, and noisy. The authors match parameter counts between architectures and vary training sample sizes, reporting that KAN outperforms MLP on regular and severe-singularity/oscillation functions while MLP is generally superior on locally irregular (non-differentiable, jump) functions. The paper also finds that KAN is orders of magnitude slower than MLP in wall-clock time and that noise generally obscures the locally irregular features from both architectures.

---

## Strengths

- **Function-type taxonomy as a diagnostic lens.** The explicit categorization of test functions into six regularity classes (Table 1) provides a structured framework for understanding architecture-specific failure modes, which is more informative than a single aggregated benchmark.

- **Optimizer and wall-clock time analysis.** The inclusion of Tables 3 and 4, comparing Adam and L-BFGS for both architectures with actual training times, is a concrete practical contribution. Showing that KAN with L-BFGS is up to 70× slower than MLP — even when convergence in epochs favors KAN — is a result that practitioners need to know and that is often omitted from KAN studies.

- **Differentiated noise analysis.** Separating the noisy-function analysis into regular, localized-irregularity, and severe-discontinuity sub-categories, and finding that noise has little additional effect on already-difficult singularity/oscillation functions, is a non-obvious and practically meaningful observation.

---

## Weaknesses

### Fatal

*(None that individually invalidate the entire paper, but the combination of Major weaknesses below substantially undermines the reliability of the reported conclusions.)*

### Major

- **Critical text–figure inconsistency for jump functions (f₅, f₆).** Section 3.3 states unambiguously: "Results show that the MLP outperforms the KAN." The Figure 3 caption, however, states the opposite: "In all cases, KAN (red dashed line) fits the target function much better than MLP." These two claims are directly contradictory on the same experiment. This is not a minor labeling slip — the jump-function result is one of the paper's four main comparative findings. Without knowing which is correct, readers cannot trust either the text or the figures for this category. Numerical tables reporting final test loss are needed to settle this, and the paper has none.

- **Optimizer confounder for coherent oscillation functions.** For f₉ in Figure 8, KAN is evaluated with L-BFGS while MLP is evaluated with Adam (the per-model best), and the conclusion is drawn that "KAN consistently surpasses MLP." This is a best-of-each comparison, not a clean architectural comparison. Presenting it as evidence of architectural superiority without clearly labeling it as a practical best-system comparison conflates optimizer-architecture interaction with architectural capability. The conclusion in Section 5 ("KAN exhibits superior performance over MLP for regular functions or functions with severe discontinuities") inherits this confound.

- **Dangling cross-reference to non-existent "section D."** Section 3.5 reads: "taking a similar approach as described in section D." No such section exists anywhere in the paper. This indicates the paper is incomplete or was not proofread, and raises doubts about whether the experimental protocol for Sections 3.5 and 4.3 is fully described.

- **Noise model never formally defined.** Section 4 uses "noise level 10" and SNR values (SNR=10, SNR=0, SNR=4) interchangeably without ever specifying: (a) additive vs. multiplicative noise, (b) distribution (Gaussian, uniform?), (c) what "noise level 10" means dimensionally, (d) whether test loss is evaluated against the clean target or the noisy observations. These choices fundamentally affect the interpretation of all Section 4 results and are a reproducibility failure.

- **All results are single runs.** No variance across random seeds or noise realizations is reported. For shallow neural networks fitted on 1D toy functions, optimization noise and initialization can materially affect outcomes. Conclusions such as "KAN achieves a lower test loss with low noise levels but performs worse under high noise conditions" are drawn from what appears to be single-realization evidence. This is inadequate for any quantitative claim at ICLR.

### Minor

- **Exclusive focus on 1D univariate functions.** The Kolmogorov-Arnold theorem is fundamentally a statement about *multivariate* function representation: KAN's theoretical motivation is the decomposition of f: [0,1]ⁿ → ℝ into combinations of univariate functions. Testing only 1D functions reduces KAN to a spline approximator and bypasses the architectural regime where KAN's structure should theoretically matter. The paper briefly acknowledges that multiplication nodes in KAN 2.0 matter "minimally" for the tested functions, but this observation actually underscores the limitation rather than addressing it.

- **Wall-clock inefficiency downplayed in the conclusion.** The conclusion emphasizes that "KAN exhibits a faster convergence rate than MLP across all tested functions" (measured in epochs) without any corresponding acknowledgment of the 10–70× wall-clock overhead. A reader taking the conclusion at face value would have a seriously misleading impression of KAN's practical utility.

- **Training loss vs. test loss not disentangled.** Only test loss is reported in the convergence curves. For functions where KAN performs worse (f₃–f₆), it is impossible to determine whether the cause is poor optimization, underfitting, or overfitting. This distinction matters for understanding the architectural inductive bias and for suggesting remedies.

- **KAN grid resolution fixed without ablation.** Grid=3, k=3 is used throughout. This is KAN's most direct capacity control. For singular and oscillatory functions — exactly where spline resolution should matter most — no ablation is presented. The reader cannot tell whether KAN's failures are architectural or a result of under-specified hyperparameters.

### Tiny

- The paper contains no formal experimental setup section. Sampling domain, test-set construction, MSE vs. other loss definitions, and stopping criteria are scattered across subsections or omitted entirely.
- Section 3.3 defines jump locations as "x = ±0.5" but Table 1 defines f₅ as {1 if |x| < 0.5, else 0}, which is symmetric — this is not an error but the language in 3.3 could cause confusion.

---

## Nice-to-Haves

- **Multivariate test functions (2D/3D).** Adding at least one 2D benchmark would engage KAN's actual theoretical regime and make the comparison meaningfully broader.
- **Training-time-normalized convergence curves.** Plotting test loss against wall-clock time (in addition to epochs) would give an honest picture of the efficiency-accuracy tradeoff.
- **Mechanistic analysis of learned KAN activations.** Visualizing KAN's learned spline activations for representative functions — especially where it fails (f₃–f₆) versus succeeds (f₇–f₁₀) — would reveal whether the splines are adapting meaningfully and provide genuine insight beyond the empirical tally.
- **Bias–variance or train/test decomposition.** Separate curves showing training loss and test loss would help readers understand whether KAN's disadvantage on irregular functions stems from optimization difficulty or overfitting.
- **Theoretical hypothesis paragraph.** A brief discussion hypothesizing *why* smooth B-spline activations (KAN) might struggle at cusps and jump discontinuities while piecewise-linear activations (ReLU MLP) adapt more easily would substantially increase the paper's depth.

---

## Removed Points

*These points are flagged for removal — treat with caution.*

- **Critic: "The function definitions are potentially confusing — f₅ uses threshold x < 0.5 suggesting one-sided behavior."** Table 1 clearly defines f₅ = {1 if |x| < 0.5, 0 otherwise}, which is symmetric. This is a misread by the critic.

- **Critic: Requesting SIREN / Fourier-feature MLP baselines.** The paper's stated scope is a KAN-vs-MLP comparison. Adding sinusoidal baselines would be useful context but is outside the paper's framing and represents scope creep rather than a genuine weakness of the paper's contribution.

- **Critic: "The functions are too easy for Section 3.1."** f₁ and f₂ serve as calibration/sanity checks for a category labeled "regular." Their simplicity is the point. Demanding harder regular functions misunderstands the section's role.

- **Critic: "Severe discontinuities is inaccurate terminology."** The paper uses this as a collective label for its own subcategory of singular/oscillatory functions across restricted domains. This is a taxonomic style choice, not a factual error.

- **Critic: "Claims about KAN 2.0 are not operationalized."** The paper explicitly states that multiplication nodes matter "minimally for the functions used in this paper" and that lower versions of PyKAN are acceptable. This is addressed.

- **Positive reviewer Strength: "Timely topic / KANs were recently introduced."** This is generic and applies to any contemporaneous benchmarking paper.

- **Positive reviewer Strength: "Controlled experimental setup (matching parameter counts)."** While this is a genuine effort, it is standard practice in architecture comparison papers and does not distinguish this paper specifically, especially given the optimizer confound noted above.

---

## Novel Insights

The observation that noise has comparatively little *additional* effect on the test loss for already-difficult singularity and coherent oscillation functions (Section 4.3) is the paper's most genuinely non-obvious finding. The intuition — that approximation error already dominates before any noise is added — is plausible and, if confirmed with proper statistical rigor, would be a useful empirical result for practitioners applying KAN or MLP to highly irregular scientific data. However, this insight is currently stated qualitatively without the statistical support needed to be trusted.

---

## Suggestions

1. **Resolve the text–Figure 3 contradiction immediately.** Add a numerical table reporting mean test loss at convergence for each model on f₅ and f₆. This is the highest-priority fix.

2. **Define the noise model precisely.** Specify in a single paragraph: distribution, parameterization, how "noise level" maps to that parameter, whether test loss targets clean or noisy observations, and whether results are averaged over multiple noise draws.

3. **Run each experiment with ≥5 random seeds.** Report mean ± std on all test-loss comparisons. For the noisy experiments, additionally average over noise realizations.

4. **Fix or remove the "section D" reference** and ensure the experimental protocol for Sections 3.5 and 4.3 is self-contained.

5. **Clearly separate "best optimizer per architecture" comparisons from "fixed optimizer" comparisons.** Figure 8 is a best-of-each comparison; label it as such and present a fixed-optimizer parallel plot as an ablation.

6. **Provide an ablation over KAN grid sizes** (e.g., grid ∈ {3, 5, 10, 20}) for at least one function from each regularity category to establish that grid=3 is not artificially handicapping KAN.

7. **Rewrite the conclusion's convergence claim** to prominently note that "faster convergence in epochs" comes with 10–70× higher wall-clock cost, and quantify the tradeoff explicitly.

---

**Evaluation axes:**

- **Novelty:** Low-to-moderate. The function taxonomy is a structured contribution, but the overall experimental setup is straightforward benchmarking with no architectural or theoretical innovation.
- **Technical soundness:** Weak. The combination of a text–figure contradiction, an optimizer confounder for a key function class, a dangling section reference, and an undefined noise model constitutes a set of methodological gaps that are difficult to overlook.
- **Empirical support:** Weak. Single runs, no confidence intervals, qualitative conclusions drawn from visual inspection of plots, and unresolved inconsistencies undermine confidence in all reported results.
- **Significance:** Limited. All conclusions come from 1D toy functions; the primary regime where KAN's theory would differentiate it from MLP (multivariate decomposition) is entirely untested. The practical utility of the findings is unclear.
- **Clarity:** Below acceptable. The missing section reference, undefined noise model, and text–figure contradiction suggest the paper was not carefully proofread prior to submission.