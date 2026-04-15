Now let me read some calibration papers for scoring.Now I have enough to write the final consolidated review. Let me synthesize everything.

---

## Summary

This paper presents **Pruning AMR**, an algorithm for adaptively discretizing a pre-trained implicit neural representation (INR) for visualization purposes, without requiring access to the original training data. The key idea is to use interpolative decomposition (ID) pruning of the INR's weight matrices as a proxy for local geometric complexity: regions where the INR can be heavily pruned with little accuracy loss are deemed low-detail and left coarse, while regions that resist pruning are refined. The method is validated on a 2D benchmark function and two 4D dynamic CT scan INRs, demonstrating DOF savings over uniform and a simple "Basic AMR" baseline in most settings.

---

## Strengths

- **Novel problem framing**: Visualizing pre-trained INRs without access to training data is a real and underserved problem. The paper makes a credible case that no prior work directly addresses this setting (Section 2.2), and the motivation from dynamic micro-CT (1024³×700 ≈ 3.6TB) is compelling and concrete.
- **Genuinely non-obvious idea**: Using weight-matrix prunability as a refinement indicator for AMR is creative cross-pollination between model compression and scientific visualization. Bridging Chee et al.'s ID pruning to an AMR criterion is not a trivial adaption.
- **Honest and credible evaluation**: Section 4.3 openly acknowledges that on the real log-pile data the gains are only marginal ("we believe this reflects the sparsity of low-detail regions"), and the paper does not oversell that result. This strengthens trust in the reported numbers.
- **Strong result in the simulated CT case**: Figure 3a shows Pruning AMR achieving comparable error to Uniform/Basic at roughly 10× fewer DOFs after 5 iterations, which is a meaningful empirical result.
- **Correct evaluation metric**: Error-vs-DOFs log-log plots are the standard metric for AMR quality, appropriately used throughout.

---

## Weaknesses

### Fatal
_None that completely invalidate the contribution._

### Major

- **Efficiency claim is unsubstantiated without runtime analysis.** The paper's primary framing is "efficient visualization" (abstract, title, conclusion), yet no wall-clock times are reported for any method. For every candidate element, Pruning AMR must perform: (i) sample `ID_samples` points, (ii) compute the rank-revealing QR factorization for each layer's weight matrix, (iii) reconstruct the pruned network sequentially, and (iv) sample `error_samples` points for error estimation. On a 5-layer width-256 INR over many elements across 5 iterations, this is non-trivial compute. Section 4.3 explicitly mentions "due to computation time constraints, we were not able to investigate this further," which signals the cost is significant. A method that saves output DOFs but is expensive to decide where to refine does not constitute "efficient visualization" in any end-to-end meaningful sense. A comparison of wall-clock time is essential.

- **Core hypothesis is stated but never validated.** Section 3 explicitly labels the prunability-to-detail link as a "hypothesis": *"we rely on the hypothesis that the less detailed a function is on a region of the domain, the smaller an INR needs to be to accurately describe the function in that region."* This is never directly tested. Pruning error is a property of the network parameterization under domain restriction, not obviously of local function variation. With Gaussian random Fourier features and global-support activations, there is no guarantee that local geometric complexity maps predictably to ID-rank of weight matrices. A direct scatter plot of local gradient magnitude vs. pruning proportion per element would either validate or falsify this central assumption. Without it, the paper's core interpretive claim is not established.

- **Weak baseline comparisons.** The only adaptive comparator is "Basic AMR," a custom heuristic designed by the authors that uses bilinear-interpolant error. No comparison is made to natural, existing AMR indicators available for differentiable INRs, such as gradient-magnitude sampling (∥∇f∥ is free from a backward pass), local variance estimators, or any method from the established AMR literature. Since INRs are fully differentiable, gradient-based refinement is the most direct competitor and could be implemented with minimal overhead. Without at least one principled alternative, it is impossible to determine whether pruning is adding value over simpler data-dependent criteria, or whether the advantage is entirely due to the Basic baseline being intentionally simple.

### Minor

- **Asymmetric hyperparameter tuning weakens the quantitative comparison.** For the simulated CT experiment, Pruning uses `error_samples = 32` while Basic uses `error_samples = 256`. This gives Pruning significantly fewer samples to compute its error estimate, yet the comparison is on the same error-vs-DOF plot. The paper does not discuss why these are set asymmetrically or what effect this has. This, combined with per-example hand-tuning of P, T, and τ, makes the quantitative margin difficult to interpret.

- **Results on log pile (the most realistic example) are marginal.** In Section 4.3, all three methods perform nearly identically for the first four iterations, with only minimal divergence at iteration 5. The paper's explanation ("sparsity of low-detail regions") is reasonable but is speculative and unquantified. Since this is the most practically realistic data (real noisy CT scan), marginal gains here temper the broad conclusion.

- **Scale gap between motivation and experiments.** The introduction motivates the work with a 1024³×700 (3.6TB) dataset, but all experiments use small-scale instances with only a few thousand DOFs. The practical applicability to the motivating scale is entirely assumed.

### Trivial

- **No ablation of the dual criterion (P and T).** Algorithm 1 refines if `error > T OR proportion > P`, but the contribution of the proportion-only criterion (P threshold) is never isolated. If T alone suffices, the distinctive pruning-based component of the algorithm may be less necessary than claimed.

---

## Nice-to-Haves

- Provide a scatter plot of per-element pruning proportion vs. local gradient magnitude across the domain. This single visualization would go a long way toward validating the core hypothesis.
- Include a wall-clock timing table comparing total runtime (not just DOFs) across methods. Even an informal estimate would address the efficiency concern.
- Provide guidance for setting P automatically (e.g., tied to a target compression ratio), since examples require different values (0.09 / 0.075 / 0.1).
- Expand the clarification of how the spatial domain restriction modifies the ID computation — whether domain restriction affects the input activation matrix or only the error estimation step.

---

## Removed Points

_These points are flagged to be removed, treat them with caution._

- **[Human Finder] Missing comparison to Instant-NGP or grid-based INR methods.** The paper explicitly scopes its contribution to pre-trained INRs with no training data access; Instant-NGP is a different paradigm (requires training/data structures) and is out of scope. Removed per scope-creep rule.

- **[Human Finder] Limited diversity in test data / modality generalization.** Requesting more modalities is reasonable, but the paper is explicitly positioned as a methods paper for a specific problem (data-free AMR for pre-trained INRs), not a general-purpose benchmark survey. Weakened to minor and folded into the scale-gap point above rather than kept as a standalone weakness.

- **[Harsh Critic] Claim that Treating Uniform as "ground truth" in Figure 2 is not justified.** The paper's caption says "Treating Uniform as 'ground truth'" as a visual reference for comparison — this is a caption shorthand, not a methodological claim. The actual error metric (Section 4.1) uses the true INR values, not uniform mesh values. This is a nitpick on caption language, not a substantive error.

- **[Harsh Critic] Instability of the error criterion near-zero INR outputs.** The paper uses `mean(|INR(X)-INR_pruned(X)| / |INR(X)|)`. While division by near-zero is a legitimate concern in principle, the CT scan INRs in this paper represent density values that are not near zero in any meaningful region. No evidence is given that this caused numerical issues. Removed as speculative.

- **[Harsh Critic] Stochastic variance of pruning not reported.** Requesting repeated-run variance for a systems/methods paper is not standard in this field for small-scale demonstrations. Moved to nice-to-have.

---

## Novel Insights

The paper's most genuinely novel observation — visible in Figure 5 and Figure 7 — is that the time-varying mesh produced by Pruning AMR naturally tracks the temporal evolution of the object being scanned, producing different optimal meshes at different time slices even though no time-specific information was provided. This emergent temporal adaptivity is a compelling demonstration that the prunability signal captures meaningful structure across the spacetime domain. The observation that prunability saturates quickly on uniformly-detailed or noisy data (the log-pile case) is also a useful diagnostic for understanding when INR-based AMR is worth deploying, and is worth formalizing in future work.

---

## Suggestions

1. **Add wall-clock timing.** Report total runtime (including pruning step) per iteration per method. If the pruning overhead is large, explore approximations (e.g., single-layer ID instead of all layers).
2. **Implement and compare against gradient-based AMR.** Compute ∥∇f∥ inside each element by backpropagating through the INR; refine where the norm exceeds a threshold. This is cheap and the most natural competitor.
3. **Directly validate the prunability-detail hypothesis.** For the 2D benchmark where ground truth is known analytically, plot per-element pruning proportion vs. local function variation (e.g., max gradient magnitude). If the correlation is tight, this alone would substantially strengthen the paper's theoretical grounding.
4. **Ablate P vs T.** Show results for T-only, P-only, and P+T conditions to clarify the contribution of each criterion.
5. **Address the error_samples asymmetry.** Either use the same value for both methods or justify the difference explicitly.

---

## Score and Decision

**Calibration:**

| Paper | Topic | Scores | Decision |
|---|---|---|---|
| DMM (hj9ZuNimRl) | Data-free mesh mover for neural PDE | 6,6,6 | Accept poster |
| ASMR (kMp8zCsXNb) | INR inference efficiency | 8,6,5 | Accept poster |
| HIWE (NLRo4qhg6t) | Importance-weighted encoding, NeRF | 3,1,3,5 | Reject |

This paper sits below the DMM paper, which had theoretical guarantees (Monge-Ampère), multiple PDE benchmarks, and runtime comparisons. DMM's core mechanism was more rigorously derived, whereas this paper's central hypothesis is explicitly acknowledged as unverified. This paper also lacks runtime data and uses much weaker baselines than DMM.

It is above HIWE, which had methodological circularity and inadequate baselines with quantitatively worse metrics on key measures. This paper's idea is more coherent and its empirical reporting is honest.

The appropriate score is **4.5** — below the 6-score bar that DMM achieved, reflecting: a genuinely novel and non-obvious idea that is underdeveloped as an empirical contribution. The absence of runtime data undermines the primary claim of efficiency, the core hypothesis is unvalidated, and the experimental scope is narrow (3 examples, 2 meaningful adaptive comparisons on the 2D case and simulated CT). With the log-pile showing marginal gains, the method's generality is in question. A stronger version of this paper with runtime data, gradient-based baseline, and hypothesis validation could be a solid poster; the current version is not ready.

**Originality:** Good — the problem is novel and the pruning-as-AMR-signal idea is non-obvious.  
**Importance of research question:** Moderate-to-high — 4D CT visualization is a real bottleneck.  
**Claims well-supported:** Weak — the primary efficiency claim lacks runtime evidence; the core hypothesis lacks direct validation.  
**Soundness of experiments:** Below average — only 3 examples, one custom weak baseline, no timing, asymmetric hyperparameter tuning.  
**Clarity of writing:** Good — the algorithm is clearly described and limitations are honestly reported.  
**Value to research community:** Moderate — introduces a problem worth studying, but current experimental evidence is insufficient for a top venue.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>