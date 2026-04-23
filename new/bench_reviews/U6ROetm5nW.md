Now I have enough calibration data. Let me write the final review.

## Summary

This paper gives the first query-time vs. space tradeoff curve for high-dimensional Gaussian KDE data structures by instantiating the Charikar–Kapralov–Nouri–Siminelakis (2020) framework with the asymmetric LSH of Andoni et al. (2017). The key insight is that the query-time and space bottlenecks in the KDE-to-ANN reduction occur at different distance scales, so an asymmetric LSH with decoupled ρ_q and ρ_s can improve both simultaneously. The main result (Theorem 16) gives, for any δ ≥ 0, a KDE data structure with space Õ(1/μ^{1+δ}) and query time Õ(1/μ^{ξ(δ)}), where ξ(δ) is non-increasing; at δ = 0 (linear space), the query exponent is 0.1865, improving the data-independent bound of 0.25 from Charikar et al. (2020).

## Strengths

- **First time-space tradeoff curve for KDE** (Theorem 16, Figure 1): Prior work operated at a single point (linear space). This paper provides a continuous family of data structures parameterized by δ, which is a genuine structural contribution regardless of the specific numerical exponents.

- **Clean and non-obvious insight about asymmetric LSH breaking the symmetry bottleneck** (Section 1.2): The observation that query-time and space bottlenecks occur at different distance scales x, and that asymmetric LSH can exploit this mismatch by setting ρ_q low where query time dominates and ρ_s low where space dominates, is a clear conceptual advance.

- **Substantial improvement in the data-independent linear-space regime**: The exponent improvement from 0.25 to 0.1865 (Theorem 17, second bullet) represents a ~25% reduction in the query time exponent for the most practically relevant setting, with a simpler (data-independent) analysis.

- **Valuable barrier identification**: The analysis showing that even with ρ_q = 0 (constant-time ANN queries), intermediate-scale collisions yield at least ≈ 1/μ^{0.09} overhead (Section 1.2, Eq. 7) identifies a genuine limitation of the ANN-to-KDE reduction approach and poses a concrete open problem.

## Weaknesses

### Fatal
None.

### Major

- **Core technical lemma (Lemma 31) not stated in the body, making the derivation chain unverifiable in the main text**: The paper's contribution rests on specific numerical exponents derived from an optimization problem (Eq. 10). This optimization depends critically on Lemma 31, which governs how the asymmetric LSH collision probabilities interact with the density-constrained KDE framework. The lemma is mentioned only in passing (Section 4, ~line 249) and its statement and proof are entirely in the appendix. For a theory paper where the specific exponents *are* the main contribution, the reader cannot verify the optimization formulation without consulting the appendix. This is a significant verifiability gap that weakens confidence in the claimed results. (The paper does acknowledge this is in Appendix C, but for a conference submission, the critical lemma should at minimum be stated with a proof sketch in the body.)

- **No error bounds or verification for numerically obtained exponents**: The headline results — query exponent 0.051 and space exponent 4.15 — come from solving the nested optimization ξ(δ) = max_x min_ρ max_y [...]. The paper provides no error analysis, no closed-form upper bounds confirming these values, no sensitivity analysis, and no discussion of numerical precision. The function ξ(δ, x) involves differences of terms that could be close in magnitude, making the optimization potentially sensitive. While the paper is transparent that "the exact optimum does not seem simple to obtain analytically, and we therefore resort to numerics" (Section 1.2), even loose analytical upper bounds (e.g., proving ξ(0) ≤ 0.2 analytically) would significantly strengthen confidence in the numerical results.

### Minor

- **The headline comparison (0.051 vs. 0.173) in the abstract conflates different space regimes**: The abstract leads with "significantly improved query time ≈ 1/μ^{0.05} at the expense of somewhat higher space complexity of ≈ 1/μ^{4.15}" versus Charikar et al.'s 1/μ^{0.173} with linear space. The paper acknowledges the caveat in Section 1.1 ("with the caveat that their space requirement is only 1/μ"), but the abstract's framing of "significantly improved query time" without prominently noting the tradeoff regime difference could mislead readers who only read the abstract. The more apples-to-apples comparison for linear space (0.1865 vs. 0.173) is a more modest improvement.

- **The claim that the analysis is "arguably much simpler" than Charikar et al. (2020)** (Section 1.2, line ~106): The paper uses the same Charikar et al. framework and swaps in the asymmetric LSH from Razenshteyn (2017), which is itself a complex data structure with a tree of random Gaussians and multi-path traversal. While simpler than data-dependent LSH, calling it "much simpler" in absolute terms is an overstatement — the core complexity is just shifted from data-dependence to the asymmetric collision probability analysis.

### Trivial
None.

## Nice-to-Haves

- Closed-form or provable upper bounds on ξ(δ) for key values of δ (even loose ones) would strengthen confidence in the numerics and provide insight into what drives the exponent.
- A plot showing the optimizing x* for each δ would reveal the structure of the tradeoff and help build intuition about where the hardness lies.
- Discussion of whether data-dependent asymmetric LSH could further close the remaining gap to 0.173 or beat it would be a natural extension to consider.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic's criticism about the section heading "Why constant query KDE is not possible with known ANN results"**: The paper's body text is careful ("not possible with present near neighbor search technology"), and the section heading is a reasonable informal summary. The body explicitly states the limitation is for the specific reduction, not all possible approaches. This is a minor presentation nitpick at best.

- **Harsh Critic's criticism about d = Õ(1) suppressing polynomial dependence on dimension (Section 2.1)**: This is standard practice following Charikar et al. (2020) and is explicitly noted in the setup. Requesting explicit polynomial dependence is a generic scope-creep criticism.

- **Harsh Critic's criticism about Eq. 7's derivation/citation provenance**: The expression for collision probability overhead with general ρ_q is derived from the asymmetric LSH collision probability bounds in Claim 23 and the constraint in Eq. 8. The technical overview provides the high-level derivation; full details are appropriately in the appendix for a theory paper.

- **Harsh Critic's criticism about Definition 14's threshold function and piecewise definitions being "stated without derivation"**: The paper explicitly references Appendix C for the derivation and motivates the intuition verbally in the text. This is a standard organization choice for a conference paper.

- **Harsh Critic's request for sensitivity analysis of exponents to c_0, c_1 constants**: The paper explicitly states these can be "set to any arbitrarily small constant" and handles boundary scales with the Charikar et al. data structure. This is a minor point that the paper has already addressed at the level appropriate for the venue.

- **Strength Finder's strength about "Careful per-scale parameter optimization with analytical threshold structure"**: While Definition 14 is indeed non-trivial, this is essentially a restatement of the paper's main contribution rather than a distinct strength. Moved to avoid inflation.

## Novel Insights

The most interesting structural observation is that the KDE-to-ANN reduction creates a fundamental asymmetry between query-time and space bottlenecks that symmetric LSH cannot exploit — this mismatch exists because query time is dominated by intermediate-scale collisions (via the max over y ∈ [x,1]) while space is dominated by the largest dataset scale. This suggests that any ANN-to-KDE reduction using symmetric hashing leaves performance on the table, and the gap between the asymmetric result (0.1865) and the data-dependent symmetric result (0.173) narrows precisely because the asymmetric approach partially captures what data-dependent methods were achieving through a different mechanism.

## Suggestions

- Include a statement and proof sketch of Lemma 31 in the main body. Even a half-page sketch would allow readers to verify the optimization formulation without flipping to the appendix.
- Provide analytical upper bounds on ξ(δ) for δ = 0 and the optimal δ that confirm the claimed exponents are correct (even if slightly looser, e.g., "ξ(0) ≤ 0.20" would be more convincing than "ξ(0) ≈ 0.1865" with no error analysis).
- Adjust the abstract to present the tradeoff more fairly: either lead with the linear-space result (0.1865 improvement) or make the space cost of the 0.051 result more prominent.

## Evaluation

**Originality**: High. The idea of using asymmetric LSH to exploit the mismatch between query-time and space bottlenecks in the KDE framework is novel and non-obvious. The first time-space tradeoff curve for KDE is a genuine structural contribution.

**Importance of research question**: High. KDE is a fundamental problem in ML, and sublinear-time data structures for high-dimensional KDE are an active area with direct applications to attention computation in transformers.

**Claims support**: Moderate. The theoretical framework is sound and the reduction is well-structured, but the specific numerical exponents rest on an appendix-only lemma and numerical optimization without error bounds, which weakens verifiability.

**Soundness of experiments**: Not applicable (theory paper, no experiments). The numerical evaluation of the optimization is presented transparently in Figure 1 but lacks precision information.

**Clarity**: Good. The technical overview is clear and progressively builds from the barrier analysis to the optimization. The reformulation in terms of a general (c,r)-ANN oracle is a useful abstraction.

**Value to research community**: High. The tradeoff curve, the barrier identification, and the technique of using asymmetric LSH for KDE all provide value for future work.

## Calibration Anchors

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| Probabilistic Kernel for ANNS (nCsF3Bsn2n) | 8.0 | Similar domain (kernel + hashing for ANN), but that paper has strong empirical results and rigorous theory. Our paper has comparable theoretical novelty but weaker verifiability due to numerics-without-error-bounds. Below this. |
| Dynamic Low-Rank Fast Gaussian Transform (dbaGyviiYF) | 5.6 | Directly competitive domain (dynamic KDE data structure). That paper was rejected despite some positive reviews due to limited scope and controversial assumptions. Our paper has a clearer conceptual contribution (first tradeoff curve) but faces the numerics verifiability issue. |
| Newton Method Revisited (0eM74HjPQA) | 5.2 | Theory paper with solid contributions but overclaimed scope; accepted as poster. Our paper has a similar profile: genuine contribution with some verifiability/presentation concerns. |
| Adversarially Robust ANN via LSH (69iBZ4DzXg) | 4.8 | LSH-based ANN theory paper; rejected due to unclear presentation and limited practical motivation. Our paper has clearer presentation and a cleaner insight, placing it above this. |
| Almost-Linear-Time Transformer Gradients (ymidNjoH4V) | 2.5 | Theory paper with unverified claims and absent experiments; withdrawn. Our paper's claims are more transparent about relying on numerics and the core framework is verifiable. Well above this. |
| Quantum Framework for Optimization (XBz6cMlv8Z) | 2.0 | Core assumptions unsubstantiated; rejected. Our paper has solid grounding in prior work (Charikar et al. framework + Razenshteyn LSH). Far above this. |

The paper sits between the medium-scoring theory papers (4.8–5.6) and the high-scoring ones (7.0+). It has a genuine conceptual contribution (first tradeoff curve, asymmetric LSH insight) but is held back by the verifiability gap (Lemma 31 in appendix, numerics without error bounds). Compared to the Dynamic FGT paper (5.6, rejected), our paper has a cleaner and more novel contribution. Compared to the Newton paper (5.2, accepted poster), it has comparable depth but a more significant structural result. I place it in the 5.5–6.0 range — a solid theory contribution that would benefit from the suggested improvements but is not fundamentally flawed.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>