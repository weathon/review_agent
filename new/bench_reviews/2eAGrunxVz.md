Now I have enough information to write the final review. Let me synthesize everything.

## Summary

The paper introduces Spherical Watermark, an encryption-free lossless watermarking framework for diffusion models that transforms binary watermark bits into approximately Gaussian noise via two modules: binary embedding (achieving 3-wise independence) and spherical mapping (leveraging spherical 3-designs). The key practical advantage is eliminating per-image key storage while still enabling per-user tracing via a fixed signature K=(T,C). Extensive experiments on Stable Diffusion show near-identical FID to unwatermarked baselines, classifier-undetectable outputs, and superior adversarial robustness over both lossy and lossless competitors, with extraction ~4 orders of magnitude faster than PRC Watermark.

## Strengths

- **Encryption-free design is a genuine and practically significant contribution.** Unlike Gaussian Shading (per-image key/nonce) and PRC Watermark (heavy cryptographic overhead), the method uses a fixed signature K=(T,C) shared across all images (Section 3.2), directly addressing a key deployment limitation. Per-user tracing is still achievable via the binary embedding matrix T and majority-vote decoding.

- **Dramatic computational efficiency advantage.** Figure 4 shows extraction is ~4 orders of magnitude faster than PRC Watermark, resulting from replacing belief-propagation decoding with simple matrix multiplication, rounding, and majority vote (Eq. 13).

- **Strong empirical undetectability.** Table 1 shows FID nearly identical to unwatermarked baselines (e.g., 48.1224 vs. 48.1256 on COCO/SD v1.5 — difference of 0.003). Figure 2 shows both latent-level and image-level classifiers at ~50% accuracy (chance), while Tree-Ring and Gaussian Shading (fixed-key) are detected at 97–100%.

- **Superior adversarial robustness.** Table 2: under WEvade attacks, the method achieves 98.12% ACC and 99.83% TPR, where lossy methods collapse to 49–52% ACC and even PRC Watermark reaches only 97.69% ACC and 95.38% TPR.

- **Principled theoretical framework.** The chain of results (Theorem 3.1 → Theorem 3.2 → Lemma 3.3 → Lemma 3.4) connecting 3-wise independence to spherical 3-designs to approximate Gaussianity is elegant and provides a clear structure for potential strengthening.

- **Well-designed ablation study.** Figure 6(b)–(c) isolates each module's contribution: removing binary embedding makes latents trivially distinguishable; removing spherical mapping drops robustness under brightness adjustment. Table 3 quantifies the s/N trade-off.

## Weaknesses

### Fatal
None.

### Major

- **Gap between the formal undetectability definition (Eq. 2) and the theoretical proof.** Eq. 2 requires computational indistinguishability: |Pr[A(z_w)=1] − Pr[A(z)=1]| ≤ negl(ρ). The theory (Section 3.3) only proves that the watermarked distribution matches the Gaussian prior up to third-order moments via the spherical 3-design property. Computational indistinguishability is strictly stronger than matching three moments — an adversary computing any degree-4 polynomial statistic (e.g., kurtosis, fourth-order cross-moments) could potentially distinguish the distributions. The paper acknowledges this in Section 5: "higher-order moments may deviate from the true prior," but this caveat is not reflected in the formal definition (Eq. 2), the title ("Loss-Less"), or the conclusion ("provably and empirically indistinguishable"). The abstract is appropriately careful ("theoretically prove...up to third-order moments, and empirically demonstrate...statistically indistinguishable"), but other parts of the paper overclaim. The security argument for adversarial robustness (Appendix E) rests on losslessness; if it only holds up to 3rd-order moments, the argument is incomplete.

- **Extraction correctness guarantee (Eq. 4) is asserted without formal proof, and the security parameter ρ is undefined.** Eq. 4 claims Pr[Extract(G⁻¹(O_w)) = m] ≥ 1 − negl(ρ), but no proof is provided. Extraction correctness depends on (a) approximate DDIM inversion introducing bounded errors, (b) the rounding/majority-vote step correctly recovering binary values from noisy continuous estimates, and (c) the r-scaling in the extraction step (C⁻¹ẑ_T yields r·z⁽²⁾, not z⁽²⁾, so rounding operates on approximately unit-scaled values only because r/√(l_x) ≈ 1 for large l_x). The empirical 99.99% clean ACC (Table 2) is excellent but does not constitute a formal guarantee. The paper would be significantly strengthened by even a simple analysis showing the rounding margin is large enough to absorb typical DDIM inversion errors.

### Minor

- **Missing higher-order statistical distinguishability test.** The most informative missing experiment is directly computing 4th-order moments (e.g., average kurtosis, fourth-order cross-cumulants) of z_w vs. standard Gaussian z. This would directly probe the boundary of the 3-design guarantee: if they match well, it strengthens the practical losslessness argument; if they don't, the theoretical limitation is concrete and measurable. The existing MLP classifier (Figure 2) is generic and not a targeted higher-order moment test.

- **Trade-off with Gaussian Shading under post-processing deserves more prominence.** Table 2 shows Gaussian Shading achieves ACC=98.43% and TPR=99.97% under post-processing, while Spherical Watermark achieves ACC=95.02% and TPR=97.50%. The paper's narrative focuses on adversarial robustness where Spherical Watermark excels, but this non-trivial trade-off (particularly the ~2.5% TPR gap) is mentioned only in passing.

- **The Lemma 3.4 applicability gap.** Lemma 3.4 requires u to be *uniformly* distributed on the sphere for r·u ~ N(0,I). But z⁽³⁾ is only a spherical 3-design (approximately uniform up to degree 3), not exactly uniform. The paper's argument from "3-design ≈ uniform" to "r·3-design ≈ Gaussian" is plausible but imprecise — the lemma does not bound the approximation error when u is only approximately uniform.

### Trivial
None.

## Nice-to-Haves

- Reporting Gaussian Shading with per-image keys as an upper-bound reference would let readers assess the cost of the fixed-key assumption.
- A discussion of constructing spherical t-designs for t > 3 (even theoretical feasibility) would provide a clear path toward closing the moment-matching gap.
- A formal bound on the extraction error under approximate DDIM inversion would strengthen the theoretical contribution.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Encryption-free" is misleading because a secret signature K=(T,C) is still required.** Removed because the paper clearly defines "encryption-free" as meaning no *per-image* key (Section 3.2: "K is kept fixed and secret during runtime"), which is the standard meaning in this context and accurately differentiates from Gaussian Shading's per-image key/nonce requirement.

- **Ablation on lm conflates two different scaling behaviors.** The harsh critic argues Figure 6(a) conflates Spherical Watermark's repetition code scaling with PRC's code-rate vs. error-correction trade-off. Removed because the comparison is fair — both methods face the same question of "what happens as watermark capacity increases?" — and the different scaling behaviors are precisely what makes the comparison informative.

- **Demand for per-image key Gaussian Shading results as a baseline.** Moved to Nice-to-Have. The paper explicitly notes that "Gaussian Shading no longer achieves true losslessness" with fixed keys, and the comparison is motivated by practical deployment considerations.

- **Formatting/notation nitpicks about Eq. 13 rendering.** Removed as these are parser artifacts.

## Novel Insights

The spherical 3-design framework provides an elegant bridge between combinatorial constructions (3-wise independent binary codes) and continuous distribution matching (Gaussian priors), but the paper exposes a fundamental tension in this approach: achieving higher-order moment matching requires higher-wise independence, which in turn requires more sophisticated binary code constructions. The gap between 3rd-order moment matching and computational indistinguishability is not merely a technicality — it represents the boundary between what combinatorial constructions can efficiently achieve and what cryptographic assumptions can guarantee. The empirical results suggest this gap is practically insignificant at the latent dimensions used (l_x = 16384), where the concentration of measure makes 3-designs an excellent approximation to uniformity, but formalizing this observation remains an open problem.

## Suggestions

- Revise Eq. 2 to define undetectability in terms that match the proof (e.g., "matching all polynomial statistics of degree ≤ 3") or explicitly acknowledge the gap between the formal definition and the proven guarantee.
- Add a 4th-order moment comparison experiment (e.g., average kurtosis and select fourth-order cross-cumulants of z_w vs. z) to directly test the boundary of the 3-design guarantee.
- Provide even a simple error analysis for the extraction pipeline (e.g., bounding the rounding margin after DDIM inversion), which would substantiate Eq. 4.
- Tone down the conclusion's claim from "provably and empirically indistinguishable" to match the abstract's more precise formulation.

## Score and Decision

**Calibration anchors:**
- **High:** PMark (avg 7.0, Accept Poster) — provably distortion-free LLM watermarking with solid theory AND strong experiments. Spherical Watermark has a bigger theory-practice gap.
- **Medium:** SERUM (avg 5.0, Accept Poster) — strong empirical diffusion watermarking with theoretical gaps. NoisePrints (avg 4.5, Accept Poster) — theoretical gaps under correlation. Ensemble Framework for Unbiased LLM Watermarking (avg 5.0, Accept Poster) — gap between unbiasedness and indistinguishability claim similar to this paper's gap. Spherical Watermark has a comparable profile but with a more principled theoretical framework and dramatic efficiency gains.
- **Low:** Catch-22 (avg 2.67, Reject) — fundamental misunderstanding of computational indistinguishability (not just incomplete). Probabilistic Automata (avg 2.0, Withdrawn) — incorrect proofs. NullGuard (avg 3.0, Withdrawn) — opaque methodology. Spherical Watermark is clearly above these.

Spherical Watermark is comparable to the medium-scoring accepted papers (SERUM, NoisePrints, Ensemble Framework) but offers stronger practical advantages (encryption-free design, 4 orders of magnitude speedup) and a more principled theoretical framework. The theoretical gap is real but the paper is partially transparent about it (abstract is careful, Section 5 acknowledges it). It falls below PMark-level due to the theory-practice gap but clearly above the low-scoring papers where theory is *wrong* rather than *incomplete*. A score of 5.5-6 is appropriate, leaning toward 6 given the genuine practical contributions and the fact that similar theoretical gaps in the watermarking literature have resulted in accepts at 5.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>