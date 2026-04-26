Now I have a thorough understanding of the paper and the calibration anchors. Let me synthesize the final review.

## Summary

Spherical Watermark proposes a novel lossless watermarking framework for diffusion models that transforms binary watermark messages into Gaussian-distributed noise via two key modules: binary embedding (mixing watermark bits with random padding to produce a 3-wise independent bitstream) and spherical mapping (projecting onto the unit sphere, applying an orthogonal rotation, and scaling by a chi-squared radius). The approach eliminates per-image key storage (unlike Gaussian Shading) and avoids cryptographic decoding overhead (unlike PRC Watermark), while achieving strong empirical undetectability and robustness.

## Strengths

- **Novel technical construction connecting spherical designs to watermarking:** The idea of using spherical t-designs to map binary watermarks into Gaussian-like noise is creative and well-motivated. Connecting coding theory (3-wise independent codes via binary embedding) with spherical design theory for watermarking is a genuine conceptual contribution, and the pipeline from binary embedding → spherical mapping → chi-squared scaling is precisely specified and well-structured.

- **Strong empirical undetectability results:** Table 1 shows FID values (e.g., 48.12 on COCO/SD v1.5) virtually indistinguishable from the unwatermarked original (48.13). Figure 2 confirms near-chance classifier detection (≈50%) for both latent-level and image-level tests, while Tree-Ring and Gaussian Shading (under fixed keys) yield 97–100% detection rates. These are compelling empirical results.

- **Substantial practical efficiency gains:** Figure 4 shows approximately four orders of magnitude speedup in extraction over PRC Watermark, which is a meaningful deployment advantage. Eliminating per-image key storage is a genuine practical benefit for real-world deployment.

- **Consistent robustness across settings:** Table 2 shows strong tracing accuracy under clean, post-processing, and adversarial conditions (e.g., 98.12% ACC, 99.83% TPR under adversarial attack), and Table 4 confirms consistency across DDIM, PNDM, and DPM-Solver++ samplers.

## Weaknesses

### Fatal
None.

### Major

- **Overclaimed theoretical guarantees — gap between 3rd-order moment matching and "computational indistinguishability":** The paper formally defines undetectability in Eq. 2 as computational indistinguishability (any PPT adversary has only negligible advantage). However, Theorems 3.1–3.2 and Lemmas 3.3–3.4 only establish that the watermarked distribution matches a standard Gaussian up to **third-order moments** via the spherical 3-design property. The paper acknowledges in Section 5 that "higher-order moments may deviate from the true prior," yet the introduction (line 17) states "we theoretically analyze each intermediate distribution and **prove** that the final noise is **statistically indistinguishable** from standard Gaussian noise," and the conclusion claims the watermarked inputs are "**provably** and empirically indistinguishable." The abstract is more careful ("theoretically prove... up to third-order moments, and empirically demonstrate... statistically indistinguishable"), but several high-visibility statements overclaim what the proofs actually establish. 3rd-order moment matching is evidence toward indistinguishability but does not constitute proof of it — a distinguisher operating on 4th-order statistics (e.g., kurtosis) could in principle detect the watermark. This gap between the formal definition and the proven result should be clearly stated throughout, not only in a brief acknowledgment in Section 5.

- **Misleading "encryption-free" framing:** The paper's title and contributions prominently claim the method is "encryption-free." While it correctly eliminates per-image key storage (unlike Gaussian Shading) and avoids cryptographic operations (unlike PRC Watermark's belief propagation), the method still relies on a **fixed secret signature** K = (T, C) that must remain confidential (Section 3.2: "K is kept fixed and secret during runtime to prevent unauthorized removal"). If an adversary obtains K, they can trivially extract and remove watermarks. The real advantage is eliminating *per-image key management* and *cryptographic computation*, not eliminating secrets entirely. The "encryption-free" label invites the interpretation that no secret parameters are needed, which is incorrect. A more precise term like "key-agnostic" or "fixed-signature" would better represent the contribution.

### Minor

- **Gaussian Shading comparison under fixed keys only:** Tables 1–2 compare against Gaussian Shading under fixed keys, which the paper explicitly acknowledges (Section 4.1) "no longer achieves true losslessness." While this is a fair comparison point (both systems under the same key management regime), it disadvantages Gaussian Shading by removing its core design mechanism. Including Gaussian Shading with per-image keys as a reference point for undetectability (even without direct robustness comparison) would more transparently isolate what the spherical mapping itself contributes.

- **No evaluation against key-recovery attacks:** The adversarial evaluation (Section 4.2) only covers WEvade (Jiang et al., 2023). Since Spherical Watermark relies on a fixed secret signature, security against key recovery (an adversary attempting to learn K from watermarked outputs) is an important practical concern. No formal threat model is specified, and no key-recovery experiments are conducted. This is a meaningful gap for a watermarking system making security claims.

- **Insufficient discussion of higher-order moment deviations:** The acknowledgment in Section 5 that "higher-order moments may deviate" is appropriate but understated. Given that 3rd-order moment matching is the theoretical foundation, a more thorough analysis — even bounding the maximum 4th-order deviation, or measuring kurtosis empirically — would substantially strengthen the paper's claims.

## Nice-to-Haves

- Evaluation on newer architectures (SDXL, FLUX) to validate the generalization claim beyond SD v1.5/v2.1
- Formal statistical tests for normality (e.g., Mardia's test, energy distance) beyond classifier-based detection
- Explicit specification of the threat model (adversary capabilities, query access assumptions)

## Removed Points

These points are flagged to be removed; treat them with caution:

- **Harsh Critic's claim that Eq. 4's extraction guarantee overmatches the actual pipeline because "DDIM inversion is approximate":** While true that DDIM inversion introduces approximation error, the high empirical extraction accuracy (99.99% clean ACC) shows this error is negligible in practice, and the formalization is an idealized bound that is standard in the field. Removed as a generic reproducibility concern.

- **Harsh Critic's claim about the theoretical analysis assuming m as random while in practice m is deterministic:** The paper's analysis of the distribution of z^(1) conditions on random padding r, and the 3-wise independence ensures each bit of z^(1) is individually Bernoulli(1/2) regardless of the message. The random-padding design specifically addresses this. Removed as it misunderstands the construction.

- **Harsh Critic's demand for evaluation on other diffusion architectures as a Major weakness:** Valid as a nice-to-have but not a core flaw — the method's design is architecture-agnostic and SD v1.5/v2.1 are standard benchmarks.

- **Harsh Critic's concern about the dimension-matching constraint (lx = N × lm + lr):** This is a standard design constraint, not a weakness. Removed as a nitpick.

- **Strength Finder's claim of "formal theoretical guarantees on distribution preservation" as a core strength claiming provable losslessness:** This conflicts with the verified Major weakness about overclaimed guarantees. Moved here — the proofs are real (3-wise independence, spherical 3-design) but do not constitute the stronger "provable losslessness" claimed.

- **Strength Finder's claim about "superior robustness under adversarial attacks" compared to PRC:** The difference (98.12% vs 97.69% ACC) is marginal and not conclusive evidence of superiority. Moved here as it overstates the empirical gap.

## Novel Insights

The interplay between the 3-wise independent code construction and spherical t-designs is genuinely novel for watermarking — the binary embedding module transforms a potentially correlated watermark bitstream into a 3-wise independent code, and the spherical mapping module exploits the fact that a set of 3-wise independent points on the unit sphere forms a spherical 3-design, providing distribution preservation up to 3rd-order moments. This yields a clean pipeline where each module's purpose is precisely specified and independently analyzable. However, the conceptual gap between 3rd-order moment matching and computational indistinguishability (which requires resistance to *all* polynomial-time distinguishers, including those operating on 4th+ order statistics) means the method's undetectability ultimately rests on empirical validation rather than proof proper. This gap could potentially be narrowed in future work by constructing spherical t-designs for higher t, or by proving that 3-wise independence combined with the chi-squared scaling provides convergence guarantees in high dimensions where Gaussian-like behavior dominates.

## Suggestions

- Revise all instances of "provably lossless" and "provably indistinguishable" to clearly state "provably preserves the prior up to third-order moments" and "empirically indistinguishable." The abstract's formulation is a good model; the conclusion should follow suit.
- Replace or clarify "encryption-free" to something like "without per-image key storage or cryptographic decoding" — this captures the real practical advantage without implying no secrets are needed.
- Add an explicit threat model section specifying adversary capabilities (access to generated images? embedding API? key?)
- Include Mardia's test or energy distance statistics to quantify higher-order distributional deviation from Gaussian.

## Score and Decision

Comparing against calibration anchors:
- **High anchors:** agHddsQhsL (avg 7.5, Accept Spotlight) — Targeted Attack paper with strong empirical results and genuine insight, but somewhat incremental; uzz3qAYy0D (avg 6.0, Accept Poster) — VideoShield, a novel watermarking framework with practical contributions.
- **Medium anchors:** aJl5aK9n7e (avg 5.25, Reject) — Novel theoretical framework for Graph Transformers but overclaimed guarantees under restrictive assumptions; wE5xp3zBaQ (avg 5.0, Reject) — Formalizing watermarks/attacks with overclaimed generality.
- **Low anchors:** O13fIFEB81 (avg 4.4, Reject) — Overclaimed watermarking framework with poor presentation; Hh0Cg4epYY (avg 2.33, Reject) — "Groundbreaking" claims with very weak theory.

Spherical Watermark sits above the medium and low anchors: it has a genuine novel construction, strong empirical results, and addresses a real practical problem. However, it sits below the high anchors because the overclaimed theoretical guarantees (presenting 3rd-order moment matching as proof of computational indistinguishability) and the misleading "encryption-free" framing are substantive issues. The paper is comparable to VideoShield (6.0) in terms of practical contribution and novelty, but with a more significant overclaiming problem that weakens its theoretical positioning. Given the calibrated anchors, I place this above medium-reject papers (which have more fundamental flaws) but below papers with clean theoretical-empirical alignment.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>