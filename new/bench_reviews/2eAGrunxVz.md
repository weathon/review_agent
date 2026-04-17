The paper content was provided in the prompt. Let me verify key claims from the paper text directly.

Looking at the paper content provided:
- The paper claims "encryption-free" and "omits key storage overhead" in contribution 3
- The Signature K = {T, C} is stated as "kept fixed and secret during runtime to prevent unauthorized removal" (§3.2)
- The paper claims "statistically indistinguishable from a standard multivariate normal distribution" (abstract)
- Section 5 acknowledges: "Our Gaussian-noise guarantee depends on spherical 3-design definition. While watermarked and random noise are empirically indistinguishable, higher-order moments may deviate from the true prior."
- The formal definition uses PPT adversary and negligible in ρ language
- Gaussian Shading is tested with fixed keys (acknowledged to break losslessness)

Now let me synthesize the final review.

## Summary

Spherical Watermark proposes a lossless, encryption-free watermarking framework for diffusion models that embeds binary watermark bits into latent Gaussian noise via a binary embedding module, spherical mapping (projecting to unit sphere, rotating, scaling by chi-square radius), and diffusion integration. The authors prove that the watermarked noise matches the target Gaussian prior up to third-order moments and empirically demonstrate undetectability, strong robustness, and significant computational efficiency gains over cryptographic baselines like PRC Watermark.

## Strengths

- **Novel and elegant mathematical construction**: The use of spherical 3-designs to approximate the uniform distribution on the sphere from binary sequences, followed by chi-square radius scaling to recover Gaussian-like noise, is a creative and mathematically principled approach. Theorem 3.2 establishing the 3-design property from 3-wise independence is a non-trivial and interesting theoretical contribution.

- **Strong computational efficiency**: The extraction is approximately four orders of magnitude faster than PRC Watermark's belief-propagation decoding (Figure 4), which is a significant practical advantage for API-scale deployment.

- **Solid empirical undetectability and robustness**: Table 2 shows competitive or superior tracing accuracy (ACC 95-99% under post-processing, TPR 97-99.8% under adversarial attacks) compared to all baselines. The FID is essentially identical to the unwatermarked baseline (Table 1), and classifier-based detection tests show near-chance accuracy (Figure 2), supporting the claim that the watermarked distribution closely matches the Gaussian prior.

- **Well-designed ablation studies**: Figure 6 and Tables 3-5 provide informative ablations on modules (B and S), parameters (N, s), and diffusion sampling settings, convincingly demonstrating the necessity of each component.

- **Simplified key management**: Compared to Gaussian Shading's per-image nonce/key requirement, using a fixed global signature K = {T, C} with random padding r for per-image variation is a meaningful practical simplification, even if it does not eliminate key management entirely (see Weaknesses).

## Weaknesses

### Major

- **Overclaiming "encryption-free" and "no key storage overhead"**: The paper prominently claims "encryption-free" (title, abstract, §1, §2) and that the method "omits key storage overhead" (contribution 3, §1). However, the Signature K = {T, C} is explicitly described as secret: *"K is kept fixed and secret during runtime to prevent unauthorized removal"* (§3.2). The matrix T involves random permutations in its construction (Algorithm 1), and C is drawn from a random Gaussian matrix. Together they constitute a high-dimensional secret linear transform—functionally a global secret key. If K is compromised, any adversary can trivially detect or remove watermarks. While the design eliminates the need for *per-image* keys, it does not eliminate key management; it consolidates it into a single global key. This is an important practical simplification but is fundamentally different from being "encryption-free." This overclaim mischaracterizes the comparison to Gaussian Shading and PRC Watermark—those methods use encryption per-image; this method replaces it with a fixed secret key, not removes it.

- **Overclaiming "lossless" and "statistically indistinguishable" beyond what the theory guarantees**: The formal definitions in §3.1 use strong cryptographic language—PPT adversaries, negligible in a security parameter ρ, and "exact extraction." Yet: (1) The theoretical analysis proves only that the watermarked distribution preserves the target prior *up to third-order moments* (spherical 3-design). The paper itself acknowledges in §5: *"While watermarked and random noise are empirically indistinguishable, higher-order moments may deviate from the true prior."* This directly contradicts the abstract's claim of being *"statistically indistinguishable from a standard multivariate normal distribution."* (2) The security parameter ρ is never defined, making the "negligible in ρ" guarantees unfalsifiable. (3) Extraction accuracy is not 100% even in clean PNG conditions (Table 2 shows 99.99% ACC and <100% TPR), and degrades to ~95% ACC under adversarial attacks—these are good engineering numbers but are not "negligible error" in any formal cryptographic sense. The scheme is more accurately described as a *high-accuracy, redundancy-based robust watermark* with moment-matching guarantees up to degree 3, rather than a cryptographically lossless one.

- **Limited undetectability evaluation**: The undetectability claim (Eq. 2, computational indistinguishability against any PPT adversary) is supported only by tests with a 2-layer MLP (latent-level) and ResNet-18 (image-level). As reviewers of comparable papers (PRC Watermark) have noted, such limited classifier evaluations do not establish undetectability against more sophisticated classifiers or statistical tests. No higher-order statistical tests (e.g., MMD with polynomial kernels of degree ≥4, Kolmogorov–Smirnov tests) are performed. Given that the theoretical guarantee stops at 3rd-order moments, this is a significant gap.

### Minor

- **Unfair comparison with Gaussian Shading**: The paper compares against Gaussian Shading with *fixed keys*, noting: *"Note that with fixed keys, Gaussian Shading no longer achieves true losslessness"* (§4.1). This creates an asymmetric comparison: Spherical Watermark is evaluated in its intended (and optimal) configuration, while Gaussian Shading is evaluated in a deliberately degraded configuration that removes its core theoretical guarantee. This makes the undetectability and robustness comparisons against Gaussian Shading uninformative. While using fixed keys may be motivated by the practical overhead of per-image keys, the resulting comparison conflates Gaussian Shading's architectural limitations with the authors' choice of evaluation configuration.

- **Limited attack evaluation**: Robustness is tested only against WEvade-based adversarial attacks and standard post-processing distortions. Recent work (e.g., adaptive attacks via optimization, surrogate key attacks, regeneration-based attacks like CtrlRegen) is not evaluated. These attacks represent realistic threats to latent-space watermarking schemes and could significantly affect the claimed robustness advantage.

- **Memory footprint of the signature**: The orthogonal rotation matrix C requires storing l_C × l_C values. For the default l_x = 16384, this could be ~1 GB in float32 if l_C = l_x. Footnote 1 mentions using l_C as a factor of l_x (e.g., ⌊√l_x⌋), but no experiments report the actual configuration used or the trade-offs on robustness and indistinguishability when using smaller l_C.

- **Low effective code rate**: With N=31 repetitions and l_r=512 padding bits for a 512-bit payload, the total latent dimension is 16384, giving an effective code rate of ~3.1%. This means the capacity for metadata is limited relative to the available channel.

### Nice-to-Haves

- Evaluation on newer architectures (SDXL, Flux) to validate the claim of generalization to any generative model with a Gaussian prior and invertible mapping.
- Formal statistical tests beyond classifiers (e.g., MMD with degree-4+ kernels) to empirically probe higher-order moment deviations claimed as a limitation.
- Analysis of signature recovery attacks: how many watermarked samples would an adversary need to estimate T and C?
- Spoofing attack evaluation where an adversary attempts to forge watermarks on non-generated images.
- Rounding tolerance analysis quantifying how much inversion error the scheme can absorb before bit flips occur.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"Not yet released" claims about baselines**: The paper references PRC Watermark, Gaussian Shading, Tree-Ring, and other methods that are assumed to exist as cited. Any concerns about their availability are reviewer knowledge gaps, not paper errors.

- **Demanding comparison with Gaussian Shading under per-image keys IS valid** (kept above), but the harsh critic's framing that this is "one-sided" in Spherical Watermark's *favor* is incorrect. As stated in the paper, fixed keys *break* Gaussian Shading's losslessness, which actually *disadvantages* Gaussian Shading, not Spherical Watermark. The comparison is unfavorable to Gaussian Shading. This is kept as a weakness but reframed correctly.

- **Formatting nitpicks**: The harsh critic noted equation duplication/garbling, but the paper text states this is likely a parser issue, not a paper problem. Removed.

- **Demanding confidence intervals for TPR**: Single-run or 5-run mean±std evaluation is standard practice in watermarking literature. Removed as nitpick.

- **Demanding total system-level wall-clock time**: The extraction time comparison specifically isolates the embedding/extraction transformation, which is the fair comparison for architectural overhead. Removed as scope creep.

## Novel Insights

The paper's insight that binary sequences with 3-wise independence can be mapped to spherical 3-designs—which then approximate the Gaussian distribution sufficiently for latent space watermarking—is genuinely novel in this domain. The decomposition of the Gaussian prior construction into a discrete binary mixing step followed by a continuous spherical projection is conceptually cleaner than the stream cipher (Gaussian Shading) or error-correcting code (PRC) approaches. However, this same insight reveals a fundamental tension: the 3-wise independence is achieved by mixing fixed message bits with random padding, which means extraction relies on majority vote (a redundancy-based code with no error-correction theory). The method works well empirically because N=31 repetitions provide adequate error tolerance, but the gap between the theoretical moment-matching framework and the practical majority-vote decoder highlights that the theoretical guarantees serve undetectability, not robustness—the latter is achieved by sheer redundancy.

## Suggestions

- **Revise the framing**: Replace "encryption-free" and "omits key storage overhead" with "global secret key" and "eliminates per-image key management." Replace "lossless" with "near-lossless" or "distribution-preserving up to third-order moments." Replace "statistically indistinguishable" with "indistinguishable under low-order statistical tests." These changes align the claims with what the theory and experiments actually demonstrate, without diminishing the practical contribution.
- **Add higher-order statistical tests**: Include MMD with polynomial kernels of degree 4+, Kolmogorov–Smirnov tests, or similar to empirically characterize where the moment-matching guarantee breaks down.
- **Test against adaptive/surrogate-key attacks**: Given that the scheme's security rests on the secrecy of K = {T, C}, evaluate whether an adversary who collects many watermarked latents can estimate K.

## Score and Decision

**Calibration**: The most directly comparable paper is PRC Watermark (ICLR 2025 poster, scores 6/6/6/8, mean ~6.5). PRC Watermark had similar concerns: loose theoretical bounds on undetectability, limited undetectability evaluation (only ResNet-18), and weaker robustness than some baselines, but still contributed a meaningful cryptographic approach to lossless watermarking. WIND (ICLR 2025 poster, scores 6/5/6/6/6/6, mean ~5.8) had novelty concerns and limited evaluation but offered practical robustness improvements. SuperMark (rejected, scores ~3-5) was much weaker, with poor novelty and unconvincing experiments.

Spherical Watermark has genuinely novel insights (spherical 3-design, binary embedding), strong empirical results, and a meaningful practical advantage (4 orders of magnitude faster extraction). However, its overclaiming is more severe than PRC Watermark's: the "encryption-free" and "lossless/statistically indistinguishable" claims are central to the paper's framing and are contradicted by the actual construction and theory. PRC Watermark's overclaim was primarily about the looseness of its undetectability bound; Spherical Watermark's overclaim extends to the fundamental nature of the scheme (claiming no key when one exists, claiming statistical indistinguishability when the theory only gives 3rd-order moments). This is a significant gap between marketing and delivery. That said, the underlying engineering contribution is solid—the scheme works well and the mathematical insight is real.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>

The paper makes a meaningful practical and theoretical contribution (spherical 3-design for watermarking, efficient extraction), but the core claims are overextended in ways that misrepresent what the method actually delivers. "Encryption-free" mischaracterizes a global secret key scheme, and "lossless/statistically indistinguishable" overclaims what spherical 3-designs guarantee. These are not minor presentation issues—they define how the method is positioned relative to its competitors and shape the reader's understanding of its properties. A revised version with corrected claims would be a solid 6.5-7, but the current framing creates a misleading impression of the contribution.