
## Summary

This paper argues that the prevailing evaluation paradigm for frequency-domain privacy-preserving face recognition (PPFR) — which measures privacy as resistance to pixel-level reconstruction (PSNR/SSIM) — does not adequately capture identity-level leakage. The authors introduce FaceLinkGen, an identity-centric attack that uses knowledge distillation to extract identity embeddings from protected templates and a diffusion model (Arc2Face) to regenerate identity-consistent faces. The attack achieves near-perfect 1-to-1 verification accuracy and above 96% regeneration success (at 5 trials) on three recent frequency-domain PPFR systems (PartialFace, MinusFace, FracFace), demonstrating a significant gap between pixel-level metrics and actual identity privacy.

## Strengths

1. **Important conceptual insight**: The paper's central argument — that pixel-level distortion is a poor proxy for identity leakage — is well-articulated and convincingly validated. Table 1 effectively illustrates that high SSIM/PSNR can coincide with low identity similarity (CanFG face, FS=0.008) while low SSIM/PSNR can accompany high identity similarity (same person, different photos, FS=0.586). This is the paper's core contribution and it is genuinely impactful.

2. **Principled threat model**: Section 2 provides a well-reasoned defense of the insider/service-provider adversary model, grounded in the original PPFR design intent [8]. The argument that oracle access is realistic (since service providers can reverse-engineer local clients or submit arbitrary faces) is convincing and carefully argued against weaker external-attacker framings.

3. **Intentional simplicity of attack methodology**: The paper deliberately uses a standard distillation pipeline (cosine similarity loss + Arc2Face generation) rather than complex adversarial optimization. This simplicity strengthens the main claim: if a straightforward, low-cost method (<2 hours, ~$1.60) suffices, the vulnerability is in the representation, not the attack sophistication.

4. **Comprehensive evaluation scope**: The study moves beyond attacking just three PPFR methods to include minimal-resource experiments (256/800 images), constrained-attacker scenarios (no oracle access, generic high-pass filter proxy), extensions to de-identification (TIP-IM) and non-frequency methods (CanFG), and soft biometric leakage analysis. Cross-verification via two commercial APIs (Face++, Amazon) is a sensible effort to avoid circular evaluation.

5. **Strong empirical results on the core claim**: The regeneration success rates (≥97% at Success@5 across all methods/datasets) and 1-to-1 verification accuracies (≥92% across all methods) are striking. The minimal-resource experiment (97% generation pass rate at FAR 1e-5 with only 800 images) further reinforces that the vulnerability is robust and accessible.

## Weaknesses

### Major

1. **The abstract's headline numbers overgeneralize from 1-to-1 verification to broader "matching accuracy"**: The abstract claims "over 98.5% matching accuracy," but this corresponds to 1-to-1 verification (Table 3: 0.92–0.99), a much simpler task than database linkage. The harder and more privacy-relevant 1-to-N closed-set linkage (Table 4) yields 0.72–0.86, substantially lower. The paper uses "matching," "verification," and "linkage" somewhat interchangeably, which materially inflates the apparent strength of the attack. Since the paper's conceptual contribution is about *identity-level leakage* (not merely 1-to-1 matching), the linkage numbers are the more relevant metric and should be foregrounded honestly. The paper states linkage results are "essentially reaching the dataset's theoretical maximum performance" due to noise, but this claim is not rigorously substantiated — an upper-bound analysis with noise-corrected ground truth would be needed.

2. **No direct comparison of reconstruction-centric attacks under identity-centric metrics**: The paper's central argument is that pixel-level attacks fail where identity-level attacks succeed, but it never measures whether the outputs of existing U-Net/StyleGAN reconstruction attacks (used by the attacked PPFR methods themselves) actually pass Face++ verification. This is a missing link: showing that blurry U-Net outputs fail Face++ while FaceLinkGen outputs pass it would make the paper's central comparison empirically self-evident rather than merely asserted. Table 6 compares FracFace's channel-disruption metric against FaceLinkGen's identity-level success, but these measure fundamentally different things; a fairer comparison would run the *same identity metric* on both attack outputs.

3. **Evaluation relies heavily on the ArcFace ecosystem**: The entire pipeline — teacher model (Antelopev2/ArcFace), generator (Arc2Face), and implicitly the commercial verifiers (Face++, likely ArcFace-family) — is anchored in one embedding ecosystem. The paper acknowledges this but provides no quantification of how much results degrade when breaking this alignment. Using FaceNet-compatible generation and non-ArcFace verification in at least one ablation would clarify whether the attack exploits a general vulnerability of utility-preserving templates or an ecosystem-specific coherence. The Amazon API results (Table 7: 45–57% vs. Face++ 94–96% in constrained setting) hint that the answer may depend on the verifier, but this is not systematically analyzed.

### Minor

4. **Overgeneralization beyond frequency-domain PPFR**: The core experiments compellingly demonstrate identity leakage in three frequency-domain PPFR methods, but the paper's rhetoric often extends to "current protection mechanisms" and even "structural limitations" of de-identification. The TIP-IM and CanFG analyses (Section 8, Table 8) are explicitly pilot-scale (small subsets, no regeneration results for TIP-IM), yet the discussion treats them as confirming a general structural limitation. These should be more carefully framed as preliminary evidence for potential broader applicability rather than established conclusions.

5. **Minimal-resource and constrained-attack experiments lack variance reporting**: With only 256 or 800 training images, very high performance is reported but without variance estimates or multi-seed results. Given the extreme resource constraints, demonstrating stability across random subsamples would significantly strengthen these claims.

6. **The constrained-attack section (Section 6) has ad hoc design choices**: The generic high-pass filter proxy is manually engineered from visual observation of template properties. No ablation tests what happens when the attacker guesses wrong about the high-pass characteristic, making it unclear how much the constrained attack depends on correctly identifying this property.

### Trivial

7. The soft biometric leakage analysis (Section 9, Table 9) is suggestive but thinly specified (one epoch on FairFace, no direct-from-template baseline without distillation). This is supporting evidence rather than a decisive result.

## Nice-to-Haves

- Running reconstruction-centric attacks (U-Net, StyleGAN as used by FracFace's own evaluation) through Face++ verification to directly quantify the gap between pixel-level and identity-level evaluation on the same benchmark.
- Testing simple template-level defenses (Gaussian noise addition, quantization, embedding space randomization) against FaceLinkGen to determine whether the vulnerability requires representational redesign or merely hardened deployment.
- At least one ablation with a fully non-ArcFace verification/generation pipeline to quantify ecosystem dependence.
- Failure case analysis characterizing the 1–10% of failures (demographic patterns, pose/quality factors).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Limited technical novelty of the attack pipeline"** (Human Finder): While true that distillation + Arc2Face is not individually novel, this is the paper's explicit design choice — the simplicity is the point. The paper's contribution is the *evaluation paradigm critique*, not the attack method's originality. This criticism misidentifies the paper's primary contribution.

- **"Missing comparison with prior attack methods"** (Human Finder citing KAN See Your Face reviews): The paper does compare (Table 6) with FracFace's own evaluation protocol. The relevant comparison is not between FaceLinkGen and prior attacks' success rates on pixel-level metrics (which would be apples-to-oranges), but rather running *both* attack types under *identity-centric* metrics — which is a valid request (see Major weakness 2) but the phrasing "no systematic comparison with prior methods" misunderstands the paper's focus on metric evaluation rather than attack method competition.

- **"No evaluation against cryptographic or key-based PPFR"** (Spark): This is requesting the paper attack methods it explicitly identifies as potentially robust alternatives. The paper clearly scopes its focus to frequency-domain PPFR and acknowledges cryptographic methods as a different, stronger defense category. This is scope creep.

- **"Lack of immediate defensive solutions"** (Neutral Reviewer): The paper is an attack/evaluation paper, not a defense paper. Suggesting future directions (as done in Section 10) is appropriate; demanding immediate countermeasures is outside the scope of a vulnerability demonstration.

- **"Ethical/dual-use concerns"** (Neutral Reviewer): While always worth noting, this is a standard consideration for security/privacy attack papers and not a weakness specific to this work.

- **"Dependency on generative model quality"** (Neutral Reviewer): The paper already reports linkage accuracy independently from regeneration success, effectively dissociating identity extraction from identity rendering. This concern is partially addressed.

- **"Demographic bias and generalization"** (Human Finder): This is a valid general concern but is not raised as a weakness by any reviewer based on specific evidence in the paper. Demographic bias analysis would be an extension, not a core flaw in the current claims.

## Novel Insights

The most significant insight emerging from the reviews is the fundamental tension between utility and privacy in PPFR: any representation that preserves enough identity information for matching inherently preserves enough for extraction. The paper's argument that this creates a structural vulnerability — not just an implementation weakness in specific methods — is its most provocative claim. If true, it suggests that the entire paradigm of "obfuscate-but-preserve-utility" (as opposed to cryptographic or key-based approaches) may be inherently insufficient for identity privacy, which would be a foundational challenge to the frequency-domain PPFR research direction.

## Suggestions

1. **Rewrite the abstract to accurately represent both 1-to-1 verification and 1-to-N linkage results**, or clearly specify that "matching accuracy" refers to 1-to-1 verification specifically, with linkage results discussed separately.
2. **Run at least one prior reconstruction attack (U-Net or StyleGAN) through Face++/Amazon API** with the same identity-centric evaluation to directly demonstrate the gap in metrics on the same benchmark — this would turn the paper's central claim from well-argued assertion into direct empirical evidence.
3. **Add one ablation using a non-ArcFace verification system** (e.g., a FaceNet-compatible generator verified by a different FR model) to bound the ecosystem-dependence of results, even if preliminary.
4. **Frame TIP-IM and CanFG results explicitly as "pilot" or "preliminary case studies"** in the conclusion, not as general confirmations of structural limitations.

## Score and Decision

**Calibration references:**
- *KAN See Your Face* (PPFR attack, rejected, avg ~3): Similar domain but weaker — no conceptual insight about evaluation metrics, no comparison, white-box-only. Our paper is substantially stronger.
- *On the Inadequacy of Similarity-based Privacy Metrics* (privacy metric critique + attack, scores 5,3,6,8,8, avg ~6): Most directly comparable — identifies metric inadequacy and demonstrates with attack. That paper had more conceptual/theoretical contribution but weaker empirical demonstration. Our paper has much stronger empirical results but overclaims somewhat.
- *Poisoned Forgery Face* (face attack, accepted spotlight, scores 6,8,8): Strong attack paper with novel threat model. Our paper has comparable importance but is somewhat weaker on experimental completeness (no comparison with prior attacks on same metrics) and has the overclaiming issue.
- *Diff-Privacy* (face privacy protection, rejected, scores 8,6,8,5): Defense paper, different category.

This paper has a genuinely important core contribution (the evaluation paradigm critique is impactful and timely) with strong empirical results, but the overclaiming in the abstract and the missing direct comparison with prior attacks on identity-level metrics are substantive weaknesses. The paper is clearly above the quality of the rejected PPFR attack papers (KAN See Your Face, Revealing Unintentional Information Leakage) but below the level of a clean accept due to the overgeneralization and evaluation gaps. The core finding — that three recent top-venue PPFR systems fail identity-level privacy under a realistic threat model — is well-supported and significant.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>

**Predicted score: 6.0**
