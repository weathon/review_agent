# Review of ../paper.md

Now let me search for calibration papers to score this properly.Now I have enough to write the final review. Let me produce it.

---

## Summary

FaceLinkGen challenges the prevailing evaluation paradigm in frequency-domain Privacy-Preserving Face Recognition (PPFR), which equates privacy with resistance to pixel-level reconstruction (PSNR/SSIM). The paper demonstrates that these metrics fail to capture identity-level leakage: a simple knowledge distillation pipeline that aligns protected templates to a standard ArcFace embedding space, combined with Arc2Face for face generation, achieves >98.5% matching accuracy and >96% regeneration success on three recent frequency-domain PPFR methods (PartialFace/ICCV 2023, MinusFace/CVPR 2024, FracFace/NeurIPS 2025). The paper argues for adopting identity-centric evaluation alongside pixel-distortion metrics in PPFR research.

---

## Claims and Support

**Claim 1: PSNR/SSIM do not adequately capture identity-level privacy in frequency-domain PPFR.**
*Well-supported for the three evaluated methods.* Table 1 and Figure 2 provide illustrative evidence of the decoupling; Tables 2–6 confirm that methods resistant to pixel reconstruction remain highly vulnerable to identity extraction. The abstract and conclusion are carefully scoped to "the evaluated methods," which is appropriate.

**Claim 2: FaceLinkGen achieves high-accuracy linkage on protected templates.**
*Well-supported.* Table 3 (1-to-1 LFW verification: 0.92–0.99) and Table 4 (closed-set 1-to-N recall consistently >70%, near the 88% dataset ceiling) together provide clean evidence under a proper train/test identity split.

**Claim 3: Face regeneration without pixel reconstruction is feasible from protected templates.**
*Partially supported.* Tables 2 and 5 show high pass rates under Face++ and Amazon APIs. The paper explicitly avoids circular evaluation (ArcFace not used for both extraction and verification). However, the metric measures third-party API acceptance, not universal identity recovery; Section 5.3 conflates these at points.

**Claim 4: Simplicity of the attack proves that vulnerability resides in the representation.**
*Partially supported, slightly overstated.* The minimal-resource results (256 images, 98.7% linkage) are compelling evidence of strong leakage, but "the vulnerability resides in the representation itself" is a causal claim that also depends on the smoothness of the PPFR mapping and alignment with the chosen teacher embedding. The paper does not fully disentangle these.

**Claim 5: As long as templates preserve recognition utility, they tend to retain recoverable identity information.**
*Partially supported as a conjecture; presented somewhat too broadly in Section 6's conclusion.* The evidence from three frequency-domain methods and two auxiliary pilots (TIP-IM, CanFG) is suggestive, but does not constitute a systematic demonstration across diverse PPFR paradigms.

**Claim 6: Constrained-assumption attack (no oracle access, 30 validation pairs only) also succeeds.**
*Partially supported, insufficiently specified.* Table 7 shows 92%+ matching and 94%+ Success@5 on Face++. But the selection of the Gaussian-blur proxy, its hyperparameters, and sensitivity to the 30 validation pairs is not reported. This weakens the reliability of this section's stronger threat-model claim.

**Claim 7: Soft biometrics remain recoverable from protected templates.**
*Suggestive but thin.* Table 9 shows race, gender, and age prediction close to original-image baselines for PartialFace and FracFace. However, the one-epoch MLP protocol, lack of error analysis, and absence of a direct template-feature baseline (bypassing the distilled embedding space) make this a preliminary rather than definitive finding.

---

## Strengths

- **Identifies a real and specific evaluation failure mode:** The paper demonstrates, on three ICCV/CVPR/NeurIPS papers, that pixel-level reconstruction resistance and identity privacy are decoupled—this is a concrete finding about the state of the field, not a generic observation. Table 6's direct comparison of FracFace's channel-disruption "protection" rate (1.000) against FaceLinkGen's near-zero resistance is particularly sharp.

- **Threat model is principled and well-motivated:** The paper correctly re-centers the PPFR threat model on the service provider as the primary adversary (insider threat), notes that prior work like Mi et al. [21] adopted a weaker external-attacker framing, and argues convincingly that oracle access follows naturally from model deployability. This is a cleaner framing than most prior attack work.

- **Attack simplicity strengthens the core critique:** Training completes in <2 hours at <$1.60 cost; 256 images suffice for 98.7% linkage on FracFace. The intentional minimalism of the pipeline is a scientific argument: if a baseline distillation breaks these systems, the problem is structural, not a failure requiring an adversarially strong attacker.

- **Cross-validated regeneration evaluation:** Using two independent commercial APIs (Face++ and Amazon) to assess regeneration success avoids circular evaluation (using ArcFace for both extraction and verification). The consistency between the two services (Tables 2 and 5) strengthens confidence in the regeneration results.

- **Linkage attack is technically clean:** The closed-set 1-to-N linkage experiment (Table 4) uses held-out identities with no overlap with training, attributes the ~88% original-image ceiling to documented WebFace label noise, and demonstrates cross-method linkage (template-to-template) in addition to face-to-template.

---

## Weaknesses

### Fatal
*None that would invalidate the core contribution.*

### Major

- **Constrained-assumption attack (Section 6) is under-specified for the claims it carries.** This section asserts that even an external attacker with no oracle access can mount a successful attack using a generic Gaussian-blur high-pass filter proxy. The results are strong (>92% matching, >94% Success@5), but the selection of blur parameters, the kernel range, and how the 30 validation pairs determine the final model are not reported. Since this section claims a substantially stronger threat (no oracle access) than the main attack, it requires correspondingly stronger validation. Without reporting sensitivity to proxy design and validation-set seed variance, it is impossible to tell whether this is a robust finding or a demonstration tuned to these three methods. The authors acknowledge the proxy "does not apply to non-frequency methods such as CanFG," further narrowing the claim—the text should not present this as broadly extending the vulnerability.

- **Regeneration metric not validated as a standalone privacy standard.** The paper rightly argues that PSNR/SSIM are insufficient. It then uses Face++/Amazon pass rates as the operative standard—but this metric depends on one generative model family (Arc2Face/SD1.5), one embedding space (Antelopev2/ArcFace), and third-party APIs with their own distributional biases toward synthetic faces. The linkage results (Tables 3–4) are direct and self-contained; the regeneration results are contingent on this pipeline. The paper should distinguish between "identity linkage is feasible" (strongly demonstrated) and "identity can be regenerated in a universally detectable way" (contingent on the particular Arc2Face/commercial-verifier stack). These are conflated in places, particularly in the abstract.

- **No statistical uncertainty reported despite very strong numerical claims.** Across all tables, results are single-run point estimates with no variance, confidence intervals, or multi-seed reporting. Given that the paper's core message is that prior evaluation standards are inadequate, it is ironic not to apply stronger evaluation standards to its own results. At least seed sensitivity for the main distillation training should be reported.

### Minor

- **Scope of "as long as utility is preserved" claim exceeds evidence.** Section 6's concluding paragraph reads: "as long as templates preserve recognition utility, they tend to retain recoverable identity information." This is presented as a general principle, but the empirical evidence is three frequency-domain methods plus two auxiliary pilots. Utility-preserving cryptographic template-protection systems are not evaluated, and the paper does not provide a theoretical argument. The phrasing should remain clearly at the "hypothesis suggested by current evidence" level throughout.

- **Soft biometric leakage analysis (Section 9) is preliminary and underspecified.** One epoch of MLP training, an evaluation on 10K images, and no confidence intervals or baselines that operate directly on templates (bypassing the distilled embedding) make it hard to draw strong conclusions. The claim that "soft biometrics can be inferred at nearly the same accuracy" understates that MinusFace race accuracy drops from 0.700 to 0.569—a meaningful gap. This section should be explicitly flagged as a pilot finding.

- **Minimal-resource experiment (Section 5.1) is conducted only on FracFace.** Since this section supports the "representation leaks identity structurally" argument, restricting it to one method narrows the supporting evidence. Reporting it for PartialFace and MinusFace would substantially strengthen this claim.

### Trivial

- The paper is submitted as ACM MM but is being reviewed at ICLR standards; no in-paper adjustment is needed, but ICLR reviewers will expect tighter empirical rigor than the current draft provides.

---

## Nice-to-Haves

- **Cross-backbone generalization:** Test the distillation with a non-ArcFace teacher (e.g., FaceNet, MagFace) to demonstrate that the vulnerability transfers across embedding families, separating the PPFR representation's leakiness from ArcFace's specific geometry.
- **Threshold-swept ROC curves:** Reporting full ROC/DET curves (with FPR axis) for the linkage attack would contextualize the >98.5% matching within a security-relevant operating range, rather than presenting only top-1 recall.
- **Failure case analysis for regeneration:** A gallery of regenerated faces that fail commercial verification, categorized by failure mode, would honestly communicate the practical boundary of the attack and prevent overclaiming the near-perfect success rates.
- **Demographic stratification:** Reporting attack success by demographic subgroup (gender, age, race) would reveal whether the claimed "universal" vulnerability is uniform or differentially worse for certain groups.
- **Proposed minimal countermeasure:** Briefly testing a simple defense (e.g., calibrated noise injection before template transmission) would clarify severity and demonstrate that the vulnerability is actionable rather than merely diagnostic.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**R1 – "Comparison to prior methods is unfair/not apples-to-apples" (Harsh Critic Critical Issue 4):** The paper explicitly addresses this in Section 5.3: "The evaluation of attack success in FracFace is based on a Protection (%) metric, defined as the proportion of frequency-domain channels that are filtered or structurally disrupted. This formulation establishes a lower barrier for defensive claims than our identity-centric standard." The paper is quite transparent that Table 6 compares different target quantities—it is precisely the mismatch that constitutes the critique. This is a feature, not a flaw.

**R2 – "The attack success depends on ArcFace invertibility, not PPFR weakness" (Harsh Critic):** The paper explicitly addresses this: "To rule out the dependence on models like Arc2Face or third-party verification services like Face++ or Amazon, we compared the similarity of the extracted embeddings with the original face. As detailed in Section 7, the embedding extracted from a protected template shows higher cosine similarity to its source image than to another image of the same person." The reviewer's concern is pre-addressed.

**R3 – Availability/existence of FracFace (NeurIPS 2025):** The paper cites it with a live OpenReview URL. Removed per hard rule.

**R4 – "Soft biometric section does not disentangle leakage from residual image content" (Human Finder Weakness 3):** The evaluation operates on protected *templates* (not original images), not residual image content. The MLP is appended to the distilled student embedding, which is trained on templates. The concern is partially misplaced—what the section doesn't disentangle is whether leakage comes through the distilled embedding specifically or could be retrieved by a simpler probe directly on raw templates. This is a legitimate remaining question but is weaker than the original phrasing suggests.

**R5 – "Lack of comparison with prior face reconstruction/inversion methods" (Human Finder Weakness 2):** The paper explicitly distinguishes its contribution from Model Inversion Attacks in Section 1 (paragraph beginning "Our approach differs from standard Model Inversion Attacks"). Since the paper's key claim is that the *objective* of reconstruction is wrong (not that a better reconstructor is needed), a direct comparison to reconstruction methods would conflate paradigms. This is scope-appropriate, not an omission.

**R6 – Generic strengths removed:** "The paper is well-written," "the topic is important," and similar generic positive statements have been removed in favor of specific evidence-backed strengths above.

---

## Novel Insights

The paper's most genuinely novel insight is methodological-diagnostic rather than algorithmic: the attack *deliberately* fails to improve on prior reconstruction attackers along their own axis (pixel fidelity) and instead reframes the attack objective entirely. The observation that reconstruction-centric red-teaming is "inadvertently trapped into pursuing the specific registration image as ground truth" while "this objective is often mathematically impossible due to information loss in protection" is well-articulated and identifies why prior red teams systematically underestimated leakage. The corollary—that a blurry, identity-inconsistent reconstructed image is a success signal for the PPFR designers but a non-event for a real attacker who can simply regenerate a fresh, verification-passing face—constitutes a meaningful reframing with direct implications for how the field should conduct red-teaming exercises.

---

## Suggestions

1. **Specify the constrained attack fully:** Report the Gaussian-blur kernel range, how parameters were selected, and sensitivity across 5+ random seeds for the 30 validation pairs. If robust, this substantially strengthens Section 6; if not, reduce claims accordingly.
2. **Add seed variance / confidence intervals** to the main distillation results (at minimum, report mean ± std across 3 training seeds for each method/dataset combination).
3. **Separate "linkage" from "regeneration" in the abstract's privacy claim.** The linkage results are the cleanest evidence; regeneration is compelling but contingent on a specific generative pipeline. Frame regeneration as "a practical privacy consequence of successful linkage," not as an independent metric of identity recovery.
4. **Broaden the minimal-resource experiment** to all three methods to shore up the "representation leakage is structural" argument.
5. **Report MinusFace soft biometric numbers with appropriate hedging** (race: 0.569 vs. 0.700 original is a visible gap) and add a direct template-space baseline to confirm the distilled embedding is doing the work.

---

## Score and Decision

**Calibration papers consulted:**

1. **"On the Inadequacy of Similarity-based Privacy Metrics"** (`g16vmAtJ8x.md`): Scores 5, 3, 6, 8, 8 → Rejected. Structurally very similar (critiques metric inadequacy, shows reconstruction attacks, argues for paradigm shift). Rejected partly due to narrow datasets and thin constrained-attack specifications—exactly the same weaknesses present here. However, that paper's MNIST/tabular setting is arguably more trivially attackable than faces; this paper's face domain is more challenging and the attack is cleaner. Suggests floor ~5–6.

2. **"KAN See Your Face"** (`razAcpFapu.md`): Scores 3, 5, 1, 3, 3 → Rejected. Also attacks PPFR with a distillation-style method but lacks the evaluation-paradigm framing, has insufficient baselines, and narrower results. This paper is clearly stronger. Not a ceiling but a lower anchor.

3. **"Revealing Unintentional Information Leakage in Low-Dimensional Facial Portrait Representations"** (`48CXLrx7K3.md`): Scores 5, 8, 3 → Rejected. Similar spirit—demonstrating information leakage from face encoders using generation. One reviewer gave 8 ("very interesting idea"), another gave 3. Variance reflects the divide in the community on how to value this type of diagnostic contribution.

**Assessment axis summary:**
- **Novelty:** Moderate-high for the evaluation-paradigm reframing; moderate for the method itself (straightforward distillation).
- **Technical soundness:** Solid for the linkage attack; weaker for the constrained attack (under-specified) and regeneration metric (contingent on specific pipeline).
- **Empirical support:** Strong in effect size on the primary evaluated systems; limited statistical rigor; single-run estimates.
- **Significance:** Meaningful for the PPFR subfield (three top-venue methods exposed); limited broader impact given narrow scope.
- **Clarity:** High on the core attack; lower on the constrained attack protocol.

Relative to the calibration set: this paper is meaningfully above "KAN See Your Face" (avg ~3) and roughly on par with the "Inadequacy of Similarity-based Privacy Metrics" paper (which was rejected ~avg 6 despite two high scores). The three frequency-domain methods attacked here are at higher-tier venues (ICCV, CVPR, NeurIPS) than typical targets, and the attack's deliberate minimalism is both a rhetorical and scientific contribution. However, at ICLR standards—which expect theoretical depth, statistical rigor, or broader scope for empirical systems papers—the constrained-attack under-specification and single-run evaluation are genuine problems.

**Final score: 5.5** — Marginally below the ICLR acceptance threshold. The core finding is genuine and the best parts are convincing, but the paper is missing the statistical rigor and scope that ICLR expects, and the secondary claims (constrained attack, universal utility-privacy principle) are not substantiated to the level they're presented.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>

**Predicted score: 5.5**
