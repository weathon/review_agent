# Review of facelinkgen_icml.md

Now let me search for calibration papers to properly score this work.Now I have enough information to write the consolidated review. Let me synthesize all reviewer inputs against what the paper actually says.

---

## Summary

FaceLinkGen presents an identity extraction attack against transformation-based privacy-preserving face recognition (PPFR) systems, demonstrating that three recent frequency-domain obfuscation methods (PartialFace, MinusFace, FracFace) leak identity information despite high pixel-level distortion. The attack uses knowledge distillation to align protected templates with standard ArcFace embeddings, then uses Arc2Face for face regeneration, achieving >98.5% matching accuracy and >96% regeneration success. The paper's primary contribution is a paradigm critique: that pixel-level metrics (PSNR/SSIM) are structurally inadequate as privacy proxies for PPFR evaluation.

---

## Strengths

- **Genuine and important paradigm critique.** Figure 2 and Table 1 provide concrete, compelling evidence that pixel-level similarity and identity-level similarity are uncorrelated. The point that PPFR systems optimized against PSNR/SSIM are evaluated by a metric that does not capture the realistic threat (identity extraction) is timely, correct, and actionable.

- **Linkage attack results are strong and cleanly evaluated.** Table 4 shows top-1 closed-set recall between all pairs of protected templates and original images, ranging from ~72%–87%, closely approaching the original-image-to-original-image upper bound (~88%). Crucially, the paper attributes the ~12% gap to 9–13% label noise in CASIA-WebFace (Wang et al., 2018), suggesting nearly ceiling-level performance. This evidence is unambiguous: the templates function as effective identity descriptors.

- **Intentional simplicity of the attack strengthens the point.** The attack—knowledge distillation with a shallow student model + Arc2Face generation—is a standard pipeline costing ~$0.80–1.60 for the full distillation. That even this lightweight method succeeds is a stronger indictment of these systems than a complex tailored attack would be.

- **Multi-dataset, multi-API evaluation provides meaningful triangulation.** Testing regeneration on CASIA hold-out, LFW, and synthetic TPDNE, and cross-validating with both Face++ and Amazon APIs, substantially reduces concerns about evaluation circularity.

- **Near-zero-knowledge experiment (Section 6) is an interesting and impactful result.** Showing that a generic high-pass filter proxy suffices to train a working student model is a practically important finding—it implies that even without oracle access to the conversion process, identity extraction remains feasible.

- **The 1-to-1 verification results (Table 3) are impressive.** Template-to-face verification on LFW reaches 98.8–99.2% accuracy, virtually matching the 99.8% face-to-face upper bound. This is quantitatively damning for the attacked systems.

---

## Weaknesses

### Fatal
*None.*

### Major

- **All three attacked methods share the same design family, but the paper frames its conclusions as applying to "visual distortion" and "conversion-based defenses" broadly.** PartialFace, MinusFace, and FracFace all come from the same research group (Mi et al.) and all rely on frequency-domain channel manipulation that visibly preserves high-frequency identity-carrying content. The abstract says "visual obfuscation leaves identity information broadly exposed" and the introduction says "protection by visual distortion is inherently insufficient." These are categorical claims. Section 6 appropriately hedges: *"Whether this generalizes to future methods that do not rely on frequency-domain obfuscation remains an open question."* However, the abstract, introduction, and conclusion use systematically broader framing than the experiments support. This gap between the headline framing and the actual evidentiary scope is the paper's most consequential weakness. The experiments support a strong, important claim about *this family* of frequency-domain obfuscation methods; that claim does not transfer to PPFR designs based on cryptographic commitments, key-dependent transforms, or learned non-frequency-domain protection.

- **The Amazon API results in the near-zero-knowledge setting are dramatically weaker than Face++ and receive no substantive explanation.** Table 7 shows Face++ regeneration success@5 of 94.6–96.3% but Amazon success of only 44.7–57.0%—a gap of roughly 40–50 percentage points. The paper's single-sentence explanation ("likely more strict or sensitive to AI-generated images") is insufficient given that the near-zero-knowledge result is one of the paper's headline claims. This discrepancy could reflect: different identity thresholds, AI-image detection, distributional differences in what constitutes a verification match, or genuine variation in identity preservation strength. Without resolving this, it is unclear which API better reflects the real-world privacy risk. The 44–57% success rate under Amazon is a much weaker result and qualifies the near-zero-knowledge claim significantly.

### Minor

- **The soft biometrics section (Section 8) lacks a proper baseline, weakening its evidential value.** The paper trains MLP models to predict age/gender/race from extracted embeddings, but acknowledges in the same paragraph that "ArcFace embeddings retain such information" (citing Melzi et al., 2023; Osorio-Roig et al., 2023). Without showing that (a) the same predictions fail on *protected* templates without the attack, or (b) comparing prediction accuracy from the attacked embedding vs. original ArcFace embeddings, the section cannot isolate whether leakage is specific to the attack or simply inherited from ArcFace's known behavior. As written, this section adds supporting color but not independent evidence.

- **The "near-zero-knowledge" framing in Section 6 is somewhat misleading.** The method exploits a visually observable and explicit common characteristic of the three attacked methods (high-frequency preservation, low-frequency suppression). The paper states this directly: *"they all preserve high-frequency information while obfuscating low-frequency information... Based on this intuition, the attacker can bypass any system-specific modeling."* An attacker who knows this is not operating under "near-zero knowledge"—they have made a specific structural observation about the method family. The framing should be corrected to "common-property" or "family-level" knowledge rather than "near-zero-knowledge."

- **Technical novelty of the attack is limited.** Knowledge distillation to align template representations with ArcFace, followed by Arc2Face generation, is a straightforward combination of existing techniques. The paper argues that simplicity is intentional, which is methodologically defensible, but the contribution is largely conceptual rather than technical. This limits the paper's appeal at a venue that traditionally weighs technical novelty highly.

### Trivial

- **Equation (1)'s formalism is more notational than probative.** Writing $T \sim p(\cdot \mid z_I)$ effectively *defines* the template as identity-preserving rather than demonstrating it empirically. This may mislead readers into thinking the framing is more rigorous than it is, though the empirical results carry the argument in any case.

---

## Nice-to-Haves

- **Test at least one method from a meaningfully different design paradigm.** Even a brief experiment on a key-based or cryptographic method (CryptoFace is cited in Section 9) would provide crucial context for the scope of the claims. A null result would appropriately bound the finding; a positive result would substantially strengthen it.

- **Resolve the Face++ vs. Amazon discrepancy rigorously.** Controlled experiments varying the threshold, dataset, and generation model would clarify whether the gap is methodological or reflects genuine differences in identity preservation under the near-zero-knowledge attack.

- **Provide a baseline for the soft biometrics section.** Show predictions from the *original* ArcFace embeddings on the same test set, and optionally from the raw protected templates (before the distillation attack), to establish what the attack actually adds beyond inherent ArcFace behavior.

- **Ablation on teacher model.** Testing a different teacher/generator pair (e.g., FaceNet + compatible generator) would confirm the method's model-agnostic property claimed in Section 3 and distinguish whether success depends on using ArcFace-adjacent tooling on ArcFace-compatible systems.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**Harsh Critic – Point 3 (withholding implementation details):** The paper explicitly withholds attack code and trained models citing safety concerns, following precedents in high-stakes AI papers. Per hard review rules, reproducibility concerns about withholding large artifacts or implementation details under documented safety justifications are not valid weaknesses. The paper provides high-level algorithmic descriptions sufficient for conceptual replication.

**Human Finder – Missing comparison with prior reconstruction attacks:** The paper directly addresses this in Figure 3, showing that U-Net and StyleGAN attacks fail even when evaluated under the identity metric. This is the paper's thesis—prior attacks were designed for and evaluated on pixel-level objectives. Demanding a comparison on prior methods' terms would conflate what the paper is arguing *against*.

**Human Finder – Missing cancellable biometrics analysis:** The paper explicitly scopes out anonymization/cancelable biometric schemes in Section 1 with reasoned justification (different objectives—PPFR must preserve recognition utility, which cancelable schemes often do not in the same way). Criticizing the absence of methods outside this stated scope is scope creep.

**Human Finder / Spark – Request for perceptual user study:** Commercial API verification at FAR thresholds of $10^{-3}$–$10^{-5}$ is a more stringent and less subjective identity verification standard than a user study, and is standard in this literature. This is not a methodological gap.

**Spark – No comparison with simple baselines (nearest-neighbor in raw template space):** This would be interesting but the paper's thesis is that even a simple distillation suffices—not that its specific neural student is optimal. Demanding simpler baselines to falsify the neural student's necessity misreads the paper's claims.

**Harsh Critic – Regeneration evaluation conflates source-image mimicry with identity leakage:** The paper acknowledges in Section 7 that "the template is a better identity descriptor for this specific image than another image of the same person." Rather than undermining the regeneration claim, this shows the template preserves *more* than generic identity—it preserves instance-specific identity. The paper's privacy claim is that the template reveals the user's identity; whether it reveals generic identity or instance-specific identity, both are violations. The Table 5 Amazon cross-validation on LFW further supports the regeneration results.

---

## Novel Insights

The paper's most underappreciated contribution may be the *evaluation diagnosis* rather than the *attack itself*. By showing (Table 6) that FracFace's own claimed protection metric (frequency channel disruption rate) reads as 100% while FaceLinkGen simultaneously achieves ~99% identity recovery, the paper provides one of the clearest demonstrations to date that proxy metrics in security can be simultaneously maximized while the actual security property fails entirely. This reframes the paper's contribution: rather than primarily an attack on three systems, it is a demonstration of systematic evaluation gaming—systems are optimized to score well on the stated metric, which happens to be orthogonal to the actual threat. This lesson is broadly applicable beyond PPFR and is more impactful than the attack pipeline itself.

---

## Suggestions

1. **Retitle or reframe the abstract and introduction claims** to be scoped to "frequency-domain transformation-based PPFR" rather than "visual distortion" or "conversion-based defenses" broadly. This does not weaken the paper—the result is still important and embarrassing for these systems.

2. **Investigate the Amazon vs. Face++ discrepancy in Table 7 more carefully.** Explore whether it stems from AI-image detection, threshold calibration, or genuine identity representation weakness in the near-zero-knowledge setting. This is essential for correctly characterizing the near-zero-knowledge threat level.

3. **Rename Section 6 to something like "Family-Level Transfer Attack"** rather than "near-zero-knowledge," which overstates the constraint on the attacker.

4. **Add an ArcFace baseline to Table 8** (soft biometrics). This would show whether the attack adds demographic leakage or inherits it from the ArcFace embedding.

5. **Explicitly scope the conclusion** to frequency-domain methods. The current conclusion (Section 10) opens: "This paper demonstrates that current frequency-based obfuscation methods fail..."—this is well-scoped. But then it escalates to "the visual distortion paradigm for external attackers." Keeping the conclusion consistently scoped would make the paper stronger and more credible.

---

## Score and Decision

**Calibration:**

- *KAN See Your Face* (razAcpFapu.md): Scores 3,5,1,3,3 → Rejected. Much weaker than FaceLinkGen: no paradigm contribution, no compelling results, naive attack. FaceLinkGen is clearly above this.
- *Revealing Unintentional Information Leakage* (48CXLrx7K3.md): Scores 5,8,3 → Rejected. Similar spirit—demonstrates privacy leakage from face templates—but on a narrower setting (32/40-element vectors). FaceLinkGen is more comprehensive and attacks stronger recent PPFR methods.
- *On the Inadequacy of Similarity-based Privacy Metrics* (g16vmAtJ8x.md): Scores 5,3,6,8,8 → Rejected despite some high scores. Very similar framing to FaceLinkGen: show that widely-used evaluation metrics are insufficient for privacy, and demonstrate a concrete attack. Rejected because the attack relied on specific dataset properties and did not generalize well. FaceLinkGen faces an analogous criticism (limited to a narrow method family) but has much stronger empirical results.
- *Diff-Privacy* (anG2Y15mwc.md): Scores 8,6,8,5 → Rejected. A technically more novel face privacy *defense* paper. Comparable community interest.

**Assessment positioning:** FaceLinkGen sits above the rejected PPFR attack papers (KAN, Revealing Unintentional) due to stronger methodology, cleaner results, and the near-zero-knowledge experiment. It is comparable to "Inadequacy of Similarity-based Privacy Metrics" in spirit and rigor. The limiting factors are: (1) narrow experimental scope (3 methods, same family), (2) the Amazon API gap in the NZK setting undermining a headline claim, (3) limited technical novelty, and (4) systematic overclaiming in scope. For ICML—a venue emphasizing both rigor and novelty—these are substantive barriers.

The paper is borderline. Its conceptual contribution is real and important; its empirical results on the linkage attack are strong; but the scope of claims significantly outpaces the scope of evidence. This is the same pattern that led to rejection of the "Inadequacy" paper despite 8,8 from two reviewers.

**Score: 5.5**
**Decision: Reject**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>

**Predicted score: 5.5**
