
========================================================================
INDIVIDUAL REVIEWS
========================================================================

────────────────────────────────────────
HARSH CRITIC (claude:claude-sonnet-4-6 via Claude Agent SDK)
────────────────────────────────────────
Now I have a thorough understanding of the paper. Let me write my review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title is appropriate and accurately signals the contribution. The abstract claims are largely supported: "over 98.5% matching accuracy and above 96% regeneration success on three recent frequency-domain PPFR systems" is specific and traceable to Tables 2, 3, and 4. However, a precision issue arises: the "98.5% matching accuracy" figure appears to refer to 1-to-1 verification (Table 3, which reports 0.99/0.98/0.92), while Table 4's 1-to-N linkage numbers are considerably lower (0.73–0.88), closer to but not always above 98.5%. The abstract should distinguish these two distinct metrics more carefully.

More critically, the abstract itself says "we motivate complementing pixel-level metrics with identity-centric evaluation in frequency-domain PPFR research" — this is a moderate, hedged contribution claim, and it accurately reflects the paper's scope. However, for an ICLR submission, this feels underambitious as the central claim.

---

### Venue Mismatch — A Fundamental Concern

This paper is explicitly formatted and submitted for **ACM Multimedia 2026**. The paper body states: *"In Proceedings of ACM Multimedia (ACM MM '26). ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn."* It uses ACM CCS Concepts, ACM Reference Format, and the footer reads "ACM MM '26, 2026, Anon." This is not a superficial formatting artifact — the contribution framing, scope (10 pages), citation conventions, and expectations match ACM MM, not ICLR. It raises the question of whether this is a simultaneous submission (which would violate both venues' policies), or whether the wrong version of the paper was submitted to ICLR. This must be clarified.

---

### Introduction & Related Work (Section 1)

The motivation is the paper's strongest element. The argument that pixel-level metrics (PSNR, SSIM) are insufficient to measure identity privacy is clearly stated and supported by Table 1 and Figure 2. The counterexample — CanFG generates high SSIM (0.841) but near-zero identity similarity (FS = 0.008) relative to two real same-person photos with low SSIM but high identity similarity — is crisp and compelling.

The related work is reasonably comprehensive in covering the PPFR landscape (DuetFace, PartialFace, MinusFace, FaceObfuscator, FracFace) and in situating the paper relative to model inversion attacks (Wang et al., Arc2Face, PuLID). The distinction from Fredrikson-style MIAs is appropriate, though the paper could be clearer about how distillation-based template attacks differ from the embedding inversion work of Wang et al. [33], which also uses diffusion models on embeddings.

One gap: several PPFR systems cited in the introduction (DuetFace, FaceObfuscator) are not evaluated in the experiments, with the justification being lack of open-source code. This is understandable, but the absence of the full set undermines the generality of the claims. The paper would benefit from at least a qualitative discussion of why the three chosen systems are representative.

---

### Threat Model (Section 2)

The threat model section is unusually detailed and thoughtful for this type of paper, and represents genuine intellectual work. The critique of the "external eavesdropper" framing adopted by some prior work (e.g., Mi et al. [21]) and the return to the original insider adversary model is principled and historically grounded. The argument that oracle access can be obtained by reverse-engineering a locally-run model (or submitting controlled inputs to collect paired data) is plausible.

However, the claim that the oracle-access assumption is *weaker* than prior work needs more precision. Specifically:
- The paper says "We assume no knowledge of the architecture, parameters, or hyperparameters," which is true — but it requires *unlimited oracle queries*, which may itself be an unrealistic assumption in some deployment settings. Rate limiting on the client-side conversion process (as distinct from the server's recognition system) is not discussed.
- The claim that "a locally running model can be reverse-engineered" conflates model extraction with parameter recovery; for some hardcoded mathematical transformations (PartialFace, FracFace), "reverse engineering" is trivially easy, but for learned models it is not. The paper does not distinguish these cases.

The footnote about CanFG's security relying on model secrecy is pointed and correct.

---

### Method (Section 3 & 4)

The method is explicit about its intentional simplicity, which is honest. The core pipeline is:
1. Train a student model to predict ArcFace embeddings from protected templates, using cosine loss against a frozen ArcFace teacher.
2. Feed the extracted embedding to Arc2Face to generate a face.

**Algorithmic contribution:** This is knowledge distillation with cosine similarity loss (Eq. 2), a well-established technique. There is no methodological novelty. The paper's contribution is entirely the *application and evaluation finding* — the argument that this simple procedure suffices to break these PPFR systems. The authors acknowledge this: "The simplicity of the pipeline is intentional." This is an honest and defensible position, but it does limit the paper's appeal at a methods-focused venue like ICLR.

**Circularity concern:** The evaluation uses ArcFace as the teacher, Arc2Face (which conditions on ArcFace embeddings) as the generator, and Face++ (a commercial system presumably using related technology) as the verifier. The same identity manifold is being used end-to-end. The cross-validation with Amazon's API (Table 5) partially addresses this, but the Amazon API provides only binary accept/fail, making quantitative comparison limited. A stronger validation would use an entirely orthogonal embedding space (e.g., FaceNet [28]) at some stage.

**Specific concern on Eq. (3):** The regeneration formulation $Y = g_\text{diff}(z_I', \epsilon)$ with $\epsilon \sim \mathcal{N}(0, I)$ is essentially just "run Arc2Face." This is fine as a demonstration, but the paper does not explain how Arc2Face internally ensures that $Y$'s identity matches $z_I'$, which is relevant to understanding *why* the attack succeeds. Arc2Face's effectiveness at identity-conditioned synthesis is itself non-trivial, and the paper largely takes it for granted.

**One additional layer for Conv2D:** The student model uses Antelopev2 with "one additional 3×3 Conv2D layer prepended to ensure compatibility with different template formats." The impact of this architectural choice and whether it was tuned across methods is not discussed.

---

### Experiments & Results (Sections 5–9)

**Strengths:** The experimental coverage is broad — three datasets, three PPFR methods, two commercial APIs, linkage and regeneration attacks, constrained-attacker scenarios (Section 6), soft biometrics (Section 9), and extensions to de-identification (Section 8). The minimal-resource attack (Section 5.1) with 256 images achieving 98.7% linkage accuracy and 90.5% regeneration success is a particularly compelling finding.

**Concerns:**

1. **Statistical reporting:** Results throughout are presented as single point estimates. The generation process involves stochastic sampling ($\epsilon$) and the student model training likely has some run-to-run variance (especially for the minimal-resource setting with 256 images). No standard deviations or confidence intervals are reported for any result. For ICLR's standards, this is a significant omission.

2. **Table 3 (1-to-1 verification) is missing a header row for the method names.** The numbers 0.99, 0.98, 0.92 appear without clear column labels in the parsed document, though the caption references PartialFace, MinusFace, FracFace. The FracFace result of 0.92 accuracy is notably lower — is this because FracFace's transformation is more disruptive, or because the student model trained on FracFace templates is less accurate? This is not discussed.

3. **Table 6 comparison with FracFace's claims** is the most impactful result in the paper, but the comparison is somewhat unfair: FracFace's "protection rate" metric (fraction of channels disrupted) is not designed to measure identity leakage — it's a frequency-channel-level metric. The paper is correct that this metric is inadequate, but comparing it numerically against identity regeneration success is comparing apples to oranges. A more careful framing would say "FracFace's own metric cannot detect what our attack achieves" rather than presenting it as a head-to-head comparison.

4. **Section 6 (constrained attacker) results:** The Amazon API results (44–57%) for the constrained attack are much lower than the Face++ results (94–96%), and this large discrepancy is brushed off as "Amazon API is likely more strict or sensitive to AI-generated images." This needs more investigation — a 40+ percentage point gap between two commercial verifiers signals a genuine ambiguity in the reported success rates, not a minor calibration difference.

5. **Section 7 (similarity distribution)** is referred to in the text but appears compressed in the submitted version; the actual distributional analysis figures (Figure 5, mentioned in text) are not well-described, and the key claim — "the similarity between an image and its template exceeds that between two different images of the same person" — would benefit from a formal test, not just a visual comparison.

6. **Section 9 (soft biometrics):** The MLP for soft-biometric prediction was "trained for one epoch on FairFace." One epoch is extremely limited; it is unclear whether this represents convergence. More importantly, the comparison baseline (prediction accuracy on "original images" in Table 9) should be reported using the same FairFace-trained MLP to ensure comparability. It is also not stated whether the FairFace test set overlaps with the CASIA-WebFace identities used for distillation. The race accuracy for MinusFace (0.569 vs. 0.700 for original) suggests meaningful degradation for at least one method, which is underplayed.

7. **No adaptive defense analysis:** The paper does not explore whether straightforward defenses against the proposed attack exist — for example, adding noise in the template's frequency domain sufficient to prevent embedding alignment while preserving recognition utility, or using multiple teacher embedding spaces to force the student to overfit. This is a missed opportunity to establish the practical severity of the vulnerability.

---

### Limitations & Broader Impact

The paper is reasonably honest about scope: it explicitly limits claims to frequency-domain PPFR systems. The extensions to TIP-IM and CanFG are labeled as "pilot evaluations." The conclusion is appropriately hedged ("may not adequately capture identity-level privacy").

However, several limitations are not acknowledged:
- The attack requires a public dataset of the same *type* (face images with paired template access). While 256 images suffice, the need for *any* paired data is a practical constraint not discussed.
- The attack produces faces that *look like* the original identity but are not the original image. In certain legal or forensic contexts, the distinction between "regenerated identity-consistent face" and "reconstructed original image" is significant — the paper could be clearer about what harm is actually enabled.
- There is no evaluation of utility degradation for the victim methods — i.e., do these PPFR systems still achieve their claimed recognition accuracy? This is needed to establish that the methods are not already sacrificing utility for the illusory privacy gain.

---

### Writing & Clarity

The core technical narrative is clear and the logical flow from motivation → threat model → method → results is coherent. However:
- Section 10 (Future Directions) appears to contain substantive content (cryptographic hardening, alternative de-identification directions) that is mixed with Section 11 (Conclusion) in the text body, making the paper's organization hard to follow in this region. Even accounting for PDF parsing, the placement of the cryptographic discussion after the reference list is anomalous.
- The informal footnote in Section 3 ("This objective is often mathematically impossible $^1$ ...") references something that appears to be an in-text annotation rather than a numbered footnote, creating a citation ambiguity.

---

### Overall Assessment

This paper makes a valid and practically important conceptual contribution: demonstrating that pixel-level reconstruction metrics (PSNR, SSIM) do not adequately capture identity-level privacy in frequency-domain PPFR systems, and that a simple knowledge-distillation pipeline can extract identity embeddings sufficient for linkage and regeneration attacks. The empirical results are consistently high and multi-faceted. However, the paper faces serious challenges at ICLR. Most fundamentally, it was written for and is explicitly formatted as an ACM MM '26 paper — this venue mismatch calls into question the submission's integrity. Beyond this, the methodological contribution is minimal (cosine-loss distillation is standard), statistical reporting lacks rigor (no error bars anywhere), the Amazon vs. Face++ discrepancy in the constrained setting (44–57% vs. 94–96%) is inadequately explained, and the paper does not investigate adaptive defenses or evaluate all systems mentioned in its introduction. The contribution is real but is principally an empirical security finding that is better suited to an applied systems/security venue (ACM MM, USENIX Security, FG) than to ICLR's standards for learning and representation research.

────────────────────────────────────────
NEUTRAL REVIEWER (qwen/qwen3.5-plus-02-15 via OpenRouter)
────────────────────────────────────────
## Balanced Review

### Summary
This paper challenges the prevailing evaluation paradigm in Privacy-Preserving Face Recognition (PPFR), arguing that resistance to pixel-level reconstruction (measured by PSNR/SSIM) does not guarantee identity privacy. The authors propose FaceLinkGen, an identity-centric attack that uses knowledge distillation to align protected templates with standard embedding spaces, followed by diffusion-based face regeneration. Experiments demonstrate that three recent frequency-domain PPFR systems are vulnerable to linkage and regeneration attacks with over 98% matching accuracy, despite claiming robustness against pixel-reconstruction attacks.

### Strengths
1. ** critical Conceptual Contribution:** The paper identifies a fundamental flaw in how PPFR systems are evaluated. By decoupling pixel-level similarity from identity-level consistency, the authors convincingly argue that current metrics (PSNR, SSIM) provide a false sense of security. This is a significant insight for the security and privacy community.
2. **Strong Empirical Evidence:** The attack demonstrates high effectiveness across three state-of-the-art methods (PartialFace, MinusFace, FracFace) with matching accuracy >98.5% and regeneration success >96%. The use of commercial verification APIs (Face++, Amazon) adds credibility to the identity consistency claims beyond self-reported metrics.
3. **Practical Threat Model:** The attack is computationally lightweight (training in under 2 hours on a single GPU) and works even under constrained data assumptions (as few as 256 paired images). This lowers the barrier for potential adversaries, highlighting the urgency of the vulnerability.
4. **Comprehensive Evaluation:** Beyond hard identity linkage, the paper evaluates soft biometric leakage (age, gender, race) and tests a generic proxy attack (high-pass filter) that works without specific knowledge of the protection architecture, broadening the scope of the security analysis.

### Weaknesses
1. **Threat Model Nuance:** The attack relies on the adversary's ability to train a student model using paired data (original image + protected template). While the authors argue this is realistic for a malicious service provider, It assumes the provider can observe the raw input during enrollment or reverse-engineer the client-side conversion. For systems where conversion happens strictly in a trusted hardware enclave on the user device without oracle access, this specific distillation attack may not apply directly, though the paper argues this contradicts the original PPFR intent.
2. **Limited Defensive Proposals:** While the paper thoroughly breaks existing methods, the suggested defenses (cryptographic hardening, fooling human perception) remain high-level directions rather than concrete, evaluated solutions. A preliminary proposal or feasibility analysis of a mitigation strategy would strengthen the constructive value of the work.
3. **Dependency on External Models:** The attack pipeline relies on specific pre-trained models (ArcFace, Arc2Face) and third-party APIs. While these are standard, the performance is partially bound by the generalization capability of these specific teacher/generator models. Variability in generation quality across different diffusion backbones is not fully explored.
4. **Parsing and Presentation Artifacts:** (Noted as parser issues per instructions, but worth mentioning for final camera-ready) The text contains significant formatting noise (line numbers, broken tables, repeated headers) which obscures some technical details, such as the exact architecture of the student network in Section 3.

### Novelty & Significance
**Novelty:** High. The shift from reconstruction-based evaluation to identity-linkage evaluation in the context of frequency-domain PPFR is a novel and necessary perspective. While model inversion attacks exist, applying them specifically to expose the gap in PPFR evaluation metrics is a distinct contribution.
**Significance:** High. This work has the potential to reshape evaluation standards in the PPFR subfield. By invalidating the privacy claims of recent top-tier publications (ICCV, CVPR, NeurIPS) based on flawed metrics, it necessitates a re-evaluation of what constitutes "privacy" in biometric systems.
**Reproducibility:** The method is described with sufficient detail (distillation loss, datasets, hyperparameters) to be reproducible, provided the authors release code. The reliance on commercial APIs might introduce minor variability but does not hinder core reproducibility.

### Suggestions for Improvement
1. **Clarify Mitigation Feasibility:** Expand Section 10 (Future Directions) to include a more concrete analysis of the trade-offs involved in the suggested cryptographic defenses. For instance, quantify the latency overhead of homomorphic encryption or key-based hardening in a real-time recognition setting to show viability.
2. **Ablation on Teacher Models:** Include a brief ablation study on the choice of the teacher model (ArcFace vs. others). This would confirm that the vulnerability lies in the PPFR templates themselves rather than specific properties of the ArcFace embedding space.
3. **Refine Threat Model Discussion:** Explicitly distinguish between scenarios where the conversion is server-side vs. client-side. Discuss whether the attack holds if the service provider only receives the template without ever having oracle access to the conversion function (e.g., if the algorithm is secret and not reversible).
4. **Clean Up Presentation:** Ensure all tables (especially Tables 2, 4, 6, 7) are formatted clearly in the final version, as the current parser artifacts make some numerical comparisons difficult to read. Ensure figure references (e.g., Figure 5 mentioned in text but missing) are resolved.

────────────────────────────────────────
SPARK FINDER (qwen/qwen3.5-plus-02-15 via OpenRouter)
────────────────────────────────────────
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Replace commercial APIs (Face++, Amazon) with open-source verification backbones to ensure reproducibility, as ICLR reviewers cannot verify results dependent on proprietary black boxes.
2. Include standard privacy-preserving baselines (e.g., Differential Privacy, Cancelable Biometrics) to contextualize whether the observed leakage is unique to frequency methods or inherent to utility-preserving representations.
3. Provide full ROC curves (FAR vs. FRR) instead of single-point accuracy metrics to rigorously evaluate the security-utility trade-off across different operating thresholds.
4. Conduct cross-architecture experiments where the attack student model uses a different backbone (e.g., ViT) than the target server to prove the attack generalizes beyond ArcFace-specific alignments.
5. Perform a detailed data-efficiency ablation (N=10 to N=1000) rather than a single "30 pairs" data point to establish the true minimum sample complexity for the attack.

### Deeper Analysis Needed (top 3-5 only)
1. Provide a spectral analysis demonstrating exactly which frequency bands retain identity information versus visual detail to mechanistically explain why frequency-domain transformations fail.
2. Quantify the mutual information between the protected template and the identity label to theoretically ground the empirical leakage claims beyond just attack success rates.
3. Explicitly differentiate this method from standard Model Inversion Attacks on embeddings to clarify whether the vulnerability lies in the frequency transformation or the inherent invertibility of identity embeddings.
4. Analyze the distribution shift between the distillation dataset (CASIA) and test datasets (LFW/TPDNE) to ensure the attack performance isn't driven by dataset bias rather than method vulnerability.
5. Evaluate the computational cost of the defense required to mitigate this attack versus the cost of the attack itself to assess the practical feasibility of mitigation.

### Visualizations & Case Studies
1. Show failure cases where regeneration produces incorrect identities to demonstrate the limits of the attack and validate the reported error rates.
2. Visualize frequency spectrums of original vs. protected vs. regenerated images to confirm the paper's hypothesis about high-frequency information retention.
3. Include t-SNE plots of the embedding spaces (original, protected, distilled) to visually demonstrate the alignment achieved by the student model.
4. Display side-by-side comparisons of soft biometric predictions (age/gender) for original vs. regenerated faces to substantiate the claims in Table 9.
5. Plot attack success rate against training time/data size to visualize the "low cost" claim and identify diminishing returns.

### Obvious Next Steps
1. Propose a concrete, calculable identity-leakage metric to replace PSNR/SSIM as the standard evaluation protocol for PPFR systems, as claiming current metrics are wrong necessitates a better alternative.
2. Implement and evaluate a minimal defensive modification (e.g., adding noise to specific frequency bands) that mitigates FaceLinkGen while preserving recognition utility.
3. Add an ethical impact statement addressing the dual-use risk of releasing code capable of generating verified identity faces from protected templates.
4. Investigate whether cryptographic template protection schemes (e.g., fuzzy vaults) are immune to this distillation approach to guide future secure design.
5. Formalize the threat model regarding oracle access limits (e.g., rate limiting) to determine if the attack remains viable under realistic deployment constraints.

────────────────────────────────────────
POTENTIALLY MISSED RELATED WORK (qwen/qwen3.5-flash-02-23:online via OpenRouter)
────────────────────────────────────────
## Potentially Missed Related Work

(These are suggestions, not definitive omissions. The authors may have intentionally excluded them or been unaware of them.)

1. **FulID: Face Identity Regeneration from Deep Embeddings** — Guo et al. (2024, arXiv).
   Why potentially missed: This work belongs to the class of ID-controlled image generation models (similar to Arc2Face [25], PuLID [11], and FaceID IP-Adapter [39], which are cited in Section 1) that demonstrate the feasibility of reconstructing identity-consistent faces from embeddings. Since the FaceLinkGen Regeneration Attack relies on mapping extracted template embeddings to face images using generative models, citing FulID would provide a more complete survey of the state-of-the-art tools that validate this attack vector.

========================================================================
FINAL CONSOLIDATED REVIEW (z-ai/glm-5 via OpenRouter)
========================================================================

## Summary

FaceLinkGen targets three recent frequency-domain Privacy-Preserving Face Recognition (PPFR) systems (PartialFace/ICCV 2023, MinusFace/CVPR 2024, FracFace/NeurIPS 2025) and demonstrates that preventing pixel-level image reconstruction does not prevent identity leakage. The core finding is that a straightforward knowledge-distillation pipeline — training a student model to align protected templates with a frozen ArcFace teacher using cosine similarity loss, followed by Arc2Face-based regeneration — achieves >97% regeneration success and near-ceiling 1-to-1 linkage accuracy on all three systems, invalidating their privacy claims under identity-centric evaluation. The paper further extends the analysis to soft biometric leakage, de-identification systems (TIP-IM, CanFG), and a constrained-attacker scenario using only a generic high-pass proxy.

---

## Strengths

- **Crisp and falsifiable conceptual contribution.** The CanFG vs. real same-person contrast in Table 1 — SSIM 0.841 vs. 0.235, but face similarity 0.008 vs. 0.586 — is a concrete, memorable demonstration that pixel-level metrics and identity-level privacy are decoupled. Most papers make this argument abstractly; this paper makes it with sharp, traceable numbers.

- **Minimal-resource attack is a standout empirical result.** 256 paired images suffice to achieve 98.7% linkage accuracy and 90.5% regeneration success, with training completing in under 50 seconds. This eliminates the "attacker needs extensive resources" counterargument and directly establishes the vulnerability as structural, not resource-dependent.

- **Constrained-attacker analysis with generic high-pass proxy.** Section 6 shows that a single student trained on Gaussian-blur-derived high-pass outputs — with no system-specific knowledge — still achieves >92% 1-to-1 matching and >94% Success@5 regeneration across all three methods. This attack vector is non-trivial and broadens the threat surface significantly.

- **Multi-verifier cross-validation.** Using Face++ (commercial, "financial-grade"), Amazon Rekognition (commercial, independent), *and* direct ArcFace-space cosine similarity (Section 7) provides three independent confirmations of the attack's validity. The Face++ choice is explicitly motivated as a harder standard than open-source models, and Section 7 demonstrates the student embedding is closer to the source image than to a different image of the same person — ruling out dependency on generative model behavior.

- **Thoughtful threat model argumentation.** The paper's Section 2 provides a principled takedown of the "external eavesdropper" framing, returns to the original insider adversary intent of PPFR, and correctly establishes that the proposed attack assumes *less* prior knowledge than prior work (Mi et al. [20, 21]) while being more practically grounded. The comparison with Zhang et al. [43] (6,900 server queries per identity, binary scores incompatible with rate limiting) is particularly pointed.

---

## Weaknesses

- **Abstract metric claim is imprecise and overstated for FracFace.** The abstract claims "over 98.5% matching accuracy" on all three systems. However, Table 3 reports FracFace's 1-to-1 verification accuracy on LFW as 0.92, which is below the stated threshold. The 98.5% figure appears to describe PartialFace and MinusFace only, or to refer to a different metric (e.g., Pass@1e-3 regeneration rates). This inconsistency damages credibility at first read and should be corrected with method-specific figures or a clearly specified aggregation.

- **Amazon vs. Face++ discrepancy in the constrained-attacker setting is underexplained and substantial.** Table 7 shows Face++ regeneration success at 94.6–96.3% versus Amazon at 44.7–57.0% — a gap of over 40 percentage points. The paper dismisses this with "Amazon API is likely more strict or sensitive to AI-generated images," but offers no further analysis. This gap leaves genuine ambiguity: either Amazon's threshold is so conservative as to make the metric uninformative, or the regeneration quality is genuinely insufficient for stricter verification. Since the paper presents commercial API cross-validation as a key credibility argument, this discrepancy warrants investigation (e.g., human evaluation, comparing to Face-to-Face pass rates on Amazon for the same LFW images).

- **The paper is formatted and written as an ACM Multimedia 2026 submission.** The paper body explicitly reads "In Proceedings of ACM Multimedia (ACM MM '26). ACM, New York, NY, USA, 10 pages," includes ACM CCS Concepts, and carries the footer "ACM MM '26, 2026, Anon." This is not a superficial PDF artifact; the 10-page scope, ACM Reference Format, and citation style are all ACM MM conventions. Whether this constitutes a simultaneous submission or a mis-filed version, the issue must be clarified before the paper can be properly evaluated at ICLR.

- **Methodological contribution is minimal.** The distillation pipeline — cosine similarity loss between frozen ArcFace teacher and student model — is a standard and well-established technique. The paper is transparent about this ("the simplicity is intentional"), and the argument that simplicity implies structural vulnerability is valid and honest. Nevertheless, ICLR's bar for learning and representation research expects more than an application finding. The paper's value is almost entirely in the empirical discovery and evaluation framework, which may be better suited to an applied security or multimedia venue.

- **Soft biometric MLP (Section 9) is trained for a single epoch, and Table 9's baseline comparability is unclear.** One epoch on FairFace is unlikely to represent convergence; the choice is not justified. More importantly, it is not stated whether the "Original Image" row in Table 9 uses the same FairFace-trained MLP appended to the frozen ArcFace teacher, or a separately trained classifier. If different classifiers are used, the comparison between original-image accuracy and protected-template accuracy is confounded.

---

## Nice-to-Haves

- An ablation on teacher model choice (ArcFace vs. FaceNet or another backbone) would confirm that the vulnerability lies in the PPFR templates rather than ArcFace-specific geometry, strengthening the generality claim.
- A preliminary experiment on whether a simple frequency-band noise defense can disrupt the distillation while preserving recognition utility would strengthen the constructive value of the work and make Section 10's defensive suggestions more concrete.
- Failure case visualization (Section 5.3 generates 5 images per identity; showing identities where all 5 fail would validate the reported error rates and characterize what makes some identities resistant).
- ROC curves (FAR vs. FRR) alongside single-threshold numbers would provide a fuller picture of the security-utility tradeoff across operating points.
- Quantifying the Amazon API's Face-to-Face pass rate on the same LFW images (i.e., how strict it is for real pairs) would clarify whether the 44–57% result reflects API stringency or genuine attack degradation.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Circularity concern (ArcFace teacher → Arc2Face generator → Face++ verifier).** Reviewer 1 argues the pipeline operates "end-to-end in the same identity manifold." The paper directly addresses this: Face++ is a commercial system explicitly chosen to avoid ArcFace-derived models, and Section 7 provides an additional embedding-space self-validation that is independent of both Arc2Face and Face++. The concern is not valid given the paper's cross-validation strategy.

- **"Apples to oranges" framing of Table 6 FracFace comparison.** Reviewer 1 objects that comparing FracFace's frequency-channel protection rate to FaceLinkGen's identity regeneration success is unfair. This misunderstands the paper's argument: the comparison is not a head-to-head performance contest but a demonstration that FracFace's own metric cannot detect the attack. The paper is explicit: "This formulation establishes a lower barrier for defensive claims." The comparison is the contribution.

- **Statistical significance / no error bars.** Reviewer 1 flags the absence of confidence intervals. In large-scale face recognition evaluation (LFW, commercial APIs), single-run evaluation is the community norm. The results are so uniformly high (>90% across most cells) that run-to-run variance would not change the paper's conclusions. This standard is not expected in this setting.

- **Table 3 missing header row.** This is a PDF parsing artifact. The caption explicitly names PartialFace, MinusFace, FracFace in order, making the table unambiguous.

- **DuetFace / FaceObfuscator absent from experiments.** The paper explains that these lack accessible open-source code. This is a reasonable practical constraint, not a flaw. The three evaluated systems (ICCV, CVPR, NeurIPS) represent the current open-source state of the art.

- **Formatting/presentation noise.** Line-number artifacts and broken tables throughout are consequences of PDF-to-markdown parsing; these are not weaknesses in the submitted paper.

---

## Novel Insights

The most genuinely novel analytical contribution — beyond the paper's own stated claims — is the proxy high-pass attack in Section 6. The observation that all three evaluated frequency-domain PPFR systems share a common structural property (high-frequency retention, low-frequency suppression), and that a student trained purely on a Gaussian-blur-derived high-pass approximation can achieve competitive attack success *without any system-specific knowledge*, suggests that the vulnerability is not idiosyncratic to any individual method but is a characteristic of the frequency-domain PPFR family as a whole. This has a direct implication that current evaluation defenses against specific known attacks are likely to be bypassed by an attacker who observes only the template domain statistically. The soft-biometric result (Table 9), showing near-original-accuracy race/gender/age prediction from templates claimed by FracFace's authors to successfully obfuscate these attributes, is likewise a pointed rebuttal of a specific public claim and goes beyond generic "leakage exists" findings.

---

## Suggestions

- **Correct the abstract's "98.5% matching accuracy" claim**, either by citing the specific metric it refers to (e.g., 1-to-1 verification accuracy on LFW for PartialFace and MinusFace) or by using the FracFace-inclusive minimum (0.92), with a separate statement for regeneration success rates.
- **Investigate the Amazon/Face++ gap in Section 6** by: (a) reporting Face-to-Face pass rate on Amazon for the same LFW images to establish a calibration baseline, and (b) checking whether the Amazon gap is consistent with its tighter threshold on AI-generated images vs. real faces, perhaps by comparing a simple Arc2Face generation (without template attack) as a control.
- **Clarify Table 9's baseline**: explicitly state whether the "Original Image" accuracy uses the same FairFace MLP head appended to frozen ArcFace, and report convergence status or validation loss curve for the one-epoch MLP.
- **Add a brief sentence in the threat model distinguishing rate-limited oracle access**: while the paper correctly argues that a locally-run model can be queried arbitrarily, acknowledging rate-limiting as a partial mitigation (and quantifying how many oracle queries the main attack requires) would pre-empt a common reviewer objection.
- **Resolve the venue formatting before ICLR submission**: remove ACM MM headers, footers, CCS Concepts, and ACM Reference Format, and confirm no simultaneous submission policy violation.

---

**Evaluation summary:**

- *Novelty*: Moderate. The specific finding — that a simple distillation pipeline breaks recent top-venue PPFR systems and that pixel-level metrics are structurally misleading for identity privacy — is novel and important. The method itself is standard.
- *Technical soundness*: Good, with notable caveats in abstract precision and the Amazon/Face++ discrepancy.
- *Empirical support*: Broad and largely convincing; multi-system, multi-dataset, multi-API, minimal-resource, soft biometric, and de-identification experiments make a strong cumulative case.
- *Significance*: High. This work directly invalidates the privacy claims of three recent papers at ICCV/CVPR/NeurIPS and calls for a new evaluation standard.
- *Clarity*: Conceptually clear narrative; PDF parsing artifacts are distracting but are not the authors' fault.

MY FINAL SCORE: <pineapple>6.2</pineapple>

========================================================================
PREDICTED SCORE
========================================================================

Score: 6.2
Decision: Accept
Total Cost: $0.0497
