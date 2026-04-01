
========================================================================
INDIVIDUAL REVIEWS
========================================================================

────────────────────────────────────────
HARSH CRITIC (claude-sonnet-4-6 via Claude SDK)
────────────────────────────────────────
## Section-by-Section Critical Review

---

### Title & Abstract

The title accurately reflects the contribution: a re-evaluation of identity leakage in PPFR, not a design of a new protective method. The abstract clearly states the problem (pixel-level metrics insufficient to capture identity leakage), method (distillation pipeline + diffusion generation), and key quantitative results (>98.5% matching, >96% regeneration). The claim of "black-box oracle" access is stated, but the abstract does not clarify that the attack requires a *training set of paired original/template data*—which is actually the operative assumption. A reader could misconstrue "black-box oracle" as requiring only query access at inference time, when in fact the attacker must collect ~90K training pairs. This distinction is consequential for threat modeling and should be surfaced in the abstract.

---

### Introduction & Related Work

The motivation is well-constructed. The core argument—that pixel-level metrics are decoupled from identity-level privacy—is both novel in this context and well-supported by the concrete example in Table 1 and Figure 2. The comparison between CanFG's high SSIM/low identity similarity and same-person cross-photo scenarios is apt.

**Concern 1 (conflation of design and vulnerability):** The introduction frames identity leakage as a flaw in these PPFR systems, but any PPFR system that supports recognition utility *must* preserve identity-discriminative features. What the paper is really exposing is not that these systems fail at privacy by design, but that their *evaluation methodology* falsely implies stronger privacy than the design actually provides. The framing in the introduction merges these two issues. To be fair: the paper does partially acknowledge this in the methods section ("As long as I persists in t, an attacker can recover..."), but this fundamental tension—between recognition utility and privacy—deserves more precise treatment in the introduction. Specifically, is the paper claiming (a) the systems are poorly designed, or (b) they are well-designed but evaluated with the wrong metrics? The answer appears to be (b), but the framing leans toward (a).

**Concern 2 (TPDNE as a "this person does not exist" dataset):** The introduction discusses identity-level verification, but the use of AI-generated faces as a test population raises an unaddressed question: what constitutes "ground truth" identity for a face that has no real-world correspondent? This concern applies to the experiments and is flagged here because the conceptual setup is introduced here.

The coverage of related work is solid, distinguishing from model inversion attacks, embedding-inversion attacks (Arc2Face, PuLID), and federated learning semantic recovery. The clarification that this attack targets the template generation process rather than the recognition embedding is important and well-made.

---

### Related Work

Related work is well-integrated into the introduction section rather than isolated. The coverage of prior PPFR systems (DuetFace, PartialFace, MinusFace, FaceObfuscator, FracFace), MIA literature, and de-identification methods is comprehensive for the scope of the paper.

**Concern 3 (missing comparison to Zhang et al. [43]):** The paper characterizes Zhang et al.'s method as requiring ~6,900 queries per identity and returning continuous similarity scores. While the paper argues this is impractical, it does not provide empirical comparison of attack success rates between FaceLinkGen and Zhang et al. on any shared benchmark, leaving the relative performance gain unmeasured.

**Concern 4 (missing DuetFace evaluation):** DuetFace [19] is listed as a representative PPFR system in the introduction and referenced repeatedly, but it is not included in the experimental evaluation. The paper states it selected systems with "accessible source code," which is a reasonable practical constraint, but this should be stated explicitly in the experimental section to avoid the appearance of selective evaluation.

---

### Method / Approach

The method is clearly described at a conceptual level. The distillation objective (Eq. 2) is standard cosine similarity loss, and the generative component (Eq. 3) is a straightforward use of Arc2Face. The simplicity is intentional and the authors make this explicit.

**Concern 5 (the theoretical framing overstates novelty):** Equations 1–3 formalize the intuition that if a template t retains identity information I, a student model can recover an embedding aligned to I. This is mathematically correct but essentially restates a known property: any useful PPFR system must have I(t; I) > 0 for recognition to work. The formal notation adds a veneer of rigor without yielding non-obvious theoretical insight—for example, there is no information-theoretic bound on how much identity information must be retained for recognition at a given accuracy, nor is there a characterization of when this threshold becomes exploitable. The paper would be strengthened by at least acknowledging this gap.

**Concern 6 (attack requires oracle access for training data collection):** The threat model in Section 2 correctly identifies oracle access (submit arbitrary inputs, observe outputs) as the attack surface. However, Section 3 does not explicitly state how many paired samples the attacker must collect to train the student model. Section 5 mentions ~90K training images and Section 5.1 reduces this to 256. But how many *template queries* does collecting 90K pairs require? If the protected template conversion runs on the attacker's local service infrastructure, collecting 90K pairs is trivial. If it requires interaction with a rate-limited server, this is a meaningful constraint. This operational detail is important for practitioners assessing realistic risk and is not addressed.

**Concern 7 (no analysis of what the student model actually learns):** The paper claims the student model learns to extract identity-discriminative features from protected templates. It would be useful to show—even qualitatively—what features the student is attending to (e.g., via saliency or activation maps). Without this, the paper cannot distinguish between two hypotheses: (a) the student learns a meaningful inverse of the frequency-domain transform, or (b) the student exploits residual low-level cues that happen to correlate with identity in the CASIA-WebFace training set. The minimal-resource experiment (Section 5.1) is suggestive of (a), but not conclusive.

**Concern 8 (edge case: stochastic or key-based transformations):** The threat model explicitly excludes systems with secret keys (citing Yuan et al. [41]). However, the paper does not analyze what happens when moderate randomization is added to the oracle's conversion process—not a full secret key, but enough to make paired data collection noisy. Even a small random perturbation in the frequency domain would increase the variance of the training signal. The robustness of the distillation pipeline to oracle noise is not discussed.

---

### Experiments & Results

The experimental design is generally sound: three PPFR systems, three test datasets, a commercial verifier (Face++) different from the training model, and cross-validation with a second API (Amazon). The minimal-resource experiments and constrained-attacker experiments add useful breadth.

**Concern 9 (TPDNE dataset methodological issue):** The paper uses the "This Person Does Not Exist" dataset—synthetic faces generated by StyleGAN—as one of three test datasets. The rationale is to "avoid data cross-contamination from Stable Diffusion 1.5 and Arc2Face training data." However, the evaluation pipeline asks Face++ to verify that the *regenerated face* (output of Arc2Face, conditioned on the extracted embedding) matches the *original TPDNE face* (a StyleGAN output). This verification is valid only if Face++ reliably assigns consistent identity to both StyleGAN- and Arc2Face-generated faces of the "same" person. It is not self-evident that commercial face verifiers are well-calibrated across generative model domains—they are typically trained and validated on real photographs. The paper should either (a) verify that Face++ is reliable on cross-model synthetic-to-synthetic comparisons or (b) rely primarily on the real-image datasets (CASIA hold-out, LFW) for identity claims.

**Concern 10 (Table 6 is difficult to interpret as written):** The column headers of Table 6 appear garbled in the parsed manuscript. More substantively: the rows label three different methods (PartialFace, MinusFace, FracFace), but the column "Protection Tested in FracFace [6]" shows a single value per row (0.680, 0.850, 1.000). It is unclear whether these are FracFace's *claimed protection rates for its own method against FracFace's simulated attack* or FracFace's evaluation of the other methods. The table needs a clearer caption explaining exactly what metric is being shown in the first data column and from which paper that metric is drawn.

**Concern 11 (linkage gap not fully explained):** Table 4 shows top-1 recall for the linkage attack: FracFace-to-FracFace achieves 0.786, and Original-to-Original achieves 0.882. The paper claims this "essentially reaches the dataset's theoretical maximum" citing 9.3–13% label noise in CASIA-WebFace. However, with 13% noise the theoretical ceiling for any method would be approximately 0.87, and Original-to-Original achieves 0.882 (slightly above, likely a sampling artifact). FracFace-to-FracFace at 0.786 is 0.096 below the upper bound—a non-trivial gap that is dismissed without analysis. What is driving this gap? Is it the distillation error, the template's information loss, or dataset noise? Understanding this would clarify whether the residual gap is fundamental or tractable.

**Concern 12 (no ablation on distillation loss or architecture):** The student model is described as Antelopev2 with one prepended Conv2D layer. There is no ablation on (a) whether the prepended layer is necessary, (b) whether a simpler or shallower student works equally well, or (c) whether the cosine similarity loss is meaningfully better than, say, an L2 loss in embedding space. Given that the paper argues the vulnerability is in the *representation* rather than the attack design, demonstrating that a wide range of student architectures achieves similar results would strengthen this claim.

**Concern 13 (Amazon API results in constrained setting):** Table 7 shows Amazon API regeneration success rates of 44–57% in the constrained setting, compared to 92–96% on Face++. The paper dismisses this discrepancy with "The Amazon API is likely more strict or sensitive to AI-generated images," but this is speculative. A 40-50 percentage point gap between two commercial verifiers on the same data is a substantive inconsistency that weakens confidence in the results. The paper should either provide a direct calibration comparison of the two APIs on real-to-real face pairs from LFW, or acknowledge this as a genuine uncertainty about the constrained setting's results.

**Concern 14 (soft biometric results, MinusFace gap):** Table 9 shows MinusFace has notably lower race classification accuracy (0.569 vs. 0.700 baseline) compared to PartialFace (0.663) and FracFace (0.673). This suggests MinusFace may be doing something meaningfully different in the frequency domain that partially disrupts soft biometric extraction. The paper notes this but does not investigate why. This is a meaningful signal worth pursuing.

**Concern 15 (no direct baseline: what does the raw template yield?):** A natural and important ablation is missing: what happens if the attacker skips distillation entirely and directly feeds the protected template into ArcFace (or another off-the-shelf FR model)? If this already achieves high accuracy, the distillation step is unnecessary. If it achieves low accuracy, the distillation's contribution is clear. This baseline would sharpen the paper's central claim that distillation is sufficient but not that it is necessary.

---

### Writing & Clarity

The paper is generally well-written and the central argument is clearly articulated. The distinction between pixel-level and identity-level privacy is effectively motivated.

**Concern 16 (Section 10 is underdeveloped):** "Future Directions" is referenced in the table of contents and section header but the content is split between Section 10 and the conclusion. The cryptographic hardening discussion and the "fooling human perception" section are presented in the conclusion without the analytical depth the rest of the paper maintains. Specifically, the suggestion to use depth maps or Canny edges as identity-agnostic representations is speculative—depth maps from a face image still encode considerable identity information (face geometry is identity-discriminative). This claim needs either a citation or a caveat.

**Concern 17 (Section 7 "Similarity Distribution" is referenced but figures are absent in parsed text):** Section 7 references Figure 5 ("Some near-zero similarities are due to dataset noise") but the figure is not accessible in the parsed manuscript. While this may be a parsing artifact, the section itself is brief and relies heavily on the figure without sufficient textual description of the distribution's shape or statistics.

---

### Limitations & Broader Impact

The paper acknowledges several limitations: (a) it evaluates only frequency-domain PPFR methods primarily; (b) CanFG and TIP-IM results are "pilot evaluations"; (c) the attack assumes oracle access. The suggestion of cryptographic defenses as a path forward is appropriate.

**Concern 18 (fundamental limitation understated):** The paper's strongest implicit limitation is that it demonstrates a problem without a fix. The "future directions" section gestures at cryptographic methods but does not show that these defenses are actually resistant to FaceLinkGen. For instance, if the key-based conversion still outputs templates that can be queried by the attacker (with the same key applied on the attacker's device), the same distillation attack applies. The paper should more explicitly characterize which design properties would make this attack infeasible, not just cite existing methods that *claim* to resist certain attacks.

**Concern 19 (societal impact of demonstrating the attack):** The paper presents an attack that can regenerate the faces of real people from protected templates. Given that CASIA-WebFace and LFW contain images of real individuals, the paper should discuss the ethical considerations of releasing this attack code and demonstration images. The regenerated faces in Figure 1 include recognizable individuals. While the paper frames this as a necessary red-team exercise, the potential for misuse—particularly the regeneration of identity-consistent faces from leaked template databases—deserves explicit discussion.

**Concern 20 (ISOstandard claim requires precision):** The paper states that "ISO/IEC 24745 explicitly requires resistance against template-to-template search, but does not address face-to-template search." This is a specific normative claim. The paper should either cite the exact clause or acknowledge that this is an interpretation, because ISO/IEC 24745 deals with biometric information protection more broadly, and readers unfamiliar with the standard cannot verify this claim as written.

---

### Overall Assessment

FaceLinkGen makes a genuine and well-motivated contribution: it demonstrates that the prevailing evaluation paradigm for frequency-domain PPFR—measuring attack success via PSNR and SSIM—may not adequately capture identity-level privacy risk, and provides a lightweight distillation attack that achieves high identity-linkage and face-regeneration success across three recent PPFR systems. The core experimental findings are credible, the choice of an independent commercial verifier (Face++) avoids circular evaluation, and the minimal-resource experiments (down to 256 images) make the threat concrete and practically significant. However, the paper has three substantive issues that should be addressed before publication. First, the TPDNE dataset methodology is questionable for identity verification across generative model domains, and the Amazon API discrepancy in the constrained setting (44–57% vs. 92–96% on Face++) is not satisfactorily explained—together these raise uncertainty about roughly a third of the reported results. Second, an obvious and important ablation is absent: what does a naïve baseline (feeding the protected template directly into an off-the-shelf FR model without distillation) achieve, and how does it compare? Without this, the necessity of the distillation step is unclear. Third, the paper's theoretical framing conflates two distinct claims—that the systems are poorly designed versus that they are measured with the wrong metrics—and the stronger version of the contribution (it is the metrics, not necessarily the systems, that are inadequate) is not argued with the rigor the data supports. Addressing these three issues would substantially strengthen a paper whose central insight is both timely and important for the PPFR research community.

────────────────────────────────────────
NEUTRAL REVIEWER (z-ai/glm-5 via OpenRouter)
────────────────────────────────────────
## Balanced Review

### Summary
This paper challenges the prevailing evaluation paradigm in frequency-domain Privacy-Preserving Face Recognition (PPFR), which equates privacy with resistance to pixel-level reconstruction measured by PSNR and SSIM. The authors demonstrate that these metrics inadequately capture identity-level privacy by proposing FaceLinkGen, a distillation-based attack that extracts identity embeddings from protected templates and regenerates identity-consistent faces without recovering original pixels. The attack achieves over 98.5% matching accuracy and above 96% regeneration success on three recent PPFR systems (PartialFace, MinusFace, FracFace), revealing a fundamental gap between pixel-distortion metrics and actual identity protection.

### Strengths
1. **Compelling conceptual contribution**: The paper makes a persuasive argument that pixel-level similarity metrics are fundamentally misaligned with identity privacy goals, supported by concrete examples (Figure 2, Table 1) showing that high PSNR/SSIM does not guarantee identity preservation, and vice versa.

2. **Strong empirical results with practical significance**: The attack achieves near-perfect success rates across three recent PPFR methods using a straightforward distillation pipeline, completed in under two hours on a single GPU at minimal cost (~$0.80-$1.60). The minimal-resource experiment (800 images, 50 seconds) demonstrates the vulnerability is fundamental, not requiring sophisticated optimization.

3. **Comprehensive evaluation methodology**: The authors test on three datasets (TPDNE, CASIA-WebFace hold-out, LFW), cross-verify with two commercial APIs (Face++ and Amazon), report multiple metrics (Success@5, Pass@FAR thresholds), and appropriately use no dataset/architecture overlap between training and target methods.

4. **Threat model rigor**: The paper returns to first principles by evaluating against the intended adversary (service provider with oracle access) and extends analysis to constrained attacker scenarios with limited samples (30 pairs) and no knowledge of protection internals, showing >92% matching success even under these restricted conditions.

5. **Broader applicability demonstrated**: The extension to de-identification methods (TIP-IM, CanFG) and soft biometric leakage analysis (age, gender, race prediction from templates) strengthens the argument that identity leakage is a systemic issue beyond frequency-domain PPFR.

### Weaknesses
1. **Limited scope of evaluated PPFR methods**: Only three frequency-domain methods are tested. While these are the main open-sourced representative systems, the paper would benefit from evaluating additional PPFR approaches (e.g., DuetFace, FaceObfuscator) or non-frequency methods to strengthen generality claims.

2. **Regeneration attack depends on compatible generative models**: The approach relies on Arc2Face being compatible with ArcFace embeddings. For systems using different embedding models, finding compatible generative models may be challenging, which is acknowledged but not empirically addressed.

3. **No empirical evaluation of proposed countermeasures**: The paper suggests cryptographic approaches and alternative de-identification strategies as future directions but provides no preliminary experiments to validate whether these would actually resist the proposed attack.

4. **Preliminary results for non-frequency methods**: The CanFG and TIP-IM evaluations are described as "pilot" and limited in scope. More thorough evaluation would strengthen claims about vulnerability generalization.

5. **Dataset noise acknowledgment without remediation**: The paper notes CASIA-WebFace contains 9.3%-13.0% noise affecting linkage accuracy (ceiling of 88%), but does not provide cleaner subset evaluation to isolate method performance from dataset quality issues.

### Novelty & Significance
The paper makes a significant conceptual contribution by challenging the reconstruction-centric evaluation paradigm in PPFR. The insight that identity information can be extracted without pixel-level reconstruction is both novel and practically important. The attack methodology is straightforward by design, which strengthens the core argument: if a simple distillation pipeline suffices, the vulnerability lies in the representation itself rather than requiring sophisticated adversarial techniques. The work has substantial implications for how PPFR systems should be designed and evaluated going forward.

### Suggestions for Improvement
1. **Expand method coverage**: Include evaluation on additional PPFR systems (e.g., DuetFace, FaceObfuscator) to strengthen the claim that the vulnerability is systematic rather than specific to the three tested methods.

2. **Provide empirical countermeasure analysis**: Conduct preliminary experiments on whether cryptographic hardening or key-based approaches (as mentioned in Section 10) resist the proposed attack, even if limited in scope.

3. **Curate a clean evaluation subset**: Create or use a manually verified identity subset to establish cleaner performance bounds without dataset noise interference, enabling more precise comparison of attack efficacy across methods.

4. **Strengthen non-frequency method evaluation**: Expand the CanFG and TIP-IM experiments with full regeneration attack evaluation (not just linkage) to fully substantiate claims about vulnerability generalization.

5. **Add ablation studies**: Investigate which components of the protected template contribute most to identity leakage—this could inform more targeted protection strategies and strengthen the analytical contribution.

────────────────────────────────────────
SPARK FINDER (claude-sonnet-4-6 via Claude SDK)
────────────────────────────────────────
## Strengthening Opportunities

### Missing Experiments

1. **Broader PPFR method coverage**: DuetFace [19] and FaceObfuscator [14] are cited as representative systems but are not attacked. Including them would make the evaluation more comprehensive and strengthen the claim that the vulnerability is systemic across frequency-domain PPFR, not specific to the three tested methods.

2. **Systematic data-efficiency curve**: The paper shows results at ~800 images and ~256 images (Section 5.1) but only for FracFace. A full learning curve (e.g., 64 → 256 → 1K → 10K images) across all three methods, reporting both linkage accuracy and regeneration success rate, would give a much clearer picture of how quickly identity information becomes extractable and where the practical floor is.

3. **Ablation on teacher/student model families**: The attack uses Antelopev2 as both teacher and student backbone. Testing with FaceNet or CosFace as the teacher, and a lighter backbone (e.g., MobileNet) as the student, would show whether the attack generalizes across embedding spaces or is tuned to ArcFace-family representations — a critical point since Arc2Face is tightly coupled to ArcFace embeddings.

4. **Demographic-stratified attack success**: Attack success rates broken down by race, gender, and age group (using FairFace labels) would reveal whether certain demographic groups face disproportionate privacy risk — important given Section 9's findings on soft biometric leakage and the paper's citation of privacy regulations protecting race and age.

5. **Direct comparison with prior attacks as baselines**: The paper compares claimed protection rates from FracFace's own evaluation (Table 6) but does not run the U-Net or StyleGAN reconstruction attacks head-to-head under identical conditions. Side-by-side numbers from the same test split — FaceLinkGen vs. prior pixel-level attackers — would more rigorously demonstrate the gap.

6. **Amazon vs. Face++ discrepancy investigation**: Table 7 shows Amazon API success rates of 44–57% vs. Face++ rates of 94%+ for the constrained attack. This ~2× gap is unexplained. A targeted experiment varying the verification backend (open-source ArcFace, FaceNet, DeepFace) across the same generated images would determine whether the discrepancy is due to AI-image sensitivity, operating threshold differences, or systematic model bias.

---

### Deeper Analysis Needed

1. **Mutual information estimation between template and identity**: The paper argues informally that identity information "persists" in templates. Formalizing this with an estimator of I(T; Z_id) — where T is the template and Z_id is the identity embedding — would provide a principled lower bound on extractable identity, strengthening the theoretical grounding of the vulnerability claim.

2. **Utility–privacy trade-off analysis**: Because any PPFR system that supports recognition must preserve identity-discriminative features, there exists a fundamental tension between recognition utility (TAR@FAR) and identity leakage. A curve plotting attack success rate against the defended system's recognition accuracy would make explicit how much "room" current methods have to improve privacy without sacrificing utility.

3. **Failure case characterization**: The paper achieves near-ceiling performance but does not analyze the ~1–8% of failures. Understanding whether failures cluster around certain identities, certain template characteristics, or certain image conditions (e.g., extreme pose, low resolution, multiple faces) would clarify whether any structural defense is already partially working.

4. **Formal connection to established privacy frameworks**: The paper would benefit from connecting its identity-centric metric to formal notions such as k-anonymity (how many identities are indistinguishable in template space?) or a differential-privacy-style bound on how much a single template reveals about the underlying face. Even a brief discussion of why formal privacy guarantees are violated would strengthen the "motivating stronger defenses" narrative.

5. **Sensitivity of distillation to template perturbations**: An ablation adding Gaussian noise or quantization to the protected template before distillation would reveal how much the attack degrades with lightweight post-processing defenses — useful for guiding practitioners who might consider simple noise injection as a low-cost mitigation.

---

### Untapped Applications

1. **Video-based face recognition**: Static-image PPFR is one setting, but video surveillance (where multiple frames per identity are available) is arguably the higher-stakes deployment context. Testing whether aggregating temporal embeddings from multiple video frames further improves attack success — or whether any frame is already sufficient — would extend the work's relevance to surveillance applications.

2. **Federated learning aggregation setting**: In federated biometric systems, protected templates from multiple clients are aggregated server-side. Testing the attack in a scenario where the adversary receives averaged or aggregated templates (rather than per-image templates) would address whether the vulnerability survives aggregation-based defenses.

3. **Other biometric modalities**: The distillation pipeline is modality-agnostic in principle. A brief pilot on iris or periocular templates protected by frequency-domain methods would test whether the "identity information preserved for recognition utility → extractable by distillation" argument generalizes beyond face, broadening the impact claim.

4. **Cross-age and cross-pose scenarios**: Evaluating the attack when the enrolled template comes from a young face and the attacker verifies against an older face image (or severe pose change) would stress-test whether the recovered embedding captures identity robustly enough for real-world re-identification, where time gaps between enrollment and probe are common.

---

### Visualizations & Case Studies

1. **Embedding-space t-SNE/UMAP plots**: Visualizing clusters of original embeddings alongside student-recovered embeddings from protected templates (colored by identity) would compellingly show that the distillation output lies in the same identity-clustered geometry as the original ArcFace space — making the linkage vulnerability immediately intuitive.

2. **Cosine similarity histograms for all three methods**: Section 7 analyzes similarity distributions but the paper references Figure 5 without reproducing it in the submission. Overlaid histograms comparing (1) same-identity image pairs, (2) image-to-template pairs, and (3) different-identity pairs for all three methods in one figure would make the privacy violation concrete and comparable across methods.

3. **Failure case gallery**: A small gallery showing the ~5% of cases where regeneration fails — alongside the original image and the template — would help readers understand what (if any) visual or structural property of a template resists the attack, and guide future defensive design.

4. **Attention/saliency maps on the student model**: Visualizing which regions of the protected template the student model attends to when recovering the identity embedding would reveal whether the attack exploits residual facial geometry, texture, or other structural artifacts — giving defenders a concrete target for disruption.

5. **Side-by-side multi-identity regeneration grid**: A large grid showing 10–15 distinct identities, each with their original image, protected template, and five regenerated images (to show the stochasticity of Arc2Face), would let readers qualitatively assess identity consistency and soft biometric preservation in a single glance.

---

### Natural Next Steps

1. **Standardized identity-centric benchmark toolkit**: The most immediate contribution this work could support is a public benchmark suite — templates from multiple PPFR systems, a standardized evaluation protocol using open-source face matchers (to avoid commercial API dependency), and reference attack implementations — so that future PPFR papers can report identity-centric metrics alongside PSNR/SSIM without re-implementing the evaluation from scratch.

2. **Empirical validation of proposed defenses**: Section 10/11 proposes cryptographic hardening (key-based transformation, fully homomorphic encryption) and a novel low-frequency-only de-identification direction. Implementing and attacking even a prototype of these defenses with FaceLinkGen would move the paper from "exposing a problem" to "providing actionable guidance," and would also confirm whether the proposed directions actually resist the distillation attack.

3. **Adaptive defense against distillation attacks**: The distillation attack succeeds because protected templates lie in a learnable neighborhood of the original embedding space. A natural next step is to design a "distillation-resistant" PPFR objective — e.g., adding a term that maximizes the distance between the template's alignment to any publicly accessible embedding model — and measuring the recognition utility cost of such resistance.

4. **Longitudinal re-identification risk study**: Beyond a single attack, a study simulating a realistic data-breach scenario — where an attacker obtains a leaked database of templates and a public social-media image corpus — would quantify the real-world re-identification rate and expected time-to-link, translating the technical vulnerability into a concrete risk assessment useful for policy and compliance contexts.

5. **Extension to generative template systems**: Methods like CanFG and FaceAnonyMixer generate synthetic faces that preserve soft biometrics by design. A follow-up study systematically characterizing the privacy–utility boundary for generative (as opposed to transform-based) PPFR, and whether distillation-style attacks transfer, would be a natural and high-impact extension of the "pilot evaluation" already begun in Section 8.

────────────────────────────────────────
POTENTIALLY MISSED RELATED WORK (perplexity/sonar-pro via OpenRouter)
────────────────────────────────────────
Related work search was skipped.

========================================================================
FINAL CONSOLIDATED REVIEW (claude-sonnet-4-6 via Claude SDK)
========================================================================

```json
{
  "summary": "FaceLinkGen challenges the prevailing evaluation paradigm for frequency-domain Privacy-Preserving Face Recognition (PPFR) systems, demonstrating that pixel-level metrics (PSNR, SSIM) fail to capture identity-level privacy leakage. The authors propose an identity-centric attack that trains a student model via distillation to extract ArcFace-compatible embeddings from protected templates, then uses Arc2Face to regenerate identity-consistent faces without recovering original pixels, achieving >98.5% matching accuracy and >96% regeneration success across three recent PPFR systems (PartialFace, MinusFace, FracFace). The core argument is that since any useful PPFR system must preserve identity-discriminative features for recognition, that same information remains extractable by a distillation pipeline even when pixel-level reconstruction is infeasible.",
  "strengths": [
    "Compelling and well-evidenced conceptual contribution: The argument that PSNR/SSIM are fundamentally misaligned with identity-level privacy is concretely supported by Table 1 and Figure 2, which show that high pixel-level similarity (CanFG face, SSIM=0.841) corresponds to near-zero identity similarity (FS=0.008), while two real photos of the same person show the reverse. This is a genuinely important insight for the PPFR evaluation community.",
    "Strong empirical results with practical significance: Near-ceiling attack success (>98.5% matching, >96% regeneration Success@5) is demonstrated across three distinct PPFR systems on three datasets using only ~$0.80–$1.60 of compute and under two hours on a single A6000 GPU. The minimal-resource experiment (256 images, 98.7% linkage accuracy) makes the threat concrete and shows the vulnerability is structural rather than requiring sophisticated optimization.",
    "Credible and independent evaluation methodology: Commercial face verification (Face++) is used for evaluation, deliberately avoiding circular use of the same model (ArcFace) used for training. Cross-verification with Amazon's API on 700 LFW images provides a second independent data point. Training and test data are disjoint in identity, and there is no architecture overlap between the student model (CASIA-WebFace-trained Antelopev2) and the attacked systems (PartialFace/FracFace are training-free; MinusFace is trained on MS1Mv2).",
    "Rigorous threat model grounded in original PPFR design intent: The paper returns to first principles, correctly identifying the service provider as the intended adversary, and explicitly adopts a weaker assumption than prior work (no knowledge of architecture, parameters, or hyperparameters — only oracle access). The constrained-attacker analysis (Section 6, using only 30 paired validation samples and a generic high-pass proxy) still achieves >92% matching and >94% regeneration Success@5 on Face++, demonstrating robustness of the vulnerability.",
    "Breadth of attack coverage: The paper extends evaluation to adversarial de-identification (TIP-IM) and a non-frequency PPFR method (CanFG), and adds a soft-biometric leakage analysis (Table 9) showing that race, gender, and age can be predicted from templates at near-baseline accuracy. Together these substantiate the claim that identity leakage is a systemic issue beyond frequency-domain PPFR.",
    "Directness of comparison with defended methods' own claims: Table 6 directly compares FracFace's claimed protection rates (based on frequency-channel disruption) against FaceLinkGen's identity-centric metric on the same method, demonstrating that channel-disruption protection rates of 68–100% do not prevent identity extraction. This is an unusually direct falsification of a prior defensive claim."
  ],
  "weaknesses": [
    "Missing critical ablation: what does a naïve baseline (feeding protected templates directly into off-the-shelf ArcFace without any distillation) achieve? This is the single most important ablation missing from the paper. If direct template-to-ArcFace already yields high matching accuracy, distillation is unnecessary; if it fails, distillation's contribution is demonstrated. Without this, the paper cannot isolate whether the vulnerability is in the representation itself or requires the specific distillation step, which directly bears on the central claim.",
    "Amazon API vs. Face++ discrepancy in the constrained-attacker setting is not satisfactorily explained: Table 7 shows Amazon API regeneration success of 44–57% versus Face++ rates of 92–96% on the same data — a ~40 percentage point gap. The paper dismisses this as 'The Amazon API is likely more strict or sensitive to AI-generated images,' but this is speculative. This discrepancy significantly undermines confidence in the constrained-attack results, since the two verifiers disagree by a factor of roughly 2x. A calibration experiment on real-to-real LFW pairs using both APIs, or a principled discussion of why the gap arises, is needed.",
    "The linkage gap of ~0.10 in Table 4 between the upper bound (Original-to-Original: 0.882) and the FracFace-to-FracFace result (0.786) is attributed to dataset noise without analysis. The paper claims results 'essentially reach the dataset's theoretical maximum' given 9.3–13% label noise, but 0.786 is 0.096 below the upper bound — a non-trivial gap. Whether this residual gap is due to distillation error, template information loss, or dataset noise is not investigated, which matters for understanding whether any structural defense is partially working.",
    "TPDNE dataset methodology is potentially problematic for identity verification: The paper uses StyleGAN-generated synthetic faces as one of three test datasets and asks Face++ to verify that Arc2Face-regenerated faces match these StyleGAN originals. It is not established that commercial face verifiers, trained and validated on real photographs, are reliably calibrated for cross-generative-model synthetic-to-synthetic comparisons. The paper should verify Face++ reliability on this cross-domain synthetic setting or rely primarily on real-image results (CASIA hold-out, LFW) for its strongest identity claims.",
    "The operational cost of oracle data collection is not addressed: Section 3 describes the attack as requiring ~90K training pairs (reduced to 256 in the minimal experiment), but does not discuss how many queries to the conversion oracle this requires. If the conversion process runs on attacker-controlled service infrastructure, collection is trivial; if it requires querying a rate-limited server, this is a meaningful practical constraint. This detail is important for practitioners assessing realistic risk and is absent from the threat model discussion.",
    "Section 10 (Future Directions) is underdeveloped: The cryptographic hardening and de-identification direction suggestions are presented without analysis of whether they would actually resist FaceLinkGen. For instance, if a key-based conversion still allows the attacker to query the oracle with the key applied locally, the same distillation attack applies. The claim that depth maps or Canny edges are 'identity-agnostic' is unsupported — face geometry encoded in depth maps is identity-discriminative and this specific claim needs either a citation or a caveat."
  ],
  "nice_to_haves": [
    "Ablation on teacher/student architecture choices (FaceNet teacher, MobileNet student) to show the attack generalizes across embedding families rather than being tuned to the ArcFace-Arc2Face coupling.",
    "Systematic data-efficiency learning curve (64 → 256 → 1K → 10K images) for all three PPFR methods, not only FracFace, to characterize where the practical vulnerability floor lies.",
    "Demographic-stratified attack success rates (by race, gender, age group using FairFace labels) to reveal whether certain demographic groups face disproportionate privacy risk — especially given Section 9's soft biometric findings.",
    "Evaluation of DuetFace and FaceObfuscator if source code can be obtained, to strengthen the claim that the vulnerability is systemic across all open-sourced frequency-domain PPFR methods.",
    "Embedding-space t-SNE/UMAP visualizations showing that student-recovered embeddings from protected templates cluster with original ArcFace embeddings by identity, making the linkage vulnerability intuitively immediate.",
    "Direct head-to-head comparison of FaceLinkGen versus the U-Net and StyleGAN reconstruction attacks from prior work under identical test conditions (same split, same metrics) rather than relying on reported numbers from different evaluation setups.",
    "Utility–privacy trade-off curve plotting attack success rate against the defended system's recognition accuracy, making explicit how much headroom exists for improving privacy without sacrificing utility — useful for future PPFR design.",
    "Analysis of failure cases (the ~1–8% where regeneration fails) to determine whether any structural property of templates or identities partially resists the attack.",
    "Formal connection to established privacy frameworks (k-anonymity, differential privacy) even in a discussion capacity, to motivate stronger defensive guarantees.",
    "Pilot test of distillation attack robustness to lightweight post-processing defenses (Gaussian noise injection, quantization of templates) as guidance for practitioners considering low-cost mitigations."
  ],
  "novel_insights": "The paper's most significant insight is the decoupling of pixel-level distortion from identity-level privacy: the information that PPFR systems must preserve for recognition utility (identity-discriminative features) is precisely the information that makes them vulnerable to a distillation attack, and this is a structural property of any recognition-utility-preserving template scheme rather than a flaw in specific implementations. The demonstration that this vulnerability holds even with 256 training images and a generic high-pass proxy (no knowledge of the specific protection mechanism) is particularly compelling, suggesting the vulnerability floor is extremely low. The soft biometric leakage finding (Table 9) — that attributes like race, gender, and age are recoverable from templates at near-original-image accuracy — adds a practically important dimension beyond identity matching. The observation that adversarial de-identification (TIP-IM) suppresses identity in one model's embedding space while leaving it recoverable by a student aligned to a different space is a genuinely novel characterization of why adversarial perturbation-based privacy is fragile: it targets a specific model's decision boundary rather than removing information.",
  "missed_related_work": [],
  "suggestions": [
    "Add the missing naïve baseline: directly feed protected templates into off-the-shelf ArcFace (without distillation) and report matching accuracy and regeneration success. This one experiment would either confirm that distillation is necessary (low baseline accuracy) or reveal that the vulnerability is even simpler than claimed (high baseline accuracy), and in either case it is essential for interpreting the central contribution.",
    "Investigate the Amazon API vs. Face++ discrepancy in Table 7 by running both APIs on a set of real-to-real LFW face pairs to establish their relative operating points, then report this calibration alongside the constrained-attack results. This would allow readers to assess whether the 44–57% Amazon result is a genuine limitation of the constrained attack or an artifact of differing API thresholds.",
    "Clarify the TPDNE evaluation by either (a) reporting whether Face++ achieves expected accuracy on synthetic-to-synthetic real-identity-matched pairs from the same generative model, or (b) marking the CASIA-WebFace hold-out and LFW results as the primary evidence and TPDNE as supplementary.",
    "Reframe Table 6 with a clearer caption specifying exactly which paper and which experimental protocol produced the 'Protection Tested in FracFace' column values, and clarify whether these are FracFace's self-reported protection rates against its own simulated attackers or against the other methods.",
    "In Section 2, explicitly state how many paired oracle queries the training process requires (e.g., ~90K in the full setting, ~256 in the minimal setting) and discuss whether this is feasible under the oracle-access threat model for different deployment scenarios (locally running conversion vs. remote server).",
    "Investigate why MinusFace shows notably lower soft biometric leakage (race accuracy 0.569 vs. 0.673–0.700 for the other methods) — this is a meaningful signal that MinusFace's frequency-domain design may be doing something qualitatively different, and understanding it could guide future defensive design.",
    "Add a caveat to the depth-map/Canny edge suggestion in Section 10, noting that face geometry captured in depth maps is itself identity-discriminative, and that the proposed direction requires empirical validation before being presented as a promising defense."
  ],
  "score": 6.2,
  "score_justification": "Clear conceptual contribution with strong empirical support on an important problem, but held back by a missing critical baseline ablation, an unexplained ~40pp API discrepancy in the constrained setting, and an underdeveloped future-directions section; merits acceptance with revisions but is not a clean accept.",
  "decision": "Accept"
}
```
