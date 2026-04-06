
========================================================================
INDIVIDUAL REVIEWS
========================================================================

────────────────────────────────────────
HARSH CRITIC (deepseek/deepseek-v3.2 via OpenRouter)
────────────────────────────────────────
## Section-by-Section Critical Review

### Title and Abstract
The title accurately reflects the paper's contribution: an attack focused on identity leakage in PPFR. The abstract clearly states the core problem (pixel-level metrics inadequately capture identity privacy) and summarizes the attack's high success rates. However, it omits a critical detail: the attack's feasibility hinges on the adversary having **oracle access** to the conversion process (the ability to generate protected templates for arbitrary faces). This is a central assumption of the primary threat model and should be mentioned to avoid overstating the attack's applicability to more constrained settings.

### Introduction and Related Works
The motivation is strong and well-argued. The paper convincingly critiques the over-reliance on pixel-reconstruction metrics (PSNR/SSIM) in the PPFR literature and highlights the conceptual gap between pixel distortion and identity privacy. Contributions are clearly stated. The related work appropriately distinguishes the attack from model inversion attacks on deep embeddings (e.g., Arc2Face) and training-data reconstruction MIAs. A minor weakness is that the novelty of the distillation-based approach relative to prior knowledge distillation attacks on other privacy-preserving systems could be more sharply delineated.

### Threat Model
This section is a strength. It correctly recenters the adversary on the **insider service provider** (the original PPFR motivation) and explicitly compares the assumptions to prior work (e.g., less knowledge required than Mi et al.). The discussion of alternative trust models (local conversion, third-party servers) is thoughtful. The clarification that attacks requiring thousands of online queries (e.g., Zhang et al.) are often impractical is important context. The threat model is consistent with the main experiments.

### Methods
The distillation-based method is elegantly simple, which supports the paper's thesis that the vulnerability is inherent in the representation. The formulation separating identity (\(z_I\)) and nuisance (\(z_N\)) information is intuitive. However, key implementation details are missing, hindering reproducibility:
*   The exact architecture of the student model \(f_s\) is not specified beyond "one additional 3x3 Conv2D layer prepended." What is the base network? How many layers? What is the output dimension?
*   Training hyperparameters (optimizer, learning rate, batch size, weight decay) are omitted, except for brief mentions in Section 5.1.
*   Equation (2) uses cosine similarity, but the loss is written as \(L = 1 - \frac{1}{N}\sum \text{cos}(f_s(t_k), f_t(i_k))\). Is this the actual loss, or is a margin or temperature applied? This detail matters for training stability and performance.

### Attack Vectors
Clear and well-illustrated by Figure 4. The description of linkage (face-to-template, template-to-template) and regeneration attacks is straightforward. A minor point: the linkage attack description assumes a nearest-neighbor search in embedding space but does not discuss scalability for large-scale databases (though this is not a primary concern for the proof-of-concept).

### Experiments and Results
Overall, experiments are extensive and support the claims. Three representative PPFR methods are tested across multiple datasets with commercial verification (Face++, Amazon API). However, several issues need addressing:

1.  **Evaluation Metrics Need Precise Definition:** "Success@5" and "Pass@τ" are used throughout but are ambiguously defined. For example, in Table 2, does "Pass@1e-5" mean the generated image's verification score exceeds the threshold corresponding to a FAR of 1e-5? The caption says "per-image pass rate at the Face++ threshold corresponding to false acceptance rate \(\tau\)." This should be stated explicitly in the main text. For "Success@5", is it sufficient for *any one* of the five generated images to pass at *any* threshold, or at a specific \(\tau\)? The text says "at least one of five generated images passes Face++ verification," which is vague.
2.  **Statistical Reporting:** The results report high percentages but no measures of variance (e.g., confidence intervals, standard deviations over multiple runs/seeds). This is particularly important for the minimal-resource experiments (Section 5.1) and the constrained attack (Section 6).
3.  **Baseline Clarity:** The comparison with FracFace's "Protection" metric in Table 6 is conceptually valid but risks comparing apples and oranges. The paper should more clearly state that FracFace's metric measures frequency-channel disruption, while FaceLinkGen measures *actual identity recovery* via verification. The takeaway is that the former does not guarantee the latter.
4.  **Constrained Attack (Section 6) is Confusing:** The scenario is under-specified. The attacker has "only 30 paired image–template samples for validation (not for training)" but also trains a student model. What data is used for training? The text implies the attacker uses a *high-pass filter proxy* applied to a public dataset (e.g., CASIA-WebFace) for training, and the 30 real pairs are for validation/calibration. This pipeline must be described step-by-step for clarity. The claim that this is "stricter than the 'black-box' scenario in Mi et al. [20]" requires justification, as Mi et al. assume knowledge of the conversion process architecture.

### Attack Under Constrained Assumptions
(See points under Experiments.) The idea of using a high-pass filter as a universal proxy is clever and demonstrates a worrying robustness of the vulnerability. However, the experimental setup needs a clear, detailed description.

### Similarity Distribution
This analysis provides strong, model-agnostic evidence of identity leakage: the embedding from a template is more similar to its source image than two different images of the same person are to each other. This is a compelling result that does not depend on generative models or commercial APIs.

### Transfer to Attacking De-Identification Systems
This is a valuable extension showing the generality of the distillation approach. The results on TIP-IM and CanFG are convincing. However, the paper should note that CanFG's security is explicitly based on model secrecy, which the threat model already invalidates, so this result is somewhat expected. The lack of regeneration results for CanFG is a minor omission.

### Soft Identity Leakage
An important and well-executed analysis. The results show that soft biometrics are largely preserved in the extracted embeddings, raising additional privacy concerns. The comparison to the original image embeddings provides a good upper-bound reference.

### Future Directions
The suggestions are reasonable and point toward more rigorous defenses (cryptographic, key-based). The proposal to invert de-identification goals (remove identity for machines, preserve for humans) is thought-provoking. However, the paper lacks a dedicated **Limitations** section, which is a significant omission for ICLR. Key limitations that must be discussed include:
*   **Dependence on Oracle Access:** The core attack requires the ability to generate protected templates for a large set of known identities. While this fits the primary threat model, its practicality in real-world infiltration scenarios could be discussed.
*   **Dependence on Compatible Generative Model:** The regeneration attack uses Arc2Face, which is tied to ArcFace embeddings. The attack's feasibility for other FR backbones (e.g., FaceNet) depends on the availability of analogous high-quality generative models.
*   **Generalization Beyond Frequency-Domain Methods:** The paper shows preliminary results on CanFG and TIP-IM, but a systematic evaluation of non-frequency-based PPFR is beyond its scope. This limitation should be stated.
*   **Assumption of Retained Utility:** The attack exploits the fact that templates must retain identity information for recognition. It does not apply to methods that fundamentally break this link (e.g., some cryptographic approaches).

### Conclusion
The conclusion succinctly summarizes the paper's message and call to action. It aligns with the abstract and introduction.

### Writing and Clarity
The paper is generally well-written and logically structured. The primary clarity issues are in the description of the constrained attack (Section 6) and the precise definitions of evaluation metrics. Some technical details in the Methods section are too sparse for full reproducibility.

### Limitations & Broader Impact
As noted, a formal limitations section is absent and required. The broader impact is implicit: the work exposes critical vulnerabilities in existing PPFR systems, which should spur the development of more robust, formally evaluated methods. A brief discussion of potential negative societal impact (e.g., the attack methodology could be misused) and mitigation (the work is a security analysis intended to improve defenses) would be appropriate.

## Overall Assessment
This paper makes a significant and timely contribution. It successfully challenges a deeply entrenched evaluation paradigm in PPFR, providing rigorous evidence that state-of-the-art frequency-domain methods leak identity information despite appearing secure under pixel-reconstruction metrics. The attack is simple, effective, and convincingly demonstrated across multiple systems and datasets. The core finding is important for the privacy and security community. For acceptance at ICLR, the paper requires revisions to: 1) add a clear limitations section, 2) provide full methodological details for reproducibility, 3) precisely define evaluation metrics and report statistical variance, and 4) clarify the ambiguous constrained attack scenario. Once these issues are addressed, the paper's contribution is substantial and merits publication.

────────────────────────────────────────
NEUTRAL REVIEWER (deepseek/deepseek-v3.2 via OpenRouter)
────────────────────────────────────────
## Balanced Review

### Summary
This paper critiques the prevailing evaluation paradigm for frequency-domain privacy-preserving face recognition (PPFR) systems, which relies on pixel-level reconstruction metrics (PSNR, SSIM) to measure privacy. The authors argue this fails to capture identity-level leakage. They introduce FaceLinkGen, a simple distillation-based attack that extracts identity embeddings from protected templates and uses them for two tasks: (1) linkage attacks (matching templates to identities) and (2) face regeneration via a diffusion model (Arc2Face). The attack achieves high success rates (e.g., >98.5% matching, >96% regeneration) on three recent PPFR methods (PartialFace, MinusFace, FracFace), demonstrating a significant gap between pixel-distortion metrics and actual identity privacy.

### Strengths
1. **Important Conceptual Critique**: The paper convincingly argues that the field's focus on preventing pixel-level reconstruction is misguided for measuring identity privacy. The evidence (e.g., Figure 2, Table 1) that PSNR/SSIM do not correlate with identity similarity is clear and impactful.
2. **Elegant and Effective Attack Methodology**: FaceLinkGen is notably simple—a standard distillation pipeline to map templates to a known face embedding space (ArcFace). Its high success across multiple state-of-the-art methods, demonstrated with extensive experiments (Tables 2, 3, 4), strongly supports the core claim that identity information is preserved in these "protected" templates.
3. **Comprehensive Evaluation**: The paper validates the attack thoroughly across multiple datasets (TPDNE, CASIA-WebFace, LFW), includes linkage and regeneration attacks, explores a constrained-attacker scenario (Section 6), and extends the analysis to soft biometric leakage (Table 9) and de-identification methods (TIP-IM, CanFG). This breadth strengthens the conclusion about a systemic vulnerability.
4. **Clarity and Reproducibility**: The threat model (Section 2) and method (Section 3) are well-explained. Computational costs and training details (e.g., ~2 hours on an A6000) are provided, facilitating reproduction. The use of public datasets and models (ArcFace, Arc2Face) aids verifiability.

### Weaknesses
1. **Limited Discussion of Attack Assumptions**: The attack requires an "oracle access" threat model (ability to query the transformation with chosen inputs) to collect paired data for distillation. While this aligns with the original PPFR threat model of a malicious service provider, the paper could more explicitly discuss the practical barriers for an *external* attacker without this access, even though the constrained-attack experiment (Section 6) partially addresses this.
2. **Superficial Discussion of Countermeasures**: Section 10 ("Future Directions") suggests paths like cryptographic hardening or de-identification focused on human perception, but these are not explored in depth. The paper would be stronger if it included preliminary experiments or a more rigorous theoretical analysis of what properties a defense would need to resist identity extraction.
3. **Writing and Structure Could Be Tighter**: Some sections (e.g., parts of the Introduction and Threat Model) are verbose. The core conceptual point about the evaluation paradigm mismatch is powerful but sometimes gets buried in detailed descriptions of related work.
4. **Lacks In-Depth Analysis of *Why* the Methods Leak**: While the attack shows *that* identity leaks, a deeper analysis of the inherent properties of frequency-domain transformations that make them vulnerable to this simple distillation (e.g., what information is preserved/alignable) would increase the paper's insight. The high-pass filter proxy attack (Section 6) hints at this but could be expanded.

### Novelty & Significance
**Novelty**: The paper's primary novelty is the shift in perspective from pixel-level to identity-centric privacy evaluation for PPFR. The FaceLinkGen attack itself is a novel application of a straightforward distillation pipeline to expose this vulnerability; its simplicity is a feature that underscores the severity of the problem. The extension to soft biometrics and de-identification methods also shows broader relevance.
**Significance**: The findings are highly significant for the PPFR research community. They directly challenge the privacy guarantees claimed by several recent, well-cited papers (PartialFace, MinusFace, FracFace). By demonstrating that high "protection rates" in frequency channels do not prevent identity extraction, the paper necessitates a reevaluation of both defense design and evaluation metrics. This aligns well with ICLR's emphasis on foundational insights and rigorous evaluation.

### Suggestions for Improvement
1. **Strengthen the Analysis of the Vulnerability's Root Cause**: Include a more formal or empirical analysis (e.g., probing the information content of templates) to explain *why* these frequency-domain representations remain so linearly alignable to identity embeddings. This would transform the finding from an empirical demonstration to a deeper principle.
2. **Deepen the Discussion on Defenses**: Expand Section 10 with a more concrete analysis. For example, briefly experiment with or simulate a simple key-based approach to show how it would break the distillation attack, or more rigorously define the necessary conditions for a template to be both useful for recognition and resistant to identity extraction.
3. **Clarify and Contextualize Threat Models**: More clearly delineate the capabilities required for the main attack (oracle access) versus the constrained attack. Discuss the practical implications for different real-world adversaries (service provider vs. eavesdropper) to help readers assess the real-world risk.
4. **Improve Presentation**: Streamline the introductory sections to foreground the core conceptual contribution faster. Consider moving some detailed related work comparisons to an appendix. Ensure figure references in the review copy are correctly parsed.
5. **Add a Limitation Section**: Explicitly acknowledge limitations, such as the attack's dependence on the availability of a powerful face generator (Arc2Face) aligned with the teacher embedding space, and the assumption that the attacker can train on a dataset without identity overlap with the target.

────────────────────────────────────────
SPARK FINDER (deepseek/deepseek-v3.2 via OpenRouter)
────────────────────────────────────────
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Direct comparison against state-of-the-art reconstruction attacks.** The paper claims pixel-level reconstruction is insufficient, but it does not benchmark FaceLinkGen against recent, strong reconstruction baselines (e.g., advanced MIAs, diffusion-based inversion) on the *same* protected templates to quantitatively show its superior identity leakage. Without this, the claim that identity leakage is a more critical metric is not fully substantiated.
2. **Evaluation on PPFR methods designed to explicitly thwart identity linkage.** The attack is only tested on three frequency-domain methods. To prove the pervasiveness of the vulnerability, it must be evaluated on PPFR methods that claim to protect against linkage (e.g., crypto-based or key-binding approaches like CryptoFace). Their absence weakens the claim that identity leakage is a fundamental flaw.
3. **Ablation on the distillation pipeline's data efficiency and model dependence.** While a minimal-resource experiment is mentioned, a systematic ablation is missing: how does performance degrade with fewer paired samples (e.g., 10, 50)? How crucial is the choice of teacher model (ArcFace vs. others)? Without this, the attack's practical feasibility under realistic constraints is unclear.
4. **Testing on the most recent and strongest PPFR methods.** The paper attacks methods from 2023-2025. ICLR expects evaluation on the most current state-of-the-art. Omitting very recent high-profile PPFR works (especially those possibly designed after awareness of such attacks) leaves the paper's impact statement incomplete.

### Deeper Analysis Needed (top 3-5 only)
1. **A theoretical or empirical analysis of *why* identity information persists.** The paper shows it *can* be extracted, but not *why* the transformations fail to remove it. A quantitative analysis (e.g., mutual information estimation, rank of feature space) between templates and identity embeddings is needed to move from demonstration to diagnosis. Without this, the work is more of an attack demo than a foundational critique.
2. **Clarification and defense of the threat model.** The threat model discussion is convoluted and mixes different adversary capabilities. A clear, consolidated threat model table contrasting assumptions with prior work (and justifying the "oracle access" assumption) is essential for a security paper at ICLR. The current text risks being dismissed as attacking a strawman.
3. **Analysis of potential contamination in evaluation datasets.** The use of Arc2Face (trained on synthetic data from SD 1.5) and commercial APIs (trained on large, unknown datasets) for verification raises concerns about data contamination, especially on LFW/WebFace. A sensitivity analysis or justification that this does not inflate results is critical for trust in the reported high success rates.
4. **Failure case analysis.** The paper reports near-perfect success. Showing and analyzing cases where the attack fails (e.g., certain identities, template types) is crucial to understand the method's limitations and the defenses' residual strengths, which is expected for a balanced ICLR submission.

### Visualizations & Case Studies
1. **Side-by-side visual comparisons of failed reconstruction attacks vs. successful FaceLinkGen regenerations.** Figure 3 is a start, but a systematic grid for many identities, comparing outputs of pixel-reconstruction baselines (U-Net, StyleGAN) and FaceLinkGen against originals, would visually prove the core claim that identity persists when pixels do not.
2. **Visualization of the template space.** t-SNE or UMAP plots comparing original image embeddings, protected template embeddings (from the student model), and embeddings of different identities would powerfully illustrate that template embeddings cluster by identity, directly revealing the leakage.
3. **Case studies on "non-ideal" faces.** Show results on faces with heavy occlusion, unusual poses, or low quality to demonstrate the attack's robustness or expose its boundaries, which is important for assessing real-world impact.

### Obvious Next Steps
1. **Propose and evaluate a concrete, measurable identity-leakage metric.** The paper criticizes PSNR/SSIM but only uses ad-hoc verification pass rates. For ICLR, the authors should propose a standardized identity leakage metric (e.g., linkage accuracy under a defined threat model, verified by a standardized model) and benchmark all methods against it. This is the logical next step for a paper "rethinking evaluation."
2. **Investigate simple defenses or modifications to existing PPFR.** A compelling next step would be to take one attacked method (e.g., FracFace) and show that even a simple modification (e.g., adding a secret key, non-linear projection) can break the distillation attack, providing a constructive path forward. Without this, the critique feels purely destructive.
3. **Extend evaluation to a broader suite of non-frequency-domain methods.** The paper has a small section on TIP-IM and CanFG, but a comprehensive evaluation on a wider range of de-identification and adversarial perturbation methods is needed to solidify the claim that identity leakage is a widespread issue beyond frequency-domain PPFR.

────────────────────────────────────────
POTENTIALLY MISSED RELATED WORK (deepseek/deepseek-v3.2:online via OpenRouter)
────────────────────────────────────────
Related work search was skipped.

========================================================================
FINAL CONSOLIDATED REVIEW (deepseek/deepseek-v3.2 via OpenRouter)
========================================================================

## Summary
This paper critiques the standard evaluation paradigm for frequency-domain privacy-preserving face recognition (PPFR), which measures privacy via resistance to pixel-level reconstruction (PSNR/SSIM). The authors argue this fails to capture identity-level leakage. They introduce FaceLinkGen, a simple distillation-based attack that extracts identity embeddings from protected templates and uses them for high-accuracy linkage and face regeneration, demonstrating that three recent PPFR methods (PartialFace, MinusFace, FracFace) leak identity information despite appearing secure under pixel-distortion metrics.

## Strengths
- **Important conceptual critique and shift in perspective:** The paper convincingly argues that pixel-level reconstruction metrics are insufficient for assessing identity privacy, a foundational point supported by clear evidence (e.g., Figure 2, Table 1) that PSNR/SSIM do not correlate with identity similarity.
- **Elegant and effective attack methodology:** The attack is notably simple—a standard distillation pipeline to align protected templates with a public face embedding space (ArcFace). Its high success across multiple state-of-the-art methods (>98.5% matching, >96% regeneration) strongly proves the core claim that identity information is preserved in these "protected" representations. The simplicity is a feature that underscores the severity of the vulnerability.

## Weaknesses
- **Lacks a formal limitations section:** For an ICLR submission, the omission of a dedicated section discussing the attack's assumptions and boundaries is a significant shortcoming. Key limitations that should be explicitly stated include: the attack's dependence on oracle access to the conversion process (fitting the primary threat model but limiting external applicability); its reliance on the availability of a compatible, high-quality generative model (Arc2Face) for regeneration; and its focus on frequency-domain methods, with only preliminary results on other paradigms.
- **Ambiguous description of the constrained attack scenario:** Section 6 introduces an attack under "constrained assumptions" but the setup is confusingly described. It is unclear what data is used for training versus validation, and the claim that this scenario is "stricter" than prior black-box models requires clearer justification. This undermines the clarity and reproducibility of an otherwise important experiment.

## Nice-to-Haves
- **Deeper analysis of the vulnerability's root cause:** A more formal or empirical analysis (e.g., probing the information content or linear alignability of the template representations) could transform the finding from a powerful demonstration into a deeper diagnostic principle.
- **Proposal for a standardized identity-leakage metric:** While the paper effectively criticizes PSNR/SSIM, suggesting a concrete, standardized metric for identity leakage (e.g., linkage accuracy under a defined threat model) would be a constructive step toward improving evaluation practices.

## Novel Insights
The paper's primary novel insight is the clear demonstration that the prevailing evaluation paradigm for frequency-domain PPFR—which equates privacy with resistance to pixel-level reconstruction—is fundamentally misaligned with the goal of protecting identity. The attack shows that identity-discriminative information can be easily extracted via a standard distillation procedure even when pixel recovery is infeasible, revealing a systemic vulnerability in current methods. This insight necessitates a reevaluation of both defense design and evaluation metrics in the field.

## Suggestions
- Add a dedicated "Limitations" section to explicitly discuss the attack's assumptions, dependencies, and scope.
- Revise Section 6 ("Attack Under Constrained Assumptions") to provide a clear, step-by-step description of the experimental setup, including what data is used for training the proxy model and how the 30 validation pairs are utilized. Justify the comparison to prior black-box models more clearly.
- Ensure all evaluation metrics ("Success@5", "Pass@τ") are precisely defined in the main text, specifying the verification threshold and success criterion unambiguously.

========================================================================
PREDICTED SCORE
========================================================================

Score: 5.3
Decision: N/A
Total Cost: $0.0270
