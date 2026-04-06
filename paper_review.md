
========================================================================
INDIVIDUAL REVIEWS
========================================================================

────────────────────────────────────────
HARSH CRITIC (deepseek/deepseek-v3.2 via OpenRouter)
────────────────────────────────────────
## Section-by-Section Critical Review

### Title & Abstract
The title clearly reflects the paper's contribution: rethinking identity leakage and introducing FaceLinkGen. The abstract concisely states the problem (over-reliance on pixel-level metrics), summarizes the attack, and presents strong quantitative results (>98.5% linkage, >96% regeneration). The claims are well-supported in the body. However, the abstract could more precisely state that the attack is demonstrated on three specific frequency-domain PPFR methods (PartialFace, MinusFace, FracFace) to avoid overgeneralization. The final sentence about motivating identity-centric evaluation is appropriate.

### Introduction & Related Works
The introduction effectively motivates the core issue: pixel-level reconstruction metrics (PSNR/SSIM) do not adequately capture identity-level privacy. The distinction between pixel similarity and identity similarity is illustrated well with examples (e.g., CanFG, daily photos). The related work covers relevant PPFR methods and model inversion attacks, but the distinction between FaceLinkGen and embedding-based inversion attacks (e.g., Arc2Face) is somewhat blurred. The authors argue that their attack targets structural vulnerabilities in the template generation process, but the method essentially learns a mapping from template to embedding, similar to representation inversion. This distinction should be clarified. The contributions are clearly stated: critiquing the current evaluation paradigm, introducing an identity-centric attack, and showing vulnerabilities across multiple systems.

### Threat Model
This section is a strength. It correctly identifies the primary adversary as the curious/malicious service provider with oracle access to the conversion process, aligning with the original PPFR motivation. It argues convincingly against weaker threat models (e.g., external eavesdroppers) and unrealistic assumptions like model secrecy (e.g., in CanFG). The assumption of no knowledge of architecture/parameters is appropriately strict. However, the discussion about local vs. remote conversion could be streamlined; the key point is that the attacker can obtain paired (original, template) data via queries, which is sufficient for the attack.

### Methods
The method is elegantly simple: distill identity information from protected templates into a standard face embedding space (e.g., ArcFace) using a student network, then regenerate faces via a diffusion model (Arc2Face). This simplicity is intentional and effectively demonstrates that the vulnerability lies in the representation itself. The formulation separating identity and nuisance factors is clear. However, several details need clarification for reproducibility and robustness:
1. **Training data requirement:** The attack requires a dataset of paired (original, template) images, which the attacker can collect under the threat model by querying the oracle. This should be explicitly stated.
2. **Dependency on compatible generative model:** The regeneration attack relies on Arc2Face, which is tied to ArcFace embeddings. The authors note that other embedding/generator pairs could be used, but the general availability of such models is not guaranteed. This limitation should be acknowledged.
3. **Architectural details:** The student model is described as "Antelopev2 with one additional 3x3 Conv2D layer prepended." More specifics (e.g., input/output dimensions, training hyperparameters) would aid reproducibility. The choice of Antelopev2 (as the backbone Arc2Face accepts) versus ArcFace (as the teacher) should be clarified to avoid confusion.

### Attack Vectors
This section clearly describes the two attack vectors: linkage (nearest-neighbor search in embedding space) and regeneration (using Arc2Face). Both are straightforward extensions of the extracted embedding. The discussion of template-to-template vs. face-to-template linkage is useful and connects to standards (ISO/IEC 24745).

### Experiments & Results
The experimental evaluation is extensive and generally convincing. Key strengths:
- Testing three recent frequency-domain PPFR methods (PartialFace, MinusFace, FracFace) across multiple datasets (TPDNE, CASIA-WebFace hold-out, LFW).
- Minimal-resource experiments showing high success even with only 256-800 training images.
- Constrained-assumption attack using a generic high-pass filter as a proxy, demonstrating vulnerability even without knowledge of the conversion process.
- Cross-verification with commercial APIs (Face++, Amazon) adds credibility.
- Comparison with the original reconstruction attacks (e.g., FracFace's protection rates) effectively highlights the discrepancy between pixel-level and identity-level privacy.
- Extension to de-identification (TIP-IM) and non-frequency methods (CanFG) provides preliminary evidence of broader vulnerabilities.

**Weaknesses and missing analyses:**
1. **Ablation studies:** The paper lacks systematic ablations on key components: teacher model choice, student architecture, training set size/distribution, and the impact of the generative model. For instance, how does performance vary with different embedding models (e.g., FaceNet)?
2. **Direct identity classification baseline:** A simpler baseline could directly classify identities from templates (without distilling into a known embedding space). Comparing to such a baseline would help isolate the benefit of the distillation step.
3. **Quantification of identity similarity:** While commercial API verification is a pragmatic metric, reporting cosine similarity between original and extracted embeddings (or between original and regenerated faces) would provide a more direct, model-agnostic measure of identity preservation. Section 7 touches on this but could be integrated into the main results.
4. **Statistical significance:** Results are reported as point estimates without confidence intervals or statistical tests, though sample sizes are large.
5. **Generalization claims:** The attack is demonstrated on three frequency-domain methods, with preliminary results on TIP-IM and CanFG. The claim that similar vulnerabilities "may extend beyond the frequency-domain PPFR family" is reasonable but should be tempered; more evidence would be needed for strong generalization.

### Writing & Clarity
The writing is generally clear and logically structured, though parser artifacts (random numbers, misplaced text) disrupt the flow. In a clean version, the paper would be easy to follow. Some sections (e.g., threat model) are slightly verbose and could be tightened. Figures and tables are referenced appropriately, but their content cannot be assessed due to the extraction. The paper effectively uses examples and tables to support arguments.

### Limitations & Broader Impact
The paper implicitly acknowledges some limitations (e.g., dependency on paired training data, generative model compatibility) but would benefit from a dedicated limitations section. Key limitations to discuss explicitly:
- The attack requires collecting a training set of paired images; while feasible under the threat model, practical barriers (e.g., rate-limiting, enrollment-only conversion) could exist.
- The generative model may not be available for arbitrary embedding spaces.
- The evaluation focuses on specific methods; vulnerabilities in other PPFR families (e.g., cryptographic approaches) are not explored.
- Soft-biometric evaluation is preliminary and limited to FairFace.

Broader impact is positive: the paper reveals critical privacy vulnerabilities in existing systems and advocates for improved evaluation metrics, which could lead to more robust privacy-preserving technologies. Potential negative impact (malicious use) is mitigated by responsible disclosure and the academic context.

### Overall Assessment
This paper makes a significant contribution by challenging the prevailing evaluation paradigm in privacy-preserving face recognition. The core argument—that pixel-level reconstruction metrics do not guarantee identity privacy—is well-motivated and supported by strong empirical evidence. FaceLinkGen is a simple yet powerful attack that achieves high linkage and regeneration success on three recent PPFR systems, demonstrating that identity information remains easily extractable. The paper is timely, as identity-centric privacy is crucial for real-world applications. While the paper could be strengthened by more thorough ablation studies, baseline comparisons, and explicit discussion of limitations, the main findings are compelling and likely to influence future research. For ICLR, which values novel, impactful insights with rigorous evaluation, this paper meets the acceptance bar. I recommend **acceptance with minor revisions**, requesting the authors to address the noted weaknesses, particularly clarifying methodological details and tempering overgeneralization claims.

────────────────────────────────────────
NEUTRAL REVIEWER (deepseek/deepseek-v3.2 via OpenRouter)
────────────────────────────────────────
## Balanced Review

### Summary
This paper argues that the current evaluation paradigm for frequency-domain Privacy-Preserving Face Recognition (PPFR) systems is flawed. The community primarily measures privacy via resistance to pixel-level reconstruction (using PSNR/SSIM), but the authors demonstrate this does not guarantee protection against identity leakage. They introduce FaceLinkGen, a simple distillation-based attack that extracts identity-preserving embeddings from protected templates to perform identity linkage and face regeneration, achieving high success rates on three recent PPFR methods. The work advocates for supplementing pixel-level metrics with identity-centric evaluations.

### Strengths
1.  **Compelling Core Insight and High Significance:** The paper identifies a critical, overlooked flaw in a major sub-field's evaluation methodology. The argument that "preventing pixel reconstruction ≠ preventing identity leakage" is well-motivated, clearly explained (e.g., via the CanFG vs. real-photo example in Table 1/Fig 2), and has significant implications for how PPFR research is conducted and evaluated. This aligns with ICLR's interest in foundational critiques and paradigm-shifting ideas.
2.  **Strong, Reproducible Empirical Evidence:** The attack demonstrates remarkably high effectiveness (e.g., >98.5% linkage, >96% regeneration success) across three distinct, recent PPFR systems (PartialFace, MinusFace, FracFace). Experiments are thorough, using multiple datasets (TPDNE, CASIA-WebFace, LFW) and commercial verification APIs (Face++, Amazon) to ensure robustness. The "Minimal-Resource Attack" (Sec 5.1) and "Assumption-Constrained Attack" (Sec 6) effectively stress-test the vulnerability, showing it persists even with limited data/ knowledge.
3.  **Effective Simplicity and Generalization:** The attack pipeline is intentionally simple (standard distillation + off-the-shelf generator), which strengthens the core message: the vulnerability is inherent to the representations, not a flaw in a complex attack. The successful extension to de-identification systems (TIP-IM, Sec 8) and soft-biometric inference (Sec 9) convincingly shows the generality of the identity-leakage problem beyond the specific frequency-domain PPFR systems initially targeted.

### Weaknesses
1.  **Unclear and Convoluted Threat Model Discussion:** Section 2 meanders through historical intent, different attacker framings, and comparisons to prior work. The core attacker assumption—a service provider with oracle access to the conversion process—is clear, but the lengthy justification and tangents (e.g., discussing TLS, separate entities) detract from clarity. A more focused, structured threat model definition would improve readability.
2.  **Limited Ablation and Analysis of the Attack's Components:** While the attack's simplicity is a strength, a more detailed ablation study would solidify the claims. For instance: How crucial is the choice of teacher model (ArcFace)? What is the impact of the student architecture? How does performance degrade with different amounts of training data or variations in the distillation objective? This would help readers understand the boundaries of the vulnerability.
3.  **Insufficient Comparison to Prior Inversion Attacks:** The paper distinguishes itself from Model Inversion Attacks (MIAs) that aim to recover training data or exploit model embeddings. However, a more direct and quantitative comparison to state-of-the-art MIAs or regeneration attacks (like Zhang et al. [43], which is only discussed briefly) would better position FaceLinkGen's novelty and efficiency. The claim that prior attacks fail (Fig 3) is visual; a quantitative comparison on the same benchmark would be stronger.

### Novelty & Significance
**Novelty:** The paper's primary novelty is conceptual: shifting the evaluation focus in PPFR from pixel-level distortion to *identity-level leakage*. While the use of distillation and generative models for attack is not novel in isolation, its application to reveal this fundamental flaw in frequency-domain PPFR systems is. The demonstration that a simple, generic pipeline suffices is a powerful and novel result.
**Significance:** The significance is very high. The work successfully challenges a foundational assumption in an active research area. If accepted, it should immediately influence how researchers design and evaluate PPFR systems, pushing the community towards more meaningful, identity-centric privacy guarantees. The high success rates against recent, peer-reviewed methods make the critique impossible to ignore.

### Suggestions for Improvement
1.  **Streamline and Clarify the Threat Model:** Restructure Section 2 to first clearly state the adopted threat model (oracle-access insider) and its justification, then briefly contrast it with alternatives (external eavesdropper, knowledge-constrained attacker) to frame the subsequent constrained-attack experiment (Sec 6). Remove tangential discussions.
2.  **Include a Detailed Ablation Study:** Add a subsection or table analyzing the sensitivity of the FaceLinkGen attack. Key variables to test: size/quality of the distillation dataset, architecture of the student network, choice of teacher embedding model, and the loss function. This would provide a more complete picture of the attack's requirements and robustness.
3.  **Strengthen the Comparative Analysis:** Conduct a fair, quantitative comparison between FaceLinkGen and one or two strong baseline attacks (e.g., the U-Net/StyleGAN attacks from FracFace, or a recent MIA) on the same set of protected templates. Report identity verification success rates (e.g., using Face++) to concretely demonstrate the superiority of the identity-centric approach over pixel-reconstruction approaches.
4.  **Expand Discussion on Defenses and Limitations:** Section 10 offers good future directions. It would be beneficial to also explicitly discuss the limitations of the *current* attack. For instance, would it fail against a system that deliberately perturbs the template in the identity-embedding space itself? A brief discussion on what properties a representation *would* need to have to resist such an attack would enhance the paper's constructive impact.

────────────────────────────────────────
SPARK FINDER (deepseek/deepseek-v3.2 via OpenRouter)
────────────────────────────────────────
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare against state-of-the-art model inversion attacks.** The paper only contrasts with the original papers' U-Net/StyleGAN attacks. Without a comparison to recent, strong inversion methods (e.g., Diffusion-Driven Universal Model Inversion [33]), it's unclear if FaceLinkGen's success stems from a novel vulnerability or is simply a different implementation of known attacks.
2. **Test on a broader set of PPFR methods, including non-frequency-domain ones.** The attack is demonstrated on three frequency-domain methods and briefly on CanFG/TIP-IM. To substantiate the claim that identity leakage is a general pitfall, it must be tested on other prominent PPFR families (e.g., FaceObfuscator, DuetFace) and non-frequency cancelable biometrics. If it fails on some, the generality claim is weakened.
3. **Perform an ablation study on the distillation pipeline.** The success of the attack hinges on the student model design. Ablations on the teacher model (e.g., using FaceNet instead of ArcFace), loss functions, and network architecture are needed to show the attack's robustness and that the results are not an artifact of a specific setup.
4. **Evaluate under a more stringent, "no paired data" threat model.** The constrained attack still uses 30 paired samples. A more realistic scenario is an attacker with zero paired samples, potentially using transfer from a public dataset or a different protection method. This tests the core assumption that obtaining some paired data is always feasible.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantify the amount of identity information retained in the template.** The paper shows high attack success but does not measure how much identity-discriminative information remains in the protected template versus the original image (e.g., via mutual information or dimensionality analysis). This is critical to explain *why* the attack works and to assess the fundamental privacy-utility trade-off.
2. **Analyze failure cases and conditions for successful attacks.** The paper reports near-perfect success rates. A systematic analysis of when the attack fails (e.g., for certain demographics, poses, or protection parameter settings) is needed to understand the method's limitations and to inform potential defenses.
3. **Correlate recognition utility with attack success.** The claim is that preserving utility leads to leakage. This should be tested by varying the protection strength (e.g., the number of filtered channels in FracFace) and plotting the corresponding recognition accuracy versus attack success rate. A strong positive correlation would powerfully support the paper's core argument.
4. **Provide statistical significance for embedding similarity claims.** The statement that an image-template similarity is higher than cross-image similarity for the same identity is based on a visual cluster (Fig. 5). Statistical tests (e.g., paired t-test) and effect sizes are needed to confirm this difference is meaningful and not due to noise.

### Visualizations & Case Studies
1. **Visualize the embedding spaces via t-SNE/UMAP.** Show clusters of original image embeddings, protected template embeddings, and the student's recovered embeddings. This would visually demonstrate whether the student successfully maps templates into the correct identity clusters, revealing the structural vulnerability.
2. **Showcase failure cases and borderline successes.** Display original and regenerated faces for identities where the verification score is low or the attack barely passes the threshold. This would concretely illustrate the attack's limitations and the types of faces that are more privacy-resilient.
3. **Visualize the protected templates themselves.** For each method, show what the protected template looks like (e.g., in the spatial domain after inverse transform) to help readers understand the common high-frequency retention that the constrained attack exploits.

### Obvious Next Steps
1. **Propose and evaluate a concrete defense or mitigation.** The paper identifies a vulnerability but leaves defenses as future work. A simple countermeasure (e.g., adding calibrated noise to break the alignment with public embedding spaces) should be proposed and tested to show that the community can act on this finding.
2. **Formalize and standardize an identity-centric evaluation metric.** The paper critiques pixel-level metrics but does not propose a standard alternative. Defining a rigorous metric (e.g., "Identity Leakage Rate" based on commercial verification API success) and benchmarking existing methods with it would be a major contribution.
3. **Release full code and pre-trained models.** For reproducibility and to accelerate follow-up research, the authors must commit to releasing their attack pipeline code, trained student models, and evaluation scripts. This is a standard expectation for ICLR.

────────────────────────────────────────
POTENTIALLY MISSED RELATED WORK (deepseek/deepseek-v3.2:online via OpenRouter)
────────────────────────────────────────
Related work search was skipped.

========================================================================
FINAL CONSOLIDATED REVIEW (deepseek/deepseek-v3.2 via OpenRouter)
========================================================================

## Summary
This paper challenges the prevailing evaluation paradigm in frequency-domain Privacy-Preserving Face Recognition (PPFR), arguing that pixel-level reconstruction metrics (PSNR/SSIM) are insufficient proxies for identity privacy. It introduces FaceLinkGen, a simple distillation-based attack that extracts identity-preserving embeddings from protected templates to perform identity linkage and face regeneration, demonstrating high success rates (>98.5% linkage, >96% regeneration) on three recent PPFR systems.

## Strengths
- **Compelling and Significant Core Insight:** The paper successfully identifies and demonstrates a critical, overlooked flaw in a major research area's evaluation methodology. The argument that preventing pixel reconstruction does not equate to preventing identity leakage is well-motivated with clear examples (e.g., CanFG vs. real photos) and has significant implications for how PPFR research is conducted.
- **Strong and Extensive Empirical Evidence:** The attack achieves remarkably high success across three distinct, recent PPFR systems (PartialFace, MinusFace, FracFace) on multiple datasets, verified by commercial APIs. The "Minimal-Resource" and "Assumption-Constrained" experiments effectively stress-test the vulnerability, showing it persists even with limited data or knowledge of the protection mechanism, reinforcing that the flaw is inherent to the representations.

## Weaknesses
- **Insufficient Analysis of Attack Components and Baselines:** While the attack's simplicity is a strength, the paper lacks systematic ablation studies (e.g., impact of teacher model choice, student architecture, training data size) and a direct, quantitative comparison to the prior reconstruction attacks (U-Net/StyleGAN) it critiques. This makes it harder to fully understand the boundaries of the vulnerability and to precisely quantify the advantage of the identity-centric approach over pixel-reconstruction baselines.
- **Overly Verbose and Occasionally Unclear Threat Model Discussion:** Section 2 meanders through historical intent and comparisons to prior work, which detracts from clarity. The core attacker assumption (service provider with oracle access) is sound, but the presentation could be streamlined to improve readability and focus.

## Nice-to-Haves
- Testing the attack pipeline on a wider range of PPFR methods, including non-frequency-domain approaches, would help assess the generality of the identified vulnerability.
- A more formal proposal for a standardized identity-centric evaluation metric would provide a direct, constructive path forward for the community.
- A t-SNE/UMAP visualization of the original, protected, and recovered embedding spaces could offer an intuitive visual demonstration of the structural vulnerability.

## Novel Insights
The paper's primary novel insight is conceptual: it shifts the focus of privacy evaluation in PPFR from pixel-level distortion to identity-level leakage. While distillation and generative models are not novel in isolation, their application reveals that preserving recognition utility in the evaluated frequency-domain systems inherently preserves identity information in a form easily aligned with public embedding spaces. This demonstrates a fundamental disconnect between the stated privacy goals and the achieved protection, suggesting that the prevailing evaluation paradigm is misleading. The successful preliminary extension to de-identification systems (TIP-IM) further suggests this identity leakage problem may be a more general pitfall in methods that attempt to hide identity from machines while preserving human-perceivable or machine-usable signals.

## Suggestions
- Add an ablation study section or table analyzing the sensitivity of FaceLinkGen to key components (teacher model, student architecture, training data scale) to solidify understanding of the attack's requirements.
- Conduct a direct, quantitative comparison between FaceLinkGen's regeneration success rate (using a commercial verification API) and the pixel-reconstruction attacks from the original PPFR papers (e.g., FracFace's U-Net/StyleGAN) on the same set of protected templates, clearly demonstrating the superiority of the identity-centric objective.

========================================================================
PREDICTED SCORE
========================================================================

Score: 4.8
Decision: N/A
Total Cost: $0.0235
