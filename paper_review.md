
========================================================================
INDIVIDUAL REVIEWS
========================================================================

────────────────────────────────────────
HARSH CRITIC (deepseek/deepseek-v3.2 via OpenRouter)
────────────────────────────────────────
## Critical Issues
None. The paper’s central claims—that pixel-level reconstruction metrics are inadequate for assessing identity privacy in frequency‑domain PPFR, and that identity information can be extracted via a simple distillation pipeline even when pixel recovery fails—are well supported by the evidence presented. There are no structural flaws in the argument or methodology, and the experimental results are sufficient to establish the claims.

## Section‑by‑Section Notes

**Abstract & Introduction** – Clearly states the problem and contribution. The critique of the prevailing evaluation paradigm (reliance on PSNR/SSIM) is well‑motivated and grounded in prior work.

**Threat Model (Section 2)** – Thorough and convincingly argued. The paper correctly recenters the adversary as the service provider with oracle access, aligning with the original PPFR intent. It also fairly compares its assumptions to those of prior work (e.g., Mi et al. assume more knowledge).

**Methods (Section 3)** – The distillation pipeline is straightforward but appropriate for the demonstration. The choice of ArcFace as teacher and Arc2Face as generator is justified and does not limit generality, as the approach can work with other embedding/generator pairs.

**Experiments (Sections 5–8)** – Extensive and rigorous. The attack is evaluated on three recent systems across multiple datasets, with both linkage and regeneration metrics. The minimal‑resource experiments (800, 256 images) convincingly show that very little data is needed. The constrained‑assumption attack (Section 6) using a generic high‑pass filter is a clever extension that further underscores the vulnerability. The extension to de‑identification systems (TIP‑IM, CanFG) is preliminary but adds breadth.

**Soft‑Biometric Leakage (Section 9)** – A valuable addition that shows the attack also recovers sensitive attributes (age, gender, race), highlighting a broader privacy risk.

**Future Directions & Conclusion** – The suggestions (cryptographic hardening, human‑perceptual de‑identification) are thoughtful and point to meaningful next steps.

**Minor points** –  
- The paper occasionally uses “we show” or “we demonstrate” without immediate reference to a figure/table (e.g., the claim about similarity distributions in Section 7). The relevant figure (Figure 5) is not included in the extracted text, but this is likely a parser artifact.  
- The discussion of dataset noise (WebFace 9.3%–13.0% noise) is used to explain less‑than‑perfect linkage results; this is reasonable but could be quantified more precisely.

## Strengths
- **Important conceptual contribution:** Identifies a critical gap in how PPFR systems are evaluated—pixel‑level reconstruction metrics do not capture identity‑level privacy. This insight is likely to influence future research in the field.
- **Effective, simple attack:** FaceLinkGen is a straightforward distillation pipeline that requires only paired data and a standard generative model. Its high success rates (≥98.5% linkage, ≥96% regeneration) on three recent systems demonstrate that the vulnerability is inherent in the representations, not an artifact of a complex attack.
- **Comprehensive experimentation:** Tests on multiple datasets (TPDNE, CASIA‑WebFace hold‑out, LFW), includes minimal‑resource ablations, extends to a constrained‑assumption setting, and even examines soft‑biometric leakage. The use of commercial verification APIs (Face++, Amazon) adds credibility.
- **Clear threat‑model analysis:** Correctly argues that the service provider is the primary adversary and that oracle access is realistic, addressing potential objections head‑on.

## Overall Assessment
This paper makes a strong, well‑supported contribution that meets ICLR’s standards for novelty, technical soundness, and impact. It convincingly shows that current frequency‑domain PPFR systems leak identity information despite their claims of privacy based on pixel‑level reconstruction metrics. The attack is simple yet devastating, the experiments are thorough, and the critique of the evaluation paradigm is timely and important. The work should be accepted as is; its findings will likely spur needed changes in how privacy is assessed in face recognition research.

Additional transferable weaknesses from matched human reviews:
I'll help you retrieve relevant human-written reviews and identify applicable strengths/weaknesses for the current paper. Let me start by reading the paper and then systematically searching for related reviews.Now let me search for relevant human reviews in the ICLR 2025 dataset. I'll use multiple search queries to find papers and reviews that address similar topics, methods, and potential weaknesses.Let me search for more relevant reviews on evaluation protocols, attack validation, and privacy evaluation methodologies:Let me read a few more relevant review files to identify common weakness patterns that apply to the FaceLinkGen paper:Now let me search for reviews that discuss specific evaluation methodologies, baseline comparisons, and attack robustness:Let me read the full OE67D1Oatr.md review for more insights:Based on my analysis of relevant human reviews in the ICLR 2025 dataset, here are the identified **weaknesses** that plausibly apply to the FaceLinkGen paper:

---

## Weaknesses

### 1. Limited Evaluation Against Privacy Defenses
The paper does not evaluate how the attack performs against various privacy-preserving defenses that could be deployed in PPFR systems (e.g., differential privacy, noise injection, gradient perturbation techniques). Similar to a related privacy attack paper that faced this critique, this limits the practical understanding of the attack's robustness: 

> "While the paper compares FGL with prior attack methods, it does not assess its effectiveness against some privacy defenses, such as gradient perturbation or differential privacy techniques. They are widely used in federated learning." (from LJULZNlW5d.md)

### 2. Generalizability Limited Primarily to Frequency-Domain PPFR
While the paper briefly evaluates TIP-IM and CanFG, the main focus remains on frequency-domain PPFR systems. The limited evaluation on non-frequency-domain and alternative PPFR paradigms constrains the generalizability claims. As noted in a similar privacy study:

> "The paper mainly focuses on high-resolution facial images, which limits generalizability to other data types, such as medical or financial datasets often used in federated learning." (from LJULZNlW5d.md)

This applies here to the domain/method generalization: the frequency-domain assumption is core to the attack's success.

### 3. Insufficient Analysis of Result Dependencies on Key Parameters
The paper presents strong attack results but provides limited analysis on how different system parameters, architectural choices, or training configurations affect the attack success. Following critiques of related adversarial work:

> "The experimental section (4) lacks some comments on the results, i.e., the results are presented without analyzing if they support the claims presented...an evaluation of the trade-offs between the number of triggers and their stealthiness would strengthen the paper." (from OE67D1Oatr.md)

Specifically, the paper could analyze sensitivity to: student model architecture variations, dataset quality/noise levels, and template format differences across systems.

### 4. Limited Comparison with Alternative Attack Methods on PPFR
The paper does not compare with other potential attack methodologies on privacy-preserving face recognition systems or discuss how FaceLinkGen compares to alternative inversion attacks adapted to this domain. Similar reviews note:

> "Comparison with baseline methods is missing. What are the alternative approaches...It is not clear in the current paper." (from V7PYbRzD0h.md)

Cross-method comparison would strengthen the contribution by clarifying what makes the distillation-based approach particularly effective compared to other template-attack strategies.

### 5. Lack of Formal Privacy Guarantees and Theoretical Analysis
While the paper provides empirical evaluation with high success rates, it lacks formal privacy-theoretic analysis or bounds on the privacy leakage. Privacy-focused methods are critiqued when relying solely on empirical evidence:

> "The main flaw is that the proposed method is just an empirical method and does not have any theoretical privacy-preserving analysis/guarantee." (from onvN3zsNMI.md)

This applies inversely here: the attack demonstrates empirical success but provides no formal analysis of when and why identity information is recoverable from templates.

---

## Strengths

### 1. Comprehensive Identity-Centric Evaluation Paradigm
The paper correctly identifies and addresses a fundamental gap in PPFR evaluation—that pixel-level metrics (PSNR, SSIM) do not capture identity-level privacy. The shift to identity-centric metrics using commercial face verification systems (Face++, Amazon) is a significant methodological contribution that moves beyond traditional reconstruction-focused evaluation.

### 2. Extensive Cross-Dataset Validation
The evaluation across multiple datasets (CASIA-WebFace, LFW, TPDNE) with attention to distribution shift and dataset noise demonstrates thorough experimental design. The inclusion of synthetic data (TPDNE) to avoid training data contamination shows methodological rigor.

### 3. Practical and Accessible Attack Design
The simplicity and low computational cost of the attack (0.80-1.60 USD on A6000 GPU) demonstrates that identity leakage is fundamental to the representation rather than dependent on complex adversarial optimization, strengthening the argument about structural vulnerabilities.

────────────────────────────────────────
NEUTRAL REVIEWER (deepseek/deepseek-v3.2 via OpenRouter)
────────────────────────────────────────
## Balanced Review

### Summary
This paper challenges the prevailing evaluation paradigm in frequency-domain privacy-preserving face recognition (PPFR), which equates privacy with resistance to pixel-level reconstruction (measured by PSNR/SSIM). The authors argue that preventing pixel recovery does not necessarily prevent identity leakage. They introduce FaceLinkGen, a simple distillation-based attack that extracts identity embeddings from protected templates and uses them for linkage attacks and identity-consistent face regeneration, demonstrating high success rates on three recent PPFR systems.

### Strengths
1. **Compelling Critique of Evaluation Paradigm**: The paper provides a clear, well-argued case that pixel-level metrics are insufficient for measuring identity privacy. The evidence—showing that high PSNR/SSIM does not correlate with identity similarity and vice-versa—effectively undermines a foundational assumption in much prior work.
2. **Strong Empirical Results**: The attack achieves remarkably high success rates (e.g., >98.5% matching, >96% regeneration) across three distinct, recent PPFR methods (PartialFace, MinusFace, FracFace) and multiple datasets (TPDNE, CASIA-WebFace, LFW). The use of commercial APIs (Face++, Amazon) for verification adds credibility.
3. **Minimal Attack Assumptions and Low Cost**: The attack is practical, requiring only black-box oracle access to the transformation and a small public dataset. The "minimal-resource" experiment (success with only 256 training images) and low estimated cost (~$0.80-$1.60) powerfully demonstrate the vulnerability's severity and accessibility.
4. **Broad Analysis**: The paper extends the analysis beyond core PPFR to include adversarial de-identification (TIP-IM) and a non-frequency method (CanFG), showing the generality of the identity-leakage concern. The investigation of soft-biometric (age, gender, race) leakage further highlights the privacy risks.

### Weaknesses
1. **Insufficient Explanation for Attack's Effectiveness**: While the attack is simple and effective, the paper lacks a deep analysis of *why* the distillation works so well. A more detailed theoretical or empirical investigation into what identity-revealing information remains in the templates (e.g., analyzing the feature space) would strengthen the contribution.
2. **Limited Discussion of Defense Implications**: The suggested future directions (cryptographic methods, human-perception-focused de-identification) are brief and somewhat speculative. A more concrete discussion of how existing systems could be patched or what properties a robust defense must have would be valuable for the community.
3. **Missing Implementation Details**: Key details about the student model architecture (beyond "one additional 3x3 Conv2D layer") and specific training hyperparameters (learning rate, optimizer) are omitted, which slightly hinders reproducibility despite the overall simple pipeline.
4. **Overstatement on Generality**: The claim that the vulnerability "may extend beyond the frequency-domain PPFR family" is supported only by preliminary results on TIP-IM and CanFG. A more systematic evaluation of diverse non-frequency methods would be needed to substantiate this broader claim.

### Novelty & Significance
**Novelty:** The core insight—decoupling pixel-level reconstruction from identity-level leakage and proposing an identity-centric evaluation—is novel and impactful. While semantic-level inversion attacks exist in other domains (e.g., federated learning), their application to challenge the foundational evaluation metrics of frequency-domain PPFR is new.
**Significance:** The work is highly significant for the PPFR and face privacy communities. It exposes a critical flaw in how state-of-the-art methods are evaluated and validated. The high success rates of a simple attack necessitate a re-evaluation of the privacy guarantees offered by these systems. This paper could shift the evaluation standards in future research, pushing the field towards more rigorous, identity-focused privacy metrics.

### Suggestions for Improvement
1. **Deepen the Analysis of Template Vulnerability**: Conduct an ablation study or feature visualization to identify what specific information in the protected templates is being leveraged by the student model. This would help explain the root cause of the leakage and guide future defense design.
2. **Provide More Concrete Defense Guidelines**: Expand the "Future Directions" section into a more substantive discussion. For example, analyze the trade-offs between recognition utility and identity privacy more formally, or propose a specific benchmark for identity-centric evaluation.
3. **Clarify Methodological Details**: Include a clear description of the student network architecture and training parameters in an appendix or supplementary material to ensure full reproducibility.
4. **Strengthen the Generality Claim**: If space permits, include a more extensive evaluation of one or two additional non-frequency-based PPFR or de-identification methods to better support the claim that the issue is not confined to frequency-domain techniques.

────────────────────────────────────────
SPARK FINDER (deepseek/deepseek-v3.2 via OpenRouter)
────────────────────────────────────────
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Test a broader range of PPFR methods, especially non-frequency-domain ones.** The paper only provides pilot studies on CanFG and TIP-IM. To substantiate the claim that identity leakage is a general flaw, rigorous attacks on other major PPFR systems (e.g., FaceObfuscator, DuetFace) and de-identification methods are necessary. Without this, the scope of the vulnerability remains unclear.
2. **Ablate the dependency on specific face recognition and generative models.** The attack uses ArcFace and Arc2Face. To claim independence from the server's model, experiments must show the attack works when the student/teacher uses a different embedding model (e.g., FaceNet) and a different compatible generator.
3. **Directly compare against state-of-the-art model inversion attacks (MIAs).** The paper dismisses pixel-reconstruction MIAs but does not quantitatively compare FaceLinkGen's identity leakage performance against modern MIAs on the same PPFR systems. This comparison is needed to prove the superiority of the identity-centric approach.
4. **Validate the constrained attack (high-pass proxy) on a more diverse set of systems.** The high-pass filter attack is shown to work on three frequency methods, but its effectiveness on non-frequency or more complex obfuscation schemes is untested. This limits the claim about shared vulnerabilities.

### Deeper Analysis Needed (top 3-5 only)
1. **Analyze what identity-relevant information is retained in the templates.** Use feature visualization or attribution methods (e.g., Grad-CAM on the student model) to show which components of the protected template are leveraged for identity extraction. This is critical to prove the vulnerability is inherent to the representation, not just the attack.
2. **Systematically analyze failure cases.** The attack has near-perfect but not 100% success. A qualitative and quantitative analysis of failures (e.g., specific identities, image conditions) is needed to understand the attack's limitations and the conditions under which privacy might be preserved.
3. **Provide a quantitative correlation analysis between pixel-level and identity-level metrics.** The paper claims a disconnect but only shows anecdotal examples (Table 1). A large-scale correlation study (e.g., computing Pearson correlation between PSNR/SSIM and face verification scores across a dataset) is required to statistically support this core argument.

### Visualizations & Case Studies
1. **Show comprehensive visual results for all attack vectors, including failures.** Figure 1 shows limited regeneration examples. To properly assess the attack, include side-by-side visualizations for linkage attacks (e.g., query face, top-matched template, regenerated face) and for all evaluated methods, highlighting both successes and characteristic failures.
2. **Visualize the protected templates themselves.** Display frequency-domain visualizations (e.g., magnitude/phase spectra) of the original and protected images to illustrate what information is retained/removed. This would ground the claim that high-frequency identity cues persist.
3. **Case studies on cross-database linkage.** Demonstrate concrete examples where the attack links a protected template from one service to a public face image from another source (e.g., social media), illustrating the real-world privacy threat.

### Obvious Next Steps
1. **Test a simple defensive baseline, such as a key-based approach.** The paper suggests cryptographic hardening as a future direction but does not test even a naive version (e.g., XOR with a secret key) to show it mitigates the attack. Including this would strengthen the call for new defenses.
2. **Discuss the attack's feasibility under more restrictive threat models.** The paper focuses on an oracle-access insider. A more detailed analysis of the attack's data and query requirements under a pure black-box scenario (no access to conversion process) is needed to fully assess the practical risk.

### Other stuff you can add
1. **Include a responsible research statement.** Given the high attack success, a discussion on the potential misuse and ethical considerations of releasing such an attack is expected for ICLR.
2. **Clarify the evaluation metric for "regeneration success."** The paper uses Face++ verification at certain FAR thresholds. It should explicitly discuss whether this commercial API's decision threshold is an appropriate standard for judging identity leakage, or if a more fundamental metric (e.g., cosine similarity in embedding space) should be the primary measure.

────────────────────────────────────────
POTENTIALLY MISSED RELATED WORK (deepseek/deepseek-v3.2:online via OpenRouter)
────────────────────────────────────────
Related work search was skipped.

========================================================================
FINAL CONSOLIDATED REVIEW (deepseek/deepseek-v3.2 via OpenRouter)
========================================================================

## Summary
This paper challenges the prevailing evaluation paradigm in frequency-domain Privacy-Preserving Face Recognition (PPFR), which equates privacy with resistance to pixel-level reconstruction (measured by PSNR/SSIM). The authors argue that preventing pixel recovery does not necessarily prevent identity leakage. They introduce FaceLinkGen, a simple distillation-based attack that extracts identity embeddings from protected templates for linkage attacks and identity-consistent face regeneration, achieving high success rates on three recent PPFR systems.

## Strengths
- **Foundational Critique of Evaluation Standards:** The paper provides a compelling, evidence-backed argument that pixel-level metrics are insufficient for assessing identity privacy, a core assumption in much prior PPFR work. This is demonstrated through examples (Table 1, Figure 2) and the high success of an identity-centric attack, which could shift evaluation practices in the field.
- **Strong Empirical Validation with a Simple Attack:** The proposed FaceLinkGen attack is straightforward but devastatingly effective. It achieves over 98.5% linkage accuracy and above 96% regeneration success across three distinct, recent PPFR methods (PartialFace, MinusFace, FracFace) and multiple datasets. The "minimal-resource" experiments (success with only 256 training images) powerfully demonstrate that the vulnerability is inherent and easily exploitable.
- **Comprehensive Threat Model and Scope Extension:** The threat model analysis correctly recenters the primary adversary as the service provider with oracle access, aligning with the original PPFR intent. The work also extends beyond core PPFR to show preliminary vulnerabilities in adversarial de-identification (TIP-IM) and a non-frequency method (CanFG), and quantifies soft-biometric (age, gender, race) leakage, broadening the privacy implications.

## Weaknesses
### Major:
*None.* The paper's core claims are well-supported, and no weakness fundamentally undermines its contribution that current pixel-level evaluation is inadequate and identity information is easily extractable.

### Minor:
- **Analysis of the Vulnerability's Root Cause:** While the attack is highly effective, the paper provides limited analysis of *why* the distillation works so well or what specific identity-revealing information persists in the protected templates. A deeper investigation (e.g., feature space analysis) would strengthen the understanding of the representation's inherent flaw.
- **Defense Discussion is Preliminary:** The suggested future directions (cryptographic methods, human-perception-focused de-identification) are brief. A more concrete discussion of the properties a robust defense must have or the fundamental trade-offs involved would provide greater guidance to the community.

### Trivial:
- **Sparse Implementation Details:** Key details like the specific student model architecture beyond a prepended convolutional layer and exact training hyperparameters are omitted. However, the pipeline is described as simple and standard, and the core results are highly reproducible from the described methodology.

## Nice-to-Haves
- A more systematic evaluation of one or two additional non-frequency-based PPFR or de-identification methods would strengthen the claim that identity leakage concerns extend beyond the frequency-domain family.
- A quantitative, large-scale correlation analysis between pixel-level metrics (PSNR/SSIM) and identity-verification scores would provide stronger statistical support for the central argument of their disconnect.
- Visualizing the protected templates (e.g., frequency spectra) could help illustrate what visual information is retained or removed, grounding the technical discussion.

## Removed Points
*These points are flagged to be removed; treat them with caution.*

**Strengths Removed:**
- *"The paper is well-written."* (Generic, applies to any competent paper.)
- *"The topic is important."* (Generic.)
- *"The experiments are extensive."* (Already covered by the specific, evidence-backed strength of "Strong Empirical Validation".)

**Weaknesses Removed:**
- **"Missing comparison with alternative attack methods."** *Justification:* The paper's contribution is to establish a new identity-centric evaluation paradigm and a simple, effective attack that reveals a fundamental flaw. A direct comparison with pixel-reconstruction attacks is not required to prove this point, as the paper already shows those attacks fail (Figure 3). Demanding this is scope creep.
- **"Limited evaluation against formal privacy defenses (e.g., differential privacy)."** *Justification:* The paper evaluates state-of-the-art PPFR systems on their own terms. These systems do not incorporate such defenses, so criticizing their absence is evaluating the paper against a standard outside its stated scope.
- **"Overstatement on generality."** *Justification:* The paper makes a measured claim: "preliminary results suggesting that similar vulnerabilities *may* extend." It explicitly evaluates TIP-IM and CanFG to support this. This is a reasonable scope extension, not an overstatement.
- **"Lacks formal privacy-theoretic analysis."** *Justification:* This is an empirical security evaluation paper demonstrating a concrete attack. Theoretical guarantees are not standard for this type of contribution; the paper is evaluated correctly on its empirical soundness.
- **"Insufficient analysis of result dependencies on parameters."** *Justification:* The paper includes meaningful ablation on training data size (Section 5.1). A full hyperparameter sweep is not required to establish the core claim that the attack works with minimal resources.
- **"Missing implementation details."** *Justification:* While more details are nice-to-have, the description of the student model and training is sufficient to understand and likely replicate the simple distillation pipeline. This is a minor point, not a major weakness.

## Suggestions
- In the revision, consider adding a short ablation or visualization (e.g., using Grad-CAM on the student model) to illustrate which regions or features of the protected template are most leveraged for identity extraction. This would directly address the minor weakness concerning the root cause analysis.
- Expand the "Future Directions" section by one paragraph to more concretely discuss the necessary properties for a defense that prevents both pixel-level and identity-level leakage, potentially framing it as a set of design principles or open challenges.

**Assessment by Axis:**
- **Novelty:** High. The core insight of decoupling pixel-level reconstruction from identity-level privacy and proposing an identity-centric evaluation paradigm is novel and impactful for the PPFR field.
- **Technical Soundness:** High. The methodology is simple but appropriate, the threat model is rigorously argued, and the experiments are comprehensive, controlled, and support all claims.
- **Empirical Support:** High. Extensive experiments across three systems, multiple datasets, resource constraints, and even preliminary tests on other privacy paradigms provide strong, convincing evidence.
- **Significance:** High. The work exposes a critical flaw in the evaluation of state-of-the-art PPFR systems, demonstrating they do not provide the identity privacy they claim. This should motivate a significant shift in how privacy is assessed in this research area.
- **Clarity:** High. The paper is well-structured, the argument is logical and easy to follow, and the figures and tables effectively support the narrative.

========================================================================
PREDICTED SCORE
========================================================================

Score: 8.0
Decision: N/A
Total Cost: $0.0231
