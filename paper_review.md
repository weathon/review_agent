
========================================================================
INDIVIDUAL REVIEWS
========================================================================

────────────────────────────────────────
HARSH CRITIC (qwen/qwen3.5-plus-02-15 via OpenRouter)
────────────────────────────────────────
## Section-by-Section Critical Review

### Abstract & Introduction
The abstract clearly articulates the core thesis: pixel-level reconstruction metrics (PSNR/SSIM) are insufficient for evaluating Privacy-Preserving Face Recognition (PPFR) systems. The claim that preventing pixel recovery does not prevent identity leakage is compelling and well-motivated. However, the abstract states FaceLinkGen achieves "over 98.5% matching accuracy." It is crucial to distinguish whether this refers to the attacker's ability to link identities (privacy breach) or bypass the PPFR server's verification (security breach). The introduction clarifies this leans towards privacy/linkage, but the phrasing "matching accuracy" could be misinterpreted as authentication bypass. The motivation regarding the "implicit assumption" that pixel reconstruction is necessary for identity leakage (Section 1) is strong, particularly the argument that identity and pixel similarity are decoupled (Figure 2).

### Threat Model (Section 2)
The threat model focuses on the "curious or malicious service provider" with oracle access to the conversion process. The authors argue that even client-side models can be reverse-engineered to provide this oracle access. While plausible for pure software clients, this assumption may not hold for systems utilizing secure enclaves or hardware-backed keys, which are increasingly common in high-security biometric deployments. The paper acknowledges this limitation in Section 10 ("Cryptographic and Key-Based Hardening"), but the primary empirical evaluation (Section 5) relies heavily on this oracle access. The constrained adversary model in Section 6 (no oracle access, only proxy filtering) is a vital complement that strengthens the paper, showing vulnerability even without full oracle access for frequency-domain methods. However, the transition between the "insider with oracle" and "external with proxy" scenarios could be sharper in defining the scope of vulnerability for each PPFR variant.

### Methods (Section 3 & 4)
The method relies on knowledge distillation to align protected templates with a public embedding space (ArcFace), followed by diffusion-based regeneration (Arc2Face). The simplicity is intentional, as stated, to highlight representation vulnerability.
*   **Eq. 2 & 3:** The formulation is standard for distillation and generation. The novelty lies not in the equations but in their application to critique the PPFR evaluation paradigm.
*   **Embedding Alignment:** The method assumes the protected template retains identity information alignable with ArcFace. Section 3 states, "Our attack method is independent of the specific face recognition model employed by the PPFR server." This is a strong claim. If the PPFR system uses a proprietary backbone significantly divergent from ArcFace, the distillation might fail to capture the specific identity manifold preserved by that system. While the paper argues identity is "generic," empirical validation against a PPFR system using a non-ArcFace-compatible private backbone would strengthen this claim. Currently, the evaluation assumes alignment with ArcFace is sufficient to demonstrate leakage, which is valid for *linkage* against public databases but less clear for *utility* recovery relative to the specific server.
*   **Attack Vectors:** The distinction between Linkage (Section 4.1) and Regeneration (Section 4.2) is clear. The linkage attack is particularly damaging for privacy (associating anonymous templates with public IDs), while regeneration demonstrates the severity of the leakage visually.

### Experiments & Results (Section 5, 6, 7, 8, 9)
The experimental setup is comprehensive, covering three recent methods (PartialFace, MinusFace, FracFace) and multiple datasets (CASIA-WebFace, LFW, TPDNE).
*   **Baselines:** The comparison against FracFace's claimed protection rates (Table 6) is effective in highlighting the metric gap. However, it relies on reported numbers from FracFace rather than re-implementing their baseline attack (U-Net/StyleGAN) under the same conditions. Re-running the baseline attack would eliminate potential discrepancies in evaluation protocols.
*   **Metrics:** Using commercial APIs (Face++, Amazon) for verification is a strong choice for realism, avoiding overfitting to open-source models. Table 2 and Table 5 show consistent high success rates.
*   **Typo/Clarity:** In Section 6, the text states "about 4457% on the Amazon API." This is clearly a numerical error (likely 44-57% based on Table 7). While likely a parser artifact, it impedes immediate understanding of the constrained attack's efficacy compared to the oracle attack.
*   **Soft Biometrics (Section 9):** The extension to soft biometric leakage (age, gender, race) is valuable and often overlooked. Table 9 shows high inference accuracy, supporting the claim that frequency-domain protection fails to hide semantic attributes.
*   **De-identification Transfer (Section 8):** Applying the method to TIP-IM and CanFG broadens the impact beyond frequency-domain PPFR. The finding that adversarial de-identification also leaks identity under distillation (Table 8) is a significant insight, suggesting a broader issue with utility-preserving transformations.

### Writing & Clarity
The paper is generally well-structured, though the text contains numerous interruptions (line numbers, fragmented sentences) due to the extraction process. Per instructions, I will not penalize these as formatting artifacts. However, there are logical flow issues independent of parsing. For instance, Section 7 ("Similarity Distribution") is referenced in Section 5.3 ("As detailed in Section 7..."), but Section 7 appears after Section 6. The numbering seems consistent in the source but the flow in the provided text is jumbled. More critically, the distinction between "breaking verification" and "breaking privacy" should be maintained consistently. Table 3 reports "1-to-1 verification accuracy," which might imply the regenerated face can spoof the system. If the system uses ArcFace, this is true; if it uses a private model, the regenerated face might not spoof it, but the identity is still leaked. The text should clarify that high Face++ accuracy proves identity recovery, not necessarily authentication bypass against the specific PPFR server.

### Limitations & Broader Impact
The paper acknowledges limitations in Section 10, suggesting cryptographic hardening as a solution. This is honest but slightly generic. A more specific discussion on *why* frequency-domain methods fail (e.g., spectral overlap with identity features) would deepen the analysis. The broader impact is significant: if adopted, the proposed identity-centric evaluation could prevent the deployment of falsely "secure" PPFR systems. The societal risk of soft biometric leakage (Section 9) is well-highlighted. However, the paper does not discuss potential misuse of FaceLinkGen itself. While it is an evaluation tool, the ability to regenerate faces from templates could be exploited by adversaries if templates are leaked. A brief mention of responsible disclosure or dual-use concerns would be appropriate for ICLR.

### Overall Assessment
This paper makes a substantial contribution to the privacy-preserving computer vision community by challenging the prevailing evaluation paradigm of frequency-domain PPFR. The core insight—that pixel-level reconstruction failure does not imply identity privacy—is empirically well-supported and critically important. The FaceLinkGen framework, while methodologically simple (distillation + generation), effectively exposes vulnerabilities in recent SOTA methods (PartialFace, MinusFace, FracFace). The use of commercial APIs for verification adds real-world validity to the claims.

Key concerns remain regarding the threat model's assumption of oracle access for client-side models, though Section 6 mitigates this by showing vulnerability even under constrained proxy assumptions. Additionally, the reliance on ArcFace alignment assumes a degree of universality in identity representation that, while likely, could be further stress-tested against proprietary backbones. Despite these points, the empirical evidence of identity leakage is robust. The paper meets ICLR's standards for significance and empirical rigor, offering a necessary correction to how PPFR systems are benchmarked. I recommend acceptance, contingent on clarifying the distinction between privacy leakage and authentication bypass, and correcting numerical reporting errors.

Additional transferable weaknesses from matched human reviews:
I'll help you retrieve relevant human reviews and extract weaknesses that apply to the current paper. Let me start by reading the paper to understand its core topic, method, and evaluation claims.Let me read the paper in sections:Now let me explore the human review dataset to find relevant papers. I'll use a specialized agent to efficiently search for similar papers based on the key topics.Great! The agent found many relevant papers. Let me now read the most relevant review files to extract weakness patterns. I'll start with the core papers on privacy-preserving face recognition and model inversion attacks.Let me check the directory structure first:Now let me look inside the human_reviews directory:Let me search for the most relevant papers using grep with keywords:Let me try different search keywords:Let me search more specifically for papers on face recognition, biometric security, and privacy attacks:Let me search for papers on privacy metrics and evaluation:Now let me read the most relevant reviews. I'll start with the one that mentions FaceLinkGen directly:This review isn't about the FaceLinkGen paper. Let me search more carefully for papers directly related to privacy-preserving face recognition:Perfect! Let me read these reviews about the PPFR methods:Let me search more specifically for papers about this work:Let me try broader search terms:Let me read papers on privacy evaluation and metrics:Let me search for papers using evaluation metrics relevant to face recognition:Let me read papers that likely discuss the PPFR methods:Let me read another relevant paper:Now let me search for papers on face recognition attacks and biometric security:Let me read papers on de-identification and embedding attacks:Let me use the Explore agent to more strategically search for relevant papers:Based on the exploration, there are limited papers directly on privacy-preserving face recognition in this dataset. Let me read some of the closest matches to find relevant weaknesses:Let me read more papers on privacy metrics and evaluation:Let me search for papers on adversarial robustness and recognition systems:Let me search for papers on attack evaluation and threat models:

────────────────────────────────────────
NEUTRAL REVIEWER (qwen/qwen3.5-flash-02-23 via OpenRouter)
────────────────────────────────────────
## Balanced Review

### Summary
This paper challenges the prevailing evaluation paradigm for frequency-domain Privacy-Preserving Face Recognition (PPFR) by arguing that pixel-level reconstruction metrics like PSNR fail to capture identity-level privacy leakage. The authors introduce FaceLinkGen, an identity-centric attack that extracts identity embeddings from protected templates to perform linkage and face regeneration without recovering original pixels. Empirical results demonstrate that recent PPFR systems (PartialFace, MinusFace, FracFace) are vulnerable to high-accuracy identity linkage and regeneration despite high pixel-level protection claims.

### Strengths
1.  **Critical Evaluation of Standards:** The paper provides strong evidence that current PPFR evaluation metrics (PSNR/SSIM) are insufficient proxies for privacy, demonstrating cases where high protection scores coexist with successful identity leakage. (Evidence: Section 5.3, Table 6 shows FracFace with 1.000 channel protection but high identity recovery under FaceLinkGen).
2.  **High Attack Effectiveness:** The proposed method achieves near-universal success rates across three distinct, state-of-the-art PPFR methods, validating the severity of the identified vulnerability. (Evidence: Table 2 reports 99.2% Success@5 regeneration on PartialFace and 99.6% on MinusFace).
3.  **Low Resource Overhead:** The attack is computationally lightweight (under 2 hours, < USD 2.00 cost), demonstrating that the vulnerability stems from the representation itself rather than a lack of computational effort by the defender. (Evidence: Section 5, cost analysis and training time).
4.  **Broader Applicability:** The investigation extends beyond frequency-domain PPFR to include adversarial de-identification (TIP-IM) and other PPFR methods (CanFG), suggesting the vulnerability is systemic to mutual information preservation in biometric templates. (Evidence: Section 8, Table 8).

### Weaknesses
1.  **Methodological Incrementality:** The core attack technique is a standard distillation process (aligning student to teacher embeddings), which may lack the algorithmic novelty expected for ICLR without deeper theoretical characterization of the embedding space. (Evidence: Section 3 describes a standard cosine similarity objective for distillation).
2.  **Dependency on External Generators:** The regeneration claim relies heavily on Arc2Face, a specific identity-controlled generator. The success rate is tied to the generator's ability to match the embedding, rather than a direct proof of template leakage independent of generative priors. (Evidence: Section 4.2 and Section 5.3 discuss reliance on Arc2Face for image synthesis).
3.  **Ambiguity in Constrained Attack:** Section 6 describes a "constrained adversary" scenario using only 30 validation pairs, yet claims the training process remains consistent with the main experiment using 90K images. The data efficiency of the student model under this strict calibration constraint is not fully detailed. (Evidence: Comparison between Section 5 training data and Section 6 validation/setup descriptions).
4.  **Limited Empirical Defense:** The paper proposes mitigation strategies (cryptographic hardening, inverted de-identification) in Section 10 but offers no experimental validation or quantitative comparison of these defenses against the FaceLinkGen attack. (Evidence: Section 10 outlines directions without empirical results).

### Novelty & Significance
**Novelty:** The novelty of this work is primarily in the *evaluation framework* and the identification of a systemic gap between pixel-level privacy and identity-level privacy in frequency-domain PPFR. While the distillation mechanism is established, its application to expose flaws in recent PPFR standards (ICCV 2023 to NeurIPS 2025) is a significant contribution. The approach shifts the focus from "unreconstructability" to "indistinguishability in embedding space," offering a new perspective on privacy in representation learning.

**Significance:** The significance is high within the ML and Security communities. As PPFR adoption grows, ensuring that "privacy" actually protects identity is critical. This paper demonstrates that current frequency-domain solutions, despite recent publication, fail to meet identity-level privacy expectations, necessitating a reevaluation of design choices and evaluation metrics. It fits ICLR's focus on evaluating representation learning properties, though it leans heavily toward security implications.

### Suggestions for Improvement
1.  **Enhance Reproducibility:** To align with ICLR standards, the authors should provide the code for the FaceLinkGen student model and training pipeline. Given the simplicity of the method, code release would verify the reproducibility claim immediately.
2.  **Deepen Theoretical Analysis:** Incorporate an information-theoretic analysis (e.g., Mutual Information estimation) between the protected templates and the identity embedding to explain *why* these specific frequency-domain transformations retain sufficient identity information to be distilled.
3.  **Clarify Constrained Attack Details:** Provide a detailed curve or ablation showing how the student model performance scales as the number of calibration/validation pairs decreases in the Section 6 scenario, distinguishing between zero-shot and few-shot capabilities explicitly.
4.  **Provide Defensive Experiments:** Validate the proposed "Future Directions" from Section 10 with at least one concrete defensive modification (e.g., adding noise specifically in the identity embedding dimension) to demonstrate that the leakage can be mitigated without destroying utility.

────────────────────────────────────────
SPARK FINDER (qwen/qwen3.5-plus-02-15 via OpenRouter)
────────────────────────────────────────
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation on distillation pipeline components** — Remove the student model, change the teacher model, or vary training data size systematically. Without this, it's unclear whether success comes from the attack design or simply from template utility preservation inherent to any working PPFR system.

2. **Comparison to existing model inversion attacks** — No baseline comparison to prior MIA methods (e.g., Wang et al. [33], FaceMAE [34] attacks). Without showing FaceLinkGen outperforms or differs meaningfully from existing attacks, the novelty claim is weakened.

3. **Evaluation on non-frequency-domain PPFR methods** — Only 3 frequency-domain methods tested. The claim that "identity leakage extends beyond frequency-domain PPFR" rests on minimal evidence (one CanFG evaluation in Table 8). Test on at least 2-3 additional non-frequency methods to support generalizability.

4. **Empirical validation of proposed defenses** — Section 10 recommends cryptographic hardening and perception-based de-identification but provides zero experimental validation. For ICLR, proposed solutions must be tested, not just described.

### Deeper Analysis Needed (top 3-5 only)
1. **Statistical correlation analysis between pixel metrics and identity leakage** — The claim that PSNR/SSIM don't capture identity privacy needs quantitative evidence (e.g., correlation coefficients, scatter plots). Currently this is asserted with only 2 examples in Table 1.

2. **Analysis of WHY frequency-domain methods fail** — Is this inherent to preserving recognition utility, or a specific design flaw? Without mechanistic analysis, the paper identifies a symptom but not the root cause, limiting actionable insights for defenders.

3. **Train-test identity separation verification** — The distillation uses CASIA-WebFace for training and testing uses CASIA hold-out, LFW, and TPDNE. Need explicit verification that no identity leakage occurred through dataset overlap, especially given known CASIA noise issues [32].

4. **Embedding space analysis** — Show cosine similarity distributions between extracted embeddings and original identity embeddings vs. different-identity embeddings. This would directly demonstrate whether identity information is truly preserved or if the high success rate comes from generator artifacts.

### Visualizations & Case Studies
1. **Failure cases** — Show regenerated faces that failed verification and analyze why. Understanding failure modes is critical for assessing attack limitations and for designing effective defenses.

2. **Template information visualization** — Visualize what frequency components or features are preserved vs. suppressed in protected templates. This would expose whether the vulnerability is in high-frequency retention as Section 6 suggests.

3. **Embedding space projection** — Use t-SNE or similar to show clustering of original embeddings, protected template embeddings, and extracted embeddings. This would visually demonstrate whether the distillation successfully recovers identity structure.

### Obvious Next Steps
1. **Test proposed cryptographic defenses experimentally** — If key-based hardening is recommended as a solution, implement and evaluate at least one variant to show it actually prevents FaceLinkGen.

2. **Propose concrete identity-centric evaluation metrics** — The paper critiques existing metrics but doesn't provide standardized alternatives. ICLR expects constructive contributions, not just criticism.

3. **Evaluate under realistic deployment constraints** — Test attack success when templates are quantized, compressed, or when rate-limiting prevents oracle queries. The current evaluation assumes idealized access that may not reflect real systems.

────────────────────────────────────────
POTENTIALLY MISSED RELATED WORK (qwen/qwen3.5-flash-02-23:online via OpenRouter)
────────────────────────────────────────
Related work search was skipped.

========================================================================
FINAL CONSOLIDATED REVIEW (z-ai/glm-5.1 via OpenRouter)
========================================================================

## Summary

This paper challenges the prevailing evaluation paradigm for frequency-domain Privacy-Preserving Face Recognition (PPFR), arguing that pixel-level reconstruction metrics (PSNR/SSIM) inadequately capture identity-level privacy leakage. The authors introduce FaceLinkGen, a distillation-based pipeline that extracts identity embeddings from protected templates to perform linkage attacks (>98.5% accuracy) and face regeneration (>96% success at commercial-grade verification) on three recent PPFR systems—PartialFace, MinusFace, and FracFace—without recovering original pixels, demonstrating that high channel-protection scores coexist with near-total identity leakage.

## Strengths

- **Compelling paradigm critique with strong empirical counter-evidence.** The paper's central claim—that preventing pixel-level reconstruction does not prevent identity leakage—is backed by concrete, striking evidence. Table 6 is particularly damning: FracFace claims 1.000 protection under its own frequency-channel metric, yet FaceLinkGen achieves near-total identity recovery through commercial-grade verification. Table 1 provides a clear motivating example showing that high SSIM (0.841) can coincide with near-zero identity similarity (FS=0.008), while low SSIM (0.235) can coincide with meaningful identity similarity (FS=0.586). This directly challenges a foundational assumption in the field.

- **Extremely low attack cost demonstrating representation-level vulnerability.** The distillation completes in under 2 hours on a single GPU for ~$0.80–1.60 per method (Section 5), and the minimal-resource experiment (Section 5.1) achieves 97.0% generation pass rate with only ~800 images trained in under 50 seconds. This is a strength because the paper's argument is precisely that the vulnerability is in the representation, not in needing a sophisticated attack—the simplicity is load-bearing for the claim.

- **Constrained adversary scenario strengthens real-world relevance.** Section 6 demonstrates that even without oracle access to the conversion process, a generic Gaussian-blur high-pass filter suffices as a universal proxy to extract identity from all three frequency-domain methods (92–96% matching, 94.6–96.3% regeneration@5 on Face++). This significantly broadens the vulnerability beyond the insider-with-oracle model.

- **Soft biometric leakage analysis extends the privacy concern.** Section 9 shows that race, gender, and age can be inferred from protected templates at near-original-image accuracy (e.g., gender accuracy 0.925–0.932 for FracFace/PartialFace vs. 0.949 for originals), directly contesting prior claims of soft-biometric obfuscation and raising regulatory implications.

- **Cross-verification using multiple independent services.** The use of both Face++ and Amazon APIs (Tables 2, 5, 7) for identity verification, plus direct cosine-similarity analysis in embedding space (Section 7), rules out dependence on any single evaluation pipeline. The embedding similarity analysis (Section 7) further demonstrates identity extraction independent of generative models.

## Weaknesses

### Major:

- **Limited mechanistic analysis of why frequency-domain methods fail.** The paper demonstrates *that* identity information is preserved and extractable, but provides insufficient analysis of *why* these specific transformations retain it. Section 6 offers the intuition that "they all preserve high-frequency information while obfuscating low-frequency information," but this is stated rather than rigorously analyzed. A spectral decomposition or information-theoretic analysis (e.g., mutual information between template and identity embedding as a function of frequency band) would transform this from a well-executed attack paper into a deeper scientific contribution. Without it, the paper identifies a critical symptom but not its root cause, limiting actionable guidance for defenders beyond "use cryptography."

- **No empirical comparison with existing attack methods.** The paper discusses differences from standard Model Inversion Attacks (MIA) in the introduction but does not empirically compare FaceLinkGen against any existing attack on the same PPFR systems. Re-running the U-Net or StyleGAN attacks used by the original PPFR papers under the same identity-centric evaluation (rather than just citing their reported PSNR/SSIM numbers) would directly quantify the gap the paper claims. Without this comparison, it remains possible that prior attacks also achieve high identity recovery but were simply not measured on the right metric—a possibility that, if true, would reframing the contribution from "new attack exposes hidden vulnerability" to "old attacks were evaluated on wrong metrics," which is a different (though still important) claim.

- **The decoupling of pixel-level metrics and identity leakage is asserted with limited quantitative evidence.** The paper's foundational claim is that PSNR/SSIM do not correlate with identity privacy. However, this is supported by only two anecdotal examples in Table 1 and Figure 2. A systematic correlation analysis—e.g., computing PSNR/SSIM vs. identity similarity (cosine similarity or verification accuracy) across hundreds of protected templates—would substantially strengthen this pillar of the argument. As presented, the reader must take the decoupling on faith from 2–3 cherry-picked examples rather than statistical evidence.

### Minor:

- **Privacy leakage vs. authentication bypass distinction could be sharper.** Table 3 reports "1-to-1 verification accuracy" between template-to-face pairs, which could be misread as demonstrating authentication bypass against the PPFR server. The paper's contribution is about identity *privacy* (leakage to the service provider or third parties), not about *spoofing* the verification system. The text in Section 3 states the attack is "independent of the specific face recognition model employed by the PPFR server," but this important distinction is muddied when verification accuracy is reported without clearly reiterating that this measures identity recovery against a public reference (Face++), not bypass of the PPFR system's own recognition pipeline. A brief clarifying note would eliminate ambiguity.

- **Typographical error in Section 6.** The text states the Amazon API achieves "about 4457%," which from Table 7 should read "44–57%." While minor, this error in a key result summary could mislead readers assessing the constrained attack's efficacy relative to the oracle attack.

- **Evaluation beyond frequency-domain PPFR is preliminary.** The extension to CanFG and TIP-IM (Table 8) is valuable but thin—each tested on a small subset (2,082 images from 408 identities). The paper itself labels these as "pilot evaluations," which is honest, but the abstract's claim that results "motivate complementing pixel-level metrics with identity-centric evaluation in frequency-domain PPFR research" is appropriately scoped. The broader generalization claim in Section 8 ("suggesting a broader issue with utility-preserving transformations") is suggestive but not yet convincing from n=2 non-frequency methods with limited data.

### Trivial:

- None significant.

## Nice-to-Haves

- **Experimental validation of at least one proposed defense.** Section 10 suggests cryptographic hardening and inverted de-identification as future directions. Even a simple proof-of-concept—e.g., adding structured noise to templates and measuring the privacy-utility tradeoff under FaceLinkGen—would demonstrate that the vulnerability is actionable and not purely theoretical.

- **Formalized alternative evaluation metric.** The paper effectively critiques existing metrics but does not propose a standardized replacement. A concrete identity-centric metric (e.g., maximum achievable linkage accuracy under a distillation budget, or mutual information between template and identity embedding) would make the contribution more constructive.

- **Failure case analysis.** The paper reports near-perfect success rates but does not analyze the ~1–3% failure cases. Understanding when and why identity extraction fails could inform both defenders (what properties help?) and future attackers (what are the boundary conditions?).

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Secure enclaves undermine oracle access assumption"** (from Harsh Critic). The paper explicitly addresses this in Section 2 (arguing that client-side models can be reverse-engineered) and Section 10 (acknowledging cryptographic approaches as a separate, stronger defense). The oracle-access threat model is consistent with the PPFR literature's original design intent [8] and prior evaluations [6, 20, 21]. Secure enclaves represent a fundamentally different system architecture; critiquing the paper for not addressing them is scope creep.

- **"Methodological incrementality — standard distillation"** (from Positive-Leaning Reviewer). The paper explicitly states "The simplicity of our method is intentional" (Section 3) and explains that the point is to show the vulnerability is in the representation, not in attack sophistication. The contribution is the evaluation paradigm critique and the empirical demonstration, not the attack's algorithmic novelty. This is a feature, not a bug.

- **"Dependency on Arc2Face for regeneration claims"** (from Positive-Leaning Reviewer). The paper addresses this in Section 5.3 and Section 7 by (a) cross-verifying with two commercial APIs (Face++ and Amazon), (b) showing direct cosine similarity between extracted embeddings and original embeddings independent of any generator, and (c) the linkage attack (Section 4.1) operates entirely in embedding space without Arc2Face. The regeneration is a *demonstration* of leakage severity, not the sole evidence.

- **"Constrained attack Section 6 is ambiguous about training data"** (from Harsh Critic). Re-reading Section 6 clarifies: the 30 paired samples are for *validation only*, and training uses a generic Gaussian-blur high-pass filter applied to public face images (not PPFR templates). The paper states: "the training process is the same as our main text, with simply the known PPFR conversion process replaced with a high-pass filter." This is a misreading by the critic.

- **"Train-test identity separation concerns"** (from Spark Finder). The paper explicitly states: "there is no dataset or architecture overlap" and "The hold-out set is used to test the ability of our method in real images while ensuring no ID duplications." The dataset noise issue [32] is acknowledged. This concern is addressed.

- **"Re-run baseline attacks (U-Net/StyleGAN) under same conditions"** (from Harsh Critic). While this would strengthen the paper (and is noted above as a valid weakness about lack of empirical comparison), the specific demand to re-implement prior attacks under identical conditions goes beyond what's needed. The paper's key comparison is metric-level: FracFace's own protection metric vs. identity-centric evaluation, which does not require re-running their attack.

- **"Dual-use / responsible disclosure concerns"** (from Harsh Critic). This is a generic ethics concern applicable to any security paper and not a specific weakness of this work.

- **"Reproducibility concerns — code release"** (from Positive-Leaning Reviewer). Per hard rules, reproducibility nitpicks about undisclosed implementation details are removed. The method is described in sufficient detail (standard distillation + Arc2Face) for reproduction.

## Novel Insights

The paper reveals a fundamental tension in PPFR design that has been hiding in plain sight: any transformation that preserves recognition utility necessarily preserves identity-discriminative information, and this information is extractable via a trivial alignment to any public embedding space. This is not merely a weakness of specific methods but a structural problem for the entire family of utility-preserving biometric protections that rely on representation-level obfuscation without cryptographic guarantees. The constrained adversary experiment (Section 6) further reveals that frequency-domain PPFR methods share an exploitable inductive bias—the retention of high-frequency components—that allows a single generic proxy (Gaussian high-pass filtering) to substitute for system-specific oracle access. The soft-biometric leakage finding (Section 9) suggests an even more troubling implication: the identity manifold preserved by these systems encodes not just identity but demographic attributes, meaning the privacy harm extends beyond re-identification to enabling discriminatory profiling.

## Suggestions

- **Provide a systematic correlation analysis** between PSNR/SSIM and identity similarity across all protected templates in your evaluation. Even a simple scatter plot with Pearson/Spearman correlation coefficients would transform the current anecdotal evidence (Table 1, Figure 2) into a rigorous statistical argument.

- **Empirically compare with at least one prior attack** (e.g., the U-Net reconstruction from FracFace or MinusFace) evaluated under your identity-centric metrics. This would cleanly separate the contribution of "new attack method" from "old attacks evaluated on wrong metrics"—both are publishable, but the distinction matters for the community's understanding.

- **Add spectral analysis of what frequency components carry identity information.** Given that Section 6 identifies high-frequency preservation as the common vulnerability, a frequency-band ablation (progressively removing frequency bands and measuring identity extraction success) would provide the mechanistic insight the paper currently lacks and give defenders concrete design guidance.

- **Clarify in Table 3 and surrounding text** that verification accuracy measures identity recovery against a public reference system, not spoofing of the PPFR server's own recognition pipeline, to prevent misinterpretation.

## Evaluation Axis Assessment

- **Novelty**: Moderate. The distillation+generation pipeline is standard; the novelty lies in the evaluation paradigm critique and the empirical demonstration that this simple pipeline catastrophically breaks recent PPFR systems. The constrained adversary extension adds incremental novelty.

- **Technical soundness**: Generally sound. The experimental design is thorough (multiple systems, datasets, verification APIs, constrained adversary, soft biometrics). The main gaps are the lack of baseline attack comparisons and the limited quantitative evidence for the pixel-metric vs. identity decoupling claim.

- **Empirical support**: Strong for the main claim (identity leakage from frequency-domain PPFR). The attack success rates are consistently high across methods, datasets, and even constrained settings. Weaker for the broader generalization claims (non-frequency methods) and the mechanistic "why" question.

- **Significance**: High. If the community adopts identity-centric evaluation as this paper advocates, it would meaningfully change how PPFR systems are designed and benchmarked. The finding that three recent, top-venue PPFR systems all fail under identity-centric evaluation is impactful.

- **Clarity**: Good. The threat model is well-argued, the method is clearly described, and the intentional simplicity of the approach is well-motivated. Minor issues with the privacy-leakage vs. authentication-bypass distinction and one typographical error.

========================================================================
PREDICTED SCORE
========================================================================

Score: 7.1
Decision: N/A
Total Cost: $0.0709
