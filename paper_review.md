
========================================================================
INDIVIDUAL REVIEWS
========================================================================

────────────────────────────────────────
HARSH CRITIC (qwen/qwen3.6-plus:free via OpenRouter)
────────────────────────────────────────
## Section-by-Section Critical Review

### Title & Abstract
- **Title accuracy**: The title accurately reflects the paper's core contribution: proposing an identity-centric attack (FaceLinkGen) to demonstrate that pixel-level metrics fail to capture identity leakage in PPFR.
- **Abstract clarity**: The abstract clearly outlines the problem (pixel-level metrics overestimate privacy), method (black-box distillation to align templates with a standard embedding space, followed by diffusion-based regeneration), and key results (>98.5% matching, >96% regeneration). 
- **Unsupported claims**: The quantitative claims are well-supported by the experimental results in Tables 2–4. No obvious overstatements in the abstract. One minor note: stating the attack works "directly from protected templates without recovering original pixels" is accurate, but the abstract could briefly acknowledge the prerequisite of a paired training set for the distillation phase, as this contextualizes the threat model from the start.

### Introduction & Motivation
- **Motivation & gap**: The gap is well-motivated and clearly articulated in Sections 1 and 2. The distinction between pixel/structural similarity and identity similarity (illustrated in Figure 2 and Table 1) effectively challenges the prevailing reconstruction-centric evaluation paradigm in PPFR literature.
- **Contributions**: Clearly stated at the end of Section 1. The contributions are accurate and appropriately scoped.
- **Over-claim/under-sell**: The introduction slightly underplays the methodological familiarity of the approach. In the paragraph beginning "Our approach differs from standard Model Inversion Attacks," the authors frame the method as distinct because it targets "structural vulnerabilities in the template generation process." However, the core pipeline (Equation 2: cosine similarity distillation from template to a standard embedding, followed by generative synthesis) is a standard knowledge distillation + ID-consistent generation setup. The novelty lies in its application to expose an evaluation gap in PPFR, not in the adversarial pipeline itself. This positioning should be sharpened to avoid misreading by reviewers expecting a novel algorithmic contribution.

### Method / Approach
- **Description & reproducibility**: The method is clearly described. Equations 1–3 succinctly formalize the information decomposition, distillation objective, and generation step. The pipeline is reproducible given the specified datasets (CASIA-WebFace subset), architecture (Antelopev2 + 3x3 Conv2D), teacher model (ArcFace), and generator (Arc2Face).
- **Key assumptions**: The method assumes (1) oracle access to the conversion function to collect paired `(original, protected)` data for training, and (2) that the protected template retains identity-discriminative information that is learnably aligned to the ArcFace manifold. These are explicitly stated and justified under the insider/service-provider threat model. However, the assumption that 10K paired identities from the target domain are available is strong. While Section 5.1 tests down to 256 images, the main results rely on dataset-scale distillation.
- **Logical gaps & edge cases**: A logical step that lacks justification is *why* frequency-domain transformations (PartialFace, MinusFace, FracFace) are so easily aligned to ArcFace. The paper empirically demonstrates high alignment but does not analyze whether this stems from the linear/subspace properties of frequency filtering, the preservation of high-frequency identity cues, or the inherent smoothness of the ArcFace embedding space. Additionally, the method assumes the PPFR system uses a deterministic or fixed-parameter transform. Systems employing per-user secret keys or randomized subspace selection per query (common in cryptographic or cancelable biometrics) would break the alignment assumption. This edge case should be acknowledged as a boundary condition.
- **Theoretical claims**: No formal proofs are provided or claimed; the work is empirical. This is appropriate for the scope, but the authors should explicitly state that the guarantees are empirical lower bounds on leakage.

### Experiments & Results
- **Testing claims**: The experiments directly test the core claim that identity persists and can be leveraged for linkage and regeneration despite low pixel-level similarity. Results across three datasets (TPDNE, CASIA hold-out, LFW) and two commercial verification APIs strongly support the claim.
- **Baselines & fairness**: The paper compares against the original defensive claims of FracFace (Table 6) and standard FR upper bounds (Table 3). However, it lacks comparison to other established template inversion or distillation baselines applied to the same PPFR outputs (e.g., standard MIAs, GAN-based inversions used in prior PPFR papers). A direct baseline comparison would better isolate whether FaceLinkGen's success stems from the specific distillation setup or simply from the templates' inherent vulnerability.
- **Missing ablations**: Ablations on the choice of teacher/generator architecture (e.g., FaceNet vs. ArcFace, Arc2Face vs. ID-Adapter) are absent. Since the attack's success hinges on Arc2Face's compatibility with ArcFace embeddings, showing robustness across alternative embedding-generation pairs would strengthen the claim that the vulnerability lies in the template, not the specific auxiliary model.
- **Error bars / significance**: No variance or error bars are reported across the tables. While success rates consistently exceed 90% and templates are deterministic, reporting standard deviation across multiple runs or dataset subsets is standard practice at ICLR and would bolster confidence in the stability of the results.
- **Cherry-picking / dataset appropriateness**: Results are consistent across domain-shifted datasets, reducing cherry-picking concerns. The reliance on commercial APIs (Face++, Amazon) is practical but limits independent reproducibility. The authors mitigate this with embedding cosine similarity analysis (Section 7), but an open-source verification benchmark (e.g., MagFace, CosFace) should be included to ensure results aren't biased toward proprietary thresholding.

### Writing & Clarity
- **Clarity of sections**: The method and threat model are clearly written. Section 8's transition to attacking adversarial de-identification (TIP-IM) is conceptually sound but feels slightly abrupt. The link between PPFR (machine-readable, human-unreadable) and de-identification (human-readable, machine-resistant) is stated, but the experimental pivot lacks a clear methodological justification for why TIP-IM specifically was chosen over other prominent adversarial masking methods.
- **Figures & tables**: Figure 1 effectively illustrates the regeneration qualitative results. Table 1 clearly demonstrates the decoupling of pixel metrics from identity similarity. Table 4 (linkage) and Table 7 (constrained setting) are well-structured. Table 2 is dense; consider separating Success@5 and Pass@1e-* metrics into distinct sub-tables for readability. Overall, figures and tables are informative and support the narrative.

### Limitations & Broader Impact
- **Acknowledged limitations**: The authors acknowledge the reliance on paired training data and focus primarily on frequency-domain PPFR. They also note in Section 6 that the proxy high-pass filter attack does not apply to non-frequency methods like CanFG.
- **Missed limitations**: The paper does not discuss the dependency on Arc2Face's generation capacity. If the student extracts a degraded embedding, Arc2Face may still synthesize plausible faces that pass commercial APIs due to the generator's prior smoothing biases, potentially inflating "success" rates. Additionally, the computational and data collection costs for an external attacker (outside the oracle model) are under-discussed; 10k paired queries may trigger rate-limiting or anomaly detection in production systems.
- **Failure modes / societal impact**: The broader impact discussion is appropriately calibrated toward system hardening. However, the paper could better address how legitimate users or system designers should respond. The recommendation in Section 10 (cryptographic hardening or N-to-1 human-perception mapping) is useful but lacks a discussion of the utility-privacy trade-off implications, which is central to PPFR deployment.

### Overall Assessment
This paper makes a timely and substantively important contribution to the secure biometrics literature by rigorously demonstrating that current frequency-domain PPFR systems fail an identity-centric privacy standard, despite passing pixel-level reconstruction metrics. The distillation-based attack pipeline is simple, highly effective, and empirically well-validated across multiple methods, datasets, and verification backends. However, for ICLR's standards, the methodological novelty is moderate, as the core approach relies on standard knowledge distillation and ID-consistent diffusion rather than a novel adversarial formulation. The paper would be strengthened by (1) clarifying why frequency transforms inherently preserve learnable identity manifolds (e.g., spectral analysis of the template space), (2) including comparisons to established model inversion baselines and open-source FR verifiers to ensure reproducibility, (3) reporting statistical variance, and (4) discussing the practical feasibility of acquiring the required paired training data in real-world, rate-limited deployments. Despite these points, the empirical rigor, clear threat model, and actionable findings on evaluation paradigms make this a strong candidate for publication, provided the authors address the reproducibility and positioning concerns in the revision.

────────────────────────────────────────
NEUTRAL REVIEWER (qwen/qwen3.6-plus:free via OpenRouter)
────────────────────────────────────────
## Balanced Review

### Summary
This paper challenges the dominant pixel-level evaluation paradigm (e.g., PSNR, SSIM) for frequency-domain privacy-preserving face recognition (PPFR) systems, demonstrating that preventing visual reconstruction does not equate to protecting identity-level information. The authors propose FaceLinkGen, a distillation-based attack that aligns protected templates with standard facial embeddings and leverages diffusion models for identity linkage and face regeneration, achieving >98.5% matching and >96% regeneration success across three recent PPFR methods. The findings advocate for shifting PPFR research toward identity-centric evaluation metrics and stronger cryptographic or information-theoretic guarantees.

### Strengths
1. **Well-motivated critique of prevailing evaluation metrics:** The paper clearly demonstrates the decoupling between pixel-level similarity and identity consistency (Table 1, Figure 2), effectively arguing that current PPFR evaluations may provide a false sense of security.
2. **Comprehensive and rigorous empirical evaluation:** The authors test across three distinct PPFR methods, multiple datasets (CASIA-WebFace hold-out, LFW, TPDNE), and varied threat models (full oracle, constrained 30-pair black-box, minimal-resource 256/800 images). Results are consistently robust (Tables 2, 4, 7), and computational costs are transparently reported (~$0.80–$1.60 training, <2 hours on A6000), highlighting real-world feasibility.
3. **Broader impact analysis:** The attack pipeline successfully extends to adversarial de-identified systems (TIP-IM) and non-frequency methods (CanFG), while also quantifying soft-biometric leakage (Section 9, Table 9). This demonstrates the generality of the identity extraction vulnerability beyond a single PPFR family.
4. **Transparent and reproducible methodology:** The use of publicly available baselines, standard datasets, open-source components (Antelopev2, Arc2Face), and explicit training protocols ensures high reproducibility, aligning well with ICLR's emphasis on empirical rigor.

### Weaknesses
1. **Limited algorithmic novelty:** The core attack pipeline (cosine similarity distillation + diffusion synthesis) is conceptually straightforward. While the authors intentionally emphasize simplicity to highlight representation flaws, ICLR typically expects deeper algorithmic or theoretical innovation. The paper lacks a formal analysis of why frequency-domain transformations fail to decouple identity mutual information from nuisance factors.
2. **Absence of comparative baselines against state-of-the-art MIAs:** The paper contrasts its approach with prior MIA definitions and FracFace's internal attacks but does not empirically benchmark FaceLinkGen against recent embedding inversion or model inversion techniques (e.g., Diffusion-Driven Universal MIA [33]). This makes it difficult to gauge whether the distillation pipeline offers a meaningful advantage in attack success, efficiency, or query constraints.
3. **Under-analyzed linkage performance gap:** The closed-set linkage accuracy caps at ~84.8% (Table 4), while the original-image baseline reaches ~88.2%. The paper attributes this to dataset noise but does not thoroughly investigate whether the distillation bottleneck, cross-embedding alignment limits, or threshold selection explains the ~3.5% gap. Missing ROC/DET curves and false-positive analysis for linkage limits practical deployment insights.
4. **Threat model scope and defense analysis are somewhat narrow:** The proxy high-pass filter attack (Section 6) is explicitly method-specific and does not generalize to non-frequency PPFR. Furthermore, proposed defenses (Section 10) lean heavily toward cryptographic hardening without engaging with recent learnable defenses such as differential privacy in template generation, information bottleneck formulations, or adversarial training against distillation.

### Novelty & Significance
**Novelty:** Moderate. The methodological contribution is primarily empirical and conceptual rather than algorithmic. The novelty lies in reframing PPFR evaluation from pixel reconstruction to identity-centric linkage/regeneration, combined with a lightweight, reproducible distillation pipeline.  
**Clarity:** High. The paper is well-structured, defines the threat model explicitly, and presents experiments in a logical progression. The intentional simplicity of the attack is clearly communicated, and tables/figures effectively support claims.  
**Reproducibility:** High. Relies on open-source PPFR implementations, public datasets, standard commercial APIs (Face++, Amazon), and reports compute costs and training configurations transparently. The pipeline is straightforward to replicate.  
**Significance:** High for the biometric security and trustworthy ML communities. The paper exposes a critical gap between current defensive claims and realistic identity leakage, which will likely influence future PPFR benchmarking, standardization (e.g., ISO/IEC), and system design. It aligns well with ICLR's growing focus on robust, secure, and privacy-aware representation learning.

### Suggestions for Improvement
1. **Include direct empirical comparisons with recent MIA/baseline attacks:** Benchmark FaceLinkGen against state-of-the-art embedding inversion or frequency-domain reconstruction attacks under identical settings (data, compute, query constraints) to contextualize its effectiveness and highlight trade-offs.
2. **Provide theoretical or information-theoretic analysis:** Formalize why frequency masking (high-pass/low-pass channel retention) preserves identity-discriminative signal. Even a mutual information or representation bottleneck analysis would strengthen the conceptual contribution and meet ICLR's expectations for methodological depth.
3. **Expand linkage evaluation metrics:** Report precision-recall or ROC/DET curves for the 1-to-N linkage attack, analyze false positive/negative trade-offs at varying similarity thresholds, and investigate the ~3.5% performance gap relative to the original-image baseline.
4. **Broaden defense discussion:** Ground suggested mitigations in recent ML literature (e.g., differentially private template projection, adversarial distillation resistance, N-to-1 identity anonymization mappings) and discuss practical deployment constraints (latency, accuracy drop, key management) for each.
5. **Clarify open-set vs. closed-set implications:** Briefly discuss how the extracted embeddings and regeneration pipeline would behave in open-world identification or cross-database scenarios, which are more representative of real-world PPFR deployments.

────────────────────────────────────────
SPARK FINDER (qwen/qwen3.6-plus:free via OpenRouter)
────────────────────────────────────────
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add direct comparisons against established inversion baselines (e.g., GMI, LIA, DLG) under identical black-box constraints, because without showing that your distillation pipeline outperforms or fundamentally differs from existing reconstruction attacks, the claim of a representation-specific vulnerability is unproven.
2. Add a strict cross-domain generalization experiment where the distillation training set shares zero dataset lineage or visual prior with the target PPFR methods, because hidden data overlap could artificially inflate success rates and invalidate the claim of inherent template insecurity.
3. Clarify and ablate the exact sample complexity for the "minimal-resource" setting (specify total vs. per-identity images and report variance), because ambiguous training data requirements leave the real-world feasibility and threat severity of the attack unquantified.

### Deeper Analysis Needed (top 3-5 only)
1. Add a frequency-band leakage analysis quantifying the mutual information or feature attribution between specific spectral components and identity embeddings, because without isolating exactly which frequencies preserve discriminative signals, the attack mechanism remains descriptive rather than diagnostic.
2. Add a population-level statistical test correlating pixel-metric degradation (PSNR/SSIM) with embedding cosine similarity and verification pass rates across thousands of samples, because the core thesis that pixel evaluations are misleading requires distributional evidence beyond isolated examples.
3. Add an ablation isolating the distillation step from the Arc2Face generator by evaluating raw extracted embeddings against multiple open-source verification backbones, because success driven by a single commercial API or generative model could reflect model overfitting rather than inherent template leakage.

### Visualizations & Case Studies
1. Add a t-SNE/UMAP projection plotting original faces, protected templates, and student-extracted embeddings for mixed identities, because visually confirming that obfuscated templates retain tight identity clustering is the most direct proof that the frequency transform fails to destroy structural discriminability.
2. Add a scatter or hexbin plot showing per-sample reconstruction fidelity (PSNR/SSIM) versus attack success rate or FAR, because this visualization would immediately and empirically validate the central argument that current PPFR evaluation metrics decouple from actual identity risk.
3. Add a systematic error gallery documenting failure cases (e.g., extreme poses, heavy occlusion, low resolution) alongside quantitative confusion matrices, because transparent error analysis is required to trust the >98% success claims and rule out dataset bias or favorable sampling.

### Obvious Next Steps
1. Formalize and release a standardized identity-centric evaluation protocol with open-source scripts and fixed baselines, because merely demonstrating a single attack pipeline does not establish a reproducible community standard as the paper advocates.
2. Propose and preliminarily test at least one targeted countermeasure (e.g., embedding-space noise injection, selective high-frequency randomization, or utility-privacy regularization) that degrades the distillation alignment, because ICLR expects critical vulnerability analyses to be paired with actionable defense pathways rather than ending in pure critique.
3. Replace or fully supplement proprietary commercial API verification with documented open-source FR backbones, because black-box evaluation hinders independent audit, violates reproducibility expectations, and leaves the core vulnerability claims vulnerable to API-specific implementation artifacts.

────────────────────────────────────────
POTENTIALLY MISSED RELATED WORK (qwen/qwen3.6-plus:free:online via OpenRouter)
────────────────────────────────────────
Related work search was skipped.

========================================================================
FINAL CONSOLIDATED REVIEW (qwen/qwen3.6-plus:free via OpenRouter)
========================================================================

## Summary
This paper challenges the prevailing reconstruction-centric evaluation paradigm for frequency-domain privacy-preserving face recognition (PPFR) systems, demonstrating that high pixel-level distortion (e.g., low PSNR/SSIM) does not equate to identity-level privacy. The authors introduce FaceLinkGen, a lightweight distillation pipeline that aligns protected templates with standard facial embeddings and leverages diffusion models for identity linkage and face regeneration. Evaluated across three recent PPFR methods, multiple datasets, and varied threat models, the attack achieves consistently high success rates, prompting a call for identity-centric evaluation standards and stronger cryptographic guarantees in PPFR research.

## Strengths
- **Compelling critique of a flawed evaluation paradigm:** The paper rigorously demonstrates the decoupling between pixel-level similarity and identity consistency (Table 1, Figure 2), effectively arguing that optimizing defenses against visual reconstruction provides a false sense of security while leaving identity-discriminative manifolds intact.
- **Empirically rigorous and highly reproducible design:** The evaluation spans three distinct PPFR architectures, distribution-shifted datasets (CASIA-WebFace hold-out, LFW, TPDNE), and progressively constrained threat models (full oracle, 30-pair black-box proxy, <1000 image distillation). Transparent reporting of compute costs (~$0.80–$1.60, under 2 hours) and explicit dataset isolation (Section 5) ensures the vulnerability claims are grounded in realistic, low-barrier attack scenarios rather than overfitted setups.
- **Demonstrates broad systemic relevance:** Beyond frequency-domain PPFR, the pipeline successfully exposes identity leakage in non-frequency methods (CanFG) and adversarial de-identification systems (TIP-IM), while quantifying persistent soft-biometric leakage (Section 9). This establishes identity preservation as a structural vulnerability across multiple privacy-preserving paradigms, not merely an artifact of a single frequency-family design.

## Weaknesses
- **Utility-privacy boundary remains empirically unquantified:** The paper correctly notes that as long as PPFR templates preserve recognition utility, identity signals remain extractable. However, it does not empirically characterize *how much* recognition utility must be degraded for the distillation alignment to fail. Providing a utility-privacy trade-off curve would be critical for system designers to understand the threshold at which identity-centric leakage becomes unavoidable versus when it can be mitigated by acceptable accuracy drops.
- **Ambiguity in minimal-resource sample complexity:** Section 5.1 reports high attack success using "800 images" and "256 images," but does not clarify whether this refers to total paired samples or per-identity samples. Since model inversion and distillation attacks scale nonlinearly with identity coverage, this ambiguity makes it difficult to precisely gauge the data-collection barrier for an external adversary deploying this pipeline in real-world, rate-limited environments.

## Nice-to-Haves
- **Supplement proprietary API verification with open-source backbones:** While Face++ and Amazon APIs provide practical verification and are mitigated by the open embedding cosine similarity analysis in Section 7, reporting results against documented open-source FR models (e.g., IResNet/MagFace) would fully satisfy community reproducibility standards and rule out proprietary thresholding artifacts.
- **Spectral or representation-level leakage analysis:** A brief mutual information or feature attribution analysis mapping specific frequency components or subspace dimensions to identity retention would transition the paper from descriptive attack demonstration to diagnostic vulnerability analysis, explaining *why* frequency transforms inherently fail to scatter identity manifolds.
- **Open-set linkage evaluation:** Closed-set 1-to-N linkage (Table 4) establishes an upper bound, but real-world PPFR deployment often involves open-world cross-database matching. Briefly evaluating the extracted embeddings in an open-set retrieval setting would strengthen the practical deployment implications of the attack.

## Novel Insights
The paper's most significant insight is the formal decoupling of visual obfuscation from identity privacy in the facial domain. By demonstrating that a standard, off-the-shelf knowledge distillation step can seamlessly bridge protected template spaces to public embedding manifolds, the authors reveal that frequency-domain PPFR systems are not failing to hide pixels; rather, they are succeeding at preserving a highly structured, machine-readable identity signal that inherently carries soft-biometric and linkage capabilities. The finding that simple high-pass filtering proxies suffice for black-box alignment further implies that current defenses treat visual frequency as a privacy barrier, ignoring that identity-discriminative features naturally concentrate in learnable spectral subspaces regardless of channel shuffling or subtraction. This fundamentally reframes the privacy problem: protecting biometric templates requires formally severing mutual information with identity spaces, not merely increasing reconstruction ill-posedness.

## Suggestions
- **Clarify sample complexity in Section 5.1:** Explicitly state whether the 256/800 image counts represent total paired samples or images per identity, and briefly report variance across multiple random seeds or subset samplings to solidify the minimal-data threat claim.
- **Characterize the utility-privacy trade-off:** Add a controlled experiment or analysis showing how the distillation success rate degrades as the underlying PPFR method's recognition accuracy (e.g., 1-to-1 verification on LFW) drops, establishing a practical boundary for when identity-centric evaluation becomes the limiting factor versus utility preservation.
- **Release an open-source evaluation script alongside the paper:** Package the distillation training loop, embedding similarity computation, and open-source verification inference into a standardized, reproducible benchmark to operationalize your call for identity-centric evaluation metrics.

========================================================================
PREDICTED SCORE
========================================================================

Score: 6.2
Decision: N/A
Total Cost: $0.0002
