
========================================================================
INDIVIDUAL REVIEWS
========================================================================

────────────────────────────────────────
HARSH CRITIC (gpt-5.4 via OpenAI)
────────────────────────────────────────
## Section-by-Section Critical Review

### Title & Abstract
- **Does the title accurately reflect the contribution?**  
  Yes. The title is accurate and appropriately emphasizes the central contribution: rethinking identity leakage in privacy-preserving face recognition via the proposed FaceLinkGen attack. It does not oversell beyond the paper’s actual scope.

- **Does the abstract clearly state the problem, method, and key results?**  
  Mostly yes. The abstract clearly identifies the paper’s main thesis: current PPFR evaluation based on pixel-level reconstruction metrics is insufficient, and the paper proposes an identity-centric attack/evaluation pipeline. It also states the black-box/oracle framing and reports concrete headline results.

- **Are any claims in the abstract unsupported by the paper?**  
  The main issue is **ambiguity in the reported metrics**. The abstract claims “over 98.5% matching accuracy and above 96% regeneration success on three recent frequency-domain PPFR systems.” The matching claim appears supported by the 1-to-1 verification results in **Table 3** (0.988–0.992), but the paper also reports a much harder **1-to-N linkage** setting in **Table 4**, where top-1 recall is often in the 0.72–0.86 range. So the abstract highlights the strongest matching setup without clarifying which task is meant.  
  Similarly, “above 96% regeneration success” seems true if this refers to **Success@5** in **Table 2**, but it is not true for the strictest **Pass@1e-5** numbers, where some entries are below 96% (e.g., FracFace and MinusFace). The abstract should specify which metric it is referring to.

### Introduction & Motivation
- **Is the problem well-motivated? Is the gap in prior work clearly identified?**  
  Yes. The introduction does a strong job motivating why **pixel-level reconstruction resistance** is not the same as **identity privacy**, and this is an important critique for the PPFR literature. The examples around **Figure 2**, **Table 1**, and the discussion of failed pixel-centric attacks in **Figure 3** effectively convey the intuition. For an ICLR audience, this framing is compelling because it challenges the validity of an accepted evaluation practice rather than merely proposing another attack variant.

- **Are the contributions clearly stated and accurate?**  
  The contribution paragraph at the end of **Section 1** is clear and mostly accurate: the paper contributes (i) an identity-centric critique of prior evaluation, (ii) an attack pipeline for linkage and regeneration, and (iii) pilot transfer results beyond the main frequency-domain PPFR setting. That said, the transfer claims to **TIP-IM** and **CanFG** are much more limited in scale than the core experiments, so the wording should remain cautious.

- **Does the introduction over-claim or under-sell?**  
  There is some **overgeneralization**. Statements like the current paradigm being “significantly limited” and that prior red-team researchers are “inadvertently trapped” by pixel losses are directionally persuasive, but the paper only empirically evaluates **three open-source frequency-domain PPFR systems** plus small pilot studies outside that family. For ICLR standards, the central insight is strong, but the introduction occasionally reads as if it has settled the broader question for PPFR as a whole. The evidence clearly supports the critique for the evaluated methods; it does not yet fully establish a general impossibility or universal failure of all reconstruction-based privacy evaluations.

### Related Work
- **Are the most relevant baselines and prior work cited?**  
  The paper cites the main recent PPFR systems it attacks or discusses: **DuetFace [19]**, **PartialFace [20]**, **MinusFace [21]**, **FracFace [6]**, **FaceObfuscator [14]**, and also touches model inversion, ID-conditioned generation, and de-identification literature. For the paper’s empirical scope, this is reasonably strong.

- **Is the positioning against prior work fair and accurate?**  
  Mostly yes. The paper is careful to distinguish its setting from classic **training-data model inversion** (**[9], [36]**) and from the more “invertible embedding” generation works such as **Arc2Face [25]** and **PuLID [11]**. The distinction between attacking the template-generation pipeline and inverting a standard face embedding is important and generally well explained.  
  One place where the positioning could be more careful is in the comparison with **FracFace** in **Table 6**. The paper explicitly notes that FracFace’s “Protection (%)” is a different metric, but the side-by-side table still risks implying a more direct apples-to-apples comparison than is actually justified.

- **Are there obvious missing references?**  
  The related work coverage is solid for the specific attacked literature, but for an ICLR submission I would have expected a somewhat broader connection to:
  1. **biometric template protection / cancelable biometrics** beyond the specific face papers;
  2. prior work on **identity leakage from intermediate or transformed representations**;
  3. evaluation methodologies for **privacy leakage beyond reconstruction**, especially in representation learning and privacy/security communities.  
  This is not a fatal omission, but stronger positioning against the broader privacy literature would improve the paper’s contribution framing for ICLR.

### Method / Approach
- **Is the method clearly described and reproducible?**  
  The high-level idea is very clear: distill from protected templates into a standard face-embedding space using a teacher-student objective (**Eq. 2**), then use nearest-neighbor matching (**Eq. 4**) and a generative model (**Eq. 3**) for linkage/regeneration. This simplicity is a strength.  
  However, **reproducibility is currently below ICLR expectations**. Important implementation details are missing or too sparse:
  - the exact student architecture beyond “Antelopev2 with one additional 3×3 Conv2D layer” in **Section 5**;
  - training hyperparameters (optimizer, learning rate, epochs, augmentation policy, preprocessing, batch size except indirectly in Section 5.1);
  - how template formats from different PPFR methods are normalized before input;
  - precise settings for the constrained attack in **Section 6** (blur kernel sizes, strengths, augmentation ranges, how the 30 validation pairs are used).  
  The idea is reproducible in principle, but not yet at the level I would expect for a top-tier ICLR paper without code.

- **Are key assumptions stated and justified?**  
  The **threat model in Section 2** is one of the paper’s stronger aspects. The insider/service-provider adversary with oracle access is well motivated and arguably closer to the original PPFR threat model than some external-attacker framings in prior work. This is an important and fair reframing.  
  That said, the main assumption still deserves more calibration: the attacker effectively needs access to enough paired examples to train the student, or at least practical means to query/collect them. The paper argues this is realistic for a service provider and partially explores reduced-data settings in **Section 5.1**, which helps. But for deployments where the provider does not have practical paired data collection or where the transform is keyed/user-specific, the attack may be less straightforward. The paper acknowledges some of this in **Section 10**, but it should be more explicit earlier.

- **Are there logical gaps in the derivation or reasoning?**  
  A few:
  1. **Equation (1)** models templates as \(T \sim p(\cdot \mid z_I)\), effectively suppressing nuisance information \(z_N\). This is useful intuition, but it is too strong as a formal statement for the attacked methods. Later sections actually rely on the fact that templates preserve visually meaningful high-frequency information (**Section 6**) and can leak soft biometrics (**Section 9**), which suggests \(T\) is not close to identity-only.  
  2. The claim in **Section 3** that “As long as \(z_I\) persists in \(T\), an attacker can recover \(z'_I\)” is stronger than what the paper proves. The paper gives strong empirical evidence for the evaluated systems, but not a theoretical guarantee or even sufficient conditions for recoverability.
  3. The method’s success may depend materially on the existence of a strong public embedding/generator pair such as **Antelopev2 + Arc2Face**. The paper says other embedding models could also be used, but this is asserted rather than demonstrated.

- **Are there edge cases or failure modes not discussed?**  
  Yes. The paper would benefit from discussing:
  - keyed or user-specific transforms;
  - stronger stochastic transforms;
  - PPFR methods that deliberately degrade recognition more aggressively;
  - scenarios where public embedding spaces are poorly aligned with the protected-template geometry;
  - non-frontal or low-quality image regimes;
  - demographic or domain shifts affecting attack success.  
  **Section 6** begins to probe a more constrained attacker, but it does not systematically characterize when the approach fails.

- **For theoretical claims: are proofs correct and complete?**  
  There are no real theoretical results or proofs. That is acceptable for an empirical security paper, but then the claims should stay empirical and avoid sounding universal.

### Experiments & Results
- **Do the experiments actually test the paper's claims?**  
  Broadly, yes. The main claim is that pixel-level privacy metrics can miss identity leakage in frequency-domain PPFR. The experiments on **PartialFace, MinusFace, and FracFace** directly test this via:
  - **linkage** (**Tables 3 and 4**),
  - **regeneration** verified by external APIs (**Tables 2 and 5**),
  - **limited-resource training** (**Section 5.1**),
  - **constrained assumptions** (**Section 6**),
  - and **soft-biometric leakage** (**Table 9**).  
  This is a fairly comprehensive empirical package, and for ICLR the diagnostic value is substantial.

- **Are baselines appropriate and fairly compared?**  
  The chosen attacked systems are appropriate: they are recent, visible, and open-source. However, some important **attack-side baselines are missing**:
  1. What happens if one applies the teacher embedding model **directly** to protected templates, without student distillation?  
  2. How much of the gain comes from the extra **Conv2D adapter** versus full student training?  
  3. Could a much simpler linear probe or MLP suffice?  
  These ablations matter because the paper’s key rhetorical point is that the vulnerability is severe enough that a **simple** procedure works. To support that, the paper should quantify how simple the attack can be before performance drops.  
  Also, the comparison to FracFace’s U-Net/StyleGAN attack in **Table 6** is only partially fair because it uses a different success metric. The paper explains this, but the direct juxtaposition still overstates comparability.

- **Are there missing ablations that would materially change conclusions?**  
  Yes, several would materially strengthen the paper:
  - **Teacher model ablation**: Is this specific to Antelopev2/Arc2Face, or does it hold with other embedding spaces?
  - **Training data size ablation**: Section 5.1 is helpful, but only on FracFace. It should be extended across all attacked methods.
  - **Identity overlap control for LFW**: the student is trained on CASIA-WebFace, and the paper does not state whether LFW identities overlapping with CASIA were removed. Since both are celebrity-heavy datasets, this matters.
  - **Seed variance / repeatability** for both student training and generation. With results so high, this may not change the core conclusion, but ICLR expects some measure of robustness.
  - **Attack without Arc2Face**: Section 7 helps by showing embedding-space similarity, but more systematic non-generative evidence would strengthen the claim that identity leakage is not an artifact of a powerful generator prior.

- **Are error bars / statistical significance reported?**  
  No. This is a weakness, especially for:
  - the 1-to-N linkage results in **Table 4**,
  - the reduced-data experiments in **Section 5.1**,
  - and the constrained-assumption attack in **Table 7**.  
  For the near-100% verification results, confidence intervals may be narrow, but some notion of uncertainty or variance across seeds would still be expected at ICLR.

- **Do the results support the claims made, or are they cherry-picked?**  
  The core claim is supported: identity leakage clearly exists despite prior reconstruction-centric evaluations. The results are striking and consistent across multiple evaluations.  
  That said, some **claim calibration** is needed:
  - In **Section 5.2**, the statement that the attack “essentially reach[es] the dataset’s theoretical maximum performance” is too strong given that several cross-method linkage numbers in **Table 4** are around 0.72–0.78 versus the original-original upper bound of 0.882.
  - The paper often foregrounds the strongest 1-to-1 or Success@5 numbers, while the more demanding 1-to-N and strict-threshold numbers are lower.
  - The extension claims beyond frequency-domain PPFR (e.g., **TIP-IM**, **CanFG**) are based on smaller pilot experiments and should remain framed as preliminary.

- **Are datasets and evaluation metrics appropriate?**  
  Mostly yes. Using **CASIA-WebFace hold-out**, **LFW**, and **TPDNE** gives a reasonable mix of in-domain, cross-domain, and synthetic-no-overlap evaluation. The use of external commercial APIs for verification is defensible and, in some ways, strengthens the independence of the evaluation.  
  Still, a few concerns remain:
  - **LFW overlap with CASIA-WebFace** is not addressed and could inflate cross-dataset claims.
  - **Face++ and Amazon APIs** improve independence but reduce reproducibility; the paper should ideally provide more detail on API settings and exact protocols.
  - Vendor thresholds (FAR \(10^{-3},10^{-4},10^{-5}\)) are convenient, but they are not necessarily calibrated for these specific generated-image distributions, so interpreting them as absolute security levels should be done cautiously.

### Writing & Clarity
- **Are there sections that are confusing or poorly explained?**  
  The paper is generally easy to follow, and the storyline is strong. **Section 2 (Threat Model)** is particularly clear and helpful. **Figure 4** gives a good high-level overview of the pipeline.  
  The main clarity issues are conceptual rather than stylistic:
  - The paper sometimes slides between **verification accuracy**, **closed-set linkage recall**, and **regeneration success** without always making the distinction prominent enough, especially between the abstract, **Table 3**, and **Table 4**.
  - **Equation (2)** and the notation around teacher/student alignment are understandable, but the formalism is lighter than the paper’s confident language suggests.
  - **Table 6** is potentially misleading because it compares two very different notions of “protection.”
  - The constrained-assumption setup in **Section 6** is interesting but under-specified; it is not fully clear how the 30 paired samples are used operationally.

- **Are figures and tables clear and informative?**  
  Yes overall.  
  - **Figure 1** effectively communicates that regenerated faces preserve identity-like traits.  
  - **Figure 3** supports the argument that pixel-level reconstruction objectives fail differently from ID-guided generation.  
  - **Figure 5** is conceptually useful because it supports the paper’s claims without relying on Arc2Face.  
  The tables are mostly informative, though **Table 6** needs more careful framing.

### Limitations & Broader Impact
- **Do the authors acknowledge the key limitations?**  
  Partially. The paper acknowledges some scope limitations, especially by describing the non-frequency-domain results as **pilot evaluations** and by discussing stronger defenses in **Section 10**. It also explicitly separates the main oracle-access threat model from the more constrained setting in **Section 6**, which is good.

- **Are there fundamental limitations they missed?**  
  Yes:
  1. **Generality**: the strongest evidence is for three frequency-domain PPFR methods. The paper’s broader claims about PPFR or de-identification mechanisms in general are plausible, but not fully established.
  2. **Dependence on public embedding/generation models**: the attack may be easier because modern face-generation models are now very strong. This is part of the threat, but it also means conclusions should be contextualized as partly contingent on the current generative ecosystem.
  3. **Paired-data acquisition**: although realistic for a malicious service provider, the practicality of collecting paired template-image data varies across deployments and should be discussed more explicitly.
  4. **Potential overlap between training and evaluation identities** on LFW is not discussed and may affect the strength of some claims.

- **Are there failure modes or negative societal impacts not discussed?**  
  The broader-impact discussion is underdeveloped for an ICLR submission. This paper presents a potent deanonymization/regeneration attack against facial privacy systems, yet there is little discussion of:
  - responsible disclosure to the affected method authors,
  - safe release of code or generation models,
  - possible misuse for doxxing, surveillance, or identity fraud,
  - demographic disparities in attack success or harm,
  - risks from using scraped face datasets and commercial APIs.  
  Given the domain, ICLR reviewers will likely expect a more explicit and serious broader-impact section.

### Overall Assessment
This is a strong and important empirical critique of current evaluation practice in frequency-domain PPFR: the central insight—that low pixel-level reconstructability does not imply identity privacy—is convincing, timely, and supported by substantial experimental evidence on three prominent open-source methods. The threat-model discussion is thoughtful, and the results in Tables 2–5 and 9 are hard to ignore. The main reasons this feels **borderline rather than clearly above the ICLR bar** are rigor and calibration: the paper lacks several key ablations, does not report variance/statistical uncertainty, leaves some reproducibility details underspecified, and occasionally overstates the generality of conclusions beyond the evaluated methods. If the authors tighten the claim scope, clarify metric reporting, control for possible dataset overlap, and add stronger attack-side baselines/ablations, the contribution would be much more competitive for ICLR. As written, the core contribution is real and meaningful, but the paper needs more experimental discipline to fully clear ICLR’s standards.

────────────────────────────────────────
NEUTRAL REVIEWER (gpt-5.4-mini via OpenAI)
────────────────────────────────────────
## Balanced Review

### Summary
This paper argues that frequency-domain privacy-preserving face recognition (PPFR) systems are often evaluated with the wrong privacy target: resistance to pixel-level reconstruction does not necessarily imply resistance to identity leakage. To support this, it introduces FaceLinkGen, a simple distillation-based attack that extracts identity embeddings from protected templates and uses them for linkage and face regeneration, reporting high success on three recent PPFR systems and additional evidence on de-identification methods.

### Strengths
1. **Important and timely evaluation insight.**  
   The paper makes a strong conceptual point that is highly relevant to ICLR’s interest in evaluation methodology: pixel similarity metrics like PSNR/SSIM are not a reliable proxy for identity privacy. The argument is well motivated with concrete examples and supported by results showing high regeneration and linkage success despite poor pixel reconstruction.

2. **Empirical evidence across multiple systems and settings.**  
   The attack is tested on three recent frequency-domain PPFR methods (PartialFace, MinusFace, FracFace), multiple datasets (TPDNE, CASIA-WebFace hold-out, LFW), and multiple attack modes (linkage, regeneration, constrained-assumption setting, soft-biometric leakage). This breadth strengthens the claim that the issue is not isolated to one method.

3. **Simple attack pipeline with practical implications.**  
   A notable strength is that the attack is intentionally lightweight: a student model is trained via distillation, then identity-consistent faces are generated via a diffusion model. The paper’s point is persuasive because the attack does not require complex optimization or access to the target model internals, yet still yields strong results.

4. **Good use of complementary metrics.**  
   The paper goes beyond reconstruction metrics and includes commercial face verification, linkage accuracy, cosine similarity distributions, and soft-biometric prediction. This multi-angle evaluation helps substantiate the claim that identity information remains accessible even when images are not pixel-reconstructible.

5. **Clear security motivation and threat-model discussion.**  
   The paper spends substantial effort clarifying the intended adversary in PPFR and why the usual “reconstruction attack” framing may be incomplete. This is useful for the community, especially if future PPFR work adopts more identity-centric evaluations.

### Weaknesses
1. **Technical novelty is more in reframing/evaluation than in attack methodology.**  
   The attack itself is fairly straightforward: distill to a face embedding space, then feed the embedding to a known identity-conditioned generator. While effective, the core method appears to be a relatively direct composition of existing components rather than a fundamentally new learning or inversion technique. For ICLR, this may reduce the paper’s novelty score unless the evaluation insight is treated as the main contribution.

2. **The scope of the evidence is narrower than some of the claims.**  
   Most of the strongest results are on frequency-domain PPFR systems with accessible source code. The paper also includes pilot evidence on CanFG and TIP-IM, but these are secondary and do not fully establish the broader claim that “privacy-preserving face recognition” generally fails under identity-centric evaluation. The generalization beyond the evaluated family is suggestive, not conclusive.

3. **Dependence on proprietary verification services affects reproducibility and interpretability.**  
   Regeneration success is measured primarily with Face++ and Amazon verification APIs, which are not fully transparent or reproducible. Although the paper adds ArcFace-space analysis as a cross-check, the main headline success numbers rely on closed systems whose calibration, thresholds, and failure modes are not inspectable.

4. **Threat-model assumptions are not fully settled.**  
   The paper argues for an insider/service-provider adversary with oracle access, but some parts of the evaluation also consider constrained external attackers. This broadening is reasonable, yet the paper sometimes moves between these settings in a way that may blur what exactly is assumed in the main claims. A stricter separation of attacker capabilities would improve rigor.

5. **Limited analysis of why the attack works and when it might fail.**  
   The empirical results are strong, but the paper offers relatively little mechanistic explanation of the representation geometry, the role of template design, or the conditions under which identity extraction degrades. For ICLR, a deeper analysis would help turn the paper from a strong empirical warning into a more general scientific understanding.

6. **Some evaluation choices may overstate performance or leave confounds unresolved.**  
   The same general face-embedding ecosystem is used for distillation, linkage, and parts of the evaluation, which may make the attack look stronger than it would under a more disjoint setup. The paper does attempt to mitigate this with different datasets and third-party APIs, but a more systematic cross-model / cross-embedding ablation would be valuable.

### Novelty & Significance
The paper’s main novelty is in its **identity-centric reinterpretation of privacy leakage** for PPFR and in showing that simple distillation can recover enough identity signal to enable both linkage and regeneration. This is a meaningful and potentially influential result, especially because it challenges a common evaluation practice; however, the attack mechanism itself is not highly algorithmically novel.

In terms of significance, the work is strong: if the results hold broadly, they expose a serious weakness in an active research area and could influence how PPFR and face de-identification methods are evaluated. Against ICLR’s standards, I would rate it as **important and empirically compelling, but somewhat narrow in technical depth and breadth of validation**.

### Suggestions for Improvement
1. **Add deeper ablations on the attack pipeline.**  
   Quantify the contribution of each component separately: student architecture choice, teacher embedding choice, generator choice, training set size, and the effect of using different public embedding models beyond ArcFace/Antelopev2.

2. **Strengthen reproducibility with open, inspectable evaluation.**  
   Report more results using fully reproducible metrics and models, not only proprietary verification APIs. If closed APIs must be used, provide calibration details, confidence intervals, and a fallback open-source verification protocol.

3. **Clarify and formalize the threat model.**  
   Separate the insider/oracle attacker from the constrained external attacker and state clearly which claims hold under which assumptions. A concise table mapping each experiment to its threat model would help.

4. **Broaden the comparison set and failure analysis.**  
   Include more baselines, especially simpler embedding attacks or direct inversion baselines under the same conditions, and analyze cases where the attack fails or weakens. This would make the paper’s claims more precise and scientifically informative.

5. **Add mechanistic analysis of template information content.**  
   Provide evidence about what information is preserved in the protected template and why identity is recoverable even when pixels are not. For example, analyze embedding separability, template-to-identity mutual information proxies, or frequency-band sensitivity.

6. **Tighten the scope of the main claim.**  
   Consider phrasing the contribution more carefully as a strong evaluation result for the evaluated PPFR family, rather than implying universal vulnerability across all privacy-preserving face methods. This would make the paper more rigorous and credible under ICLR review standards.

────────────────────────────────────────
SPARK FINDER (gpt-5.4 via OpenAI)
────────────────────────────────────────
## Strengthening Opportunities

### Missing Experiments
1. **Broaden the attacked-method coverage to better meet ICLR’s generalization bar.**  
   In addition to PartialFace, MinusFace, and FracFace, the paper would be stronger with evaluations on other PPFR families such as **DuetFace**, **FaceObfuscator**, and **learnable privacy budget in frequency domain** methods, plus at least one **non-frequency cancelable-biometric** baseline. This would show whether the identity-centric vulnerability is a property of a broad design pattern rather than three representative systems.

2. **Add stronger attack baselines beyond the proposed distillation pipeline.**  
   ICLR reviewers will likely want to know how much of the gain comes from the paper’s framing versus the exact attack implementation. It would help to compare against:  
   - prior **U-Net / StyleGAN / reconstruction** attacks from the original PPFR papers,  
   - a **linear probe** from template to ArcFace embedding,  
   - a small **MLP/CNN classifier** trained directly for identity retrieval,  
   - a **contrastive** student instead of cosine distillation,  
   - a direct **template-to-template kNN** baseline without distillation.  
   This would clarify the minimal attack needed and underscore how accessible the leakage is.

3. **Systematically vary the attacker’s supervision budget across all methods, not just one.**  
   The 800-image and 256-image FracFace study is very compelling. The paper would become even stronger with **data-scaling curves** for all three methods (e.g., 32, 64, 128, 256, 512, 1k, 5k, 10k paired samples), reporting both linkage and regeneration. This would quantify sample efficiency and align well with ICLR’s preference for mechanistic understanding.

4. **Include cross-backbone transfer experiments.**  
   Since the attack uses ArcFace/Antelopev2 plus Arc2Face, it would be valuable to train/extract with one identity model and evaluate with others such as **AdaFace, CurricularFace, FaceNet, MagFace**, or a commercial API not used during training. This would show whether the leakage is tied to a specific embedding space or is backbone-agnostic.

5. **Evaluate open-set identification, not only closed-set top-1 linkage.**  
   Closed-set top-1 on a 2,115-gallery set is useful, but ICLR reviewers will likely expect more realistic retrieval settings. Adding **open-set ROC/DET/CMC curves**, **larger galleries** (e.g., CASIA full hold-out, MegaFace-style distractors if feasible), and **rank-k retrieval** would better characterize practical privacy risk.

6. **Use more challenging face benchmarks with greater nuisance variation.**  
   The paper would be strengthened by testing linkage/regeneration on **CFP-FP** (frontal-profile), **AgeDB** (age variation), **IJB-C** (unconstrained templates), and **RFW** (demographic diversity). This would reveal whether leakage persists under the kinds of pose, age, and domain shifts ICLR reviewers often look for.

7. **Expand the constrained-attacker study into a full spectrum of assumptions.**  
   Section 6 is promising. It would be stronger with a more systematic ladder:  
   - oracle access,  
   - limited paired samples,  
   - no paired samples but same template family,  
   - unknown preprocessing/alignment,  
   - stochastic templates / multiple random seeds,  
   - quantized or noisy templates.  
   This would sharpen the boundary of where the attack remains effective.

8. **Test revocability and cross-key linkage if the PPFR systems support randomness.**  
   A highly relevant privacy question is whether the same identity can be linked across two independently protected versions. The paper would be stronger with experiments on **same-person different random seeds / channel selections / sessions**, measuring whether FaceLinkGen still links them. This connects directly to cancelability and template renewability.

9. **Strengthen regeneration evaluation with more identity criteria than pairwise verification.**  
   In addition to Face++ and Amazon pass/fail, it would help to report whether regenerated faces are **retrieved as the correct identity in a large gallery**, and whether they match **other photos** of the same person, not only the source image. This would show that the attack recovers identity rather than just source-specific resemblance.

10. **Add demographic-disaggregated leakage results.**  
    Since the paper already studies soft biometrics, it would be valuable to break down linkage/regeneration success by **gender, age bin, and race** (e.g., using FairFace labels or RFW). ICLR often values analysis of whether a method behaves uniformly or unevenly across subpopulations.

11. **Include calibration against utility-preserving baselines.**  
    A particularly informative experiment would be a **privacy-utility frontier**: for each PPFR method’s hyperparameter settings, plot recognition utility versus identity leakage under FaceLinkGen. This would show whether existing methods are operating on an unfavorable frontier or whether some settings meaningfully trade utility for privacy.

12. **Validate on video or multi-image enrollment if feasible.**  
    Many practical systems use multiple views or video frames. Evaluating on **video face tracks** or multi-image per identity could show whether template averaging or multiple observations make leakage even easier, which would increase the applied relevance.

### Deeper Analysis Needed
1. **Formalize the utility–leakage connection.**  
   A major ICLR-strengthening opportunity is a theorem or proposition explaining when recognition utility necessarily implies extractable identity information. Even a modest result using **data processing**, **Bayes risk**, or **mutual-information-style arguments** would elevate the paper from an empirical attack paper to a more general scientific contribution.

2. **Characterize why distillation works so easily.**  
   The paper would benefit from analysis showing whether the template-to-embedding mapping is largely **linear**, **low-complexity**, or **spectrally structured**. For example, linear-probe results, singular value spectra, CKA/CCA similarity, or probe-depth ablations could reveal whether the representation is inherently aligned with standard FR spaces.

3. **Provide a sample-complexity perspective on attackability.**  
   Since 256 images already work surprisingly well, the paper would be stronger with either empirical fits or lightweight theory on how performance scales with the number of paired samples. This would turn the “low-cost attack” claim into a more principled observation.

4. **Analyze which template components leak identity.**  
   For frequency-domain PPFR, a natural addition is a **channel/frequency ablation study**: which bands, channels, or template regions are most predictive of identity? This could connect the attack back to design choices in PartialFace/MinusFace/FracFace and offer insight useful to both attackers and defenders.

5. **Separate identity leakage from soft-biometric leakage more explicitly.**  
   The soft-biometric section is strong; it would become even stronger with analysis of whether age/gender/race leakage is merely correlated with identity leakage or partially independent. For instance, predicting attributes after removing identity-related nearest neighbors could help disentangle these effects.

6. **Clarify the role of the teacher space.**  
   Because the method distills to ArcFace-compatible embeddings, one important analysis is: would leakage still be strong if the teacher were an unrelated public FR model? Cross-teacher robustness would support the broader claim that the template retains identity, not just ArcFace-specific information.

7. **Quantify attack cost more comprehensively.**  
   The runtime/cost numbers are excellent. The paper would be even more persuasive with a compact complexity analysis covering **time, paired samples, GPU hours, and query requirements** for linkage and regeneration under each threat model. ICLR reviewers often appreciate this kind of “attack surface accounting.”

8. **Relate the findings more explicitly to representation learning literature.**  
   The work would deepen with a connection to **knowledge distillation**, **domain adaptation**, and **semantic inversion**. Framing PPFR templates as a learnable alternate view of identity could position the paper as more than a niche security result.

9. **Formalize the constrained high-pass proxy attack.**  
   Section 6 would be stronger with a clearer hypothesis test: e.g., measure the correlation between true protected templates and generic high-pass-filtered images, and show that this proxy alignment predicts attack success. That would make the constrained-assumption result feel more principled.

10. **Define privacy metrics beyond pass rates.**  
    A useful addition would be identity-centric metrics such as **linkability score**, **re-identification risk**, **min-entropy reduction**, or **expected rank of true identity**. This could help the paper propose not only an attack but also a better evaluation toolkit for future PPFR work.

### Untapped Applications
1. **Cancelable biometrics beyond faces.**  
   The core idea should transfer naturally to **fingerprint, iris, palmprint, voiceprint, and gait** template protection. Showing even one additional biometric modality would substantially broaden the contribution in a way that fits ICLR’s interest in general mechanisms.

2. **Video anonymization and person re-identification.**  
   The paper could be extended to **video face de-identification**, **tracking across cameras**, or **person re-ID** settings, where linkage risk is often more important than exact reconstruction.

3. **Synthetic or renewable biometric systems.**  
   Since the paper already discusses CanFG and revocability, a natural next application is **revocable/renewable templates**, where cross-version linkage is a key privacy concern.

4. **Federated or on-device biometric systems.**  
   The identity-centric evaluation protocol could be applied to **federated face recognition**, **mobile unlock embeddings**, or **edge-device biometric APIs**, where representations are shared but raw images are not.

5. **Human-recognizable de-identification pipelines.**  
   The TIP-IM result suggests a broader opportunity: benchmark the same attack on **makeup-based**, **adversarial patch**, and **diffusion-based de-identification** methods. This could connect two currently separate literatures.

6. **Sensitive deployment domains.**  
   Demonstrating the attack in settings like **KYC/onboarding**, **healthcare image anonymization**, **education proctoring**, or **border control** would make the practical stakes even clearer.

7. **Privacy auditing tools for commercial APIs.**  
   A very practical extension is turning FaceLinkGen into a general **privacy audit protocol** for vendors claiming “protected template” or “face anonymization” systems.

### Visualizations & Case Studies
1. **A privacy–utility frontier figure for each attacked method.**  
   Plot the original method’s recognition utility against FaceLinkGen leakage as protection strength varies. This single figure would likely be very effective for ICLR readers.

2. **Embedding-space visualizations.**  
   UMAP/t-SNE plots showing **original images**, **template-derived embeddings**, and **generated faces** would make the alignment effect intuitive and visually compelling.

3. **Data-scaling and threat-scaling curves.**  
   Showing attack performance versus **number of paired samples**, **student capacity**, and **knowledge assumptions** would make the paper’s central message much more quantitative.

4. **Retrieval case studies for linkage.**  
   For several identities, show the **query image/template**, **top-5 retrieved templates/images**, and where the correct identity appears. This would make the linkage risk concrete.

5. **Failure-case analysis.**  
   ICLR reviewers often value understanding when a method breaks. It would help to show cases where regeneration or linkage fails, organized by **pose**, **occlusion**, **age gap**, **lighting**, **alignment error**, or **dataset noise**.

6. **Spectral heatmaps or saliency maps.**  
   Since these are frequency-domain methods, visualizing which template regions/channels contribute most to the student embedding would provide insight into what the protection leaves exposed.

7. **Similarity-distribution plots across more conditions.**  
   Figure 5 is a good start. It would be even stronger to add distributions for:  
   - different identity pairs,  
   - same identity across datasets,  
   - source vs generated image,  
   - template vs regenerated image.  
   This would better triangulate the leakage story.

8. **Cross-method linkage matrices under multiple teachers.**  
   Table 4 is informative; a heatmap version, repeated for different FR backbones, would make cross-protection interoperability of leakage immediately visible.

9. **Soft-biometric leakage visualizations.**  
   Confusion matrices for race/gender and error histograms for age would make the attribute leakage section more complete.

10. **Case studies on renewability.**  
    If random seeds/keys are available, visual examples of two different protected templates for the same user linking back to one identity would be especially impactful.

### Natural Next Steps
1. **Establish an identity-centric benchmark for PPFR.**  
   A major next step is a standardized benchmark with **linkage**, **open-set identification**, **regeneration**, **soft-biometric leakage**, and **utility-preservation** metrics. This could become a community resource and would align very well with ICLR’s appetite for benchmark-shaping work.

2. **Develop defenses explicitly against distillation-based identity extraction.**  
   Since the paper shows simple distillation is enough, the next generation of PPFR could be trained adversarially against **cross-model distillation**, not only pixel reconstruction.

3. **Study keyed or user-specific transformations under identity-centric evaluation.**  
   The paper already points toward this. A natural follow-up is to test whether **keyed/randomized/revocable** designs reduce cross-template linkage while retaining verification accuracy.

4. **Explore formal privacy guarantees or lower bounds.**  
   A very strong research direction is to derive **provable limits** on privacy given target recognition utility, potentially using information-theoretic tools. That would turn the current empirical insight into a broader theory of PPFR feasibility.

5. **Move from paired supervision to self-supervised or unpaired attacks.**  
   Since the constrained attack already uses a generic proxy, the next step is to see how far attackability extends with **no paired data**, **synthetic pairing**, or **domain adaptation**. This would further probe the real-world risk.

6. **Design representations that preserve verification but suppress identity transferability.**  
   One promising avenue is to preserve utility only for a **task-specific matcher** while minimizing compatibility with public embedding spaces. Whether this is achievable is itself a valuable research question.

7. **Extend the attack to multimodal identity leakage.**  
   Future work could investigate whether protected face templates also allow recovery of **attributes**, **kinship**, **emotion priors**, or linkage to **voice / gait / social media photos**.

8. **Audit human-perception versus machine-perception privacy more directly.**  
   Since the paper argues against reconstruction-centric evaluation, a next step is a joint benchmark combining **commercial FR**, **open-source FR**, and **human judgment** to understand which privacy notions are actually protected.

9. **Investigate defense–attack co-training.**  
   A particularly ICLR-friendly next step is a min-max setup where PPFR methods are trained against identity-centric attackers like FaceLinkGen, analogous to adversarial robustness but in representation privacy space.

10. **Generalize beyond face recognition to representation privacy.**  
    The paper’s central message may apply broadly: “non-reconstructable does not imply non-linkable.” Extending this principle to other learned protected representations could turn the work into a wider machine learning privacy agenda.

────────────────────────────────────────
POTENTIALLY MISSED RELATED WORK (gpt-5.4-nano via OpenAI)
────────────────────────────────────────
## Potentially Missed Related Work

(These are suggestions, not definitive omissions. The authors may have intentionally excluded them or been unaware of them.)

1. **Template Inversion Attack against Face Recognition Systems using 3D Face Reconstruction (GaFaR)** — Hatef Otroshi Shahreza, Sébastien Marcel (2023, ICCV).  
   Why potentially missed: Provides a strong, directly relevant precedent for **recovering identity-revealing face information from protected representations/templates**, supporting the paper’s core argument that “non-reconstructability under pixel metrics” does not imply “no identity leakage.”

2. **Face Reconstruction from Facial Templates by Learning Latent Space of a Generator Network** — Hatef Otroshi Shahreza, Sébastien Marcel (2023, NeurIPS).  
   Why potentially missed: A closely aligned template-inversion/regeneration approach showing **face synthesis from templates/embeddings** (not necessarily pixel recovery), which matches the paper’s “regenerate without original pixel recovery” framing.

3. **Vulnerability of State-of-the-Art Face Recognition Models to Template Inversion Attack** — Hatef Otroshi Shahreza, Vedrana Krivokuća Hahn, Sébastien Marcel (2024, IEEE TIFS).  
   Why potentially missed: Offers rigorous evaluation evidence for **template inversion feasibility and identity leakage**, which complements FaceLinkGen’s claim that existing PPFR evaluations may be incomplete without identity-centric metrics.

4. **Privacy-Preserving Face Recognition Utilizing Differential Privacy (PEEP)** — M.A.P. Chamikara, P. Bertok, I. Khalil, D. Liu, S. Camtepe (2020, Computers & Security).  
   Why potentially missed: An earlier PPFR line grounded in **privacy guarantees via DP**, potentially relevant to the paper’s discussion of why purely transformation-based/protection-by-obfuscation approaches may be insufficient and to motivate more formal defenses.

========================================================================
FINAL CONSOLIDATED REVIEW (gpt-5.4 via OpenAI)
========================================================================

{
  "summary": "This paper makes a strong and important empirical point: for the evaluated frequency-domain privacy-preserving face recognition (PPFR) systems, resistance to pixel-level reconstruction is not a reliable proxy for identity privacy. The proposed FaceLinkGen pipeline is simple but effective: distill protected templates into a standard face-embedding space, then use the recovered identity signal for linkage and identity-consistent regeneration. The experiments on PartialFace, MinusFace, and FracFace are striking, and the broader message about identity-centric evaluation is valuable for the community. The main limitations are that the methodological novelty is modest relative to the strength of the evaluation insight, the strongest evidence is concentrated on three open-source frequency-domain methods, and several ablations/reproducibility details are missing. Overall, this is a meaningful and timely critique of current PPFR evaluation practice, but the paper would be stronger with tighter claim calibration, better attack-side baselines, and more rigorous experimental reporting.",
  "strengths": [
    "Important conceptual contribution: the paper clearly shows that low PSNR/SSIM reconstruction quality does not imply low identity leakage, which is a consequential evaluation lesson for PPFR research.",
    "Strong empirical package on the main target family: three recent frequency-domain PPFR methods are attacked across multiple datasets and via multiple attack modes (1-to-1 verification, 1-to-N linkage, regeneration, constrained setting, soft-biometric leakage).",
    "The threat-model discussion is unusually thoughtful and mostly well aligned with the original PPFR motivation: the insider/service-provider adversary with oracle access is well justified rather than assumed away.",
    "The attack pipeline is intentionally simple and cheap, which strengthens the paper's practical significance: if a lightweight teacher-student distillation already suffices, the leakage is likely in the representation itself rather than in a specially engineered attack.",
    "The paper uses several complementary forms of evidence beyond visual examples, including commercial verification APIs, linkage metrics, embedding-space similarity analysis, and soft-biometric prediction.",
    "The reduced-resource results are especially compelling: high attack success with very small amounts of supervision suggests the leakage is not fragile or narrowly overfit."
  ],
  "weaknesses": [
    "The main novelty is more in reframing and evaluation than in the attack mechanism itself. FaceLinkGen is effective, but it is largely a composition of known ingredients (distillation into a public FR embedding space plus identity-conditioned generation).",
    "The strongest conclusions are supported for three open-source frequency-domain PPFR systems. The paper includes pilot evidence beyond that family, but some wording still overgeneralizes from these cases to PPFR or de-identification more broadly.",
    "Several important attack-side baselines and ablations are missing. In particular, it would help to compare against simpler alternatives such as direct teacher inference on templates, a linear probe / small MLP, template-space kNN, and cross-teacher or cross-backbone variants.",
    "Metric reporting and claim calibration need tightening. The abstract highlights the strongest matching and regeneration numbers without clearly distinguishing 1-to-1 verification from harder 1-to-N linkage, or Success@5 from stricter pass-rate thresholds. Table 6 also juxtaposes different notions of 'protection' in a way that can be misread as a direct apples-to-apples comparison.",
    "Reproducibility is somewhat weaker than ideal: important training details for the student model and the constrained attack are underspecified, and the headline regeneration results rely heavily on proprietary APIs (Face++, Amazon), even though the paper does provide some independent embedding-space analysis.",
    "No uncertainty estimates are reported (e.g., seed variance or confidence intervals), and a potentially important confound is not resolved: LFW/CASIA identity overlap is not discussed, which could affect the claimed cross-dataset generalization.",
    "Some of the paper's stronger interpretive statements are overstated relative to the data, such as the suggestion that the reported linkage results essentially reach the dataset's theoretical maximum across settings."
  ],
  "nice_to_haves": [
    "Broader attacked-method coverage, especially additional PPFR families such as DuetFace or FaceObfuscator, and at least one stronger non-frequency cancelable-biometric baseline.",
    "Open-set identification / retrieval metrics and larger-gallery experiments, not only closed-set top-1 linkage.",
    "A privacy-utility frontier showing whether stronger privacy settings actually trade off utility or simply fail on both axes.",
    "Demographic-disaggregated leakage analysis for linkage, regeneration, and soft-biometrics.",
    "Failure-case analysis and visualizations showing when regeneration or linkage breaks under pose, age, occlusion, or domain shift.",
    "Frequency/channel ablations to reveal which parts of the protected template retain most identity information."
  ],
  "novel_insights": "The paper's most valuable insight is not the specific network architecture but the reframing of what privacy should mean in PPFR. It argues, and empirically supports, that the relevant privacy question is identity recoverability/linkability, not whether the original enrollment image can be reconstructed pixel-by-pixel. Once protected templates preserve enough signal for face-matching utility, a simple distillation into a public face-embedding space can recover that signal well enough for both linkage and regeneration. In other words, 'unreconstructable under PSNR/SSIM' can still mean 'highly deanonymizable.' That is a useful shift in evaluation perspective.",
  "missed_related_work": [
    "Template inversion / face-from-template recovery literature, especially GaFaR (ICCV 2023) and related 2023-2024 works by Shahreza and Marcel on reconstructing or synthesizing faces from facial templates/embeddings.",
    "Additional template inversion evidence such as 'Vulnerability of State-of-the-Art Face Recognition Models to Template Inversion Attack' (IEEE TIFS 2024), which could help position the paper relative to broader face-template leakage results.",
    "Differential-privacy-based PPFR approaches such as PEEP (Computers & Security 2020), as a possible contrast to transformation-only methods and to motivate discussion of stronger privacy guarantees."
  ],
  "suggestions": [
    "Tighten the scope of the main claim: present the paper as a strong result on the evaluated frequency-domain PPFR family, with transfer to TIP-IM/CanFG framed explicitly as preliminary evidence rather than a settled general statement.",
    "Clarify all headline metrics in the abstract and main tables, explicitly distinguishing 1-to-1 verification, 1-to-N linkage, per-image pass rate, and Success@5.",
    "Add stronger attack-side baselines and ablations: direct teacher-on-template, linear probe / shallow MLP, template-space kNN, different teacher backbones, and data-scaling curves for all attacked methods.",
    "Provide fuller implementation details for reproducibility, especially optimizer/hyperparameters, preprocessing, template normalization, and the exact protocol for the constrained high-pass proxy attack.",
    "Reduce dependence on proprietary verification services by adding a fully reproducible open-model evaluation protocol, or at minimum provide more calibration detail and uncertainty estimates.",
    "Control for possible CASIA/LFW identity overlap or re-run the cross-dataset evaluation with explicit overlap removal.",
    "Include some robustness analysis across random seeds and report confidence intervals or equivalent uncertainty estimates.",
    "Deepen the analysis of why the attack works, for example via frequency/channel ablations, representation-alignment analysis, or cross-backbone transfer experiments."
  ]
}

========================================================================
PREDICTED SCORE
========================================================================

Score: 5.9
Decision: Accept
