Now I have a thorough understanding of the paper. Let me write the final review.

## Summary

The paper documents racial and gender biases in SDXL across six races, two genders, 32 professions, and eight attributes at unprecedented scale (320K profession images, 80K attribute images). It introduces the novel concept of "racial homogenization" — the tendency of T2I models to depict all members of a race as visually similar — and provides a quantitative operationalization using pairwise cosine similarity of face embeddings. The paper proposes two debiasing solutions (SDXL-Inc via LoRA routing and SDXL-Div via fine-tuning on FFHQ) and conducts four preregistered RCTs showing that exposure to inclusive AI-generated images reduces human biases while non-inclusive images increase them.

## Strengths

- **Novel concept of racial homogenization and rigorous quantification (Section 4.5):** The idea that T2I models depict same-race individuals as overly similar (e.g., Middle Eastern men with mean cosine similarity 0.61, reduced to 0.41 by SDXL-Div) is genuinely novel and addresses a form of representational harm that prior work overlooked. The ~50 million pairwise comparisons per race provide a robust measurement.

- **Large-scale bias audit comprehensiveness:** The 320,000 profession images, 80,000 attribute images, and LAION-5B comparison across six races go well beyond prior work (Bianchi et al. examined 3 races/10 professions; Wang et al. examined 2 races/8 professions). Finding that "Terrorist" generates 100% Middle Eastern individuals and that Security Guard/Cleaner generate >50% Black individuals are striking and socially important findings.

- **Preregistered RCTs (Study 3–4 on homogenization):** The studies measuring whether homogenized AI images affect perceptions of beard prevalence (Study 3) and headcover prevalence (Study 4) for Middle Eastern individuals test a novel and specific causal hypothesis. Results showing p < 0.0001 in multiple conditions are compelling.

- **Training data comparison isolates model-level bias amplification (Figure 1):** The comparison of LAION-5B subsample demographics vs. SDXL output (e.g., LAION has ~equal male/female, SDXL generates 65% male) demonstrates that biases are not merely inherited from training data but amplified by the model.

## Weaknesses

### Fatal

None.

### Major

- **SDXL-Inc's "generalization" claim is misleading — the debiasing mechanism is demographic routing, not learned debiasing:** SDXL-Inc trains 12 LoRA adapters (one per race×gender combination) and randomly selects one at inference time. This enforces demographic balance by construction through quota-based routing. The paper's claim that SDXL-Inc "can be generalized beyond the features it was fine-tuned on" (Section 4.4) obscures this: any prompt will produce balanced demographics because the *routing* enforces it, not because the model learned debiasing. The 21/11 profession split and the attribute generalization test are therefore not tests of debiasing generalization — they merely confirm that the routing mechanism works regardless of prompt. What would constitute genuine generalization would be testing whether each LoRA adapter produces quality, profession-appropriate images for unseen prompts, which is never evaluated. The GPT-in-the-loop alternative (which simply injects demographic terms into prompts) further underscores that the core mechanism is demographic routing, suggesting a prompt-engineering baseline could achieve similar results without any fine-tuning. This does not invalidate the finding that demographic routing *works*, but the paper's framing as a "debiasing solution" that "generalizes" overstates what has been demonstrated.

- **User studies demonstrate anchoring effects, not AI-specific bias influence:** Studies 1–2 show participants viewing 6 all-White or all-male images estimate higher White/male percentages in real-world professions than a no-image baseline. This is fully consistent with a basic anchoring/priming effect: presenting 6 images of a specific demographic before asking for demographic estimates will anchor responses regardless of whether the images are AI-generated, photographed, or hand-drawn. The paper's manipulation — labeling images as "AI-generated" vs. "produced by an artist" — tests source attribution, not whether the AI-generation aspect matters. Without a control condition using real photographs with matched demographics, the paper cannot support its core claim that "AI-generated faces influence gender stereotypes" specifically. The abstract's framing ("This persists regardless of whether the images are labeled as AI-generated") further conflates the source-label manipulation with the AI-generation question. Studies 3–4 (on homogenization) are somewhat stronger because they test whether homogenized vs. diverse images shift perceptions of specific features (beards, headcovers), but these too lack a real-photograph control.

- **Classifier circularity and inadequate validation on neutral-prompt synthetic images:** The same classifier pipeline (MTCNN→VGGFace→SVM) is used to (a) identify SDXL biases, (b) generate the fine-tuning data for SDXL-Inc (by labeling SDXL outputs with explicit race/gender prompts), and (c) evaluate whether SDXL-Inc reduces biases. This creates structural circularity: the classifier defines what "balanced" means, the solution is built to satisfy those categories, and the same classifier confirms success. While classifier accuracy is reported on FairFace (natural images) and on SDXL images with *explicit* demographic prompts ("a photo of a [race]"), the classifier is applied to images generated from *neutral* prompts ("a photo of a person," "a photo of a [profession]") in the main bias analysis. These images may have different visual properties (e.g., more ambiguous racial features) than either FairFace or the explicit-prompt images. The paper does not report per-class accuracy or confusion matrices for this critical distribution. Given that many race-profession combinations show 0.00% (e.g., all non-White/Black races are 0.00% for most professions in Table 1), systematic misclassification could materially affect the findings.

### Minor

- **LAION-5B comparison has confounds (Section 4.2):** The subsample was filtered by keywords ("face, person, child, woman, or man") and by image quality/resolution. These filters could themselves introduce systematic biases, making the comparison with SDXL suggestive but not definitive as a claim that "SDXL contains biases that cannot be fully explained by the data."

- **User study design transparency (Section 4.6):** The power analysis references "paired-sample comparison" but the design is between-subjects. Per-condition cell sizes are not clearly stated in the main text (135 total per study, split across 4 conditions = ~34 per cell). The single-item percentage estimation measure is susceptible to demand characteristics given the transparent manipulation (viewing 6 skewed images then immediately estimating demographics).

- **Cosine similarity as a proxy for perceived homogenization (Section 4.5):** VGGFace embeddings may not capture the features humans find most salient for perceived homogenization (e.g., headdress, beard — these are semantic features). The paper treats the quantitative measure and the user-study measure as interchangeable when they measure different constructs.

- **Uniform distribution as the default for GPT-in-the-loop race selection (Section 3.2.3):** Random selection assumes a uniform distribution over six racial categories, which is itself a normative choice not discussed. In practice, real-world distributions differ.

### Trivial

- None.

## Nice-to-Haves

- A simple prompt-injection baseline (randomly prepending demographic terms to prompts) would clarify what SDXL-Inc's LoRA routing actually contributes beyond routing.
- Per-class classifier metrics on neutral-prompt images would strengthen confidence in the bias audit.
- A real-photograph control in the user studies would separate anchoring effects from AI-specific effects.
- Evaluation of image quality and prompt fidelity for SDXL-Inc/SDXL-Div outputs would ensure debiasing does not come at unacceptable quality costs.

## Removed Points

- **Harsh Critic's claim that "none of the prior studies proposed debiasing solutions" is inaccurate (Introduction):** The harsh critic noted the paper states no prior studies "proposed debiasing solutions" while Friedrich et al. (2023) and Zhang et al. (2023a) are both discussed as debiasing proposals. However, reading the actual text (p. 2, lines 21–22), the paper states "none of these studies proposed debiasing solutions" referring specifically to the bias-audit papers (Bianchi et al., Wang et al., Ghosh & Caliskan), then separately discusses Friedrich and Zhang as having proposed debiasing but with limitations. The claim is structured to distinguish the audit papers from the debiasing papers, and while the phrasing could be clearer, it is not outright inaccurate.

- **Harsh Critic's demand for per-class classifier accuracy on the exact distribution:** While per-class metrics would strengthen the paper, the classifier is validated on FairFace (natural images) and on SDXL with explicit prompts. The FairFace validation set includes per-class sample sizes. The concern about classifier performance on neutral-prompt images is legitimate but is a minor rather than fatal gap — the extreme bias patterns (e.g., 0% Middle Eastern for most professions) are consistent with qualitative inspection and are unlikely to be entirely artifacts of classifier failure.

- **Formatting/style nitpicks from the parser output:** Removed as per instructions.

- **Claiming the 0.00% values in Table 1 are parser artifacts:** These appear to be real data values (e.g., 87.53% White for Dietitian and 0.00% for all other races), not artifacts. While they could potentially reflect classifier limitations, they are consistent with the extreme bias patterns the paper documents.

- **Demand for reproducibility details like training logs:** Removed per instructions — these are impractical to include.

## Novel Insights

The introduction of "racial homogenization" as a distinct form of representational harm in T2I models is the paper's most important conceptual contribution. Prior work focused exclusively on under/over-representation; the idea that even when a race *is* represented, all members may be depicted as visually identical captures a qualitatively different harm. The finding that Middle Eastern faces have a mean cosine similarity of 0.61 (compared to lower values for other races) quantifies this for the first time. The user studies (Studies 3–4) provide initial causal evidence that homogenized representations shift human perceptions of specific demographic features, which is an important new direction.

## Suggestions

- Reframe SDXL-Inc honestly: describe it as "demographic routing via LoRA adapters" rather than "learned debiasing." Clearly state that the generalization test demonstrates routing works for unseen prompts, not that the model learned debiasing.
- Add a simple prompt-injection baseline (random demographic term injection) to isolate what the LoRA fine-tuning contributes beyond routing.
- Narrow the user-study claims to what was demonstrated: "exposure to demographically skewed AI-generated images shifts demographic estimates" rather than "AI-generated faces influence biases." Acknowledge the absence of a real-photograph control as a limitation.
- Report per-class classifier accuracy on neutral-prompt SDXL images in an appendix to address the distribution-shift concern.

## Score and Decision

**Calibration anchors:**

- **High (avg > 7):** OASIS (7.20, Spotlight) — T2I bias audit with sociological metrics, comprehensive but no human studies; CEB (7.50, Spotlight) — compositional fairness benchmark for LLMs; MMDT (7.00, Poster) — multimodal fairness/safety benchmark with human studies. This paper has stronger contributions (novel homogenization concept + RCTs) but also deeper methodological concerns.

- **Medium (avg 4–6):** DebiasDiff (4.67, Withdrawn) — debiasing T2I diffusion with attribute latent directions, criticized as "simple routing in latent space"; Balancing the Picture (4.67, Reject) — synthetic contrast sets for debiasing VLMs, limited to single attribute; GRADE (5.33, Reject) — measuring attribute diversity in T2I generation using VQA classifiers. This paper's contributions are stronger than these but share some weaknesses around methodological novelty.

- **Low (avg < 3):** Fair Image Generation from PCs (2.33, Withdrawn) — simple probabilistic circuit approach for debiasing, limited validation; SimpleStrat (3.67, Reject) — simple stratified sampling for LLM diversity, criticized as lacking novelty.

This paper's core contributions — the homogenization concept and measurement, the large-scale bias audit with LAION comparison, and the preregistered RCTs — are substantial and would place it above medium-scoring anchors. However, the misleading framing of SDXL-Inc's "generalization" (which is really routing) and the absence of real-photograph controls in the user studies are significant issues that prevent it from reaching the level of the high-scoring anchors, which had cleaner methodology. The OASIS paper (7.20) is the closest comparison — also a T2I bias audit but with more novel metrics and fewer methodological concerns, though without human behavioral studies. This paper adds human studies (a strength) but with the anchoring confound (a weakness). Overall, this is a solid contribution with real novelty in the homogenization concept and valuable empirical findings, but with overclaimed scope on the debiasing and human-influence dimensions.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>