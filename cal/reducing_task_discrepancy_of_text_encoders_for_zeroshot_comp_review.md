=== CALIBRATION EXAMPLE 30 ===

# Final Consolidated Review
## Summary
RTD proposes a post-processing approach for projection-based Zero-Shot Composed Image Retrieval (ZS-CIR) that fine-tunes only the CLIP text encoder using cheap, automatically generated text triplets (T_r, T_c, T_t) via a target-anchored text contrastive loss. The core insight is that frozen CLIP text encoders suffer from a task discrepancy: they were pre-trained on image-text alignment but must process compositionally modified prompts ("a photo of [S] that [T_c]") at inference time. RTD integrates with existing projection-based ZS-CIR methods (Pic2Word, SEARLE, LinCIR), achieves consistent improvements across five benchmarks and multiple backbones, and does so with substantially lower computational cost than competing approaches.

---

## Strengths

- **Pinpointed and empirically validated problem identification.** Table 1 provides a clean quantitative demonstration of the task discrepancy: a frozen text encoder retrieves with mAP@5 = 10.12 using T_{r+c}, vs. 18.96 using T_t, while RTD closes the gap to 15.12. The additional cosine similarity measurement (0.10 → 0.29 for composed query vs. target image) independently corroborates the finding. Few recent ZS-CIR papers provide such direct evidence for the problem they claim to solve.

- **The target-anchored design is principled and empirically critical, not incidental.** Table 8 shows that naive fine-tuning of the text encoder with the same base loss catastrophically degrades Pic2Word from R@10 = 64.43 to 27.51, while RTD's anchor mechanism raises it to 69.77. This dramatic contrast demonstrates that the anchor is a strict requirement, not a heuristic improvement, and justifies the specific method design.

- **Breadth and consistency of improvements.** RTD improves every base method (Pic2Word, SEARLE, LinCIR) × every CLIP backbone tested (ViT-B/32, ViT-L/14, ViT-G/14) × every benchmark (CIRR, CIRCO, FashionIQ, COCO, GeneCIS). Minimum average improvement exceeds 2 R@10 points in all configurations. This breadth is strong evidence that RTD addresses a structural limitation rather than overfitting to a specific setting.

- **Rule-based triplets perform comparably to LLM-generated ones.** The finding in Table 7 that rule-based triplets (+2.85 avg) achieve near-parity with expensive LLM-generated CompoDiff triplets (+3.34 avg) is practically significant and non-obvious. It implies that the key driver of improvement is the training format/objective, not the semantic richness of the triplets themselves—a meaningful mechanistic claim.

- **Efficient scalability to ViT-G.** Because training is text-only (no image encoder forward passes), RTD scales to LinCIR ViT-G/14 with minimal overhead and achieves state-of-the-art performance among CLIP-based ZS-CIR methods, which would be infeasible for image-triplet-based approaches at this backbone scale.

---

## Weaknesses

### Fatal
None.

### Major

- **The alignment-preservation claim is unverified on standard benchmarks.** Section 3.2 states that fixing the target textual embedding to the frozen encoder "helps maintain the pre-trained alignment," and Section 3.3 shows the cosine similarity between composed queries and target images improves. However, there is no direct evaluation of whether the updated text encoder preserves standard zero-shot image-text retrieval performance (e.g., on COCO or Flickr30k). Given that naive tuning completely destroys CLIP alignment (Table 8), this is not a trivial concern. Without this verification, the claim that "pre-trained alignment is maintained" remains partially unsubstantiated and is a meaningful gap for a method that modifies a general-purpose CLIP text encoder.

- **Table 6 ablation contains a suspected formatting error that makes two key rows uninterpretable.** Rows 3 and 4 appear identical in the displayed columns (TCL text pair = (T_{r+e}, T_t), Anchor = ✓, RB = ✓, RC = ×) yet show different results (avg 39.17 vs. 39.64). The paper text says row 3 isolates the effect of using generated triplets and row 4 adds refined batch sampling (RB), meaning row 3 should have RB = ✗. If row 3's RB column is incorrectly marked as ✓, the contribution of RB cannot be correctly read from the ablation table. This should be corrected.

### Minor

- **The noise injection approximation for the modality gap lacks distributional justification.** The refined concatenation scheme injects isotropic Gaussian noise into the textual embedding before passing it through φ to simulate inference-time behavior (where φ receives an image embedding). The paper acknowledges this is an approximation (Appendix B.6), but the systematic directional nature of the modality gap (text and image embeddings occupy different cones in embedding space) means isotropic noise only partially captures the gap in expectation and not per-sample. An empirical check showing that the distribution of φ(t_r + noise) actually resembles the distribution of φ(v_r) for matched pairs would meaningfully justify this design choice.

- **The +0.31 performance gain with CASE triplets is attributed to "poor quality" without characterization of what quality means.** This is the one setting where RTD barely improves. The paper dismisses it with reference to Table A.1, but the failure is informative: it suggests RTD is sensitive to some property of triplet quality. A brief qualitative or quantitative analysis (e.g., conditioning text diversity, T_t specificity) would help practitioners know what makes triplets suitable for RTD and whether other triplet sources could similarly fail.

- **Catastrophic collapse under naive tuning is unexplained mechanistically.** The naively tuned Pic2Word degrades from R@10 = 64.43 to 27.51 (Table 8), an extreme drop that deserves more than a note about "misalignment." A brief analysis—e.g., showing how standard image-text cosine similarity changes under naive vs. anchored tuning—would clarify why the anchor mechanism is strictly necessary and strengthen the paper's methodological narrative.

- **Whether improvements reflect genuine compositional reasoning or improved keyword matching is not analyzed.** The target-anchored loss trains the model to match T_{r+c} embeddings to T_t embeddings. Since T_t often contains the keywords of T_r modified by T_c, it is unclear whether the encoder learns to process the modification instruction (T_c) compositionally or simply learns to better match the keywords appearing in T_t. An analysis probing whether T_c tokens contribute meaningfully to the composed query embedding would substantiate the "task discrepancy reduction" interpretation.

### Tiny

- **Notation inconsistency:** In Eq. 1 and surrounding text, the target caption embedding uses subscript $t_i$ (where $i$ is the batch index), while the target caption is denoted $T_t$ everywhere else. This overlap between index and name subscripts is slightly confusing and could be avoided with cleaner notation (e.g., $t_i^+$ for the positive target or using consistent naming).

- **GeneCIS and COCO results are relegated entirely to the appendix** despite being listed in the introduction as part of the paper's empirical scope. Summary numbers in the main body would complete the empirical narrative.

---

## Nice-to-Haves

- **t-SNE/PCA embedding visualization** of T_{r+c}, T_t, and corresponding image embeddings before and after RTD fine-tuning, to visually demonstrate the claimed discrepancy reduction.
- **Full wall-clock cost breakdown** including triplet generation time alongside training time in the main paper, not just appendix, to fully validate the efficiency claims.
- **Layer-wise fine-tuning analysis** (Appendix B.3) promoted to the main paper: knowing which layers drive the improvement vs. those that preserve alignment would illuminate the mechanism and support more efficient deployment.
- **Comparison with PEFT alternatives** (LoRA, prefix tuning) using the same objective, to determine whether the improvements are due to the training objective or the full-parameter update regime.
- **Temperature sensitivity analysis** for τ = 0.07, which is used directly as the CLIP default. Even a brief confirmation that nearby values perform similarly would increase confidence.
- **Cross-domain triplet generalization:** training on COCO-domain triplets and evaluating on fashion/scene-change benchmarks, to assess how much distributional overlap between triplet source and test domain matters.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Statistical significance testing / confidence intervals (Critic, W4):** Single-run evaluation is the norm for large-scale CIR benchmarks. Demanding confidence intervals imposes a non-standard requirement for this field. Removed.

- **Comparison fairness with CoVR/CASE using BLIP backbone (Critic, E1):** The paper explicitly states this comparison is not entirely fair and tables note the architecture difference. The comparison is offered as context, not equivalence. The unfairness is asymmetric in the baseline's favor (larger pre-training data, different architecture), which actually makes the comparison conservative for RTD's claims. Removed per policy.

- **CIReVL not sufficiently credited (Critic, Intro2):** The paper explicitly discusses CIReVL in the related work section (Section 2), contrasting its inference-time captioning against RTD's training-time fine-tuning. The structural similarity is acknowledged. This criticism misreads the paper. Removed.

- **Human preference evaluation (Spark Finder, Next Steps 1):** Human evaluation is not standard practice for CIRR/CIRCO/FashionIQ benchmarks, which have established automatic metrics. This is not a weakness within the field's norms. Removed.

- **Missing related works:** Not assessed, as external references cannot be independently verified. Removed per policy.

- **Claim that RTD being a "post-processing" method and needing φ is an unacknowledged limitation (Critic, Limitations 1):** The paper's entire framing is as a plug-in post-processor for existing methods. That it requires a pre-trained φ is a design property, not a hidden limitation—it is stated in the abstract and introduction. Removed.

- **Missing ViT-B/32 LinCIR in Table 2 (Critic, E3):** Likely because LinCIR was not trained at ViT-B/32 with public weights and the ViT-B/32 LinCIR results are already present in Table 3. A minor asymmetry, not a meaningful weakness. Removed.

---

## Novel Insights

The most substantive novel observation emerging from these reviews is the stark empirical asymmetry between naive text encoder tuning and anchored tuning: the catastrophic collapse of Pic2Word under naive fine-tuning (R@10: 64.43 → 27.51) versus its substantial improvement under RTD (64.43 → 69.77) demonstrates that the target-anchored design is not merely a regularization nicety but a structural requirement for this setting—any update to a shared text encoder that moves embeddings away from the pre-trained CLIP manifold severs the text-image alignment that the frozen image encoder and retrieval database depend on. The implication is that the projection-based ZS-CIR pipeline creates a brittle shared embedding space where even well-intentioned fine-tuning of one modality can destroy retrieval if not carefully anchored. This insight generalizes beyond RTD: any future method that updates the text encoder within projection-based ZS-CIR must contend with this fragility and should design objectives that explicitly preserve the target text embedding distribution.

---

## Suggestions

1. **Verify alignment preservation on standard benchmarks.** Add zero-shot COCO or Flickr30k image-text retrieval results before and after RTD fine-tuning. Even a single number showing negligible degradation would close the most substantive open question about the method.

2. **Fix the Table 6 ablation formatting.** Confirm whether row 3 is intended to have RB = ✗ (as implied by the text) and correct the table accordingly so the RB contribution can be read directly.

3. **Characterize triplet quality for the CASE failure case.** Add a short analysis in Section 4.5 quantifying what distinguishes CASE triplets from CompoDiff or CoVR triplets (e.g., T_c specificity, T_t descriptiveness), giving practitioners actionable guidance on what kinds of text triplets are suitable for RTD.

4. **Provide a mechanistic account of naive tuning collapse.** A brief additional experiment—e.g., plotting image-text cosine similarity on a held-out set as a function of training steps under naive vs. anchored tuning—would explain the collapse and reinforce the necessity of RTD's specific design.

5. **Add a compositionality probe.** For a sample of queries, ablate T_c (e.g., replace with a null token) and measure the change in retrieval rank to empirically distinguish between compositional reasoning and keyword matching as the driver of improvement.

---

**Evaluation across axes:**

- **Novelty:** Moderate-to-good. The specific diagnosis of text encoder task discrepancy in ZS-CIR is original within the field; the contrastive fine-tuning framework is established but the anchoring mechanism and text-only formulation are non-trivial adaptations.
- **Technical soundness:** Good. The method is principled, ablation studies cover each component, and the anchor design is empirically justified. The noise injection approximation is the weakest technical element.
- **Empirical support:** Strong. Five benchmarks, three base methods, multiple backbones, consistent positive results throughout. The breadth of evaluation is above average for this sub-field.
- **Significance:** High from a practical standpoint. The method converts an expensive image-triplet training problem into a cheap text-only one, with consistent gains and no inference overhead. The scalability to ViT-G is a concrete practical benefit.
- **Clarity:** Good overall, with the ablation table formatting error and minor notation inconsistencies as the main clarity issues.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 3.0, 3.0, 5.0]
Average score: 3.4
Binary outcome: Reject
