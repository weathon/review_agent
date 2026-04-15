Now let me search for calibration papers to determine the appropriate score.Now I have enough information to write the final consolidated review. Let me verify one more key claim about the λ_oc=0 ablation.Now I have everything needed to write the final consolidated review.

---

## Summary

OCEBO proposes a self-distillation bootstrapping framework for training object-centric (slot attention-based) models from scratch on real-world images, eliminating the dependency on frozen, non-object-centric pretrained encoders (e.g., DINOv2) that impose a performance ceiling. The key contributions are (1) an EMA target encoder updated from the object-centric student to inject instance-level inductive biases into the teacher, and (2) a cross-view patch filtering mechanism that gates supervision to patches with consistent cross-view correspondences, preventing slot collapse from the cold-start random initialization. Trained on ~241k COCO images, OCEBO achieves FG-ARI scores comparable to DINOv2-target methods pretrained on 142M images, establishing for the first time that object-centric pretraining from scratch on real-world data is feasible.

---

## Strengths

- **Principled, novel solution to a real bottleneck.** The frozen-target ceiling in object-centric learning is a well-documented problem (Didolkar et al., 2024); OCEBO's EMA bootstrapping approach provides a conceptually clean solution with a clear mechanistic motivation (semantic drift → object-centric drift in the target encoder).
- **Cross-view patch filtering is a genuinely critical and novel component.** Table 1(a) demonstrates that omitting the filtering immediately causes slot collapse, and Figure 2 provides intuitive support that ~90% of patches are uninformative in early training. This is not a cosmetic contribution.
- **The λ_oc=0 ablation is a strong baseline.** Section 4.2 explicitly notes that setting λ_oc=0 "reduces OCEBO to pretraining of a DINO model on COCO followed by FT-DINOSAUR fine-tuning with a frozen COCO-pretrained DINO target encoder" — and this baseline collapses. This directly validates the necessity of the object-centric training objective and preempts the most obvious confound.
- **Competitive FG-ARI despite much less pretraining data.** Table 2 shows OCEBO (COCO+) achieving MOVi-E FG-ARI of 66.8 and EntitySeg FG-ARI of 44.2 — comparable to DINOSAUR (71.1 / 43.5) and FT-DINOSAUR (71.1 / 48.1) trained with DINOv2 pretrained on 142M images. Given the orders-of-magnitude data gap, this is a meaningful empirical result.
- **Honest evaluation design.** The paper argues convincingly for zero-shot unsupervised object discovery over in-distribution evaluation, and the limitations section is refreshingly concrete about dataset suitability constraints.
- **Quantitative collapse metric.** The d metric (content vs. positional similarity across views) is a useful diagnostic tool for the community, even if informally validated.

---

## Weaknesses

### Fatal
*None. The paper makes a real contribution and its core claim — that EMA bootstrapping with object-centric inductive biases enables training from scratch on real data without collapse — is supported by adequate ablations.*

### Major

- **Substantial mBO gap vs. state-of-the-art.** On EntitySeg, OCEBO achieves 16.0 mBO vs. FT-DINOSAUR's 28.4 — a 12.4-point gap. On MOVi-E, 22.1 vs. 29.9. While the authors correctly attribute this partly to decoder differences (MLP vs. top-k MLP) and the absence of a high-resolution training stage, this is a confound the paper chose not to control. Given that the core scientific claim involves the *training scheme*, not the decoder, it is a meaningful gap. The FG-ARI–mBO trade-off discussion in Section 4.3 is plausible but is asserted rather than demonstrated to be intrinsic vs. architecture-specific. This limits the strength of comparative conclusions.

- **Mask sharpening stage raises questions about self-distillation objective completeness.** Section 3.4 introduces a 100-epoch sharpening stage that switches to a frozen target encoder and ℓ₂ loss — essentially re-using the DINOSAUR recipe. Table 1(c) shows this adds ~10 FG-ARI points on MOVi-E (44.0→54.8). The paper attributes boundary fuzziness to a "constantly changing target encoder" but does not analyze *why* this occurs or whether it is fundamental to the approach or addressable within the self-distillation framework. This feels like a pragmatic workaround rather than an understood design choice. Additionally, the final performance of OCEBO is then partly dependent on the DINOSAUR-style stage, making the attribution of gains ambiguous.

### Minor

- **Scalability evidence is thin (only two data points).** The claim that OCEBO "removes the performance upper bound" and "enables large-scale pretraining" rests primarily on one comparison: COCO (~118k) vs. COCO+ (~241k). This 2x increase shows some benefit in FG-ARI but a slight regression in mBO on MOVi-E (25.8→22.1). The paper provides no intermediate data points (e.g., 32k, 64k images) that would reveal a trend. At minimum, this supports "does not immediately plateau between 118k and 241k" — a much weaker claim than "removes the upper bound." The appendix reportedly contains a more detailed scaling plot, but this does not substitute for a systematic analysis in the main paper.

- **"Large-scale pretraining" is overstated relative to the actual scale.** 241k images is modest even by 2022 SSL standards. The paper itself admits in the conclusion that "the scale of dataset used for pretraining" is a notable limitation and that a suitable large-scale dataset remains an open question. The headline framing should be adjusted to match this acknowledged limitation. "First pretraining from scratch" is an accurate and strong claim; "large-scale pretraining" is not yet realized.

- **The mechanism claim (object-centric inductive biases in the EMA target) is partially interpretive.** Figure 3 (PCA visualizations) is qualitative and cannot establish the causal claim that EMA updates inject instance-level representations into the teacher. The λ_oc=0 ablation is the strongest evidence, but it changes the entire training objective, not just the inductive bias of the teacher. This is a reasonable methodological gap for an empirical paper, but the causal claims in the text are somewhat ahead of the evidence.

- **Selective reporting of 4 out of 7 evaluation datasets in the main paper.** The paper states conclusions on the remaining three are "identical." This may be true, but selective reporting — especially when the paper argues for a more comprehensive zero-shot protocol — reduces confidence. The appendix results should at minimum be highlighted more clearly.

### Trivial

- The global loss ablation is stated to be "necessary for training stability" (Section 3.2) but no ablation isolates this component; the claim is unsupported by any table entry.
- The k hyperparameter in the cross-view patch filtering mask (Eq. 7) is never ablated or numerically reported in the main text.

---

## Nice-to-Haves

- **Downstream task evaluation** (robotic control, compositional generation, VQA — applications listed in the introduction) to demonstrate that the learned object-centric representations transfer beyond the unsupervised discovery metric.
- **Controlled decoder comparison.** Running OCEBO with FT-DINOSAUR's top-k MLP decoder and high-resolution stage (or vice versa) would isolate the effect of the training scheme from decoder architecture, resolving the mBO gap ambiguity.
- **Computational cost comparison.** Training 300 + 100 epochs from scratch vs. fine-tuning a pretrained encoder is a meaningful trade-off not quantified in the paper.
- **Segmentation mask visualizations** alongside DINOSAUR/FT-DINOSAUR on the same images, to complement the PCA visualizations in Figure 3 with direct segmentation comparisons.
- **Quantitative validation of the collapse metric d** against human judgment or alternative metrics to establish interpretability thresholds.
- **Attempt pretraining on a larger, more suitable dataset** (e.g., Open Images with multi-object scenes); even a negative or limited result would be informative.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Spark: "DINO-on-COCO baseline missing."** This is a strawman. Table 1(b) (λ_oc=0) is *explicitly described* in Section 4.2 as "pretraining of a DINO model on COCO followed by FT-DINOSAUR fine-tuning with the frozen COCO-pretrained DINO target encoder." The baseline exists and leads to collapse, directly validating the authors' claim. Removed.

- **Harsh Critic: Rhetoric/formatting nitpicks ("for the first time ever," "paves a way").** While the scalability framing is partially overstated (addressed as a Major weakness), removing specific rhetorical phrases is a style nitpick. Removed.

- **Harsh Critic: Notation inconsistency in Eqs. 1–4.** The harsh critic explicitly notes this may be parser artifacts, and the paper's logic is coherent. Removed.

- **Harsh Critic: Reproducibility concerns (undisclosed hyperparameters, training logs).** The paper reports all key hyperparameters (EMA momentum schedule, temperatures, learning rate, λ values, k in patch filtering, slot count, epochs) in Section 4.1. Code/model release is promised. Removed.

- **Harsh Critic: No confidence intervals / multiple seeds.** Single-run evaluation is the norm in object-centric learning SSL benchmarks. Removed (moved to nice-to-have tier, not implemented here since even this is standard practice in the field).

- **Human Finder: Fixed slot number constraint.** The paper trains with s=7 slots but evaluates with per-dataset slot counts (11, 24, 7, 7). The fixed training count is a design choice common to all slot-based methods and is not specific to OCEBO. Removed.

---

## Novel Insights

The most genuinely novel insight in this paper — beyond its technical contribution — is the *mechanism* characterization for why EMA updating fails in prior object-centric methods: standard SSL features organize representations semantically, but slot attention's inductive bias reorganizes them toward instance boundaries, causing the student and teacher to drift apart in a reinforcing cycle. This "semantic-to-positional drift" hypothesis is compelling and distinct from what prior work has articulated. The paper also makes an interesting empirical observation that a curated multi-object dataset (COCO) outperforms a much larger but object-poor dataset (ImageNet) for object-centric pretraining — pointing toward the need for data curation pipelines as a future research direction, analogous to early curated dataset efforts in other pretraining paradigms. These observations are likely to shape the field's approach to object-centric pretraining for some time.

---

## Suggestions

1. **Add a decoder-controlled comparison** in a single table entry (e.g., OCEBO + top-k MLP decoder) to isolate mBO gap sources definitively.
2. **Provide the full 7-dataset table in the appendix and reference it prominently** in Section 4.1, so readers can assess breadth of generalization.
3. **Add ablation of the global loss** (Table 1, row e) to support the stability claim, and of the k hyperparameter in Eq. 7.
4. **Revise "large-scale pretraining" framing** to "object-centric pretraining from scratch" throughout, reserving the former claim for when a genuinely large-scale dataset is available.
5. **Analyze the sharpening stage more carefully** — e.g., how much benefit comes from the frozen OCEBO target vs. the switch to ℓ₂ — to understand whether the fuzzy boundary issue is structural or correctable within the self-distillation framework.

---

## Score and Decision

**Calibration:**
- `kZvor5aaz7` (SlotAdapt — novel object-centric method, strong results): Scores 8, 6, 5, 6 → Accepted (Poster). SlotAdapt also leverages pretrained diffusion models and has broader generation evaluation. OCEBO is arguably more fundamental in its contribution but has a more significant results gap.
- `bSq0XGS3kW` (Object-centric transfer benchmark + FT-DINOSAUR baseline): Scores 6, 5, 3, 6 → Accepted (Poster). Primarily an analysis paper with incremental method contribution; OCEBO's contribution is more technically novel and addresses a deeper bottleneck.
- `EaLfdBPlIh` (Adaptive Slot Attention): Scores 5, 5, 5, 5 → Rejected. Addresses a real limitation but with less rigorous experiments; OCEBO's ablations are stronger.
- `aBUidW4Nkd` (Slot Mixture Module): Scores 5, 8, 6, 6 → Accepted (Poster). Incremental improvement to slot design; OCEBO's contribution is more foundational.

OCEBO sits above `bSq0XGS3kW` in novelty and technical depth, comparable to `kZvor5aaz7` in poster-level contribution, and well above `EaLfdBPlIh`. The major mBO gap (addressed by the sharpening stage and decoder choice but not fully controlled) and the thin scaling evidence prevent it from reaching the 7+ range of a spotlight-level paper. The fundamental contribution — first successful object-centric pretraining from scratch on real-world data — is solid and reproducible via the ablations. A score of **6.0** (borderline accept) is appropriate: the paper makes a genuine contribution to an important problem, but the strongest claims outpace the evidence and several experiments needed to close the loop are missing.

**Originality:** High — first to solve the EMA bootstrapping failure in object-centric learning.
**Importance:** High — removes a fundamental bottleneck that has blocked the field.
**Support for claims:** Moderate — core feasibility claim is well-supported; scalability and mechanism claims are underdeveloped.
**Soundness:** Moderate — ablations are solid; mBO results and sharpening stage raise questions.
**Clarity:** Good — paper is well-written and transparent about limitations.
**Value to community:** High — opens a new direction for object-centric pretraining.

**Decision: Accept (Poster)**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>