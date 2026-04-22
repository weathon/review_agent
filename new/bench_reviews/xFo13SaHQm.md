Now I have a good set of calibration anchors. Let me compile my findings and write the final review.

Let me verify one more thing — the user study figure labels issue. The harsh critic says "Cure," "iDetch," "Uniformal" don't match method names. Looking at the paper, Figure 8's caption says the methods are "Cure, UNO, iDetch, Uniformal, OmniGen." The paper text (line 302-304) confirms this. "Cure" = WithAnyone, "iDetch" = ID-Patch, "Uniformal" = UniPortrait — these appear to be abbreviated/nickname versions. This is confusing but not a "labeling error" per se. The paper does not provide a mapping table, which is a real presentation issue.

Let me also check the claim that multi-person evaluation is missing key baselines:

From Table 2, the multi-person baselines are: DreamID, OmniGen, OmniGen2, GPT, UNO, USO, UniPortrait, ID-Patch. The single-person baselines (Table 1) include: DreamO, OmniGen, OmniGen2, FLUX.1 Kontext, Qwen-Image-Edit, GPT-4o Native, UNO, USO, UMO, UniPortrait, ID-Patch, InfU, PuLID, InstantID.

Indeed, InstantID, PuLID, UMO, DreamO, and InfU are absent from Table 2. However, the paper explains on line 209: "only those supporting multi-ID generation were additionally tested on the multi-person subset." This is a reasonable explanation — some methods are single-ID only. But for multi-ID capable methods like PuLID and InstantID that are missing, the paper doesn't explain their absence. Actually, PuLID and InstantID are primarily single-ID methods; they may not natively support multi-ID generation. UMO is also mentioned as a multi-ID method but appears absent.

Now let me write the final review.

## Summary

The paper introduces WithAnyone, a diffusion-based model for controllable, identity-consistent image generation that addresses the "copy-paste artifact" — the tendency of models to replicate reference images rather than generating diverse appearances of the same identity. The paper contributes three components: (1) MultiID-2M, a large-scale dataset of ~500k paired multi-identity images with diverse references per identity; (2) MultiID-Bench, a benchmark with a novel Copy-Paste (CP) metric that quantifies reference-bias vs. ground-truth alignment; and (3) WithAnyone, a FLUX-based model trained with a GT-aligned ID loss, ID contrastive loss with extended negatives, and a four-phase training pipeline that transitions from reconstruction to paired tuning.

## Strengths

- **Novel formalization of the copy-paste artifact**: The paper identifies and operationalizes a real, previously under-examined failure mode in ID-consistent generation. Figure 2's density plot compellingly shows that existing models (InstantID, PuLID) concentrate at near-perfect face similarity while real photo pairs and WithAnyone show broader, more natural distributions. The CP metric (Eq. 2) provides a principled, normalized measure of reference-bias that addresses the flaw in Sim(Ref)-only evaluation.

- **MultiID-2M dataset fills a real gap**: The dataset of ~500k paired multi-ID images with hundreds of references per identity across ~3k identities (Section 3) directly addresses the data bottleneck that forces prior work into reconstruction-only training. No prior multi-ID dataset provides this scale of paired references.

- **Strong empirical evidence for trade-off breaking**: Figure 5 is the paper's most compelling evidence — scatter plots show all baseline methods lying approximately on a regression curve between Sim(GT) and CP, while WithAnyone clearly sits off the curve in the desired upper-right region. Table 1 quantifies this: WithAnyone achieves Sim(GT)=0.460 (competitive with InstantID's 0.464) while reducing CP from 0.337 to 0.144.

- **GT-aligned ID Loss is a technically sound contribution**: Using GT landmarks to align generated faces for ArcFace extraction (Section 5.1, Eq. 4) avoids unreliable landmark detection in noisy intermediates, enabling ID supervision at all noise levels. Figure 7 provides convincing evidence that GT-Align yields lower and more informative ID loss across noise levels, and Table 3 confirms removing it drops Sim(GT) from 0.405 to 0.385.

- **Phased training pipeline is well-motivated**: The four-phase training (Section 5.2) that transitions from reconstruction to paired tuning is cleanly designed. Table 3 shows Phase 3 reduces CP from 0.239 to 0.161 with nearly unchanged Sim(GT) (0.406→0.405), validating paired tuning as the key mechanism for suppressing copy-paste.

## Weaknesses

### Fatal
None.

### Major

- **Ablation data partially contradicts the narrative about contrastive loss**: Table 3 shows that removing extended negatives (w/o Ext. Neg.) *decreases* CP from 0.161 to 0.074 while also decreasing Sim(GT) from 0.405 to 0.368. This means the ID contrastive loss with extended negatives *increases* copy-paste artifacts — it operates squarely on the very trade-off curve the paper claims to break. The actual component that breaks the trade-off is Phase 3 (paired tuning), which reduces CP from 0.239→0.161 with virtually no Sim(GT) cost. However, the abstract and introduction present the contrastive loss as part of the solution (e.g., "a novel training paradigm with a contrastive identity loss that leverages paired data to balance fidelity with diversity"), and the ablation discussion (Section 6.3) focuses only on the Sim(GT) drop from removing extended negatives while ignoring that CP simultaneously *improved*. The paper should honestly acknowledge that the contrastive loss trades higher identity fidelity for higher CP, and assign proper credit for the trade-off breaking to the paired training phase.

- **Missing key face-customization baselines in the most relevant multi-person evaluation**: Table 2's multi-person evaluation omits PuLID, InstantID, UMO, InfU, and DreamO — methods that appear in Table 1's single-person evaluation. The paper states (line 209) "only those supporting multi-ID generation were additionally tested on the multi-person subset," but does not clarify which of these methods lack multi-ID support. For the multi-ID capable baselines (e.g., UMO is described in Section 2 as a multi-ID method), their absence needs explanation. The multi-person setting is the paper's core use case (the dataset is called *Multi*ID-2M), making these gaps consequential.

- **No variance or significance information for any quantitative result**: The paper claims to "break the long-observed trade-off" between identity fidelity and copy-paste. This is a strong claim requiring strong evidence. None of the tables report standard deviations, confidence intervals, or results from multiple runs. Figure 5 shows WithAnyone positioned above the fitted curve, but without variance estimates it is unclear whether this deviation is statistically significant or within noise. While the gap appears large in absolute terms (CP 0.144 vs. the next competitive method at ~0.233), reporting variance would substantially strengthen the evidence.

### Minor

- **CP metric requires a GT image, limiting real-world applicability**: The CP metric (Eq. 2) normalizes by the reference-GT angular distance, requiring access to a ground-truth target image. In real-world deployment — the scenario the paper targets — no GT image exists. The paper acknowledges this implicitly by using the benchmark setting, but should discuss how copy-paste artifacts could be detected in the absence of GT (e.g., via self-similarity across generations with different references for the same identity).

- **The Sim(GT) > 0.40 threshold filter for CP ranking is not discussed**: Table 1's caption notes that "only cases with Sim(GT) > 0.40 are considered" for Copy-Paste ranking, and Table 2 uses Sim(GT) > 0.35. This filtering could systematically advantage or disadvantage certain methods by excluding cases where they produce low-fidelity results from the CP comparison. The paper does not analyze the effect of this threshold choice.

- **User study presentation is limited**: Figure 8 uses method nicknames ("Cure", "iDetch", "Uniformal") that do not appear elsewhere in the paper, making it hard to map these to actual method names (presumably Cure=WithAnyone, iDetch=ID-Patch, Uniformal=UniPortrait). The study has only 10 participants, and no inter-rater agreement statistic is reported. The paper references Appendix H for statistical details, which are not available in the main text.

- **Identity matching threshold of 0.4 cosine similarity needs analysis**: The dataset construction pipeline (Section 3) assigns identities by matching ArcFace embeddings at a 0.4 cosine similarity threshold. For ArcFace embeddings, this is relatively low and could introduce false positive identity assignments. No precision/recall analysis is provided for this critical parameter that governs the quality of the entire paired supervision.

### Trivial
- Figure 8's method labels should include a clear mapping to the paper's method names.

## Nice-to-Haves
- Report results on non-celebrity identities to demonstrate generalization beyond the training distribution of publicly photographed individuals.
- Provide failure cases where avoiding copy-paste leads to identity drift (wrong person generated).
- Evaluate whether the CP metric correlates with human judgments of copy-paste on a per-image basis (not just rank-level), to validate the metric's perceptual validity.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic: "No statistical significance, variance, or confidence intervals reported"** — This is valid but overstated as "Critical Issue #2." For large-scale benchmark evaluation with deterministic models, single-run reporting is standard practice in the generative image community. The concern is real but not "severe" — it's a major weakness, not a fatal one. Kept in Major but downgraded from "Critical Issue."

- **Harsh critic: "The benchmark conflates matching a specific GT photo with correctly rendering identity"** — The paper explicitly uses Sim(GT) rather than Sim(Ref) precisely to penalize copying. While there's a valid concern that a model generating a valid but GT-different image might score poorly, this is inherent to any reference-based evaluation, and the paper's CP metric is specifically designed to separate these concerns. The critique also claims "CP metric cannot be computed without GT, making it inapplicable to real-world deployment" — this is true but is a property of any benchmark, not a flaw. Benchmarks use GT for evaluation; real-world use doesn't need GT. Partially kept as a Minor concern about real-world CP applicability.

- **Harsh critic: "GPT-4o result should be disqualified for prior knowledge"** — The paper already footnotes this issue in Table 2's caption. The information is disclosed for the reader to interpret, which is the right approach. Removed as a weakness.

- **Harsh critic: Training phase transition criteria are soft (20k steps, "satisfactory")** — These are standard hyperparameter choices, not a structural problem. Training recipes in practice always use such heuristics. This is a minor reproducibility concern at most, not a critical issue. Removed.

- **Harsh critic: Ethics statement about CC filters being unreliable** — The paper describes its data collection methodology; questioning the reliability of search engine CC filters is speculative and outside the paper's scope. Removed.

- **Harsh critic: User study figure labels are a "labeling error"** — The labels are abbreviated nicknames, not errors. The real issue is the missing mapping, which is a presentation concern. Moved to Minor.

- **Harsh critic: Train-test asymmetry from GT landmarks** — The paper's ablation already shows GT-Align helps. GT landmarks are available during both training and evaluation (the benchmark has GT images). This is not an asymmetry issue. Removed.

- **Strength finder: "Extended negative pool substantially improves discrimination"** — While Sim(GT) does improve (0.368→0.405), the extended negatives also increase CP (0.074→0.161). Presenting this as an unqualified strength conflicts with the verified Major weakness that contrastive loss increases copy-paste. Downgraded as a standalone strength.

- **Strength finder: "Comprehensive baseline comparison across 12+ SOTA models"** — This is only true for the single-person setting. In the multi-person (core) setting, key baselines are missing. Qualified as a partial strength.

## Novel Insights

The ablation structure reveals an important architectural insight: the copy-paste artifact and identity fidelity are not monolithic objectives addressable by a single mechanism. The paper's three training components — GT-aligned ID loss, contrastive loss, and paired tuning — each target different points on the fidelity-copy-paste plane. The GT-aligned loss improves fidelity without much affecting CP. The contrastive loss improves fidelity at the cost of increased CP. Paired tuning reduces CP without fidelity loss. This compositional decomposition of the problem space is arguably the paper's most interesting methodological contribution, even though the paper does not explicitly articulate it this way.

## Suggestions

- Reframe the contribution narrative to accurately attribute trade-off breaking to the paired tuning phase (Phase 3), and present the contrastive loss as an identity-fidelity booster that comes with a CP trade-off, which is then mitigated by Phase 3. This would make the ablation story coherent rather than contradictory.
- Add a mapping table in the user study section linking nickname labels to method names.
- Report standard deviations or at least results on 2-3 random seeds for key metrics.
- Clarify which Table 1 baselines support multi-ID generation and why absent ones were excluded from Table 2.
- Include a CP metric analysis section discussing threshold effects (Sim(GT) > 0.40 filter) and the metric's behavior when no GT is available.

## Evaluation

**Originality**: The copy-paste artifact formalization and CP metric are genuinely novel. The GT-aligned ID loss is a clever engineering contribution. The training pipeline (especially paired tuning) is well-designed but builds incrementally on established ideas.

**Importance**: The problem of copy-paste in ID-consistent generation is practically important and under-addressed. MultiID-2M fills a real data gap. The benchmark enables standardized evaluation in a space where prior work used ad-hoc protocols.

**Claims well supported?**: The main claim of breaking the trade-off is supported by Figure 5 and Table 1, but the ablation reveals the attribution is partially incorrect (contrastive loss contributes to the trade-off, not the breaking of it). This is an overclaim in the narrative, not in the results themselves.

**Soundness of experiments**: Comprehensive single-ID evaluation with 14 baselines. Multi-ID evaluation is weaker due to missing key baselines. No variance reported. Ablation is informative but the discussion of the contrastive loss ablation is incomplete.

**Clarity**: Generally well-written with clear motivation, but the user study figure labels are confusing and the ablation discussion is incomplete regarding the CP implications of contrastive loss.

**Value to community**: The dataset and benchmark are high-value resources. The method and training recipe provide a practical framework for future work on controllable ID-consistent generation.

## Calibration Anchors

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| /home/wg25r/review_agent/human_reviews_2026/DM0Y0oL33T.md (ViVerBench) | 8.0 | Strong benchmark+model paper with comprehensive evaluation, multiple data pipelines, and clear methodology; WithAnyone is less polished but makes a similar dataset+benchmark+model tripartite contribution |
| /home/wg25r/review_agent/human_reviews_2026/DVmR3Ij0ap.md (OmniPortrait) | 5.5 | Similar domain (face ID customization) with dataset contribution; OmniPortrait had weaker baselines and outdated architecture but no narrative-contradicts-ablation issue; WithAnyone is stronger on empirical evidence but has the ablation narrative inconsistency |
| /home/wg25r/review_agent/human_reviews_2026/x2DWTywZ1i.md (SIGMA-Gen) | 5.0 | Multi-identity generation with synthetic dataset; similar scope but WithAnyone has more real data and clearer trade-off analysis; SIGMA-Gen was accepted despite synthetic data concerns |
| /home/wg25r/review_agent/human_reviews_2026/XJ3T70nELl.md (CrossFaceID) | 2.67 | FaceID customization dataset paper with weak evaluation and missing baselines; WithAnyone is substantially stronger on all axes |
| /home/wg25r/review_agent/human_reviews_2026/rz6V1RI3rH.md (trade-off overclaim) | 4.0 | Claims fundamental "trade-off" but evidence contradicted narrative; similar pattern to WithAnyone's contrastive loss issue but WithAnyone's overall evidence is much stronger |
| /home/wg25r/review_agent/human_reviews_2026/NbdCwOgk4m.md (trajectory invariance overclaim) | 4.0 | Overclaimed trade-off resolution with limited ablation; WithAnyone has stronger empirical grounding but similar narrative issue |

WithAnyone is notably stronger than the low-scoring anchors (CrossFaceID at 2.67, trade-off overclaim papers at 4.0) because its empirical evidence (Figure 5, Tables 1-3) is genuinely compelling and its dataset/benchmark contributions are substantial. It is slightly below the high-scoring anchor (ViVerBench at 8.0) due to the ablation narrative inconsistency and missing multi-person baselines. It sits in a similar range to OmniPortrait (5.5) and SIGMA-Gen (5.0) but with a more substantial dataset contribution and clearer problem definition. The primary differentiator from these mid-range papers is the (partially) valid trade-off breaking claim and the valuable CP metric.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>