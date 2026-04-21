Now I have all the information needed to write the final consolidated review. Let me synthesize everything.

## Summary

OCEBO proposes the first self-distillation framework for pretraining object-centric models from scratch on real-world data. By updating the target encoder as an EMA of the object-centric model (rather than keeping it frozen as in prior work), object-centric inductive biases propagate into the target encoder, removing the performance ceiling imposed by frozen pretrained targets. A cross-view patch filtering mechanism prevents slot collapse during early training by restricting supervision to patches where the target encoder's features are consistent across views.

## Strengths

- **First demonstration that object-centric pretraining from scratch on real-world data is feasible without slot collapse.** Table 1 shows OCEBO trained on COCO achieves FG-ARI 54.8 / mBO 25.8 on MOVi-E, while removing patch filtering (row a) or the object-centric loss (row b) both result in collapse (d = 0.02). This is a genuine milestone for the field.

- **Cross-view patch filtering is a well-motivated and effective mechanism.** Table 1(a) shows removing it causes immediate collapse, and Figure 2 demonstrates the progressive nature of the mechanism (starting at ~10% supervised patches, rising to ~70%), providing a principled way to bootstrap from random targets.

- **Competitive FG-ARI performance despite orders-of-magnitude less pretraining data.** On EntitySeg FG-ARI (Table 2), OCEBO (44.2) outperforms DINOSAUR (43.5) and SPOT (41.7), and is competitive on MOVi-C FG-ARI (63.1 vs. DINOSAUR's 67.0), despite being trained on 241k vs. 142M images.

- **PCA visualizations (Figure 3) provide qualitative evidence that OCEBO's target encoder learns instance-level features** rather than semantic groupings, supporting the central thesis that EMA propagates object-centric inductive biases.

- **Introduction of a quantitative slot collapse metric (d)** that goes beyond qualitative visual inspection, providing a principled diagnostic (d > 0 indicates no collapse).

## Weaknesses

### Fatal
None.

### Major

- **The "comparable performance" claim is overstated, particularly on mBO.** The abstract states OCEBO "achieves unsupervised object discovery performance comparable to that of object-centric models with frozen non-object-centric target encoders." While FG-ARI numbers are competitive (especially vs. DINOSAUR and SPOT), mBO gaps are substantial: MOVi-C (27.3 vs. FT-DINOSAUR's 44.2), MOVi-E (22.1 vs. 29.9), EntitySeg (16.0 vs. 28.4). The EntitySeg mBO gap represents a ~44% relative deficit. The paper acknowledges models are "not directly comparable" due to architectural differences (Section 4.3) but this acknowledgment undermines the blanket "comparable" claim in the abstract and conclusion. The paper should either moderate the claim or provide a more nuanced comparison (e.g., separating FG-ARI vs. mBO performance).

- **The "removes the upper bound" / scalability claim is supported by limited evidence.** The only scalability demonstration is COCO (118k) → COCO+ (241k), a 2× increase (Table 1d). The ImageNet experiment (1.3M images) yields worse results, which the authors attribute to dataset composition (single-object scenes). While this explanation is plausible, it means the paper has not demonstrated that OCEBO continues to improve at data scales where frozen-target models plateau (e.g., 50k–100k COCO images, the critical range from Didolkar et al.). The paper claims to "remove the upper bound" (abstract, conclusion) but has only shown improvement over a narrow range, making this claim theoretical rather than demonstrated.

### Minor

- **No variance or stability analysis reported.** For a method whose primary failure mode is slot collapse and where Table 1 shows removing key components leads to collapse, reporting only single-run results is insufficient to assess reliability. Multi-seed runs with variance reporting would strengthen confidence in the method's robustness.

- **The ablation in Table 1(b) (λ_oc = 0) is characterized as equivalent to "DINO pretraining on COCO followed by FT-DINOSAUR" but uses OCEBO's training recipe rather than DINO's established hyperparameters.** This makes it a valid ablation of the object-centric loss component, but not a clean comparison against a proper DINO baseline. The paper's characterization on this point is somewhat misleading.

- **The mask sharpening stage (Section 3.4) partially qualifies the "from scratch" framing.** Table 1(c) shows it provides meaningful gains (FG-ARI 44.0→54.8 on MOVi-E), meaning a substantial portion of OCEBO's final performance comes from this non-from-scratch stage. While the paper correctly notes this uses OCEBO's own EMA-enriched target encoder (not a pretrained non-object-centric encoder), the "from scratch" characterization should be more precise.

- **The comparison in Table 2 is architecturally uncontrolled.** FT-DINOSAUR uses a top-k MLP decoder and high-resolution training stage that OCEBO does not. The authors acknowledge this, but it means the contribution of the training framework vs. architectural choices is confounded. A controlled comparison (same decoder for both) would more cleanly isolate OCEBO's contribution.

### Trivial
None.

## Nice-to-Haves

- A scalability experiment at the critical range (COCO subsets: 4k, 8k, 16k, 32k, 64k, 118k) compared against the Didolkar et al. plateau curve, which would directly demonstrate that OCEBO continues improving where frozen-target models plateau.

- Downstream task evaluation (e.g., object property prediction, compositional generalization) to demonstrate that the learned representations are useful beyond object discovery metrics.

- Sensitivity analysis of k in cross-view patch filtering and how it affects collapse rate and performance.

- Multi-seed training with variance reporting for all results.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Typo in Eq. 2: the second line should use h_t not h."** The second line of Eq. 2 already uses h_t — the harsh critic appears to have misread the equation due to formatting artifacts. The subscripts on the probability distributions are confusing (both labeled p_{t,1,2}) but the projection heads are correctly assigned.

- **"The projection head asymmetry (student on decoded q vs. teacher on encoder z) is not discussed."** The paper does explain this in Section 3.2: "In our case, this is possible because the target encoder does not have the slot attention encoder and decoder." The asymmetry is by design since the teacher lacks slot attention. While one could wish for more discussion of alternatives, the basic rationale is provided.

- **"The value of k in patch filtering is never specified in the main text."** The paper states "We perform additional ablations of the cross-view patch filtering approach in Appendix B." The appendix (stripped by the parser) likely contains the k value. This is a reproducibility nitpick about content that exists in the original submission.

- **"Missing downstream task evaluation" as a major weakness.** The paper's stated scope is demonstrating that from-scratch pretraining is possible and scalable — evaluating on downstream tasks is a natural extension but outside the paper's explicit scope. Moved to Nice-to-Have.

- **"Demanding controlled comparisons with same decoder architecture" as a major weakness.** The asymmetry actually favors FT-DINOSAUR (which has additional engineering components), so by the rules, this is weakened. Per the paper's own framing, OCEBO achieves its results *without* these components, which strengthens rather than weakens its case. The uncontrolled comparison is a minor limitation, not a major one.

- **Formatting/notation complaints about Eq. 2 subscripts.** Parser artifacts, not author errors.

- **Missing related works.** Removed per rules (cannot verify existence of suggested references).

## Novel Insights

The paper reveals an important asymmetry in self-distillation for object-centric learning: simply applying the DINO self-distillation recipe with EMA target updates fails (Table 1b, collapse), but adding object-centric inductive biases through the slot attention pathway and filtering unreliable supervision via cross-view consistency enables stable bootstrapping from scratch. This suggests that the failure of prior EMA-based target updates was not fundamental to the bootstrapping paradigm but rather to the lack of object-centric structure in the distillation signal — a subtle but important distinction that could inform future work on bootstrapping in other structured representation learning settings.

## Suggestions

- Moderate the "comparable performance" claim in the abstract and conclusion to acknowledge the mBO gap, e.g., "achieves competitive FG-ARI performance despite using orders of magnitude less pretraining data, though mBO remains behind methods with engineering enhancements."

- Add a scalability plot with multiple COCO subset sizes (as suggested in Nice-to-Haves) to directly demonstrate improvement over the frozen-target plateau from Didolkar et al.

- Report results from at least 3 random seeds with standard deviations for the main OCEBO configuration, given the method's sensitivity to collapse.

## Evaluation Axis

- **Originality:** High. First from-scratch object-centric pretraining on real-world data; cross-view patch filtering is a novel contribution.
- **Importance of research question:** High. Removing the frozen target encoder ceiling is a fundamental challenge for the field.
- **Claims well supported:** Mixed. The core claim (from-scratch pretraining works) is well supported. The "comparable performance" and "removes the upper bound" claims are overstated relative to evidence.
- **Soundness of experiments:** Moderate. Ablations are convincing but comparison is uncontrolled, scalability evidence is narrow, and no variance is reported.
- **Clarity of writing:** Good. The paper is well-structured and the method is clearly described.
- **Value to community:** High. This opens a new research direction and the code/pretrained models are available.

## Score and Decision

**Calibration anchors:**

- **CrIBo** (/home/wg25r/review_agent/human_reviews/3M0GXoUEzP.md): avg 8.0, Accept (spotlight) — Cross-image object-level bootstrapping for SSL. OCEBO is below this: CrIBo had extensive experiments across many downstream tasks; OCEBO's evaluation is narrower and claims are more overstated.

- **Provable Compositional Generalization** (/home/wg25r/review_agent/human_reviews/7VPTUWkiDQ.md): avg 7.33, Accept (oral) — Theoretical guarantees for object-centric learning. OCEBO is below this: that paper had strong theoretical grounding; OCEBO is primarily empirical with overclaimed results.

- **SlotAdapt** (/home/wg25r/review_agent/human_reviews/kZvor5aaz7.md): avg 6.25, Accept (poster) — Slot attention + pretrained diffusion models. OCEBO is comparable: both are meaningful contributions to object-centric learning with some weaknesses. OCEBO has a more fundamental contribution (first from-scratch pretraining) but weaker experimental validation.

- **CoSA** (/home/wg25r/review_agent/human_reviews/pBxeZ6pVUD.md): avg 6.0, Accept (poster) — Conditional slot attention. OCEBO is comparable to or slightly above: both address slot attention limitations, but OCEBO's contribution is more fundamental.

- **TapWeight** (/home/wg25r/review_agent/human_reviews/j3cBYvwyQT.md): avg 5.25, Reject — Novel pretraining framework with limited experiments. OCEBO is above this: it has a more concrete and verifiable contribution with stronger results.

- **FreeLM** (/home/wg25r/review_agent/human_reviews/qgLyKwXVDs.md): avg 2.0, Reject — Novel pretraining with unverified scalability and overclaimed results. OCEBO is clearly above this: OCEBO's core finding is real and verifiable; FreeLM's claims were not credible.

- **Harry Potter Visual Learning** (/home/wg25r/review_agent/human_reviews/3ZdGSTxKuy.md): avg 2.0, Reject — Overclaimed contribution with narrow experimental scope. OCEBO is clearly above: the from-scratch pretraining result is genuine.

OCEBO sits between the 5.25–6.25 range. Its core contribution (first from-scratch object-centric pretraining without collapse) is real and important, placing it above borderline-reject anchors like TapWeight (5.25). However, the overclaimed "comparable performance" and weak scalability evidence keep it below solidly-accepted papers like SlotAdapt (6.25) and CoSA (6.0). The contribution is more fundamental than CoSA's but the experimental validation is weaker. I place OCEBO at 5.5 — a meaningful contribution that would be stronger with more tempered claims and more thorough experimental validation.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>