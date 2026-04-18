Now I have enough calibration context. Let me carefully review the paper and synthesize the review.

Let me re-examine the key points from the harsh critic, checking each one against the actual paper content.

**Point 1: Notation ambiguity in Eq. 2/4**

Let me re-read the relevant section carefully.

The paper says:
- Eq. (2) defines two quantities with the same symbol **p**_{t,1,2}, but then Eq. (3) uses **p**_{t,1}, **p_2} etc. Looking more carefully:
  - In Eq. (2), the first line defines p_{t,1,2} = softmax(h(q_{t,1,2})/τ) — this appears to use q (the reconstruction)  
  - The second line defines p_{t,1,2} = softmax((h_t(z_{t,1,2}) - c_t)/τ_t) — this uses z_t (the target encoder features)

Wait, this is indeed confusing. The same symbol p_{t,1,2} is used for both the student and teacher outputs. But the subscript "t" in p_{t,1,2} suggests "target" - yet it's used for both.

Actually, looking more carefully at the notation convention, in Eq. (2) the first line uses h (student projection head) applied to q (reconstruction), and the second line uses h_t (teacher projection head) applied to z_t (target encoder features). The "t" subscript on p seems to refer to the fact that these are both in the target/probability space, not that they're outputs of the teacher. Then in Eq. (3), the cross-entropy uses p_{t,1} (teacher on view 1) with p_2 (student on view 2), which would be the standard DINO-style asymmetric loss.

Actually wait - re-reading Eq. (2) more carefully:
- p_{t,1,2} first definition: softmax(h(q_{t,1,2})/τ) — this is odd. It uses q (reconstructions) but passes through h (student head). The subscript "t" seems to just be part of the notation for probability distributions, not indicating the teacher. 

But then... in Eq. (3), we have H(p_{t,1}, p_2) + H(p_{t,2}, p_1) where p_{t,1} comes from the teacher on view 1 and p_2 comes from the student on view 2.

This is indeed confusing. The notation in Eq. (2) uses p_{t,1,2} for BOTH the student and teacher outputs, which is genuinely ambiguous. The paper should have used different symbols for student and teacher probabilities.

However, is this a "fatal" structural problem? Let me think... Looking at Eq. (3), the intent is clear: cross-entropy between teacher and student across views. The paper also says "the object-centric model can be viewed as the student and the target encoder as the teacher." The overall framework is well-understood (DINO-style self-distillation). The notation is confusing but the intent is decipherable from the context and the figure. This is a significant but not fatal issue - it's a presentation problem that makes the core contribution harder to verify, but the surrounding context makes the intent reasonably clear.

**Point 2: "First ever" and "comparable" claims**

The "first time ever" claim about pretraining object-centric models from scratch on real-world data... the paper does acknowledge in Related Work that "To the best of our knowledge, direct pretraining of object-centric models on real-world datasets has not been achieved yet." But in the abstract it says "for the first time ever" without qualification. This is a notable overclaim in the abstract specifically.

For "comparable" - looking at Table 2, OCEBO gets FG-ARI of 63.1, 66.8, 22.4, 44.2 compared to FT-DINOSAUR's 71.3, 71.1, 24.0, 48.1. The gaps are 8.2, 4.3, 1.6, 3.9 on FG-ARI. On mBO, the gaps are larger: 27.3 vs 44.2 (MOVi-C), 22.1 vs 29.9 (MOVi-E), 34.4 vs 37.6 (VOC), 16.0 vs 28.4 (EntitySeg). The mBO gaps are substantial. The "comparable" claim is somewhat fair for FG-ARI but quite generous for mBO.

**Point 3: EMA "removes upper bound" claim**

This is a valid concern. The paper argues that frozen targets create an upper bound and EMA removes it, but doesn't show a controlled experiment where the only difference is frozen vs. EMA target. The baseline in Table 1(b) is DINO pretrained on COCO + FT-DINOSAUR, which is a different setup. This is a real gap in the evidence. However, looking at it from the other side, they cite Didolkar et al. (2024) who showed that performance plateaus at ~16k images with frozen targets, and OCEBO shows continued improvement going from 118k to 241k. This is indirect but suggestive.

**Point 4: Cross-view patch filtering**

The lack of ablation on k and computational cost analysis in the main text is noted. The paper says additional ablations are in Appendix B. This is a valid minor concern.

**Point 5: Evaluation limited to object discovery**

This is a valid concern about scope. The paper does mention Appendix A for patch representation quality.

**Point 6: Mask sharpening stage**

The observation that mask sharpening uses frozen targets (which the paper argued against) is interesting. Table 1(c) shows the sharpening stage provides meaningful improvement. This partially contradicts the narrative but the authors acknowledge it as an "optional" stage. The concern about how much of the final performance comes from this stage vs. the core OCEBO loss is valid.

Now let me synthesize the final review, properly tiering the weaknesses.

From the calibration papers:
- Cycle Consistency Object Discovery (8,8,6,5) → Accept poster — novel method, good results, some concerns about evaluation scope
- On Transfer of OCL (6,5,3,6) → Accept Poster — useful benchmark, limited novelty, evaluation concerns  
- Adaptive Slot Attention (5,5,5,5) → Withdrawn/Reject — limited novelty, evaluation concerns
- Learning to Compose (6,6,8,6) → Accept Poster — novel objective, synthetic-only evaluation
- CrIBo (8,6,8,10) → Accept Spotlight — strong results, similar concerns about computational cost and evaluation scope

OCEBO is a genuinely novel contribution (first from-scratch pretraining of object-centric models) with encouraging results but some notable gaps in evidence and presentation. It's probably in the 5-6 range - above the reject threshold but with significant room for improvement.</think>Now I have enough context from similar papers and calibration data. Let me carefully verify the key criticisms against the actual paper content before writing the final review.

## Summary

OCEBO proposes the first framework for pretraining object-centric (slot attention-based) models from scratch on real-world data, replacing the standard frozen pretrained target encoder with an EMA-updated target encoder that progressively acquires object-centric inductive biases through the self-distillation loop. A key technical contribution is cross-view patch filtering, which prevents slot collapse by restricting supervision to patches whose features are sufficiently consistent across augmented views during early training. An optional mask sharpening stage with frozen targets and ℓ₂ loss improves mask boundaries after the main training phase.

## Strengths

- **Addresses a fundamental and well-motivated limitation in object-centric learning.** Prior methods (DINOSAUR, FT-DINOSAUR, SPOT) rely on frozen pretrained non-object-centric encoders (DINO, DINOv2), which impose an upper bound on attainable performance and exhibit poor data scalability (plateau at ~16k COCO images per Didolkar et al. 2024). Removing this dependency is an important conceptual step.

- **Novel and technically sound core mechanism.** The cross-view patch filtering idea is elegant: using mutual k-nearest-neighbor consistency across views as a proxy for target feature quality, starting from only ~10% supervised patches and progressively increasing. Table 1(a) demonstrates that removing it causes immediate collapse (d=0.02), confirming its necessity.

- **Demonstrates data scalability, unlike prior work.** Table 1(d) shows consistent FG-ARI improvements from COCO (~118k) to COCO+ (~241k), contrasting with the plateau behavior of frozen-target methods. The ImageNet negative result (FG-ARI 46.0, mBO 16.0) is also informative, showing that dataset structure matters, not just scale.

- **Competitive FG-ARI despite orders-of-magnitude less pretraining data.** Table 2 shows OCEBO achieves FG-ARI of 63.1, 66.8, 22.4, 44.2 across four benchmarks—within range of DINOSAUR/FT-DINOSAUR using DINOv2 pretrained on 142M images. This is a meaningful result given that OCEBO trains from scratch on only ~241k images.

- **Useful quantitative collapse metric.** The proposed *d* metric for measuring slot collapse (cross-view semantic vs. positional similarity) is a practical contribution that enables more rigorous evaluation beyond qualitative inspection.

## Weaknesses

### Major

- **Ambiguous notation in the core loss definition (Eqs. 2/4) makes the training objective unclear.** Equations (2) and (4) define the symbol $\mathbf{p}_{t,1,2}$ twice—once for the student branch (softmax of $h(\mathbf{q})/\tau$) and once for the teacher branch (softmax of $(h_t(\mathbf{z}_t) - \mathbf{c}_t)/\tau_t$). The same symbol is used for both branches without distinguishing subscripts. While the overall architecture (Figure 1) and the description of student/teacher roles ("the object-centric model can be viewed as the student and the target encoder as the teacher") make the likely intent decipherable, for a self-distillation method where the training objective is the central contribution, unambiguous notation indicating which probabilities are detached (teacher) and which receive gradients (student) is essential. Without explicit stop-gradient notation or distinct symbols, it is impossible to verify from the math alone that the loss avoids collapse. This is not a minor notation slip—the core contribution of the paper relies on this loss being correctly formulated.

- **The "upper bound removal" claim is asserted rather than demonstrated with controlled comparison.** The paper's central narrative is that EMA-updating the target encoder removes a performance ceiling imposed by frozen targets. However, there is no experiment that directly compares frozen-target vs. EMA-target under otherwise identical conditions (same architecture, same training data, same training schedule). Table 1(b) replaces OCEBO entirely with DINO pretraining + FT-DINOSAUR fine-tuning—this is a different method altogether, not a frozen-target version of OCEBO. What would strengthen the claim significantly is an experiment starting from a pretrained target encoder (e.g., DINOv2), keeping it frozen vs. updating it via EMA within the OCEBO framework, on the same data. The scalability result (Table 1(d)) is encouraging but indirect evidence—it shows continued improvement with more data, but does not prove that the mechanism is specifically the removal of an upper bound vs. other factors.

- **The mBO gap vs. pretrained baselines is substantial, and the "comparable performance" framing is overstated.** While FG-ARI is competitive (within ~8 points of FT-DINOSAUR), mBO is consistently and significantly lower: 27.3 vs. 44.2 on MOVi-C, 22.1 vs. 29.9 on MOVi-E, 16.0 vs. 28.4 on EntitySeg. The paper acknowledges that FG-ARI and mBO trade off against each other, but the abstract and conclusion claim "performance comparable to that of object-centric models with frozen non-object-centric target encoders pretrained on hundreds of millions of images" without this qualification. This should be stated more carefully: OCEBO achieves comparable FG-ARI but notably lower mBO, with the gap indicating weaker mask boundary quality.

### Minor

- **The mask sharpening stage partially contradicts the paper's narrative and its contribution is not clearly disentangled.** After arguing that frozen targets are a fundamental limitation, the method requires a 100-epoch mask sharpening stage with a frozen target and ℓ₂ loss. Table 1(c) shows this provides non-trivial improvement (e.g., MOVi-E FG-ARI: 44.0 → 54.8). This introduces the question of how much of the final performance is attributable to the self-distillation phase vs. this sharpening stage. The sharpening stage is described as "optional" yet appears necessary for competitive results. The relationship between this stage and the previously criticized frozen-target regime could be discussed more transparently.

- **No downstream task evaluation beyond unsupervised object discovery.** The paper positions OCEBO toward "large-scale object-centric foundation models" but evaluates only FG-ARI and mBO for zero-shot object discovery. Appendix A is mentioned for patch representation quality but not presented in the main text. Demonstrating utility on at least one downstream task (e.g., object property prediction, robotic control, image generation) would strengthen the representation-learning claims. That said, this is the first paper achieving from-scratch pretraining on real-world data, so the focus on object discovery is understandable as a first step.

- **Limited analysis of cross-view patch filtering.** The value of $k$ in Eq. (7) and the similarity measure used for nearest-neighbor computation are not specified in the main text (Appendix B is referenced for ablations). Computational overhead of the per-image k-NN computation at each training step is also not discussed, which is relevant given the stated goal of large-scale pretraining.

### Trivial

- The abstract claims "for the first time ever" without qualification. The paper's related work section does include "To the best of our knowledge, direct pretraining of object-centric models on real-world datasets has not been achieved yet," but this hedge should accompany the stronger claims in the abstract/introduction as well.

## Nice-to-Haves

- A controlled experiment comparing frozen vs. EMA-updated target within the OCEBO framework would directly validate the "upper bound removal" hypothesis.
- Scaling experiments beyond 241k images (e.g., on a curated multi-object dataset) to demonstrate the trend continues.
- Integration of architectural improvements from SPOT/FT-DINOSAUR (autoregressive decoder, top-k MLP, high-resolution fine-tuning) to close the mBO gap and demonstrate composability.
- Evaluation on downstream tasks beyond object discovery to support the "foundation model" framing.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Criticism that the ImageNet failure is "underanalyzed" or undermines scalability claims.** The paper already acknowledges this limitation in Section 4.2 ("Constructing a large-scale dataset suitable for pretraining of object-centric models remains an open question") and the negative result is informative rather than damning. The paper explicitly discusses why ImageNet's single-object scenes are unsuitable—this is a useful finding, not a missing experiment.

- **Demand for error bars/statistical significance across multiple runs.** While desirable, single-run evaluation is the norm in the object-centric learning literature (and in the baselines the paper compares against). Flagging this disproportionately for this paper alone would be inconsistent with community standards.

- **Demand for evaluation on larger architectures (ViT-B, ViT-L).** The paper uses ViT-S/16, consistent with prior work (DINOSAUR, FT-DINOSAUR). Requesting scale-up without providing the compute budget is scope creep beyond the paper's contribution.

- **Criticism that the comparison in Table 2 is "not apples-to-apples."** The paper explicitly acknowledges this (Section 4.3, first paragraph of comparison) and states it is not focused on tuning for highest numbers. The asymmetry favors the baselines (they use pretrained encoders on 100-1000x more data), so this comparison style actually makes the result harder, not easier, to achieve.

- **Demand for proving that no prior work has pretrained object-centric models from scratch.** The paper states "to the best of our knowledge" in the related work. This is the standard scholarly formulation. Asking for exhaustive proof of novelty claims is unreasonable.

## Novel Insights

The cross-view patch filtering mechanism is an interesting instance of a more general principle: in self-distillation from scratch, the quality of the teacher's supervision varies over time and across spatial locations, and explicitly gating the loss on a per-sample, per-location basis based on consistency diagnostics is a more principled approach than global confidence thresholds or curriculum schedules. This idea—using cross-view correspondence as a proxy for supervision quality—could generalize beyond object-centric learning to any self-distillation framework bootstrapping from random initialization.

## Suggestions

- **Clarify the loss formulation unambiguously.** Rewrite Eqs. (2) and (4) with distinct notation for student and teacher probabilities, and explicitly mark the teacher branch with stop-gradient (e.g., $\text{sg}[\cdot]$). This is essential for the core contribution to be verifiable.
- **Reframe the "comparable performance" claim.** State explicitly that FG-ARI is competitive while mBO remains substantially lower, and qualify the abstract/conclusion accordingly.
- **Add a direct frozen-vs-EMA ablation.** Train OCEBO with a frozen DINOv2 target on COCO+ as a controlled baseline. This directly tests whether EMA-plus-object-centric-biases vs. frozen-pretrained-encoder is the key factor, and would either substantiate or weaken the "upper bound removal" claim convincingly.

## Score and Decision

**Calibration comparison:**
- Cycle Consistency Object Discovery (8,8,6,5, Accept poster): Novel cycle-consistency regularization for slot attention, evaluated on object discovery + downstream RL. Sound empirical results, clear contribution.
- On Transfer of OCL (6,5,3,6, Accept Poster): Benchmark paper + fine-tuning strategy, limited novelty, evaluation concerns but useful contribution.
- Adaptive Slot Attention (5,5,5,5, Withdrawn/Reject): Dynamic slot number for slot attention, limited novelty, narrow evaluation.
- Learning to Compose (6,6,8,6, Accept Poster): Novel compositional objective for slot attention, evaluated on synthetic datasets only.
- CrIBo (8,6,8,10, Accept Spotlight): Cross-image object-level bootstrapping, strong results, similar concerns about computational cost.

OCEBO proposes a genuinely novel and important direction (first from-scratch object-centric pretraining) with encouraging results. It demonstrates a real conceptual advance over prior work. However, the paper has two significant issues: (1) the core loss formulation is not presented clearly enough to verify, and (2) the central claim about "removing the upper bound" lacks a direct controlled experiment. The "comparable performance" framing also overstates the mBO results. These are substantial but addressable issues. The paper is above the acceptance threshold given the novelty and importance of the direction, but not comfortably so.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>

Note: The score of 5.5 reflects that this is an important and novel direction with encouraging results, but the ambiguous loss formulation and insufficient evidence for the core "upper bound removal" claim are significant enough that they need to be resolved before publication. With clearer notation and a controlled frozen-vs-EMA experiment, this could easily be a 6.5-7. The decision is borderline—this could be a weak accept with revisions, but as submitted, the weaknesses in the core technical presentation and evidence for the central claim push it below the acceptance bar.