Now I have a thorough understanding of the paper. Let me carefully verify the harsh critic's key claims:

**Claim 1: "Comparable performance" claim is contradicted by the data** — The paper says "comparable" in the abstract, intro, and conclusion. Table 2 shows OCEBO is worse on both FG-ARI and mBO against FT-DINOSAUR (DINOv2) on most datasets. However, against DINOSAUR (DINOv2), OCEBO actually beats it on MOVi-E FG-ARI (66.8 vs 71.1... no, 66.8 < 71.1). Let me re-check:

Table 2:
- vs DINOSAUR (DINO): OCEBO is close on MOVi-C FG-ARI (63.1 vs 66.9), and beats it on EntitySeg FG-ARI (44.2 vs 43.7) and mBO (16.0 vs 25.1... no, 16.0 < 25.1). Actually let me re-read the table carefully.

MOVi-C: OCEBO 63.1 FG-ARI vs SlotDiffusion-DINO 66.9, SPOT-DINO 63.0, DINOSAUR-DINOv2 67.0, FT-DINOSAUR-DINOv2 71.3
MOVi-E: OCEBO 66.8 vs SlotDiffusion-DINO 67.6, SPOT-DINO 47.8, DINOSAUR-DINOv2 71.1, FT-DINOSAUR-DINOv2 71.1
Pascal VOC: OCEBO 22.4 vs ... various
EntitySeg: OCEBO 44.2 FG-ARI, 16.0 mBO vs FT-DINOSAUR 48.1, 28.4

The paper acknowledges this in Section 4.3: "the models are not directly comparable" and "we do not focus on tuning the performance." The claim of "comparable" is arguable — OCEBO is in the same ballpark as some baselines but not as strong as FT-DINOSAUR. The "comparable" word is a judgment call; it's somewhat overclaimed but the paper does provide context. The claim that it's "on every metric on every dataset" is factually correct for FT-DINOSAUR.

**Claim 2: Mask sharpening contradicts narrative** — The paper describes it as "optional" but Table 1(c) shows significant drops without it. The paper does frame it as similar to FT-DINOSAUR fine-tuning but with an object-centric target. This is a fair tension to note, though the paper partially addresses it by noting the target encoder has object-centric inductive biases.

**Claim 3: Scalability claim supported by only two data points** — True, only COCO vs COCO+. But the paper also mentions the ImageNet negative result and notes that scaling in the Appendix.

**Claim 4: λ_oc = 0 ablation confounded** — This is a valid methodological concern. When λ_oc = 0, the slot attention only trains during the 100-epoch sharpening, not the full 400 epochs. The paper describes this as "DINO pretraining on COCO followed by FT-DINOSAUR fine-tuning," which is a meaningful comparison but doesn't isolate the effect of object-centric inductive biases from training duration.

Now let me write the review.</think>## Summary

OCEBO proposes the first pretraining scheme for object-centric models on real-world data from scratch, replacing the frozen pretrained target encoder (e.g., DINOv2) used in prior work with an EMA-updated target encoder enriched with object-centric inductive biases via slot attention. To address slot collapse from random initialization, the paper introduces cross-view patch filtering, which progressively admits patches into the supervision signal as the target encoder's features improve, and an optional mask sharpening stage that freezes the target encoder and switches to ℓ₂ reconstruction. Evaluated via zero-shot unsupervised object discovery on multiple datasets, OCEBO trained on ~241k COCO+ images achieves performance in the range of methods using DINOv2 pretrained on 142M images.

## Strengths

- **First successful from-scratch pretraining of object-centric models on real-world data without collapse.** This is a genuine and important milestone. Prior work (Didolkar et al., 2024) showed that simply unfreezing/updating the target encoder causes collapse; OCEBO demonstrates this is solvable through object-centric inductive biases. Table 1 confirms OCEBO (d=0.13) avoids collapse while baselines fail.

- **Cross-view patch filtering is well-motivated and effective.** The insight that noisy early-stage targets cause collapse, and that cross-view correspondence serves as an informative-quality proxy, is elegant. Table 1(a) shows removing it causes immediate collapse (FG-ARI 54.8→27.7); Figure 2 reveals a natural curriculum from ~10% to ~70% supervised patches.

- **Object-centric inductive biases in the target encoder demonstrably matter.** Table 1(b) shows that λ_oc=0 (reducing to DINO pretraining + FT-DINOSAUR fine-tuning) also collapses (d=0.02), directly supporting the central hypothesis. PCA visualizations (Figure 3) provide complementary qualitative evidence of instance-level separation vs. semantic grouping.

- **The paper is honest about the asymmetry of comparison.** Section 4.3 explicitly states "the models are not directly comparable" and that OCEBO does not incorporate architectural improvements (autoregressive decoders, top-k MLP, high-resolution training) from SPOT and FT-DINOSAUR, which would likely improve its numbers further.

## Weaknesses

### Fatal

None.

### Major

- **The "comparable performance" claim is overstated relative to the strongest baselines.** The abstract and conclusion claim OCEBO achieves "performance comparable to that of object-centric models with frozen non-object-centric target encoders pretrained on hundreds of millions of images." Against FT-DINOSAUR (DINOv2)—the strongest baseline—OCEBO is worse on both FG-ARI and mBO on all four reported datasets (e.g., EntitySeg: 44.2 vs 48.1 FG-ARI, 16.0 vs 28.4 mBO; MOVi-C: 63.1 vs 71.3 FG-ARI, 27.3 vs 44.2 mBO). The paper's own Section 4.3 implicitly acknowledges this by noting it does not use architectural improvements that boost those baselines. While "comparable" is a spectrum, the consistent gap on both metrics against the SOTA baseline makes the claim misleading. The paper would be better served by framing this as "competitive in parts" or "in the same ballpark on some metrics" rather than "comparable," emphasizing the from-scratch advantage instead. — Overclaiming on the headline result can mislead readers about the current state of the method vs. the feasibility milestone it represents.

- **The mask sharpening stage creates a tension with the paper's narrative that is not adequately addressed.** The central thesis is that EMA-updated object-centric target encoders remove the ceiling imposed by frozen targets. Yet the mask sharpening stage (Section 3.4)—which Table 1(c) shows is de facto required (FG-ARI 54.8→44.0 on MOVi-E without it)—works by freezing the target encoder and using ℓ₂ reconstruction, essentially the FT-DINOSAUR paradigm. The paper describes this as "optional" and notes the target is now object-centric, which partially addresses the concern. However, the narrative framing does not honestly confront that the EMA-updated target alone does not produce clean enough masks for competitive performance, and the final 100-epoch stage reverts to the frozen-target paradigm. The contribution should be framed as EMA pretraining *plus* frozen-target sharpening, not as replacing frozen targets entirely. — This matters because it affects how readers understand the true nature and limits of the contribution.

- **The scalability claim is supported by minimal evidence.** The paper claims OCEBO "scales well beyond a few thousand training images, unlike other object-centric models" (Abstract, Introduction, Conclusion). The evidence is a single comparison: COCO (118k) vs COCO+ (241k) in Table 1(d). This is a doubling on essentially the same distribution. The ImageNet experiment shows performance *drops*, attributed to single-object scenes—but this means scalability to actual large-scale diverse data remains unvalidated. Without a proper scaling curve (e.g., subsets at 25k, 50k, 118k, 241k), the scalability claim only establishes "OCEBO does not plateau at 16k like prior work," not that it "scales well." — Scalability is central to the paper's framing and a stronger evidential basis is expected.

### Minor

- **The λ_oc = 0 ablation has a training-duration confound.** Table 1(b) sets λ_oc=0, meaning slot attention trains only during the 100-epoch sharpening stage, versus 400 total epochs in full OCEBO. The collapse observed could partly result from insufficient slot-attention training rather than purely from absence of object-centric inductive biases. A fairer ablation would train slot attention for 400 epochs with a DINO-from-scratch target encoder. — This does not invalidate the conclusion (DINO-on-COCO features may indeed be insufficient), but the ablation isolates the wrong variable.

- **The collapse metric d lacks a defined threshold.** Section 4.2 introduces d = sim(q_{1,i}, q_{2,j}) − sim(q_{1,i}, q_{2,i}), which is reasonable but ad hoc. The paper shows d=0.02 correlates with collapse and d≥0.13 does not, but no systematic sweep or threshold definition is provided. — Minor because the qualitative collapse confirmation is adequate for the claims made.

### Trivial

None.

## Nice-to-Haves

- A proper scaling curve with multiple data sizes (25k, 50k, 100k, 241k) from COCO/COCO+ would substantially strengthen the scalability claim.
- Running OCEBO with one architectural improvement from SPOT/FT-DINOSAUR (e.g., autoregressive decoder or high-resolution training stage) would demonstrate the potential upper bound.
- Ablation over k in cross-view patch filtering would clarify sensitivity of the primary collapse-prevention mechanism.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Not yet released" / reproducibility concern about code/models**: The paper provides a GitHub link; per rules, we assume cited resources exist. Removed.

- **Missing appendix / missing proofs in appendix**: Parser strips appendices; the paper references Appendix A–C which exist in the original submission. Removed.

- **Missing ablation over k**: This is partially a valid point; the paper mentions appendix ablations. Moved to Nice-to-Have as it would strengthen but not invalidate.

- **invaug interpolation artifacts**: The harsh critic speculates about artifacts from interpolation in overlapping regions but provides no evidence this is a problem in practice; the model works. Removed as speculative.

- **Slot count varies per dataset**: The paper notes this is standard practice in the field (FT-DINOSAUR also varies slot counts). Removed as not a specific weakness of this paper.

- **"We do not focus on tuning" creates asymmetry**: The paper explicitly acknowledges this in Section 4.3 and predicts that incorporating improvements from SPOT/FT-DINOSAUR would help. The criticism that "this is not a trivial engineering step" is speculative. Moved to Nice-to-Have as a suggestion for future work.

- **Formatting/style nitpicks and typos**: Removed per rules.

- **Undisclosed hyperparameters / reproducibility nitpicks about training details**: Standard DINO hyperparameters are used; code is provided. Removed per rules.

## Novel Insights

The paper reveals an interesting asymmetry in self-distillation for object-centric learning: the very inductive biases (semantic grouping) that make pretrained SSL encoders effective as frozen targets become a liability when those encoders serve as EMA teachers, because the object-centric model's encoder subtly shifts toward positional/instance-level features, causing the EMA-updated target to leak informative knowledge at the cost of positional contamination. This explains the failure of naive EMA updating observed by prior work and motivates the need for object-centric inductive biases in the teacher. The cross-view patch filtering mechanism is a clever curriculum that exploits a simple test—can the target encoder match a patch across views?—as a proxy for target quality, avoiding the need for explicit quality estimation.

## Suggestions

- Tone down "comparable performance" to more precise language (e.g., "competitive on FG-ARI with methods using orders-of-magnitude more pretraining data, though trailing on mBO") in the abstract and conclusion. Given that OCEBO is worse on both metrics vs. FT-DINOSAUR, claiming "comparable" without qualification is misleading.

- Reframe the mask sharpening stage more honestly: label it as a necessary component (not "optional") and explicitly discuss the hybrid nature of the contribution—EMA pretraining creates a good object-centric target, then frozen-target sharpening leverages it. This is actually a strength (the target is object-centric rather than semantic) but requires clearer presentation.

- Add even a simple 3-point scaling curve (e.g., 25k, 118k, 241k subsets of COCO+) to support the central scalability claim.

## Calibration

**Anchors used:**

1. **High-scoring (>7):** CrIBo (avg 8.0, Accept spotlight) — cross-image object-level bootstrapping for SSL, strong SOTA results, well-motivated, object-centric-adjacent. OCEBO is below CrIBo: CrIBo achieves clear SOTA on standard benchmarks while OCEBO trails FT-DINOSAUR and overclaims comparability. Provable Compositional Generalization (avg 7.33, Accept oral) — theoretical contribution to object-centric generalization. OCEBO lacks the theoretical depth but has a real empirical milestone.

2. **Medium-scoring (4–6):** dBOT (avg 5.25, Accept poster) — bootstrapping from randomly initialized teacher for masked autoencoders; novel training paradigm with solid but not overwhelming results. OCEBO is comparable in spirit (novel from-scratch bootstrapping scheme, feasibility milestone) but with more significant overclaiming in framing. Progressive Subnetwork Training (avg 5.75, Reject) — novel stagewise training scheme with preliminary results. OCEBO is stronger: it demonstrates a genuine first (from-scratch OCL without collapse), has clear ablations, and a meaningful (if overclaimed) comparison.

3. **Low-scoring (<3):** Unsupervised Object Detection via Contrastive Learning (avg 3.0, Reject) — overclaimed "groundbreaking" results with missing baselines. OCEBO is far above this: it has real technical contributions, proper ablations, and honest discussion (even if "comparable" is overstated). Object-centric world modelling with slot attention (avg 3.0, Reject) — incremental combination of existing architectures with insufficient novelty. OCEBO has more novelty (the from-scratch training scheme is new) and better experimental validation.

**Score reasoning:** OCEBO demonstrates a genuine first (from-scratch object-centric pretraining without collapse) with strong ablation evidence, but its headline claim of "comparable performance" is undermined by consistent gaps against the strongest baseline, the mask sharpening stage undercuts the narrative, and scalability evidence is thin. This places it in the medium band—above low anchors (3.0) that lack novelty or have fundamentally flawed claims, but below high anchors (7–8) that deliver clear SOTA or theoretical depth. Among medium anchors, it is above the rejected ones (Progressive Subnetwork at 5.75) because the contribution is more novel and timely, but the overclaiming is a real issue. Comparable to dBOT (5.25, accepted) in terms of "novel training paradigm + feasibility result," though dBOT had less overclaiming. A score of 5.5 reflects a meaningful contribution that needs better framing and more evidence for its scalability claims.

## Evaluation Axes

- **Originality:** High. First from-scratch pretraining scheme for object-centric models; cross-view patch filtering is novel.
- **Importance of research question:** High. Breaking the frozen-encoder ceiling is central to the field.
- **Claims well supported:** Mixed. The feasibility claim (from-scratch without collapse) is well-supported; the "comparable performance" and "scales well" claims are weaker than presented.
- **Soundness of experiments:** Adequate for feasibility but insufficient for scalability claims. Key ablation has a confound.
- **Clarity of writing:** Good. The paper is well-structured and the method is clearly described.
- **Value to research community:** High. Opens a new research direction even in its current form.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>