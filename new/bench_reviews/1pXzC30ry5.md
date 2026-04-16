## Summary

This paper introduces the novel setting of "real-time multi-purpose segmentation," which unifies interactive segmentation, panoptic segmentation, and video instance segmentation within a single efficient model. To address this, the authors propose RMP-SAM, featuring a lightweight encoder, a unified decoder with pooling-based dynamic convolution, and lightweight asymmetric adapters (one for object queries, one for prompt queries) to balance the conflicting needs of semantic-level and interactive-level tasks. The paper also provides a benchmark by extending existing segmentation methods to this multi-task setting, demonstrating that RMP-SAM achieves the best speed-accuracy trade-off across the three tasks.

## Strengths

- **Novel and well-motivated problem setting**: The paper identifies a genuine gap between heavy multi-purpose models and fast single-task models, and proposes a practical unified benchmark for real-time multi-purpose segmentation. This is a useful contribution for the community.

- **Effective architectural design**: The asymmetric adapter design is well-motivated — object queries need scene-level and spatio-temporal context while prompt queries need local detail — and the ablation in Tab. 6(d) clearly validates that `{DC, CA}` (asymmetric) outperforms symmetric alternatives, with 56.7 COCO-SAM mIoU vs. next best 56.2 and 44.6 PQ vs. 44.5/42.6.

- **Comprehensive benchmarking effort**: Re-implementing Mask2Former, MaskFormer, kMaX-DeepLab, and YOSO under the same training framework and evaluating them on three tasks simultaneously provides a valuable reference point. The results in Tab. 3 show clear improvements: e.g., RMP-SAM with R50 achieves 46.9 PQ vs. 42.9 for Mask2Former-R50, while being ~9 FPS faster (35.1 vs. 26.6).

- **Good speed-accuracy trade-off with clear margins**: Across all three backbone configurations (R18, R50, TopFormer), RMP-SAM consistently achieves competitive or best PQ, mIoU, and VIS mAP while maintaining real-time FPS (30–40). This demonstrates the practical utility of the dynamic convolution + adapter approach.

## Weaknesses

### Major

- **Negative transfer undermines the core "multi-purpose" motivation**: Tab. 6(b) shows that adding COCO-SAM training to the joint image+video co-training *reduces* panoptic PQ (36.2→35.7) and VIS mAP (36.0→35.3). While adding video data helps VIS, adding interactive segmentation data hurts both other tasks. This directly challenges the premise that a single multi-purpose model is superior. The paper acknowledges "different learning targets" but does not resolve or mitigate this conflict. Without comparisons to training three separate single-task models, the practical benefit of the multi-purpose framework — versus simply deploying three lightweight specialists — remains unclear.

- **Non-standard interactive segmentation evaluation**: The paper evaluates interactive segmentation using only center-point prompts at test time (Sec. 2: "we obtain the center point of the mask as a visual prompt") and box prompts from a detector (Tab. 4). This is significantly easier than standard SAM evaluation protocols (multi-click iterative refinement, diverse point placements). The center-point oracle protocol advantages the model and inflates interactive segmentation performance. While Tab. 4 does show competitive mAP with SAM variants, the evaluation setup is not directly comparable to how SAM models are typically benchmarked.

- **Missing comparison with SAM-2**: SAM-2, cited in the paper, is the most directly competing unified model that handles both image and video interactive segmentation. The paper acknowledges SAM-2 in Tab. 1 but excludes it from experimental comparisons. Even if SAM-2 is not "real-time," including it would contextualize the performance gap and demonstrate where RMP-SAM stands relative to the state-of-the-art unified model.

### Minor

- **Vague temporal modeling description**: Section 3.1 mentions extracting spatial-temporal features for video but does not detail the mechanism (3D convolution? temporal attention? frame concatenation?). The "Lite Neck" receives no architectural specification beyond FPN with deformable convolution. This hinders reproducibility.

- **FPS measured only on A100, input resolution unspecified**: The "real-time" claims are anchored to a high-end A100 GPU. While this is common practice for comparative evaluation, it limits claims about deployment practicality on edge devices. The input resolution for FPS measurements is also not clearly stated in the main text.

- **Confusion in Tab. 4 notation**: The "Interactive" column marks SAM variants (TinySAM, MobileSAM, etc.) with "✗" even though these are interactive segmentation models by design. The column refers to multi-purpose capability, but this is misleading in a table specifically comparing interactive segmentation performance (mAP).

## Nice-to-Haves

- Comparison against three separate single-task models to quantify the practical benefit (compute, latency, accuracy) of unification vs. specialization.
- Results on YouTube-VIS 2021/2022, which are more challenging and more commonly reported benchmarks than 2019.
- Per-task latency breakdown rather than a single FPS number, since interactive and video inference may have different bottlenecks.
- More realistic interactive segmentation evaluation (multi-point, iterative correction) to align with SAM literature conventions.

## Removed Points

- **"SAM-like baselines in Tab. 4 marked Interactive ✗ is logically wrong"**: This conflates two readings. The column reflects multi-purpose capability (PS + VIS + Interactive in one model), not whether a model can do interactive segmentation at all. While potentially confusing, it is not factually wrong — it represents what the paper is actually comparing on (unified multi-purpose models). *Kept as a minor notation issue rather than a fatal flaw.*

- **"Real-time claim is fundamentally misleading"**: The paper consistently reports FPS on A100 and compares against other models on the same hardware, making the *relative* speed claims valid. The "real-time" label is standard in the efficient segmentation literature (YOSO, ICNet, etc.) for models achieving >30 FPS on server GPUs. *Downgraded from fatal to minor.*

- **"Re-implemented baselines may not match published numbers"**: The authors explicitly state they re-implement baselines "using the same codebase" with identical training settings for fair comparison, which is a valid methodological choice. The concern about under-representing baselines is speculative — the paper is transparent about this. *Removed.*

- **"Per-pixel cross-attention FLOPs not reported in Tab. 6(c)"**: The text states per-pixel cross-attention "introduces extra computation costs" and the table shows dynamic convolution achieves competitive accuracy. The FLOPs difference is implied by the architectural description. *Downgraded to minor.*

- **"Eq. 4 is confusing about X_{i-1} vs X_i"**: This is a notational concern; Eq. 4 builds on Eq. 3 which defines X_i in terms of M_{i-1}, making the recurrence implicit but interpretable. Removed as a formatting nitpick.

- **"Demanding edge device benchmarks"**: The paper's scope is about relative efficiency compared to transformer baselines. Edge deployment is a separate engineering concern. Moved to nice-to-have.

- **"Demanding theoretical proofs for task interference"**: This paper is an empirical systems contribution. Theoretical analysis of why tasks conflict is outside scope. Removed.

- **"Missing OMG-Seg or other unified segmentation models"**: I cannot verify whether these references exist or are relevant, per the rules about not mentioning missing related works. Removed.

## Novel Insights

The asymmetric adapter ablation provides a genuinely insightful observation: interactive segmentation (prompt-driven) benefits more from pixel-wise cross-attention adapters while semantic segmentation (object queries) benefits more from dynamic convolution adapters. This aligns with the intuition that prompt queries need fine-grained local detail (which pooling discards) while object queries benefit from broader context aggregation (which pooling provides efficiently). This is a useful architectural takeaway for future multi-purpose model design.

## Suggestions

1. Add a comparison against single-task specialists (3 separate models of comparable total parameter budget) to demonstrate whether joint training truly provides a practical advantage beyond parameter savings.
2. Report interactive segmentation results under a more standard evaluation protocol (e.g., single-point at various locations, not just center-point) to make claims about SAM-like capability more credible.
3. Include SAM-2 in comparison tables, even if only on accuracy (not speed), to contextualize RMP-SAM's performance against the strongest unified baseline.

## Score and Decision

**Calibration**: I compared against: (1) PRO-SCALE (scores 5,6,6,6, accepted as poster) — an efficient segmentation paper with moderate novelty and solid speed-accuracy tradeoffs; (2) QueryMeldNet (scores 5,3,5,6,5, rejected/withdrawn) — a multi-task segmentation paper with task conflict issues, missing baselines, and limited novelty; (3) LegoNet (scores 5,5,8, withdrawn/rejected) — multi-task learning with modular design but novelty concerns.

RMP-SAM is stronger than QueryMeldNet and LegoNet in that it defines a clear new problem setting, provides comprehensive baseline experiments, and demonstrates consistent improvements. However, it shares the negative transfer issue with multi-task papers, and the evaluation gaps (interactive protocol, missing SAM-2) are meaningful. Compared to PRO-SCALE, RMP-SAM has a more novel problem formulation but weaker experimental validation of its core claim (that multi-purpose is beneficial).

The paper makes a genuine contribution in defining and benchmarking a new problem setting, and the proposed architecture is a reasonable baseline with clear improvements. The weaknesses are real but not fatal — the negative transfer is acknowledged and partially explained, and the interactive evaluation, while non-standard, still provides useful signal. The paper falls in the borderline range.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>