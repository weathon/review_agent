## Summary

SpatialBoost proposes a framework to enhance the spatial awareness of pre-trained vision encoders (e.g., DINOv3, SigLIPv2) by converting dense 3D spatial information from images into multi-turn chain-of-thought linguistic expressions, then injecting this knowledge via LLM-guided fine-tuning with a dual-channel attention mechanism that preserves pre-trained capabilities. The method is evaluated across depth estimation, segmentation, 3D scene understanding, robot learning, classification, and retrieval, showing consistent improvements.

## Strengths

- **Comprehensive empirical validation across diverse task families**: The paper demonstrates gains not only on spatial tasks (depth estimation RMSE: 0.31→0.25 for DINOv3 on NYUd; SQA3D: 51.4→54.9; Geometric Understanding RR@0.05m: 86.9→97.5%) but also on tasks not explicitly requiring spatial understanding (ImageNet linear probing: 88.4→90.2%), establishing that the method does not overfit to spatial features. The breadth of evaluation—covering dense prediction, 3D-centric benchmarks (Lexicon3D), robotic control (CortexBench), and instance recognition—is unusually thorough.

- **Dual-channel attention effectively mitigates catastrophic forgetting**: The ablation in Table 17 and Figure 6 provides clear evidence that full fine-tuning drops ImageNet classification from 86.3% to 79.5%, LoRA drops to 81.7%, while dual-channel attention maintains 87.6% *and* improves depth estimation. This is a practical and well-validated solution to the specific problem of preserving general visual knowledge while adding spatial specialization.

- **Hierarchical CoT reasoning order is empirically validated**: Table 7 shows that forward ordering (pixel→object→scene) outperforms reversed (0.34 vs 0.35 RMSE on depth) and random ordering, and Table 15 provides a detailed breakdown of which levels contribute most. This is not merely an intuitive design choice—it is substantiated by controlled ablations.

## Weaknesses

### Major:

- **Potential data leakage with ScanNet**: The multi-view VQA dataset construction (Appendix C) explicitly uses ScanNet (Dai et al., 2017) images to generate training QA pairs. Table 3 evaluates on ScanQA, SQA3D, and ScanRefer—all ScanNet-derived benchmarks. The paper does not state whether the ScanNet images used for training data generation are disjoint from the evaluation splits of these benchmarks. If there is overlap in scenes or images, the dramatic gains on 3D-centric tasks (e.g., SigLIPv2 3D SU mIoU jumping from 9.2 to 55.5, or OpenCLIP GU RR@0.05m from 22.6% to 78.8%) could be substantially inflated by memorization of scene layouts rather than genuine spatial reasoning generalization. This must be clarified; if overlap exists, these results are not trustworthy.

- **Table 6 ablation does not control for supervision head capacity**: The comparison between LLM (Qwen-7B) and Linear/SAM decoders as supervision heads conflates modality (language vs. pixel) with model capacity. The LLM has orders of magnitude more parameters than a linear layer or SAM decoder. A 7B-parameter decoder naturally provides richer gradient signals during backpropagation to the vision encoder. Without controlling for parameter count (e.g., comparing against a similarly-sized non-language decoder or analyzing gradient magnitudes), the conclusion that "language provides superior dense information transfer" is not well-isolated from "larger supervision networks provide better gradients." This undermines one of the paper's central claims.

### Minor:

- **The mechanism for improved 2D classification remains unexplained**: SpatialBoost improves ImageNet linear probing by 1.8% for DINOv3. The paper attributes this to dual-channel attention preserving pre-trained knowledge and scene captions providing general knowledge, but this does not explain why adding *spatial* specialization would *improve* a task where spatial reasoning is not the primary signal. An ablation isolating the spatial reasoning turns from the general scene caption turns would clarify whether the gain is from spatial understanding or incidental language alignment—this is partially addressed by Table 15 but not in a way that cleanly disentangles the two signals for classification specifically.

- **Dual-channel attention initialization at α=0.5 may destabilize early training**: With zero-initialized **a**, α = sigmoid(0) = 0.5, meaning the newly introduced Attn⁺ branch contributes 50% of the attention output from the very first training step. This is unusual compared to parameter-efficient methods like LoRA that initialize adaptation branches near-zero to start from the identity function. While the empirical results show this works, the choice lacks justification and seems risky—it may require careful learning rate tuning (which the paper acknowledges searching for) to avoid early destabilization.

- **No explicit discussion of limitations or the ceiling imposed by teacher models**: The paper does not discuss the computational overhead of the 3-stage pipeline (especially Stage 2 LLM fine-tuning) or the fundamental performance ceiling: SpatialBoost can only inject spatial knowledge as good as the upstream models (Depth-Pro, VGGT). Appendix F.5 shows marginal VFM-vs-GT differences on one dataset, but this does not establish robustness across domains where teacher models may fail (e.g., outdoor scenes, extreme viewpoints). A honest limitations section would strengthen the paper.

### Trivial:

- The paper uses "visual encoder" and "vision encoder" interchangeably throughout.

## Nice-to-Haves

- Comparison with spatial-specific baselines that also enhance spatial understanding (e.g., SpatialVLM applied as an encoder enhancement) to position the contribution more precisely.
- Ablation on LLM size (e.g., 3B vs. 7B) to determine whether spatial knowledge transfer scales with decoder capacity.
- A control experiment using the same data volume but non-spatial QA pairs, to isolate the contribution of *spatial* knowledge specifically versus general supervised training.
- Analysis of the learned α distribution across layers and training steps to verify that dual-channel attention behaves as claimed (gradual incorporation rather than abrupt switching).
- Failure case analysis showing where SpatialBoost underperforms the baseline.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Freezing contradiction in Section 4.2"**: The critic claimed Section 4.2's statement "All experiments freeze the visual backbone during training" contradicts Stage 3 fine-tuning. In context, Section 4.2 clearly refers to the downstream evaluation protocol (training linear/DPT probes on frozen features), not the SpatialBoost training itself. This is a misreading.

- **"Reproducibility concerns about API costs for GPT-4o"**: Flagging proprietary API usage as a reproducibility concern is not a valid weakness for this venue—using GPT-4o for data generation is standard practice, and the paper provides prompt templates in Appendix C.

- **"Broader impact / robot safety discussion missing"**: Demanding a broader impact discussion about robot safety for a representation learning paper is scope creep. The paper's contribution is about visual features, not deployed systems.

- **"Missing comparison to direct 3D pre-training"**: The paper explicitly scopes its contribution as enhancing existing 2D encoders without requiring large-scale 3D pre-training data. Requesting comparison to an entirely different training paradigm is outside the stated scope.

- **"Dataset scalability only tested in limited range"**: Figure 5 tests 50K–300K, which is a reasonable range given the computational constraints and the authors' resources. Requesting larger scales is a generic one-size-fits-all weakness.

- **"Formatting/style nitpicks"**: Removed per hard rules.

## Novel Insights

The hierarchical CoT reasoning structure reveals an interesting asymmetry: pixel-level QA disproportionately helps dense prediction tasks (depth, segmentation) while object-level QA disproportionately helps classification (Table 15). This suggests that different granularity levels of spatial supervision target different representational subspaces in the encoder—a finding that could inform future work on targeted spatial curriculum design. Additionally, the dramatic jump in SigLIPv2's 3D semantic understanding (6.9→55.5 mIoU in Table 3) compared to already-spatial-aware encoders like DINOv3 (69.1→70.6) suggests that language-aligned encoders have a uniquely large "spatial deficit" that linguistic injection can fill, possibly because their text-aligned features are already structurally compatible with language-guided spatial descriptions.

## Suggestions

- Explicitly state whether ScanNet images used in training data generation are disjoint from the ScanNet evaluation splits of ScanQA/SQA3D/ScanRefer. If there is overlap, re-run evaluation on held-out scenes; if there is no overlap, add a clarifying sentence in Section 3.2 or Appendix D to preempt this concern.

- In Table 6, add a capacity-controlled comparison: either use a non-language decoder of comparable parameter count to the LLM, or analyze gradient magnitudes/norms from different heads to disentangle capacity effects from language modality effects.

- Add a 2–3 sentence limitations paragraph discussing computational cost and the performance ceiling imposed by teacher model quality.

- Run a clean ablation on the classification improvement: train with multi-turn spatial reasoning data *without* the final 2 scene caption turns, and report ImageNet accuracy. This would clarify whether the classification gain comes from spatial reasoning or general language alignment.