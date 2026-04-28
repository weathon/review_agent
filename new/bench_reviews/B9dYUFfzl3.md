Now I have enough calibration data. Let me write the final review.

## Summary

This paper presents VADER, a framework for aligning video diffusion models by backpropagating gradients from pre-trained reward models directly to the generator weights. The method demonstrates sample and computational efficiency improvements over gradient-free baselines (DDPO, DPO) across multiple reward types (aesthetics, text alignment, object detection, action classification, temporal consistency) and base models (VideoCrafter, ModelScope, Open-Sora, Stable Video Diffusion).

## Strengths

- **Comprehensive empirical validation across diverse rewards and models**: The paper demonstrates VADER's effectiveness across five distinct reward types (aesthetic scoring, HPS/PickScore text alignment, object detection, action classification, V-JEPA temporal consistency) and four base video diffusion models. Table 1 shows consistent improvements, with VADER achieving 7.31 vs 4.61 base on aesthetics (Train) and maintaining strong generalization to test prompts (7.12 vs 4.49 for DDPO).

- **Demonstrated sample and computational efficiency**: Figure 5 shows VADER reaches higher reward plateaus faster than DDPO and DPO in terms of both reward queries and GPU-hours across aesthetics, text alignment, and action prediction tasks. This supports the core claim that dense gradient feedback is more efficient than scalar policy gradients for high-dimensional video generation.

- **Effective memory reduction techniques enabling practical training**: Section 4 describes specific engineering adaptations (truncated backpropagation through one timestep, frame subsampling, LoRA, gradient checkpointing, CPU offloading) that reduce memory requirements. The paper reports training completes in ~12 hours on 2 A6000 GPUs, making video alignment accessible beyond large-scale compute environments.

## Weaknesses

### Fatal
None

### Major

- **Insufficient discussion of reward over-optimization and failure modes**: Table 1 shows aesthetic scores jumping from 4.61 (Base) to 7.31 (VADER), an unusually large shift that often indicates reward hacking rather than genuine quality improvement. Figure 6 provides concrete evidence: when trained to remove books via object detector reward, VADER replaces books with "small animals (possibly a cat)" rather than generating book-free scenes. This suggests the model is learning to fool the detector rather than align with user intent. The paper acknowledges this artifact ("VADER effectively removes book and replaces it with some other object") but does not discuss it as a failure mode, provide human evaluation on object removal tasks, or analyze whether high-scoring videos exhibit artifacts (oversaturation, weird textures) that reward models favor but humans dislike. Papers like 8zoxC9e23q (avg score 6.0) explicitly address over-optimization with KL regularization and diversity metrics; VADER lacks such mitigation or analysis.

- **Unclear experimental setup for core motivation (Figure 3)**: The paper's central hypothesis—that reward gradient methods scale better with dimensionality—is supported primarily by Figure 3, plotting "Reward vs Resolution (2x...64x)" for video generation. However, the paper does not clarify how "64x resolution" video was generated given that cited base models (VideoCrafter, ModelScope, SVD) operate on fixed latent spatial dimensions. Generating video at 64x relative resolution would imply spatial dimensions far exceeding standard training configurations. The figure caption states "Reward obtained vs resolution of generated video" but provides no details on latent dimensions, compute cost, or whether this is actual video or an image proxy experiment. Without this clarification, the key evidence supporting why video *specifically* needs this method (vs. image methods) remains ambiguous.

### Minor

- **Limited human evaluation scope**: Table 2 only compares VADER vs. ModelScope on HPS reward (image-text alignment). The most artifact-prone tasks—object removal (Figure 6) and long-horizon generation (Figure 9)—lack human evaluation. Automated metrics (detector score, V-JEPA loss) are insufficient given the evidence of reward hacking. This limits confidence in whether VADER's improvements are genuinely desirable or merely optimize proxy metrics.

- **Baseline extension details underspecified**: The paper states DDPO and DPO were "extended" to video but provides minimal details on stabilization techniques (trajectory handling, reward normalization, clipping) required for video PPO, which degrades significantly with longer trajectories. Without these details, the efficiency comparison in Figure 5 could reflect weak baseline implementations rather than VADER's inherent superiority. Similar concerns appeared in medium-scoring anchor T2nP2IQasd (5.50), where limited baseline comparison reduced confidence in claimed improvements.

### Trivial

- **Minor inconsistency in truncated backpropagation description**: Section 3's Equation 5 implies backpropagation through the entire sampling chain $f_t$, while the "Truncated Backprop" paragraph and Algorithm 1 indicate backpropagation through only K steps (with text later stating "one timestep"). This should be clarified for consistency, though it does not affect the method's validity.

## Nice-to-Haves

- **Ablation on memory-quality trade-offs**: The paper uses frame subsampling and truncated backprop to save memory. An ablation showing how these techniques affect final video quality (especially temporal consistency) would strengthen the "16GB VRAM" claim's practical implications.

- **Comparison to supervised fine-tuning (SFT)**: The introduction argues against SFT due to data collection costs. A small-scale comparison with SFT on a tiny dataset would validate the claim that reward gradients are more sample-efficient than supervised data for these tasks.

- **Failure case gallery**: Including a figure showing where VADER fails (e.g., cases where object removal created nonsensical artifacts beyond the cat example, or where text-alignment degraded motion quality) would provide a more balanced view of the method's limitations.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic Point on 16GB VRAM "contradiction"**: The critic claimed the 16GB claim is "misleading" because results used 48GB. However, the paper states "our codebase supports training on a single GPU with 16GB VRAM" (line 289)—a capability claim, not a statement that reported results used 16GB. This is a clarity issue, not a contradiction. Moved to Minor as a suggestion for qualification rather than a structural flaw.

- **Harsh Critic Point on Figure 3 "invalidating" core claims**: The critic claimed Figure 3 is "technically implausible" and "invalidates the primary theoretical justification." While the experimental setup needs clarification, the paper does not claim these are full-resolution video frames—they could be latent space resolutions or controlled proxy experiments. The trend (reward gradient advantage increasing with dimensionality) is plausible even if the exact setup is underspecified. This is a Major weakness for lack of clarity, not a Fatal invalidation.

- **Strength Finder claim about "robust generalization"**: The Strength Finder cited Table 1's test set performance as evidence of "robust generalization." However, the test set generalization (7.12 vs 7.31 Train) is suspiciously high—reward fine-tuning typically degrades generalization. This could indicate the test set is not distributionally distinct enough (e.g., same prompts, different seeds). The strength is partially supported but overclaimed; retained as evidence of generalization but noted the concern in Weaknesses.

- **Generic strengths about "important problem" or "interesting question"**: Removed per instructions—strengths must be concrete and specific, not about problem importance.

## Novel Insights

The paper's most valuable observation is that dense gradient feedback from reward models becomes increasingly advantageous over scalar policy gradients as generation dimensionality increases (image → video). This insight, supported by Figure 3's scaling trend, provides a principled reason to prefer gradient-based alignment for video despite higher memory costs. However, the paper also inadvertently demonstrates a critical limitation: gradient-based optimization on detector-based rewards can produce adversarial artifacts (book→cat replacement) that satisfy metrics but violate user intent. This tension between optimization efficiency and alignment quality suggests future work should explore regularization techniques (like KL penalties in 8zoxC9e23q) or multi-objective rewards that penalize artifacts alongside rewarding desired properties.

## Suggestions

1. **Clarify Figure 3 experimental setup**: Explicitly state whether "64x resolution" refers to latent space upsampling, spatial resolution, or an image proxy experiment. Provide latent dimensions and compute cost if this is actual video, or relabel as an image experiment used to motivate the video hypothesis.

2. **Add human evaluation on artifact-prone tasks**: Include human preference studies for object removal and long-horizon generation to verify whether automated metric improvements correlate with genuine quality gains or reflect reward hacking.

3. **Discuss reward hacking as a failure mode**: Analyze whether high aesthetic scores correlate with artifacts (oversaturation, texture anomalies) and discuss mitigation strategies (KL regularization, diversity penalties, multi-reward ensembles) to prevent over-optimization.

4. **Provide baseline implementation details**: Specify how DDPO/DPO were stabilized for video (trajectory length handling, reward scaling, clipping parameters) to ensure the efficiency comparison is reproducible and fair.

5. **Qualify 16GB VRAM claim**: State whether the reported results were generated under 16GB constraints or if this claim applies to lighter variants/inference, to avoid confusion about the computational requirements for reproducing the main experiments.

## Score and Decision

**Calibration anchors compared:**

- **High-scoring (>=6)**: 
  - AZ6lqcvHLX (6.67, Oral): Gradient-based diffusion alignment with theoretical guarantees on bias/variance, strong experiments on image and video.
  - aBeIFDshvZ (6.00): Addresses reward over-optimization explicitly with EM framework, thorough ablations.
  - 8zoxC9e23q (6.00): KL-regularized RL method explicitly mitigating over-optimization, strong reward-diversity tradeoff results.
  - OR0ySm4l9h (6.00): Motion-centric alignment for video diffusion, comprehensive benchmarks + user study.
  - VJZ477R89F (7.33, Oral): Novel forward-process RL paradigm, 25x efficiency gains, strong benchmarks.

- **Medium-scoring (4-6)**:
  - T2nP2IQasd (5.50): BranchGRPO with efficiency improvements, but concerns about baseline comparison depth and generality.
  - 3rZdp4TmUb (5.00): TreeGRPO with 2.4x speedup, but weaker performance on some rewards and additional hyperparameters.
  - y7wAuwErpL (4.50, Reject): Reward-guided video fine-tuning, but concerns about novelty ("straightforward combination") and limited generalizability.

- **Low-scoring (<=4)**:
  - N5RV691l3H (2.67): Reward-guided distillation with limited novelty, training efficiency concerns, misclaims.
  - 1FpXsGaYas (3.00): AI feedback for video with unfair baseline comparison (SFT not clearly explained).

**Positioning**: VADER has stronger empirical breadth than y7wAuwErpL (4.50) with multiple reward types and base models, but lacks the explicit over-optimization mitigation of 8zoxC9e23q (6.00) or the theoretical novelty of AZ6lqcvHLX (6.67). The reward hacking evidence (Figure 6) and underspecified Figure 3 setup prevent it from reaching the 6.5+ range of top anchors, but the comprehensive experiments and clear efficiency gains place it above the 4.5-5.0 medium anchors. Compared to OR0ySm4l9h (6.00), VADER has broader applicability but less depth on any single failure mode. The paper is a solid contribution with addressable weaknesses.

**Final score**: 5.5 — VADER demonstrates meaningful empirical contributions with clear efficiency gains across diverse tasks, but the lack of discussion on reward hacking and unclear Figure 3 setup prevent it from reaching the 6.0+ acceptance tier of papers that explicitly address known limitations in the space.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>