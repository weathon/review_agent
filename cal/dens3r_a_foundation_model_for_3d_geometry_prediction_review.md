=== CALIBRATION EXAMPLE 34 ===

# Final Consolidated Review
## Summary

Dens3R proposes a unified feed-forward visual foundation model that jointly regresses multiple 3D geometric quantities—pointmaps, depth, surface normals, and image-pair matching—from unposed images. The core technical contributions are a two-stage training strategy that transitions from scale-invariant to "intrinsic-invariant" pointmap representations by incorporating normal supervision, a shared encoder-decoder backbone for efficiency, and position-interpolated rotary positional encoding (RoPE) to support high-resolution inputs.

## Strengths

- **Effective normal integration into pointmap regression:** The key insight—that surface normals provide an intrinsic, locally deterministic geometric property that can anchor and improve pointmap representations—is well-motivated and empirically validated. Table 1 shows consistent improvements over strong baselines (DSINE, StableNormal, GeoWizard) across five benchmarks, and Table 3 demonstrates that intrinsic-invariant training contributes measurably to these gains.

- **Practical high-resolution support:** The adaptation of position-interpolated RoPE from LLM context-window extension to 3D vision transformers addresses a real limitation of prior DUSt3R-based methods. Figures 8a and 21 provide convincing qualitative evidence that the model avoids the structural degradation (overlapping/inconsistent pointmaps) that occurs when extrapolating beyond training resolution, enabling inference at 2K resolution on a single RTX 3090.

- **Consistent multi-task improvement without specialization penalty:** Dens3R simultaneously achieves state-of-the-art on normal prediction (Table 1), image matching (Table 2), and competitive depth estimation (Table 7), while MASt3R—specifically optimized for matching—is outperformed on its own task. This suggests the unified representation genuinely captures geometric coupling rather than simply trading off task performance.

- **Architectural efficiency:** The shared encoder-decoder design reduces parameters from 737M to 624M and memory from 4.6 GB to 4.1 GB (Table 4), which is practically significant given that Dens3R outputs more geometric quantities than its predecessors.

## Weaknesses

- **Ambiguity in normal evaluation protocol:** Table 1 compares Dens3R against monocular normal estimators (DSINE, StableNormal, GeoWizard), but it is unclear whether Dens3R's evaluation uses strictly single-view inference or leverages the model's pair-based architecture (e.g., providing the same image as both inputs, or using a second view from the test set). Since the model natively processes image pairs with cross-attention, any multi-view cue during evaluation would constitute an unfair advantage over purely monocular baselines. The paper must explicitly state the inference protocol for Table 1. If pairs are used, a single-view ablation is needed to isolate the contribution of multi-view features.

- **Shared decoder ablation incomplete on accuracy:** Table 4 reports only compute, memory, and parameter savings for the shared encoder-decoder design but omits accuracy metrics. The text claims "without losing the prediction quality," but no numbers support this. The possibility that weight sharing degrades representational capacity—especially when the decoder must simultaneously serve pointmap regression, normal prediction, and matching—cannot be dismissed without evidence.

- **Quantitative depth results relegated to appendix:** For a paper claiming "unified geometric dense prediction" where depth is a primary output (listed in Eq. 1 and the abstract), burying quantitative depth comparisons in Appendix Table 7 weakens the central argument. The main text provides only qualitative Fig. 5 for depth/pointmaps while giving full quantitative tables for normals and matching. This asymmetry raises questions about whether depth results are comparatively weaker.

- **Multi-view consistency relies on external post-processing:** Section 3.3 describes the multi-view inference pipeline as relying on MASt3R's triangulation and SfM post-processing. The model itself operates on image pairs; global multi-view geometric consistency is not achieved by the model's internal representations but by an external optimization pipeline. This should be clearly stated upfront, as "geometrically consistent multi-view inference" (abstract) suggests the model itself ensures consistency.

- **Missing comparison against single-task specialist models:** The paper demonstrates Dens3R outperforms other methods on individual tasks but does not compare the unified model against training separate single-task models with the same backbone. Without this ablation, it remains unclear whether the "structural coupling" among tasks genuinely improves each task or whether the improvements stem primarily from better training data/strategies rather than multi-task synergy. The core claim that "explicitly modeling the structural coupling among different geometric properties" improves accuracy needs direct evidence that joint training helps beyond what independent training achieves.

## Nice-to-Haves

- A systematic evaluation across resolution ranges (512→1024→2048→4096) with quantitative metrics at each resolution to precisely characterize the position-interpolated RoPE benefit curve.

- Inference latency and FLOPs comparison against diffusion-based competitors (GeoWizard, StableNormal, Lotus) and DUSt3R-based methods (MASt3R, VGGT), to concretely quantify the efficiency advantage of regression-based unified prediction.

- A zero-shot cross-domain evaluation (e.g., train on indoor+synthetic, evaluate on outdoor benchmarks without any fine-tuning) to better support the "foundation model" framing.

- Ablation on training data composition (synthetic-heavy vs. real-heavy) to assess domain-gap sensitivity, given that Type A synthetic data constitutes ~60% of the training mix (Table 5).

## Removed Points

*These points are flagged to be removed, treat them with caution:*

- **Weakness: Training computational cost (32 H20 GPUs for 2 weeks) limits accessibility.** This is standard for foundation model training and falls under demanding practices impractical to change; large-scale training is expected in this area.

- **Weakness: "Foundation model" terminology demands scaling laws or emergent capabilities.** While the zero-shot evidence could be stronger, the paper demonstrates strong multi-task performance across diverse benchmarks. Demanding scaling law analysis is scope creep beyond the paper's stated contribution.

- **Weakness: Reproducibility of RoPE interpolation—frequency band details not specified.** The position interpolation formula (Eq. 2) follows the standard Chen et al. (2023) formulation cited in the paper. This constitutes a trivial implementation detail complaint.

- **Weakness: Broader impact / environmental cost discussion missing.** ICLR does not require this, and it is not a standard weakness for technical contributions.

- **Weakness: Equation notation ambiguity for masked point sets.** This is a formatting/style nitpick; the notation is consistent with prior DUSt3R/MASt3R papers.

- **Weakness: Lack of comparison between two-stage training and single-stage multi-task with gradient balancing.** The paper explicitly addresses this: "jointly training the pointmap and normal at the initial scale-invariant stage leads to instability and poor convergence" (Appendix A.1). While numerical evidence would strengthen this, the authors did consider and reject this alternative with justification, so demanding the ablation is weakened to a nice-to-have.

## Novel Insights

The bidirectional mechanism between pointmaps and normals is underexploited in the paper's presentation. The authors note that pointmap features help normals resolve monocular ambiguity (via cross-view information from Stage 1), while normals regularize and refine the 3D representation (Fig. 11). This reciprocal relationship suggests that the real contribution is not simply "adding normals to pointmaps" but establishing a geometric feedback loop where each representation fills the other's invariance gap—pointmaps lack intrinsic invariance (scale/shift ambiguity), while normals lack multi-view grounding (monocular ambiguity). The two-stage training can be understood as sequentially resolving these complementary weaknesses rather than merely progressively adding supervision.

## Suggestions

- Explicitly state in Section 4.1 whether Table 1 uses single-view or pair-wise inference, and add a single-view ablation if pairs are used, to definitively address fairness concerns.

- Add accuracy metrics (normal error, depth REL) to the shared decoder ablation (Table 4) to verify the claim that weight sharing does not degrade quality.

- Move depth quantitative results (Table 7) to the main text, or at minimum provide a summary table with key metrics, to support the "unified prediction" claim with equal evidential weight across all tasks.

- Add an ablation comparing the unified multi-task model against independently trained single-task models on the same backbone and data, to directly validate the coupling benefit claim.

- Clarify in the abstract and introduction that multi-view geometric consistency is achieved via post-processing rather than natively by the model, to set accurate expectations.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 4.0, 6.0]
Average score: 6.0
Binary outcome: Accept
