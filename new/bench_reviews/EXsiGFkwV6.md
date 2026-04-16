## Summary

Realistic-Gesture proposes a framework for co-speech gesture video generation that combines three components: (1) speech-aware gesture motion representation via contrastive alignment and knowledge distillation into an RVQ codebook, (2) a masked gesture generator with bidirectional decoding that enables long-sequence generation and editing, and (3) a structure-aware image refinement module using differentiable edge heatmaps to improve rendering fidelity in regions of large motion. Experiments on a 4-speaker subset of PATS demonstrate improvements over baselines in both motion and video quality metrics.

## Strengths

- **Well-motivated, modular pipeline**: Each of the three components addresses a specific, clearly identified limitation of prior warping-based methods (Sec. 3 identifies three challenges; Sec. 4 addresses each). The contrastive alignment → distillation → masked generation → refinement flow is logically coherent.

- **Substantial quantitative improvements on the chosen benchmark**: FGD drops from 23.646 (best baseline) to 1.303, Diversity improves from 10.848 to 13.260, and BAS reaches 0.996 (near GT's 1.00). FVD also improves consistently (Tab. 1).

- **Practical advantages of masked generation**: The iterative decoding requires only 5 inference steps (Tab. 3f) compared to 50–100 for diffusion models, and the bidirectional nature enables gesture editing and inpainting (Sec. 5.5), which is a tangible capability gain.

- **Thorough ablation study**: Six ablation groups (Tab. 3) cover keypoint design, gesture representation, generator architecture, image refinement, mask ratio, and inference steps, providing useful design insights.

- **Edge heatmap refinement is technically interesting**: The differentiable edge connection formulation (Eqs. 5–7) is clearly presented and demonstrably improves hand/shoulder rendering over simpler alternatives (Tab. 3d), with visual evidence in Fig. 5.

## Weaknesses

### Major:

- **Evaluation is limited to 4 speakers from a single dataset, raising generalization and overfitting concerns**: All experiments use only Noah, Kubinec, Oliver, and Seth from PATS. The extremely low FGD of 1.303 (vs. GT at 0.0) on ~400 test clips from 4 speakers is suspiciously close to ground truth, suggesting possible memorization of speaker-specific patterns rather than generalizable speech-gesture alignment. No cross-speaker or cross-dataset evaluation is provided. Similar concerns were raised for other gesture generation papers evaluated on narrow speaker sets (e.g., TANGO reviewer: "difficult to verify whether the proposed approach would outperform other methods in different circumstances").

- **Semantic alignment claims are not supported by appropriate evaluation**: The paper's central novelty claim is "speech-aware gesture representation" that captures "implicit intentions conveyed in the audio" and produces "metaphoric gestures" (Sec. 1, 4.1). However, the only alignment-related metrics are BAS (beat timing) and PCM (deviation from ground truth pose)—both low-level motion statistics. No experiment tests whether gestures change appropriately when speech *semantics* change while prosody stays similar, or ablates the contrastive alignment to isolate semantic vs. rhythmic contributions. The qualitative examples ("90 joules" = pointing) are anecdotal and not systematically verified. This gap between claim and evidence is significant.

- **Limited and asymmetric baseline comparisons**: Only 3 baselines are compared, and MM-Diffusion is a general audio-video model not designed for co-speech gestures—acknowledged by the authors who note its fixed 34-frame limitation. Recent co-speech gesture generation methods discussed in related work (e.g., Make-Your-Anchor, EMO, AnimateAnyone) are not benchmarked, and no two-stage pose+rendering pipeline is included despite being the dominant alternative paradigm for high-fidelity avatar generation. This leaves the claimed state-of-the-art status on uncertain footing.

- **VQA metrics exceeding ground truth are concerning**: VQA_A=96.326 (GT: 95.694) and VQA_T=6.081 (GT: 5.329). The paper attributes this to "structure-aware image enhancement design," but this actually suggests the refinement module may be producing artificially sharpened/denoised frames that the VQA models prefer over naturally textured ground truth, rather than generating more realistic gesture videos. This anomaly deserves explicit discussion rather than being presented as an achievement.

- **Diversity contradiction between metrics and user study**: Quantitative Diversity (13.260) outperforms S2G-Diffusion (10.848), yet the user study shows MOS₂ for diversity is 3.05 vs. S2G-Diffusion's 3.6. This discrepancy—only briefly noted as "lower in diversity than S2G-Diffusion" without explanation—undermines confidence in the feature-space diversity metric and suggests the generated gestures may be more repetitive than the numbers suggest.

### Minor:

- **Applications (Sec. 5.5) are only qualitatively demonstrated**: Long-sequence generation, gesture editing, and pattern transfer are shown in Fig. 6 without any metrics on temporal coherence, editing fidelity, or user preference. These capabilities are promising but unquantified.

- **Missing inference time comparison**: The paper explicitly motivates masked prediction as faster than diffusion (Sec. 4.2: "these strategies hinder the fast synthesis for real-time applications"), but provides no wall-clock latency, FLOPs, or throughput numbers for any method.

- **Confounding between motion quality and image quality in evaluation**: Better image rendering (from edge-heatmap refinement) likely produces cleaner pose detections, indirectly boosting motion metrics (FGD, BAS, PCM). The motion-vs-image quality ablations (Tab. 3a vs. 3d) are separate rather than joint, so this confound is not controlled.

- **Masked generation's key hyperparameters are underspecified in the main paper**: Threshold details for the iterative remasking, masking schedules, and model capacity are deferred to the appendix, making it difficult to assess the core contribution from the main text alone.

### Trivial:

- The phrase "groundbreaking framework" in the abstract is overclaiming for what is a solid but incremental integration of existing techniques.

## Nice-to-Haves

- Cross-speaker and cross-dataset evaluation (e.g., on BEAT or Trinity) to demonstrate generalizability.
- Quantitative evaluation of long-sequence generation (FGD/FVD at increasing durations) to verify no temporal drift.
- Ablation that isolates semantic alignment vs. beat-only conditioning (e.g., replace speech encoder with a beat-only signal) to validate the "speech-aware" claim.
- Inference time benchmarks across all methods.
- Failure case analysis showing where the method breaks down.

## Removed Points

- **Missing hyperparameters (learning rate, batch size, etc.) as a weakness**: The reviewers flag this as reducing reproducibility, but per the rules, undisclosed hyperparameters and trivial implementation details are not valid weaknesses—they are impractical to include in a submission.
- **"Groundbreaking framework" overclaiming**: While the language is inflated, this is stylistic. The substantive version of this point (that semantic claims outpace evidence) is retained above.
- **Concerns about training data leakage between pose features and RVQ codebook**: This is speculative without evidence—the reviewers hypothesize potential leakage but provide no concrete indication it occurs.
- **Demand for comparison with AnimateAnyone/Champ/EMO as missing baselines**: While including these would strengthen the evaluation, these are general video/avatar generation models not specifically targeting the co-speech gesture pipeline, and the paper explicitly argues they are "slow in inference speed" and "based on large amount of training data"—a scope argument that, while debatable, is reasonable.

## Novel Insights

The paper demonstrates that distilling speech-gesture contrastive alignment features into an RVQ codebook (rather than using alignment only at inference via cross-attention) yields a large FGD improvement (21.47 → 1.30 in Tab. 3b). This suggests that making the discrete token space itself speech-aware—rather than relying solely on conditional generation with speech as input—is an effective design principle for speech-conditioned discrete motion generation. This idea of infusing cross-modal alignment into tokenizer codebooks rather than only conditioning the generator deserves further investigation in other domains.

## Suggestions

- Add a cross-speaker generalization experiment to address the most significant evaluation concern.
- Add a beat-only vs. full-speech ablation to substantiate the "speech-aware" claim beyond beat timing.
- Discuss the VQA-exceeds-GT anomaly explicitly, ideally with analyses showing whether the refinement module produces perceptually "enhanced" but not necessarily more realistic outputs.
- Provide inference time numbers for all methods to support the "fast synthesis" claim.
- Reconcile the diversity metric contradiction with the user study through explicit discussion or additional analysis.

## Score and Decision

**Calibration anchors:**

- **TANGO** (LbEWwJOufy): Scores 8/8/8/10, Accept Oral. Strong novel contribution with rigorous evaluation, though also on limited speakers. Clearly stronger than the current paper.
- **CyberHost** (vaEPihQsAA): Scores 8/10/6/6/8, Accept Oral. Novel real-time framework with strong engineering. Stronger evaluation scope.
- **IMAC** (kmhNK0fs8c): Scores 3/5/5/3, Withdrawn/Reject. Limited baselines, repetitive gestures, overclaiming. The current paper is clearly stronger than IMAC.
- **CoCoGesture** (g3kK6YBSZ1): Scores 6/3/3, Withdrawn/Reject. Over-smoothed, limited novelty. The current paper has more technical substance.
- **AV-SDS** (CYUIeEBri1): Scores 5/3/3/6/5, Withdrawn/Reject. Limited evaluation scope, weak baselines.

The current paper has genuine technical contributions (distillation into codebook, masked generation for gestures, edge heatmap refinement) with substantial metric improvements. However, the evaluation is narrow (4 speakers, 3 baselines with one weak), the central semantic claim is unsupported, VQA exceeds GT, and the diversity metrics contradict the user study. These are significant but not fatal issues—it is a solid engineering contribution with overstated claims. It sits below TANGO/CyberHost (clear accept) but above IMAC/CoCoGesture/AV-SDS (clear reject).

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>