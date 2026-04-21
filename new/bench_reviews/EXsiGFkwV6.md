Now I have all the information needed. Let me compile the final review.

## Summary

Realistic-Gesture proposes a framework for co-speech gesture video generation with three components: (1) speech-aware gesture tokenization via contrastive alignment and knowledge distillation into an RVQ codebook, (2) a masked gesture generator using iterative mask prediction for fast inference (5 steps), and (3) a structure-aware image refinement module using differentiable edge heatmaps. The method demonstrates substantial quantitative improvements over prior baselines on the PATS benchmark.

## Strengths

- **Well-structured pipeline with complementary components**: The three-module design (speech-aware tokenization → masked generation → edge heatmap refinement) cleanly addresses the identified limitations of prior work (unsupervised keypoints, speech disconnection, blurry hands/shoulders). Each component is ablated independently (Table 3a–f), showing clear contributions.

- **Fast inference via iterative masked prediction**: The 5-step iterative decoding (Table 3f) is significantly faster than diffusion-based alternatives (50–100 steps), which is a genuine practical advantage for real-time applications. This is supported by the ablation showing minimal quality degradation from 5 to 20 steps (FGD 1.303→1.881).

- **Comprehensive ablation study**: Table 3 provides six distinct ablation dimensions—keypoint design (3a), gesture representation (3b), generator architecture (3c), refinement strategy (3d), mask ratio (3e), and inference steps (3f)—each with multiple configurations, making every design choice clearly quantifiable.

- **Strong quantitative improvements over existing baselines**: Table 1 shows improvements across all metrics: FGD 1.303 vs. 23.646 (S2G-Diffusion), Diversity 13.260 vs. 10.848, BAS 0.996 vs. 0.974, PCM 0.572 vs. 0.447. Even on the more conservative PCM metric, the method leads by a meaningful margin.

- **Multiple downstream applications enabled by design**: The masked token representation naturally supports long-sequence generation, gesture editing, and pattern transfer (Figure 6), which prior methods like ANGIE and S2G-Diffusion cannot support.

## Weaknesses

### Fatal
None.

### Major

- **The central claim of "speech-content-aware" gestures lacks quantitative evaluation**: The paper's primary contribution is stated as a "speech-aware gesture motion representation" that enables "conditional generation of gestures that accurately reflect the speaker's intended meaning based on the speech input" (Sec. 4.1, Introduction). The method specifically claims to fuse "semantics and contextual triggers from speech (e.g., pronouns like 'this' or 'they')" into motion embeddings (Sec. 4.1). Yet none of the quantitative metrics—FGD, Diversity, BAS, PCM—measure whether gestures *semantically* correspond to speech content. BAS captures only rhythmic/beat alignment. The "w/o align" ablation (Table 3c) shows FGD degrading from 1.303 to 8.382, but this only demonstrates that the contrastive pre-training helps gesture quality—it does not isolate whether *semantic* speech content is reflected in the generated gestures versus just better audio-gesture temporal alignment. The qualitative examples (Fig. 4, pointing when saying "90 joules") are anecdotal. Without a metric or controlled experiment that specifically evaluates semantic gesture appropriateness (e.g., gesture-type classification accuracy against speech-predicted expectations), the core novelty claim remains unsupported by evidence.

- **FGD improvement may be inflated by VQ-VAE structural advantage; FGD-PCM discrepancy is unexplained**: The reported FGD of 1.303 (vs. GT = 0.0) is dramatically better than S2G-Diffusion's 23.646, yet PCM—frame-level motion accuracy—is only 0.572 (43% of parameters incorrect). This discrepancy is not discussed. As the paper states, FGD and Diversity are "utilize an auto-encoder trained on PATS poses" (Sec. 5.2). Since the method's VQ-VAE codebook constrains outputs to poses drawn from the training manifold, this gives a structural advantage on FGD (a distribution-level metric computed in the feature space of an auto-encoder trained on the same data). The distillation ablation (Table 3b) reinforces this concern: adding distillation drops FGD from 21.473 to 1.303 (a 94% reduction) while PCM improves only from 0.412 to 0.582 (a 41% relative improvement). The method clearly improves on both metrics, but the magnitude of FGD improvement relative to PCM suggests that FGD disproportionately rewards the VQ-VAE's manifold constraint rather than reflecting genuine gesture quality gains. The paper should discuss this and provide analysis of what FGD captures in this setup.

- **Missing contemporary baselines for the video generation claim**: The paper's title promises "co-speech gesture video generation" and claims "pixel-level realism," yet only three baselines are compared: ANGIE (2022), MM-Diffusion (forced to 34-frame segments), and S2G-Diffusion. The related work section discusses Make-Your-Anchor, AnimateAnyone, Champ, and EMO—all recent methods for pose-driven or audio-driven avatar video generation—but dismisses them as "slow in inference speed" and claiming "None of them focus on the speech-gesture pixel-level video generation" (Sec. 2). However, Make-Your-Anchor directly addresses person-specific avatar video generation with disentangled face/body control—a closely related problem. If the claim is state-of-the-art video quality, at least one of these methods should be included as a baseline, even if the comparison requires adaptation. Without them, the video quality superiority claim is under-supported.

### Minor

- **VQA scores exceeding ground truth are not properly discussed**: The method achieves VQA_A = 96.326 and VQA_T = 6.081, surpassing ground truth (95.694 and 5.329). The paper attributes this to "our structure-aware image enhancement design" (Sec. 5.2), but generated video outscoring ground truth on quality metrics is a well-known artifact of smoothing/post-processing that reduces noise at the cost of fine detail. This should be acknowledged as a potential limitation rather than celebrated as unambiguous success.

- **Diversity metric contradicts user study**: The quantitative Diversity score (13.260) exceeds S2G-Diffusion's (10.848), but in the user study, participants rate the method lower on diversity (MOS₂: 3.05 vs. 3.6). This discrepancy is not discussed. Since perceptual diversity and feature-space diversity can diverge, an analysis of what drives this gap would strengthen the evaluation.

- **Claimed applications (long sequence, editing, transfer) are only qualitatively evaluated**: These are highlighted in the abstract, Figure 1, and Section 5.5, but demonstrated only through Figure 6 with no quantitative evaluation. While qualitative demonstrations are informative, some basic metrics (e.g., temporal consistency for long sequences, editing fidelity) would strengthen these claims.

### Trivial
None.

## Nice-to-Haves

- A semantic alignment evaluation protocol: annotate a subset of test clips with expected gesture types (deictic, metaphoric, beat, iconic) given the speech content, and measure classification accuracy of generated gestures against these expectations. This would directly validate the paper's central claim.
- Evaluation on a second dataset beyond the 4-speaker PATS subset to demonstrate generalization.
- Quantitative evaluation of the application scenarios (long sequence generation, editing, transfer).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic: "ANGIE is outdated / MM-Diffusion is a strawman"**: ANGIE is the most directly comparable method for the same task and was the prior SOTA; MM-Diffusion is included for breadth. Dismissing them as unfair baselines is incorrect because the asymmetry actually favors the baselines, not the authors' method.

- **Harsh critic: "The mask prediction is identical to MAGE/Muse, not novel"**: The paper clearly positions this as inspired by MAGE/Muse (Sec. 4.2), not as a novel architecture. The contribution is the *application* to gesture tokens with the specific two-generator (base + residual) design. Claiming zero novelty for a known paradigm applied to a new domain overstates the criticism.

- **Harsh critic: "The 30% temporal masking is not ablated"**: The paper describes the masking as part of the contrastive learning design and defers details to the appendix. While an isolated ablation would be nice, this is a minor presentation issue.

- **Harsh critic: "Edge heatmap improvement is modest (~3% FVD)"**: The refinement module's primary contribution is improved VQA scores and visual quality in hand/shoulder regions (Fig. 5), not FVD. The VQA_A improvement (91.2→96.3) is meaningful, and FVD is not the primary target of this module.

- **Harsh critic: "User study realism gap (3.35 vs 4.7 GT) is substantial"**: A gap between generated and ground-truth realism is expected and present for all methods. The relevant comparison is against other generated methods (3.35 vs. 3.0 for S2G-Diffusion), where the method leads.

- **Strength finder: "VQA surpassing GT" as a strength**: This is actually a concern (moved to Minor weaknesses). Generated video outscoring GT on quality metrics typically indicates smoothing artifacts.

- **Strength finder: "User study validates perceptual quality" as unqualified strength**: The user study shows mixed results—winning on realism/synchronization but losing on diversity. The strength should be qualified.

## Novel Insights

The FGD-PCM discrepancy reveals an important underappreciated issue in gesture generation evaluation: when using VQ-VAE-based tokenization, distribution-level metrics like FGD (computed via an auto-encoder trained on the same data) can be disproportionately favorable because the VQ-VAE codebook constrains outputs to the training manifold. This is not unique to this paper—it affects any method using discrete tokenization with FGD—but this paper's dramatic FGD improvement (21.473→1.303 via distillation) relative to the modest PCM improvement (0.412→0.582) provides the clearest illustration of this problem in the co-speech gesture literature. The community should develop evaluation protocols that are less susceptible to this artifact, such as using feature extractors trained on different data or emphasizing frame-level metrics.

## Suggestions

- Add a semantic alignment evaluation: even a small-scale human annotation of gesture types (deictic/metaphoric/beat/iconic) on 50–100 test clips, correlated with speech content, would substantially strengthen the central claim.
- Discuss the FGD-PCM discrepancy explicitly, including an analysis of how the VQ-VAE codebook affects FGD computation. Consider reporting FGD using a feature extractor not trained on the same codebook.
- Add at least one recent pose-driven video generation baseline (e.g., Make-Your-Anchor) to validate the pixel-level quality claim, even if it requires task adaptation.
- Acknowledge that VQA scores exceeding GT may reflect smoothing artifacts rather than genuine quality improvement.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| TANGO | /home/wg25r/review_agent/human_reviews/LbEWwJOufy.md | 8.50 | Same task (co-speech gesture video), much deeper audio-gesture alignment analysis (AuMoClip with explicit temporal+semantic evaluation), comprehensive baselines. Realistic-Gesture is notably weaker on evaluation rigor. |
| CyberHost | /home/wg25r/review_agent/human_reviews/vaEPihQsAA.md | 7.60 | Similar domain (audio-driven talking body), comprehensive experiments including video-driven and hybrid settings. Realistic-Gesture has weaker baselines and unverified core claim. |
| Synergizing Motion & Appearance | /home/wg25r/review_agent/human_reviews/LDtNetvNQp.md | 5.25 | Similar codebook-based approach for talking head video, moderate quality with some evaluation gaps. Realistic-Gesture is comparable in contribution level but has more serious evaluation concerns. |
| CoCoGesture | /home/wg25r/review_agent/human_reviews/g3kK6YBSZ1.md | 4.00 | Directly on co-speech gesture generation with scaling approach but limited evaluation. Realistic-Gesture is stronger in pipeline completeness and ablation depth. |
| VideoDiT | /home/wg25r/review_agent/human_reviews/lvgsPjRtLM.md | 2.50 | Low-quality video generation paper with unclear method and poor novelty. Realistic-Gesture is clearly above this. |
| Blind Baselines Beat MI | /home/wg25r/review_agent/human_reviews/BXMoS69LLR.md | 4.50 | Strong empirical results but overclaimed contribution and metric concerns. Realistic-Gesture shares a similar pattern—strong numbers but unsupported central claim. |

Realistic-Gesture sits between the medium-scoring anchors (4–5.25) and the high-scoring TANGO/CyberHost (7.6–8.5). It has real contributions—a well-designed pipeline, comprehensive ablations, and practical speed advantages—but its central claim about semantic speech-gesture alignment is unsupported by evaluation, the FGD metric reliability is questionable, and the baseline comparison is narrow. Compared to CoCoGesture (4.0), it is stronger in pipeline completeness and ablation depth. Compared to Synergizing Motion & Appearance (5.25), it has similar evaluation gaps but more serious issues with the FGD interpretation. I place it slightly below 5.25, as the unverified core claim is a more fundamental problem than the evaluation gaps in the Synergizing paper.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>