## Summary

4K4DGen proposes a framework to elevate a single 4K panoramic image into a 4D dynamic environment. It consists of two stages: (1) a **Panoramic Denoiser** that adapts pre-trained perspective image-to-video diffusion models to animate 360° panoramas by projecting spherical latents to perspective views, denoising them independently, and fusing them back at each step; and (2) **Dynamic Panoramic Lifting** that converts the panoramic video into time-dependent 3D Gaussians with spatial-temporal geometry alignment for consistent 4D representation. The paper claims to be the first to achieve 4K-resolution panoramic 4D generation.

## Strengths

1. **Novel and important problem formulation.** Generating immersive 4D panoramic environments from a single image addresses a real need for VR/AR content creation. The task of panoramic 4D generation at 4K resolution is genuinely underexplored and practically relevant.

2. **Clever adaptation strategy for the Panoramic Denoiser.** The project-denoise-fuse approach (Eq. 3, Fig. 2) is a principled way to leverage well-trained perspective diffusion priors for panoramic domains while maintaining cross-view consistency. The ablation in Table 2 shows it achieves 70% view-consistency vs. 33% for naive per-perspective animation, and the qualitative comparison in Fig. 5 convincingly demonstrates the failure modes of alternative approaches.

3. **Coherent two-stage pipeline design.** The decomposition into animation and lifting stages is well-motivated by the scarcity of panoramic 4D training data. The spatial-temporal geometry alignment for depth fusion and temporal regularization for 4D lifting are reasonable engineering contributions.

4. **First demonstration at this scale and format.** Producing panoramic 4D content at 4096×2048 resolution is a meaningful technical milestone, even if the effective resolution of the underlying perspective guidance is limited.

## Weaknesses

### Fatal

None.

### Major

1. **Evaluation protocol is misaligned with the core claims.** The paper claims "4D immersive environment," "spatial-temporal consistency," "free-viewpoint 360° virtual views where users can move in all directions," and "6-DoF virtual tours." However: (a) Training and evaluation cameras are all at position p=0 (origin); the only novel views differ by direction d, with a perturbation δ_p ∈ [-0.05, 0.05]³ used solely as a regularizer, not for evaluation. The paper does not demonstrate that the representation supports any meaningful camera translation. (b) The sole baseline is 3D-Cinemagraphy, an optical-flow-based 2.5D method not designed for panoramic 4D generation, making the superiority claim nearly tautological. No comparison with adapted video-to-4D methods or even static 360° GS reconstruction plus per-frame appearance modulation is provided. (c) Only 16 synthetically generated panoramas are evaluated, with no real-world panoramas, and quantitative evaluation relies exclusively on no-reference Q-Align metrics and user preference studies, with no temporal consistency, geometric accuracy, or multi-view consistency metrics.

2. **The Panoramic Denoiser (core contribution #1) is underspecified.** Eq. 3 defines the denoising procedure as argmin_S E_{d∈S²} ||γ(S,d) − Φ(γ(S^t,d), γ(I,d))||, but the optimization procedure is never made concrete. Is this solved analytically (e.g., weighted averaging in overlapping regions), iteratively via gradient descent, or some other method? How is the continuous field S parameterized — as an equirectangular grid, spherical harmonics, or something else? How are overlapping projections fused? The paper mentions "20 directions" in Sec. 4.1 but never specifies how these samples approximate the expectation over S² or how conflicts between overlapping perspective views are resolved. As one of two headline contributions, this is insufficient for reproducibility and makes it impossible to attribute the benefits to a specific, well-defined mechanism versus simple averaging.

3. **The 4K resolution claim is misleading about effective resolution.** Section 4.1 specifies 512×512 perspective views for denoising. The 4K resolution refers to the input/output panorama resolution, but the actual generative guidance operates at much lower resolution. No analysis is provided on how much high-frequency detail is preserved or lost through the projection-fusion pipeline — whether the final 4K output genuinely contains 4K-level detail or is essentially upscaled from lower-resolution guidance. This directly undermines the "4K resolution" framing in the title.

### Minor

1. **Temporal loss formulation biases toward static scenes.** Eq. 6 defines L_temporal = Σ_i ||R(G_t, 0, d_i) − R(G_{t-1}, 0, d_i)||, which penalizes any per-pixel difference between consecutive frames. This pushes the model toward static outputs unless counteracted by RGB fitting to the video. The ablation in Fig. 6 only shows that removing it causes "flashing stripes" (suggesting it mostly damps noise), but there is no analysis of whether it suppresses genuine dynamic content or how the balance between motion and stability is controlled.

2. **Depth alignment formulation has ambiguities.** In Eq. 4, the inner "min_i" across views is odd — it suggests optimizing the fused depth to match only one view at each location rather than being consistent across overlapping views. The norm type (L1, L2) and domain of the loss terms are unspecified.

3. **No timing or efficiency metrics.** The paper claims "real-time" rendering in the abstract and introduction but provides no FPS numbers, generation time, or memory breakdown. With 20 perspective views processed per denoising step, this is a notable omission for a systems-oriented paper claiming real-time capability.

4. **"6-DoF" claim is unsupported.** The abstract and introduction repeatedly mention "6-DoF virtual tours" and "free-viewpoint" rendering, but the method only supports rotational camera motion around a fixed center plus very small translations (used only as regularization). This overclaim could mislead readers about the method's capabilities.

### Trivial

- The LDM section in Sec. 3.1 repeats the I2V definition sentence twice (sentences ending with "where x_t, x_{t-1} ∈ ℝ^{l×h×w×c} represent the sampled latent codes and I the conditioning image" appear verbatim in consecutive paragraphs).

## Nice-to-Haves

- Ablation on the number of perspective views (n=20) and their arrangement to justify this design choice
- Comparison with at least one adapted baseline (e.g., DreamScene360 with per-frame appearance modulation) or a video-to-4D method applied to perspective crops
- Evaluation on real-captured panoramas to test generalization beyond synthetic data
- Quantitative temporal consistency metrics (e.g., warp error, flicker scores)
- Visualization of rendered depth maps or 3D geometry quality

## Removed Points

- *Reproducibility concern about undisclosed hyperparameters/training details* — The paper provides core hyperparameters (20 directions, 512×512 perspective views, A100 GPU, specific loss weights). Missing details like diffusion step count and total frames L are minor implementation specifics, not fatal gaps. Weakened and retained only the specific underspecification of Eq. 3 in Major #2.
- *Heavy reliance on pretrained models with limited novelty* — This is standard practice in generative model papers. The novelty is in the adaptation strategy (spherical fusion, depth alignment), not in training new models from scratch. Most papers in this area build on pretrained diffusion models.
- *User study details deferred to appendix* — This is a common and accepted practice; not a weakness of the paper.
- *Missing related works (specific papers)* — Per rules, I cannot confirm the existence of specific uncited works and should not flag missing references.
- *Concern that Q-Align metrics are biased* — While Q-Align is a learned metric, it is commonly used in the generative models community. The paper also includes user studies, which provides a secondary evaluation channel. This is a minor concern, not a major one.

## Novel Insights

The project-denoise-fuse approach for adapting perspective diffusion priors to panoramic domains is an interesting and transferable idea. The key insight is that simultaneous denoising of overlapping perspective views with fusion back to a spherical latent at each step can maintain global coherence while leveraging domain-specific (perspective) models. This is conceptually similar to multi-view diffusion synchronization (as in StochSync) but applied in a video generation (I2V) context with a different fusion mechanism. However, the paper does not clearly specify whether this fusion is simply averaging or something more sophisticated, which limits the ability to assess the true novelty of the mechanism versus a straightforward baseline.

## Suggestions

1. **Provide explicit implementation details for the Panoramic Denoiser fusion step.** Specify the parameterization of S, the optimization algorithm for Eq. 3, and how overlapping regions are weighted. Even a short paragraph in an appendix would dramatically improve reproducibility.

2. **Demonstrate meaningful camera translation in evaluation.** Render and show views with camera translations of 0.1m, 0.5m, 1m to honestly characterize the method's parallax capabilities and limitations, rather than only evaluating at or very near the origin.

3. **Add at least one additional baseline.** Even applying a static panoramic GS method (DreamScene360) and overlaying the animated video as texture would provide a more meaningful comparison than 3D-Cinemagraphy alone.

4. **Report timing numbers.** At minimum: total animation time, total lifting time, and rendering FPS at 4K.

## Score and Decision

**Calibration anchors:**
- *GaussianFlow* (4D Gaussian splatting for 4D content, scores 5-6, withdrawn): Similar 4D generation domain, but with more baselines and comparable evaluation gaps. This paper has a more novel problem setting but even thinner evaluation.
- *Optimizing 4D Gaussians for Dynamic Scene Video* (single image to 4D landscape, scores 3-8, accepted poster ~6): Very similar task scope (single image to 4D). Accepted despite evaluation concerns, but had PSNR metrics and more scenes.
- *StochSync* (panorama generation via diffusion sync, scores 6, accepted poster): Spherical diffusion synchronization — most directly related technical approach. Similar methodological novelty level, cleaner evaluation.
- *PanoDiffusion* (360° panorama generation, scores 6-8): Similar panoramic domain, stronger evaluation.

This paper has a genuinely novel and important problem, a creative technical approach, and promising visual results. However, it suffers from: (1) evaluation that is fundamentally misaligned with its claims (no geometric metrics, one weak baseline, 16 synthetic scenes, no translational novel views), (2) underspecification of the core Panoramic Denoiser contribution, and (3) the "4K" and "6-DoF" claims that overstate what is demonstrated. The evaluation gaps are more severe than most accepted papers in this space. I place this below StochSync and PanoDiffusion (6) and below the landscape 4D Gaussian paper (6), but the novel problem formulation keeps it above clearly weak papers (3-4).

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>