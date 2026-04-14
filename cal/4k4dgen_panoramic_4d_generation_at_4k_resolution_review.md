=== CALIBRATION EXAMPLE 7 ===

# Final Consolidated Review
## Summary
4K4DGen proposes a pipeline for lifting a single static 360° panoramic image into a 4D immersive environment at 4K (4096×2048) resolution. The method introduces a **Panoramic Denoiser** that adapts perspective I2V diffusion models to the spherical domain via a project-and-fuse denoising loop (a temporal/spherical extension of MultiDiffusion), and a **Dynamic Panoramic Lifting** stage that builds time-dependent 4D Gaussian Splatting representations using spatial-temporal geometry alignment. The approach requires no annotated 4D panoramic training data by transferring priors from perspective-domain diffusion models.

---

## Strengths

- **Principled and well-motivated spherical denoising.** The Panoramic Denoiser is technically coherent: projecting 20 uniform perspective views, denoising each with an unmodified I2V model, and fusing back to spherical latents at each DDPM step is a clean and reproducible idea. The ablation (Table 2) concretely validates it against "Animate Pano." (small motion, 2K ceiling) and "Animate Pers." (inconsistent cross-view, 33% view-consistency UA vs. 70% for the full model), demonstrating that the method is not just a combination of obvious alternatives.
- **Tackling a genuinely underexplored intersection.** Existing 4D generation works are either object-centric (low-resolution, no 360° FoV) or static panoramic 3D scenes. The specific combination of full-sphere, dynamic, 4K, and no specialized 4D training data is not addressed by prior work, and the paper's motivation is well-grounded.
- **Spatial-Temporal Geometry Alignment with measurable impact.** The STA module—fusing per-perspective depth maps with per-frame scale/shift parameters regularized across time—goes beyond naive monocular depth estimation. The qualitative ablation in Fig. 6 (right) demonstrates tangible geometry consistency improvement in dynamic regions (volcano smoke), showing the module is doing real work.
- **Strong user study preference for the full pipeline.** 81% user choice for quality and 59.2% for naturalness (Table 1) against 3D-Cinemagraphy, across separate forced-choice studies, are large margins that provide some confidence in visual quality gains, despite the limited dataset size.

---

## Weaknesses

### Fatal
None.

### Major

- **Severely limited baseline comparison.** The only external comparison is 3D-Cinemagraphy (Li et al., 2023b), an optical-flow-based cinemagraph method that does not do panoramic generation and was never designed for this task. Beating it is unsurprising. At minimum, the paper should include: (a) a composite generative baseline that animates each perspective view independently and lifts them to 4DGS ("Animate Pers. + 4D lift") — the ablation already generates these 2D videos but never evaluates the resulting 4D representation's quality; (b) an adapted DreamScene360 output animated with a naive I2V pass and lifted. Without such comparisons, the relative contribution of the two proposed modules to 4D rendering quality (rather than 2D video quality) remains unvalidated. This is the most significant weakness of the evaluation.

- **"Real-time" claim is never substantiated.** The abstract, introduction, and conclusion all claim the system supports "real-time exploration" via Gaussian splatting. Yet not a single FPS or rendering latency number is reported anywhere in the paper. The limitations section itself acknowledges "substantial storage capacity" requirements. This claim cannot be taken on faith — either report rendering FPS at 4K or remove the claim.

- **Missing second stage of training.** Section 3.4 states: "The training process is structured in two stages: initially, we directly supervise the 3D Gaussians using the panoramic videos." Only this first stage is described. The second stage is never disclosed, which is a material reproducibility gap.

- **Eq. 3 practical solver is unspecified.** The optimization objective in Eq. 3 is presented as an argmin, but the actual numerical procedure used to solve it is never described. For MultiDiffusion-style methods, the standard solution is a weighted average (closed-form under L2), but the paper never confirms this. Readers cannot reproduce the Panoramic Denoiser step without this information.

- **The "4K" label for dynamic content is potentially misleading.** Implementation details (Sec. 4.1) reveal that each perspective view is denoised at 512×512 resolution. The 4K output for dynamic regions is obtained by projecting back to the equirectangular panorama. The actual resolution of generated motion content is therefore bounded by the 512×512 per-view processing, not 4K. The paper should clearly separate "static background resolution" from "effective dynamic element resolution" to substantiate the 4K claim for the dynamic portions.

### Minor

- **Duplicated paragraph in Section 3.1.** The paragraph defining I2V generation ("Recently, image-to-video (I2V) generation has been realized…") appears verbatim twice in Section 3.1. This is a clear proofreading oversight.

- **No temporal consistency or novel-view quality metrics.** All quantitative evaluation (Table 1, Table 2) uses Q-Align, which assesses perceptual quality of individual frames/videos. There are no metrics that measure 3D or temporal consistency — the two core claims of the 4D representation. Warping error or depth temporal variance would directly measure the STA module's contribution quantitatively, and LPIPS/FID on held-out novel views would validate the Gaussian lifting quality. The ablation of $\mathcal{L}_\text{Temp}$ (Fig. 6 left) is qualitative only.

- **Equation 4 notation is ambiguous.** The nested `argmin_S min_{i} E[...]` is not standard. It is unclear whether the argmin over S and the min over i are solved jointly or sequentially. The downstream formulas (scale, shift regularization) are clearer, but the top-level equation needs rewriting.

- **Temporal loss tension with desired motion.** $\mathcal{L}_\text{temporal}$ (Eq. 6) penalizes rendering differences between consecutive Gaussian frames. This is applied during the *lifting* phase, which means it could also suppress some desired temporal variation in the Gaussian representation beyond flickering. The paper does not discuss the λ_temporal value or how the balance between suppressing artifacts and preserving motion amplitude is tuned.

- **Small evaluation dataset.** 16 panoramas is very limited; no variance or confidence intervals are reported. While this scale is consistent with some prior scene generation work (e.g., DreamScene360), it weakens statistical conclusions, especially for user studies.

### Tiny

- **View distribution underspecified.** The paper states "uniformly select 20 directions on S²" but does not specify the method (Fibonacci spiral, icosahedral, etc.) or the FoV of each view. This matters for reproducibility and for understanding polar vs. equatorial coverage density.
- **STA ablation is qualitative only.** The heatmap comparison in Fig. 6 (right) is not supported by a depth error or surface consistency metric.
- **$\mathcal{L}_\text{sem}$ and $\mathcal{L}_\text{geo}$ adaptation** from DreamScene360 to the dynamic/temporal setting is not explicitly discussed — it is mentioned that they are adopted but not how they are applied across frames.

---

## Nice-to-Haves

- **Real-world panorama experiments.** All 16 evaluation panoramas are generated by a text-to-panorama diffusion model. Evaluating on real captured 360° images would strengthen the generalization claim.
- **Failure case gallery.** Visualizing cases where the spherical denoising fails (cross-view seams, globally inconsistent motion, polar distortions) would provide a balanced view of the method's current limits.
- **6-DoF fly-through video demonstrations.** Continuous camera trajectories would more convincingly validate the "immersive 4D environment" claim than static novel-view snapshots.
- **Storage and generation time statistics.** Even if not real-time, reporting total generation time, VRAM usage during generation, and per-asset storage size at 4K would allow practitioners to assess feasibility.
- **Text control demonstration.** The abstract and Figure 1 caption mention "user interaction or an input mask" and text prompts, but the methodology focuses on mask-based region specification. A clearer demonstration of how text conditioning influences the dynamic content would be helpful.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **User study values summing to >100% (Harsh Critic).** The critic raised confusion about UC values summing to 100%. This is correct behavior for a 3-way forced-choice study ("compare and select the best video from candidates generated by different methods"), which by construction yields proportions summing to 100%. This is not a flaw.
- **Insufficient discussion of relationship to MultiDiffusion (Harsh Critic).** The paper does cite Bar-Tal et al. (2023) and Lee et al. (2023) in the related work under "sampling-based techniques" (Sec. 2). While a more explicit positioning of Eq. 3 vs. MultiDiffusion would be welcome, the connection is acknowledged. Per instructions, absence of further related-work discussion is not a weakness we can verify externally.
- **Claiming cited references don't exist / methods aren't released.** None of the reviewers made this claim here; no removal needed on this ground.
- **Independent per-view denoising producing globally inconsistent content (Harsh Critic).** This is the hallucination consistency concern. While theoretically valid, it is exactly the problem the spherical latent fusion is designed to solve, and the ablation demonstrates improved cross-view consistency. This concern would need empirical evidence of actual failure to rise to a weakness.

---

## Novel Insights

The most substantive novel observation — raised by the Spark Finder and supported by the implementation details — is the **resolution disconnect in the "4K" claim**: the actual dynamic content is generated at 512×512 per perspective view and then reverse-projected onto the 4K equirectangular canvas, meaning the effective resolution of *generated motion* is substantially lower than 4K, while the static background inherits full 4K fidelity from the input panorama. This architectural reality deserves explicit acknowledgment and analysis. If the paper can demonstrate that the per-view 512×512 denoising, when projected back, yields visually 4K-quality dynamic elements (e.g., through super-resolution or the native detail preservation of the Gaussian splatting rasterizer), that would be a meaningful finding worth documenting.

---

## Suggestions

1. **Add at minimum one generative 4D baseline**: run "Animate Pers. + DreamScene360-style 4DGS lift" as a full 4D system and evaluate rendered novel views with the same Q-Align + user study metrics as Table 1. This would concretely isolate the contribution of the Panoramic Denoiser to the 3D representation quality.
2. **Report FPS at inference time**, or explicitly downgrade the "real-time" claim to "near real-time" or "real-time capable (via 3DGS rasterization)."
3. **Describe the second training stage** in Section 3.4, even if briefly, to close the reproducibility gap.
4. **Clarify Eq. 3 implementation**: one sentence confirming whether the fusion is a weighted average (closed-form L2 solution) suffices, but it must be present.
5. **Add a quantitative lifting-phase ablation**: report a temporal consistency metric (e.g., per-frame LPIPS between adjacent rendered frames) for the full model vs. w/o $\mathcal{L}_\text{Temp}$ and vs. w/o STA to complement the qualitative Fig. 6.
6. **Rewrite Eq. 4** to clarify the joint vs. sequential nature of the nested argmin/min.
7. **Provide a per-view resolution analysis** for the dynamic content: demonstrate whether the 512×512 per-perspective denoising, after spherical reprojection and Gaussian rasterization, actually recovers 4K-level detail in the dynamic regions.

---

**Overall evaluation:**
- *Novelty*: Moderate-to-high — the task combination (360°, 4D, 4K, no 4D training data) is new; the Panoramic Denoiser is an incremental but principled extension of MultiDiffusion to the temporal/spherical domain.
- *Technical soundness*: Moderate — the pipeline is coherent but has reproducibility gaps (missing solver for Eq. 3, undescribed second training stage, ambiguous Eq. 4).
- *Empirical support*: Weak-to-moderate — ablations convincingly validate the animating phase, but the comparison baseline is mismatched, the evaluation set is tiny, and the core 3D/4D claims lack appropriate metrics.
- *Significance*: Moderate — genuinely useful for VR/AR content creation, with a scalable approach; significance is tempered by the reliance on I2V model quality and the unverified real-time claim.
- *Clarity*: Moderate — the pipeline overview is clear and the figures are informative, but key implementation details are missing or ambiguous.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 6.0]
Average score: 7.5
Binary outcome: Accept
