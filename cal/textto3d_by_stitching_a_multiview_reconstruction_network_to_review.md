=== CALIBRATION EXAMPLE 86 ===

# Harsh Critic Review
Now I have enough to write a thorough review. Let me compile my analysis.

---

## Section-by-Section Critical Review

### Title & Abstract

The title accurately captures the core idea—stitching a multi-view reconstruction network to a video generator—and the acronym VIST3A is clearly decoded. The abstract's claims are mostly well-supported: the paper does address both the "decoder quality" and "alignment" limitations it identifies. However, the abstract asserts "all tested pairings markedly improve over prior text-to-3D models that output Gaussian splats," a strong blanket claim that deserves scrutiny: the DPG-Bench table (Table 2 in the main text) appears to omit VIST3A's own rows in the parsed text, making full verification difficult (though this may be a PDF-parsing artifact). The claim about "no labels" is accurate but slightly misleading, since the reward finetuning stage does rely on prompts from the HPSv2 training set and uses DL3DV multi-view data.

---

### Introduction & Motivation

The motivation is clear and compelling: existing 2D-to-3D LDM pipelines retrain a decoder from scratch, which is expensive and lags behind rapidly improving 3D foundation models. The two identified limitations—weak 3D decoder from scratch, and weak alignment between generator and decoder—are well-articulated and lead naturally to the two proposed components. The paper situates itself appropriately in the landscape of concurrent work (e.g., DiffSplat, Bolt3D, VideoRFSplat).

One subtle issue: the introduction argues that "the Achilles heel" of existing methods is the from-scratch decoder, yet many baselines (Director3D, SplatFlow) do not use this paradigm at all. Director3D in particular uses a multi-stage pipeline, not a retrained VAE decoder. The framing should be more precise about which prior works the critique applies to.

---

### Methodology (Section 3)

**3.1 Model Stitching**

The core idea—searching for the 3D model layer whose activation is most linearly compatible with the video VAE latent—is elegant and well-motivated. The closed-form least-squares solution (Eq. 2) is straightforward. Several issues warrant clarification:

1. **Discrepancy between theory and implementation.** Equation 2 defines the stitching layer **S** as a linear map, and the layer is identified via a least-squares criterion. However, Appendix B.1 reveals that the actual stitching layer is a *3D convolution* (not a pure linear map), preceded by an interpolation step. The theoretical justification via Insulla et al.'s bound (Eq. 4) applies to a linear map **S**, but the implemented stitching layer is nonlinear (due to interpolation before convolution). The gap between the theory and the implementation should be explicitly acknowledged.

2. **Scope of the search.** The paper states: "We limit our search to the encoder layers of each model, as we observe that MSE values consistently increase within deeper layer indices." This means only a subset of layers is searched, and the claim that the optimal stitching layer is found is conditional on this pre-filtering. If the MSE criterion itself is used to justify limiting to encoder layers, the argument is circular: it uses MSE to prune the search, then uses MSE as a proxy for reconstruction quality to validate the criterion.

3. **Data sensitivity of layer search.** The search uses 200–3,200 scenes for different models. How sensitive is the choice of stitching layer to this dataset? Would the same layer emerge from a different distribution of images?

4. **LoRA fine-tuning scope.** Step 2 fine-tunes **S** and F_{k*+1:l} using LoRA, but the front portion F_{1:k*} of the 3D model is discarded. The paper claims this results in minimal performance loss (Table 5), and indeed the pointmap numbers are close to the original 3D models. This is one of the strongest empirical results in the paper and a genuine positive finding.

**3.2 Direct Reward Finetuning**

The choice of direct reward finetuning (DRF) is well-motivated: unlike single-step rendering losses, it propagates gradients through the full denoising trajectory, ensuring the generated latents are in-distribution for the 3D decoder. Several concerns:

1. **3D consistency reward counterproductively degrades performance in isolation.** Table 6 (ablation) shows that adding the 3D-consistency reward *alone* ("Multi-view + Consistency") dramatically hurts performance: Imaging quality drops from 54.56 to 38.67, aesthetic from 52.08 to 50.59. The explanation given is "blurred images" because "the model optimizes for geometric correctness at the expense of detail." If this reward is individually harmful, its inclusion in the full objective requires stronger justification. The weighting scheme (consistency reward scaled by 0.05 vs. quality rewards at 1/16 each) needs discussion of how this was determined.

2. **Reward metrics for 3D.** The 3D representation quality reward is computed on *rendered images* from the generated Gaussian splats, evaluated with CLIP and HPSv2—both metrics developed and calibrated for 2D image quality. It is not clear whether these metrics capture 3D-specific properties (multi-view consistency, geometric accuracy) rather than just per-frame aesthetics.

3. **K=2 gradient-enabled steps.** The gradient is only enabled for K=2 out of t denoising steps. This is extremely sparse. No sensitivity analysis for K is provided. Is this sufficient to align the full denoising trajectory?

4. **Reward weights.** The weights (1/16 for quality, 0.05 for consistency) appear to have been tuned, but no ablation on these hyper-parameters is given. These choices could substantially affect performance.

---

### Experiments & Results (Section 4)

**4.2 Main Results: 3D Generation**

The quantitative results (Tables 1, 4) show substantial improvements over all baselines on T3Bench, SceneBench, and the user study. However, several concerns undermine the strength of these conclusions:

1. **Table 2 (DPG-Bench) appears to omit VIST3A results.** The body of Table 2 lists only five baselines (Matrix3D-omni, Director3D, Prometheus3D, SplatFlow, VideoRFSplat) without corresponding VIST3A rows, while the text claims "our models greatly outperform the baselines, mostly scoring > 75 (often even ≈ 85)." Even allowing for PDF parsing artifacts, this is a significant gap. Reviewers cannot verify these DPG-bench numbers from the main paper, and the specific per-column scores for VIST3A are not given anywhere in the extracted text.

2. **The primary gain may be due to the video backbone, not the stitching methodology.** VIST3A uses Wan 2.1, a 2025 state-of-the-art video model, while most baselines use older generators. There is no ablation directly comparing "Wan 2.1 + sequential 3D reconstruction (no stitching)" vs. "Wan 2.1 + stitched 3D reconstruction." Table 3 does compare AnySplat alone vs. stitched variants for NVS, but that doesn't address the text-to-3D generation setting specifically.

3. **Post-processing for MVDUSt3R is not uniformly applied.** Appendix C.2 notes: "For our text-to-3DGS evaluation, we refined the Gaussian primitives using the source view for 100 optimization steps." This additional per-scene optimization step is applied to VIST3A (Wan+MVDUSt3R) but it is unclear whether it is applied to baselines that do not use MVDUSt3R. This is a potentially significant confound.

4. **DPG-Bench evaluation protocol change.** The paper "upgrades" the originally proposed LLM evaluator to the UnifiedReward model (Qwen 7B). If baseline numbers were computed with the original protocol, the comparison may be unfair. It is also unclear whether baseline numbers were recomputed using the upgraded evaluator.

5. **Inference time not reported.** VIST3A involves a full denoising loop followed by 3D decoding, while some baselines also include SDS refinement. Computation/inference time is entirely absent, making it impossible to assess efficiency trade-offs.

**4.3 Main Results: Model Stitching**

Table 5 is one of the strongest results: stitched models (e.g., Wan+VGGT) achieve pointmap and camera pose estimation performance nearly identical to the original 3D model (VGGT), despite receiving video VAE latents as input instead of direct image features. This validates the core claim that stitching preserves 3D capabilities. The correlation between MSE at the stitching layer and 3D performance (Figure 5) is a compelling empirical validation.

However:
- All stitched variants on 7-Scenes are slightly *worse* than the originals (e.g., MVDUSt3R: Acc. 0.026 vs. 0.026 for stitched—these are nearly identical, so the claim of "barely change" is supported).
- The evaluation of the stitching layer for CKA (Figure 6) vs. MSE is interesting but the description is fragmented in the extracted text, making it hard to assess the figure precisely.

**4.4 Ablations**

The ablation on reward components (Table 6, Appendix D.1) is informative but raises the concern noted above about the 3D-consistency reward. The comparison of stitched vs. sequential under noise injection (Figure 8, Appendix D.2) is a useful proxy for robustness, though controlled noise is an idealization of actual generation errors.

The stitching layer search ablation (Figure 5) is directly in the main paper and strongly supports the MSE criterion as a layer-selection heuristic.

**Missing ablations:**
- No ablation on the number of data points used for layer search (sensitivity to N=200 vs. N=3200).
- No ablation on LoRA rank for the stitched decoder fine-tuning.
- No quantitative evaluation of text-to-pointmap generation quality; the paper relies entirely on qualitative results (Figure 1b and Appendix E).

---

### Writing & Clarity

The main paper is well-organized and clearly written. The methodology is logically structured. Two sections of concern:

1. **Paragraph 2 of Section 2 is repeated almost verbatim at lines 782–788 and 785–788**, which reads: "Additional results in Appendix E demonstrate that VIST3A inherits prompt-based camera control... and our finetuning does not degrade — and even slightly improves — video generation quality as measured by VBench. **Additional results in Appendix E**, we further show that VIST3A inherits prompt-based camera control..." This redundancy suggests an editing artifact.

2. **Algorithm 1** appears twice in the main text (partially at lines 1616–1646, then fully at lines 1649–1670), which is also a formatting/duplication issue.

---

### Limitations & Broader Impact

The paper does not have a dedicated limitations section, and the conclusion is optimistic without acknowledging failure modes. Several limitations are missing:

1. **Dependence on pretrained model quality.** The stitching approach is only as good as the 3D foundation model it reuses. If the 3D model fails on a certain domain (e.g., extremely uncommon scenes), stitching will propagate that failure.

2. **Video-like scene layouts.** The video encoder imposes a strong prior toward scenes that look like video frames—static scenes with continuous viewpoints. The paper acknowledges that "multi-view images are arranged into sequences with smooth view transitions to resemble video input," which may bias generation.

3. **Scale of Gaussian primitives.** Appendix C.2 acknowledges "MVDUSt3R model does not generalize well across diverse domains, often failing to estimate the scale of primitives," requiring additional optimization. This is a significant limitation for the MVDUSt3R variant that should appear in the main paper.

4. **Societal impact.** No discussion of misuse potential (e.g., generating photorealistic 3D environments for disinformation) is provided.

---

### Overall Assessment

VIST3A presents a genuinely creative and technically sound idea: instead of training a 3D decoder from scratch, search for the layer in an existing 3D foundation model whose activations are most linearly compatible with the video VAE latent space, and stitch the two. The empirical evidence that this works—and works across multiple video models and 3D models—is one of the paper's strongest contributions. The direct reward finetuning for alignment is well-motivated and the ablation (though flawed by the counterproductive 3D consistency reward) provides useful insight. However, the paper has several concerns that must be addressed before acceptance at ICLR: (1) the VIST3A results appear missing from Table 2 (DPG-Bench), making one of the primary benchmark claims unverifiable; (2) the performance gains are partly confounded by the much stronger Wan 2.1 backbone, and the critical "sequential baseline" (Wan 2.1 → video decode → VGGT without stitching) is absent; (3) the 3D consistency reward alone hurts performance and this tension is not fully resolved; (4) the per-scene Gaussian primitive refinement in evaluation is not clearly applied symmetrically across methods; and (5) the text-to-pointmap contribution has no quantitative evaluation. Contingent on clarifying these points—particularly items 1 and 2—the core contribution is novel and likely above the ICLR bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces VIST3A, a framework for text-to-3D generation that integrates a pretrained text-to-video latent diffusion model with a feedforward 3D reconstruction model via a technique termed "model stitching." The framework addresses the incompatibility between generative latents and 3D decoders by identifying a compatible layer in the 3D network to stitch with the VAE latent space, followed by "direct reward finetuning" to align the generator with the stitched decoder's domain. Experiments demonstrate that VIST3A outperforms existing feedforward and Score Distillation Sampling (SDS) baselines on multiple benchmarks (T3Bench, SceneBench, DPG-Bench) while enabling additional modalities like text-to-pointmap generation.

### Strengths
1.  **Effective Leverage of Foundation Models:** The core contribution of reusing pretrained, feedforward 3D vision models (e.g., AnySplat, MVDUSt3R, VGGT) as decoders rather than training custom VAE decoders from scratch is compelling. Section 4.3 (Table 5) shows that stitching preserves the original 3D model's accuracy, avoiding the need for massive labeled datasets to rebuild 3D reasoning.
2.  **Strong Empirical Performance:** The proposed method demonstrates state-of-the-art results across diverse benchmarks. Table 1 and Table 2 show consistent superiority over baselines like Director3D and SplatFlow in metrics such as Imaging Quality and Unified Reward scores. The human evaluation in Table 4 further corroborates these findings, with VIST3A achieving the best average rank.
3.  **Innovative Alignment Strategy:** The use of Direct Reward Finetuning (DRF) to align the generative video model with the 3D decoder is a sound approach to solving the distribution mismatch between generated latents and reconstruction requirements. Abation studies in Appendix D.1 (Table 6) clearly isolate the contribution of quality vs. consistency rewards, showing that their combination yields the best results without degrading video generation quality (Table 7).

### Weaknesses
1.  **Heuristic Nature of Stitching Criterion:** The criterion for selecting the stitching layer (minimizing MSE between encoder latents and 3D model activations, Eq. 2) is empirically motivated but lacks strong theoretical justification regarding why this specific layer depth correlates with best reconstruction performance. While Figure 5 supports the claim, the relationship could be explored more deeply theoretically.
2.  **Input Ordering Constraint:** Section F (Limitations) admits that the framework requires input images to be arranged into a temporally coherent sequence due to the video VAE encoder. This restricts immediate applicability to unordered multi-view datasets without preprocessing, which is a practical engineering hurdle for general 3D reconstruction tasks.
3.  **Dependency on Proxy Rewards:** The "Direct Reward Finetuning" relies on external metrics like CLIP, HPSv2, and LPIPS. While effective, these are proxy rewards and may not capture ground-truth 3D geometric consistency perfectly. There is a risk that the model optimizes for pleasing textures or CLIP alignment at the expense of precise geometric fidelity, though the consistency reward mitigates this.

### Novelty & Significance
The paper introduces a significant novelty by reframing text-to-3D generation as a model composition problem rather than a training-from-scratch problem. The concept of "model stitching" for aligning generative and reconstruction latents is underexplored in the context of 3D. Given ICLR's focus on representation learning and foundation models, this work aligns well with the trend of combining powerful, pre-trained components (video LDMs + 3D Vision models). The significance lies in the potential to reduce training costs and improve generalization by decoupling generation priors from reconstruction priors while ensuring they work together.

### Suggestions for Improvement
1.  **Clarify Stitching Search Complexity:** Provide a more detailed discussion on the computational cost of finding the optimal stitching layer index across different model pairs. Is this search step negligible compared to the finetuning phase? This is important for the efficiency claims.
2.  **Address the Input Ordering Limitation:** Discuss potential future strategies or modifications (e.g., using a spatial pooling layer instead of temporal compression in the VAE) to overcome the requirement for sequential input in Section F.
3.  **Expand on Reward Sensitivity:** Analyze how sensitive the final results are to the reward weights (Section B.2). A small sensitivity analysis (e.g., varying the weight of consistency vs. quality rewards) would strengthen the claim that the proposed combination is robust.
4.  **Compare Inference Latency:** While training efficiency is highlighted, a comparison of inference latency (seconds per scene) against the SDS-based baselines (which require per-scene optimization) should be explicitly quantified to fully support the "efficient generation" argument.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Geometric Ground Truth Evaluation:** Add Chamfer Distance or F-Score metrics on synthetic datasets (e.g., Objaverse) with known 3D ground truth. Current metrics (CLIP, HPS) are 2D proxies that cannot verify actual geometric fidelity, which is the core claim of using a 3D reconstruction decoder.
2. **Inference Efficiency Benchmarks:** Report explicit latency (seconds/sample) and peak VRAM usage compared to feedforward baselines like LRM or Instant3D. The paper claims efficiency over SDS methods but lacks quantitative compute comparisons against other single-pass generators.
3. **Cross-Architecture Generalization:** Test stitching on unseen video/3D model pairs without layer-search hyperparameter tuning. The current results rely on specific combinations (Wan+AnySplat); broader testing is needed to validate the "general framework" claim.
4. **Data Efficiency Quantification:** Explicitly compare the number of training samples required for stitching vs. training a custom decoder from scratch. The abstract claims "only a small dataset," but this is not quantified against the baseline data requirements.

### Deeper Analysis Needed (top 3-5 only)
1. **Reward Conflict Analysis:** Table 6 shows the 3D Consistency reward *degrades* performance. Analyze gradient norms or loss landscapes to explain why enforcing consistency harms visual quality, as this undermines the alignment strategy.
2. **Latent Distribution Shift:** Quantify the distributional divergence (e.g., MMD or FID in latent space) between original 3D model activations and stitched VAE latents. This is needed to prove the stitching layer actually aligns the distributions rather than just minimizing MSE.
3. **Pose Error Propagation:** Analyze how camera pose errors from the video generator propagate to 3D reconstruction errors. Since video models hallucinate motion, quantifying this sensitivity is critical for trusting the geometric output.
4. **Stitching Layer Semantics:** Investigate *why* early layers stitch best (Fig 5). Determine if these layers capture geometry or just low-level edges, as this dictates whether the method can scale to semantic 3D generation tasks.

### Visualizations & Case Studies
1. **Epipolar Consistency Maps:** Visualize pixel-wise differences between rendered views and decoded views along epipolar lines. Single-view renderings hide 3D inconsistencies that epipolar checks would expose.
2. **Failure Mode Gallery:** Include examples where stitching fails (e.g., transparent objects, complex topology, non-rigid structures). Defining boundary conditions is essential for assessing reliability.
3. **Latent Space Interpolation:** Show smooth interpolation in the latent space to verify the stitched decoder maintains manifold continuity without collapsing or introducing artifacts between points.

### Obvious Next Steps
1. **Explicit Camera Control:** Integrate explicit camera pose conditioning into the generator rather than relying on text prompts (e.g., "aerial shot"). Text-based camera control is too imprecise for rigorous 3D reconstruction.
2. **Dynamic 4D Generation:** Leverage the video backbone's temporal capability to generate dynamic 4D Gaussian Splats. Restricting the model to static 3D scenes ignores the primary advantage of using a video generator.
3. **End-to-End Encoder Finetuning:** Experiment with unfreezing the video encoder to allow deeper alignment. Freezing the encoder creates a bottleneck that likely limits the maximum achievable reconstruction quality.

# Final Consolidated Review
## Summary
VIST3A proposes a framework for text-to-3D generation that "stitches" a pretrained video latent diffusion model to a pretrained feedforward 3D reconstruction model, rather than training a custom decoder from scratch. The key technical contributions are: (1) a layer-search procedure to find the most linearly compatible layer in the 3D model to stitch to the video VAE latent space, and (2) direct reward finetuning to align the generator with the stitched decoder.

## Strengths
- **Novel reuse of foundation models:** The core idea of leveraging existing pretrained 3D reconstruction models (AnySplat, MVDUSt3R, VGGT) as decoders rather than training custom VAE decoders from scratch is compelling. Table 5 validates this empirically: stitched models achieve pointmap and pose estimation performance nearly identical to the original 3D models, confirming that stitching preserves 3D capabilities.
- **Strong quantitative results on text-to-3DGS benchmarks:** Tables 1 and 4 demonstrate consistent improvements over baselines on T3Bench, SceneBench, and human evaluation. The gains in Imaging Quality (58.83 vs. 54.32 best baseline) and Coherence scores (3.89 vs. 3.25) are substantial.
- **Cross-architecture generalization demonstrated:** The framework is tested across multiple video generators (Wan, CogVideoX, SVD, HunyuanVideo) and multiple 3D models (AnySplat, MVDUSt3R, VGGT), showing the stitching approach is not architecture-specific.

## Weaknesses
- **DPG-Bench results are not verifiable:** Table 2 lists only baseline scores for DPG-Bench; VIST3A's scores are not shown in the table despite the text claiming "our models greatly outperform the baselines, mostly scoring > 75 (often even ≈ 85)." This prevents verification of one of the three main benchmark claims in the paper.

- **Critical sequential baseline is missing:** The paper does not compare against a "Wan 2.1 → decode to RGB → VGGT without stitching" baseline to isolate the contribution of stitching versus simply using a stronger video backbone. Table 3 shows AnySplat alone vs. stitched variants for novel-view synthesis, but this does not address the text-to-3D generation setting where latents are generated (not encoded from images). Without this comparison, it is unclear how much of the performance gain comes from the video model quality versus the stitching methodology itself.

- **3D consistency reward degrades performance in isolation:** Table 6 shows that adding the 3D-consistency reward alone ("Multi-view + Consistency") drops Imaging Quality from 54.56 to 38.67 and degrades other metrics. The paper explains this is due to "blurred images" from optimizing geometric correctness at the expense of detail, but this raises the question of why this reward is included if it is individually harmful. The weighting scheme (consistency scaled by 0.05 vs. quality at 1/16) is presented without ablation on these hyperparameters.

- **MVDUSt3R variant requires per-scene optimization not clearly applied to baselines:** Appendix C.2 states that "the Gaussian splatting produced by the MVDUSt3R model does not generalize well... often failing to estimate the scale of primitives" and requires "100 optimization steps" of additional refinement. It is unclear whether equivalent refinement is applied to baseline methods, creating potential asymmetry.

- **No quantitative evaluation for text-to-pointmap:** The paper introduces text-to-pointmap as an additional capability (Figure 1b), but provides only qualitative visual results without benchmark comparisons or metrics, leaving this contribution unsupported.

- **Inference efficiency not reported:** No latency or memory measurements are provided. SDS-based baselines are criticized as slow, but feedforward baselines like the ones compared are also single-pass; the efficiency trade-offs are unclear.

## Nice-to-Haves
- Geometric ground-truth evaluation (Chamfer Distance, F-Score) on synthetic datasets where 3D ground truth is available, to complement the 2D proxy metrics (CLIP, HPSv2).
- Ablation on the number of gradient-enabled steps K and reward weights, to justify the chosen hyperparameters.
- Sensitivity analysis of the stitching layer to the search dataset composition.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Formatting issues (duplicate paragraphs, Algorithm 1 appearing twice):** These are minor editing artifacts that do not affect the technical content.
- **Dependency on proxy rewards as a weakness:** Using CLIP/HPSv2/LPIPS as proxy metrics is standard practice in generative model evaluation and not specific to this work.
- **Theoretical justification gap for MSE criterion:** Figure 5 empirically validates MSE as a layer-selection heuristic, and the connection to the theoretical bound in Eq. 4 is provided. While not a complete theory, this is sufficient empirical grounding for a methods contribution.
- **Input ordering constraint limitation:** This is acknowledged in Section F and is inherent to using a video encoder; it does not invalidate the contribution.

## Novel Insights
The observation that early layers of pretrained 3D reconstruction models have activations linearly compatible with video VAE latents is surprising and potentially generalizable to other cross-domain stitching problems. The ablation in Figure 5 showing that lower MSE at the stitching layer correlates with better downstream performance provides empirical evidence that representation alignment is a viable proxy for transferability, even without full theoretical justification. The finding that direct reward finetuning through the full denoising trajectory is more effective than single-step rendering losses aligns with recent work in diffusion alignment and suggests that 3D consistency should be enforced end-to-end rather than locally.

## Suggestions
- **Add the missing DPG-Bench results to Table 2:** Include VIST3A scores for all columns so readers can verify the claimed improvements.
- **Add a sequential baseline for text-to-3D:** Compare "Wan 2.1 + decode to RGB + VGGT (no stitching)" to isolate the contribution of the stitching approach.
- **Provide quantitative text-to-pointmap evaluation:** Either add benchmark metrics or acknowledge this as qualitative-only in the contributions.
- **Report inference latency and memory:** Even a single representative comparison would clarify efficiency claims.
- **Clarify the refinement protocol:** State explicitly whether the 100-step refinement for MVDUSt3R was applied to baselines that use similar 3D reconstruction backbones.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
