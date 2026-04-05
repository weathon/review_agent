=== CALIBRATION EXAMPLE 97 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately reflects the core technical contribution: combining a multi-view reconstruction network with a video generator for text-to-3D synthesis.
- The abstract clearly outlines the problem (decoding gap and alignment issue in existing LDM-based 3D generators), the method (model stitching + direct reward finetuning), and the key results (state-of-the-art text-to-3DGS and text-to-pointmap generation).
- Claims in the abstract are generally supported by the experimental sections, though the phrase "markedly improve" should be tempered by acknowledging that some baselines (e.g., SDS-based methods) optimize per-scene and were evaluated post-refinement, while VIST3A is purely feedforward. The abstract appropriately notes the label-free nature of the stitching/alignment pipeline.

### Introduction & Motivation
- The problem is well-motivated. The critique that training custom decoders from scratch for LDM-based 3D generators is data-intensive and prone to misalignment with the generator's latent space is a recognized bottleneck in recent literature.
- Contributions are clearly stated and match the subsequent sections. The gap is correctly identified and positioned against the shift toward feedforward 3D foundation models.
- The introduction does not significantly over-claim, but it slightly undersells the potential computational and stability challenges of reward-based finetuning. The claim that stitching "requires only a small dataset and no labels" is accurate for the initialization phase, but the subsequent reward finetuning still relies on large-scale multi-view datasets (DL3DV-10K) and external scoring models (CLIP, HPSv2). This distinction should be more explicit to set accurate expectations.

### Method / Approach
- The method is described with reasonable clarity, and reproducibility is aided by detailed hyperparameters and architectural choices in Appendices B and C.
- **Key assumptions & justification:** The linear stitching assumption (Eq. 2) is empirically validated via MSE search across layers, and theoretically supported by citing the Lipschitz-bound on stitching risk (Eq. 4). This is sound. However, the theoretical bound assumes a known Lipschitz constant $\kappa_2$ for the downstream 3D decoder, which is not measured or reported. The practical validity of the bound remains hypothetical.
- **Logical gaps / Failure modes:**
  1. **Circular dependency in 3D consistency reward (Sec. 3.2, Reward component 3):** The reward compares decoded video frames to 3D renderings using *predicted* camera poses from the feedforward 3D model. If the 3D model predicts inaccurate poses, the consistency loss (L1 + LPIPS) will compare spatially misaligned images, potentially penalizing correct geometry or rewarding incorrect geometry that accidentally matches the erroneous pose. The paper does not discuss how pose errors are mitigated or weighted in the reward signal.
  2. **Decoder drift during reward alignment (Sec. 3.2 vs. Appendix B.1):** The reward finetuning explicitly updates only the generative model parameters $\theta$ (via LoRA). The stitched decoder $F_{k^*+1:l} \circ S$ is stated to be updated only during the initial pseudo-target finetuning phase. If the generator's latent distribution shifts significantly during reward tuning, the frozen stitched decoder could fall out of its expected input distribution, yet this is not monitored or addressed.
  3. **Truncated backpropagation (Algorithm 1, Appendix B.2):** Gradient computation is enabled for only $K=2$ steps out of $T \in [10, 50]$ denoising steps. Direct reward finetuning (DRTune) relies on gradient flow through the full denoising trajectory. Truncation likely weakens the reward signal, especially for early noisy latents. The paper justifies this for compute/memory but provides no ablation showing $K$'s impact on convergence or final reward alignment.

### Experiments & Results
- The experiments directly test the core claims across multiple benchmarks (T3Bench, SceneBench, DPG-Bench, RE10K, 7Scenes, ETH3D).
- **Baselines:** Comparisons to recent feedforward models (SplatFlow, Prometheus3D, VideoRFSplat, Director3D) are appropriate. However, Matrix3D-omni is also end-to-end latent-based but uses a differently trained decoder. The comparison to SDS-based baselines (Director3D, SplatFlow) mixes per-scene optimization with single-pass feedforward generation. While quality comparison is valid, the inference time, VRAM, and compute cost differences are never discussed, which is critical for ICLR's emphasis on practical methodology.
- **Missing ablations / statistical rigor:** 
  - All quantitative tables (Tables 1–5) report single-run results without error bars, standard deviations, or statistical significance testing. Given the known variance of LLM-as-a-judge and perceptual metrics, this is a notable omission.
  - No ablation isolates the impact of *stitching vs. training a decoder from scratch* on the same computational budget. The paper assumes stitching is superior but does not empirically prove it against a randomly initialized or lightly pretrained decoder.
  - DPG-Bench scores for VIST3A are consistently >75 (often ~85), while baselines hover around 50–70. Without prompt difficulty stratification or error analysis, it's unclear if these gains stem from genuine geometric fidelity or metric gaming by the CLIP/HPSv2 rewards.
- **Datasets & Metrics:** Benchmarks are appropriate. Relying heavily on Unified Reward LLM for evaluation is standardizing in the field but remains noisy. The evaluation protocol (Sec. 4.1, Appendix C.2) is thorough, but the lack of variance reporting weakens confidence in the claimed margins.

### Writing & Clarity
- The paper is generally well-structured and logically flows from problem formulation to method to experiments.
- **Clarity issue (Reward formulation):** Eq. 3 defines $L_{total} = L_{gen} - r(z_0, c)$, implying $r$ should be maximized. However, Appendix B.2 Eq. 6 defines $R_{consistency}$ with explicit negative signs. The notation conflates "reward" and "loss", making it unclear whether the optimization maximizes or minimizes each component. This should be reconciled for mathematical precision.
- Figures and tables are informative, though Fig. 4 lacks failure cases or boundary examples. Including at least one case where VIST3A produces artifacts or misinterprets complex spatial relations would improve critical transparency.

### Limitations & Broader Impact
- The authors acknowledge the sequential/coherent view ordering requirement (Sec. F), which is a valid and important constraint inherited from the video encoder.
- **Fundamental limitations missed:** 
  - Computational overhead of reward finetuning (DRTune + rendering loop) is not quantified. Training time, GPU requirements, and inference latency relative to baselines are absent.
  - Potential for reward hacking or saturation by CLIP/HPSv2 scores is not discussed. These 2D metrics do not strictly enforce 3D plausibility and can be gamed by texture smoothing or camera-stable hallucinations.
- **Broader impact:** ICLR expects a discussion of societal implications. The paper omits any mention of potential misuse (e.g., generating deceptive 3D scenes, copyright concerns with foundation models, environmental/compute costs of large-scale reward tuning). A dedicated paragraph on broader impact is required for ICLR standards.

### Overall Assessment
VIST3A presents a timely and conceptually elegant approach to a recognized bottleneck in latent text-to-3D generation: the weak, scratch-trained decoders and misalignment with the generative backbone. Leveraging model stitching to repurpose large, feedforward 3D reconstruction networks is a sound and empirically validated strategy, and the direct reward finetuning mechanism shows clear qualitative and quantitative improvements across multiple benchmarks. However, the paper's empirical claims are weakened by the absence of statistical rigor (no error bars, single-run tables), a potentially flawed 3D consistency reward that relies on unverified predicted camera poses, and truncated gradient backpropagation that may limit alignment efficacy. Additionally, the work lacks a comparison of computational cost/inference latency against baselines and misses required ICLR elements like a broader impact statement. These are addressable with additional experiments, clearer reward formulation, and variance reporting. If the authors provide statistical validation, clarify the reward/camera dependency, and discuss computational trade-offs and societal implications, the contribution would be solidly positioned for acceptance at ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents VIST3A, a framework for text-to-3D generation that stitches a pretrained video VAE encoder to a feedforward 3D reconstruction model (e.g., VGGT, AnySplat) via a learned linear layer, then aligns the video generator with this new decoder using direct reward finetuning. By reusing pretrained components instead of training custom 3D decoders from scratch, the method achieves state-of-the-art results on text-to-3DGS and text-to-pointmap generation across multiple benchmarks.

### Strengths
1. **Resource-Efficient Architecture Reuse:** The framework effectively bypasses the costly training of custom 3D VAE decoders from scratch by leveraging existing foundation models. The stitching layer requires minimal unlabeled data and is initialized via a closed-form least-squares solution (Sec 3.1, Eq. 2), preserving the pretrained knowledge of both modules.
2. **Theoretically and Empirically Grounded Component Selection:** The criterion for selecting the stitching layer (MSE minimization) is not only empirically correlated with final reconstruction quality but is also supported by a theoretical upper bound on stitching risk (Sec 4.4, Eq. 4, citing Insulla et al., 2025). The comparison with CKA further demonstrates the authors' rigorous approach to representation alignment.
3. **Comprehensive and Multi-Faceted Evaluation:** The paper evaluates multiple VAE-3D model pairings across object-centric (T3Bench), scene-level (SceneBench), and complex prompt (DPG-Bench) benchmarks. Strong quantitative results, a well-designed human study (Table 4), and extensive ablation studies on reward components and latent robustness (Fig. 5, 8, Table 6) provide convincing empirical validation.
4. **Practical Alignment Strategy via Direct Reward Finetuning:** Adapting direct reward finetuning (DRTune) to enforce 3D consistency and visual quality between the generator and decoder is a well-motivated solution to the latent misalignment problem. The gradient detachment strategy and randomized timestep sampling effectively address common instability issues in reward-based diffusion training.

### Weaknesses
1. **Perceived Incremental Technical Contribution:** Both model stitching and direct reward finetuning are well-established techniques. While their combination for text-to-3D is novel and timely, ICLR reviewers may view the contribution primarily as a clever engineering integration rather than a fundamental algorithmic advance in generative modeling or representation learning.
2. **Rigid Temporal/Sequential Input Requirement:** The reliance on a video VAE encoder forces multi-view images into a sequential order to mimic video coherence. As noted in the limitations (Appendix F) and evaluation protocol (Appendix C.2), this requires manual view arrangement and restricts the model's flexibility for arbitrary, unordered multi-view inputs common in real-world deployment.
3. **Heavy Reliance on Proxy VLM/Learned Rewards:** The alignment objective depends on CLIP, HPSv2, and the UnifiedReward LLM. These metrics are known to correlate imperfectly with true 3D geometry and can be susceptible to reward hacking (e.g., optimizing for sharp, high-contrast textures at the expense of structural correctness). The paper lacks an analysis of reward bias or potential mode collapse.
4. **Missing Computational & Scaling Analysis:** Direct reward finetuning through simulated denoising trajectories is notoriously compute-heavy. The manuscript does not report training wall-clock time, GPU memory requirements, or efficiency comparisons against baseline gradient-based multi-view finetuning, making it difficult to assess practical scalability.

### Novelty & Significance
**Novelty:** Moderate to high. The individual components (stitching, reward alignment) are prior art, but their targeted integration to bridge large video latent spaces with feedforward 3D decoders addresses a recognized bottleneck in current text-to-3D pipelines. The approach shifts the focus from training decoders to aligning latent distributions across independently pretrained systems.
**Clarity:** High. The paper is exceptionally well-structured, with clear mathematical formulations, logical progression from motivation to methodology to experiments, and thorough appendices detailing hyperparameters and ablation setups. Figures and tables effectively communicate key insights.
**Reproducibility:** High. The authors provide detailed training setups, loss formulations, reward definitions, and benchmark sampling protocols in the appendices. The use of standard, publicly available datasets and models, combined with a linked project page, strongly supports replication.
**Significance:** High for the 3D generation and multimodal alignment communities. VIST3A offers a practical, modular blueprint for leveraging existing foundation models without catastrophic retraining. It highlights the potential of latent-space stitching as a general strategy for combining specialized generative and discriminative models, aligning well with ICLR's emphasis on efficient, scalable representation learning.

### Suggestions for Improvement
1. **Clarify Terminology & Pipeline Structure:** The paper occasionally uses "end-to-end" to describe VIST3A, yet the method is explicitly a two-phase process (stitching initialization + reward finetuning). Adjust the terminology or explicitly frame it as a two-stage alignment pipeline to avoid overclaiming and set accurate expectations for implementation.
2. **Provide Computational Efficiency Metrics:** Include a subsection reporting training hours, GPU memory footprint for reward tuning, and inference latency compared to key baselines (e.g., Prometheus3D, VideoRFSplat). Discuss how the reward computation scales with the number of diffusion steps and latent dimensions.
3. **Analyze View-Ordering Sensitivity & Mitigation:** Systematically evaluate how performance degrades when input views are randomly shuffled versus smoothly ordered. If feasible, propose or discuss lightweight permutation-invariant adaptations (e.g., frame-level positional encoding adjustments) to relax the video-VAE constraint.
4. **Validate Reward-Guided Optimization Against Ground Truth:** While VLM rewards improve perceptual scores, add an analysis of how these proxy rewards correlate with objective geometric metrics (e.g., pointmap accuracy on 7-Scenes/ETH3D) during reward finetuning. This will help rule out reward hacking and confirm that structural fidelity is genuinely preserved.
5. **Strengthen Positioning in Related Work:** Expand the discussion to explicitly contrast VIST3A with very recent (2024/2025) latent 3D generation and decoder-free methods. Clearly articulate the unique trade-offs (e.g., flexibility vs. training cost, latent compatibility vs. custom decoder performance) to solidify the paper's contribution relative to fast-moving concurrent literature.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add ground-truth 3D geometric metrics (Chamfer Distance, F-Score) evaluated on datasets with true 3D annotations (e.g., Objaverse, ScanNet held-out splits). Without quantitative 3D shape evaluation, claims of "geometrically consistent 3D scenes" rely entirely on 2D render metrics that cannot distinguish plausible appearance from accurate structure.
2. Provide quantitative text-to-pointmap results against geometric baselines or ground-truth depth/pointmaps using metrics like point-to-plane error or cross-view reprojection consistency. Stating that benchmarks do not exist does not exempt a core claimed capability from rigorous validation at ICLR.
3. Report inference latency, VRAM usage, and FLOPs compared to baselines like Prometheus3D and Matrix3D-omni. Claims of methodological superiority are undermined without computability analysis, especially since gradient propagation through full reward-based denoising trajectories is notoriously expensive.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify the impact of sampling only two views for reward computation on global 3D consistency. This heuristic risks local overfitting; you must demonstrate how varying the reward view count affects cross-view reprojection error to prove the alignment genuinely enforces volumetric coherence rather than per-view plausibility.
2. Explain why linear MSE predicts stitching success within a single VAE but fails to correlate across different VAE families (Wan vs. SVD/CogVideoX). A spectral or effective-rank analysis of the respective latent spaces is needed; without it, the stitching layer search remains an opaque heuristic rather than a principled architectural contribution.
3. Analyze the trade-off surface between the 3D-consistency reward and texture blurriness, as evidenced by the sharp drop in metrics in Table 6. A Pareto analysis of reward weighting is required to prove the alignment strategy is robust and does not simply trade high-frequency detail for geometric over-smoothing.

### Visualizations & Case Studies
1. Display explicit failure cases highlighting severe multi-object geometric collapse or reward-induced artifacts, particularly for prompts with heavy occlusion where the 2-view reward likely fails. This exposes whether the method achieves true 3D consistency or merely masks structural errors with strong 2D texture priors.
2. Provide side-by-side visual comparisons of the original video decoder's frames versus the rendered stitched 3D outputs for identical prompts. Direct visualization of the domain gap is necessary to prove whether direct reward finetuning actually bridges the encoder-latent-decoder mismatch or leaves compounding misalignment artifacts.

### Obvious Next Steps
1. Evaluate domain generalization by testing on synthetic or rigid CAD data (e.g., ShapeNet) entirely outside the DL3DV/ScanNet finetuning distribution. This directly tests the central premise that stitching unlocks pretrained knowledge rather than merely memorizing the limited real-scene finetuning data.
2. Quantify the claimed inheritance of "prompt-based camera control" using a standardized trajectory tracking metric across a diverse prompt set. Two qualitative examples are insufficient to claim that semantic camera instructions reliably transfer to 3D space without inducing geometric distortion.

# Final Consolidated Review
## Summary
VIST3A introduces a framework for text-to-3D generation that stitches a pretrained video VAE encoder to a feedforward 3D reconstruction network via a learnable linear layer, then aligns the video generator to this hybrid decoder using direct reward finetuning. By repurposing existing foundation models rather than training custom 3D decoders from scratch, the method achieves state-of-the-art results on text-to-3DGS and enables high-quality text-to-pointmap generation across multiple benchmarks.

## Strengths
- **Principled component reuse via model stitching:** The framework bypasses costly decoder training by repurposing pretrained 3D reconstruction models. Layer selection is grounded in both empirical linear transfer error (MSE) and a theoretical Lipschitz-based upper bound on stitching risk (Sec. 3.1 & 4.4), avoiding arbitrary architectural choices.
- **Effective alignment via direct reward finetuning:** Adapting DRTune with a composite reward (perceptual quality + 3D rendering consistency) successfully bridges the latent-distribution gap between the video generator and 3D decoder. This yields consistent gains over multi-view supervised baselines and mitigates the geometric distortions typical of purely generative latent tuning (Table 6, Fig. 7).
- **Broad empirical validation across model pairings and benchmarks:** The method is tested across multiple video/3D combinations (Wan, Hunyuan, CogVideoX × AnySplat, VGGT, MVDUSt3R) and evaluated on T3Bench, SceneBench, DPG-Bench, and novel-view synthesis datasets. The strong quantitative margins, backed by a controlled human study, robustly support the core claim of improved geometric and semantic fidelity.
- **High clarity and reproducibility:** Mathematical formulations, hyperparameters, reward definitions, and dataset sampling protocols are thoroughly detailed in the main text and appendices. The closed-form stitching initialization and explicit LoRA configurations strongly support replication.

## Weaknesses
- **Heavy reliance on 2D proxy rewards without explicit 3D geometric validation:** The alignment objective optimizes CLIP, HPSv2, and LLM-based perceptual scores, supplemented by L1/LPIPS consistency between decoded frames and rendered views. Because the consistency term still operates in 2D pixel space, the model may learn to satisfy 2D plausibility priors (e.g., texture smoothing, view-stable hallucinations) rather than enforcing true volumetric correctness. The absence of ground-truth 3D metrics or cross-view reprojection error analysis during fine-tuning leaves geometric fidelity partially unverified.
- **Circular dependency in the 3D consistency reward:** The consistency reward compares decoded video frames to 3D renderings using camera poses *predicted* by the same stitched 3D model. Systematic pose estimation errors will misalign the comparison, potentially penalizing correct geometry or rewarding structures that accidentally match an erroneous viewpoint. The paper does not quantify or mitigate how pose uncertainty propagates into the reward signal.
- **Missing computational and scaling analysis:** Direct reward finetuning through simulated denoising trajectories is notoriously compute-intensive, and the pipeline requires rendering loops during training. The manuscript reports no wall-clock training time, VRAM footprint, or inference latency relative to baselines (e.g., Prometheus3D, SplatFlow). Without efficiency metrics, the practical trade-offs of this alignment strategy remain opaque, especially given the field's shift toward faster feedforward pipelines.

## Nice-to-Haves
- Quantitatively evaluate the inheritance of text-driven camera control (e.g., trajectory tracking error across diverse prompts) rather than relying solely on two qualitative examples.
- Provide an ablation on reward view sampling (e.g., 2 views vs. 4+ views) and analyze how increasing sampled viewpoints impacts global 3D coherence vs. training overhead.
- Include explicit failure cases highlighting prompts where geometric collapse or reward-induced artifacts occur, particularly for highly occluded or multi-object scenes, to clarify the method's boundaries.

## Novel Insights
The paper demonstrates that independently pretrained video latent spaces and feedforward 3D reconstruction networks share highly compatible intermediate representations, enabling seamless, label-free stitching via a simple linear transformation. This challenges the prevailing paradigm of training custom 3D decoders from scratch, revealing that the primary bottleneck in latent text-to-3D generation is not architectural design but latent-distribution alignment. Furthermore, shifting from conventional generative losses to direct reward finetuning with differentiable rendering feedback proves more effective at grounding the generator's outputs in the decoder's geometric prior than iterative rendering losses or multi-view supervision alone.

## Suggestions
- Incorporate objective 3D geometric evaluation during or after alignment, such as cross-view reprojection consistency on held-out real data or Chamfer/L1 distance on a synthetic subset (e.g., Objaverse/ShapeNet) with ground-truth meshes. This will decouple 2D perceptual optimization from true 3D structural correctness.
- Report training wall-clock hours, GPU memory usage, and average inference latency for VIST3A compared to key baselines, including the overhead introduced by the reward rendering loop and LoRA-based DRTune.
- Analyze the sensitivity of the consistency reward to pose estimation errors. A simple experiment injecting controlled pose noise or comparing reward stability with/without pose optimization would clarify whether the alignment signal remains robust in practical deployment.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
