=== CALIBRATION EXAMPLE 88 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately captures the core technical contribution: combining a video generator with a 3D reconstruction network via stitching and alignment.
- The abstract clearly outlines the problem (decoders trained from scratch in prior LDM-based 3D generators, misalignment between generative latents and 3D decoders), the proposed solution (model stitching to reuse pretrained 3D foundation models, direct reward finetuning for alignment), and key results (outperforms prior text-to-3DGS baselines, extends to pointmap generation).
- Claims in the abstract are well-supported by the experimental sections (Tables 1–3 show consistent quantitative gains over baselines; Table 5 and Fig 14 validate pointmap generation). No unsupported overclaims are present. The phrasing "All tested pairings markedly improve" is accurate per the reported results.

### Introduction & Motivation
- The motivation is strong and timely. The argument that repurposing 2D VAEs as 3D decoders is inefficient and lags behind dedicated SOTA feedforward 3D models is a genuine pain point in the community. The identification of weak alignment between generated latents and decoder input domains is also accurate.
- Contributions are clearly stated in Section 1: (1) self-supervised model stitching to build a 3D VAE with minimal fine-tuning, (2) direct reward finetuning to align the generative diffusion process with the stitched decoder, and (3) comprehensive validation across architectures and tasks.
- The introduction does not over-claim or under-sell. It correctly positions VIST3A as a framework that unlocks existing models rather than proposing a new generative architecture from scratch, which aligns well with current ICLR trends favoring modular, efficient adaptation over monolithic retraining.

### Method / Approach
- The method is clearly described and largely reproducible. Section 3.1 details the layer-search procedure (Eq. 2) and the closed-form initialization. Section 3.2 and Appendix B.2 provide reward components and Algorithm 1 for direct reward finetuning.
- **Key assumption & justification:** The core assumption is that the video VAE latent space is linearly compatible with an intermediate layer of the feedforward 3D model. This is empirically validated via MSE minimization and supported by the stitching risk bound in Eq. 4 (Insulla et al., 2025). While the bound requires a Lipschitz constant $\kappa_2$ that is rarely verified for complex 3D backbones, the empirical trend (Fig. 5) justifies its practical use.
- **Logical gap / edge case in reward design (Sec 3.2, Eq 5-6, App B.2):** The 3D consistency reward computes $\ell_1$ and LPIPS between RGB frames decoded by the *original video decoder* $D$ and RGB frames *rendered* from the stitched 3D decoder $D_{\text{stitched}}$. These two decoders have fundamentally different training objectives and priors (temporal video synthesis vs. geometric point/Gaussian rendering). The consistency penalty may inadvertently conflate true 3D geometric inconsistency with inherent differences in how $D$ and the 3D renderer map to RGB space. The authors note in D.1 (Table 6, Fig 7) that adding the consistency reward alone degrades performance, which strongly hints at this mismatch. A brief discussion or sensitivity analysis on this cross-decoder comparison would strengthen the methodological rigor.
- **Alignment scope:** Reward finetuning only updates the video generator (via LoRA), leaving the stitched decoder frozen after Step 2. While standard in direct reward tuning diffusion models, if the initial stitching mismatch is non-trivial, optimizing only the generator might struggle to fully pull generated latents into the decoder's valid manifold. The paper assumes stitching error is small post-initialization; an ablation showing what happens if decoder LoRA is also unfrozen during alignment would be valuable.

### Experiments & Results
- The experiments directly test the paper's claims across multiple video backbones, 3D decoders, and benchmarks (T3Bench, SceneBench, DPG-Bench, RealEstate10K, 7-Scenes, ETH3D). Results consistently support the claims.
- Baselines are appropriate, recent, and fairly compared. The evaluation protocol (App C.2) carefully matches settings (e.g., 13 frames, 80 denoising steps, CFG 7.5) and accounts for SDS refinement where applicable.
- **Missing statistical reporting:** No error bars or standard deviations are reported across multiple random seeds. Given the stochastic nature of diffusion sampling and reward optimization, reporting variance would be important to confirm robustness. ICLR reviewers increasingly expect at least 3 seeds or explicit confidence intervals for key tables.
- **Ablation completeness:** The reward component ablation (Table 6) is well-designed. However, the scaling factors for the reward components (1/16 for quality, 0.05 for consistency, stated in App B.2) appear empirically tuned without a stated search protocol or sensitivity analysis. Clarifying whether these were grid-searched or held fixed across architectures would improve reproducibility.
- **Compute/efficiency:** The paper emphasizes efficiency by avoiding decoder retraining, but does not report GPU hours, training time for stitching vs. reward alignment, or memory overhead. Reward finetuning with full trajectory simulation, rendering, and LLM scoring is computationally heavy. ICLR values transparency in compute budgets, especially for RL-style fine-tuning loops.
- Datasets and metrics are standard and appropriate for the field. The shift toward Unified Reward (LLM-based) aligns with current evaluation trends, though it's worth noting that LLM judges can exhibit bias on fine-grained geometric attributes.

### Writing & Clarity
- The manuscript is well-structured and technically precise. The flow from problem identification to stitching, then to reward alignment, is logical.
- Figure 5 and the CKA visualization (Fig 6) effectively communicate the layer-index vs. performance correlation and similarity metric limitations.
- One clarity gap: Section 3.2 states "reward function consists of three components... compute these scores only on two sampled decoded views". The rationale for sampling exactly two views is not provided. Given that 3D consistency typically requires multi-view agreement, restricting rewards to two views may undersample geometric errors. Clarifying this trade-off (compute vs. coverage) would help readers understand the design space.
- Overall, no major clarity issues impede understanding. The mathematical notation and algorithmic description are standard and accessible.

### Limitations & Broader Impact
- The authors correctly identify the primary limitation in Section F: the video VAE encoder expects temporally coherent input, so arbitrary unordered multi-view sets must be sorted/padded, which may not generalize to in-the-wild multi-view datasets.
- **Missed limitations:** 
  1. The manual stitching layer search per VAE/3D-model pair (Sec 3.1, App C.1) is not fully automated. If a new 3D foundation model is released, users must run the scan again. Discussing potential heuristics for automatic layer selection or transferability across architectures would be valuable.
  2. The reward loop's computational cost and latency (rendering + LLM scoring per backprop step) is non-trivial but unquantified. This affects practical deployability.
- **Broader/Societal Impact:** The paper lacks a dedicated Broader Impact statement, which is standard for ICLR submissions involving generative models. Text-to-3D generation carries implications for asset copyright, synthetic media proliferation, and potential misuse in immersive environments. A brief, standard acknowledgment would satisfy conference expectations.

### Overall Assessment
VIST3A presents a well-motivated, practically significant framework that addresses a real bottleneck in current text-to-3D generation: the inefficient, from-scratch training of 3D decoders in LDM pipelines. By cleanly separating concerns (stitching to leverage SOTA 3D foundation models, reward tuning to align generative latents), the paper delivers consistent, substantial improvements across multiple benchmarks and output types. The methodological novelty is solid, and the empirical evaluation is comprehensive. Key concerns that require clarification include the cross-decoder comparison in the consistency reward, the lack of multi-seed statistical reporting, opacity around reward scaling hyperparameters and compute budgets, and the need for a brief broader impact discussion. These are all addressable through revisions or supplementary details. The contribution stands strongly despite these points and aligns well with ICLR's standards for methodological clarity, empirical rigor, and practical utility. I recommend acceptance pending clarification of the noted technical details.

# Neutral Reviewer
## Balanced Review

### Summary
VIST3A introduces a framework that combines a pretrained text-to-video latent generator with a feedforward 3D reconstruction model via a lightweight linear stitching layer, effectively repurposing existing 3D foundation models as decoders for 3D generation. To address the typical misalignment between latent-space generators and reconstruction decoders, the authors employ direct reward finetuning that propagates gradients through the denoising trajectory using a composite reward based on 2D image quality, 3D rendering quality, and multi-view consistency. Empirically, the method achieves state-of-the-art results across multiple text-to-3DGS and text-to-pointmap benchmarks while requiring no labeled training data.

### Strengths
1. **Elegant component reuse and theoretical grounding for stitching.** Instead of training custom 3D decoders from scratch (a major bottleneck in current LDM-based 3D generation), the paper identifies compatible layers between video VAE latents and pretrained 3D models using a closed-form least-squares fit. The choice is theoretically justified via Lipschitz continuity bounds (Eq. 4) and empirically validated through layer-index ablations (Fig. 5, App. E).
2. **Well-motivated alignment strategy with comprehensive ablations.** The adoption of direct reward finetuning effectively bridges the generator-decoder gap. The multi-component reward design (2D quality, 3D rendering quality, and cross-view consistency) is carefully dissected in Table 6 and Fig. 7, demonstrating how each term contributes to reducing ghosting, improving prompt adherence, and sharpening geometry.
3. **Strong, multi-backbone empirical validation.** The framework is tested across diverse video generators (Wan 2.1, SVD, Hunyuan, CogVideoX) and 3D backbones (MVDUSt3R, VGGT, AnySplat), consistently outperforming strong baselines on T3Bench, SceneBench, and DPG-Bench (Tables 1–3). The extension to text-to-pointmap generation further highlights the generality of the stitching paradigm.

### Weaknesses
1. **Heavy reliance on 2D proxy metrics for 3D evaluation.** The quantitative evaluation almost exclusively uses image-based metrics (CLIP, HPSv2, MUSIQ, LPIPS, L1) rendered from the generated 3D representations. While standard in recent text-to-3D literature, the absence of direct 3D geometric metrics (e.g., Chamfer Distance, F-Score, or surface normal consistency against ground truth meshes/scans) makes it difficult to verify whether the reward optimization truly improves 3D structure or merely optimizes 2D appearance proxies.
2. **Alignment algorithm novelty is incremental.** The direct reward finetuning procedure largely adapts existing techniques from 2D diffusion alignment (e.g., DRTune), with gradient detachment and partial step backpropagation. The primary innovation lies in the 3D-specific reward formulation rather than the optimization framework itself, which may limit perceived algorithmic novelty for the ICLR audience.
3. **Heuristic reward scaling and opaque compute budget.** The reward weighting scheme (1/16 for quality components, 0.05 for consistency in Appendix B.2) lacks principled derivation or sensitivity analysis. Additionally, while hyperparameters and batch sizes are provided, the total training compute (GPU hours/days) is not disclosed, making it harder to assess the practical cost of running the alignment stage compared to end-to-end 3D LDM training.

### Novelty & Significance
**Novelty:** Moderate to High. Model stitching and direct reward finetuning are established concepts, but their systematic integration for text-to-3D generation is novel and thoughtfully executed. The paper provides a clear paradigm for decoupling generative priors from geometric decoders, which is a fresh perspective compared to monolithic 3D LDMs or multi-stage pipelines.
**Clarity:** High. The paper is exceptionally well-organized, with clear methodological descriptions, intuitive figures (e.g., Fig. 2, Fig. 3), and logical progression from stitching to alignment to evaluation. Mathematical formulations are standard and correctly applied.
**Reproducibility:** Good. The authors disclose backbone models, dataset splits, optimizer settings, LoRA configurations, and reward compositions. A project page is provided, and the reliance on open weights (Wan, VGGT, AnySplat) lowers the barrier to replication. Full code release and explicit compute budgets would further strengthen this.
**Significance:** High. The method directly addresses two critical pain points in contemporary text-to-3D research: the decoder bottleneck and generator-decoder latent mismatch. By enabling plug-and-play integration of future video and 3D foundation models, VIST3A offers a scalable, modular pathway that can accelerate progress in generative 3D vision.

### Suggestions for Improvement
1. **Incorporate direct 3D geometric evaluation.** Supplement the rendered-image metrics with at least one ground-truth 3D evaluation (e.g., Chamfer Distance or normal consistency on ScanNet/ETH3D subsets or Objaverse scans) to conclusively demonstrate that the alignment improves actual 3D geometry rather than just perceptual quality.
2. **Provide a sensitivity analysis or principled justification for reward weights.** Explore how performance varies across different reward scaling factors, or consider a dynamic weighting scheme (e.g., uncertainty-based or curriculum-based) to reduce reliance on manually tuned constants.
3. **Disclose training compute and clarify code release.** Report approximate GPU-hours for both the stitching finetuning and reward alignment phases. If code is available, explicitly link to training scripts, reward computation pipelines, and environment setups to meet ICLR's reproducibility standards.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a head-to-head comparison against an inference-time composition pipeline (generate multi-view video → decode with frozen video VAE → run the original 3D reconstruction model). Without this, the claim that end-to-end stitching and alignment are necessary over a simple test-time pipeline remains unproven.
2. Include evaluations with ground-truth 3D geometry and camera metrics on a standard held-out dataset (e.g., Objaverse-LVIS or synthetic scenes). Relying exclusively on 2D proxy metrics (CLIP, HPSv2, VBench) cannot validate the core claim of improved 3D consistency.
3. Train a baseline custom 3D decoder from scratch using the exact same multi-view data, compute budget, and LoRA configuration as your stitched decoder. This directly tests the critical claim that stitching avoids costly decoder training without sacrificing output quality.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify whether backpropagating 2D aesthetic rewards (CLIP/HPSv2) through the denoiser into 3D parameters causes reward hacking, such as geometry over-smoothing or texture baking to match blurry decoded frames. Show metrics that decouple 2D appearance fidelity from true 3D structural accuracy.
2. Analyze the domain mismatch between the frozen video decoder (used as the reward reference) and the target 3D representation. If the reward forces 3DGS renders to match potentially inconsistent decoded video outputs, you must demonstrate that this bottleneck does not cap the achievable 3D fidelity.
3. Provide a data-scaling curve for the stitching fine-tuning stage. The claim that stitching "requires only a small dataset and no labels" is speculative without showing how reconstruction quality degrades as training scenes/epochs decrease below your reported setup.

### Visualizations & Case Studies
1. Show explicit, documented failure cases (e.g., highly self-occluded objects, complex topologies, or prompts requiring precise spatial relationships). Demonstrating where the stitched latent or reward signal fails is essential to establish the method's operational boundaries.
2. Provide side-by-side visualizations of the 2D video decoder output vs. the 3DGS render that drove the consistency reward on identical prompts. This exposes whether the alignment step is genuinely enforcing multi-view geometric consistency or merely forcing the 3DGS to mimic video generation artifacts.
3. Visualize gradient magnitudes or latent perturbations across different denoising timesteps during reward finetuning. This would reveal whether early or late noise steps dominate the alignment signal, validating your chosen gradient detachment and timestep sampling strategy.

### Obvious Next Steps
1. Report inference latency, peak VRAM, and parameter counts compared to SDS-based and multi-stage baselines. Positioning VIST3A as a practical, feedforward alternative requires the computational efficiency metrics ICLR reviewers expect, not just qualitative claims.
2. Run a sensitivity ablation on the fixed reward scaling factors (1/16 for quality, 0.05 for consistency). Without demonstrating robustness to these hyperparameters, the alignment framework appears highly tuned and its generalizability is questionable.
3. Compare linear stitching against a small non-linear alternative (e.g., a shallow MLP) to rigorously prove that a closed-form linear map is truly sufficient, rather than an architectural convenience that may be limiting representation capacity.

# Final Consolidated Review
## Summary
VIST3A introduces a compositional framework that stitches the encoder of a pretrained text-to-video VAE to a feedforward 3D reconstruction model, creating a high-fidelity 3D VAE decoder. To resolve generator-decoder latent misalignment, the authors employ direct reward finetuning with a composite objective spanning 2D quality, 3D rendering quality, and multi-view consistency. This approach bypasses costly decoder-from-scratch training and consistently outperforms prior text-to-3DGS methods across multiple benchmarks, while also extending to text-to-pointmap generation.

## Strengths
- **Principled component reuse via model stitching:** The closed-form least-squares layer matching elegantly repurposes SOTA feedforward 3D models as decoders, avoiding the data-hungry and geometrically limited training pipelines of prior 3D LDMs. The approach is theoretically grounded (Lipschitz continuity bounds) and empirically validated via MSE-index correlation (Fig. 5, Table 5).
- **Strong, multi-backbone empirical performance:** The framework demonstrates consistent, significant gains across diverse video generators (Wan 2.1, SVD, Hunyuan, CogVideoX) and 3D decoders (MVDUSt3R, AnySplat, VGGT) on T3Bench, SceneBench, and DPG-Bench. The preservation of pointmap accuracy and camera pose estimation post-stitching confirms the approach effectively transfers 3D reasoning capabilities.
- **Targeted alignment ablations:** Table 6 and Fig. 7 systematically isolate the contributions of multi-view, quality, and consistency rewards, demonstrating that the composite signal is necessary to suppress temporal ghosting while maintaining prompt adherence and visual sharpness.

## Weaknesses
- **Circular evaluation heavily reliant on 2D proxies:** The primary text-to-3D generation evaluation (Tables 1–2) relies almost exclusively on image-based metrics (CLIP, HPSv2, MUSIQ, VLM judges). Because the reward function directly optimizes these exact proxies, the evaluation cannot disentangle whether the alignment genuinely improves 3D geometric structure or merely overfits to 2D perceptual heuristics. Table 6 explicitly shows the consistency reward alone causes structural degradation (over-blurring), indicating the optimization trajectory is highly sensitive to these proxy signals.
- **Inherent bottleneck in the cross-decoder consistency reward:** The 3D consistency loss penalizes discrepancies between RGB frames decoded by a *temporal video decoder* (trained for plausibility, not geometry) and frames rendered from the 3D representation. These decoders have fundamentally different inductive biases and error modes. Conflating true geometric inconsistency with decoder-specific rendering artifacts creates an unstable optimization signal and likely caps the achievable 3D fidelity, yet the paper provides no analysis of this domain-gap error propagation.
- **Opaque hyperparameter tuning and missing compute transparency:** Critical reward scaling weights (1/16 for quality, 0.05 for consistency) are manually fixed without sensitivity analysis or justification. More importantly, no compute budget (GPU hours, wall-clock time, memory footprint for the full trajectory sampling + reward computation pipeline) is reported. Without these details, it is impossible to verify whether the claimed efficiency gains over end-to-end 3D LDM training hold in practice, or if the method simply shifts the bottleneck to expensive RL-style finetuning loops.

## Nice-to-Haves
- Report variance/error bars across multiple random seeds for key generative tables to confirm optimization stability.
- Quantify inference latency, peak VRAM, and training wall-clock time to concretely substantiate the "lightweight/feedforward" efficiency claims.
- A brief sensitivity analysis on the fixed reward scaling factors would improve generalizability and reduce reliance on manual tuning.

## Novel Insights
The core insight—that intermediate representations of modern feedforward 3D reconstruction networks are linearly compatible with the latent space of independent video generative models—challenges the field's reliance on co-trained, monolithic 3D decoders. By treating the 3D backbone as a frozen geometric module and using reward gradients to bend the generative trajectory toward its valid input manifold, the paper demonstrates a practical, modular alternative to from-scratch training. The empirical finding that a simple closed-form linear map suffices to preserve SOTA reconstruction accuracy suggests that foundational 2D/video priors and feedforward 3D networks share deeper, transferable representational structure than previously assumed, opening a viable design space for compositional generative systems over end-to-end retraining.

## Potentially Missed Related Work
- Wu et al. (2024c) "Deep reward supervisions for tuning text-to-image diffusion models" and Clark et al. (2024) "Directly fine-tuning diffusion models on differentiable rewards" — These establish the direct reward finetuning baseline; a clearer contrast with your timestep-sampling and gradient-detachment modifications would sharpen the algorithmic contribution.
- Concurrent latent-space 3D alignment works (e.g., Lin et al. 2025, Schwarz et al. 2025) — Briefly discussing how their rendering-aligned objectives differ from your cross-decoder consistency reward would clarify the novelty of your alignment strategy.

## Suggestions
- **Decouple evaluation from optimization:** Incorporate direct 3D geometric metrics that are explicitly excluded from the reward function (e.g., Chamfer Distance, F-Score, or surface normal consistency against ground-truth meshes/scans from Objaverse-LVIS or ScanNet). This is necessary to prove that reward alignment improves true 3D structure rather than merely exploiting 2D proxy loopholes.
- **Analyze and bound the cross-decoder mismatch:** Provide a controlled experiment or theoretical discussion quantifying how much of the consistency reward signal stems from genuine 3D misalignment vs. inherent video-decoder vs. 3D-renderer prior differences. Consider decoupling consistency supervision from the video decoder entirely (e.g., using multi-view feature consistency or depth/LPIPS directly on rendered views) to remove this bottleneck.
- **Disclose full training compute and hyperparameter robustness:** Report total GPU-hours, memory usage, and wall-clock time for both stitching and reward phases. Include a small sensitivity sweep over the reward scaling factors to demonstrate the method isn't fragile to the fixed 1/16 and 0.05 weights.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
