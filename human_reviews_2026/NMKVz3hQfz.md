# OmniPainter: Global-Local Temporally Consistent Video Inpainting Diffusion Model

- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Video inpainting methods often fail to resolve the inherent trade-off between long-term global consistency and short-term local smoothness, leading to artifacts such as contextual drift and flickering. We introduce OmniPainter, a latent diffusion framework designed to address this limitation. Our framework is built on two core innovations: a Flow-Guided Ternary Control mechanism for superior structural fidelity, and a novel Adaptive Global-Local Guidance strategy. This guidance strategy dynamically blends two complementary guidance scores at each denoising step: an autoregressive score to enforce local transitional smoothness, and a hierarchical score to maintain long-range global coherence. The blending weight is determined by a function of both the video's motion dynamics and the current diffusion timestep. This adaptive blending allows the model to prioritize global structure during the early stages of generation and then shift focus to local continuity during later refinement stages, thereby achieving a robust temporal equilibrium. Extensive experiments confirm that OmniPainter significantly outperforms state-of-the-art methods, setting a new standard for temporally consistent video restoration.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces OmniPainter, a effective latent diffusion model-based video inpainting framework specifically designed to address temporal inconsistencies. The approach achieves both global and local temporal consistency by preserving overall context across extended sequences while ensuring smooth transitions throughout. The framework introduces two key contributions : (1) a ternary control mechanism that categorizes regions based on inpainting needs and leverages flow-based video completion as a prior, and (2) adaptive global-local guidance that dynamically blends two complementary strategies during
the denoising process. Extensive experiments demonstrate significant improvements compared to previous state-of-the-art methods.

### Strengths
Comprehensive quantitative evaluation against 11 methods shows consistent improvements across multiple metrics (PSNR, SSIM, VFID, warping error).

### Weaknesses
1. Limited novelty. This work appears to make incremental improvements within an existing framework, without fundamental innovation. The novelty of the overall framework should be more clearly articulated, compared with DiffuEraser and ProPainter. And the authors take an older off-the-shell U-Net architecture (e.g., AnimateDiff-style), and it is unclear whether the proposed contributions transfer to modern DiT backbones. For example, does the Ternary Mask remain effective when the latents are temporally compressed by a 3D VAE?
2. The paper does not provide sufficient details on the flow-based video completion component, and its quantitative contribution to the overall results is unclear. This component appears similar to ProPainter's image propagation module, while the authors do not provide details on the optical flow completion. 
3. Insufficient Comparison. The paper lacks comprehensive comparisons with SOTA diffusion-based approaches. 
    - VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control [SIGGRAPH 2025]
    - MiniMax-Remover: Taming Bad Noise Helps Video Object Removal
4. Poor Results: The supplementary results provided by the authors show noticeable temporal inconsistency and visual artifacts. And the scenario is very simple, featuring minimal camera motion and negligible occlusion, which limits its applicability to real-world scenarios.

### Questions
Refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces OmniPainter, a latent diffusion framework for temporally consistent video inpainting. The core contribution is the introduction of an Adaptive Global-Local Guidance strategy and a Flow-Guided Ternary Control mechanism. The Adaptive Guidance dynamically blends an Autoregressive (AR) guidance for local smoothness and a Hierarchical (HR) guidance for long-range global coherence. This guidance has an adaptive weight determined by video motion and diffusion timestep. The Ternary Control utilizes optical flow reliability to partition masked regions into three categories: full inpainting, preservation, and crucial refinement of the flow-warped prior. Effectively, it provides a strong prior for the diffusion model to condition on. Experiments demonstrate that OmniPainter significantly outperforms baselines methods in both visual quality and temporal consistency metrics on several standard datasets.

### Strengths
The paper effectively addresses a critical trade-off between global consistency (long-term structure) and local smoothness (short-term transitions) in video inpainting. To the best of my knowledge, conditioning a diffusion model on the optical flow input with a ternary control mechanism with a ternary mask is novel. Combining hierarchical and autoregressive signals is not novel (see [1,2] for example), but _dynamically_ combining them based on scene motion and the diffusion timestep is novel.

The proposed Flow-Guided Ternary Control is a smart, nuanced solution to overcome the limitations of relying solely on imperfect flow-based priors or relying entirely on the diffusion model. The ablation on the refinement weight $\beta$ clearly demonstrates the superiority of the proposed refinement approach, which significantly enhances structural stability and detail preservation.

[1] Harvey, W., Naderiparizi, S., Masrani, V., Weilbach, C. and Wood, F. Flexible diffusion modeling of long videos. NeurIPS 2022.

[2] Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M. and Fleet, D.J. Video diffusion models. NeurIPS 2022.

### Weaknesses
**Clarity and Completeness of Method Description**

- The paper suffers from missing definitions for key metrics. While references are provided, a concise, high-level explanation of the metrics (e.g., $TC\_{local}$, $TC\_{global}$) and a clear indication of whether higher or lower values are better should be included for better reader comprehension.
- A comprehensive algorithm that synthesizes the entire denoising process is necessary for reproducibility. While the multiple parts of a denoising steps is separately described in the text (the Ternary Control LDM, AR score, HR score, and Adaptive Guidance), it is difficult to fully understand how exactly all components interact per step.

**Justification for Adaptive Guidance**

- Figure 3 is very confusing. The x-axis labels are not clearly indicated in the figure itself, forcing the reader to rely on the caption. Furthermore, different metrics are provided in the left and right panel, without justification (the left panel is missing the $TC\_{global}$ curve, and the right panel is missing all metrics except the TC ones.)
- Critically, no justification is provided for why and how the right panel of Figure 3 demonstrates the superiority of the "HR first" strategy. It is not immediately obvious to me from the presented curves. A clearer explanation of why this specific pattern on the graph motivates the adaptive blend is needed.

**Related Work**
The Related Work section on Diffusion Models in video generation seems to overlook some of the earliest and most relevant foundational works, specifically those introducing the first video diffusion models (like [1,2,3] from 2022) . Notably, the "Hierarchy-2" sampling scheme in FDM appears conceptually close to the idea of combining hierarchical and autoregressive approaches for long-sequence generation, which is central to OmniPainter's contribution.

[1] Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M. and Fleet, D.J. Video diffusion models. NeurIPS 2022.

[2] Harvey, W., Naderiparizi, S., Masrani, V., Weilbach, C. and Wood, F. Flexible diffusion modeling of long videos. NeurIPS 2022.

[3] Yang, R., Srivastava, P. and Mandt, S.. Diffusion probabilistic modeling for video generation. Entropy 2023.

### Questions
- **Metric Definitions**: Please add a brief, high-level description of all reported metrics (PSNR, SSIM, VFID, $E\_{warp}$, $TC\_{local}$, $TC\_{global}$) in Section 4.1 or a table footnote, and explicitly state whether higher or lower is better for each, which is essential for a general audience.

- **VAE Encoder/Partial Frames** (Line 190): The masked video $\hat{x}\_t$ is encoded via the VAE encoder to get the latent $z_t$. Since $\hat{x}\_t$ is only a partial frame, is the pre-trained VAE encoder robust to this out-of-distribution input? Did the authors observe any adverse effects, and if so, how were they mitigated during training or inference?

- **Notation for Conditional Scores**: Eq. (5) and (6) define $s\_{AR}(t, k)$ and $s\_{HR}(t, k)$ respectively, but they do not explicitly show conditioning on the masked latent $\mathbf{z}$ and the mask $\hat{\mathbf{m}}$, which is implied by the text and Equation (8). Please clarify if $\mathbf{z}$ and $\hat{\mathbf{m}}$ are implicit conditioning inputs for these scores, and if so, update the notation in the equations for consistency and clarity.

- **Inference speed**: Given the complex inference pipeline (flow computation, encoding, base LDM denoising, AR/HR score estimation, and adaptive guidance weight), what is the latency of OmniPainter compared to the baselines? Specifically, please provide a comparison of the time taken per frame or per video for OmniPainter versus all relevant baselines?

- In the caption of figure 3: FVID -> VFID?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces OmniPainter, a latent diffusion framework for video inpainting that resolves the fundamental trade-off between long-term global consistency and short-term local smoothness. The method features two key innovations: (1) a Flow-Guided Ternary Control mechanism that partitions masked regions into three categories (full inpainting, preservation, and refinement) using optical flow priors, and (2) an Adaptive Global-Local Guidance strategy that dynamically blends autoregressive (AR) and hierarchical (HR) scores based on motion dynamics and diffusion timesteps. Extensive experiments demonstrate that OmniPainter outperforms state-of-the-art methods in both visual quality and temporal coherence, significantly mitigating artifacts like flickering and contextual drift while preserving structural fidelity through its ternary refinement approach. The framework achieves robust temporal equilibrium by prioritizing global structure in early denoising stages and local continuity in later refinement stages.

### Strengths
1. This paper presents a novel and well-motivated solution to the critical challenge of balancing global consistency and local smoothness in video inpainting through its innovative ternary control mechanism.

2. It demonstrates strong theoretical grounding with its adaptive guidance approach that intelligently combines autoregressive and hierarchical strategies based on motion dynamics and diffusion timesteps.

3. The experimental results are comprehensive and convincing, showing clear improvements over state-of-the-art methods across multiple metrics while including important ablation studies. The experiments are solid.

4. The work has significant practical value with its demonstrated effectiveness on real-world tasks and potential for broader application in video restoration and generation domains.

### Weaknesses
1. The method described in this paper was tested on the outdated SD1.5 and its migration performance was not tested on the modern DiT architecture. Currently, DiT has become mainstream in the field of video generation.
2. This article does not specifically test inpaint analysis and comparison in extreme scenarios such as rapid occlusion, dynamic blur, multi person, and multi arms situations; The reviewer believes that this is a challenge for video editing.

### Questions
1. Can this method run on edge devices such as mobile phones ?
2. Are the indicators in Table 1 calculated for the entire graph or only for the edited region?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
I think the paper addresses video inpainting with a diffusion framework that combines (1) flow-guided ternary control (choose to inpaint, refine a warped prior, or keep) and (2) adaptive global–local guidance that mixes an auto-regressive prior (nearby frames) and sparse high-recall keyframe guidance as a function of motion and denoising time. The model is trained at 512² and uses an LCM distillation for 8-step inference, with experiments on DAVIS/YouTube-VOS and additional temporal-consistency proxies.

### Strengths
(1) The ternary control is a clear way to balance preservation vs. hallucination; the formulation is simple to reproduce and matches the problem.

(2) Conditioning on previous groups vs. keyframes at different denoising stages is intuitive and helps explain performance improvements.

(3) Using an SD-inpainting backbone + motion module and LCM acceleration makes the approach practical to implement.

(4) Sweeps over ternary weighting and motion sensitivity show where gains come from, and single-component baselines (AR-only/HR-only) are helpful for attribution.

### Weaknesses
(1). I think the novelty is a little bit incremental. The adaptive guidance largely amounts to when to condition on which cached predictions; it’s a useful systems recipe, but conceptually close to prior conditioning/concatenation schedules.   

(2). Temporal metrics could be broader. The CLIP-based consistency scores are informative, but I’d like to see triangulation with t-LPIPS, VMAF-temporal, or FVD-temporal to avoid bias toward a single embedding space.  

(3). Robustness to conditioning errors. Because the method reuses intermediate predictions (AR/HR), early artifacts may propagate. A sensitivity test (noise on the conditional latents or delayed conditioning) would strengthen the robustness story.  

(4). Training is at 512² with latent upsampling to 1080p+. Clear wall-clock and memory numbers for long sequences at 1080p, and Pareto curves vs. diffusion baselines (steps vs. quality) would make the practical value more concrete.

### Questions
(1). Do your gains persist under t-LPIPS or VMAF-temporal at both 512² and upsampled 1080p?   

(2). What happens if you perturb the AR/HR conditioning latents or offset them by a few denoising steps?  

(3). Can β be learned or adapted per-pixel using estimated flow uncertainty, and does this improve stability on fast/non-rigid motion?  

(4). What is the wall-clock and VRAM for a 300-frame 1080p sequence on A100/4090 (with and without latent upsampling)?

### Soundness
3

### Presentation
3

### Contribution
3
