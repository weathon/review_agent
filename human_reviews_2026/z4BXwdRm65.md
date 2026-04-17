# CannyEdit: Selective Canny Control and Dual-Prompt Guidance for Training-free Image Editing

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4

## Abstract
Recent advances in text-to-image (T2I) models have enabled training-free regional image editing by leveraging the generative priors of foundation models. However, existing methods struggle to balance text adherence in edited regions, context fidelity in unedited areas, and seamless integration of edits. We introduce ***CannyEdit***, a novel training-free framework that addresses this trilemma through two key innovations. First, *Selective Canny Control* applies structural guidance from a Canny ControlNet only to the unedited regions, preserving the original image's details while allowing for precise, text-driven changes in the specified editable area. Second, *Dual-Prompt Guidance* utilizes both a local prompt for the specific edit and a global prompt for overall scene coherence. Through this synergistic approach, these components enable controllable local editing for object addition, replacement, and removal, achieving a superior trade-off among text adherence, context fidelity, and editing seamlessness compared to current region-based methods. Beyond this, CannyEdit offers exceptional flexibility: *it operates effectively with rough masks or even single-point hints in addition tasks*. Furthermore, the framework can seamlessly integrate with vision-language models *in a training-free manner* for complex instruction-based editing that requires planning and reasoning. Our extensive evaluations demonstrate CannyEdit's strong performance against leading instruction-based editors in complex object addition scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the fundamental trilemma in regional image editing - balancing text adherence, context fidelity, and editing seamlessness - by introducing CannyEdit, a novel training-free framework. The method features two key innovations: Selective Canny Control that applies structural guidance exclusively to unedited regions, preserving original details while enabling precise text-driven modifications; and Dual-Prompt Guidance that utilizes both local editing instructions and global scene descriptions to ensure accuracy and coherence. This approach supports flexible interactions ranging from rough masks to single-point hints, and seamlessly integrates with VLMs for complex instruction-based edits requiring planning and reasoning. Comprehensive evaluations demonstrate CannyEdit's superior performance in challenging object addition scenarios compared to state-of-the-art methods.

### Strengths
- The proposed editing method can precisely locate editing regions, support more flexible editing operations, and deliver highly faithful generation results.
- The designed approach effectively preserves the unedited regions of the image.
- The paper is clearly written, with well-presented comparative results and professionally crafted figures.

### Weaknesses
- The method appears to be heavily engineered, with the overall image generation process resembling a combination of null-text inversion, ControlNet, and FLUX.
- Given the involvement of multiple models, the inference speed requires clarification through detailed runtime analysis.
- The current strategy may struggle with editing requests involving significant spatial transformations, such as shifting the viewpoint by 60 degrees or making objects "fly" in the image. The authors should address how such limitations could be overcome.

### Questions
While current image editing models can handle complex instructions involving world knowledge and spatial transformations, and the paper compares with state-of-the-art models, these unified models offer comprehensive capabilities with user-friendly interfaces. I encourage the authors to further elaborate on what unique contributions their pipeline can provide in the context of rapidly advancing unified model research. I will determine my final score based on the authors' rebuttal and other reviewers' comments.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a training-free and model-agnostic approach for image editing. To preserve non-target pixels, the method adds a ControlNet model and injects masked ControlNet's features exclusively into unedited regions and guides edits with dual prompts (local for spatial precision, global for scene coherence). User point hints are translated by a VLM and refined into accurate masks, enabling a mask-based editing pipeline. The evaluation separates mask-based and instruction-only settings to ensure scope parity, and the appendix reports background-preservation metrics (PSNR/LPIPS/MSE) alongside Context Fidelity.

### Strengths
- The method requires no fine-tuning of the base diffusion model (i.e., FLUX.1-dev) and works with existing Canny-based ControlNets, making integration straightforward and model-agnostic.
- During denoising, Canny ControlNet feature maps are injected only into non-target pixels, stabilizing layout and preventing unintended changes in unedited regions.
- Dual (local/global) prompts and a VLM+SAM pipeline that converts point hints into accurate masks enable precise control and natural extension to a wide range of editing tasks.

### Weaknesses
**1. Minor Novelty of CannyEdit.** Despite using provided editing masks (or VLM+SAM–refined masks), background preservation lags far behind prior art: e.g., LPIPS (Appendix Tab. 4) shows KV-Edit 9.92 vs. CannyEdit 26.38, indicating that simply mixing ControlNet features into non-target pixels is insufficient to protect unedited regions. If the method’s core claim is “Selective Canny Control preserves the original structure,” then background-fidelity metrics must be strong in the main tables; moving them to the appendix while underperforming undermines the central contribution. In addition, comparisons to training-free background-preservation baselines that do not rely on masks (e.g., [1]) are limited and should be expanded. Also, the proposed dual-prompt guidance aligns with techniques widely used in layout-to-image generation [2–5]. Beyond direct application to editing, the paper should clarify what is technically new (e.g., objective, optimization, or inference mechanism) and why it matters for editing beyond prior formulations.

**2. Limited Experiments.** Although the approach should be compatible with any ControlNet-augmented UNet, experiments focus on FLUX.1-[dev] + FLUX-Canny-ControlNet only. The paper should delineate scope and limits, and test other controllable structures (e.g., GLIGEN [6], ControlNeXt [7]) where Canny/edge cues can serve as conditions. In Appendix C.4 (ControlNet Strength), background preservation should be evaluated with standard metrics (PSNR/SSIM/LPIPS refer to KV-Edit and [1]) on unedited regions to quantify whether increasing “strength” truly keeps non-targets intact. If the goal is “structure preservation,” it is natural to assess other edge/depth priors (e.g., HED, depth maps) and why Canny is preferred. The current ablations do not sufficiently justify this choice. Because ControlNet runs alongside the base model, compute and memory roughly double; the paper should report VRAM/latency vs. quality trade-offs to argue for practical value.

**3. Unclear Setup and Unfair Comparison.** Masks appear mandatory for the Selective Canny Control pipeline, yet several figures (e.g., Figure 6) are ambiguous about which mask (provided vs. refined) was used. Since Kontext/Qwen-Edit do not take masks, the paper should center comparisons within mask-based editing to avoid conflating settings and to attribute gains fairly. Background-preservation metrics must appear in the main tables (e.g., Table 1), not only in the Appendix. The primary claim is preservation of unedited regions; readers need headline numbers against strong baselines to gauge effectiveness. Hiding weaker numbers in the Appendix invites doubts about the method’s core contribution.

[1] Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing, ICCV 2025. 

[2] Region-Aware Text-to-Image Generation via Hard Binding and Soft Refinement, ICCV 2025. 

[3] DreamRenderer: Taming Multi-Instance Attribute Control in Large-Scale Text-to-Image Models, ICCV 2025. 

[4] NoiseCollage: A Layout-Aware Text-to-Image Diffusion Model Based on Noise Cropping and Merging, CVPR 2024. 

[5] GrounDiT: Grounding Diffusion Transformers via Noisy Patch Transplantation, NeurIPS 2024. 

[6] GLIGEN: Open-Set Grounded Text-to-Image Generation, CVPR 2023. 

[7] ControlNeXt: Powerful and Efficient Control for Image and Video Generation

### Questions
Q1. How does Selective Canny Control differ technically from simply mixing ControlNet features, and why should this yield superior background preservation (i.e., the original image's details)? Given this is the paper’s central claim, why are background-preservation metrics absent from the main tables (Tab. 1/2)? For instance, Appendix Tab. 4 reports LPIPS 9.92 (KV-Edit) vs. 26.38 (CannyEdit) under identical masks/inputs. What explains this gap, and can you provide failure cases (qualitative examples and per-region metrics) diagnosing where Selective Canny fails?

Q2. Dual-prompt guidance appears close to prior layout-to-image controls [2–5]. What is the editing-specific novelty beyond a simple adaptation, and can you ablate to show gains that prior prompting schemes cannot match (e.g., with/without dual prompts, prompt-mixing strategies, or attention routing ablations)?

Q3. Results focus on FLUX.1-[dev] + FLUX-Canny. To clarify scope and limits, can you report on additional control frameworks (e.g., classic U-Net + ControlNet, GLIGEN [6], ControlNeXt [7])? If the goal is structural fidelity, can you also test other priors (e.g., HED edges, depth) or provide evidence that Canny is the preferred/most stable signal?

Q4. What are the compute/memory overheads (e.g., VRAM and Latency) of running ControlNet alongside the base model? Please include a quality vs. cost table (background-preservation and perceptual metrics vs. FLOPs/VRAM/Latency) to substantiate practical viability.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents **CannyEdit**, a *training-free* image editing framework designed to balance **editability and fidelity**. The method introduces two key components: **Selective Canny Control**, which preserves unedited structures using edge-based guidance, and **Dual-Prompt Guidance**, which combines local and global text prompts for coherent editing. A progressive mask refinement strategy further supports weak or point-based editing inputs. Experiments on **RICE-Bench** show that CannyEdit achieves superior text alignment and visual realism compared to recent baselines such as KV-Edit, BrushEdit, PowerPaint-FLUX, FLUX.Fill.

### Strengths
### Originality
1. Introduces a clear and well-motivated training-free editing framework addressing the editability-fidelity-seamlessness trade-off.
2. Overall originality is moderate: innovation lies more in integration and careful design than in novel algorithms.

### Quality
1. Strong empirical validation on both mask-based and instruction-based setups.
2. Selective Canny Control effectively preserves background structure while allowing flexible local edits.
3. Dual-Prompt Guidance improves text alignment and global coherence compared with single-prompt baselines.
4. The method supports weak or point-based inputs, enabling integration with VLM-driven editing.

### Significance
1. Offers a practical, flexible editing solution that bridges mask-based and instruction-based paradigms.
2. The framework’s ability to use VLM-inferred point hints makes it relevant for future reasoning-based editing systems.

### Weaknesses
1. Limited Novelty of Mechanisms
The two key modules—selective structural control and multi-prompt attention—mainly extend existing ControlNet and attention-masking strategies rather than introducing fundamentally new formulations. In particular, **Selective Canny Control** is conceptually similar to the edge-based structural guidance used in **MagicQuill[1] (Sec. 3.1, Para. 1)**, but the paper does not explicitly clarify how it differs.

2. Efficiency Unclear
Although the method is “training-free,” it still involves inversion, ControlNet caching, and multi-prompt attention, all of which can be computationally expensive. The paper does not report quantitative runtime or memory comparisons against strong training-free/editing baselines such as KV-Edit or PowerPaint-FLUX, and the comparison with Qwen-Image-Edit in the appendix is not sufficient to assess practical efficiency — a more complete evaluation is needed.

3. Clarity and Structure Issues

The **Method** section feels overly long and dense, which makes it hard to follow. Some parts — for example, *Dual-Prompt Guidance* — would be much clearer with an attention diagram instead of long text descriptions, while the more basic *Preliminaries* could easily be moved to the appendix. 

Because method section runs almost to page 7, many important experiments and analyses (like the ablation studies) are pushed into the appendix, leaving the main paper incomplete. Ideally, the core technical and experimental content should appear in the main text, with the appendix used only for supporting details.

[1] Liu, Zichen, et al. "Magicquill: An intelligent interactive image editing system." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Questions
1. What is the actual inference cost (time and GPU memory) compared to KV-Edit or PowerPaint-FLUX?

2. The proposed Selective Canny Control appears conceptually similar to the edge-based structural guidance in MagicQuill [1] (Sec. 3.1, Para. 1). Could the authors explicitly clarify the key differences in formulation or implementation, and explain how CannyEdit advances beyond prior edge-guided editing methods?

3. CannyEdit relies heavily on Canny-based structural guidance, which may underperform in texture-rich or low-edge regions (e.g., skies or artistic styles). How does the method handle such cases? How strong is the generalization of this method?

### Soundness
3

### Presentation
3

### Contribution
3
