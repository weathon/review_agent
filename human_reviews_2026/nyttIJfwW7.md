# FlowAlign: Trajectory-Regularized, Inversion-Free Flow-based Image Editing

- Decision: Accept (Poster)
- Scores: 2, 6, 2, 6

## Abstract
Recent inversion-free, flow-based image editing methods such as FlowEdit leverages a pre-trained noise-to-image flow model such as Stable Diffusion 3, enabling text-driven manipulation by solving an ordinary differential equation (ODE). While the lack of exact latent inversion is a core advantage of these methods, it often results in unstable editing trajectories and poor source consistency. To address this limitation, we propose {\em FlowAlign}, a novel inversion-free flow-based framework for consistent image editing with principled trajectory control. FlowAlign introduces a flow-matching loss as a regularization mechanism to promote smoother and more stable trajectories during the editing process. Notably, the flow-matching loss is shown to explicitly balance semantic alignment with the edit prompt and structural consistency with the source image along the trajectory. Furthermore, FlowAlign naturally supports reverse editing by simply reversing the ODE trajectory, highliting the reversible and consistent nature of the transformation. Extensive experiments demonstrate that FlowAlign outperforms existing methods in both source preservation and editing controllability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
FlowAlign is a flow-based approach for text-guided image editing that operates without additional training or inversion. Unlike prior methods such as FlowEdit, which suffer from unstable trajectories and source degradation, FlowAlign introduces trajectory regularization using terminal structural similarity to address these issues. This allows the method to achieve a balanced trade-off between semantic alignment and structural consistency, resulting in smooth and deterministic transformations. Moreover, FlowAlign achieves computational efficiency by requiring only one additional function evaluation (NFE) per ODE step and enables near-perfect reconstruction of the source image via backward ODE integration.

### Strengths
1.FlowAlign performs efficient editing without requiring inversion or additional training. In addition, FlowAlign is more efficient than FlowEditing.

2.FlowAlign achieves smooth and stable trajectories through trajectory regularization.

3.FlowAlign It balances semantic alignment and structural consistency effectively.

### Weaknesses
1.My main concern is the comparison with FlowCycle [1]. FlowCycle presents a more advanced approach than the proposed FlowAlign.

2.There are some questionable aspects in the figures and tables presented by the authors. Please refer to the "Questions" Section for specific details.

3.The diversity and explanation of CFG settings appear insufficient. Although the authors acknowledge that CFG is a highly critical parameter, they provide limited experimental results, using only fixed CFG settings for the baselines.

[1]FLOWCYCLE: PURSUING CYCLE-CONSISTENT FLOWS FOR TEXT-BASED EDITING

### Questions
1.The most important point is the comparison with FlowCycle. FlowCycle demonstrates superior performance compared to FlowAlign — can the authors explain why FlowAlign performs better than FlowCycle?

2.As mentioned in the Weaknesses Section, there are several questionable aspects in the presented results. The main ones are as follows:

2-1. In Figure 3(a), did the authors indeed set the CFG scales to {5.0, 7.5, 10.0, 13.5}? If so, were the source CFG and target CFG values identical for the baselines? If they were the same, this does not match the baseline settings described elsewhere (in other tables, figures, or the Implementation Details section). In particular, for FlowEdit, if the source CFG values were set to {5.0, 7.5, 10.0, 13.5}, as far as I know, this is not a standard setting.

2-2. While it is reasonable to point out the limitations of CLIP scores, the Human Evaluation results in Figure 3(b) and the CLIP scores in Table 3 are highly inconsistent. The discrepancy is too large to attribute merely to the limitations of CLIP score. Why does FlowAlign show better Human Evaluation results but lower CLIP scores?

3.The key idea of FlowAlign seems to be its improved efficiency by computing CFG only for 'v'. However, to convincingly demonstrate superior performance over the baselines, more comparisons involving 'source CFG' and 'target CFG' variations should be provided. How does the performance vary across baselines depending on the source and target CFG values?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces FlowAlign, a flow-based image editing framework that builds upon a prior inversion-free approach, FlowEdit. While this method demonstrated that direct editing without inversion is feasible, FlowAlign focuses on improving the smoothness and consistency of the editing trajectory. To achieve this, it introduces source similarity at the terminal point as a regularization term, encouraging trajectories that preserve structural consistency with the source image while following the intended semantic edits.

### Strengths
1. Good results with clear qualitative comparisons – the paper presents convincing visual examples that highlight the differences between FlowAlign and baselines. The figures clearly show improved source structure preservation and accurate semantic edits.
2. Human preference study – beyond quantitative metrics, the authors conduct a user study showing that participants consistently prefer FlowAlign’s outputs over baselines. As far as I know, this is the first time human preference is reported for these low-based editing methods.
3. Demonstrated across multiple settings – The method is evaluated on 2D image editing, video editing, and 3D Gaussian splatting, showing versatility and robustness across different tasks.

### Weaknesses
1. Trajectory regularization intuition – while trajectory regularization appears effective, the paper could provide a deeper explanation of why smoother trajectories improve editing outcomes. It seems intuitive that smoother trajectories help preserve the original image structure, but it is unclear what trade-offs or costs this introduces. For example, could this regularization limit the strength or flexibility of edits, or bias the model toward minimal changes? 

2. Figure 2 clarity – figure 2 is somewhat confusing. It is not clear which panels correspond to FlowEdit versus FlowAlign. What does “R” represent? This term does not appear in the text. Additionally, the velocity field mentioned in the caption is not visually represented in the figure, making it difficult to interpret the intended trajectory differences. Clarifying these points would improve readability and understanding of the proposed method.

3. It is unclear whether the paper uses the same early-timestep skipping as FlowEdit.

### Questions
I raised question in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes FlowAlign, an inversion-free image editing method for rectified flow models. Through terminal point regularization, the balance between semantic alignment and structural consistency can be improved. Extensive experiments illustrate the superior performance of the proposed method.

### Strengths
- The method is training-free and inversion-free, which can be applied on various off-the-shelf flow-based models.
- Compared to the baseline methods such as FlowEdit, the proposed method is more computationally efficient.
- The paper is well-written and easy to follow.

### Weaknesses
- The motivation of this paper is not clear.  Authors claim that 'these limitations of FlowEdit arise from non-smooth and unstable
editing trajectories' (Line 86). However, there is no experimental analysis in the paper to support this, making this claim unpromising.
- Authors claim that FlowEdit is sensitive to the hyperparameters. However, there are also many hyperparameters need to be tuned in the proposed method (including but not limited to the ω and ζ ). Could FlowAlign perform better than FlowEdit in terms of hyperparameter sensitivity?
- A comprehensive discussion of the related work is missing, there has emerged a number of image editing methods recently, such as RF-Edit [1], QK-Edit [2], FluxSpace [3], FireFlow [4]. Authors are highly suggested to discuss and compare the performance between the proposed method and these baselines.
- Authors say that 'FlowAlign achieves superior source preservation and competitive or improved editing quality compared to FlowEdit and also other existing approaches in image, video, and 3D editing.' However, they do not compare the performance of their method with any baselines of video editing. By the way, the prompt in Figure 13 ( A black swan is swimming...) does not match the video.

[1]. Taming Rectified Flow for Inversion and Editing

[2]. QK-Edit: Revisiting Attention-based Injection in MM-DiT for Image and Video Editing

[3]. FluxSpace: Disentangled Semantic Editing in Rectified Flow Transformers

[4]. FireFlow: Fast Inversion of Rectified Flow for Image Semantic Editing

### Questions
See weakness.

### Soundness
2

### Presentation
2

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
This paper introduces FlowAlign, an inversion-free, flow-based framework for text-guided image editing built on pre-trained rectified flow models such as Stable Diffusion 3. It formulates editing as an optimal control problem and derives a modified velocity field that combines FlowEdit-style semantic drift with a source-consistency term expressed via Tweedie denoising of source and target latents. The resulting ODE is training-free, adds only a small computational overhead, and aims to produce more stable and deterministic editing trajectories. Experiments on PIEBench with SD3 show improved background preservation metrics and human preference over SDEdit, DDIB, RF-Inversion, and FlowEdit, while keeping CLIP and HPS scores competitive.

### Strengths
- The optimal-control formulation leads to a drift that explicitly combines semantic editing and source-consistency terms, offering a clear interpretation of how FlowAlign stabilizes trajectories.

- Using a shared SD3 backbone and comparable NFEs, FlowAlign consistently improves background PSNR/SSIM/LPIPS/DINO over SDEdit, DDIB, RF-Inversion, and FlowEdit while maintaining competitive CLIP and HPS scores.

- Reconstructing the original image by integrating backward yields much better fidelity with FlowAlign than with FlowEdit, reinforcing the claim of more stable, ODE-like behavior.

- Applying the trajectory regularizer as a guidance signal in these settings suggests that the idea could generalize beyond 2D image editing and inspire further work on structured generative representations.

### Weaknesses
- The optimal-control derivation relies on linear assumptions and a first-order approximation to the integrated drift, so the implemented ODE is not tightly characterized by theory and functions more as a guided heuristic.

- The paper positions FlowAlign mainly against multi-step ODE/SDE editors and does not clearly situate it relative to recent fast or one-step editing approaches that optimize for speed–quality trade-offs.

- Results on video and 3D Gaussian splatting are purely qualitative, without metrics for temporal or multi-view consistency, making it hard to assess robustness in these scenarios.

### Questions
How closely does the approximated drift in the implemented ODE track the exact optimal-control solution on simple toy problems, and did you observe regimes where the approximation qualitatively breaks down?

Are there characteristic failure modes (e.g., multi-object insertion, strong domain shifts) where FlowAlign underperforms FlowEdit or RF-Inversion, and could you provide concrete examples or diagnostics?

To what extent can the core idea be ported to non-rectified-flow backbones e.g. standard diffusion models, and what modifications would be required?

### Soundness
3

### Presentation
3

### Contribution
3
