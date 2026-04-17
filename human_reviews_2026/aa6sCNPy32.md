# Motion-Aware Concept Alignment for Consistent Video Editing

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
We present MoCA-Video, a training-free framework for semantic mixing in videos.
Operating in the latent space of a frozen video diffusion model, MoCA-Video utilizes class-agnostic segmentation with diagonal denoising scheduler to localize and track the target object across frames. 
To ensure temporal stability under semantic shifts, we introduce momentum-based correction to approximate novel hybrid distributions beyond trained data distribution, alongside a light gamma residual module that smooths out visual artifacts.
We evaluate model's performance using SSIM, LPIPS, and a proposed metric, MoCA-Video, which quantifies semantic alignment between reference and output. 
Extensive evaluation demonstrates that our model consistently outperforms both training-free and trained baselines, achieving superior semantic mixing and temporal coherence without retraining. Results establish that structured manipulation of diffusion noise trajectories enables controllable and high-quality video editing under semantic shifts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles video semantic mixing, which is blending a reference image’s concept into a target object in a base video, without retraining the diffusion model. The proposed MoCA-Video pipeline uses (i) DDIM inversion of a frozen text-to-video model (VideoCrafter2), (ii) class-agnostic segmentation + IoU-based overlap maximization to track the edited region in latent space, (iii) latent feature injection from the reference image within a “soft” mask, (iv) a diagonal (FIFO-style) denoising scheduler to propagate edits temporally, (v) a momentum-corrected DDIM update to stabilize trajectories perturbed by semantic shifts, and (vi) a lightweight gamma residual noise regularizer to reduce flicker.

### Strengths
- **Training-free approach and simple design**
The paper provides a controllable hybrid concept mixing without fine-tuning the whole model, operating purely at inference time, which reduces the resource consumption compared with training-based methods. Combining soft masked latent fusion, overlap-max tracking, a diagonal scheduler, and a momentum term is conceptually straightforward.

- **Wide ablations**
The paper tests imperfect masks and multi-object scenes.

### Weaknesses
- **Incremental novelty**
The core ingredients of this paper, which are DDIM inversion plus masked latent injection and consistency propagation, are similar to image mixing (MagicMix, FreeBlend) and video editing propagation (TokenFlow, AnyV2V, and other recent tuning-free V2V pipelines). The paper claims to be the first training-free video semantic mixing, but the difference from tuning-free region-aware video editing is unclear and questionable. A sharper novelty boundary and deeper comparison to the most relevant video editing methods are needed.

- **Efficiency and practicality issues**
The pipeline requires DDIM inversion per video, and long sequences (147 frames) take 45 minutes on a V100. That is heavy for training-free editing. A runtime comparison vs. baselines is needed.

- **Missing details**
How the reference image latent is computed and aligned (spatial rescaling, aspect mismatch) for injection at time t? How robust is tracking under large appearance changes? How the diagonal FIFO scheduler is implemented in the editing process?

The problem is interesting and practical, and the pipeline is coherent and training-free. However, the contribution is incremental and overclaimed; core designs are common in previous works. I lean toward a negative score.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a training-free framework (MoCA-Video) for video semantic mixing with latent noise manipulation. IoU-based object tracking in noisy latent, momentum-corrected denoising, and gamma residual stabilization are introduced to achieve the MoCA-Video.

### Strengths
Belows are strong points that the paper has:

1. The paper presents thoughtful efforts to enhance video diffusion performance by fusing latents between the original video and the reference image, mitigating motion-induced artifacts through momentum-corrected DDIM, and stabilizing the denoising process with the proposed gamma residual noise.

2. The introduction of an entity blending dataset and the use of task-specific evaluation metrics indicate that the authors carefully designed the experimental setup to assess the proposed model’s performance.

### Weaknesses
Belows are weak points that the paper has:

**1. Clarity and Explanation**
-  The paper is difficult to follow due to insufficient self-contained explanations. Specifically, MoCA-Video is built on the denoising scheduler of FIFO-Diffusion, but the paper does not adequately describe FIFO-Diffusion itself. This lack of context makes it challenging for readers unfamiliar with FIFO-Diffusion to fully grasp the proposed methodology.
- Figure 2 is also unclear. The images following DDIM inversion are confusing. it's not evident whether they represent predicted clean $x_{0}$​ images or noisy latents. Moreover, the “Extract Mask Region” step lacks visual or textual clarity, and the meaning of the angle in the “Motion Correction” part is not explained. Without additional clarification, these visual elements are likely to confuse readers.

**2. Limited Model Compatibility**
- The proposed approach is only evaluated with VideoCrafter2, which restricts the generality of the results. In contrast, FIFO-Diffusion validated its framework with multiple video diffusion models. It would strengthen the paper to demonstrate MoCA-Video’s compatibility and performance across other diffusion backbones.

**3. High Sensitivity to Hyperparameters**
- The method relies on several hyperparameters ($\gamma$, IoU threshold, $\beta$, base correction weight $k_{o}$), which may complicate reproduction and tuning. The absence of guidance or ablation on their selection further limits the practical applicability of the approach.

**4.  Lack of Failure Case Analysis**
- The paper does not sufficiently discuss failure cases. For instance, the proposed approach appears to be highly sensitive to the quality of the mask generated by the segmentation model. I also find the authors’ interpretation of Table 4 unconvincing. This table, in fact, suggests that the semantic mixing performance is strongly dependent on mask quality, as evidenced by the substantial performance gap between the CASS and relCASS metrics. A more detailed clarification and analysis of this observation would significantly improve the readers’ understanding of the method’s limitations and robustness.

### Questions
Please answer the questions listed in the above Weaknesses section.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MoCA-Video, a training-free framework for video semantic mixing—blending features from a reference image into a target object within a video while maintaining temporal consistency. The method operates in the latent space of a frozen diffusion model (VideoCrafter2). It localizes the target by applying segmentation (Grounded-SAM2) to the predicted clean latent ($\hat{x}_0$) and stabilizes tracking via IoU maximization. To handle the generation of novel, out-of-distribution (OOD) hybrid concepts, the authors propose a momentum-corrected DDIM scheduler, introduced as a heuristic to adjust the denoising trajectory. The paper also introduces a new evaluation metric (CASS) and a benchmark dataset. Experiments show MoCA-Video outperforms existing video editing baselines in semantic blending quality.

### Strengths
*   **S1. Novel Task Definition:** The paper addresses the challenging and under-explored problem of temporally consistent video semantic mixing.
*   **S2. Strong Empirical Performance:** The qualitative results are visually compelling. Quantitatively, MoCA-Video significantly outperforms baselines on the proposed CASS metric (4.93 vs. the next best 3.80), which is supported by a user study (Appendix 6.3).
*   **S3. Useful Evaluation Metrics:** The introduction of CASS/relCASS provides a well-motivated metric for evaluating semantic shifts, addressing limitations in prior approaches (Appendix 6.2).

### Weaknesses
*   **W1. Lack of Theoretical Rigor:** The core innovation, the momentum correction (Alg. 2), is presented as a heuristic (L213) without theoretical grounding for how it approximates OOD hybrid distributions. The contribution remains largely empirical rather than analytical.
*   **W2. Algorithmic Ambiguity and Missing Details:** Algorithm 2 contains a critical ambiguity: Line 4 defines $g_{t} \leftarrow x_{t}-x_{t-1}+\lambda dir_{t}$. However, $x_{t-1}$ is computed later in Line 7. This ambiguity must be resolved. Additionally, the explicit schedule for the adaptive injection intensity $\lambda$ (L177) is missing, hindering reproducibility.
*   **W3. Severe Computational Overhead:** The inference time is extremely slow. Appendix 6.4 reports 45 minutes to process a 147-frame video on a V100 GPU. This significantly limits practical utility and scalability.
*   **W4. Sensitivity to Segmentation:** The framework's performance is sensitive to the quality of the external segmentation applied to the noisy $\hat{x}_0$. Table 4 demonstrates a significant performance drop (CASS from 4.93 to 2.69) when using coarse bounding boxes, indicating potential fragility in complex scenarios.

### Questions
1.  **[W2] Clarification on Algorithm 2:** In Algorithm 2, Line 4: $g_{t} \leftarrow x_{t}-x_{t-1}+\lambda dir_{t}$. Which value of $x_{t-1}$ is used here, given that the updated $x_{t-1}$ for the next step is calculated in Line 7? Does it refer to $x_{t-1}^{(DDIM)}$ calculated in Line 3? Please clarify the notation and the exact update sequence.
2.  **[W1] Theoretical Justification:** Beyond the empirical ablation, can you provide a deeper analysis or theoretical intuition regarding how the "geometric correction" term $x_t - x_{t-1}$ specifically helps reorient the denoising process toward OOD hybrid distributions?
3.  **[W2] Implementation Details:** Please specify the explicit schedule used for the adaptive feature injection intensity ($\lambda$) mentioned in L177.

### Soundness
2

### Presentation
3

### Contribution
2
