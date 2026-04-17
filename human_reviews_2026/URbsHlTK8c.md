# Let Features Decide Their Own Solvers: Hybrid Feature Caching for Diffusion Transformers

- Decision: Accept (Oral)
- Scores: 4, 8, 6, 10

## Abstract
Diffusion Transformers offer state-of-the-art fidelity in image and video synthesis, but their iterative sampling process remains a major bottleneck due to the high cost of transformer forward passes at each timestep. To mitigate this, feature caching has emerged as a training-free acceleration technique that reuses hidden representations. However, existing methods often apply a uniform caching strategy across all feature dimensions, ignoring their heterogeneous dynamic behaviors. Therefore, we adopt a new perspective by modeling hidden feature evolution as a mixture of ODEs across dimensions, and introduce \textbf{HyCa}, a Hybrid ODE solver inspired caching framework that applies dimension-wise caching strategies. HyCa achieves near-lossless acceleration across diverse tasks and models, including 5.55$\times$ speedup on FLUX, 5.56$\times$ speedup on HunyuanVideo, 6.24$\times$ speedup on Qwen-Image and Qwen-Image-Edit without retraining.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Diffusion Transformers (DiTs) offer state-of-the-art fidelity in image and video synthesis, but their iterative sampling process remains a major bottleneck due to the high cost of transformer forward passes at each timestep. To mitigate this, feature caching has emerged as a training-free acceleration technique that reuses or forecasts hidden representations. However, existing methods often apply a uniform caching strategy across all feature dimensions, ignoring their heterogeneous dynamic behaviors. Therefore, this paper adopts a new perspective by modeling hidden feature evolution as a mixture of ODEs across dimensions, and introduces \textbf{HyCa}, a Hybrid ODE solver inspired caching framework that applies dimension-wise caching strategies. HyCa achieves near-lossless acceleration across diverse domains and models, including 5.56× speedup on FLUX and HunyuanVideo, 6.24× speedup on Qwen-Image and Qwen-Image-Edit without retraining.

### Strengths
1. Performance is solid. The experiments are detailed, and the visualizations are rich.
2. Achieves high-speedup gains (specifically 5.56× and 6.24×) across diverse domains and multiple models, including FLUX, HunyuanVideo, Qwen-Image, and Qwen-Image-Edit.  
3. Maintains synthesis fidelity close to that of the original model while delivering acceleration.

### Weaknesses
1. The most important part of the ODE modeling should be elaborated in detail.  (see Question 4)
2. Missing related works with bespoke solver[1, 2, 3], which also searches the optimal solver parameters of a pretrained diffusion model.

[1] Xue, Shuchen, et al. "Accelerating diffusion sampling with optimized time steps." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[2] Wang, Shuai, et al. "Differentiable Solver Search for Fast Diffusion Sampling."  International Conference on Machine Learning (ICML) 2025. 

[3] Shaul, Neta, et al. "Bespoke solvers for generative flow models." arXiv preprint arXiv:2310.19075 (2023).

### Questions
1. What's $g_\theta$ mentioned in eq.3 ?
2. I am still confused about ODE modeling of the feature cache. Since there are discrete feature cache sets $[t_i, F(x_i, t_I)]_{i=0}^j$, estimating the next feature $F(x_l, t_I)$ constitutes a classic extrapolation problem, as stated in TaylorSeer [1].

3. Alternatively, if we treat the feature difference $F(x_i, t_i) - F(x_{i-1}, t_{i-1})$ as the subject of the ODE, a closed-form solution exists, which can be simplified to a classic extrapolation problem.

4. Line 206-207: 'sample the trajectory on a discrete timestep grid, enabling numerical integration using only cached feature values', Why? 

 Finally, I would be willing to raise the score if my concerns (even partially) can be addressed.

[1] Liu, Jiacheng, et al. "From reusing to forecasting: Accelerating diffusion models with taylorseers." arXiv preprint arXiv:2503.06923 (2025).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents Hybrid Caching (HyCa), a general acceleration framework for Diffusion Transformers (DiTs) that interprets feature caching as a hybrid ODE-solving process.

Instead of applying a single extrapolation rule across all timesteps or layers, HyCa clusters latent features by temporal dynamics and assigns each cluster a numerical solver (e.g., RK, AB, BDF, AM, or Taylor).

This design enables feature-adaptive extrapolation with improved stability and reduced computation.

Experiments on three DiT applications—text-to-image (Qwen-Image, FLUX), text-to-video (Hunyuan-Video), and image editing (Qwen-Image-Edit)—show 1.6–2.3× acceleration with minimal quality degradation, suggesting broad applicability and practical efficiency.

### Strengths
1. **Breaking the “Unified Caching Strategy” Assumption — Aligning with the Dynamic Nature of DiT Features**  
   Existing feature caching methods (e.g., FORA, TaylorSeer) typically assume that all feature dimensions follow a *single dynamic system*, adopting a unified caching/prediction strategy that overlooks the *heterogeneous dynamic behaviors* inherent in high-dimensional DiT features:
   - **New dynamic modeling perspective:** It models feature evolution as a *multi-dimensional ODE hybrid system*, revealing through clustering that DiT features exhibit two typical dynamics — oscillatory trajectories (Cluster 1, requiring stable solvers) and smooth continuous trajectories (Cluster 2, predictable with efficient solvers).  
   - **Clustering stability verification:** Using the Adjusted Rand Index (ARI), the authors demonstrate consistent clustering stability (ARI > 0.8) across prompts, time steps, and resolutions, supporting the claim that *dimension-wise dynamics are input-invariant* and thus enabling *“once-off offline clustering, lifelong online reuse.”*  
   This design fundamentally overcomes the “one-size-fits-all” issue that causes *sharp quality degradation at high acceleration*, which explains its strong performance gains.

2. **Near-Lossless Acceleration with Superior Speed-Quality Trade-off**  
   Across three major tasks — text-to-image generation, text-to-video generation, and image editing — HyCa achieves an optimal balance between speed and quality **without retraining**:  

3. **Seamless Integration with Distilled Models — “Stackable Acceleration”**  
   Traditional caching methods struggle to handle distilled models (where sampling steps shrink from 50 to 4–8, making features more discrete and harder to predict). HyCa addresses this with a *hybrid solver pool* (including explicit/implicit methods — RK, AM, TF, etc.) that adapts to varying dynamics.
   This property enables deployment in *low-latency scenarios* (e.g., mobile, real-time generation), broadening HyCa’s application scope.

4. **Unifying Feature Caching through the ODE Solver Perspective — High Interpretability**  
   HyCa formalizes *feature caching* as an *ODE numerical solving* problem.  
   By deriving $\frac{d}{dt}F(x_t) = g_\theta(F(x_t), t)$ (where $g_\theta$ is an implicit vector field), it shows that caching essentially *numerically integrates past features to predict future ones*.  
   Building on this, HyCa’s solver pool (explicit RK/AB and implicit AM/BDF) selectively matches dynamics: smooth trajectories use efficient TF (Taylor formula) prediction, while oscillatory ones use stable AM (Adams-Moulton) solving. This provides both clear interpretability and a foundation for future extensions (e.g., adding new solvers for specific dynamics).

### Weaknesses
**Incomplete Computational Complexity Analysis**  
   The paper only reports *end-to-end latency*, omitting details on offline clustering and solver overheads:  
   - **Offline Phase:** For high-dimensional models, k-means complexity is ma. Such large-scale clustering may require substantial preprocessing time, yet this is undocumented.  
   - **Online Phase:** Different solvers have varying computational loads (e.g., implicit AM requires nonlinear equation solving, explicit TF only polynomial evaluation). However, the paper does not quantify each solver’s latency impact, leaving it unclear whether further speed gains are possible through solver re-selection.

### Questions
1. How much time does the offline clustering step require
2. Can you include a single-solver ablation to isolate the hybrid solver’s benefit

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes HyCa, a training-free acceleration framework for Diffusion Transformers. The authors observe that different hidden feature dimensions in DiTs exhibit distinct temporal dynamics—some smooth, others highly oscillatory—making uniform caching suboptimal. HyCa reformulates feature caching as a mixture of ODE solving problem, where each feature cluster automatically selects its best numerical solver. During an offline stage, clusters are formed based on dynamic indicators and matched with optimal solvers; at inference time, each cluster reuses its solver to skip redundant computations under a “one-time choosing, all-time solving” mechanism. Experiments on image, video, and editing tasks demonstrate 5–6× acceleration with negligible quality loss. The method is simple, generalizable, and practically useful.

### Strengths
1. The paper introduces a novel idea of modeling feature caching as a mixture of ODE solving process, offering a dynamic and interpretable perspective on diffusion sampling. The approach is simple, training-free, and practical.

2. The experiments are comprehensive, comparing against strong baselines while maintaining image quality under significant acceleration. Ablation studies and visualizations clearly support the proposed design, and the framework generalizes across different models.

### Weaknesses
1. **Limited validation of robustness and generalization.** The assumption that feature clusters remain stable across prompts, resolutions, and timesteps seems optimistic. The presented evidence is limited and lacks failure or stress cases.

2. **Missing analysis of error accumulation.** The method minimizes one-step prediction error, but inference involves multi-step extrapolation, potentially causing cumulative errors. No analysis or long-horizon stability evaluation is provided.

3. **Unclear offline cost.** The paper does not report the computational cost or scalability of the offline clustering and solver-selection process, which is critical for reproducibility and practicality.

### Questions
1. Under what conditions does the clustering become unstable? How do prompt, resolution, or timestep variations affect stability? Could you show a few failure cases?

2. Please provide details on the offline clustering and solver-selection cost—runtime, data scale, and scalability for larger models.

3. If a LoRA or fine-tuning is applied on top of the same base model, can the previous configuration still be reused, or must the offline process be repeated?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
The paper propose a method for feature caching/predicting in Diffusion Transformers (DiT). The main contribution of the paper upon TaylorSeer [1] are the introduction of feature dimension clustering which allows to use different estimation methods (i.e., ODE solvers) for different cluster group. Importantly the proposed method is able to accelerate even distilled models, improving both speedup 5.2 Db in PSNR compared to the runner-up baselines [1].

[1] Liu, Jiacheng, et al. "From reusing to forecasting: Accelerating diffusion models with taylorseers." arXiv preprint arXiv:2503.06923 (2025).

### Strengths
1. The method is validated across multiple tasks such as image generation, image editing, and video generation showing very strong results with significant speedups.

2. Importantly, the method is able to accelerate the a state-of-the-art (SOTA) distilled flow model FLUX.1 Schnell [2] by a factor of 2x while maintaining  SOTA performance (measured by CLIPScore and ImageReward) as we as achieve 34.37 Db in PSNR w.r.t. to ground truth generated by the original FLUX.1 Schnell model.

3. The method is training free.

4. Fast inference of diffusion/flow model is of interest for many researcher. Therefore, given the two points mentioned above, this paper has the potential to make a significant impact.

[2] FLUX.1 Schnell (Black Forest Labs, 2024).

### Weaknesses
1. The presentation is somewhat lacking. Adding explicit examples illustrating how different solvers are applied to predict future feature would significantly help the reader understand the method.

### Questions
1. In appendix A.2.1 does $v_k$ stands for the features?

2. Could the authors please provide explicit example of how does the Runge-Kutta method is applied to predict features?

### Soundness
4

### Presentation
3

### Contribution
3
