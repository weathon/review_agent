# Direction-Magnitude Decoupling for Fast Video Generation with Flow Matching Models

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 8, 4, 4

## Abstract
Flow matching models for video generation achieve impressive performance but suffer from high computational overhead due to iterative denoising. In fact, the original model is not necessary for all denoising steps, allowing some steps to use lightweight alternatives for faster processing. However, directly using caching or lightweight models can deviate from the original denoising trajectory, resulting in suboptimal performance. Through empirical analysis, we find that lightweight models can robustly capture the magnitude components of the original model's output, while caching provides reliable directional guidance. Building on this insight, we propose the Direction-Magnitude Decoupling (DMD) method, which adaptively employs a direction-calibrated lightweight model as a substitute for the original model to accelerate inference and effectively correct deviations in the denoising trajectory. Moreover, DMD further reduces inference costs by reusing magnitude information under classifier-free guidance (CFG). As a result, DMD offers a more reliable and lightweight solution to accelerate denoising. Experiments show that DMD outperforms existing acceleration methods, delivering significant speedups (e.g., up to 2.95× on Wan2.1) while maintaining visual fidelity.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a video diffusion acceleration method. The authors observe that in existing caching-based acceleration methods, the generated flow trajectory can deviate from original one due to differences in the magnitude component. To address this issue, they propose modeling the scalar magnitude with a lightweight model, while caching the original diffusion model outputs only for the direction component. Additionally, a CFG variant is introduced to reuse the magnitude information. The proposed method is evaluated on Wan 2.1, demonstrating a 2.95x speedup without much performance degradation.

### Strengths
1. The proposed method is well motivated. The authors show an empirical analysis that the error in caching-based methods mainly stem from the magnitude component, which which motivates the proposed magnitude-direction decoupling method.
2. Extensive experimental results demonstrate that the proposed model improves inference speed without much performance degradation.

### Weaknesses
1. Lack of user study. The main experiments show improvement is LPIPS, PSNR, SSIM, while the VBench score degrades. This discrepancy raises concern about the actual visual quality. The authors could analyze the VBench results in detail and conduct additional user study to validate both the advantages and limitations of the proposed method.
2. Limited applicability. The proposed acceleration method is restricts to video models that are released with a smaller variant. However, many existing models like HunyuanVideo and Wan2.2-A14B do not provide comparable smaller versions. The authors should discuss the transferability of their approach across different models to address this concern.
3. Lack of experiments at higher resolution. Since the main time overhead of Wan2.1 14B arises from high-resolution inference (>10 min for a 720p video), it would be more informative to evaluate whether the proposed method remains effective at higher resolutions. Such experiments would also help verify the generality of the empirical observations and the robustness of the involved hyperparameters.

### Questions
1. The proposed CFG reuse strategy seems related to the velocity rescaling in CFG-Zero* [1] which demonstrates performance gains on its own. Would a simpler velocity normalization like CFG-Zero*, rather than rescaling by a small model's prediction, yield better performance?
2. Small models typically produce motions that are smaller and more subtle. Would approximating the magnitude of a larger model's velocity using the small model's outputs affect the motion intensity in the generated videos?

---

[1] Fan, et al. CFG-Zero⋆: Improved classifier-free guidance for flow matching models. arXiv:2503.18886

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the high computational overhead of flow matching models for video generation by proposing Direction-Magnitude Decoupling (DMD), a training-free acceleration method. DMD leverages two key empirical insights: (1) residual reuse reliably captures the directional component of the original large model’s output, and (2) lightweight small models accurately estimate the magnitude component. It combines these to approximate the original denoising trajectory, adds classifier-free guidance (CFG) reuse to halve small-model inference cost, and uses adaptive switching to mitigate directional error accumulation. Experiments on Wan2.1 and EasyAnimateV5.1 show DMD outperforms baselines (TeaCache, SRDiffusion) with up to 2.95× speedup on Wan2.1 while preserving visual fidelity (e.g., LPIPS 0.178, PSNR 22.72).

### Strengths
1. Targeted Empirical Insight: The decoupling of direction/magnitude components is well-motivated and validated (via cosine similarity, ℓ₂ norms), avoiding generic "lightweight substitution" and enabling principled acceleration.

2. Practicality: As a training-free method, DMD requires no retraining or extra data—critical for real-world adoption. CFG reuse is a clever optimization that reduces small-model cost without sacrificing quality.

3. Rigorous Evaluation: Experiments span two base models (Wan2.1, EasyAnimateV5.1) with multiple scales, and use comprehensive metrics (latency, LPIPS, PSNR, SSIM, VBench) to compare against relevant baselines. Multi-GPU results further demonstrate scalability.

4. Error Mitigation: The adaptive switching mechanism (via cumulative directional error) addresses residual reuse’s inherent error accumulation, a key limitation of cache-based methods like TeaCache.

### Weaknesses
1. Though the paper positions DMD as a training-free method (avoiding costly retraining or extra data, ), it relies on manual threshold adjustment to balance speed and visual quality—undermining its broad adaptability. 

2. The paper acknowledges multiple acceleration paradigms in related work, such as efficient ODE/SDE solvers (e.g., DPM-solver, ), post-training quantization, and progressive distillation. However, it only validates DMD as a standalone method, with experiments limited to comparisons against cache-based (TeaCache) and large-small model collaborative (SRDiffusion) baselines. There is no exploration of combining DMD with other acceleration categories—for example, whether integrating DMD with DPM-solver.

### Questions
1. Given that this manual tuning limits DMD’s generalization across different flow matching models or video generation scenarios, does the authors have any preliminary ideas or plans to design an adaptive mechanism (e.g., learning-based threshold adjustment using lightweight signals) that allows DMD to automatically determine the optimal \(\tau\) without manual intervention for new models?

2. Would pairing DMD with quantization distort the model’s ability to accurately estimate direction components, thereby affecting DMD’s performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a training-free acceleration method for flow matching video generation models, called DMD (Direction-Amplitude Decoupling). The core idea is to leverage empirical insights to exploit the complementary advantages between small models and cache residuals: small models within the same family can reliably capture the amplitude information from the original large model's output, while cache residuals can provide precise directional guidance. DMD combines the amplitude estimation from small models with directional calibration guided by cache residuals, adaptively replacing part of the denoising steps. It also introduces CFG (Classifier-Free Guidance) amplitude reuse to further reduce computational overhead, and mitigates error accumulation through an accumulated directional error threshold mechanism.

### Strengths
1. Efficiency of CFG reuse and adaptive strategy: Taking advantage of the characteristic of CFG mechanism where conditional and unconditional output amplitudes are similar, an amplitude reuse strategy is proposed to halve the small model inference overhead; meanwhile, an adaptive switching mechanism based on accumulated directional error is designed, which allows for moderate error accumulation while timely invoking the large model for calibration, effectively balancing inference speed and visual quality, and avoiding the error amplification caused by static caching or simple replacement of small models.

2. Practical value of training-free plug-and-play: DMD requires no additional training or fine-tuning, does not rely on large-scale annotated data, and can be directly integrated into existing flow matching models, compatible with multi-scale model architectures (shared VAE), with low deployment cost, possessing strong engineering compatibility and potential for promotion, solving the high cost problem of training-based acceleration methods (such as distillation, quantization).

3. Sufficiency and persuasiveness of experimental validation: The paper conducted comprehensive experiments on two mainstream flow matching models, compared with caching-based and collaborative small-large model baselines, and verified the effectiveness of the method through quantitative metrics (speedup, LPIPS, etc.) and qualitative visualization; meanwhile, it supplemented threshold sensitivity analysis, CFG reuse ablation experiments, and multi-GPU scenario testing, with rigorous experimental design.

### Weaknesses
1. The experiments are validated on two models, Wan2.1 and EasyAnimateV5.1. I believe the results on more other mainstream flow matching video generation models (such as HunyuanVideo, Open-Sora) would further enhance the persuasiveness;

2. Detailed metrics on VBench would further enhance the persuasiveness.

3. The direction error accumulation threshold (τ=0.005) in the adaptive strategy is entirely based on empirical settings, without providing theoretical basis or adaptive optimization methods (such as dynamically adjusting the threshold according to the dynamic characteristics of the video), raising doubts about its generality across different scenarios.

4. Can DMD accelerate smaller-sized models?

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors address the significant computational latency in flow matching models for video generation, a well-known bottleneck for iterative sampling. They observe that existing training-free acceleration methods, such as caching mechanisms or small-model substitution, often compromise visual fidelity. The reason, they posit, is that these substitutes can deviate from the original model's denoising trajectory, leading to accumulated errors. The paper's primary investigation involves decoupling the model's velocity field output into its direction and magnitude components. Their central empirical finding is that these two components are best approximated by different lightweight methods: cached residuals (residual reuse) provide a highly reliable estimate for the direction, while lightweight models from the same family robustly capture the magnitude.

Building on this analysis, the authors propose a novel training-free method, Direction-Magnitude Decoupling (DMD). The core of this technique is to construct a more faithful lightweight substitute by synthesizing a new velocity vector. This vector combines the magnitude estimated by the small model with the direction provided by the cached residual. This "direction-calibrated" approach allows the model to stay significantly closer to the original denoising path, thus preserving quality. The framework is made adaptive, using a cumulative error threshold to determine when to briefly invoke the large model for correction. Furthermore, a CFG reuse strategy is introduced to exploit redundancies in classifier-free guidance, further reducing inference costs. The authors report that DMD achieves substantial speedups, such as $2.95\times$ on the Wan2.1 model, while maintaining superior visual fidelity compared to existing acceleration baselines.

### Strengths
+ The central insight of decoupling the model output into direction and magnitude components and leveraging the complementary strengths of small models and residual reuse is novel. This  design leads to improved performance.

+ DMD achieves superior performance across all key metrics (Speedup, LPIPS, PSNR, SSIM) on two different model families (Wan2.1 and Easy Animate V5.1). The $2.95\times$ speedup on Wan2.1 while improving LPIPS to $0.178$ is a significant result.

+ As a training-free acceleration method, DMD is accessible and practical for immediate adoption with existing large flow matching models.

+ The introduction of the adaptive strategy using cumulative directional error $\mathcal{E}$ (Equation 5) and the CFG reuse mechanism demonstrates a robust and well-thought-out system design for maintaining fidelity under acceleration.

### Weaknesses
The empirical analysis focuses heavily on the $20\%$ to $95\%$ range of the diffusion process (Figure 2). While this covers the bulk of the steps, a brief discussion or analysis of the decoupling behavior in the extreme initial (0% to 20%) and final (95% to 100%) stages would complete the picture. Specifically, the paper mentions preserving the first 20% of steps with the large model but does not fully detail the inherent component reliability in those initial steps. The cumulative error threshold $\tau$ is set empirically, and while the ablation is good, a deeper theoretical justification or guidance for setting $\tau$ based on model architecture or dataset properties would be a valuable addition.

### Questions
- The paper mentions the magnitude for conditional output $v_{\varphi}(x_{t},t|c)$ and unconditional output $v_{\varphi}(x_{t},t|c=\emptyset)$ is nearly identical (Figure 2c, 2f). Is the small model used in these plots a different version for conditional vs. unconditional generation, or is it a single model trained to predict both? 

- Could the authors provide a brief analysis or discussion of the component reliability (directional vs. magnitude) in the initial $0\%$ to $20\%$ of the diffusion process, which is currently performed by the large model? 

- While the speedup is excellent, the comparison in Figure 4 is limited to a small number of GPUs (up to 6). Providing scalability analysis up to $8$ or $16$ GPUs would further strengthen the claim of high efficiency in multi-GPU setups.

### Soundness
3

### Presentation
2

### Contribution
3
