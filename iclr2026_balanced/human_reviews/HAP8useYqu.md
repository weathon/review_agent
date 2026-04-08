## Human Reviewer 1

### Summary
This paper presents a method called TempO, which evolves the solution of PDEs using flow matching. First, TempO projects the input into a latent space. The evolution of the solution in latent space is trained with flow-matching, including sparse conditioning on the previous latent space values. Here, TempO uses a time-conditioned FNO to learn the latent flow.  The authors evaluate TempO against other flow-matching video-generation methods based on vision transformers and UNets in combination with different flow-matching paths. The experiments are performed on the Navier-Stokes, shallow water, and reaction-diffusion PDEs, showing that TempO reaches a lower MSE and diverges more slowly than the other models.

### Strengths
1. Novel combination of Flow-matching and FNO.
2. Theoretical bounds on the model approximation error are provided.
3. Extensive evaluation against other flow-matching approaches.
4. The paper is clear and well written.

### Weaknesses
1. No evaluation against PDE-specific models, only other flow-matching models from the video generation domain are tested. There are a number of PDE models that employ diffusion [1,2,3], for example, and also improve the rollout stability in that way.  Additionally, since the model uses an FNO in latent space, a plain FNO should also be used as a baseline. 
 
2. In the introduction, the stochasticity of diffusion-based models is described as a disadvantage for PDEs. However, as shown in [2], the stochasticity is still helpful, even in deterministic PDEs, since it can act as an uncertainty measure. It is especially helpful for chaotic PDEs, where diffusion can help to describe the distribution of plausible trajectories (since any estimator will diverge at some point for chaotic dynamics)



[1] Serrano, L., Wang, T. X., Le Naour, E., Vittaut, J. N., & Gallinari, P. (2024). AROMA: Preserving spatial structure for latent PDE modeling with local neural fields.  
[2] Lippe, P., Veeling, B., Perdikaris, P., Turner, R., & Brandstetter, J. (2023). Pde-refiner: Achieving accurate long rollouts with neural pde solvers.  
[3] Holzschuh, B., Liu, Q., Kohl, G., & Thuerey, N. (2025). PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations.

### Questions
1. For the sparse conditioning, the experiments (line 269) mention that the last 15 frames are used to condition. During inference, how does the sparse conditioning work at the beginning (ie, when you only have the embedding of the initial condition as the input)? Does the model need to encode multiple timesteps for that?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 2

### Summary
The authors propose using FNO as the denoiser during flow matching, motivated by some theoretical insight. When compared to ViT or Unet based denoisers across different noise schedules, the method works well and seems to improve on 2D benchmarks.

### Strengths
- The authors present a variety of baselines, including different noise schedules and backbones.
- The theoretical motivation for using FNO as a denoiser is apparent.
- The performance gains are good, with consistent gains across different noise schedules/datasets/baselines.

### Weaknesses
- I’m having a hard time differentiating this work from prior work (Functional Flow Matching https://arxiv.org/pdf/2305.17209), where an FNO is also used as the denoiser and an OT schedule is used. Is the novelty using sparse conditioning and channel folding? The additional experiments to a longer horizon are also somewhat lacking (an ablation in Table 5 for some unspecified dataset).  
- Reporting next-step error is a good start, but in general, most works are concerned with rollout error since the main driver of error in neural PDE surrogates is autoregressive error accumulation. MSE/time is shown in Table 5, but it would be good to see this for all models (backbones + noise schedules), since it is more informative than single-step error. 
- There seems to be better spectral performance at the highest frequency bands for ViT-based models, which may be related to FNO’s mode truncation. This might be more relevant for spectral accuracy in more complex systems (turbulence, multiscale phenomena)
- The ablation in Table 5 on the effect of context length on accuracy could be expanded on. Is the model with a sequence length of 25 predicting 15 unseen frames and the model with sequence length of 2 predicting 38 unseen frames? 
    - This also seems to contradict prior results (https://arxiv.org/pdf/2507.02608,  https://arxiv.org/abs/2111.13802) that suggest that the context length either does not have an effect on rollout error or harms it. 
- Some comparisons to deterministic models (FNO/Unet) for SWE/NS/RD would be beneficial just to calibrate what is the baseline performance for neural surrogates on these common systems.

### Questions
- There could be a few more relevant works (related to flow matching for PDEs) that could be cited:
    - Latent flow matching (https://arxiv.org/abs/2503.22600)
    - Flow matching for PDEs (https://arxiv.org/abs/2506.08604)
    - Using FNO as a Denoiser (https://arxiv.org/abs/2302.07400)
- What dataset is used in Table 5? 
- Is Figure 1 generated with a single trajectory or averaged across the validation set? 
- Not a problem, but there seem to be a lot of red references to lead nowhere. 
- The performance of the UNet denoiser seems to be very poor based on qualitative observations (Figure 6), but since the NS dataset has been standard for quite some time now, there is a lot of prior work that shows that vanilla FNO/Unet can approximate the system well + even more challenging systems (https://arxiv.org/abs/2209.15616, https://arxiv.org/abs/2309.01745). 
    - This isn’t explicitly an issue, but just curious that Unet struggles so much.
- Also not a clear issue, but the use of a transformer as a diffusion/flow matching backbone is very ingrained in modern machine learning, not only in image generation, but also in PDEs. There is a lot of prior work that uses this paradigm successfully, so suggesting an alternate and expecting people to adopt it would need to include a very rigorous set of experiments, likely on more challenging PDEs/benchmarks (https://arxiv.org/abs/2412.00568).
    - Perhaps a thought experiment would be: "If I am building a large, latent generative model, would I use a transformer or FNO as the backbone?" How would you convince someone of one method or another?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper tackles long-horizon forecasting for high-dimensional PDE dynamics , proposing Tempo, a novel latent flow matching model. Tempo's core innovation is using a time-conditioned Fourier Neural Operator (FNO) as its velocity field regressor. The model operates in a latent space, employing sparse conditioning and channel folding for efficient 3D spatiotemporal processing. The authors provide theoretical justification that this FNO-based design is asymptotically more parameter-efficient than sampler-based architectures like Transformers or U-Nets. Experimentally, Tempo outperforms SOTA baselines on three PDE datasets, demonstrating highly stable long-horizon forecasting where competitors fail , while also being significantly more parameter-efficient

### Strengths
* Proposes **Tempo**, a novel model combining latent flow matching with an FNO regressor, which is well-motivated by aligning the FNO's spectral bias with continuous PDE dynamics.

* Provides a theoretical analysis (Theorem 3.1, Prop 3.2) to justify the architecture, suggesting FNOs can achieve a target accuracy with asymptotically fewer parameters than sampler-based models like U-Nets or Transformers.

* Demonstrates highly stable 40-step autoregressive forecasting on the NS-w dataset. This significantly outperforms ViT and U-Net baselines, which show clear degradation.

* The model is highly parameter-efficient. It also shows superior sampling efficiency, requiring the fewest Number of Function Evaluations (NFEs) for inference.

### Weaknesses
* The paper suffers from significant clarity issues. It lacks a high-level overview of the proposed method, making it difficult to grasp the core components. The writing style relies on overly long and convoluted sentences, hindering readability. The overall structure feels disjointed, making the paper's narrative hard to follow.

* The paper's core motivation—that existing methods fail due to "cumulative errors and discretisation artifacts"—is not sufficiently substantiated. This claim is presented as a given, but the paper lacks the necessary citations or analysis to support it.

* The connection between the stated problem and the proposed solution is weak. The paper does not adequately explain *why* flow matching is the right choice to mitigate "cumulative errors" or *how* the FNO architecture specifically addresses "discretisation artifacts" better than other regressors. The design choices feel disconnected from the initial problem statement.

* The methodological novelty is unclear. The primary components, Flow Matching and FNO, are both well-established. The paper fails to clearly articulate what makes their specific combination (Tempo) a significant and novel contribution beyond a straightforward engineering application.

* The empirical results are not fully convincing.
    * The performance gains over the ViT and U-Net baselines are marginal in several next-step prediction tasks. The strong long-rollout performance is only demonstrated on one dataset (NS-w).
    * The baseline comparison is insufficient. The paper omits direct comparisons against the original FNO (which should be a key baseline/ablation) and other strong operator learning models (e.g., WNO, DeepONet), making it difficult to properly contextualize the results.

### Questions
1.  The paper's motivation hinges on addressing "cumulative errors" and "discretisation artifacts." Could the authors elaborate on the explicit mechanism by which the flow matching framework and the FNO's spectral bias inherently mitigate these specific error types, in a way that standard autoregressive U-Nets or Transformers do not?

2.  The experimental comparison focuses on ViT and U-Net regressors *within* the flow matching setup. Could the authors provide a comparison against a standard autoregressive FNO (or other strong neural operators like WNO/DeepONet)? This seems crucial to disentangle the benefits of the proposed flow matching framework from the known benefits of the FNO architecture itself.

See the above weaknesses for further details.

### Soundness
1

### Presentation
2

### Contribution
2

### Rating
2

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper introduces TempO, a novel framework for PDE forecasting based on flow matching in latent space. Unlike stochastic diffusion models, TempO leverages deterministic ODE-based sampling through a time-conditioned Fourier Neural Operator (FNO) to capture both global and local spectral dynamics. The paper provides theoretical error bounds, showing that FNOs can approximate flow fields more efficiently than Transformer- or U-Net–based samplers. Experiments are conducted on three PDE benchmarks (Navier–Stokes, Shallow Water, Reaction–Diffusion).

### Strengths
1. The paper is technically solid, with a clear theoretical contribution (Theorem 3.1, Proposition 3.2) establishing approximation bounds for FNO-based flow matching.
2. The idea of coupling flow matching with operator learning (FNO) is elegant and well-motivated.
3. The experimental evaluation is rigorous and diverse, covering several PDE datasets and comparing against recent ViT-based and diffusion-based baselines.

### Weaknesses
1. The training details (e.g., hyperparameter sensitivity, stability beyond 40-step rollouts) are under-discussed. Since the main claim is long-horizon stability, results on longer or more chaotic regimes would strengthen the argument.
2. The novelty claim could be better contextualized: related works such as Functional Flow Matching (Kerrigan et al., 2023) and Conditional Flow Matching (Tamir et al., 2024) are mentioned but not deeply contrasted with TempO in terms of scalability or architecture

### Questions
1. Please see the weaknesses above.
2. Could the model extend to irregular sampling or missing data, for example through Neural ODE–like continuous-time conditioning?
3. How sensitive is TempO’s performance to the number of Fourier modes or truncation level (beyond the eight-mode empirical finding)?

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
2