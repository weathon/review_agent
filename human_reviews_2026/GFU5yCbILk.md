# Uniform Discrete Diffusion with Metric Path for Video Generation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Continuous-space video generation has advanced rapidly, while discrete approaches lag behind due to error accumulation and long-context inconsistency. In this work, we revisit discrete generative modeling and present Uniform discRete diffuSion with metric pAth (URSA), a simple yet powerful framework that bridges the gap with continuous approaches for the scalable video generation. At its core, URSA formulates the video generation task as an iterative global refinement of discrete spatiotemporal tokens. It integrates two key designs: a Linearized Metric Path and a Resolution-dependent Timestep Shifting mechanism. These designs enable URSA to scale efficiently to high-resolution image synthesis and long-duration video generation, while requiring significantly fewer inference steps. Additionally, we introduce an asynchronous temporal fine-tuning strategy that unifies versatile tasks within a single model, including interpolation and image-to-video generation. Extensive experiments on challenging video and image generation benchmarks demonstrate that URSA consistently outperforms existing discrete methods and achieves performance comparable to state-of-the-art continuous diffusion methods. Code and models are available at https://github.com/baaivision/URSA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a generative model for image and video generation using discrete rectified flows. To achieve this, it introduces a new metric path to measure the token embedding distances from a pre-trained token cookbook. This, combined with other training stability tricks such as resolution-dependent timestep shifting and randomised frame-level timestep sampling, enables the proposed method to achieve comparable text-to-image and text-to-video generation performance on standard benchmarks to continuous generative methods. Ablation studies demonstrate the effectiveness of its design components.

### Strengths
The current video generation methods predominantly rely on continuous representation via latent diffusion or flow matching. This paper explores the potential of using discrete representation combined with flow matching for video generation. The paper is well-written, and the parameterisation of the sampling ratio via timestep is both interesting and intuitive. The results on the standard benchmarking are strong.

### Weaknesses
- I feel like the paper touches on a wider range of topics. While the title and method section focus on video generation, the experiment also briefly touches on image generation, which I think is unnecessary.

- The paper introduces several potential applications, such as long video generation and interpolation through diffusion-forcing training strategies. However, it lacks qualitative and quantitative results. For example, the paper could easily compare self-forcing, vanilla diffusion-forcing, on long video generation using the same VBench scores, or attach some generated videos. For a video generation paper, it’s a bit unacceptable not to include any generation examples, as the figures provided in the paper simply couldn’t reflect the generation qualities.
- Resolution-dependent timestep shifting is introduced and well-validated in SD3, so it shouldn’t be considered a proper technical contribution. However, the paper only acknowledges this in the experiment section, which seems suspicious.
- Asynchronous timestep scheduling is introduced in diffusion forcing, which the paper has acknowledged in the main paper. However, no visualisation or quantitative evaluation of the long/autoregressive video generation performance is discussed in the main paper.

### Questions
- Provide the generation examples of text-to-video, video interpolation, as well as long-video generations.
- If time allowed, please compare with self-forcing / diffusion forcing to validate the claim on long video generation.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Uniform Discrete diffusion with Metric path (UDM), a framework designed to improve discrete generative modeling for video synthesis by incorporating iterative refinement over spatio-temporal tokens. The method's key contributions are a linearized metric-path and resolution-dependent timestep shifting, which aim to stabilize training and enable scaling to long-sequence data. Furthermore, the authors propose an asynchronous timestep scheduling strategy to unify multiple video tasks, claiming that this approach closes the performance gap and achieves results comparable to state-of-the-art continuous diffusion models.

### Strengths
1. The linearized metric-path and timestep shifting mechanism looks sound to me. The motivation is clear and the theoretical explanation is reasonable. I think the authors find an effective manner to make the discrete diffusion model more powerful.

2. Experienmental analysis are comprehensive. The authors compare many state-of-the-art advances on three tasks: text-to-image generation, text-to-video generation, and image-to-video generation on several benchmarks. The results show that the proposed discrete diffusion model can outperform many continuous diffusion baseline methods.

3. The authors also conduct a series of ablation studies to fully probe the role of each modules. The performance across different inference steps and training iterations are also well ablated.

### Weaknesses
1. The scalability of the proposed uniform discrete diffusion method seems to be unclear. As the authors claim that scalability is one of the key highlight factors of the proposed method, it is crucial to conduct experiments on different model sizes to probe its scalability. This part of the experiment is completely missing both in the main manuscript and the attachment.

2. Some model details are missing. What is the size of the tokenizer and the Qwen LLM? What is the rationale for choosing Qwen instead of other open-source LLMs for training? Is there any ablation?

3. The authors do not provide any qualitative demos in the supplementary materials. It is always hard to distinguish the model performance from the concatenated screenshots.

### Questions
1. What is the average inference time for the proposed model? Comparison with previous discrete and continuous diffusion methods could be helpful.

2. How will the model degrade when generating longer video sequences?

3. It seems that the final model performance is not that sensitive to variations in the hyperparameters (α and c) that control the relationship between the sampling distance d(xt, x1) and time t from Appendix Figure 3. What is the rationale to choose the best path for different paths?

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
4

### Summary
The paper introduces a Uniform Discrete Diffusion approach that brings iterative, continuous-style refinement to discrete tokens for image/video generation. Core ideas include a metric-induced probability path over token embeddings, a resolution-dependent timestep shift to align effective SNR with sequence length, and frame-level (asynchronous) timesteps to unify T2V, I2V, interpolation, and extrapolation within one model. Training uses a cross-entropy objective; sampling integrates the learned probability velocity with few steps. Experiments report competitive image (GenEval/DPG) and video (VBench) quality and consistency, suggesting discrete diffusion can match continuous counterparts while remaining tokenizer/LLM-friendly.

### Strengths
* Clarity & focus: Clean objective and sampling recipe; equations and schedules are easy to implement.
* Unification: One model handles multiple video tasks via per-frame timesteps.
* Practicality: Simple schedules; no exotic losses or architectures required.
* Results: Broad benchmarks indicate strong quality and temporal stability with modest steps.
* Positioning: Bridges discrete tokenization with diffusion-style global refinement, relevant for LLM-aligned video generation.

### Weaknesses
* Novelty overlap: The *frame-level/asynchronous timestep* contribution is close to prior video schedulers (e.g., SkyReels-V2) and essentially the same idea as Pusa & FVDM; the paper should clarify what is new beyond this reuse.
* Relation to DFM/schedulers: The metric path and schedule resemble discrete flow matching/kinetic schedules; limited theory for superiority beyond heuristic tuning.
* Ablation depth: Missing systematic sweeps for the shift parameter \lambda; marginal gains of asynchronous vs. synchronous timesteps are not isolated.

### Questions
1. Differentiate from SkyReels-V2/FVDM&Pusa (especially): What is *new* about your frame-level timestep usage (objective, schedule, conditioning, or analysis)? 
2. Asynchronous impact: Quantify incremental gains over a global-t schedule for T2V at fixed steps/compute.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes UDM, a uniform discrete diffusion framework with a metric path for image and video generation. Its key designs include a linearized metric path for probability trajectory control and a resolution-dependent timestep shifting mechanism for high resolutions. An asynchronous timestep scheduling strategy is also adopted to unify multiple generation tasks within a single model. The paper demonstrates competitive performance against state-of-the-art continuous methods and improved performance over previous discrete methods on standard benchmarks.

### Strengths
This work represents a solid application of discrete diffusion models to video generation. The experiments comprehensively show UDM's advantages on multiple generation benchmarks, achieving competitive results with both discrete and continuous baselines. The proposed framework demonstrates the potential for scalable and unified visual generation.

### Weaknesses
The paper's claimed key innovations lack sufficient novelty and rigorous justification.

*   **Metric Path Novelty:** The core concept of a metric path has been previously proposed and applied to multimodal tasks[1,2]. While cited, the discussion of the relationship to these works is inadequate. The primary novelty lies in the "linear relationship" established by Eq. 4. However, the motivation and precise meaning of preserving a "linear relationship between $t$ and $d(x_t, x_1)$" in line 229-230 are unclear and not rigorously derived.

*   **Timestep Shifting Novelty:** The resolution-dependent timestep shifting is a common technique in continuous diffusion/rectified flow models [3,4]. Notably, Eq. 5 in this paper is functionally equivalent to Eq. 23 in [3] under simple variable substitutions: $t_n \to 1 - t,t_m \to 1 - \tilde{t},\sqrt{\frac{m}{n}} \to \lambda$. While its validation in the discrete setting is valuable, the connection to prior art is under-discussed, mentioned only briefly in the experiments.

*   **Typo:** There is a likely typo in Eq. 6, where `p_{t|1}` should probably be `p_{1|t}`.

[1] Shaul, Neta, et al. Flow Matching with General Discrete Paths: A Kinetic-Optimal Perspective. ICLR 2025.

[2] Jin Wang, Yao Lai, et al. Fudoki: Discrete flow-based unified understanding and generation via kinetic-optimal velocities. NeurIPS 2025.

[3] Esser, Patrick, et al. Scaling rectified flow transformers for high-resolution image synthesis. ICLR 2024.

[4] Hoogeboom, Emiel, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. ICML 2023.

### Questions
1.  Could the authors please elaborate on the motivation and precise meaning behind the claim of preserving a a "linear relationship between $t$ and $d(x_t, x_1)$" in line 229-230? A more detailed theoretical explanation or an empirical plot showing this relationship would be helpful.

2.  Figure 4 compares uniform diffusion with and without the metric path. To better validate the core innovation of the linearized path (Eq. 4), could you provide an ablation comparing performance *with the metric path* but *with and without* the specific parameterization given by Eq. 4?

### Soundness
2

### Presentation
3

### Contribution
3
