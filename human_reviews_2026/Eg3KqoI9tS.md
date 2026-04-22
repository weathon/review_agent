# TPDiff: Temporal Pyramid Video Diffusion Model

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4, 4, 6

## Abstract
The development of video diffusion models unveils a significant challenge: the substantial computational demands. To mitigate this challenge, we note that the reverse process of diffusion exhibits an inherent entropy-reducing nature. Given the inter-frame redundancy in video modality, maintaining full frame rates in high-entropy stages is unnecessary. Based on this insight, we propose TPDiff, a unified framework to enhance training and inference efficiency. By dividing diffusion into several stages, our framework progressively increases frame rate along the diffusion process with only the last stage operating on full frame rate, thereby optimizing computational efficiency. To train the multi-stage diffusion model, we introduce a dedicated training framework: stage-wise diffusion. By solving the partitioned probability flow ordinary differential equations (ODE) of diffusion under aligned data and noise, our training strategy is applicable to various diffusion forms and further enhances training efficiency. Comprehensive experimental evaluations validate the generality of our method, demonstrating 50% reduction in training cost and 1.5x improvement in inference efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a staged frame-rate reduction strategy to accelerate the training of diffusion-based generative models. The method progressively lowers the temporal resolution during training, aiming to reduce computational cost while preserving generation quality.

### Strengths
S1. The core idea is intuitive and conceptually easy to grasp.
S2. The empirical results, based on three baselines, demonstrate consistent improvements and validate the effectiveness of the proposed approach.

### Weaknesses
W1. Lack of comprehensive baselines. The experimental comparisons are rather limited. The field of video diffusion acceleration has seen rapid development since 2023, with numerous relevant approaches proposed recently. However, this paper only compares its method against a vanilla diffusion baseline, without evaluating against other contemporary acceleration techniques. As a result, it is difficult to assess the relative advantage or practical relevance of this approach. Readers cannot determine whether the proposed strategy offers clear benefits over more recent methods.

W2. Outdated experimental setup. Most of the compared methods date back to 2024 or earlier, while the video generation domain evolves extremely fast. Many 2023-era methods are no longer representative of the current state of the art. For instance, the authors compare against Open-Sora 1.3 (2023), but Open-Sora 2.0 was released in March 2025—roughly six months before the submission deadline. This omission raises concerns about the timeliness and completeness of the evaluation. Incorporating more recent baselines would provide a fairer and more convincing assessment of the proposed contribution.

### Questions
Q1. Could the authors include comparisons with more recent (2024–2025) approaches in video generation acceleration to contextualize their results?

Q2. It would be beneficial to add experiments involving newer baselines such as Wan 2.2 or other contemporary diffusion acceleration models.

Overall, I would be open to increasing my evaluation if these issues are addressed in the rebuttal. However, given the current limitations in experimental scope and currency, my initial assessment remains conservative.

### Soundness
2

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
The paper proposes a temporal pyramid video diffusion model that accelerates both training and inference. The authors finds that consecutive frames are temporally redundant and that early high-noise steps do not need full frame rate, so generating video in a coarse-to-fine temporal pyramid is intuitive. To make this work in a pyramid format, they introduce a stage-wise diffusion objective that treats each temporal stage as its own denoising trajectory, and also align each training clip with a consistent noise sample to stabilize those trajectories. The authors also extend to support DDIM-style sampling. Experiments show up to around 2× faster training convergence and 1.5–1.8× faster inference while maintaining comparable results with baseline methods.

### Strengths
1. The paper introduces a novel stage-wise temporal pyramid diffusion process that reduces frame rate in early noisy steps and restores it later. This aligns well with the intuition that video modality is usually quite redundant, especially between consecutive frames.
2. The proposed stage-wise diffusion objective and sample-noise alignment seem to solve the main problem in pyramid format modeling shown in the experiment section.
3. The method is shown to work beyond typical flow matching objectives and extend to DDIM-style objectives
4. Experiments faster training and inference speed while maintaining similar results with baseline methods.

### Weaknesses
1. The approach depends on a fixed multi-stage pyramid schedule (how frames are downsampled/upsampled across stages), and the ablation study in the paper explores limited variations (3-4 stage) of the fixed schedule. 
2. While data–noise alignment improves determinism and stability for training, will this lead to less diversity in generation?
3. It would be great to include some visual failure cases, especially related to your ablation studies, as quantitative evaluation suites like VBench tend to be biased, and sometimes can not align with visual perception.

### Questions
1. Not a limitation but a curious question. How robust is the temporal pyramid schedule? Will the inference break if the user chooses more stages (beyond 3) in the inference for better quality?
2. Do authors observe any semantic or visual artifacts at stage transitions (such as duplicated frames or sudden velocity jumps)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the author argues that maintaining full frame rates in high-entropy is unnecessary considering the nature of inter-frame redundancy in video. The author proposes TPDiff framework to facilitate the training and inference efficiency. Specifically, the proposed method divides the diffusion process into several stages and gradually increases the frame rate. To achieve this goal, the author proposes a stage-wise training strategy that leverages ordinary differential equation of diffusion under aligned data and noise. As a result, the training and inference efficiency can be improved and faster. And the proposed method can be generalized to various types of diffusion methods.

### Strengths
* The paper is well-organized and easy to follow.
* The motivation for leveraging the inter-frame redundancy to increase the training and inference efficiency is intuitive and the proposed solution is effective.
* The proposed solution is applicable to various diffusion forms such as DDIM and flow matching, demonstrating flexibility and extensibility across major generative modeling approaches.
* The proposed method achieves better performance compared to the reference work. Specifically, the results show that the proposed method outperforms vanilla diffusion baselines like MiniFlux-vid and AnimateDiff, achieving improvements in motion smoothness, object fidelity, and spatial coherence as measured by VBench metrics and FVD.
* The authors added a discussion and experiments on the effect of the number of stages. This improves the completeness of the study.
* The authors added a detailed analysis of computational overhead, demonstrating that the proposed data-noise alignment incurs negligible extra cost.
* The authors discuss the data selection and generalization. They use the full OpenVID1M dataset and include experiments on Wan to validate generalization across models, avoiding the potential dataset bias.

### Weaknesses
* While the paper introduces the stage-consistent positional encoding to maintain temporal coherence across stages, the experimental validation of this component is limited. There is no dedicated ablation or quantitative evaluation to isolate its contribution, making it unclear how much improvement comes from this module versus other parts of the framework.

* The computational overhead analysis provides CPU latency data, but there is little discussion of GPU memory usage, scalability with video length, or training cost under large-scale settings. A more comprehensive efficiency breakdown would strengthen the claims of computational efficiency.

### Questions
Please refer to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work introduces TPDiff, a temporal-pyramid scheme that progressively increases the frame rate across denoising stages. Early, high-noise steps operate at lower frame rates (exploiting redundancy), while later steps use full frame rate. A stage-wise training framework solves a partitioned probability-flow ODE with data-noise alignment, making the approach applicable to both flow matching and DDIM; a single model handles multiple frame-rate regimes. The latent is written in unified form $x_t = \gamma_t x_0 + \sigma_t \epsilon$ with $\epsilon \sim \mathcal{N}(0,I)$. The result is better training/inference efficiency with competitive quality.

### Strengths
1. Intuitive curriculum on temporal redundancy with a single model across frame‑rate regimes.

2. Implementation appears simple and potentially compatible with other accelerations.

3. Some metrics on temporal consistency improve, suggesting a useful inductive bias.

### Weaknesses
1. Comparative breadth. More baselines (esp. temporal interpolation modules and caching) would sharpen the empirical case.

2. Temporal fidelity. Limited analysis of temporal artifacts vs. full-rate baselines.

3. Scalability details. Memory/computation profiles for long videos and high resolutions could be elaborated.

4. Ablations: stage count and frame‑rate schedules lack systematic exploration and guidance.

### Questions
1. How sensitive are results to the number of temporal stages and their schedule?

2. Does the single-model design struggle with extreme motion or long-range dependencies?

3. Can the PF-ODE training be combined with distillation from a full-rate teacher?

4. Any failure cases for long videos ($>$16 s) or high resolutions (e.g., $720\text{p}$, $1080\text{p}$)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes TPDiff, a temporal-pyramid video diffusion framework that aims to improve training and inference efficiency for text-to-video diffusion models. The key idea is to progressively increase the frame rate along the reverse diffusion process: early steps operate on a temporally down-sampled video (to avoid processing redundant, low-SNR frames), while only the final stage runs at the full frame rate. To support this, the authors introduce a stage-wise diffusion training scheme that decomposes the overall ODE into several sub-ODEs, and they show how to obtain the intermediate latents and targets for each stage via data–noise alignment so that one single network can handle all stages. The method is instantiated on both DDIM-style diffusion and flow-matching–style video generation, and evaluated on several video backbones (AnimateDiff, MiniFlux-vid, Wan). Experiments show comparable or slightly better VBench scores while reducing training cost (≈2× speedup) and improving inference latency (≈1.5×)

### Strengths
Video diffusion indeed suffers from temporal redundancy and from the fact that early denoising steps carry very little signal; using a lower frame rate in these steps is a sensible way to cut quadratic attention cost. Aligning video samples to nearby noises reduces variance of the ODE path and seems to be the reason why multi-stage training can converge, which is a neat implementation detail often ignored in similar works

### Weaknesses
The paper is very close in spirit to pyramid flow / spatial-pyramid diffusion: the new part is mainly “do it in temporal dimension + make one model handle all stages + add data–noise alignment”. Some readers may feel this is more an engineering generalization than a fundamentally new generative formulation.

### Questions
1. For videos with large or non-smooth motion (e.g., camera panning, fast human motion), how often does the temporal upsampling between stages fail? Could you provide a metric or user study that shows interpolation does not become the bottleneck?
2. The authors should explicitly state the coloring rule (per-backbone improvement? column-wise best?
3. The training illustration in Fig. 4(b) resembles prior “trajectory-aligned” or “consistency/distillation” style diffusion works, where multiple timesteps are forced to follow the same noise / data direction to reduce variance and enable parameter sharing. What is actually specific to this paper is that the alignment is used to support multi-stage, multi-frame-rate video diffusion. I would suggest the authors clarify this connection and better distinguish their alignment from earlier consistency / rectified-flow / progressive-distillation lines.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes TPDiff, a temporal pyramid video diffusion framework that progressively increases the frame rate during the diffusion process to reduce computational cost. The method divides the diffusion process into multiple stages, with only the final stage operating at full frame rate. A stage-wise training strategy based on partitioned probability flow ODEs and data-noise alignment is introduced to support multi-stage learning. Experiments show that TPDiff achieves 2× faster training and 1.5× faster inference while maintaining or improving video generation quality on benchmarks like VBench.

### Strengths
Novelty: The temporal pyramid structure is a novel and intuitive way to exploit temporal redundancy in video generation, extending the spatial pyramid idea to the temporal dimension.

Efficiency: The method significantly reduces both training and inference costs without sacrificing quality, making it highly practical for large-scale video generation.

Generality: The framework is applicable to multiple diffusion forms (DDIM, Flow Matching) and models (AnimateDiff, MiniFlux-vid, Wan), demonstrating strong generalization.

### Weaknesses
Theoretical grounding: While the stage-wise ODE derivation is sound, the justification for why the temporal pyramid does not harm motion modeling is somewhat heuristic.

Limited video length evaluation: The experiments focus on short video clips; it is unclear how the method scales to longer sequences.

Comparison with recent methods: Missing comparisons with other efficient video diffusion methods (e.g., LVD, VideoPoet) limits the context of its advantages.

Complexity of implementation: The need for data-noise alignment and stage-consistent positional encoding adds implementation overhead.

### Questions
Can the authors provide a theoretical analysis or intuition for why the temporal pyramid does not degrade motion consistency?

Please include experiments on longer video generation to assess the method’s scalability.

Consider comparing with more recent efficient video diffusion models to better position TPDiff’s contributions.

The paper would benefit from a discussion on the trade-off between the number of stages and video quality.

### Soundness
3

### Presentation
3

### Contribution
2
