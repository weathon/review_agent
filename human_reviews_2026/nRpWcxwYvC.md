# SRDiffusion: Accelerate Diffusion Inference via Sketching-Rendering Cooperation

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Leveraging the diffusion transformer (DiT) architecture, models like Sora, CogVideoX and Wan have achieved remarkable progress in text-to-video, image-to-video, and video editing tasks. Despite these advances, diffusion-based video generation remains computationally intensive, especially for high-resolution, long-duration videos. Prior work accelerates its inference by skipping computation, usually at the cost of severe quality degradation. In this paper, we propose SRDiffusion, a novel framework that leverages collaboration between large and small models to reduce inference cost. The large model handles high-noise steps to ensure semantic and motion fidelity (Sketching), while the smaller model refines visual details in low-noise steps (Rendering). Experimental results demonstrate that our method outperforms existing approaches, over 3 $\times$ speedup for Wan with nearly no quality loss for VBench, and 2 $\times$ speedup for CogVideoX. Our method is introduced as a new direction orthogonal to existing acceleration strategies, offering a practical solution for scalable video generation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a training-free algorithm to accelerate the sampling process. The core idea is to use a large model for the initial denoising steps and then switch to a smaller model for the later stages. This reduces inference time compared to exclusively using the large model.

However, the work's primary weakness is its simplicity and lack of novelty. The contributions are twofold: (1) proposing to substitute the large model with a small one in the later phase of denoising, and (2) providing a metric to determine when to make this switch. It is important to note that the proposed metric is not original, as a similar concept has already been discussed in Teacache.

### Strengths
The article's strength lies in its well-written and easy-to-understand content, but that is where its merits end.

### Weaknesses
1. Limited Novelty and Practicality: The core idea of substituting a large model with a smaller one during the denoising process is overly simplistic and lacks sufficient novelty for a publication at this venue. Moreover, the method's practical applicability is questionable. It relies on the pre-existence of an official, smaller version of the large model. The paper fails to address how the method would be applied to models without a corresponding small variant, such as HunyuanVideo. The authors should clarify whether a new small model must be trained, and how that would impact the "training-free" claim of the algorithm.

2. Unconvincing Motivation: I have reservations about the paper's central motivation. The authors justify their approach by citing VBench results, claiming that the quality gap between the Wan 14B and 1.3B models is negligible. This premise is weak for two reasons: VBench is not a widely accepted authoritative benchmark, and its metrics may not reliably capture perceptual quality. The claim contradicts the general consensus in the community that the visual quality of WanX-14B is significantly superior to WanX-1.3B, a difference that is readily apparent to the human eye. Relying on a questionable benchmark that conflicts with perceptual reality undermines the paper's foundation.

3. Incomplete Survey of Related Work: The related work section is missing a crucial category of acceleration techniques: step distillation. Methods like DMD2, used in works such as FastVideo [1], achieve substantial acceleration (20-30x) and are highly relevant. The authors should include a thorough discussion of these methods and position their work in relation to them.

4. Redundant Writing: The manuscript is repetitive. The core methodology is restated multiple times throughout the paper without offering new insights or details, which detracts from its clarity and impact.

[1]. FastVideo: A Unified Framework for Accelerated Video Generation, https://github.com/hao-ai-lab/FastVideo

### Questions
No

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a training-free acceleration method for video diffusion sampling, called SRDiff, which leverages collaboration between large and small models to reduce the inference computational cost. The main idea behind the method is like the speculative decoding. From the experiments on VBench, RSDiff can accelerate the sampling speed up to 3x and 2x on Wan and CogVideoX, respectively.

### Strengths
The core idea of this paper is initutive and straightforwad. It utilizes the large model handles high-noise steps to ensure semantic and motion fidelity, while utilizes the smaller model refines visual details in low-noise steps. The authors provide the solutions on the same VAE family and cross-frame VAE family.

### Weaknesses
First, I need to point out the writing issues of this paper. The reference format is totally wrong. I think the authors mis-use \citep and \citet, which makes the paper hard to read. And the core method are introduced several times during the whole paper. I hope the authors can provide more interesting phonemenon or insights from this method.

Next, I need to point out the technical issues.

1. Insufficient related work.  As a training-free acceleration method, this paper misses a lot of disscussions with other acceleration methods like FastVideo [1], CausVid [2] etc.

2. Limited novelty. As I mentioned in before, the idea of this method is initutive and straightforward, but this method has been applied on several models like $\textbf{Wan2.2}$ [3], which utilizes a MoE structure with a high-denoise model and low-denoise model, and $\textbf{SDXL-refiner}$ [4], an ensemble of experts pipeline.

3. Insufficient experiment results. Through the whole paper, all experiments are conducted on the VBench.

4. The method utilizes two models for a single video generation, how about the memory cost of SRDiff?

[1] FastVideo: https://github.com/hao-ai-lab/FastVideo

[2] From Slow Bidirectional to Fast Autoregressive Video Diffusion Models: https://github.com/tianweiy/CausVid

[3] Wan2.2: https://github.com/Wan-Video/Wan2.2

[4] SDXL-refiner: https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0

### Questions
I think the authors need to clarify the contributions and the significance of this paper. Provide more insights behind the proposed methods. More details please see the weaknesses.

### Soundness
3

### Presentation
3

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
The paper proposes SRDiffusion, a training-free method to accelerate video diffusion inference by combining a large model for early high-noise steps (semantic fidelity) with a smaller model for later low-noise steps (detail refinement). An adaptive switching metric ensures a good balance between speed and quality. Experiments on Wan and CogVideoX show up to 3x speedup with negligible quality loss, outperforming caching-based baselines and complementing other optimizations.

### Strengths
1) The paper is well written; it is easy to read and understand.
2) The core idea is very simple and shows good performance as evidenced by quality metrics.
3) The method is training-free, does not require any additional training or finetuning.
4) The approach is flexible and extendable to different video diffusion transformer model families; cross family experiments are especially interesting.
5) The idea is orthogonal to other acceleration techniques, so it can be combined with other methods to improve efficiency even further.

### Weaknesses
1) While effective and new, the core concept (switching between large and small models) is relatively simple and may be seen as incremental rather than fundamentally novel.
2) The paper relies solely on automated metrics (VBench, LPIPS, SSIM, PSNR), which are imperfect proxies for perceptual quality; lack of user studies or preference tests is concerning for a video generation task.
3) No video examples in supplementary: Only rolled-out frames are shown, making it hard to judge temporal consistency and overall perceptual quality.
4) Maintaining two or more models during inference may pose memory and deployment challenges, especially in resource-constrained environments; this is not addressed in detail.

### Questions
1) How sensitive is the delta threshold to different prompt types? Could a learned policy outperform the current heuristic?
2) Have you considered latent alignment techniques (e.g., learned mapping) to reduce quality loss when switching across different VAEs?
3) What are the memory and compute implications of maintaining two models during inference? Any strategies for deployment on resource-constrained devices?
4) Why were no user studies or preference tests included? Do you plan to validate perceptual quality beyond automated metrics?
5) Will SRDiffusion work for step-distilled models (e.g., DMD) that achieve high-quality generation in 1–4 steps? If not, doesn’t this severely limit the practical relevance of your approach compared to these highly efficient alternatives?

### Soundness
3

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
3

### Summary
The paper observes large models excel at semantics in high-noise steps, while small models are efficient at low-noise steps. SRDiffusion therefore uses a large model for early steps (“Sketching”) and a small model for later steps (“Rendering”), with an adaptive switch metric. Standard diffusion notation is used: forward process $x_t=\sqrt{\alpha_t},x_{t-1} + \sqrt{1-\alpha_t},\varepsilon_t$ with $\varepsilon_t!\sim!\mathcal{N}(0,I)$, and decoding via a scheduler $\Phi$ to compute $x_{t-1}=\Phi!\big(x_t,t,O(x_t,t)\big)$. The switch metric is $D_t=\tanh!\big(\lVert x_t-x_{t+1}\rVert_1/\lVert x_{t+1}\rVert_1\big)$, and the hand-off is triggered when $0<D_t-D_{t+1}<\delta$ after a minimum number of steps. The method composes with caching (TeaCache) and FP8 attention (SageAttention), reporting up to $3.9\times$ speedups with small quality deltas; combined gains can reach $6.22\times$ on recent GPUs.

### Strengths
1. Originality. “Sketch-then-render” across noise regimes is a crisp insight.

2. Strong empirical speedups with minimal quality loss; synergy with TeaCache/FP8 attention.

3. Method, switch metric, and system composition are described concretely; diffusion equations are explicit.

### Weaknesses
1. Resource footprint. Requires hosting two models; memory and orchestration overheads on single-GPU edge devices need quantification.

2. Switch robustness. Generalization of the switch threshold $\delta$ across datasets/models is only partially explored.

3. Evaluation scope. Broader perceptual/temporal metrics and user studies would strengthen the evidence.

4. Reproducibility: missing details on schedulers and guidance settings that strongly affect quality/latency.

### Questions
1. What are peak VRAM and end‑to‑end latency (including encode/decode) under single‑ and multi‑GPU serving?

2. Can $\delta$ be auto‑calibrated from runtime signals (e.g., similarity proxies) instead of manual tuning?

3. How does performance change for long clips, high resolutions, and strong camera motion?

4. What happens if the small model’s capacity is substantially lower—does the handoff still preserve semantics?

### Soundness
3

### Presentation
3

### Contribution
3
