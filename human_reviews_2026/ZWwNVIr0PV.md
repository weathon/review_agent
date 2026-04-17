# DiSA: Diffusion Step Annealing in Autoregressive Image Generation

- Decision: Reject
- Scores: 6, 4, 6

## Abstract
An increasing number of autoregressive (AR) models, such as MAR, FlowAR, xAR, and Harmon adopt diffusion sampling to improve the quality of image generation. However, this strategy leads to low inference efficiency, because it usually takes 50 to 100 steps for diffusion to sample a token. This paper explores how to effectively address this issue.
Our key motivation is that as more tokens are generated during the AR process, subsequent tokens follow more constrained distributions and are easier to sample. To intuitively explain, if a model has generated part of a dog, the remaining tokens must complete the dog and thus are more constrained. Empirical evidence supports our motivation: at later generation stages, the next tokens can be well predicted by a multilayer perceptron, exhibit low variance, and follow closer-to-straight-line denoising paths from noise to tokens. 
Based on our finding, we introduce diffusion step annealing (DiSA), a training-free method that gradually uses fewer diffusion steps as more tokens are generated, e.g., using 50 steps at the beginning and gradually decreasing to 5 steps at later stages. Because DiSA is derived from our finding specific to diffusion in AR models, it is complementary to existing acceleration methods designed for diffusion alone. 
DiSA can be implemented in only a few lines of code on existing models, and albeit simple, achieves $5-10\times$ faster inference for MAR and Harmon and $1.4-2.5\times$ for FlowAR and xAR, while maintaining the generation quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the inefficiency problem in autoregressive image generation models that incorporate diffusion-based sampling (e.g., MAR, FlowAR, xAR, Harmon). These models typically require 50–100 diffusion denoising steps for each token, leading to high inference latency. The paper proposes Diffusion Step Annealing, a training-free inference-time strategy that gradually reduces the number of diffusion steps as more tokens are generated

### Strengths
1. The paper presents a convincing empirical study demonstrating that later AR steps have more constrained distributions.
2. Experiment results demonstrate consistent speed-ups across four major AR-diffusion models (MAR, FlowAR, xAR, Harmon) with minimal loss in quality.
3. The paper is well-written, logically structured, concise, and clear, making it easy for readers to understand.

### Weaknesses
1. The annealing schedule (linear, cosine, two-stage) and the choice of T_early, T_late are not extensively analyzed. The robustness of these settings across datasets and models could be better demonstrated.
2. The idea of step annealing has precedents in pure diffusion models (e.g., DDIM, DPM-Solver). The novelty here lies in transferring and validating this principle within autoregressive-diffusion frameworks.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors show that in AR with diffusion architectures, the later AR steps have tighter token distributions and straighter denoising paths, so diffusion can run with fewer steps without hurting quality. They propose DiSA, a training‑free schedule that uses more diffusion steps early and gradually fewer later (two‑stage / linear / cosine schedulers), reducing per‑token diffusion effort as conditions strengthen. Evidence includes: (i) MLPs can better predict later tokens; (ii) the variance of diffusion‑sampled tokens decreases with AR progress; and (iii) path straightness increases (Fig. 3). Finally, the authors show the effectiveness of DiSA on ImageNet‑256 with various AR models.

### Strengths
* Clear empirical insight (straighter late‑stage denoising) turned into a simple, general sampler schedule that’s easy to be equipped with various AR + diffusion architectures (Fig. 1)

* Strong evaluation: various image‑level metrics (such as FID/IS/Precision/Recall), per‑image time, and complements existing diffusion accelerators (Table 3).

* Practical wins on both ImageNet 256 x 256 and T2I GenEval (Harmon) with concrete speed–quality curves (Fig. 5).

### Weaknesses
I'm not an expert in this area, but I have some concerns and questions based on my understanding.

* About novelty

I appreciate the practical acceleration idea, but the paper mainly relies on the diffusion-step annealing strategy without much theoretical/mathematical evidence. Even a bit of math or intuition on why this schedule makes sense would make the work solid.

* Scheduler robustness

How sensitive is performance to the exact annealing schedule (e.g., 50 -> 5)? Could the authors provide a hyper‑sweep and an auto‑tuning rule per model?

* Automatic scheduling

The proposed heuristics look promising but ad‑hoc. Can the method learn a schedule online from uncertainty/variance signals?

* About experiments

The experiments appear to focus mainly on 256×256-scale data (largely ImageNet or similar), which raises questions about generalization. It would be helpful to see results on higher-resolution settings (e.g., 512×512, 1024×1024) since frequency characteristics can change substantially with resolution and texture complexity.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a training-free strategy to accelerate AR diffusion models by gradually reducing the number of diffusion steps during token generation. Through empirical analysis, the authors find that as more tokens are generated, the diffusion head becomes increasingly constrained, and later tokens require fewer denoising steps. Based on this observation, DiSA linearly anneals the diffusion steps, achieving notable speed-up while maintaining comparable FID and IS scores. The method is plug-and-play and complementary to existing diffusion samplers.

### Strengths
DiSA introduces a new interpretation of diffusion dynamics within AR generation. As conditioning strengthens across timesteps, the diffusion process becomes inherently easier. Unlike prior accelerators (e.g., DDIM, DPM-Solver, LazyMAR), which assume uniform difficulty and globally reduce steps, DiSA models the heterogeneity of diffusion necessity over AR progression. This observation is theoretically supported through denoising-path straightness analysis, linking DiSA to the geometry of diffusion ODEs and providing a principled foundation rather than a heuristic adjustment.

The paper substantiates its hypothesis through three orthogonal metrics (prediction accuracy, variance reduction, and trajectory straightness) forming a multi-faceted empirical argument. This triangulated evidence distinguishes DiSA from earlier works that rely solely on output metrics like FID or IS. The combination of quantitative analysis and visual trajectory interpretation strengthens the empirical credibility of its claims.

DiSA is a training-free and architecture-agnostic plug-in, requiring no parameter updates or structural modifications. By only modifying the diffusion step schedule (e.g., from 50→5), it achieves up to 5–10× acceleration on MAR and 1.4–2.5× on FlowAR/xAR with negligible degradation in generation quality. The efficiency-to-complexity ratio clearly surpasses methods like FAR or speculative decoding, demonstrating elegance through minimal intervention.

### Weaknesses
While DiSA is empirically well-justified, it remains largely heuristic. The diffusion-step schedule is fixed (typically linear), without a principled derivation from diffusion dynamics or uncertainty theory. In contrast, prior works like AdaDiff or Rectified Flow introduce adaptive step sizes based on explicit error or confidence estimation. DiSA assumes the AR step index monotonically correlates with conditional strength, an assumption not guaranteed for complex prompts. A theoretical analysis linking token entropy or local curvature to optimal diffusion steps would strengthen generality and interpretability.

The training-free nature is practical but introduces a potential mismatch: DiSA modifies the inference-time denoising schedule without retraining the diffusion head, which was originally optimized for uniform timesteps. This causes instability in some models (e.g., MAR required time-offset corrections). By contrast, retraining-based accelerators (e.g., FAR) maintain consistency between training and inference dynamics. Exploring fine-tuning or joint schedule learning could reduce this gap.

All evaluations are limited to ImageNet-256 and GenEval benchmarks. The method’s behavior under higher resolutions, complex spatial layouts, or multimodal conditioning remains untested. Additionally, the study reports only FID and IS, omitting perceptual (LPIPS), semantic (CLIP-Sim), or human-alignment metrics used in recent acceleration research. This narrower evaluation spectrum makes it difficult to quantify subtle degradation patterns or aesthetic trade-offs.

### Questions
The experiments focus on 256×256 ImageNet and GenEval.
Could the authors comment on expected behavior for higher resolutions or long-horizon text prompts? For instance, when later tokens represent finer local details, does the “easier-later” assumption still hold?

DiSA shows minimal quality loss, but where does degradation start?
Please provide an analysis or visualization showing at what step reduction threshold (e.g., 50→1, 50→10) artifacts begin to appear. This would help position DiSA’s safe operating range.

Section 5 briefly mentions heuristics (variance, uncertainty, straightness) for online adjustment.
Could the authors expand on how these heuristics performed and whether they can form the basis of a truly adaptive DiSA variant? This seems like a promising direction that could elevate the contribution beyond a fixed schedule

### Soundness
3

### Presentation
3

### Contribution
2
