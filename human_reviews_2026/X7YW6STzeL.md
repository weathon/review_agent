# Streaming Autoregressive Video Generation via Diagonal Distillation

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
Large pretrained diffusion models have significantly enhanced the quality of generated videos, and yet their use in real-time streaming remains limited. Autoregressive models offer a natural framework for sequential frame synthesis but require heavy computation to achieve high fidelity. Diffusion distillation can compress these models into efficient few-step variants, but existing video distillation approaches largely adapt image-specific methods that neglect temporal dependencies. These techniques often excel in image generation but underperform in video synthesis, exhibiting reduced motion coherence, error accumulation over long sequences, and a latency-quality trade-off. We identify two factors that result in these limitations: insufficient utilization of temporal context during step reduction and implicit prediction of subsequent noise levels in next-chunk prediction (i.e., exposure bias). To address these issues, we propose Diagonal Distillation, which operates orthogonally to existing approaches and better exploits temporal information across both video chunks and denoising steps. Central to our approach is an asymmetric generation strategy: more steps early, fewer steps later. This design allows later chunks to inherit rich appearance information from thoroughly processed early chunks, while using partially denoised chunks as conditional inputs for subsequent synthesis. By aligning the implicit prediction of subsequent noise levels during chunk generation with the actual inference conditions, our approach mitigates error propagation and reduces oversaturation in long-range sequences. We further incorporate implicit optical flow modeling to preserve motion quality under strict step constraints. Our method generates a 5-second video in 2.61 seconds (up to 31 FPS), achieving a 277.3× speedup over the undistilled model.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a new method for distilling diffusion models into autoregresive models. It includes three designs: (1) To mitigate the exposure bias in autoregressive generation, a diagonal distillation strategy is proposed that allocates more inference steps in early chunks and fewer steps in later chunks. (2) A diagonal forcing method is introduced that simulates the same denoising strategy during training to further reduce error accumulation. (3) The authors propose flow distribution matching that performs DMD on motion flow field. Experiments based on Wan2.1 1.3B shows the model achieves real-time performance without much quality degradation.

### Strengths
1. The proposed designs (diagonal sampling trajectory and DMD loss for motion field) are well motivated and effectively address exposure bias while enhancing motion quality.
2. Experimental results demonstrate that the proposed strategy preserves performance with fewer number of steps, thereby improving throughput and reducing latency.

### Weaknesses
1. Missing critical details. The proposed flow distribution matching method (Section 3.3) depends on estimating a "motion flow field" using $\mathcal{F}$. However, the definition of this flow field is unclear, does it refer to optical flow, motion vector, or motion feature? Furthermore, the paper has not discussed the design of the motion estimator, making it difficult to assess its reliability, especially given that it operates on potentially noisy latents.
2. Insufficient comparisons. Since the proposed diagonal distillation method is essentially re-allocating the number of inference steps, it is important to compare against baselines (e.g. self-forcing) under comparable number of steps. For example, it would be better to evaluate (1) both self-forcing and diagonal forcing at 2 NFEs per later chunk, (2) both at 4NFEs per latent chunk.
3. Lack of generalization study. As the inference schedule is largely hand-crafted, it would be better to test the proposed strategy on alternative models (e.g. Wan2.1 14B) to evaluate its transferability without inference hyperparameter tuning.
4. Incomplete related work. The proposed flow distribution matching is related to many motion-based video modeling methods such as VideoJAM [Chefer'25], Video-LaVIT [Jin'24], MicroCinema [Wang'24]. A comprehensive discussion about prior work would help clarify the paper's contribution and improve its clarity.

### Questions
1. How are the training loss coefficients $\lambda_{\textrm{spatial}}$, $\lambda_{\textrm{flow}}$, $\gamma$ determined, and how sensitive is the model's performance to these training hyperparameters?

### Soundness
3

### Presentation
1

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
The paper proposes Diagonal Distillation (DiaDistill) for real-time, streaming autoregressive (AR) video generation. The key idea is an asymmetric, chunk-wise denoising schedule, more steps for early chunks, fewer for later ones, combined with Diagonal Forcing, which conditions each new chunk on a noised version of the previous chunk (reusing the KV cache) to align training and inference and reduce exposure bias. A complementary Flow Distribution Matching loss is introduced to preserve motion amplitude when using very few denoising steps. Empirically, the method is evaluated on a Wan2.1-based text-to-video system and VBench metrics; it reports 31 FPS, 0.37 s first-frame latency, and a 277.3× speedup over an undistilled baseline while retaining visual quality.

### Strengths
1. The diagonal allocation of denoising effort across time (many steps early->few later) is a simple but appealing scheduling concept for AR diffusion models, explicitly exploiting temporal priors accumulated early. The Diagonal Forcing mechanism, feeding noised previous-chunk states (rather than clean frames) as the KV cache, targets exposure bias in a way that is tailored to AR diffusion, not borrowed wholesale from image distillation. 
2. Adding Flow Distribution Matching to align motion distributions (rather than only framewise fidelity) is a sensible, task-appropriate extension to DMD.
3. Streaming, low-latency T2V is important for interactive and real-time applications; a method that roughly doubles acceleration vs. prior AR-diffusion baselines (e.g., vs. “Self Forcing”) while keeping quality could materially impact practice.

### Weaknesses
1. Early sections mention a “diagonal attention mechanism operating jointly across time and denoising steps,” but the implementation centers on scheduling (step counts per chunk) plus conditioning with a noised KV cache. It’s unclear whether there is any architectural change to attention patterns (e.g., block-sparse or strided attention over (time × step) axes) beyond cache reuse. If there is special attention, the paper needs explicit architecture diagrams and tensor shapes; if not, the phrase “diagonal attention mechanism” is misleading.
2. The paper defines a KL over optical-flow distributions and shows a regression term, but it does not specify: which flow estimator is used (RAFT? GMFlow? in-house?), whether it is frozen, if gradients flow through it, how flow fields are normalized, or how flow-estimation errors bias training. Without these details, the flow objective’s reproducibility and reliability are uncertain.
3. VBench categories and the exact metric subsets used (“Temporal Quality,” “Frame Quality,” “Text Alignment”) are referenced, but the paper doesn’t enumerate which VBench dimensions or how they’re aggregated; this matters because VBench includes many sub-metrics with different sensitivities.
4. First-frame latency and throughput are reported, but the streaming protocol (chunk size, overlap, buffering, tokenization, KV cache size, prefill costs) isn’t fully specified. For real deployments, memory footprint from KV-cache reuse and any stall during step-count transitions matter; these are not reported. 
5. The paper claims to mitigate over-saturation and motion attenuation, but does not quantify residual failure rates, sensitivity to prompt types (e.g., rapid scene changes), camera motion vs. object motion, or scaling to higher resolutions and frame rates beyond the default (832×480 @ 16 FPS generation).
6. The quality of the figures provided in the paper appears to be relatively low.

### Questions
1. Is there a true attention change? Please clarify whether “diagonal attention mechanism” denotes a new attention pattern (e.g., attention over (time, step) diagonals) or simply the training/inference schedule + KV-cache conditioning. If it is architectural, provide equations, masks, and complexity; if not, please remove or rephrase to avoid confusion.
2. Which optical-flow network is used? Is it frozen? Do you backprop through it? How are flow distributions parameterized for KL (bins, continuous density, or feature-space scores)?
3. What is the runtime overhead of computing flows during training, and how stable is optimization without the regression term?
4. What chunk size and overlap were used, and how do they affect latency and quality? What is the KV-cache memory footprint over time, and how does cache reuse interact with the noised conditioning frames? Any degradation when streaming beyond 45 s or at higher frame rates/resolutions?
5. The paper highlights schedules like 4322222 for 7 chunks (5s). How is the schedule chosen for other durations (e.g., 10s, 45s)? Is there a principled rule (e.g., warm-up length, geometric decay) or an auto-tuner? Please include results for algorithmic schedule selection (not hand-tuned) and its effect on quality/latency.
6. Can you provide a direct measurement that the method reduces exposure bias (e.g., error growth curves, saturation shift statistics across time, calibration of the “implicit next-noise prediction” claim), rather than only qualitative frames?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Diagonal Distillation for streaming autoregressive video generation distilled from a diffusion teacher. The core idea is a diagonal denoising schedule—early chunks use more steps, later chunks fewer—so later chunks inherit stronger priors while keeping latency low. Two components support this: Diagonal Forcing (training on noisy previous-chunk states/KV cache to better match inference and reduce exposure bias) and Flow Distribution Matching (a temporal loss to preserve motion amplitude otherwise damped by few-step distillation). Experiments on a modern T2V backbone show substantially lower first-frame latency and higher throughput with comparable or slightly better quality on standard video benchmarks, plus qualitative improvements on long-horizon and dynamic-prompting scenarios.

### Strengths
* Timely problem: addresses online/streaming latency, not just offline T2V.
* Coherent design: diagonal schedule + noisy conditioning + flow loss form a simple, compatible recipe.
* Strong practicality: low first-frame latency, high FPS, and straightforward cache reuse.
* Empirical support: consistent speedups with minimal quality loss; informative ablations on step allocation and losses.
* Clarity: figures and narrative make the training–inference mismatch and diagonal rationale intuitive.

### Weaknesses
* Longer Videos Test: While 45 seconds is impressive, many streaming use cases (e.g., live streams) require minutes of content. Does error accumulation reemerge for 1–5 minute videos, and if so, can the diagonal strategy be extended (e.g., adaptive step resets)?
* Insufficient Analysis of Step Allocation Heuristics: A quantitative comparison of more step sequences (beyond the 6 evaluated) would clarify how step allocation impacts the quality-efficiency frontier. For example, does 5422222 yield better early-frame quality at the cost of marginal latency? Besides, the paper assumes step reduction is monotonic (fewer steps over time), but dynamic allocation (e.g., more steps for high-motion chunks) could further optimize performance. No analysis of non-monotonic strategies is provided.

### Questions
Please see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Diagonal Distillation, a novel framework for efficient streaming autoregressive video generation. The method addresses the limitations of existing diffusion-based autoregressive models, i.e., the insufficient utilization of temporal context during step reduction and implicit prediction of subsequent noisel evels in next-chunk prediction (exposure bias). The core idea is to allocate more denoising steps to early chunks and fewer to later ones, forming a diagonal denoising trajectory. To support this, the authors propose two key techniques: Diagonal Forcing which explicitly simulating diagonal denoising paths through controlled noise injection, allowing the model to leverage partially denoised previous chunks as contextual priors. Flow Distribution Matching (FDM) that aligns optical flow distributions between teacher and student models to preserve motion consistency during step reduction.

Experimental results show substantial gains: the method achieves 1.53× speedup over Self Forcing, with comparable visual quality and temporal coherence. Ablation studies validate the gain of each proposed component, and qualitative examples demonstrate stability in long (45s) video generation and dynamic prompting.

### Strengths
1. The proposed design is novel and well-motivated. The diagonal denoising idea is conceptually elegant, aligning the temporal and diffusion-step dimensions in a unified framework. This bridges autoregressive conditioning with step-efficient diffusion distillation.

2. This work addresses critical problems in streaming generation: insufficient utilization of temporal context during step reduction and implicit prediction of subsequent noise levels in next-chunk prediction (exposure bias).

3. The authors demonstrate strong empirical results, including both quantitative and qualitative evaluations across different benchmarks, and also include a thorough user study in the appendix. The reported 1.53× speedup with minimal quality drop is impressive and relevant for real-time applications.

4. The ablation study is solid. Table 2, 3; Figure 5 convincingly show how Diagonal Forcing and Flow Distribution Matching contribute to temporal coherence and motion fidelity.

### Weaknesses
1. Limited novelty in the distillation objective. While the diagonal scheduling and forcing mechanism are new, the underlying distillation objective remains close to prior work Self Forcing with Distribution Matching Distillation. The conceptual leap may be seen as an engineering refinement rather than a fundamentally new learning principle.

2. Some format flaws, e.g., order of Table 1, 2 and 3. Caption font is too small. The resolution seems to be very different in Figure 7.

### Questions
1. Can you provide the total scores in ablation study (Table 2)? 
2. The generated videos in the demo have some scene cut and flickering, what is the reason behind that? Is the proposed method tends to cause more artifacts compared to the original Self Forcing? 
3. Can you provide a breakdown of the speedup?

### Soundness
4

### Presentation
4

### Contribution
4
