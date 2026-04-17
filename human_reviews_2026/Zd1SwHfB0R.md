# OUSAC: Optimized gUidance Scheduling with Adaptive Caching for DiT Acceleration

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Diffusion models have emerged as the dominant paradigm for high-quality image generation, yet their computational expense remains substantial due to iterative denoising.
Classifier-Free Guidance (CFG) significantly enhances generation quality and controllability but doubles the computation size by requiring both conditional and unconditional forward passes at every timestep. 
We present OUSAC ($\textbf{O}ptimized$ $g\textbf{U}idance$ $\textbf{S}cheduling$ with $\textbf{A}daptive$ $\textbf{C}aching)$, 
a framework that accelerates diffusion transformers (DiT) through systematic optimization. 
We begin with two key observations that   reveal acceleration opportunities:
 first, the importance  of guidance varies dramatically across timesteps -- while a few critical steps require strong guidance, most steps need minimal or even no guidance; second, variable guidance patterns introduce denoising deviations that undermine the standard caching methods, which assume constant CFG scales and future similarity across steps.
Moreover,  different transformer  blocks are affected with different levels under dynamic conditions.
This paper develop a two stage approach leveraging these insights. Stage-1 employs  evolutionary algorithms to discover sparse guidance schedules that 
apply CFG only at critical timesteps, which eliminates up to 82\% of unconditional passes. Stage-2 introduces an adaptive rank allocation strategy that tailors 
calibration efforts per transformer block, maintaing caching effectiveness under variable guidance.
Experiments demonstrate that OUSAC significantly outperforms the state-of-the-art acceleration methods.
Specifically, it achieves 
53\% computational savings and a 15\% improvement in generation quality on DiT-XL/2 (ImageNet 512$\times$512), as well as 60\% savings with 16.1\% quality improvement on PixArt-$\alpha$ (MSCOCO)
.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
OUSAC accelerates DiT by applying Classifier-Free Guidance only at key timesteps and adaptively caching intermediate features. Its two-stage optimization—first using evolutionary search to find sparse CFG schedules, then applying coordinate descent to determine per-block cache ranks—reduces computation on both DiT-XL/2 and PixArt-α while improving FID.

### Strengths
The paper proposes a dynamic CFG weighting method and a corresponding tailored feature caching method for image generation, which maintains similar performance while saving computation.

### Weaknesses
1. Lack of novelty: Only optimized FID scores using coordinate descent, which are unknown for the performance gains of other more important metrics, and a lack of more visualization on T2I tasks, leaving real-world improvement unclear.
2. Outdated Model Evaluation: Evaluates on DiT, PixArt-α, which is outdated; lacks validation on modern models like SD3.5, FLUX, or LightingDiT.
3. Limited Benchmarks: Only uses FID; missing key T2I benchmarks like CLIP Score, Geneval, DPG-Bench, and T2I-CompBench for compositional and semantic evaluation.
4. Overly Engineering: Relies heavily on evolutionary and coordinate descent optimization; lacks theoretical insight or general principles.
5. Optimization Cost Unclear: Offline optimization cost not reported; unclear if practical for large-scale deployment.

### Questions
1. How do modern text-to-image (T2I) models perform on the benchmarks mentioned above?
2. How do few-step generation methods, such as shortcut models or MeanFlow, compare in terms of performance?
3. What is the offline optimization cost, and do these methods scale effectively to larger models?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes OUSAC, a training-free acceleration framework for Diffusion Transformers (DiTs) that jointly optimizes Classifier-Free Guidance (CFG) scheduling and feature caching to reduce inference cost without sacrificing image quality. Through this joint optimization, OUSAC achieves 53–60% computational savings while improving FID by up to 16% on DiT-XL/2 (ImageNet-512) and PixArt-α (MSCOCO). Importantly, both optimizations are performed once per pre-trained model, producing fixed schedules and rank configurations that generalize across prompts, enabling plug-and-play inference acceleration without retraining.

### Strengths
1. OUSAC is the first to recognize and systematically handle the interdependence between variable guidance and cache calibration. The use of an evolutionary optimization framework for discovering sparse guidance schedules is particularly original, allowing training-free, gradient-free optimization over a hybrid discrete–continuous search space.

2. The methodological formulation is technically solid and well-motivated. Stage-1’s optimization objective is rigorously defined, balancing fidelity and sparsity via a population-based search that avoids vanishing gradients. Stage-2’s adaptive rank allocation builds upon empirical observations of heterogeneous reconstruction errors across Transformer blocks, leading to a principled coordinate-descent optimization under compute constraints. The experimental section is extensive, covering multiple datasets (ImageNet, MSCOCO) and comparing against both training-free (ICC) and learned (L2C, HarmoniCa) baselines. The quantitative improvements—up to 60% compute savings and 15–16% FID gains—support the method’s validity.

### Weaknesses
1. Limited evaluation diversity and generalization scope.

Although the paper demonstrates consistent improvements on DiT-XL/2 (ImageNet) and PixArt-α (MSCOCO), both models share a diffusion-transformer backbone and similar sampling dynamics. It remains unclear whether OUSAC’s sparse scheduling and adaptive caching generalize to recent state-of-the-art text-to-image models, such as FLUX.1-dev, Stable Diffusion 3.5 (SD3), or Qwen-Image, which feature substantially different architectures, noise schedules, and guidance implementations. Evaluating OUSAC on these newer large-scale generative models—or at least discussing potential adaptation challenges—would significantly strengthen the claim of architectural generality and practical relevance to current T2I systems.

2. Static (non-adaptive) schedule limits flexibility.

The discovered CFG schedule and caching ranks are optimized once per model and then fixed during inference. While this ensures training-free usage, it also prevents adaptation to prompt-dependent complexity — for example, highly detailed or abstract prompts may benefit from different guidance densities. Incorporating a lightweight inference-time mechanism (e.g., dynamic thresholding based on intermediate latent variance or CLIP similarity) could improve robustness and better exploit the sparse guidance principle.

### Questions
Missing evaluation on generative fidelity benchmarks beyond FID.

The experiments focus on FID and IS, but omit metrics that describe semantic and relational consistency in text-to-image generation, such as DPG-Bench or GenEval. These benchmarks capture alignment and structural quality beyond low-level distribution matching. Including such evaluations could reveal whether sparse guidance schedules preserve global–local coherence under complex prompts.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes OUSAC, a two-stage, training-free framework to accelerate diffusion transformers (DiT). Stage-1 uses an evolutionary (gradient-free) search to discover per-timestep, sparse Classifier-Free Guidance (CFG) schedules so that CFG (which normally requires an unconditional and conditional forward pass per step) is applied only at a small subset of timesteps. Stage-2 compensates for the larger temporal feature shifts introduced by non-uniform guidance by performing adaptive rank allocation (regionwise SVD calibration) for cached transformer features via coordinate descent. On DiT-XL/2 (ImageNet) and PixArt-α (MSCOCO), OUSAC claims very large compute reductions (e.g., using guidance at 9/50 timesteps) with substantial MACs/latency savings and improved FID vs baselines (tables show up to ≈50–60% MAC savings while matching or improving FID). The paper also reports ablations showing adaptive rank allocation outperforms uniform calibration.

### Strengths
1. **Practical, training-free approach.** The method operates post-hoc on pretrained DiT models (no extra model training), which is attractive for adoption. The two stages are performed offline once per model and then used at inference. 
    
2. **Diverse experiments.** Evaluation on large, realistic models/datasets (DiT-XL/2 on ImageNet 256/512; PixArt-α on MSCOCO) with standard metrics (FID, IS, sFID, CLIP score) and multiple baselines (ICC, L2C, Harmonica, DDIM) gives the paper empirical breadth. Reported improvements in both compute and FID are compelling in the tables.
    
3. **Ablations that probe components.** The paper includes ablations showing (a) the role of reference trajectory length in schedule discovery, and (b) the benefit of adaptive rank allocation vs a set of uniform ranks. These ablations increase trust that each component contributes.

### Weaknesses
- **Missing/insufficient reporting of optimization cost & reproducibility concerns.** The method’s Stage-1 uses evolutionary optimization (population sampling, multiple generations) to search a T-dimensional schedule; Stage-2 uses coordinate descent over region ranks. The paper gives search hyperparameters (e.g., 15 generations for DiT) but does not report the wall-clock compute, GPU hours, or search cost needed to discover schedules and rank assignments nor whether the search is practical for large models. This is crucial: a large offline search cost could negate the stated inference gains. 
    
- **Ablation depth and hyperparameter sensitivity.** The method introduces multiple design choices and hyperparameters (e.g., threshold τ to disable unconditional pass, wmax, population size, number of generations, K regions for rank allocation, rmin/rmax). The paper gives some settings but lacks a systematic sensitivity analysis showing the method is not fragile to these choices.
    
- **Lack of relevant work.** The works TGate (which considers CFG and caching)[1], the classic cache acceleration method Delta-DiT[2], and the recent state-of-the-art cache methods TeaCache[3] and TaylorSeer[4] are currently not mentioned in the related work or included among the baselines.
    
- **Old pretrain model.** In this paper, we conduct experiments on two basic models, DiT-XL/2 and pixart-alpha, which are both models from a long time ago.
    

[1] Faster Diffusion via Temporal Attention Decomposition

[2] Δ-DiT: A Training-Free Acceleration Method Tailored for Diffusion Transformers

[3] Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model

[4] From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers

### Questions
1. **Robustness / transfer.** How does a schedule discovered for one guidance scale (e.g., baseline w=1.5) behave when used with different guidance scales, different prompts (e.g., long captions vs short), or different samplers? Can schedules transfer between model sizes (e.g., DiT-XL/2 → smaller DiT) or do they need to be rediscovered per model? Provide experiments or analysis. 
    
2. **Hyperparameter sensitivity.** How sensitive are results to the threshold τ, wmax, number of generations, population size, and K (number of regions)? A short sweep or sensitivity table would strengthen claims of robustness.
    
3. **Interaction with other accelerations.** Can OUSAC be combined with sampling-reduction methods (DPM-Solver, progressive distillation) or pruning/quantization? If yes, does the guidance schedule discovery need adaptation? Experimental demonstration would increase the paper’s impact.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces OUSAC, a two-stage optimization framework to accelerate Diffusion Transformers (DiT) under Classifier-Free Guidance (CFG). Stage 1 uses evolutionary search to learn sparse, timestep-specific guidance schedules that skip unconditional passes when guidance is unimportant. Stage 2 adds adaptive rank allocation for increment-calibrated caching, assigning different SVD ranks to transformer regions to handle the feature inconsistencies caused by variable guidance. Experiments on DiT-XL/2 (ImageNet 512×512) and PixArt-α (MSCOCO 256×256) report up to 53–60 % computation reduction with modest or improved FID (e.g., 2.72 vs 3.20 on DiT-XL/2). The authors claim that OUSAC is training-free, improves both efficiency and quality, and generalizes across prompts.

### Strengths
1. The paper is clearly written and the motivation is easy to follow.

2. Integrating guidance scheduling and caching is a practical and coherent idea.

3. Results on DiT and PixArt-α show consistent speedup with comparable quality.

4. The approach is training-free and could be added to existing DiT models.

### Weaknesses
1. Limited relevance: The method assumes Classifier-Free Guidance, but many recent models (e.g., FLUX) do not use CFG at all. The paper does not explain how OUSAC would work in those settings.

2. Missing search-cost details: Stage-1 evolutionary search likely requires many full generations, but the paper gives no runtime, GPU hours, or total evaluations.

3. Generalization claims unproven: The paper claims that the discovered guidance schedule “generalizes across different prompts and conditions,” but provides no explanation or evidence for why this should hold. In practice, the optimal guidance steps are likely input-dependent, since noise trajectories and attention dynamics vary with prompt complexity. The current approach learns a single fixed sparse pattern, which seems more dataset-specific than truly generalizable.

4. Incremental novelty: Both sparse CFG and caching are known ideas; OUSAC combines them but adds limited new insight.

5. Small improvements: The reported FID gains (≈0.2–0.3) are minor and within normal variance.

### Questions
1. How does OUSAC apply to CFG-free models like FLUX?

2. What is the total cost of the evolutionary search (generations, population size, GPU hours)?

3. Why should the discovered schedule generalize to new prompts or datasets?

4. What is the actual cache-reuse rate in practice?

### Soundness
3

### Presentation
3

### Contribution
2
