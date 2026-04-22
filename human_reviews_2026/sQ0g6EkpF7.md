# RAPID$^3$: Tri-Level Reinforced Acceleration Policies for Diffusion Transformer

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Diffusion Transformers (DiTs) excel at visual generation yet remain hampered by slow sampling.  Existing training-free accelerators—step reduction, feature caching, and sparse attention—enhance inference speed but typically rely on a uniform heuristic or manually designed adaptive strategy for all images, leaving quality on the table. Alternatively, dynamic neural networks offer per-image adaptive acceleration, but their high fine-tuning costs limit broader applicability. To address these limitations, we introduce RAPID^3: Tri-Level Reinforced Acceleration Policies for Diffusion Transformer framework that delivers image-wise acceleration with zero updates to the base generator.  Specifically, three lightweight policy heads—Step-Skip, Cache-Reuse, and Sparse-Attention—observe the current denoising state and independently decide their corresponding speed-up at each timestep. All policy parameters are trained online via Group Relative Policy Optimization (GRPO) while the generator remains frozen.  Meanwhile, an adversarially learned discriminator augments the reward signal, discouraging reward hacking by boosting returns only when generated samples stay close to the original model’s distribution. Across state-of-the-art DiT backbones including Stable Diffusion 3 and FLUX, RAPID^3 achieves nearly 3$\times$ faster sampling with competitive generation quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces RAPID3, a framework that enables per-image adaptive acceleration for large DiT models such as Stable Diffusion 3 and FLUX, without modifying the base generator’s parameters. RAPID3 formulates the sampling process of a DiT as a Markov Decision Process (MDP), in which each denoising step corresponds to a state transition. It attaches three lightweight policy heads, each trained via RL, to dynamically determine Step-Skip Policy, Cache-Reuse Policy, Sparse-Attention Policy. Empirically, RAPID3 achieves nearly 3× faster sampling on SD3 and FLUX with minimal image fidelity degradation.

### Strengths
RAPID3 reframes the inference-time efficiency problem of diffusion transformers as a reinforcement learning (RL) control problem, treating each denoising step as a state-action transition. This perspective is both conceptually elegant and practically impactful. The proposed tri-level policy design—encompassing Step-Skip, Cache-Reuse, and Sparse-Attention strategies—is a novel unification of previously disparate acceleration techniques under a single learnable framework.

### Weaknesses
1. Limited scope of evaluation metrics and qualitative diversity.

The paper’s evaluation primarily relies on image-level alignment metrics such as CLIP-Score, Aesthetic, and GenEval, which mainly capture prompt consistency or aesthetic preference. However, it overlooks popular benchmarks that directly assess the intrinsic quality and compositional correctness of generated images, such as DPG-Bench or other perception-grounded generation metrics. These benchmarks are designed to measure fidelity, entity consistency, and spatial relations—aspects that are particularly sensitive to dynamic acceleration strategies like cache reuse and sparse attention.

2. Limited exploration of policy coordination and interdependence

While RAPID3 introduces three distinct policy heads (Step-Skip, Cache-Reuse, Sparse-Attention), each is trained independently under a shared reward function. The paper does not fully analyze the inter-policy dependencies—for example, how aggressive step-skipping interacts with sparse attention or cache reuse. In practice, these actions may have compounding or conflicting effects on visual fidelity and stability. A more detailed study on joint versus decoupled optimization, or a hierarchical policy controller that adaptively balances the three heads, would strengthen the understanding of policy synergy and improve robustness.

### Questions
Lack of comparison with learned feature caching approaches

The experimental comparisons mainly include training-free baselines (e.g., TeaCache, SpargeAttn) and dynamic fine-tuned models (DyFLUX, TPDM), but they omit recent learned feature caching methods such as HarmoniCa [1] and Learning-to-Cache [2], which also learn adaptive caching or reuse strategies through training. These approaches represent a closer baseline to RAPID3, as they similarly attempt to learn data-dependent cache update schedules rather than rely on manual thresholds. Without direct comparison or discussion, it remains unclear how RAPID3’s reinforcement learning–based policy differs from or improves upon these learned caching paradigms in terms of adaptivity, efficiency, and quality preservation. Including such baselines would better position RAPID3 within the broader family of learned, model-aware acceleration methods.

[1] HarmoniCa: HarmonizingTraining and Inference for Better Feature Caching in Diffusion Transformer Acceleration, ICML 2025.

[2] Learning to-cache: Accelerating diffusion transformer via layer caching, NeurIPS 2024.

### Soundness
2

### Presentation
2

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
This paper presents RAPID3, a reinforcement-learning-based acceleration framework for Diffusion Transformers (DiTs). The key idea is to view the diffusion sampling process as a Markov Decision Process (MDP) and attach three lightweight policy heads—Step-Skip, Cache-Reuse, and Sparse-Attention—in a clean and neat design.

### Strengths
1. The story of this paper is reasonable, and the design is clever. 

2. The proposed method does not modify or fine-tune the base generator. Training only the small policy heads with GRPO is resource-efficient, which makes the framework practical even for closed-source large models.

### Weaknesses
1. It is a little bit weird that the main experiments in Table 1 choose 28 steps as the baseline, which is not a commonly used setting. I want to know why.

2. Given that the method is resource-efficient, more validations on video diffusion or image editing can further strengthen the impact of this paper.

3. The ablation studies are solid but remain surface-level. For example, while the tri-level design (Step/Cache/Sparse) shows quantitative gains, there is little insight into *when* and *why* each policy activates.

### Questions
See Weakness

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
4

### Summary
This paper proposes RAPID3, a system that attaches three tiny, trainable policy heads to a frozen Diffusion Transformer (DiT) to adaptively accelerate inference per image without fine-tuning the generator. The three policies are:

- Step-Skip — learns per-timestep jumps (a Beta-parameterized jump).
    
- Cache-Reuse — decides whether to reuse cached residuals vs recompute.
    
- Sparse-Attention — chooses among discrete attention sparsity patterns per timestep.
    

Policies are trained with a group based policy-optimization (GRPO) objective and an adversarial discriminator is used alongside an image reward model to reduce reward-hacking. Experiments on Stable Diffusion 3 and FLUX show ≈3× speedups while retaining competitive metrics (CLIP, Aesthetic, GenEval/HPS) and much lower training cost than dynamic fine-tuning approaches.

### Strengths
1. **Practical, low-cost adaptivity.**
    
    The paper addresses a highly practical problem (fast inference for large DiTs) and presents a solution that keeps the generator frozen, adding only lightweight heads (~0.025% of generator params) — an attractive tradeoff for deployment and proprietary models. The claimed GPU-hour savings vs fully fine-tuned dynamic models is strong evidence of real utility.   
    
2. **Tri-level design is well motivated and complementary.**
    
    Combining step skipping, cache reuse, and sparse attention covers orthogonal sources of compute (temporal, recomputation, attention cost). Ablations show incremental gains when adding strategies, supporting the architectural choice.   
    
3. **Careful training objective to mitigate reward hacking.**
    
    The use of an adversarial discriminator alongside an image-reward model is an effective and conceptually clean way to prevent policies from exploiting an automatic metric. The Algorithm and RL setup (GRPO) are reasonable for the deterministic frozen-model environment. 
    
4. **Empirical evidence on strong backbones.**
    
    Results on SD3 and the larger FLUX backbone are provided, including comparisons to training-free baselines (TeaCache, ∆-DiT, SpargeAttn), single-strategy RL (TPDM), and a dynamic fine-tuned baseline (DyFLUX). Quantitative results and visualization suggest RAPID3 attains large speedups with competitive quality.   
    
5. **Thorough ablations and implementation detail.**
    
    The appendix and tables include implementation defaults, sensitivity to RL method (GRPO vs RLOO), and data-scale experiments, which strengthen reproducibility and trust in robustness.

### Weaknesses
1. **Benchmarks and baselines — potential fairness concerns.**
    
    The paper compares to many baselines, but details matter: how were hyperparameters for baselines tuned (equivalent computational budget / equivalent “equivalent steps”)? The paper claims equivalent-step fairness but could be clearer on whether baselines were given the same effort for tuning (especially manual adaptive methods). Without fully transparent tuning procedures and seeds, it’s hard to judge whether gains stem from method novelty or hyperparameter selection. 
    
2. **Computational cost during deployment & overheads.**
    
    Although policy heads are tiny, the per-step decision logic, discriminator CPA during training, and any additional memory for caches add overhead. The latency numbers report the DiT forward pass only (excluding text encoder and VAE decoder); reporting wall-clock end-to-end costs (including policy inference) would give a clearer picture of real speedups. The paper should more explicitly quantify policy inference cost and memory overhead.

### Questions
1. **Policy overhead & reproducibility:** Please report the exact inference overhead of running the three policy heads (ms per step) and memory used for caches, and confirm whether the reported latency numbers include policy compute. Also, can you release seeds/hyperparameter configs used for baselines so comparisons are fully reproducible? 
    
2. **Ablation on discriminator strength & dataset bias:** How sensitive are results to the discriminator model choice (architecture, adapter size) and to the reward weight ω? Table 4 shows some effect, but a sweep and discussion of potential adversarial equilibria (where policies game D instead of Q) would be helpful.
    
3. **Generality to other samplers and guidance strengths:** Does RAPID3 interact with different samplers (DPM-solver variants) and classifier-free guidance scales? Appendix claims compatibility, but please show a small table demonstrating sensitivity to sampler/guidance variations.

### Soundness
3

### Presentation
3

### Contribution
3
