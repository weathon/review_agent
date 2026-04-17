# Evolutionary Caching to Accelerate Your Off-the-Shelf Diffusion Model

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Diffusion-based image generation models excel at producing high-quality synthetic content, but suffer from slow and computationally expensive inference. Prior work has attempted to mitigate this by caching and reusing features within diffusion transformers across inference steps. These methods, however, often rely on rigid heuristics that result in limited acceleration or poor generalization across architectures. We propose **E**volutionary **C**aching to **A**ccelerate **D**iffusion models (ECAD), a genetic algorithm that learns efficient, per-model, caching schedules forming a Pareto frontier, using only a small set of calibration prompts. ECAD requires no modifications to network parameters or reference images. It offers significant inference speedups, enables fine-grained control over the quality-latency trade-off, and adapts seamlessly to different diffusion models. Notably, ECAD's learned schedules can generalize effectively to resolutions and model variants not seen during calibration. We evaluate ECAD on PixArt-alpha, PixArt-Sigma, and FLUX.1-dev using multiple metrics (FID, CLIP, Image Reward) across diverse benchmarks (COCO, MJHQ-30k, PartiPrompts), demonstrating consistent improvements over previous approaches. On PixArt-alpha, ECAD identifies a schedule that outperforms the previous state-of-the-art method by 4.47 COCO FID while increasing inference speedup from 2.35x to 2.58x. Our results establish ECAD as a scalable and generalizable approach for accelerating diffusion inference. Our project page and code are available here: https://research.aniaggarwal.com/ecad

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a training-free accelerated sampling algorithm for diffusion transformers. The method achieves its speedup by caching and reusing features across multiple inference steps, thereby avoiding redundant computations within the model.

The core innovation of this work is the introduction of the Pareto frontier to construct efficient, model-specific caching schedules. By leveraging evolutionary algorithms to search for this frontier, the authors identify optimal schedules that achieve a favorable trade-off between inference speed and generation quality, using only a minimal set of calibration prompts.

The authors evaluate their method, ECAD, on state-of-the-art models such as PixArt-α, PixArt-Σ, and FLUX-1.dev. Their experiments are conducted across diverse benchmarks (COCO, MJHQ-30k, and PartiPrompts) and assessed with multiple metrics (FID, CLIP Score, and Image Reward), demonstrating consistent performance improvements over previous approaches.

### Strengths
1. This paper presents a highly novel approach to the cache-based accelerated sampling problem in diffusion models by leveraging concepts from traditional machine learning. I find the most compelling contribution to be the elegant design of the cache schedule and, in particular, the use of an evolutionary algorithm to efficiently search for the optimal schedule. This represents a creative and effective way to frame and solve the problem. 💡

2. The experimental validation in this work is exceptionally thorough and convincing.

- Comprehensive Evaluation: The authors employ a suite of evaluation metrics, including FID, CLIP Score, and ImageReward, ensuring a multi-faceted assessment of generation quality. Their method is rigorously tested on a diverse range of powerful, state-of-the-art models like PixArt-α, PixArt-Σ, and FLUX-1.dev.

- Insightful Ablations and Visualizations: The paper is further strengthened by thorough ablation studies and clear visualizations. The ablation on the number of evolutionary generations, for instance, provides a clear insight into the trade-off between quality (FID) and efficiency (acceleration ratio), showing how FID first decreases before increasing while the speedup consistently grows. Furthermore, the visualizations of the discovered cache schedules and the points on the Pareto frontiers provide strong, intuitive evidence for the effectiveness of the proposed search strategy.

Overall, the rigorous and well-executed experiments strongly substantiate the claims made in the paper.

### Weaknesses
I find several points in the paper confusing, primarily as follows:

1. On the Behavior of Optimization Objectives (ImageReward vs. MACs). I have a question regarding the behavior of the evaluation metrics during the evolutionary search. The paper's objectives are to maximize ImageReward (quality) and minimize MACs (computational cost). I understand how the NSGA-II algorithm operates to find a Pareto frontier for this multi-objective problem. However, the results show that as the number of generations increases, the MACs consistently decrease, while ImageReward does not exhibit a monotonic improvement; instead, it appears to fluctuate. Could the authors clarify this dynamic? My understanding is that while reducing MACs can be achieved by simply caching more, this aggressive caching might negatively impact image quality. Is the fluctuation in ImageReward a result of the algorithm exploring this trade-off, where some generations might sacrifice quality for significant computational savings before finding a better balance? A brief explanation of this trade-off during the search process would be beneficial.

2. On the Representation of the Cache Schedule (S). I know $S \in \{0,1\}^{N,B,C}$, but I don't seem to see $C$ in the visualization. I'd like to understand what role $C$ actually plays.

3. I observe two implicit but critical constraints in the cache schedules: Initial Step Constraint: 

- The first inference step (n=0) must always be a full recomputation, as there is no prior cache to reuse. This is reflected in the visualizations. 

- Intra-Step Dependency: Within a single inference step n, if a block b is set to recompute, all subsequent blocks (b+1,b+2,…) in that same step must also recompute. A reuse decision for a later block would be illogical, as its input depends on the output of the recomputed block b, invalidating the stored cache.

My question is: how are these logical constraints enforced during the evolutionary search, specifically within the Crossover and Mutation operations? A random mutation or crossover could easily generate an invalid schedule that violates these rules. Do the authors employ constraint-aware genetic operators, or is there a "repair" function that corrects invalid schedules after they are generated? Detailing this mechanism would strengthen the paper's description of the algorithm.

### Questions
No

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Evolutionary Caching to Accelerate Diffusion models (ECAD), a genetic algorithm-based framework designed to optimize caching schedules for diffusion models, significantly improving inference speed without compromising image quality. Unlike heuristic-based methods, ECAD formulates caching as a Pareto optimization problem, enabling fine-grained trade-offs between speed and quality. The method is scalable, generalizable across model architectures, and requires no modifications to network parameters. Experiments on PixArt-α, PixArt-Σ, and FLUX-1.dev demonstrate state-of-the-art performance, achieving up to 3.37x speedups while maintaining or improving image quality metrics like FID and Image Reward. ECAD also exhibits strong generalization to unseen resolutions and model variants.

### Strengths
+ The proposed evolutIionary caching framework is novel and effective.

### Weaknesses
- The authors emphasize in the introduction that "there are no restrictions on batch size." However, it is unclear whether they measure generation performance per image and acceleration ratio under different batch sizes. Doing so would provide strong evidence to support this claim.

- For the strongest model, FLUX, it would be more compelling to evaluate its performance on more comprehensive benchmarks, such as GenEval and DPG-Bench, rather than relying solely on toy datasets.

### Questions
see weakness

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
2

### Summary
This paper introduces ECAD, a method that uses genetic algorithms to learn optimal caching schedules for accelerating Diffusion Transformers. The authors demonstrate speedups on models like PixArt and Flux while preserving or even improving image quality across multiple benchmarks.

### Strengths
- framing diffusion caching as a multi-objective Pareto optimization problem to balance quality and efficiency is novel.
- the proposed method is lightweight, and can adapt across prompts, model variants, and resolutions.
- strong empirical performance that achieves state-of-the-art inference acceleration with competitive image quality across several models.
- the paper is well-written and the comprehensive ablation studies in the appendix provide supports for the design choices.

### Weaknesses
- the performance is dependent on the choice of the quality metric used during optimization, and the caching schedules may not generalize perfectly to all possible evaluation metrics.
- the experiment setup (e.g., 100 prompts) may not be representative of complex prompts from real-world applications.

### Questions
- what's the generalizability of a schedule optimized for a certain reward, e.g. Image Reward, when evaluated with a different metric?
- how stable are the Pareto frontiers across random seeds and initializations of population, versus being seeded with heuristic schedules?
- can you analyze the selected prompt in terms of how representative they are of real-world use, and how sensitive is the performance of the proposed method towards prompts with different length, granularity, etc?

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
This paper, “Evolutionary Caching to Accelerate Your Off-the-Shelf Diffusion Model (ECAD)”, addresses the problem of slow and computationally intensive inference in diffusion-based generative models, especially Diffusion Transformers (DiTs). Traditional caching methods reuse previously computed features during sequential inference steps to reduce redundant computation, but existing approaches rely heavily on rigid heuristics, hand-tuned hyperparameters, and limited trade-off flexibility. ECAD reframes feature caching as a multi-objective optimization problem over image quality and computational efficiency, solved via a genetic algorithm (specifically, NSGA-II). The algorithm evolves caching schedules—which specify what components (self-attention, cross-attention, feedforward outputs) to cache, and when—to form a Pareto frontier of quality–latency trade-offs. ECAD is model-agnostic, training-free, and learns these schedules using only a small set of “calibration prompts” and a quality metric (e.g., Image Reward). Experiments on three major diffusion models (PixArt-α, PixArt-Σ, FLUX-1.dev) show that ECAD provides higher-quality outputs at faster inference speeds compared to previous caching baselines such as FORA, ToCa, DuCa, and TaylorSeer. The method is also shown to generalize across resolutions (e.g., 256×256 to 1024×1024) and related model architectures without retraining.

### Strengths
1. **Novel framing and methodological clarity.**  
    The paper introduces a principled optimization perspective (Pareto frontier with NSGA-II) that replaces heuristic-based caching rules. This reframing is conceptually elegant and well-motivated.
    
2. **Training-free and model-agnostic.**  
    ECAD operates without any weight updates or retraining, making it practical for various diffusion models—especially for users without access to large compute resources.
    
3. **Comprehensive experimental evaluation.**  
    Evaluations cover multiple high-profile models (PixArt and FLUX families), a range of datasets (COCO, MJHQ-30K, PartiPrompts), and metrics (FID, CLIP, Image Reward), showing consistent improvements.on prompts, and different reward functions
    
4. **Clear writing and structured presentation.**  
    The paper is well-organized, readable, and includes detailed algorithmic pseudocode and visualizations.

### Weaknesses
1. **Compute inefficiency during optimization.**  
    Although no network training is required, ECAD’s optimization process (hundreds of generations, thousands of images per prompt) might still be computationally expensive. The paper could better quantify this cost relative to real-world savings.
    
2. **Reliance on automatic quality metrics.**  
    The optimization depends on Image Reward or similar automatic metrics, whose correlation to human preference can be imperfect or domain-specific. The paper does not show any human evaluation.
    
3. **Potential overfitting to calibration prompts.**  
    While generalization is discussed, there’s limited theoretical grounding for why a small prompt set suffices to generalize to unseen data.
    
4. **Reproducibility complexity.**  
    The large-scale evolutionary optimization (e.g., 500 generations × 72 candidates) may be difficult for others to reproduce without substantial GPU time.

### Questions
1. How computationally heavy is ECAD’s optimization stage in practice (e.g., total GPU-hours)? Is the cost justified by the inference savings for typical users?
    
2. What are the performance trade-offs if the number of calibration prompts is reduced further (e.g., to 10 or 20)? How stable is ECAD under smaller calibration sets?
    
3. Could ECAD be combined with other acceleration strategies (e.g., distillation or quantization)? Would the Pareto optimization remain consistent?

### Soundness
3

### Presentation
3

### Contribution
3
