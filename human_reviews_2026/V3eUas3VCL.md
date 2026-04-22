# DiffSparse: Accelerating Diffusion Transformers with Learned Token Sparsity

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Diffusion models demonstrate outstanding performance in image generation, but their multi-step inference mechanism requires immense computational cost. Previous works accelerate inference by leveraging layer or token cache techniques to reduce computational cost. However, these methods fail to achieve superior acceleration performance in few-step diffusion transformer models due to inefficient feature caching strategies, manually designed sparsity allocation, and the practice of retaining complete forward computations in several steps in these token cache methods.
To tackle these challenges, we propose a differentiable layer-wise sparsity optimization framework for diffusion transformer models, leveraging token caching to reduce token computation costs and enhance acceleration. Our method optimizes layer-wise sparsity allocation in an end-to-end manner through a learnable network combined with a dynamic programming solver. Additionally, our proposed two-stage training strategy eliminates the need for full-step processing in existing methods, further improving efficiency.
We conducted extensive experiments on a range of diffusion-transformer models, including DiT-XL/2, PixArt-$\alpha$, FLUX, and Wan2.1. Across these architectures, our method consistently improves efficiency without degrading sample quality. For example, on PixArt-$\alpha$ with 20 sampling steps, we reduce computational cost by 54% while achieving generation metrics that surpass those of the original model, substantially outperforming prior approaches. These results demonstrate that our method delivers large efficiency gains while often improving generation quality.
.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on the issue of high computational cost of diffusion models, and proposes the **DiffSparse** framework, which achieves acceleration through optimization via learnable layer-wise sparsity. This method eliminates the need for manual adjustment of sparsity parameters and realizes automated sparsity allocation through end-to-end learning, making it more efficient than traditional feature caching methods that rely on manual sparsity allocation. DiffSparse performs excellently on a variety of diffusion models and possesses cross-resolution generalization capability. Meanwhile, compared with search methods such as genetic algorithms, it also features higher training efficiency.

### Strengths
### 1. Innovative Method Design & Problem Formulation
Redefines diffusion transformer acceleration as a **differentiable layer-wise sparsity allocation problem**, abandoning the manual sparsity schedules of prior token cache methods (e.g., ToCa). It combines a learnable sparsity cost predictor, dynamic programming solver, and adaptive token selector into a unified framework. Additionally, it eliminates predefined full-step computation via a two-stage training strategy to unlock token caching’s full acceleration potential, and achieves cross-resolution generalization where low-resolution-trained sparsity predictors work for high-resolution models without retraining.

### 2.  Robust Experimental Validation
It conducts comprehensive experiments, validating on 4 models (DiT-XL/2, PixArt-α, FLUX, Wan2.1) across image/video generation—for example, PixArt-α achieves 54% cost reduction (1.91× speedup) with better FID than ToCa. Thorough ablation studies confirm the optimality of two-stage training, LPIPS loss, and 0.25 sparsity interval. It also outperforms search methods like genetic algorithms with 4× faster training.

### Weaknesses
### 1. Insufficient Analysis of Computational Overhead
- **Unreported training/inference overhead of DiffSparse components**: The paper claims the dynamic programming solver has low complexity $O((L \cdot T)^2 \cdot |S|)$ but does not report the **actual runtime overhead** of the sparsity cost predictor or the dynamic programming solver during inference. 
- **Lack of memory consumption breakdown**: While noting cross-resolution generalization reduces memory usage, the paper does not provide quantitative data (e.g., peak memory for 256×256 vs. 512×512 training with DiffSparse) or compare it to baselines.

### 2. Limited Generalization Validation
- **No testing on ultra-low-step models**: The paper excludes distillation methods but does not test DiffSparse on **1–2 step distilled diffusion models** (e.g., SDXL-Turbo). 

### 3. Incomplete Ablation of Key Components
- **No ablation of dynamic programming solver necessity**: The paper claims the dynamic programming solver optimizes global sparsity, but it does not compare against a "no solver" baseline (e.g., using the cost predictor’s output directly without global constraint optimization). This fails to prove the solver’s contribution to performance.

### Questions
### **1. Similarity of Dynamic Sparsity Allocation in Other Feature Caching-Based Diffusion Acceleration Works**
 Are there other papers that use Feature caching to accelerate diffusion models and have similar dynamic sparsity allocation ideas?

### **2. Feature Recomputation and Utilization Logic After Sparsity Determination**
Although the method in this paper aims to achieve dynamic sparsity allocation, it lacks mention of how features are recomputed and utilized after sparsity is determined. For example, does this paper directly adopt previous work, or it make modifications based on previous methods?

### **3. Input, Cost Definition, and Calculation Logic of Learnable Sparsity Cost Predictor**
What is the input of the Learnable Sparsity Cost Predictor? What is "Cost" defined as, and how is the Cost of a specific layer of the model at a certain timestep calculated?

### **4. Rationale for Using Formula (9) to Integrate Step and Layer Costs**
 Formula (9) is intended for warm-start. Why can it be used to integrate step and layer costs? I cannot see the logic connection for now.

### **5. Lack of Performance Comparison with Other Non-Feature-Caching-based Acceleration Methods**
Although this paper focuses on accelerating diffusion models using Feature Caching, the experiments lack performance comparisons with non-Feature caching acceleration methods, such as Model pruning, Rectified Flow, and knowledge distillation.

### Soundness
3

### Presentation
1

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
This paper introduces DiffSparse, a framework to accelerate diffusion transformer models by optimizing layer-wise token sparsity during the sampling process. It employs a learnable sparsity cost predictor and dynamic programming to allocate computation efficiently, eliminating the need for manual sparsity tuning and full-step computations. The proposed two-stage training strategy enhances acceleration while preserving image generation quality. Extensive experiments on various models demonstrate significant speedups (up to 1.91×) with improved or comparable performance to state-of-the-art methods.

### Strengths
+ The proposed differentiable layer-wis sparsity optimization is novel and effective.

### Weaknesses
1. What is the training cost for FLUX and Wan?

2. The paper would benefit from reorganization. For instance, the related work section is overly long and could be trimmed and it would be better to include visualizations directly in the main paper to enhance clarity and accessibility.

3. The discussion should also include other token-based acceleration methods, such as DyDiT [1].

[1] Dynamic Diffusion Transformer, 2025 ICLR

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
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes DiffSparse, a learned token-sparsity allocation framework for accelerating diffusion transformer models.It addresses inefficiencies in existing token caching methods by automating layer-wise sparsity allocation using a differentiable cost predictor and dynamic programming solver.Experiments demonstrate moderate acceleration with limited quality degradation across several diffusion transformer backbones.

### Strengths
* The motivation behind the proposed method is well articulated. 

* Validates across diverse architectures (DiT, PixArt, FLUX, Wan2.1) and tasks (text-to-image/video, class-conditional), demonstrating robustness.

### Weaknesses
* The training process is relatively complex, requiring a bespoke two-stage strategy and 4-10 hours of training on 8 high-end GPUs.

* Results on higher resolution (512×512) are based on one configuration and show only marginal benefit,Table 4 shows the proposed method does not outperform all baselines in FID.

### Questions
1.How sensitive is performance to δ and |S|? The limited ablations do not paint a full picture of robustness.

2.Could you include comparisons with state-of-the-art training-free caching approaches such as DiCache and DeepCache.

3.Since the Token Selector computes attention-based ranking signals and dynamically applies binary masks at each step, could the authors quantify the corresponding inference-time overhead and its impact on real wall-clock latency?

4.Given the training cost (4–10 GPU hours with full teacher inference), is the amortized gain meaningful vs. training-free caching or distillation approaches?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
DiffSparse is a novel framework for accelerating Diffusion Transformer models through learnable token-level sparsity optimization. Unlike existing token caching methods that require manual sparsity configuration and depend on predefined full-step computations, DiffSparse automatically learns optimal layer-wise sparsity allocation in an end-to-end manner while eliminating the need for complete forward passes at certain steps.

### Strengths
The method comprises three key components: (1) a Token Selector that computes importance scores based on self-attention influence, cross-attention entropy, cache reuse frequency, and spatial distribution to determine which tokens should be recomputed versus reused from cache; (2) a Learnable Sparsity Cost Predictor with trainable parameters that outputs a normalized cost matrix quantifying the quality impact of different sparsity rates for each layer at each timestep; and (3) a Dynamic Programming Solver that finds the optimal sparsity configuration across all layers and timesteps by minimizing cumulative cost while satisfying a global sparsity constraint , with gradients flowing through discrete decisions via Straight-Through Estimation.

### Weaknesses
1. Symbol M is overloaded with two conflicting meanings within the same section.  First, M (introduced in Section 3.2, "Learnable Sparsity Cost Predictor") represents a binary mask matrix for token selection at each layer and timestep,  with dimension equal to the number of tokens N. Second, M appears as the  summation upper bound in Equation 5, representing the count of importance  criteria combined in the token ranking score. These two usages denote entirely  different mathematical objects—a vector versus a scalar integer—creating  potential confusion for readers.
2.Compared to "training-free" competing methods (such as ToCa), this approach  requires additional training but does not demonstrate a clear advantage in  acceleration performance.

### Questions
1. Does the method require retraining for each new model, new resolution, or  new sampling step count? Is this training overhead worthwhile?
2.While Table 5 demonstrates that the Attention strategy outperforms other  strategies overall, it fails to analyze the individual contributions of 
the four components (self-attention influence, cross-attention focus, cache-reuse frequency, etc)  within the Attention method .

### Soundness
3

### Presentation
2

### Contribution
2
