# Test-Time Iterative Error Correction for Efficient Diffusion Models

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
With the growing demand for high-quality image generation on resource-constrained devices, efficient diffusion models have received increasing attention. However, such models suffer from approximation errors introduced by efficiency techniques, which significantly degrade generation quality. Once deployed, these errors are difficult to correct, as modifying the model is typically infeasible in deployment environments. Through an analysis of error propagation across diffusion timesteps, we reveal that these approximation errors can accumulate exponentially, severely impairing output quality. Motivated by this insight, we propose Iterative Error Correction (IEC), a novel test-time method that mitigates inference-time errors by iteratively refining the model’s output. IEC is theoretically proven to reduce error propagation from exponential to linear growth, without requiring any retraining or architectural changes. IEC can seamlessly integrate into the inference process of existing diffusion models, enabling a flexible trade-off between performance and efficiency. Extensive experiments show that IEC consistently improves generation quality across various datasets, efficiency techniques, and model architectures, establishing it as a practical and generalizable solution for test-time enhancement of efficient diffusion models. The code is available at
https://github.com/zysxmu/IEC.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel inference-time method called Iterative Error Correction (IEC) to enhance the performance of efficient diffusion models that often degrade in quality due to approximations introduced by quantization, feature caching, or other efficiency techniques. The authors introduce IEC, a test-time, plug-and-play refinement procedure that corrects inference-time errors without modifying model parameters. IEC is evaluated on multiple diffusion models (DDPM, LDM, and Stable Diffusion) and efficiency schemes (quantization, feature caching, and hybrid quantization-caching), across datasets such as CIFAR-10, LSUN-Church, LSUN-Bedroom, ImageNet, and MS-COCO.

### Strengths
1. The paper introduces Iterative Error Correction (IEC), a new test-time method for improving diffusion model performance without retraining or architecture changes. This is highly practical because deployed diffusion models—especially quantized or cached versions—are often immutable, and IEC offers a plug-and-play solution that directly operates at inference time.

2. The authors provide a clear mathematical analysis of error accumulation in efficient diffusion models and show that approximation errors grow exponentially across timesteps. They then rigorously prove that IEC reduces this exponential growth to linear error propagation through a fixed-point convergence analysis. This theoretical grounding strengthens the credibility and generality of the proposed method.

3. Since IEC operates purely at test time, it does not require access to the original full-precision model or re-application of the efficiency pipeline. This feature is particularly valuable for real-world scenarios where deployed models are frozen or stored under strict hardware and policy constraints.

### Weaknesses
1. The experiments are limited in CNN-based methods. The effectiveness of the proposed methods are not validated in Transformer-based methods such as DiT and Flux.

2. The theoretical and experimental analysis builds on top of DDIM sampler. I wonder if this error reduction strategy is compatible with other samplers such as DPM-Solver?

3. Some critical implementation details are missing such as the exact version of SD.

### Questions
1. Would this strategy also applicable in other acceleration methods such as adaptive computation[1]? Could you provide some empirical or theoretical analysis for this?

[1] Tang S, Wang Y, Ding C, et al. Adadiff: Accelerating diffusion models through step-wise adaptive computation[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 73-90.

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
3

### Summary
This paper proposes Iterative Error Correction (IEC), a test-time method to improve the quality of efficient diffusion models suffering from approximation errors (e.g., from quantization or feature caching). The authors analyze error propagation in DDIM sampling and claim errors accumulate exponentially. IEC iteratively refines outputs at each timestep using a fixed-point iteration scheme, theoretically proven to reduce error growth from exponential to linear. Experiments on CIFAR-10, LSUN, ImageNet, and MS-COCO demonstrate FID improvements when combining IEC with quantization and caching methods.

### Strengths
1. Clear problem motivation (deployed models can't be easily modified) thus a training free method is needed.
2. The ability to apply IEC to selected timesteps (±1/10, ±1/20) provides efficiency options, though this flexibility is expected from any refinement method
3. IEC shows gains across all tested scenarios, demonstrating broad applicability
4. The paper is generally easy to follow with good visual aids

### Weaknesses
Fundamentally Unfair Comparison:
1. Baseline uses 100 forward passes, while IEC with K=1 applied to all T=100 steps uses 200 forward pass.
2. Missing crucial baseline: what is the FID with T=200 steps without IEC?
3. This is like claiming "Method A is better than Method B" when Method A uses 2× computation

IEC is Multi-Step Refinement essentially
1. Eq. 10 is mathematically equivalent to: repeatedly calling the denoising function with corrected inputs
2. This is essentially running multiple sampling steps per timestep
3. Authors admit in A.2 that "naively adding 100% iterations" achieves FID 3.99, close to IEC's 3.76

### Questions
1. What is the FID when using T=200 sampling steps without IEC, matching the computational budget of T=100 with IEC (K=1)?
2. How does IEC compare to other methods that increase sampling quality with more compute (e.g., DPM-Solver++ with more steps)?
3. Can you provide rigorous proof that errors grow exponentially without IEC? The current analysis only shows they propagate through products of matrices.
4. How does IEC compare to consistency models or other iterative refinement approaches?

### Soundness
2

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
This paper addresses the performance degradation problem in efficient diffusion models, which arises from approximation errors introduced by techniques like quantization and feature caching. It shows that these errors can accumulate exponentially throughout the sampling.

The core contribution is a test-time-scaling-like method called Iterative Error Correction (IEC). It requires no retraining or architectural changes, and operates by introducing an inner-loop fixed-point iteration at each sampling timestep to refine the model's output, which reduces the error accumulation from exponential to linear.

The paper provides both theoretical and empirical analysis showing that IEC can reduce the error accumulation. Experiments on various models (DDPM, LDM, SD), datasets (CIFAR, LSUN, ImageNet), and efficiency techniques (quant, cache, quant+cache) demonstrate that IEC consistently improves generation quality.

### Strengths
-- Problem: It targets the practical, post-deployment scenario for efficient models, which is somewhat under-explored.

-- Idea: It is related to test-time scaling, and potentially provides another new dimension in which we can scale (the inner-loop iteration at each timestep).

-- Analysis: The theoretical analysis is of good quality. It is also appreciated that the authors try to validate their method on multiple models and settings.

### Weaknesses
-- The theoretical framework seems general to not be limited to errors from efficiency techniques.
It is unclear how the definition of $\tilde x_t = x_t + \delta_t$ and $\tilde\epsilon_\theta = \epsilon_\theta + \epsilon_\theta^\delta$ are necessarily linked to quantization or caching.
It seems potentially applicable to any diffusion model, where $x_t$ and $\epsilon_\theta$ correspond to an ideal denoiser (please refer to Figure 1 in EDM).
It would strengthen the paper if the authors could show the method can also improve regular diffusion models (e.g., trained with fewer iterations and thus have imperfections).

--The compute-performance trade-off is not fully explored.
According to my understanding, the total inference compute is determined by at least three factors: (1) DDIM sampling steps (2) hyper-parameters in efficiency methods (3) hyper-parameters in IEC. Each of these can be adjusted to trade more computation for better performance.
While IEC shows improvements when (1) and (2) are fixed, it is unclear whether it provides a better Pareto frontier in the broader context. For example, a real, comprehensive compute-performance plot (please refer to Figure 9 in EDM2 and Figure 10 in DiT) with respect to all these factors would be helpful. The current discussion in A.1 and A.2 is insightful but seems too simple.

--Theoretical and empirical analysis is based only on DDPM/DDIM variants, without showing applicability to modern flow-based models.

--Presentation
Tables: Table 1 and Table 3 take too much space, and their subtables could be merged like Table 2. Additionally, it is unclear why results for certain settings are omitted (e.g., W4A8 in Table 1 for LSUN-Bedrooms, and W8A8 in Table 3). I assume W8A8 and W4A8 are both meaningful in both tables.
Section A.2: some cited data (e.g., 3.99, 8.58) do not appear in Table 5. The overhead and trade-off (claimed in Abstract) analysis is important and should be moved to the main text.
Typo: "ImageNet 256x25" in Table 4.
Citation: most should be in parenthesis using \citep{}.

EDM: Elucidating the Design Space of Diffusion-Based Generative Models, NeurIPS 2022

EDM2: Analyzing and Improving the Training Dynamics of Diffusion Models, CVPR 2024

DiT: Scalable Diffusion Models with Transformers, ICCV 2023

### Questions
Please refer to the weaknesses about generalization and compute-performance trade-off.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the degradation in generation quality that with quantization or feature caching. The authors first analyze how approximation errors propagate across diffusion timesteps. Motivated by this, they propose Iterative Error Correction (IEC), a plug-and-play, test-time refinement that injects a fixed-point iteration at selected timesteps. They provide a convergence argument via Banach's fixed-point theorem and show that IEC reduces error growth from exponential to linear in theory. Empirically, IEC consistently improves metrics across multiple datasets and efficiency techniques, with a tunable compute–quality trade-off.

### Strengths
- IEC is model-agnostic, requires no retraining, and can be dropped into existing inference pipelines. This plug-and-play nature makes it highly practical for real deployments.
- The method is evaluated across several models and multiple datasets. Improvements are consistent, with especially notable gains for aggressive efficiency settings.
- The paper includes ablations over λ, iteration count K, and which timesteps are refined, and a brief comparison to naïvely adding iterations, helping understand the method better.

### Weaknesses
- For Stable Diffusion, IEC is applied only at the first timestep, yielding modest improvements. This raises concerns about the practical benefits at scale if the method cannot be applied more broadly due to compute.
- W4A8 on LSUN-Bedrooms “collapses” without detail. A short analysis of when IEC helps vs. cannot rescue severe degradation would be valuable, potentially guiding users to regimes where IEC is most beneficial.
- Most experiments use T=100 (or 250). Since many efficient systems run at T≤20–50, it would be useful to report results in that low-step regime to demonstrate relevance to contemporary fast samplers.
- Showing ||At + BtJt|| > 1 indicates potential amplification but does not alone establish exponential growth of the end-to-end error. More direct measurement across models/datasets would strengthen the claim.

### Questions
Please check Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
