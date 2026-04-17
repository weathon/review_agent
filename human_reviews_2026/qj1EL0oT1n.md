# Inference-Time Search using Side Information for Diffusion-based Image Reconstruction

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Diffusion models have emerged as powerful priors for solving inverse problems.
However, existing approaches typically overlook side information that could significantly improve reconstruction quality, especially in severely ill-posed settings.
In this work, we propose a novel inference-time search algorithm that guides the
sampling process using the side information in a manner that balances exploration
and exploitation. This enables more accurate and reliable reconstructions, providing an alternative to the gradient-based guidance that is prone to reward-hacking
artifacts. Our approach can be seamlessly integrated into a wide range of existing
diffusion-based image reconstruction pipelines. Through extensive experiments on
a number of inverse problems, such as box inpainting, super-resolution, and various deblurring tasks including motion, Gaussian, nonlinear, and blind deblurring,
we show that our approach consistently improves the qualitative and quantitative
performance of diffusion-based image reconstruction algorithms. We also show
the superior performance of our approach with respect to other baselines, including
reward gradient-based guidance algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposed a method to take advantage of side information in solving ill-posed inverse problems with diffusion prior. Side information is incorporated by a reward function and is integrated in a plug-and-play manner into any inverse problem solvers using pre-trained diffusion model, such as DPS and DAPS. Instead of using reward gradient guidance, the paper adopts inference-time search techniques to accommodate general reward functions. Extensive experiments show superior performance comparing to baselines across various inverse problems on a number of base models.

### Strengths
1. Novelty: The paper is the first to incorporate side information for diffusion inverse problems.
2. Flexibility: The proposed method integrates smoothly into a wide range of diffusion inverse problem solvers, and is compatible with general reward functions.
3. Effectiveness: Experimental results suggest that the method outperforms baselines under various scenarios.

### Weaknesses
1. Although experiments show that the proposed method significantly outperforms baseline methods without side information, it is not clear how side information quality could affect the reconstruction.
2. Since highly informative side information is provided, the authors should consider testing on "difficult" scenarios such as high measurement noise and severe ill-posedness. 
3. The method involves various approximations such that theoretical insights are intractable.

### Questions
1. Is it possible to compare reconstruction quality with side information with different quality? For example, in a fixed inverse problem, it could be interesting to see the difference between text "golden retriever sitting on a snowy frozen lake, facing forward", "golden retriever sitting on a snowy frozen lake", "golden retriever sitting", and "dog". It could be also interesting to see how a blurry image (or a side view when the ground truth is a front view) side information could affect the reconstruction.
2. Can the authors apply the algorithm on more challenging inverse problems as mentioned in the second point of Weaknesses?
3. Why does DPS fail completely in the results of Figure 4?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a training-free inference-time search framework that leverages side information to guide diffusion-based image reconstruction. By modeling side information as a reward-tilted prior and applying greedy or recursive fork-join search, the method balances exploration and exploitation without gradient guidance.

### Strengths
- The paper introduces inference-time search to diffusion-based inverse problems with side information, a setting largely unexplored in prior work.
- The proposed reward-tilting formulation allows the use of arbitrary side information (image, text, MRI contrast) without retraining or model modification.

### Weaknesses
- The proposed search algorithms lack formal theoretical guarantees for exploration–exploitation optimality or convergence.
- The method relies on heuristic scheduling and requires manual tuning across tasks.
- The computational trade-offs between particle count, reward evaluation cost, and performance are not systematically discussed.
- The experimental comparisons are not fully consistent across baselines. Different methods are evaluated on different task settings or degradation levels, which weakens the fairness and interpretability of the reported improvements.

### Questions
- How robust is the search procedure when the side information is noisy, partially mismatched, or misleading?
- The reported DPS baseline results appear significantly weaker than those in prior literature. Could the authors clarify whether this is due to different degradation settings, implementation details, or the influence of side-information conditioning?

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
The authors propose to solve inverse problems by using side information in diffusion model based inverse problem samplers. This could be useful for cases where the observed degradation signal itself might not provide useful information for generating high quality reconstructions and thus using side information could help in these scenarios. In addition to the reward gradient guidance, the authors also propose search methods for non-differentiable rewards. Empirical results are illustrated on the FFHQ dataset for different inverse problems like Super-resoution, non-linear deblur etc.

### Strengths
1. The flow of the paper is straightforward to understand although some claims in the main text need more clarification (see weaknesses below)

2. The problem setup seems relevant in the context of diffusion inverse problem solvers and could be useful for a lot of other applications.

### Weaknesses
**Re. Theoretical assumptions:**

1. The authors propose to use the reward model r(x_0, s). However, the right distribution to sample from would be the following tilted distribution p(x_0|s,y) \propto p(y|x_0)p(s|x_0)p(x_0). Do the authors assume that p(x_0|s, y) \approx p(x_0|s) which would imply that the side information completely explains the ground truth observation. This is a very strong assumption and might not hold in practice. Can the authors clarify more on this aspect?

2. What is the intuition behind setting \eta=0 in Eq. 6. This implies that the authors assume E[x_0|x_t, y] \approx E[x_0|x_t] which is again a strong assumption? Is this primarily for computational convenience? I see that the authors have some theoretical results which validate these assumptions to some extent for a small t but the claims are still seem dubious here. I would request the authors to clarify this aspect in more details in the main text.

**Re. Missing related work**:  The authors highlight some work under Reward-gradient guidance in Section 2 and highlight that recent works are typically used for semantic generation tasks rather than inverse problems (line 126). However this claim is incorrect and I think some related work is missing. For instance the idea of using reward maximization with KL regularization for test time inference is not new and has been explored in [1] from the perspective of optimal control and in [2] from the perspective of variational inference. Both works explore these ideas in the context of inverse problems.

[1] Variational Control for Guidance in Diffusion Models - Pandey et al.

[2] Divide-and-conquer posterior sampling for denoising diffusion priors - Janatai et al.

**Re Empirical Comparisons:**

1. Given the large body of work on solving inverse problems, the paper lacks empirical comparisons with state of the art methods which do not require any side information. For instance comparisons are made against DPS which is outdated. This is important since if the method cant outperform solvers which dont require any side information for the same input degradation, its practical utility is very limited.

2. While the reconstructions in Fig. 3 look decent and the differences between the proposed method and baseline DPS look noticeable, I wonder how cherry picked these samples are? This is because in terms of quantitative results, the reconstruction gains (in terms of PSNR and SSIM) and perceptual gains (LPIPS) are marginally better than the baselines. I can see that the main differences are in terms of the FaceSimilarity metric but Im curious why the other metrics are only marginally better.

3. Can the authors report runtime estimates in Table 1 too? This would help in a holistic comparison between different baselines and the proposed method in terms of the improvement in different metrics vs the additional compute required.

### Questions
See weaknesses above

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
4

### Summary
This author proposes a reward-model based method to aid the inverse problem solving with side information with diffusion models such as reference image, text and so on.

### Strengths
1. This reward function to utilize side information is a novel contribution.
2. The injection of side information seems to improve performance

### Weaknesses
1. The side information is explored in prior works such as [1], [2], [3]. Authors should extensively discuss with those methods
2. Even though reward function may not be differentiable, why not using DPO to fine tune the model. How is the fine-tuned performance compared with inference-time search? Is the inference-time searching adding more computational cost.
3. The experiments do not show the advantage of this method significantly. Inverse problem solvers already achieve quite good performance in most common cases. I expect this method should be more effective in challenging scenarios, for example 32x super-resolution, inpainting with heavy noise, or heavy blur etc. I also expect your method largely outperform DPS if implemented correctly. 
4. The code is not available and I cannot validate without the code. 


[1] CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets

[2] Generative Diffusion Prior for Unified Image Restoration and Enhancement

[3] Prompt-tuning Latent Diffusion Models for Inverse Problems

### Questions
Please show the experimental results on hard inverse problems, compare with [1],[2],[3], and at least provide some pseudo-code, and then I will consider improving my rating.

### Soundness
3

### Presentation
2

### Contribution
2
