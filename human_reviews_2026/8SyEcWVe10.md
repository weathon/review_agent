# LVTINO: LAtent Video consisTency INverse sOlver for High Definition Video Restoration

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Computational imaging methods increasingly rely on powerful generative diffusion models to tackle challenging image restoration tasks. In particular, state-of-the-art zero-shot image inverse solvers leverage distilled text-to-image latent diffusion models (LDMs) to achieve unprecedented accuracy and perceptual quality with high computational efficiency. However, extending these advances to high-definition video restoration remains a significant challenge, due to the need to recover fine spatial detail while capturing subtle temporal dependencies. Consequently, methods that naively apply image-based LDM priors on a frame-by-frame basis often result in temporally inconsistent reconstructions. We address this challenge by leveraging recent advances in Video Consistency Models (VCMs), which distill video latent diffusion models into fast generators that explicitly capture temporal causality. Building on this foundation, we propose LVTINO, the first zero-shot or plug-and-play inverse solver for high definition video restoration with priors encoded by VCMs. Our conditioning mechanism bypasses the need for automatic differentiation and achieves state-of-the-art video reconstruction quality with only a few neural function evaluations, while ensuring strong measurement consistency and smooth temporal transitions across frames. Extensive experiments on a diverse set of video inverse problems show significant perceptual improvements over current state-of-the-art methods that apply image LDMs frame by frame, establishing a new benchmark in both reconstruction fidelity and computational efficiency. The
code is available on https://github.com/LATINO-PRO/LVTINO.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper produces LVTINO, a video reconstruction model, based on Latino. It "handcrafts" a prior, using state of the art generative models. This special mixture of experts prior has the consequence that it is capturing the video as a whole object, instead of operating on a frame to frame basis.  Furthermore, relying only on "denoising" operations, this algorithm does not need to backprop through any ODE or even network evaluation. The resulting algorithm is applied to video datasets, and compared against two other models.

### Strengths
I find the idea very innovative. Indeed, training a full usual generative model, on a video space is prohibitively expensive. So defining a mixture of experts prior (based on generative models), one for "smoothness", one for the temporal relations, and one for frame consistency. The evaluation is good, it is nice that the authors performed a full grid search for finding suitable hyperparameters, and also showed that others work too, solidifying the trust in the method. Further, I commend the authors for providing many examples and an explanation of the latino algorithm. Furthermore, that it is gradient free makes that it is super cheap as opposed to gradient based diffusion restoration algorithms.

### Weaknesses
The downside is that this paper is very densely written. I dont find the intro to diffusion/consistency models helpful i would rather want to paper to introduce the latino framework and then delve into the video model. I dont think the long equation 7 is necessary and the spelling out of the Euler steps, too. Rather focus on more explanation in a simpler setting to make it easier to digest. 

 There is a plethora of hyperparameters. While the grid search is nice, and good that there area other vals I would like to see that this method is not overengineered. If put on a compute (tuning budget), how does it compare to the other models? 

Does the efficiacy of the approach depend a lot on the chosen consistency/distillation strategy?

Why is the hyperparameter between the video regularizer and the image one? I would rather see a tuning tradeoff between the smoothing regularizer and the image one?

### Questions
see weaknesses. Overall I really appreciate the novelty and lightweightness of the idea :) my main issues are making sure the algorithm is robust and understandable.

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
This work proposes a novel framework that enables high-definition video restoration in a plug-and-play manner using Video Consistency Models (VCMs).
Unlike previous methods that directly apply image LDMs to video inverse problems, the proposed approach explicitly captures temporal causality through VCMs. Leveraging the temporal priors encoded in VCMs, the method achieves substantial perceptual improvements over current state-of-the-art techniques.

Specifically, the video inverse problem formulated as $y=Ax+n$ is solved as follows.
The sampling process begins by initializing $x_0$ with the pseudo-inverse reconstruction $A^{\dagger}y$.
The subsequent diffusion sampling proceeds in three stages:

1. VCM sampling: the model first evolves the initial estimate using only the VCM prior for better temporal coherence.

2. Alternating VCM–ICM: the sampling then alternates between the VCM and Image Consistency Model (ICM) priors to progressively refine both temporal and spatial consistency.

3. ICM refinement phase: finally, the process is completed using the ICM prior alone for better spatial details.

Each after ICM sampling, a few iterations of the conjugate gradient (CG) method are performed to enforce data consistency,
while each after VCM sampling, the Adam optimizer is used to align the samples with the measurement constraint $y=Ax+n$.

### Strengths
1. Using Video Consistency Models (VCMs) to solve video inverse problems represents a novel and timely contribution, especially given the current interest on efficiency in high-definition video inverse problem solvers.

2. The idea of alternating between VCM and ICM is conceptually interesting, offering a fresh perspective on how temporal and spatial priors can be jointly leveraged.

3. The authors conduct comprehensive ablation studies, examining different design choices such as whether to enforce measurement consistency using the conjugate gradient or Adam optimizer, and whether to alternate between VCM and ICM priors or rely solely on VCM.
The results empirically demonstrate that alternating between VCM and ICM yields the most effective performance.

4. I really enjoyed reading the submission. The discussion of open problems provides valuable insights and motivates further research, making the paper timely and impactful.

### Weaknesses
1. I understand the empirical evidence showing that alternating VCM and ICM is effective for solving video inverse problems. However, there is no clear explanation as to why alternating between VCM and ICM yields better results than using VCM alone. Providing a clear theoretical or intuitive explanation would make the paper stronger.

2. The authors state that $p_\phi(x|\lambda)$, representing the total variation (TV) prior, helps prevent the ICM from introducing flickering artifacts. If that is the case, the TV regularizer should logically be applied after the ICM step. However, in the current formulation, $p_\phi(x|\lambda)$ is applied after the VCM step. Clarifying this design choice would be beneficial.

3. Since runtime and VRAM usage are also direct measures of a method’s efficiency, the authors should report these metrics for all comparative methods to ensure a fair and comprehensive comparison.

### Questions
1. Please provide the runtime and VRAM usage for all comparative methods to ensure a fair and comprehensive comparison.

2. Please provide a clear explanation addressing Weakness 1 and Weakness 2.

3. Please clarify whether the code will be publicly released, as this would enhance the reproducibility and impact of the work.

### Soundness
4

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
This paper proposes LVTINO, a zero-shot inverse solver for high-definition video restoration. It uses Video Consistency Models (VCMs) as a prior. The method combines a VCM prior (for temporal), an ICM prior (for spatial), and a classical TV regularizer in a unified framework. The method is based on a Langevin posterior sampler (like LATINO) and does not require gradient calculation through the network, making it efficient. The experiments show good results on several video restoration tasks like a combination of temporal super-resolution and deblurring.

### Strengths
1. LVTINO is computationally efficient. A very big advantage is that it does not need to compute gradients in the latent space, which is a heavy problem for methods like DPS (Chung et al., 2023). This is a very good point for practical use.

2. The paper combines the video prior (VCM), the image prior (ICM), and a classical prior (TV3) in a unified and clean way. I think this product-of-experts prior is an elegant and good way to solve this video restoration problem.

### Weaknesses
1. Lack of baseline comparison: This paper compares with only a very small number of baselines (VISION-XL, ADMM-TV). This is much less than other papers in this field. It is not necessary to solve all problems, but maybe the authors should focus on one task, for example Frame Interpolation (Temporal SR), and compare their method with more specialized models for that task. For example, comparing with BiM-VFI (Seo et al., in CVPR 25).

2. Limitied to *Linear* inverse problem: The method is proposed for linear inverse problems ($y = Ax + n$ where $A$ is a matrix). This is a fundamental limitation. It is not clear if it can be used for non-linear problems.

3. Somewhat impractical metric (NFE): The NFE (neural function evaluations) metric is not very practical (it is more like theoretical). For a better comparison, it would be good to add more practical metrics like runtime (seconds) and memory usage (GB) to Table 1.

4. Hyperparameter sensitivity: Table 2 shows that the optimal hyperparameters are very different for each problem (A, B, C). This suggests the method might be sensitive to these parameters. Also, the paper has not enough ablation study on how to choose these different hyperparameters (e.g., the lambdas for TV, eta, delta).

5. Missing Error Maps: The qualitative results (Figures 4, 5, 6) are good, but it would be much easier for readers to understand how well the model is performing if the authors provide error maps (e.g., Difference from GT).

### Questions
1. In Equation (2), the coefficient for the score term is $\beta_t$. But in Equation (3) (the ODE form), this coefficient changes to $\beta_t / 2$. Could you explain why there is a such change?

2. In Equation (4), the term $\tilde{x}_s$ is used in the integral, which is not defined (I can't find it). What is this variable?

3. On L175, the superscript (k) is used. Is superscript different from other subscripts k? Or is it just a typo?

4. On L213, the paper introduces a TV3 regularizer. Is this 3D Total Variation a widely used regularizer? It would be great if you can provide a reference for this.

### Soundness
2

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
The paper proposed LATINO, a plug-and-play inverse solver for high definition video restoration with latent consistency model (LCM) priors. In particular, LATINO leveraged a video consistency model to capture global dependencies and causalities, and a image consistency model to recover local spatial detail and enhance perceptual quality on each frame. The two LCMs are integrated via a Langevin-type algorithm, together with measurement gradient guidance and regularizations. Experiments show superior performances comparing to baseline algorithms.

### Strengths
1. Successfully combined VCM and ICM to generate with both temporal causality and spatial quality. 
2. Developed Langevin-type splitting algorithm to effectively balance measurement consistency with prior data consistency.
3. Demonstrated highly competitive experimental results under low NFEs.

### Weaknesses
1. The paper only shows experiments on a single dataset and limited baselines. Also, videos are short and noise level is low. Would be more convincing if more experiments could be added. 
2. Since Langevin dynamics typically require the score function to be "well-behaved" to converge, fundamental questions could be raised on why LCM scores could lead to good sampling results. Further explanation should be provided to justify the choice of developing a Langevin-type algorithm for such tasks. 
3. There seems to be no ablation studies on \eta, which is a crucial hyper-parameter in balancing the algorithm.
4. LATINO is based on various models and algorithms. It seems to me that introducing all base models dilutes the core motivation and method.

### Questions
1. Is it possible to experiment on more datasets and inverse problems?
2. Can the authors provide more insight of the Langevin dynamics algorithm?
3. Is it possible to carry out an ablation study on \eta?

### Soundness
4

### Presentation
3

### Contribution
3
