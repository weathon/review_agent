# Repulsive Latent Score Distillation for Solving Inverse Problems

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Score Distillation Sampling (SDS) has been pivotal for leveraging pre-trained diffusion models in downstream tasks such as inverse problems, but it faces two major challenges: $(i)$ mode collapse and $(ii)$ latent space inversion, which become more pronounced in high-dimensional data. 
To address mode collapse, we introduce a novel variational framework for posterior sampling. 
Utilizing the Wasserstein gradient flow interpretation of SDS, we propose a multimodal variational approximation with a \emph{repulsion} mechanism that promotes diversity among particles by penalizing pairwise kernel-based similarity. 
This repulsion acts as a simple regularizer, encouraging a more diverse set of solutions. 
To mitigate latent space ambiguity, we extend this framework with an \emph{augmented} variational distribution that disentangles the latent and data. 
This repulsive augmented formulation balances computational efficiency, quality, and diversity. 
Extensive experiments on linear and nonlinear inverse tasks with high-resolution images ($512 \times 512$) using pre-trained Stable Diffusion models demonstrate the effectiveness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes (Repulsive Latent Score Distillation) RLSD, a new method leveraging pre-trained diffusion models for inverse problems. RLSD uses a repulsive term in its variational sampling process, which aims to address mode collapse by promoting diversity. Experimental results suggest that RLSD may offer improvements in both quality and diversity for tasks like image inpainting.

### Strengths
The RLSD approach is novel, with a theoretically sound foundation in Wasserstein gradient flow and variational sampling. Additionally the repulsion term does appear to promote diversity.

### Weaknesses
While comparisons with a few models are presented, the paper does not address how RLSD performs against the broader set of models for inverse problems using diffusion prior methods. For example how about comparisons with DDNM, Pi-GDM, FPS, FPS-SMC, ect.? Without these comparisons it makes it difficult to gauge just how much of an improvement this approach is compared to similar approaches.

### Questions
How sensitive is RLSD to the initialization of particle positions? 

In cases where diversity is essential, what strategies do you suggest to improve convergence to distinct modes?

Can RLSD achieve similar performance when the number of particles is reduced? If so, what modifications, if any, would be required?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focus on solving two challenges in inverse problem solving with latent score distillation: mode collapse and latent space inversion. To solve mode collapse, the authors add a repulsive regularization loss to push different samples away from each other. To improve the latent space inversion, the authors define an augmented joint distribution $p(z_0, x_0 | y)$ and propose a two-step optimization algorithm. The authors demonstrate their model on FFHQ 512 dataset using the inpainting, phase retrieval and HDR tasks.

### Strengths
The paper presents a clear and well-structured introduction to the concepts of repulsive regularization and augmented variational distribution. The motivation behind these methods is well-founded, and the theoretical derivation is both detailed and easy to understand.

### Weaknesses
My major concern is that the effectiveness of the two proposed techniques in this paper does not seem to be fully supported by the experiments. The proposed techniques do not consistently yield good results under all conditions. In more detail:

1. Table 2,3,4 show that there is no single setting that can consistently perform the best (or almost the best) across all the three tasks in this paper. For example, NonRepuls-RLSD works the best in HDR, but it performs badly  (second-to-worst) in Phase retrieval. NonAug-RLSD shows best results in Phase-retrieval, but it performs bad in HDR and inpainting. And the full RLSD model has never achieved the best results across the three tasks.

2. While directly enforcing dissimilarity between particles can intuitively enhance their diversity, pushing particles apart may also drive them away from the true data manifold and affect the condition $y = f(x) + v$. Therefore, the strength of this term should be highly influenced  the landscape of the underlying distribution—specifically, whether it is indeed highly multimodal. This likely explains why incorporating repulsive regularization does not always yield improved results.

3. The two proposed techniques introduce several new hyperparameters that require tuning (for example, $N, \gamma, \rho, l$). While some quantitative results are presented in D7, a comprehensive qualitative ablation study examining the sensitivity of performance to different parameter selections might be needed.

### Questions
1. In the Sampling setup, the proposed method employs Stable Diffusion v2.1. Do the baseline methods (PLSD, latent RED-Diff, latent DPS) also use Stable Diffusion v2.1? It appears that in the original paper, PLSD used Stable Diffusion v1.5. Additionally, it may not be fair to directly compare methods using Stable Diffusion as a prior with those using pixel-level diffusion models as a prior, as the pixel-level diffusion models are smaller in size and trained on less data.

2. In the RLSD, N=4 particles are generated per noisy measurement. How are these four different results used to calculate the metrics reported in Tables 2, 3, and 4? Are the metrics calculated by selecting the best result among the four, or are they averaged across all four results?

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
3

### Summary
This paper presents Repulsive Latent Score Distillation (RLSD), a method that addresses the challenges of mode collapse and latent space inversion in Score Distillation Sampling (SDS) for high-dimensional data. RLSD introduces a repulsion mechanism to promote diversity among solutions and an augmented variational distribution to disentangle latent and data spaces. Inspired by the Wasserstein gradient flow, RLSD balances computational efficiency, quality, and diversity in inverse problems like inpainting phase retrieval, and HDR.

### Strengths
I appreciate the overall strong contribution of this paper. It is well-written and easy to follow. The proposed method introduces a repulsion mechanism to enhance diversity among solutions and an augmented variational distribution to separate latent and data spaces. Experiments on both linear and nonlinear inverse problems demonstrate the method’s effectiveness.

### Weaknesses
1. A recent baseline published in ICLR 2024 [1], which also uses a latent-diffusion-based approach for inverse problem-solving, is missing. 

2. I found the coverage of linear inverse problems to be limited. Previous works like DPS, PLSD and RED-DIFF typically include four types of linear inverse problems—such as inpainting, super-resolution, Gaussian deblurring, and motion deblurring—which would strengthen the analysis.


[1] Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency. Bowen Song, Soo Min Kwon, Zecheng Zhang, Xinyu Hu, Qing Qu, Liyue Shen. ICLR 2024.

### Questions
1. A recent baseline published in ICLR 2024 [1], which also uses a latent-diffusion-based approach for inverse problem-solving, is missing. Can you clarify the advantages of your method over this baseline? I suggest that a direct comparison would be beneficial.

2. The variety of linear inverse problems in the main text is limted. I reviewed the appendix for additional experiments on other tasks. Why not include these results in the main text? A fair and complete comparison with PSLD, Latent RED-Diff, RED-Diff, DPS, and [1] across various linear inverse problems would strengthen the analysis.

3. Since Algorithm 1 involves two optimizer steps, could you clarify the computational efficiency of the proposed method?


[1] Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency. Bowen Song, Soo Min Kwon, Zecheng Zhang, Xinyu Hu, Qing Qu, Liyue Shen. ICLR 2024.

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
The paper introduces Repulsive Latent Score Distillation (RLSD), a novel variational posterior sampling framework designed to address key challenges faced by Score Distillation Sampling (SDS) in high-dimensional data, specifically mode collapse and latent space inversion. By leveraging an interactive particle-based variational approach with a repulsion mechanism, RLSD encourages diversity among solutions through a kernel-based regularization strategy. Additionally, the framework incorporates an augmented variational distribution that decouples the latent space from the data space, optimizing computational efficiency while preserving detail and quality. The authors validate RLSD through extensive experiments on high-resolution (512 × 512) inverse tasks, showcasing improvements in linear inverse problems like inpainting, deblurring and nonlinear inverse problems like phase retrieval using Stable Diffusion models.

### Strengths
1. In Table 1, the authors compared RLSD with all the relevant prior works in a very clear way, emphasizing the scalability over dimensions and linear / nonlinear inverse problems as well as the avoidance of score Jacobian computations. 
2. The introduction of a repulsion regularization mechanism that leverages kernel-based similarity to prevent particles from collapsing into a single mode is a step forward. This approach promotes exploration of diverse solutions, a common challenge in high-dimensional generative tasks.
3. The presentation of this paper is good, the reference is relatively complete. 
4. The paper provides robust experimental results across different types of inverse problems, including linear and nonlinear tasks. The diversity and quality trade-off demonstrated by RLSD is well-supported by quantitative metrics and visual comparisons.

### Weaknesses
1. While the proposed approach is well-executed, its novelty compared to existing methods could be more explicitly justified. I think using repulsive forces for diversity and variational formulations for latent space decoupling are not truly unique and novel, compared the existing framework of variational formulations. 
2. For the experiment section, is it possible for you to add more discussion or results on the performance of RLSD with larger-scale datasets and higher-dimensional spaces to clarify practical usage limits?
3. For the related works section, can you include a part (or put in appendix) with the comparison to other variational samplers as well as the difference between their pseudo-codes?

### Questions
Please refer to the "weaknesses" section for questions.

### Soundness
3

### Presentation
3

### Contribution
3
