# Energy-oriented Diffusion Bridge for Image Restoration with Foundational Diffusion Models

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Diffusion bridge models have shown great promise in image restoration by explicitly connecting clean and degraded image distributions. However, they often rely on complex and high-cost trajectories, which limit both sampling efficiency and final restoration quality. To address this, we propose an Energy-oriented diffusion Bridge (E-Bridge) framework to approximate a set of low-cost manifold geodesic trajectories to boost the performance of the proposed method. We achieve this by designing a novel bridge process that evolves over a shorter time horizon and makes the reverse process start from an entropy-regularized point that mixes the degraded image and Gaussian noise, which theoretically reduces the required trajectory energy. To solve this process efficiently, we draw inspiration from consistency models to learn a single-step mapping function, optimized via a continuous-time consistency objective tailored for our trajectory, so as to analytically map any state on the trajectory to the target image. Notably, the trajectory length in our framework becomes a tunable task-adaptive knob, allowing the model to adaptively balance information preservation against generative power for tasks of varying degradation, such as denoising versus super-resolution. Extensive experiments demonstrate that our E-Bridge achieves state-of-the-art performance across various image restoration tasks while enabling high-quality recovery with a single or fewer sampling steps. Our project page is https://jinnh.github.io/E-Bridge/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper aims to address an important and challenging problem in the field of image restoration: how to achieve extremely high sampling efficiency while maintaining high-quality restoration. The authors propose a novel framework called the **Consistency Geodesic Bridge (CGB)**, whose core idea is to ingeniously integrate three cutting-edge research directions: (1) bridge diffusion models that directly connect the distributions of degraded and clean images; (2) a geometric perspective that views the “geodesic” path on the data manifold as the most efficient restoration trajectory; and (3) consistency models that enable single-step inference without the need for ordinary differential equation (ODE) solvers. The CGB framework claims to construct a low-energy geodesic trajectory and derives a closed-form “CGB solver” that can be trained through a consistency objective function. As a result, it achieves state-of-the-art performance in perceptual quality metrics with a very small number of sampling steps (NFE = 5–10).

### Strengths
1. This paper proposes the CGB framework, which remodifies the generation process based on the idea of minimum energy, demonstrating a certain level of insight.
2. It achieves industry-leading inference efficiency with very few function evaluations, significantly enhancing the practical value of generative tasks.
3. By introducing a single tunable parameter $T_0$, it achieves task adaptivity and maintains a balance between performance and flexibility across different restoration tasks.

### Weaknesses
**Theory:**
1. The core motivation of the paper is to construct a path of minimum energy. The authors state in the appendix that the energy of the geodesic trajectory is 0, so they attempt to construct a geodesic named CGB. However, the construction on the right-hand side of equation (6) is merely a linear interpolation. Although the authors mention that this construction starts from $T_0$, reducing time and energy—thus seemingly consistent with the motivation of minimizing energy—this does not provide a reasonable scheme for constructing a geodesic. Hence, there is a certain conflict between the motivation and the construction.

2. In Proposition 4.1, the authors claim that the CGB trajectory achieves lower energy, providing two reasons: (i) a shorter integration interval; (ii) at $t = T_0$, the denoiser is given a more “in-distribution” input. This reasoning lacks rigor. The first point is trivial: integrating a non-negative function over a shorter interval naturally yields a smaller value, but this does not imply that the path itself is more efficient—a shorter but sharper path may have higher energy than a longer but smoother one. The second point is more of an intuitive conjecture. Theoretically, given the energy formula, the construction in equation (6) and a certain diffusion bridge equation could be quantitatively compared.

**Experimental:**

3. The comparisons are unfair. The paper uses Flux-dev as the base model, which already has strong generative capabilities, while some other methods, such as UNIDB, are based on SD. Since the paper mainly presents a theoretical improvement and claims superiority over diffusion bridges, to prove that the CGB path is optimal, comparisons should be made between models that are as similar as possible but differ in theory.

**Overall:**
I believe the authors essentially constructed a model:
$\mathbf{X}_t = (1-t)\left[\left(1-\left(\frac{t}{T_0}\right)^{\kappa}\right)\mathbf{X}_0 + \left(\frac{t}{T_0}\right)^{\kappa}\mathbf{Y}\right]+ t\mathbf{X}_T,$
which can essentially be viewed as learning a mapping from $X_0$ to the distribution $N((1-T_0)Y,T_0^2I)$, representing a conditional mapping from a high-quality image to a low-quality one plus a certain amount of noise. Moreover, both training and inference are conducted within the range $[0,T_0]$ (note: $X_T$ is independent of $T$ and just a purely noise). Therefore, although the integration length is nominally reduced, theoretically it is not fundamentally different from previous bridge models ending in Gaussian distributions, and in fact it is essentially time varying Ornstein–Uhlenbeck process—it is merely a scaled version. Considering points 1, 2, and 3, this does not demonstrate that such a construction is truly of minimum energy or superior. I recommend that the authors further verify the theoretical optimality of CGB from two aspects:

1. **Theoretically**, directly provide an inequality proof showing that the energy of CGB is lower than that of other bridge models;
2. **Experimentally**, validate the superiority of CGB through experiments using models that are as similar as possible but differ in theoretical design.

Finally, I would like to discuss the fundamental motivation proposed by the authors—that minimizing energy leads to an optimal path. In the paper, the authors state that typical diffusion-bridge models add noise to the image during the reverse process and then denoise it, with the noise level peaking at intermediate steps. They argue that this noise-adding process during inference is unnecessary and thus propose to start generation directly from $T_0 < T$, restoring high-quality images directly from low-quality ones, which they claim yields minimal energy and an optimal path.

However, I believe that the noise-adding process in the reverse procedure serves an important purpose: it helps to destroy certain information (such as degradation information) before reconstructing a high-quality image, thereby achieving better results. If, as the authors suggest, one wishes to directly transform a low-quality image into a high-quality one without the add-noise-then-denoise process, then models like Rectified Flow (RF) could be directly applied, since RF can progressively transform a low-quality image into a high-quality one as described by the authors. Moreover, RF is also an energy-minimizing model in the context of optimal transport. However, according to experimental results, using RF for image restoration does not perform as well as diffusion-bridge models that incorporate stochastic noise. 

If my understanding is incorrect, I welcome the authors to provide clarification.

### Questions
See weekness.

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
4

### Summary
This paper proposed a novel geodesic diffusion bridge framework through constructing a efficient and geodesic trajectory, which effectively avoids redundant re-noising phases in traditional diffusion bridges. To realize one-step mapping on the data manifold, a pretrained denoiser is proposed and the continuous-time consistency objective is adopted to analytically map any state to the target distribution. Five different image restoration experiments demonstrate state-of-the-art performance of CGB while ensuring a single or fewer sampling steps.

### Strengths
1. The proposed concept "geodesic bridge" looks interesting, reasonable and works as effective tools to solve the inefficient, re-noising trajectories used in traditional diffusion bridge models.

2. The main experiments are reasonable, covering five different image restoration tasks (super-resolution, denoising, raindrop removal, low-light image enhancement, underwater image enhancement, and image demoir´ eing) and six metrics (PSNR, LPIPS, FID, NIQE, MUSIQ, NFE). 

3. The paper demonstrates the superior performance across different image restoration tasks with better perceptual realism and achieves a trade-off between efficiency and quality in image restoration.

### Weaknesses
1. The novelty and contribution of CGB solver seem incremental. The solver (Eqn. 9) is pratically the relationship between the noise prediction $\epsilon_\theta$ and data prediction $x_\theta$ (predict $x_0$) in many diffusion models [1] [2] and the distillation training objective (Eqn. 10) is not in new formulation since it is directly adopted from [3].

[1] Lu et al. "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models.", 2022.

[2] Zhou, Linqi, et al. "Denoising diffusion bridge models.", 2023.

[3] Lu et al. "Simplifying, stabilizing and scaling continuous-time consistency models.", 2025.

2. The paper's main contribution of enabling "**direct and single-step** inference" is misleading. The proposed CGB solver is described as a single-step mapping and the training objective is for distillation. However, all experiments are conducted with **5-10 NFEs** inference, and no real single-step results are shown. The experiments are contradicted with the statements of contribution.

3. The motivation for geodesic trajectory and the related Proposition 4.1 seem intriguing but I'm not convinced by Proposition 4.1. Proposition 4.1 claimed CGB trajectory defined on $[0, T_0]$ for $T_0<1$ achieves a lower total energy (Eqn. 4) than standard bridge models operating on [0,1] for two reasons: (1) reduced integration upper bound, (2) a smaller initial control at time $T_0$. Although the integration upper bound is reduced from 1 to $T_0$ which results in the reduced trajectory cost (first term in Eqn. 4), the terminal cost $\gamma/2 ||X_T-Y||^2_2$, second term in Eqn. 4, could not be ignored: the terminal cost is near or even equal to zero in standard bridge models (e.g. DDBMs and UniDB), while, as for CDB, the starting point $X_{T_0}$ of its reverse trajectory is a noise mixture distinct from $Y$, which appears to result in a non-zero and potentially substantial terminal cost. Therefore, it's not obvious for the two total costs to demonstrate which is smaller and it's better for the authors to provide a rigorous mathematical proof of Proposition 4.1 instead of only the text explanation. Otherwise, the correctness of Proposition 4.1 remains to doubt.

4. The comparison to baselines seems insufficient. Since the authors included DA-CLIP [4], MaRS [5] (the training-free accelerated-sampling version of DA-CLIP), and UniDB [6] in their main experiments, they should compare UniDB++ [7] with the same NFEs as CGB/MaRS, which is a specific training-free acceleration algorithm for UniDB and achieves better results in lower NFEs, as mentioned in Related Works.

[4] Luo et al. "Controlling Vision-Language Models for Multi-Task Image Restoration.", 2023.

[5] Li et al. "MaRS: A Fast Sampler for Mean Reverting Diffusion based on ODE and SDE Solvers.", 2025.

[6] Zhu et al. "UniDB: A Unified Diffusion Bridge Framework via Stochastic Optimal Control.", 2025.

[7] Pan et al. "UniDB++：Fast Sampling of Unified Diffusion Bridge.", 2025.

5. The ablation study seems also insufficient. As in Inherent Data Geodesic Transition (Eqn. 6), except for $T_0$, another hyperparameter is the curvature parameter $\kappa$, which should also be tuned and tested in ablation study.

### Questions
1. There seems to be some typos:
   + In line 364 $X_0$ should be bold, yes? 
   + Eqn 19 appears some error. 
   + Commas and periods should be added at the end of the equations in Appendix as in main paper. 
   + There are some errors of the best results highlighted in bold in Table 1. 
   + ''to approach the a manifold geodesic transition process'' contains an extra article "a" in Line 206.

2. Some motivation statements lack clarity. There are many kinds of interpolant coefficients as mentioned in Stochastic Interpolants [1], why choosing $1 - (t/T_0)^\kappa$ and $(t/T_0)^\kappa$ in Eqn. 6? Is there any relationships between this kind of coefficients and geodesic trajectories?

[1] Albergo et al. "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.", 2023.

After these clarifications, I would be better able to evaluate the overall contributions and potentially raise my rating.

### Soundness
3

### Presentation
2

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
This paper proposes a Consistency Geodesic Bridge (CGB) framework based on pretrained diffusion models for efficient and high-quality image restoration. The framework constructs low-energy trajectories that are near-geodesic paths, effectively avoiding redundant re-noising. In addition, the proposed consistency solver further improves both the efficiency and quality of image restoration.

### Strengths
1.This paper develops a novel consistency geodesic bridge framework, with rigorous theoretical derivation, demonstrating a solid theoretical foundation.

2.The proposed method demonstrates highly competitive performance across diverse restoration tasks while supporting high-quality few-step sampling.

### Weaknesses
1.While the proposed consistency solver is interesting, its level of novelty appears somewhat limited, as it seems to build upon and combine ideas from existing approaches [1,2] rather than introducing a fundamentally new concept.

2.The experimental evaluation is insufficient. Although the paper claims the method can handle five different tasks, the ablation and qualitative experiments are only presented for super-resolution and denoising.

3.The experimental details are incomplete. The paper does not specify how the baseline models were trained, making it difficult to ensure the fairness of the comparisons.

4.The ablation study only analyzes the consistency solver, without examining the role or impact of the geodesic trajectory.

5.In Table 1, for the "Denoising" and "Demoiréing" tasks, the results for LPIPS and FID show that CGB (NFE=5) performs better than CGB (NFE=10), whereas the opposite trend is observed for the MaRS model. No corresponding explanation for this observation is provided in the text.

6.The paper claims to achieve single-step inference along the geodesic path，however, it does not provide qualitative and quantitative results for the true single-step case (NFE = 1). The best performances reported in Tables 1 and 2 is obtained with NFE = 5 or 10.

7.The paper lacks a more comprehensive efficiency analysis, including metrics such as the number of parameters, computational complexity, and actual inference time.

8.The proposed method performs poorly in the raindrop removal task, however, it excels in all other tasks. The paper did not analyze this contrast.

[1] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In International conference on machine learning, 2023.

[2] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. In International conference on learning representations, 2025.

### Questions
1.The paper points out the limitations of IRBridge [1], but why is there no performance comparison with it in the experiments?

2.Is the strong performance of the proposed model primarily attributed to the powerful Flux-dev [2] model? Would similar results be achieved if this model were replaced with another restoration model or generative model?

3.Regarding the parameters κ and T_{0}, the paper does not specify their values for different tasks. What are the specific hyperparameter settings for each task? Could additional ablation studies be provided to perform a sensitivity analysis?

4.On page 7, last line, it is unclear why “underwater image enhancement” appears here. Please clarify whether it is part of the proposed tasks or a mistake.

5.On page 8, line 411, why is PSNR evaluated in the YCbCr color space? Please provide the PSNR results in the RGB space for comparison.

[1] Hanting Wang, Tao Jin, Wang Lin, Shulei Wang, Hai Huang, Shengpeng Ji, and Zhou Zhao. Irbridge: Solving image restoration bridge with pre-trained generative diffusion models. In International conference on learning representations, 2025a.

[2] Blattmann Andreas, Sauer Axel, Lorenz Dominik, Podell Dustin, Boesel Frederic, Saini Harry,Muller Jonas, Lacey Kyle, Esser Patrick, Rombach Robin, Kulal Sumith, Dockhorn Tim, Levi Yam, and English Zion. Flux, 2024. URL https://blackforestlabs.ai/.

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
The paper proposes the Consistency Geodesic Bridge (CGB) framework for image restoration, aiming to improve efficiency and restoration quality by constructing low-cost manifold geodesic trajectories. The method evolves over a shorter time horizon and starts the reverse process from an entropy-regularized point that mixes the degraded image with Gaussian noise, reducing the required trajectory energy. A pretrained denoiser is used as a dynamic geodesic guidance field, and a single-step mapping function is learned via a continuous-time consistency objective to efficiently map any state on the trajectory to the target image. Experiments show that CGB achieves state-of-the-art performance across multiple image restoration tasks while allowing high-quality recovery with a single or very few sampling steps.

### Strengths
1. The method tackles an interesting problem which is the problem of image restoration.

1. The method achieves competitive results compared to existing methods.

### Weaknesses
1. The method relies on paired data for training, which limits its practicality compared to zero-shot and blind methods.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
