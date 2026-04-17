# ReGuidance: Diffusion Steering with Strong Latent Initializations Solves Hard Inverse Problems

- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
In recent years there has been a flurry of activity around using pretrained diffusion models as informed data priors for solving inverse problems, and more generally around steering these models towards certain reward models. Training-free methods like gradient guidance have offered simple, flexible approaches for these tasks, but when the reward is not informative enough, e.g., in inverse problems with highly compressive measurements, these techniques can veer off the data manifold, failing to produce realistic data samples. To address this challenge, we devise a simple algorithm, ReGuidance, that leverages prior methods' solutions as strong initializations and substantially enhancing their realism. Given a candidate solution $x$ produced by a given method, we propose inverting the solution by running the unconditional probability flow ODE in reverse starting from $x$, and then using the resulting latent as an initialization for a simple instantiation of diffusion guidance. 
In toy settings, we provide theoretical justification for why this technique boosts the reward and brings $x$ closer to the data manifold. Empirically, we evaluate our algorithm on difficult image restoration tasks including large box inpainting, heavily downscaled superresolution, and high noise deblurring with both linear and nonlinear blurring operations. We find that, using a wide range of baseline methods as initializations, applying our method results in much stronger samples with better realism and measurement consistency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes a training-free algorithm for diffusion-based inverse problem solver, with the key idea that inverse an initial recovery to latent representation through probability flow ODE, and then adopt the DPS from this initialization. Experiments show the effectiveness.

### Strengths
1. The author provide theoretical analysis of ReGuidance in improving reward and realism in Gaussian mixture toy setting;
2. Experiments show improvements over baselines on ImageNet;

### Weaknesses
1. The proofs are restricted to mixtures of Gaussians and linear inverse problems and far from real-world settings;
2. Only evaluated with ImageNet DDPM, and the evaluation set is quite small (100 samples);
3. There is no ablations on guidance strength, ODE steps, or computation trade-off;
4. The baselines are old, it is suggested to add some new strong baselines;

### Questions
1. It is unclear whether the reverse ODE process is stable or sensitive to score errors in used diffusion models, also, how sensitive are the improvements to the initial reconstruction quality, Does ReGuidance fail if the ODE reverse results is far from the manifold?
2. How does runtime scale with diffusion steps compared to DPS/DAPS?

### Soundness
3

### Presentation
2

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
The paper introduces an algorithm called REGUIDANCE that improves the solution of existing diffusion-based plug-and-play inverse problem solvers. Specifically, given a solution $x$ of an existing diffusion-based inverse problem solving algorithm, REGUIDANCE proposes to run the probability flow ODE in reverse, i.e., starting from the clean image $x$ to generate an intermediate latent $x_T$ at time $T$ and then use this $x_T$ as an initialization to run the DPS method using the ODE formulation, i.e., DPS-ODE. Considering a standard toy case, the paper conducts a theoretical analysis and provides bounds on certain quantities, which were useful in inferring about the realism and measurement consistency of the solution returned by the REGUIDANCE procedure. Quantitative metrics such as LPIPS have been shown to improve when REGUIDANCE is applied on top of existing algorithms.

### Strengths
The proposed procedure, called REGUIDANCE, seems simple yet very effective in improving the solution returned by existing inverse problem solvers.  The approach is also modular, which implies that the procedure can be used on top of any existing solvers, which widens its impact. 

In some toy settings, the claims about the solutions returned by REGUIDANCE are supported by a thorough analysis, which adds validity to the proposed method and indicates a principled nature of the approach.

Most inverse problem solvers can perform well for simple inverse problems, but they do really struggle with “hard inverse problems,” and yet this hasn’t been paid much attention to in the literature. I strongly agree and appreciate the authors’ choice of using “hard datasets” like ImageNet and CIFAR rather than FFHQ, etc., and also “hard inverse problems” such as extreme mask inpainting and extremely downsized super-resolution, etc. Improvements on such hard problems showcase the true effectiveness of REGUIDANCE in practice.

Though minor concerns exist, I find the paper well written, easy to read, and organized quite well. I find the roadmap in the appendix very helpful, along with the informal remarks and the simplified explanations in the main text and appendix. I appreciate the authors’ thoughtfulness in this regard.

### Weaknesses
Though the paper presents an interesting find, I believe it has the following weaknesses regarding the plausibility of conclusions drawn from the theoretical analysis, and about where the actual effectiveness of the method stems from. These concerns further highlight the need for more experiments and ablations (some suggested below), which I believe are critical to address before the paper can be recommended for acceptance.

Q1. Remark 1 (line 329) mentions that REGUIDANCE improves the posterior likelihood. But prior works such as DMAP[1] have shown that the DPS procedure itself encourages MAP solutions, i.e., boosts posterior likelihood. In light of this fact, I believe the true effectiveness of REGUIDANCE arises from the fact that the DPS-ODE component of REGUIDANCE encourages MAP solution by default, i.e., irrespective of initialization. This might also explain why it has to be DPS-ODE and not some other posterior sampling algorithm?  Also, from remark 1, if  REGUIDANCE’s effectiveness is because it improves the posterior likelihood, then can one also expect it to perform well if DPS-ODE is replaced with other MAP solvers? I understand that the authors seemed to be completely unaware of the work DMAP[1], as it hasn’t been cited in the paper. From the above perspective, I find REGUIDANCE less novel than posed in the paper (of course, this doesn’t overshadow the other contributions of the paper). Still, I see REGUIDANCE as yet another improvement of DPS (like DMAP[1]) for MAP estimation. However, the most novel aspect of REGUIDANCE is that this improvement is based on good initializations and no procedural changes to DPS, unlike DMAP[1], which alters the DPS procedure slightly. I would ask the authors to clarify and justify their case if something is different from my understanding above. 

Q2. The paper only considers posterior sampling algorithms such as DDRM, DPS, and DAPS, but fails to consider MAP solvers such as DMPlug[2], MAP-GA[3], ProjDiff[4], etc. Again, from remark 1, if the ultimate reason behind REGUIDANCE’s effectiveness is that it returns high-likelihood sample of the posterior $p(x|y)$, then it should absolutely be (1) thoroughly compared to existing MAP-based methods, such as checking whether REGUIDANCE+DDRM/DPS/DAPS solutions perform comparably to the solutions returned by MAP methods, and (2) if (1) holds, then it should be checked how much improvement REGUIDANCE+MAP solver offers over the vanilla MAP solver solution.


Q3. Concerning the point above, I understand that MAP solvers can be computationally expensive than posterior sampling; however, the REGUIDANCE procedure also seems to be quite expensive as it first needs to run the original posterior sampler, then run PF ODE to generate the latents, and finally run DPS-ODE. However, no mention of the computational efficiency of REGUIDANCE was discussed. Also, no mention of NFEs, the noise schedules, and other hyperparameters for REGUIDANCE and baselines was made in the whole paper. A study on how the performance of REGUIDANCE depends on the hyperparameters of DPS, especially such as $\rho$, and NFEs, noise schedule, etc., is crucial and quite essential for empirical validity in my opinion. 

Q4. Regarding Theorem 1 and Theorem 3, I find it difficult to understand the extent to which the conclusions apply beyond toy cases to the real case of Image inpainting. Especially, Theorem 3 holds if the initial solution $x$ is “sufficiently” close to the mode $z_1$? This is a very unrealistic assumption, for (1) this assumption clearly doesn’t hold in the case of the initial solutions returned by DDRM/DAPS/DPS for image inpainting. (2) If the initial solution itself satisfies measurement and is closer to a mode, i.e., a highly likely sample of the posterior, then REGUIDANCE might further push the solution closer to the mode, but one would expect the returned solution of REGUIDANCE to be perceptually similar to the initial solution, which may result in marginal improvements of LPIPS. I’d recommend an empirical verification of this if the authors would consider MAP solvers, as mentioned in point 2 above. 


Q5. Another crucial aspect of Theorems 1 and 3 concerning why initializing (for DPS-ODE) at a higher $T$ is better:  Theorem 3 bounds say, if $T$ is large, then $x_{T}^{DPS}$ has more realism, (but from Theorem 1 bound, if $T$ is large, $x_{T}^{DPS}$ has low reconstruction error, but it also says that the unobserved pixels remain closer to the initial solution $x$.) this clearly implies that it is heavily based on the fact that $x$ is already very close to a realistic sample. Since $x$ is not already a realistic sample in practice, I find the argument for a higher $T$ quite unconvincing. In my opinion, this aspect, like point 4 above, needs an empirical validation by considering intermediate time initializations (with the same NFEs, however) and not just max $T$ initialization. 


Q6. With intermediate initialization in point 5 above, I strongly recommend an ablation to check if REGUIDANCE can still improve the initial solution if the latent is generated with SDE instead of PF-ODE. I believe this aspect of why only the PF-ODE-generated latent has to be used is not discussed in the paper (i.e., why not SDE-generated latent + DPS-ODE with the same NFEs, but with intermediate time initializations, because if $T$ is large, since the latent becomes random, REGUIDANCE essentially becomes DPS).



[1] Xu et al. Rethinking Diffusion Posterior Sampling: From Conditional Score Estimator to Maximizing a Posterior 
[2] Wang et al. DMPlug: A Plug-in Method for Solving Inverse Problems with Diffusion Models 
[3] Gutha et al. Inverse Problems with Diffusion Models: A MAP Estimation Perspective 
[4] Zhang et al. Unleashing the Denoising Capability of Diffusion Prior for Solving Inverse Problems 
 

Minor weaknesses/clarifications: 

Q7. The notation of $x_t$ can be made more clear. In Sec 2.1, $t$ always goes from 0 to T. The left and right arrows clarify whether we talk about the reverse or the forward process, but in some later parts of the text, such as line 203 (one would think $x_{0}^{DPS}$ to be a clean image) and line 290 (one would think $t=0$ in $v_{t}^{DPS}$ is at the start of the reverse process because of earlier notation in line 203). Although I find the paper very interesting, the notation inconsistencies make it difficult to read.


Q8. I’d appreciate it if the authors could share additional qualitative visualizations of imagenet examples over the large mask image inpainting task. From my own experience, I find that sometimes we can achieve better LPIPS if the reconstructed images have rich texture, but are not necessarily semantically meaningful (This can imply unrealistic samples getting better LPIPS. However, I understand this can also happen due to other reasons, such as diffusion models not being perfect, etc., so I’m not very critical about this.)

### Questions
Please see the weaknesses mentioned above.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a simple yet effective method to enhance diffusion-based inverse problem solving by leveraging strong latent initializations. The core idea is to take an initial reconstruction \(x\) from any baseline method, invert it via the unconditional probability flow ODE to obtain a latent $x_T^*$, and then run a deterministic DPS-ODE from this latent to produce an improved sample $\hat{x}$. Theoretically, the authors prove in Gaussian mixture models that REGUIDANCE boosts both reward (measurement consistency) and realism (data likelihood). Empirically, REGUIDANCE consistently improves state-of-the-art baselines (DDRM, DPS, DAPS) across challenging image restoration tasks (large inpainting, super-resolution, deblurring) on ImageNet and CIFAR-10, measured by LPIPS and CMMD.

### Strengths
- **Theoretical Soundness:** Provides the first rigorous guarantees for DPS in mixture models, explaining both reward and realism improvement.
- **Empirical Effectiveness:** Demonstrates strong, consistent improvements across multiple tasks, datasets, and baselines.
- **Clarity:** Exceptionally well-written and easy to follow.
- **Significance:** Offers a practical, low-cost method to enhance existing diffusion-based solvers, especially for highly compressive inverse problems.

### Weaknesses
- **Theoretical Scope:** The theoretical guarantees are currently limited to Gaussian mixture models. While insightful, extending them to more complex distributions remains future work.
- **Computational Overhead:** REGUIDANCE doubles the inference time (inversion + DPS-ODE), though the absolute cost (≤7 GPU minutes) is reasonable. A more detailed runtime comparison would be helpful.
- **Initialization Dependency:** The method’s performance depends on the quality of the initial reconstruction. While it boosts weak baselines, poor initializations may limit gains.

### Questions
1. The paper shows the space of good latents is disconnected. Could this be exploited to generate diverse solutions, e.g., by sampling multiple latents from a baseline and applying REGUIDANCE?
2. While REGUIDANCE improves sample quality, does it also improve convergence speed or stability compared to running DPS from random initialization?
3. Have you explored adaptive strategies for choosing the guidance strength $\rho$ or time horizon $T$ based on the task or baseline method?

### Soundness
4

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
3

### Summary
This paper introduces ReGuidance, a simple yet effective method for improving diffusion-based inverse problem solvers in highly compressive or weak-reward settings. The key idea is to invert an existing reconstruction into the diffusion latent space via the reverse probability flow ODE, then re-run deterministic diffusion posterior sampling (DPS) initialized at this latent.

### Strengths
- The paper clearly articulates a key weakness of current diffusion guidance techniques in settings with limited measurement information or weak reward signals, and introduces a an approach that systematically addresses this issue. 
- ReGuidance is conceptually simple yet powerful. It can be applied to any pretrained diffusion model or inverse problem solver without retraining, making it broadly useful. 
- The paper is clearly written, with well-organized motivation, method, and theory.

### Weaknesses
- ReGuidance always re-samples from one recovered latent, offering limited posterior diversity. Exploring multiple inverted latents or stochastic variants might provide richer solutions.
- The paper would benefit from a comparison or discussion with D-Flow [1], which similarly optimizes the diffusion starting point to improve reconstruction and control.
- Both the reverse ODE and DPS-ODE stages are deterministic with fixed hyperparameters $\rho$ and $T$. It would be good to have ablations or robustness analysis. 

[1] Ben-Hamu et al. D-Flow: Differentiating through Flows for Controlled Generation. ICML 2024.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3
