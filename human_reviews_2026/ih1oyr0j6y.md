# Accelerate Diffusion Transformers with Feature Momentum

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Diffusion models have demonstrated outstanding generative capabilities in image and video synthesis. However, their heavy computational burden, particularly due to the sequential denoising process and large model sizes, makes them challenging to meet real-time application demands. In this paper, motivated by the continuity of diffusion models in the feature space, we introduce FeMo, which employs a momentum mechanism to stabilize the dynamics of diffusion models in different timesteps, allowing us to accurately predict the features in the future timesteps based on the historical information. Additionally, we further propose an Adapted-FeMo, which allows for adaptive searching for the optimal coefficient for each generated sample. Extensive experiments demonstrate its effectiveness, e.g., a 4.99$\times$ acceleration on FLUX with 0.86% improvements on image reward.Under the condition of maintaining generation quality, Adapted-FeMo achieves a maximum speedup of 7.10$\times$ on DiT and 6.24$\times$ on FLUX. Our codes are available in the supplementary material and will be released on Github.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a feature momentum mechanism called FeMo, which aims to accelerate diffusion transformers by predicting future step features in the diffusion process using a momentum-inspired adaptive method. Subsequently, Adapted-FeMo is constructed by introducing a momentum coefficient for each sample, further minimizing prediction errors. To verify the effectiveness of FeMo and Adapted-FeMo, the authors conducted extensive comparisons with the latest state-of-the-art baselines in image and text-to-image generation tasks. The results show that while maintaining generation quality, Adapted-FeMo achieves a maximum speedup of 7.10x on the DiT model and a maximum speedup of 6.24x on the FLUX model.

### Strengths
1. The experiments are sufficiently comprehensive with rich metrics. This paper conducts benchmark tests on text-to-image and class-conditional tasks, and provides multiple metrics (such as FLOP, latency, FID, sFID, LPIPS, etc.) as well as quantitative performance of various acceleration mechanisms.
2. The paper conforms to the paradigm of "cache first, then predict" and effectively solves the problem of finite difference approximation oscillation existing in this paradigm.
3. The mathematical derivation of the paper is relatively sound, with clear update equations and error proofs for FeMo.

### Weaknesses
1. The idea of introducing momentum methods into diffusion models for acceleration has been widely proposed and implemented [1-2], which affects the innovativeness of this paper to a certain extent.
2. The task scenarios do not cover other diffusion model tasks such as video generation and image editing, and the acceleration effect of FeMo in dynamic sequences or non-generative tasks remains to be verified.
3. Key parameters (e.g., $\beta$) need to be set empirically with defined ranges, and the adaptive mechanism is subject to manual constraints.

[1] Wang X, Dinh A D, Liu D, et al. Boosting Diffusion Models with an Adaptive Momentum Sampler[C]//IJCAI. 2024: 1416-1424.

[2] Wu Z, Zhou P, Kawaguchi K, et al. Momentum-accelerated Diffusion Process for Faster Training and Sampling[J].

### Questions
1. Tables 1 and 2 in the appendix show the experimental results of FeMo and Adapted-FeMo respectively, but the difference between the two is not significant under certain configurations (e.g., $\mathcal{N}=10, \mathcal{O}=1$ and $\mathcal{N}=3, \mathcal{O}=2$). How do the authors view the issue of Adapted-FeMo's ineffectiveness in such cases?
2. Some previous studies have shown that the sampling trajectories of diffusion models in 3D space are very similar.[1] Is this contradictory to the findings in Figure 2 of this paper?
3. Have the authors explored the maximum acceleration effect of FeMo and Adapted-FeMo?

[1] Defang Chen, Zhenyu Zhou, Can Wang, Chunhua Shen, and Siwei Lyu. On the trajectory regularity of ode-based diffusion sampling. arXiv preprint arXiv:2405.11326, 2024.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes the Feature Momentum (FEMO) algorithm, which leverages a “cache-then-reuse” strategy based on the assumption that features from previous timesteps are similar to those in subsequent ones. The authors incorporate a momentum mechanism that predicts features across different timesteps, which they claim capture the temporal dynamics of diffusion models. Additionally, they propose Adapted-FEMO, an extension that adaptively searches for the optimal momentum coefficient for each generated sample. According to the authors, their approach achieves up to 7.10× speedup on DiT and 6.24× on FLUX.

### Strengths
The paper proposed a momentum mechanism to model the dynamics of diffusion models in different timesteps to predict features of diffusion models.

### Weaknesses
The paper needs to be rewrite to make sure clarity, especially the description of the algorithm. For example,
The sub-section: Feature Caching and predicting for Diffusion model did not provide reference.

### Questions
What does eqn. (3) means
Figure 4: what are (a) and (b) referred to in the caption, and the color used is too light. It is hard to read. 
% in T%N is not an standard operator. What is the definition?
The authors stated that: "T is the full activation step closest to the first feature reuse step. It
typically does not directly equal the total number of timesteps"
Possible mistakes:
1. I believe there is a mistake in eqn. (1). It should be $\bar{\alpha}$, not $\alpha$, in order to have eqn. (2) follows. Equ. (1) is an approximate of the diffusion model. I suggest to use the SDE form for definition. 
2. $\epsilon_t$ or  $\epsilon_{\theta}$

### Soundness
3

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
3

### Summary
This paper proposes a training-free acceleration framework for diffusion transformers, namely Feature Momentum (FEMO) and its adaptive variant Adapted-FEMO. FEMO leverages a momentum mechanism to predict future features from the derivatives of historical representations, thereby skipping redundant computations and significantly accelerating inference. Adapted-FEMO further introduces an adaptive weighting mechanism that dynamically adjusts the momentum coefficient based on the feature trajectory of each sample, improving robustness and generality. Experimental results show that FEMO and Adapted-FEMO achieve up to 7.10× acceleration and 6.24× on DiT and FLUX models while maintaining image quality, demonstrating their effectiveness and applicability in accelerating diffusion model inference.

### Strengths
1. The paper introduces the FEMO framework, which employs a momentum-based prediction mechanism to estimate future features and skip computation steps, achieving substantial inference acceleration.

2. Experiments show that FEMO and Adapted-FEMO can achieve high speedup ratios on FLUX and DiT models with almost no degradation in generation quality, indicating that they have high practical value and robustness.

3. The work provides an error-bound analysis and bias correction mechanism, combining theoretical reasoning with empirical validation to support the proposed approach.

### Weaknesses
1. **Scalability of the paper.** The experiments focus on visual diffusion models, such as DiT and FLUX, but there are relatively few experiments on other types or for different modalities. It is hoped that relevant experiments can be added to demonstrate the model's generality and scalability.

2. **Limited theoretical depth.** Although the paper provides error bound analysis, the discussion of the theoretical performance guarantees of FEMO and Adapted-FEMO in different models and tasks is not in-depth enough, such as the convergence and stability analysis of the momentum mechanism under different models.

3. **Lack of discussion on limitations** The paper does not adequately analyze potential failure cases. For instance, when feature trajectories change abruptly under high-noise or complex conditions, the momentum mechanism may mispredict feature directions, and the adaptive $\beta$ update may become unstable under distribution shifts. A more thorough discussion of these limitations would improve transparency.

### Questions
1. From a theoretical perspective, can the momentum mechanism in FEMO be interpreted as a first-order integral form of an ODE solver? If so, could this connection to numerical integration theory provide stronger mathematical support for the method’s design?

2. Does the $\beta$-update rule in Adapted-FEMO have convergence or stability guarantees? If $\beta$ is iteratively updated, could momentum drift or oscillation occur, and how might that affect robustness? Including a formal analysis or empirical validation would be valuable.

3. Regarding reproducibility: what are the additional computational and memory costs of FEMO? Does the caching process introduce significant overhead? Could the authors provide more details or visualizations on hyperparameter sensitivity and memory consumption?

### Soundness
3

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
5

### Summary
The authors propose FEMO (Feature Momentum), a training-free method for cache-based sampling acceleration in Diffusion Transformers. This work improves upon the "cache-then-forecast" paradigm by introducing a momentum mechanism. Instead of relying on noise-sensitive, high-order derivatives like previous methods , FEMO predicts future features by calculating a stable, weighted trend of historical features, allowing computation steps to be skipped. The method achieves state-of-the-art acceleration compared to existing cache-based methods.

### Strengths
+ The method achieves a large speedups on Diffusion Transformers sampling acceleration compared to existing cache-based methods.
+ FEMO is a plug-and-play accelerator that can be applied to pre-trained models without any retraining or fine-tuning.

### Weaknesses
+ The paper's central claim to novelty is applying "Feature Momentum" for diffusion acceleration. However, the use of momentum in diffusion model acceleration is not new, and the paper fails to cite or discuss the extensive literature on this topic. While the authors may argue that FEMO applies momentum to *feature caching* rather than the *forward/reverse* process, these are deeply related concepts, as both leverage historical information throughout the diffusion process (forward or reverse). This omission prevents a clear differentiation of FEMO's contribution and significantly overstates its originality. For example:
  +  Sampling Acceleration:

  [1] Wizadwongsa, S., et al. "Diffusion Sampling with Momentum for Mitigating Divergence Artifacts." ICLR 2024.

  [2] Daras, G., et al. "Soft Diffusion: Score Matching with General Corruptions." TMLR.

  [3] Wang, X., et al. "Boosting Diffusion Models with an Adaptive Momentum Sampler." IJCAI 2024.

  + Diffusion Process Acceleration:

  [4] Dockhorn, T., et al. "Score-Based Generative Modeling with Critically-Damped Langevin Diffusion." ICLR 2022.

  [5] Wu, Z., et al. "Fast diffusion model." arXiv 2023.

+ The paper's title and claims are overly broad, suggesting a general method to "Accelerate Diffusion Transformers". However, the proposed method is specifically a **cache-based acceleration** technique, and its novelty is primarily within this sub-field. More, the experiments are made exclusively against other feature-caching baselines. The paper does not compare against other major acceleration families, such as fast samplers or model distillation. The claims and title should be refined to more accurately reflect this contribution.

+ The method's mechanism for momentum accumulation requires further justification. It builds a momentum term from a sparse history of computation steps (e.g., <6 steps in a 7.1x speedup setting), which differs significantly from traditional momentum (in optimizers or EMA) that relies on many steps for a stable estimate. The paper would benefit from a theoretical or empirical analysis of the stability and accuracy of this sparse-sampling approach.

+ The "Adapted-FEMO" variant introduces a notable number of hyperparameters (e.g., $\mathcal{N}$, $\mathcal{O}$, initial $\beta$, $\gamma$, and $\beta$-bounds) that require careful tuning, which could present challenges for reproducibility. There also appears to be a disconnect between the theoretical optimum derived for $\beta$ (Eq. 13) and the final implementation, which relies on a complex, heuristic-based adaptive search. It is unclear how this compares to prior momentum-based works [4, 5] that directly use the theoretical optimal coefficients.

+ There appear to be citation errors in the manuscript. For instance, the primary baseline "TaylorSeer" is repeatedly cited, but the corresponding bibliography entry seems to point to an unrelated paper ("Timestep embedding tells: It's time to cache for video diffusion model").

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
2
