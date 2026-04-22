# Contrastive Diffusion Guidance for Spatial Inverse Problems

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
We consider a class of inverse problems characterized by forward operators that are partially specified, non-smooth, and non-differentiable.
Although generative inverse solvers have made significant progress, we find that these forward operators introduce a distinct set of challenges.
As a concrete instance, we consider the problem of reconstructing spatial layouts, such as floorplans, from human movement trajectories, where the underlying path-generation process is inherently non-differentiable and only partially known.
In such problems, direct likelihood-based guidance becomes unstable, since the underlying path-planning process does not provide reliable gradients.
We break-away from existing diffusion-based posterior samplers and reformulate likelihood-based guidance in a smoother embedding space.
This embedding space is learned using a contrastive objective to bring compatible trajectory-floorplan pairs close together while pushing mismatched pairs apart.
We show that this surrogate likelihood score in the embedding space provides a valid approximation to the true likelihood score, making it possible to steer the denoising process towards the posterior.
Across extensive experiments, our model CoGuide produces more consistent reconstructions and is more robust than existing inverse-solvers and guided diffusion.
Beyond spatial mapping, we show that our method can be applied more broadly, suggesting a route toward solving generalized blind inverse problems using diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the spatial inverse problem of reconstructing 2D floorplans from human trajectories. The proposed method, CoGuide, is a diffusion-based posterior sampler that avoids the instability of path-planning forward operators by learning a smooth, shared embedding space for floorplans and trajectories using contrastive learning. The learned contrastive similarity score functions as a stable surrogate for the true likelihood, steering the reverse diffusion process toward a compatible layout. CoGuide was evaluated on reconstructing $64\times64$ floorplan images from synthetic trajectories. Experiments show the method produces more robust and visually consistent floorplans than baselines using differentiable planners (e.g., DPS+Neural A*) and achieves the best overall performance.

### Strengths
- The paper clearly motivates its approach by demonstrating that the core path-planning forward operator is unstable, non-differentiable, and non-smooth.

- The central proposal to learn a smooth likelihood surrogate in a contrastive embedding space is a novel and well-reasoned solution to bypass the problematic forward operator of the studied problem.

- The experiments are thorough, comparing CoGuide against six relevant baselines (including DPS+planners and CFG) and showing superior results in sparse and moderate data regimes.

### Weaknesses
- The method might suffer from a significant "sim-to-real" gap, as the encoders are trained exclusively on synthetic $A^*$ (shortest-path) trajectories and on empty floorplans lacking real-world obstacles like furniture. The authors should discuss how the approach would generalize to more complicated real-world problem settings.

- There is no explanation for why the CFG baseline outperforms CoGuide in dense trajectory settings.

- The method's practical applicability is questionable due to scalability, as it is only demonstrated on small $64\times64$ images, a limitation the authors attribute to the poor scaling of path planners.

### Questions
- The use of an Adam optimizer inside each DDIM guidance step is an unusual modification; what are the theoretical implications of this on the mathematical validity and convergence properties of the reverse diffusion process? Why not use other higher-order ODE solvers? 

- How sensitive is the model to the choice of the synthetic data generator? Would performance degrade significantly if the trained models are directly tested with real-world human trajectories?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces CoGuide, a diffusion based approach to solve the inverse problem of reconstructing a floorplan from a user's movement trajectories. A key challenge of the approach is that the forward operator in a floor plan is non-differentiable and non-linear, which destabilizes standard diffusion-based solvers that rely on a likelihood score. CoGuide's key contribution is to bypass this issue by reformulating the likelihood score in a smooth, learned embedding space. This space is trained using a contrastive loss that pulls compatible floorplan-trajectory pairs together while pushing mismatched pairs apart. This creates a stable surrogate likelihood score that can effectively guide the diffusion process, enabling the model to produce more consistent and robust floor plans than baseline methods.

### Strengths
1. The work is well written and easy to follow. With enough preliminaries, backgrounds and technical details, it easy for readers not familiar with spatial inverse problems to understand the problem setting and the work's contributions.
2. Each component of the proposed approach is well-motivated, including the learning of the embedding space using contrastive learning, adapting and improving the diffusion models for the spatial inverse problems. The choice of these components are also well supported by ablation study results.
3. The quantitative evaluation results of the work demonstrates competitive performance of the proposed approach against various baseline models.

### Weaknesses
My biggest concern of the approach is on its comparison against the classifier free guidance approach. The performance gap between CFG and COGUIDE is not significant enough in either sparse, moderate, or dense setting. Is the performance gap a results of randomness or hyper-parameter selection? There're not enough details on the exact instantiation of the baseline CFG model which is also a diffusion-based model and its difference with the proposed COGUIDE approach. More ablation study focusing on the difference between CFG and COGUIDE might help with answer this question.

### Questions
Floorpans are downsampled to a resolution of 64x64 in the work's experiment setting. Is this a common practice in solving spatial inverse problem setups? Is there any challenge preventing the approach from being applied to higher resolution inputs?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper proposes CoGuide to infer a floorplan from human-walked trajectories. It learns a joint contrastive embedding for trajectory and floorplan pairs and uses the embedding similarity as a likelihood surrogate to guide diffusion sampling. Practical additions include a trajectory–wall intersection penalty and DDIM steps optimized with Adam + cosine LR + early stop. On HouseExpo, CoGuide beats the baselines under sparse–moderate trajectories and is comparable to CFG when trajectories are dense.

### Strengths
Clear derivation linking InfoNCE logits to a likelihood ratio and gradients to the likelihood score; switching to unit-norm embeddings yields an equivalent squared-distance form that is smooth for guidance.
Using Adam within each step, cosine LR ramp + hard gating, and a trajectory–wall intersection penalty are simple, effective, and empirically ablated.
The paper diagnoses instability from 𝐴(⋅), which has large/sensitive Jacobians, as well as non-smooth choices, motivating the embedding approach.

### Weaknesses
Evaluation is entirely synthetic on 64×64 binary maps, which makes the sim2real validity uncertain.

While baselines are diverse, CFG wins in dense trajectories. It would be fair to analyze why and whether CFG + contrastive guidance is helpful. Additionally, consider stronger conditional diffusion variants, such as those with modern UNets or latent-diffusion backbones.

Only Gaussian perturbations to trajectories are tested. Yey, real sensors have biases and drift. A more realistic corruption suite would be convincing.

### Questions
Any results at higher resolution or with grayscale walls/soft obstacles to test robustness beyond binary maps?

How does performance change with real trajectories, e.g., human teleop or logs from indoor SLAM?

Can you combine CoGuide with CFG to close the dense-trajectory gap?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes CoGuide, a diffusion-based method for spatial inverse problems with the example in the paper being room layout estimation from (simulated) human trajectories. The authors formalize this task as an inverse problem, where the trajectories have been generated by a forward process that applies an operator (in this case a path planning process by a human) on a certain room layout.
The goal now is estimate a plausable room layout from a given trajectory. For this, the authors follow the existing framework of Diffusion Posterior Sampling [1], which leverages a trained diffusion model as a data prior and additionally maximizes the likelihood of the observation from the foward process (the trajectory inside the room in this case) during the generation / reverse process, i.e., sampling of the unobserved signal (the room layout). However, for the application to room layout estimation from trajectories, the authors identify the use of differentiable path planner (A*) approximations for the likelihood term as unstable / very sensitive to small sample perturbations.
Therefore, the authors propose a new method CoGuide that trains encoders for trajectories and floor plans to a common embedding space with contrastive losses and approximates the likelihood term via similarities of embeddings. Additionally, the paper introduces an additional intersection penalty term and explores using Adam instead of SGD for the gradient-based guidance during sampling.
Finally, an experimental evaluation includes qualitative and quantitative results for different trajectory sparsity levels as well as an ablation study w.r.t. the intersection penalty and the use of Adam.

References:
- [1] Diffusion Posterior Sampling for General Noisy Inverse Problems. ICLR 2023

### Strengths
- The paper is very well written and easy to follow and understand.
  - The introduction does a great job in formalizing inverse problems, the application to floor plan estimation from human trajectories, and already summarizes the main idea of the paper in an understandable way.
  - Preliminaries cover the important existing works to understand the contributions of this paper.
  - The method section first formulates the problem (in particular why a straightforward use of a differentiable A* approximation in the DPS framework does not work well), also visualizes it in Fig. 1, then describes the approach well and proofs that it is a valid approximation of the true likelihood, and finally visualizes the embedding space in a t-SNE plot to give insight about why their method works intuitively.
- The method is interesting, general, and to my knowledge sound and novel.
  - Making use of InfoNCE linking contrastive learning and density estimation to define a likelihood surrogate via similarity in an embedding space learned contrastively seems elegant and also generally applicable to more problems.
- The qualitative and quantitative evaluation shows strong performance compared to a variety of baselines.
- The ablation study validates the effectiveness of the wall and trajectory intersection penalty as well as the use of Adam for gradient-based guidance.
- The appendix provides further qualitative results, implementation details, and information about used path planning algorithms.

### Weaknesses
- Limited evaluation:
  - Even though the method seems to be general and therefore applicable to other (spatial) inverse problems, the paper is limited to the single task of floor plan estimation from trajectories on a single dataset. The paper could be much stronger, if the authors would provide at least one more application.
  - For quantitative results, the paper only provides IoU and F1 score as metrics. If I am not mistaken, both metrics do not account for multiple possible floor plans given a trajectory. Would it be possible provide distribution-based metrics similar to FID and possibly check validity in form of no wall-trajectory intersections? At least, these would allow for multiple possible solutions instead of just comparing with the single ground truth sample.
  - The authors claim that CoGuide is "more robust than differentiable-planner baselines and guided-diffusion methods" but they only show this robustness in terms of trajectory sparseness, where CoGuide outperforms all baselines for sparse and moderate trajectory densities. Robustness in terms of the level of noise in the measurements is only analyzed for CoGuide (Fig. 4) but not for baselines.
- The paper misses to discuss the floor plan and trajectory representations:
  - I find the representation as grayscale images questionable. To this end, the authors unfortunately do not discuss whether this choice is in line with previous related works and whether there could be any alternatives.
  - The authors already mention that even though both trajectories and floor plans are binary, planners would have to "cope with continuous-valued inputs (gray pixels) during the reverse diffusion process" (line 161). If both are binary, would a discrete diffusion approach be more applicable for example?
- Concern about comparison with baselines and their selection:
  - The quantitative and qualitative results are not completely convincing in comparison with the Classifier-Free Guidance baseline. It outperforms CoGuide for dense trajectories and is pretty much on par for moderate trajectory density (Tab. 1) for the two used metrics.
    - Also, how is the guidance scale tuned? Is it the best performance possible w.r.t. this hyperparameter?
  - Since both the intersection penalty and Adam are shown to be important for strong performance in their ablations (Tab. 2 and Tab. 3), the question arises whether the baselines also make use of these improvements, which are independent of the actual CoGuide method or not.
    - To this end, the intersection penalty is a straightforward way of incorporating manual inductive bias specific to floor plan estimation.
      - If I am not mistaken, it should be possible to use the intersection penalty also in combination with CFG or not? This is important since without the intersection penalty, CFG would mostly outperform CoGuide.
    - The use of Adam for the gradient-based guidance during the reverse process seems like something completely general to me independent of the application or the rest of the CoGuide method. Is this paper really the first one to do this?

- Minor weaknesses:
  - In the method section, it could be beneficial to already show some snippet of failure cases from baselines using differentiable A* approximations for the likelihood computation as evidence for the claims and explanations that these are unstable.
  - The Sec. 3.2 (and the rest of the main paper) does not include any details about the encoder architectures.
  - It could be interesting to show t-SNE plots for a larger part of the dataset as well.
  - The GitHub links directly in text in the paragraph about baselines should be avoided. One could just say that you follow the original configurations or at most have them as footnotes.

### Questions
The authors should discuss the representation of floor plans and trajectories as continuous grayscale images, also w.r.t. existing works. Furthermore, I would suggest to address my concerns regarding fair baseline comparisons w.r.t. intersection penalty, Adam, and guidance scale for the Classifier-Free Guidance baselines and others. Furthermore, additional metrics that account for the existence of multiple possible floor plans matching a trajectory would be interesting as well as a comparison with baselines in terms of robustness to measurement noise.

Possibly out of scope for rebuttal: The paper would be much stronger, if it would show the generalization of CoGuide to other applications than the single, synthetic floor plan estimation from human trajectories.

### Soundness
2

### Presentation
3

### Contribution
3
