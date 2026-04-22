# Caffarelli Regularity and Hierarchical Phase Boundaries in Diffusion Models

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 6, 6, 6

## Abstract
Recent studies have shown phase-transition-like behavior in diffusion models, where a small perturbation of the initial Gaussian noise sample can cause an abrupt change in the generated image. The underlying mechanism of these transitions, however, remains theoretically underexplored. In this work, we investigate this phenomenon through the lens of the pullback metric on the latent space induced by the perceptual similarity between generated images. We observe a hierarchical emergence of phase boundaries: coarse boundaries appear in the early denoising steps, while finer boundaries progressively emerge within these regions as the denoising process advances. Moreover, we observe that diffusion distillation shifts boundary formation towards earlier denoising steps and reduces final complexity by decreasing the number of sharp boundaries. To provide a theoretical foundation, we follow the JKO scheme and approximate the reverse diffusion dynamics by a discrete-time sequence of quadratic-cost optimal transport maps between successive noisy marginals. We show that mode splitting forces the diffusion generative map to develop large Lipschitz constant. Using Caffarelli’s regularity theory, we argue that these high-Lipschitz regions form contiguous sets, driven by the disjoint support of the real data distribution and giving rise to phase boundaries. We further note that the proposed theoretical framework does not depend on models' design, but describes the general properties of unimodal-to-multimodal diffusion mappings. This leads to an important practical implication: non-Lipschitzness of generative mapping is necessary for good mode coverage.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper attempts to formalize the phase transition affect in diffusion model, where at certain reverse diffusion timestep (noisy level), and certain noisy images, feature abruptly appear. The author borrow some interesting theory from physics. And use these idea to create an visualization tool to visualize where these phase transition occur during the diffusion process.

### Strengths
I think it's amazing the author is able to track down and visualize phase transition. I'm not really familiar with this literature, but I think this idea of phase transition during reverse diffusion process is only vaguely defined (suddenly an object appear in the reverse diffusion process). The slice latent space + pull back metric and jacobian visualization seems novel.
The author seems to find a deep connection between this ML phenomenon to some theory in physics.  
Unfortunately I'm not similar with the theory that they author uses. So I cannot judge how relevant they are.

### Weaknesses
From a practitioner point of view, it seems less clear to me how the theory connects to practice. 
The author did not write the practical application part clear. What is ($\alpha$, $\beta$), or $\log(\det G)$ has to do with mode-splitting times, and how does that predict the steps for editing images, and how does that depends on the images and edit prompt.

### Questions
At high level, how is the theory section related to your experiment, specifically, parametrize slice of latent space, clip pull back metric and determinant of Jacobian measure? How are these related to JKO, Caffarelli’s regularity, etc... 

Could you explain the image editing section a little bit more? Staring with a clean image, to edit, you said go back to a specific intermediate time step $x_{t_k}$, meaning you add specific amount of noise. At this point, the noisy image $x_{t_k}$ has different ($\alpha$, $\beta$) and $\log(\det G)$. Could you explain the intuition on how does each parameter, time step $t_k$, ($\alpha$, $\beta$) and $log(\det G)$ affect image editing?

I'm not confident to review this paper but I'm happy to change my score if I have a better understanding of the paper.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors study the emergence of phase-transition-like behavior in diffusion models: small perturbations in latent noise can cause abrupt changes in generated outputs. They propose a theoretical framework based on optimal transport theory (JKO scheme) and Caffarelli regularity, by linking high-Lipschitz regions of the generative map to “phase boundaries” in latent space. The authors support the theory with empirical analyses using pullback metrics derived from perceptual similarity (e.g., CLIP embeddings), showing hierarchical phase boundary formation and its modification under LoRA, diffusion distillation, and classifier-free guidance. Their study suggests that such singularities structure the latent space in a hierarchical, ultrametric fashion, impacting image editing and generative smoothness.

### Strengths
1. The authors integrate Caffarelli regularity with the JKO approximation, providing a rigorous mathematical framework for understanding discontinuities in diffusion mappings
2. The authors provide empirical validations via visualizations on Wan 2.1 and Stable Diffusion 1.5
3. The theory has practical intuition for diffusion inversion, editing difficulty, and LoRA effects

### Weaknesses
1. Although the paper visualizes phase boundaries via heatmaps of $\sqrt{det(G)}$ and feature entropy, it does not provide quantitative metrics that relate the sharpness or density of these boundaries to concrete model behaviors such as FID, reconstruction error, or editing success rate. This limits the paper's practical insights.
2. It is unclear whether assumption 1 can fully correspond to empirical diffusion behavior, and explanations or supports of the assumption will make the paper sound.

### Questions
See weakness.

### Soundness
3

### Presentation
2

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
This work proposes a theoretically grounded way to understand where and how the phase boundaries arise in the latent space of diffusion models. The experiments are performed to understand how phase boundaries evolve. The impact of CFG and LoRA on phase boundaries is studied. Further visual experiments are validatied on 2D latent slices of images.

### Strengths
* Interesting findings that both classifier-free guidance (CFG) and LoRA increase the average geometric sensitivity of the slice
* Highlights how distilled models, CGF, LoRA degrade mode coverage
* The study is based on Caffarelli’s theory and combine it with the Jordan Kinderlehrer-Otto (JKO) scheme for the Fokker-Planck equation.
* Figure 2 nicely illustrates the idea of cut-trick and the problems with existing models.

### Weaknesses
I had struggle understanding how this analysis helps develop better generative models or how it can improve existing diffusion models.

### Questions
The work is mainly theoretical and is beyond my mathematical skills. I see the problems raised by the paper and how they propose a nice theoretically sound way to address these issues. I am somehow left wondering what is this study be useful for? How can we design a model that can build on top of the theoretical guarantees promised in this work? 
I believe it is very important to answer these questions, at least, somewhat partially. This would greatly help adoption of the ideas in practice.

### Soundness
3

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
2

### Summary
This paper explains why diffusion models exhibit phase boundaries in their latent spaces, where nearby latent points can produce drastically different outputs. The authors analyze the reverse probability flow of diffusion models through the lens of JKO (Jordan-Kinderlehrer-Otto) gradient flow and show that the generative process can be understood as a sequence of optimal transport maps. Using Caffarelli regularity theory, they prove that when the data distribution is multimodal, the generative map must develop non-smooth regions (singular sets) separating different modes. These regions manifest as sharp ridges in the pullback metric computed from perceptual feature embeddings such as CLIP. Experiments on Stable Diffusion and Wan 2.1 confirm that these boundaries emerge hierarchically along the reverse diffusion trajectory. The paper further shows that LoRA fine-tuning and distillation shift and simplify these boundaries, explaining the observed fidelity&diversity trade off in practical text2image generation.

### Strengths
1. Applying the JKO scheme and Caffarelli regularity theory to the analysis of phase boundaries in diffusion models is a new and theoretically sound contribution.
2. The paper connects the theoretical analysis and the experimental results in a clear way, providing a unified explanation of how phase boundaries form.
3. The robustness analysis using different feature extractors supports that the observed phase boundaries are not dependent on a particular embedding choice.

### Weaknesses
1. The cut trick is central to the paper, but its methodological justification is not sufficiently explained. It is unclear whether this operation preserves the essential structure of the phase boundaries or introduces artificial artifacts. A sensitivity analysis of the cut threshold and a comparison with alternative approaches would be necessary.
2. The claim regarding an ultrametric structure is supported primarily by 1D experiments, and there is no clear evidence that the same structure holds in high-dimensional image spaces. It is likely that the behavior may differ in higher dimensions.
3. The paper does not provide quantitative metrics for the image editing experiments. Relying solely on qualitative figures makes the argument less convincing.

### Questions
How is OT implemented in practice?
How was the cut-trick parameter \eta selected?
What is the finite difference step size used in the Jacobian estimation?
What are the LoRA rank and α values?
What sampler and hyperparameters were used?

### Soundness
3

### Presentation
3

### Contribution
3
