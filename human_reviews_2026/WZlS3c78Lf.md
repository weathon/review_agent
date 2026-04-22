# EquiReg: Equivariance Regularized Diffusion for Inverse Problems

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Diffusion models represent the state-of-the-art for solving inverse problems such as image restoration tasks. Diffusion-based inverse solvers incorporate a likelihood term to guide prior sampling, generating data consistent with the posterior distribution. However, due to the intractability of the likelihood, most methods rely on isotropic Gaussian approximations, which can push estimates off the data manifold and produce inconsistent, poor reconstructions. We propose Equivariance Regularized (EquiReg) diffusion, a general plug-and-play framework that improves posterior sampling by penalizing those that deviate from the data manifold. EquiReg formalizes manifold-preferential equivariant functions that exhibit low equivariance error for on-manifold samples and high error for off-manifold ones, thereby guiding sampling toward symmetry-preserving regions of the solution space. We highlight that such functions naturally emerge when training non-equivariant models with augmentation or on data with symmetries. EquiReg is particularly effective under reduced sampling and measurement consistency steps, where many methods suffer severe quality degradation. By regularizing trajectories toward the manifold, EquiReg implicitly accelerates convergence and enables high-quality reconstructions. EquiReg consistently improves performance in linear and nonlinear image restoration tasks and solving partial differential equations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces EquiReg, a plug-and-play regularization framework for diffusion-based inverse problem solver by keeping sampling trajectories close to the data manifold. It leverages Manifold-Preferential Equivariant (MPE) functions, whose equivariance error is small for on-manifold and large for off-manifold data, to penalize implausible samples. EquiReg integrates seamlessly with existing diffusion solvers across pixel, latent, and PDE domains. Experiments on diverse restoration and physical modeling tasks show effectiveness.

### Strengths
1. The work connects geometric deep learning (equivariance) with probabilistic sampling (diffusion) in a theoretical way.
2. The work demonstrated improvements across pixel diffusion, latent diffusion, and PDE solvers, showing strong versatility.
3 The definitions of distribution-dependent equivariance and manifold-constrained equivariance are well-motivated.

### Weaknesses
1. No formal analysis shows why equivariance error correlates with manifold distance, and the impact of the EquiReg term on sampling stability and likelihood gradients is mostly qualitative. A detailed study (e.g., trajectory visualization, convergence proofs) would strengthen understanding.
2. The framework’s success hinges on selecting a good MPE function and appropriate symmetry group, which may be non-trivial or domain-specific. The paper provides examples but lacks systematic guidelines.
3. Although the authors mention low overhead, the additional forward passes through MPE networks could be significant for high-resolution diffusion.
4. The paper compares against general diffusion baselines but omits direct comparisons to manifold-preserving or geometry-constrained approaches
5. Figures 2 could be better visualized on how equivariance loss actually affects trajectory correction.

### Questions
1. Can MPE functions be learned jointly with the diffusion model instead of being pre-trained? Would that further improve alignment?
2. Whether EquiReg conflict with guidance-based conditioning (e.g., classifier-free guidance) in diffusion models?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an explicit regularizer, called EquiReg, for solving inverse problems in imaging with diffusion models. The method is framework-agnostic. Because natural images have equivariance, the paper suggests adding the regularizer's gradient when differentiating the likelihood. They apply this to ReSample, DPS, and PSLD and get better LPIPS, FID, and PSNR.

### Strengths
1. The story is not too complex, so the message is clear.

2. It is model-agnostic, so many users can adopt EquiReg for their own framework.

3. It generally improves fidelity metrics like PSNR and LPIPS.

### Weaknesses
This paper might be sacrificing the true strength of diffusion models for inverse problems. The reason diffusion models are used (like in DPS - Diffusion Posterior **Sampling**) is that they are **Samplers**, not mean-estimators. The advantage is that they can sample many different solutions that are all good. This paper has no consideration for this. However, this paper just says EquiReg helps getting good PSNR and SSIM. This is obvious. Equivariance is a somewhat classical regularization method. Of course, if one add any classical regularizer, the PSNR and SSIM will improve. This is a common idea. For example, a paper rejected at ICLR last year (https://openreview.net/forum?id=GQnR7L6SmA) used Total Variation (TV) with ADMM, another classical regularization, and it worked well, but I think that paper shares the same problem with this one: DIVERSITY. Diversity is a core philosophy of using diffusion models for inverse problems. The author of DPS has already pointed out that subtle PSNR improvement is not the thing for this problem (https://x.com/hyungjin_chung/status/1788861058309902633). Without showing a proper consideration for diversity, it is hard to prove that this simple regularizer is truly beneficial.

Minor weaknesses:
- L34: There is a citation error for Charles W Groetsch and CW Groetsch.

- It is better importing figures as pdf, rather than png or jpg.

### Questions
In L89, 97, this paper points out relying on the isotropic Gaussian assumption is one of limitations of prior work. Could you please elaborate why is it so, and how this paper addressed it?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes EquiReg, an equivariance-based regularization method for diffusion inverse problems. By penalizing samples that break learned symmetries, it keeps diffusion trajectories closer to the data manifold and stabilizes posterior sampling.

### Strengths
- Introduces equivariance-based regularization as a proxy for manifold consistency, connecting geometric symmetry with probabilistic sampling in a fresh way. 
- The method is architecture-agnostic and simple to implement that can be directly added to existing diffusion frameworks (DPS, PSLD, SITCOM) without retraining.

### Weaknesses
- The link between low equivariance error and on-manifold behavior is intuitive but not rigorously proven.
- The approach assumes an explicit group $G$ (e.g., rotation, reflection), which may not exist or be meaningful for all tasks. 
- The method relies on pre-trained MPE encoders, but how to systematically obtain or generalize them is not well discussed. 
- The paper frequently refers to an Appendix for details of tasks, proofs, and additional experimental results, but the Appendix is not provided. As a result, several important aspects cannot be verified. This omission limits the paper’s clarity.

### Questions
- See weaknesses
- The statement “MPE can emerge when functions are trained with symmetry-preserving mechanisms such as data augmentation” is somewhat ambiguous. Almost all modern pretrained models are trained with some form of data augmentation, yet clearly not all of them behave as MPEs. How do the authors determine whether a given model qualifies as an MPE?

### Soundness
2

### Presentation
2

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
The paper proposed Equivariance Regularized (EquiReg) diffusion, a plug-and-play framework to solve Bayesian inverse problems with pre-trained diffusion prior. In inference time, equivariance loss is incorporated as reward gradient guidance, penalizing reconstructions that deviate from the data manifold. Experimental results across various tasks and existing inverse problem solver demonstrates the effectiveness of EquiReg.

### Strengths
1. Novelty: The paper proposed a novel reward gradient guidance that leads to on-manifold sampling. Instead of digging deep into the underlying data distribution, the reward discriminates on-manifold samples from off-manifold samples simply using symmetry arising from data itself or the training process. 
2. Effectiveness: Numerical experiments demonstrates that EquiReg loss could be easily incorporated into gradient guidance-based diffusion inverse problem solvers, and achieve better performances.

### Weaknesses
1. Though it is mentioned in the main paper that various details are deferred to the appendix, the appendix is not included in the submission, which substantially affects readability.
2. Theoretical insights of the EquiReg loss is not sufficiently explored. It remains unknown how EquiReg loss could affect generation consistency. The authors mentioned some insights through Wasserstein gradient flow. It will be helpful if it could be further discussed.
3. Though it is natural to use symmetry when data is inherently symmetric, there seems to be little intuition in the paper on how to choose a symmetry group and the corresponding MPE functions in general. Yet the choice could be crucial in sampling quality. 
4. Symmetry group size too small among all experiments. It remains unclear whether a large symmetry group, or even an infinite group such as SO(3), could affect the algorithm.

### Questions
My questions follow from the Weaknesses.
1. Is it possible to solve box inpainting so that reviewers could recover the appendix? Will be happy to see this paper published if it is complete.
2. Is it possible to demonstrate the effectiveness of EquiReg loss through low-dimensional experiments?
3. Can the authors compare different MPE functions and symmetry groups on the same task and same diffusion inverse problem solver?
4. Is it possible to experiment with a symmetry group with large cardinality?

### Soundness
4

### Presentation
2

### Contribution
4
