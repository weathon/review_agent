# Score-based generative modeling through anisotropic SPDEs

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Score-based generative modeling (SBGM) has achieved state-of-the-art performance in image generation, with the quality of generated images highly dependent on the design of the forward (diffusion) process. Among these, models based on stochastic differential equations have proven particularly effective.
 
While traditional methods aim to progressively destroy all image information to enable reconstruction from pure noise, we introduce a novel class of anisotropic stochastic partial differential equations (SPDEs) that preserve the geometric structure of the data throughout the transformation. These SPDEs consist of a drift term that enforces deterministic destruction via structured smoothing, and a diffusion coefficient that enables random destruction through noise injection. Both components are governed by anisotropy coefficients, enabling controlled, direction-dependent information degradation. 

This framework provides the theoretical foundation for a novel anisotropic SBGM. Due to geometry-aware degradation, the data generation process can exploit residual geometric cues, leading to improved fidelity in image reconstruction. We empirically validate this improvement in a proof-of-concept implementation on unconditional image generation, showing that anisotropic diffusion can achieve superior image quality metrics.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes an anisotropic nonlinear diffusion process. The authors claim to allow both the drift and diffusion coefficients to evolve dynamically based on the current state. However, their implementation in this paper appears to focus solely on the drift term. While stochastic differential equations (SDEs) with state-dependent drift and diffusion coefficients are well-established in theory, in my view, incorporating such dynamics into a generative diffusion model requires a rigorous framework to demonstrate that the backward diffusion process can recover the initial prior distribution. However, I could not find a clear explanation  of such framework presented in this paper. It needs a major revision.

### Strengths
This paper proposes an anisotropic nonlinear diffusion process.

### Weaknesses
This paper is poorly written. It does not flow well. It lacks clarity and coherence, making it difficult to follow.  I listed below some examples of specific problems:
1. The authors used non-standard  statistical terms such as "$\mu$-distributed sequence" 
2. P3. Line 128, That authors stated that, during the forward process, "the dynamics
of the transformation realized by the forward process are learned (by a neural network)". This seems like a wrong statement.
3. Lack proper reference:  anisotropic nonlinear diffusion process has been investigate intensively in multiscale image analysis, however, this paper did not discuss the relevant of this topic in their introduction. Here are some references:
 1).  P Perona and J Malik, Scale-space and edge detection using anisotropic diffusion.
 2).  P. Guidotti  Anisotropic Diffusions of Image Processing From Perona-Malik
 3). Y Bao; H. Krim Smart nonlinear diffusion: a probabilistic approach
:4). W Feng, P Qiao, X Xi, and Y Chen,  Image Denoising via Multiscale Nonlinear Diffusion Models
 5). Y You , W Xu, A Tannenbaum, M Kaveh, Behavioral analysis of anisotropic diffusion in image processing

4. Repeating: 
for example P7, Line 361: We now begin by describing specific instances of our framework
                    P7, Line 363: We now describe the specific instances of our framework considered
5, The selection of drift and diffusion coefficients in Eqns (7)-(9): The definition of sigma in eqn. (8) was not used in the new algorithm in section 5.2 and 5.3. More importantly, there is no proof on why the selection of the coefficients will lead to preserving structures in images.

### Questions
1. Is there a timeline problem in the following statement:
P2. Lin 90: The authors stated that:
Rissanen et al. (2023) considered a stochastic heat equation with isotropic noise, which is effectively destroying the data by blurring up to complete dissipation. This is in contrast to earlier approaches that typically destroyeddata into pure noise. Hoogeboom & Salimans (2022) extended this idea by introducing a temporally increasing isotropic noise term, further refining the blurring process over time.

2. Eqns. (4) (5) is not used anywhere, any discussion on this and how it is used in the proposed algorithm?

3.There is a $v$ in Eqn. (8), but there is no $v$ given in Eqn. (6).

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method that employs an anisotropic destruction process, rather than the isotropic noising common in existing diffusion models. Through small-scale experiments, the paper demonstrates performance that is superior or comparable to existing diffusion model families, such as score-based models and flow-matching models.

### Strengths
- The paper challenges the convention in the diffusion model literature that we should use an isotropic diffusion process, and it demonstrates the potential of anisotropic SPDEs through comparisons with existing models.

- It clearly explains the conceptual similarities and differences compared to prior work.

### Weaknesses
- Standard Gaussian noise-based diffusion models already benefit from schedules that rapidly destroy the image, such as the cosine noise schedule in IDDPM or timestep shifting in Stable Diffusion 3, improving both training and inference.
It is questionable whether the proposed method's superior performance could be achieved in existing score-based or flow-matching models simply by applying more advanced noise schedules.

- The "blurring diffusion models" (Hoogeboom & Salimans), which are only briefly mentioned in the related works, are conceptually very similar as they also use both noise and a blurring drift. A more detailed conceptual and experimental comparison against them is necessary.

### Questions
While qualitative comparisons are provided in the appendix, they are limited to CIFAR-10. Can you show qualitative results for ImageNet and LSUN?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a novel framework for Score-Based Generative Modeling that
uses anisotropic Stochastic Partial Differential Equations to govern the
diffusion process. The main goal is to enhance image generation quality by preserving the
geometric structure of data during the forward (destruction) process, a departure from
traditional methods that aim to destroy all image information to pure noise. Their forward
process is modeled as the formal solution to SPDE, where it has two components, namely
drift term which enforces deterministic destruction through structural smoothing, and
diffusion term which enables random destruction through noise injection.

### Strengths
1). This paper introduces a novel framework that keeps some geometrical structural clues when data destruction, helping the resemblance of geometric features in generative sampling.


2). This paper showcases the proposed method through experiments, and it obtains superior results on both qualitative and quantitative comparisons.

3). One of the main strengths is this unifies formulation of SBGM. i.e., providing a common framework for existing SBGMs and a new anisotropic diffusion process.

### Weaknesses
1). I feel even though this paper has the mathematical rigor, it lacks the intuitive and logical building of the proposed method. I suggest authors to add more intuitive explanation that will enable readers to understand the paper much better. Otherwise, the current version is
bit hard to follow and grasp the concepts. If possible, try to add a figure that explaining the concept.


2). The experimental section only compares against methods up to 2023. For completeness, I suggest including comparisons or discussions of more recent works such as “Edge-Preserving Noise for Diffusion Models” (2024), which shares a similar motivation of geometry-aware corruption.

3). The proposed method requires almost 2000 score evolutions. This I feel a major limitation compared to recent works.

4). The authors mainly used \( \ell = 0 \) for the noise process. I would like to know whether introducing a finite \( \ell > 0 \) — i.e., spatially correlated noise could help capture textured patterns or reduce artifacts, or if it would mainly complicate the sampling
process. 

5). In this paper, both the drift and diffusion terms depend on the local gradient. I would like to know whether it is possible to understand how sensitive the model’s performance is to this gradient dependence. For instance, if ( g_1 ) varies too sharply with ( \nabla u ), could it lead to unstable training dynamics? I feel some empirical or theoretical insight into this behavior would be useful.


6). The core idea centers on the argument of a “residual dependence on the initial image.” I would like to know whether the authors attempted to measure how much information about the initial image remains at t=T. Can this dependence be quantified using a specific metric or statistical measure? I believe such an analysis would provide deeper insight into the behavior and effectiveness of the proposed method.

### Questions
see the weaknesses section.

### Soundness
3

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
3

### Summary
This paper proposes an extension of score-based generative models (SBGMs) by formulating the forward diffusion process using anisotropic Stochastic Partial Differential Equations (SPDEs) instead of the more common Stochastic Differential Equations (SDEs). The authors argue that traditional SDE-based methods, which are typically isotropic, destroy all information uniformly, including valuable geometric structures.

The core contribution is the introduction of nonlinear, spatially-dependent SPDEs where the drift and diffusion coefficients are influenced by anisotropy coefficients. The goal with this formulation is to more generally preserve spatially relevant information such as edges during the forward process, so that the reverse process can in turn leverage these geometric cues to sample images with higher fidelity.

### Strengths
I believe the paper's formulation is novel and interesting. Some strengths are:  

**(S1)**: Novel theoretical contribution. A general framework for spatially-dependent diffusion processes is quite valuable as it can capture more complex dynamics in structured data like images. I particularly like how many different forms of data corruption in the forward process are subsumed under the same framework. I would like to see this extended further across modalities.

**(S2)**: Experimental validation on pre-trained model. The authors demonstrate that their training on top of an existing diffusion model yields improvements in standard image generation quality metrics such as FID and inception score.  

**(S3)**: Clear presentation for the theoretical section. I appreciate the care with which the authors explained the differences in diffusion frameworks and then unified it under one umbrella. This set up the motivation and intuition for the core method well-- preserving geometric structures via anisotropy will aid in reconstruction.  

Overall, I think the idea is interesting and deserves further exploration.

### Weaknesses
**(W1)**: Lack of thorough experimental validation. This is my main concern with the paper. There are a number of baselines and prior relevant work that have not been included in the experimental results (eg: Table 1). This makes it hard to judge the efficacy of the method. E.g. [1] achieved an FID of 1.97 on CIFAR-10. [2] tackles a similar problem as this paper, but results have not been compared. 

**(W2)**: Computational cost. The forward and backward process requires the computation of spatial gradients (eq 7, 8) via finite differences, for every time step. This would arguably make training and inference much slower. A comparison between the quality / performance tradeoff with prior work and baselines is critical and is missing.

**(W3)**: Extension to latent diffusion models. While the theoretical framework should apply equally to standard and latent diffusion models, I would be curious to see empirical results on latent diffusion models, which have become the standard today. Do the findings still hold? This is important for broader relevance and applicability. 

**(W4)**: While the authors do discuss the anisotropy, diffusivity and intensity coefficients in Appendix A, experimental validation for different choices of hyperparameters is missing. There are multiple new hyperparameters introduced in this paper and a detailed ablation would be quite important.

The weaknesses slightly outweigh the strengths for me. I would encourage the authors to present more extensive empirical results to support their claims.

---  
References:  
[1] Elucidating the Design Space of Diffusion-Based Generative Models, NeurIPS 2022.  
[2] Edge-preserving noise for diffusion models, arXiv 2410.01540.

### Questions
**(Q1)**: L456. "Notably, according to the original authors, continuing training with their own method did not yield further metric improvements."  
Was this verified via experimental results on your side?

**(Q2)**: Table 1. Why is the Ours (Isotropic) version so much worse on FID than the anisotropic version? What are the results on the remaining datasets? 

**(Q3)**: Are there any results on the fully anisotropic variant? Where both the diffusion and drift terms are anisotropic. 

**(Q4)**: The fine-tuning experiment (Figure 2)  is interesting. Does this imply that the primary benefit is in the sampling path (i.e., the backward SDE derived from the anisotropic SPDE is simply a better path from noise to data), or is the score model itself being fundamentally retrained to leverage geometric information that it was previously ignoring?

**(Q5)**: The exposition on different diffusion methods was clear and well-written. Section 4.1 was a bit opaque to me. It was a bit difficult for me to get an intuitive sense of the terms in eqs. 7, 8, 9. Some additional explanation would be valuable for this section.

### Soundness
3

### Presentation
3

### Contribution
3
