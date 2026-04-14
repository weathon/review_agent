# Blind Inversion using Latent Diffusion Priors

- Decision: Reject
- Scores: 6, 6, 5, 5

## Abstract
Diffusion models have emerged as powerful tools for solving inverse problems due to their exceptional ability to model complex prior distributions. However, existing methods predominantly assume known forward operators (i.e., non-blind), limiting their applicability in practical settings where acquiring such operators is costly. Additionally, many current approaches rely on pixel-space diffusion models, leaving the potential of more powerful latent diffusion models (LDMs) underexplored. In this paper, we introduce LatentDEM, an innovative technique that addresses more challenging blind inverse problems using latent diffusion priors. At the core of our method is solving blind inverse problems within an iterative Expectation-Maximization (EM) framework: (1) the E-step recovers clean images from corrupted observations using LDM priors and a known forward model, and (2) the M-step estimates the forward operator based on the recovered images. Additionally, we propose two novel optimization techniques tailored for LDM priors and EM frameworks, yielding more accurate and efficient blind inversion results. As a general framework, LatentDEM supports both linear and non-linear inverse problems. Beyond common 2D image restoration tasks, it enables new capabilities in non-linear 3D inverse rendering problems. We validate LatentDEM's performance on representative 2D blind deblurring and 3D sparse-view reconstruction tasks, demonstrating its superior efficacy over prior arts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a new diffusion-based model for blind inverse problems, which alternates between a diffusion step with a fixed parameter (eg a fixed blur kernel), and a step estimating the kernel given an estimate of the image. The paper motivates this algorithm as an instance of the expectation-maximization (EM) framework. In particular, the approach leverages powerful latent diffusion models such as Stable Diffusion v1.5, and performs the bulk of the diffusion in the latent space. 

The method is demonstrated in blind image deblurring and sparse view 3D reconstruction, obtaining a better performance in terms of both distortion and perceptual quality than the competing baselines.

### Strengths
- introduces an approach that leverages pretrained encoder/decoders which reduces the computational burden of working in the pixel space.
- good experimental results in blind deblurring and sparse view 3D reconstruction.

### Weaknesses
- I believe the link to the EM framework is not sufficiently well founded: the MAP kernel estimation step uses \hat{x_0} instead of a sample from the posterior p(x|y, \phi). It is not clear to me what is the role of q(x|y) in equation 5, and more generally the equation does not add much to the explanation in my opinion.
- The experimental setup is not sufficiently well described in the main text, i) raises doubts about testing on a training set (see related questions below) and ii) the papers seems to use circular padding which is not a very realistic setting.
- The presentation of the paper could be improved:
    - the citation style is incorrect in many parts of the paper (use citep{} for references that are not part of a sentence)
    - the mathematical notation is not consistent: equation 11 uses Z uppercase for denoting a vector, whereas the rest uses lowercase, the letter 'z' is used for both latent space vectors and auxiliary variables in 11, the letter D is used for a denoiser and a decoder. What is the  'F^2' operator in equation 11? Acronyms are defined multiple times throughout the text (eg. EM). The method Zero123 is not followed by a reference when introduced on page 7.

### Questions
- What is the denoiser used for estimating the blur kernels? how was that denoiser trained? 
- Is there an overlap between training kernels and test kernels? How were the test kernels generated/chosen? 
- Was the Stable Diffusion model was trained on the FFHQ and ImageNet images used for testing the models?

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
4

### Summary
This paper introduces LatentDEM, a novel method for addressing blind inverse problems through latent diffusion models. The proposed method leverages an Expectation-Maximization (EM) framework: in the expectation step, a clean image is recovered using a latent diffusion prior, and in the maximization step, the forward model is estimated based on the recovered clean image. The authors demonstrate the method’s application to blind deblurring and 3D pose-free sparse-view reconstruction.

### Strengths
The paper is well-organized and clearly communicates its results. 

The authors provide extensive experimental results, comparing their method to prominent existing methods for blind inverse problems, such as BlindDPS and FastEM. 

They provide an ablation study examining the influence of gradient estimation skipping and annealing consistency on performance.

### Weaknesses
1. A primary concern is the paper’s original contribution. The methodology appears to combine elements previously explored in the literature. For instance, latent diffusion models for inverse problems have been investigated in [1], while Expectation-Maximization approaches for blind inverse problems have been studied in [2]. Simply substituting a latent diffusion model as the prior does not, in itself, seem to constitute a novel contribution. Explicitly listing the contributions could improve clarity.

2. Another concern is the runtime of the proposed method. As seen in Table 2, the method appears to trade time for performance, with comparable results to BlindDPS but with similar computational demands. In other words, we would not gain any performance advantage compared to BlindDPS, if are limited by the runtime of the method. 

3. The reported performance metrics for FastEM and BlindDPS appear inconsistent with previous results. For instance, in the FastEM paper, performance on the FFHQ dataset is as follows: MPRNet: 19.52, BlindDPS: 24.05, and FastEM: 25.75, demonstrating comparability to DPS given the GT kernel. However, in this paper, the results are MPRNet: 21.60, BlindDPS: 22.58, and FastEM: 17.46. Could the authors provide an explanation for this performance discrepancy? Is the comparison of FastEM accurate? Can they reproduce FastEM’s results by strictly following its settings for a more direct comparison?

[1] Litu Rout, Negin Raoof, Giannis Daras, Constantine Caramanis, Alex Dimakis, and Sanjay Shakkottai. Solving linear inverse problems provably via posterior sampling with latent diffusion models. Advances in Neural Information Processing Systems, 36, 2024.

[2] Charles Laroche, Andrés Almansa, and Eva Coupete. Fast diffusion em: a diffusion model for blind inverse problems with application to deconvolution. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 5271–5281, 2024.

### Questions
Have the authors observed any instances of non-converging patterns in their algorithm? Specifically, does a particular choice of annealing or tunable parameters (Equations 11 or 12) lead to divergence?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposed a method for solving blind restoration problems (examined mostly on blind deblurring) using a pretrained latent diffusion model (LDM). It is based on adding to the iterations of unconditional LDM sampling steps (which are sometimes skipped) where the the observation/forward model parameters (e.g., blur kernel) are updated and being used for guiding the diffusion (I do not see a clear relation to expectation maximization). The proposed method is quite heuristic and shares similar ideas with existing works that use guided LDM for non-blind restoration and works that use guided pixel-space DMs for blind restoration. In the reported experiments it outperforms alternative methods.

### Strengths
- As far as I know, this is the first work that uses a pretrained LDM for blind restoration.

- The proposed method appears to have empirical advantages over competitors.

### Weaknesses
- The contribution seems incremental in terms of the techniques being used and the lack of theoretical motivation and analysis of the proposed method. 

- The presentation is not good in terms of explaining why the authors refer to the proposed method as an expectation maximization (EM) technique.

### Questions
- The presentation of EM in Section 3 differs from the plain EM formulation (as in Dempster et al., 1977 and textbooks).
For example, what are the unobserved/missing data? Where in the E-step do you compute (conditional) expectation of the log likelihood/posterior with respect to the missing data?
The authors may refer to some variational Bayes modification of EM but not to the actual EM. Make the discussion more clear and point to the reference that justifies your formulation.

- Consequently, the presentation of the method in Section 4 needs to be improved. The derivation of the method does not match the EM paradigm. It looks like some alternating optimization scheme that lacks formal connection to EM.

- Lack of theoretical analysis for the proposed modifications to the baseline alternating optimization approach.  
For "Technique 1" there is some motivation in the appendix, but no formal mathematical argument that actually considers the effect of error in phi. The other modifications are quite heuristic.  

- Regarding the experiments, you need to provide more details on the kernels, noise levels, etc., that you use in training and in testing.  
Also, state in some table the hyperparameters that you use. Do you use the same hyperparameters for all the settings and for both 2D and 3D?

-  The reported results of the competitors for blind deblurring on FFHQ are lower than those reported in previous works  (Chung et al., 2022a), (Laroche et al., 2024) (e.g., worse kernel estimation for BlindDPS and huge deterioration for FastEM).  
How can you explain it? Do you use the same setup as these works? If not, then explain why.

- Please compare also with GibbsDDRM:  
Murata, Naoki, et al. "Gibbsddrm: A partially collapsed gibbs sampler for solving blind inverse problems with denoising diffusion restoration." ICML, 2023.

- Present also the results for the non-blind methods on which BlindDPS and FastEM, and your approach, are based. They can be interpreted as the upper bound on the results, as done in GibbsDDRM and FastEM.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper aims to tackle the challenging problem of blind inverse tasks by employing latent diffusion priors within an expectation-maximization (EM) framework, which iteratively estimates both the unknown measurement operator (within M-step) and the underlying image (within E-step). The proposed approach has been validated for tasks of 2D blind deburring and 3D pose-free sparse-view reconstruction. The parameters of the forward model are estimated in the M-step through using a pre-trained denoiser of Gaussian noise within a PnP framework. In the E-step, a so called "annealing consistency" approach is proposed through introducing a tunable hyper-parameter in the score of likelihood term to stabilize the training. In addition to the score of likelihood term,  a "gluing" regularization (Rout et al, 2024) is employed within diffusion posterior sampling for 2D deblurring task.

### Strengths
This problem holds broad interest across the machine learning and signal processing communities.
The paper is generally well-structured and somewhat clear.

### Weaknesses
Several technical ambiguities require clarification. These include the choice of likelihood formulation, the role of “gluing” regularization versus annealing consistency, and the performance discrepancies across datasets. More experimental validation, particularly isolating the effects of different regularization techniques, would strengthen the claims. Additionally, a detailed comparison of hyperparameters with competing methods would improve transparency and reproducibility. For further information of the detected ambiguities please refer to the section "Questions".

### Questions
the following issues and ambiguities require further clarification or validation:

1. While (Chung et al. 2022b) formulates the score of the data likelihood as $\nabla_{z_t} \log p_t(y|z_t)$ in Eq.(14), this paper opts for Eq.(9), citing “annealing consistency” for enhanced stability and performance. The authors should provide empirical evidence or theoretical justification for this choice. How does “annealing consistency” compare to the “vanilla” latent DPS term? Also, clarify the behavior of Eq.(9) and Eq.(18) when measurements are noiseless, i.e.  $\sigma = 0$.

2. For the 2D blind deblurring task, Eq.(16) includes a “gluing” regularization term (in addition to “vanilla” latent DPS-based term) as per (Rout et al. 2024). It is unclear whether the reported improvements are due to this regularization or the proposed “annealing consistency.” An experiment isolating these factors in 2D deblurring would be helpful.

3. The paper should explain why BlindDPS performs much better on FFHQ than ImageNet, and detail the training setup. Additionally, comparing FID metrics between this work, FastEM (Rombach et al. 2022), and BlindDPS (Chung et al. 2022a) would improve clarity [as used in the aforementioned papers as well]. Why does FastEM show negligible improvement over initial observations (e.g., see the estimated blurring kernel which is close to Dirac delta)?

4. Clarify the parameter $\phi$ for widely studied image inverse tasks (e.g., compressed sensing, tomography, super-resolution, inpainting, spectral imaging) to enhance understanding of its role across applications.

5. The paper aims to tackle [general] ‘blind’ inverse problems through estimating the parameters of forward measurement operator in the M-step of the EM framework as eq.(10).  To this aim, the parameter $\phi$ (e.g., blurring kernel for the task of 2D deblurring) is estimated through a PnP denoising step as eq.(19). Please elucidate on whether a denoiser pre-trained on Gaussian noise can generalize to other inverse tasks like compressed sensing, and explain what signal regularities it might exploit in estimating $\phi$.

6. A comprehensive list of trainable and tunable hyperparameters for the proposed method, along with those used in Fast Diffuse EM (Laroche et al. 2024) and BlindDPS (Chung et al. 2022a), would facilitate fair comparisons.

7. Please specify the network used in the E-step and provide experimental setup details for Table 1.

Minor Comments:

- In the context “these priors cannot capture the complex natural image distributions, limiting the solvers’ ability to produce high-quality reconstructions”, please consider replacing the (Candes & Romberg, 2007) citation with the following, which better aligns with the stated limitations of image priors:
https://doi.org/10.1109/TIP.2005.863057
https://doi.org/10.1109/ICIP.2007.4379013
https://doi.org/10.1109/GlobalSIP.2013.6737048
https://doi.org/10.1137/16M1102884

- Please clarify initialization of $\mathcal{A}_{\phi^{(t)}}$ in Algorithm 1 and specify corresponding M-step equations (perhaps Appendix C.2 ?)

- Differences between Eq.(11) of this paper and Eq.(24)/Eq.(B.9) in (Laroche et al. 2024) require clarification.
    
- Please correct discrepancies in Section 4.3 regarding the noiseless assumption in Eq.(1).

- Minor typographical and grammatical errors include:
  * Typo corrections: “a intermediate” and “an one-to-many.”
  * Redundant citations: “Liu et al. (2023b)” and “Liu et al. (2023c)” refer to the same work.
  * Misuse of “While” and “However” following Eq.(4).
  * Replace “introduce” with “utilize” or “employ” when referring to “gluing” regularization as it is not part of the contribution of this work.
  * The paper overlooks the following relevant work:
        Murata et al., “Gibbsddrm: A partially collapsed Gibbs sampler for solving blind inverse problems with denoising diffusion restoration.” ICML, 2023.

### Soundness
2

### Presentation
3

### Contribution
2
