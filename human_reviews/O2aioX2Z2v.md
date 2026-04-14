# Stochastic Deep Restoration Priors for Imaging Inverse Problems

- Decision: Reject
- Scores: 6, 1, 6, 5, 3

## Abstract
Deep neural networks trained as image denoisers are widely used as priors for solving imaging inverse problems. While Gaussian denoising is thought sufficient for learning image priors, we show that priors from deep models pre-trained as more general restoration operators can perform better. We introduce Stochastic deep Restoration Priors (ShaRP), a novel method that leverages an ensemble of such restoration models to regularize inverse problems. ShaRP improves upon methods using Gaussian denoiser priors by better handling structured artifacts and enabling self-supervised training even without fully sampled data. We prove ShaRP minimizes an objective function involving a regularizer derived from the score functions of minimum mean square error (MMSE) restoration operators, and theoretically analyze its convergence. Empirically, ShaRP achieves state-of-the-art performance on tasks such as magnetic resonance imaging reconstruction and single-image super-resolution,  surpassing both denoiser- and diffusion-model-based methods without requiring retraining.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Stochastic deep Restoration Priors (ShaRP), a method that leverages an ensemble of deep restoration models to regularize imaging inverse problems. ShaRP improves upon Gaussian denoiser-based methods by handling structured artifacts more effectively, enabling self-supervised training without fully sampled data.

### Strengths
- The paper is well-written and easy to follow.
- The theoretical analysis of the convergence is thorough and well-explained.

### Weaknesses
1. **Comparison to Diffusion-Based Methods:**
   The paper lacks a comprehensive comparison to diffusion-based methods, such as DiffIR [1] and DDRM [2]. Including these comparisons would strengthen the results and provide clarity on how ShaRP performs relative to other leading methods in the field.

2. **Supervised vs. Self-Supervised ShaRP:**
   In line 154, the authors mention a key contribution: "We implement ShaRP with both supervised and self-supervised restoration models as priors and test it on two inverse problems: CS-MRI and SISR." However, the experimental section does not provide a direct comparison between the self-supervised and supervised versions of ShaRP for the same task. This comparison is necessary to assess the benefits of the self-supervised approach.

3. **Self-Supervised Nature:**
   For a restoration network to be trained on a set of tasks, such as a set of blur kernels \(H_i\), access to ground truth data is still required. This approach, which involves sampling multiple times from the ground truth, raises questions about whether the method can truly be considered self-supervised. Clarification is needed regarding the self-supervised claim.

4. **Use of Multiple Degradation Operators:**
   The rationale for using a set of degradation operators \(H_1, H_2, \ldots, H_k\) in cases where the target problem involves only a single fixed operator (e.g., \(H_1\)) is unclear. It would be helpful if the authors could explain why introducing multiple degradation operators is necessary or beneficial when solving a fixed-task problem.

**References:**

- [1] DiffIR: Efficient Diffusion Model for Image Restoration
- [2] Denoising Diffusion Restoration Models

### Questions
What is the practical inference time of the proposed method in comparison to state-of-the-art (SOTA) methods? Additionally, the visual comparisons presented do not clearly demonstrate significant improvements. It would be beneficial to include more compelling visual examples to better illustrate the advantages of the proposed approach.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
This paper extends the "deep restoration priors" approach from Hu et al 2024c from one degradation operator to multiple ones.

### Strengths
The paper is clearly written, for the most part.

### Weaknesses
This paper is a trivial extension of Hu et al 2024c.  In this paper, the degradation operator is randomly chosen from a set, whereas in Hu et al 2024c there was a single operator.  But this change led to no new challenges or questions.  For example, the "ShaRP" regularizer in (6) is a trivial extension of (9) in Hu et al 2024c that now takes an expectation over H.  Figure 2 in this paper is a direct copy of Figure 1 from Hu et al 2024c.  The experiments are conducted on different linear inverse problems, but there is no intellectual contribution there.

### Questions
I don't have any questions for the authors.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces a novel concept termed ShaRP for stochastic priors for the regularization of inverse problems. At its core, a set of $b$ restoration problems is considered, for which the MMSE is minimized. In this setting, ShaRP is the expectation of the probability of the degraded version of the images given the probability density of the observations. Under certain assumptions, a closed-form solution of the gradient of the regularizer is derived and the convergence of $\nabla f$ is shown. The results are numerically tested for CS-MRI and SISR.

### Strengths
The paper introduces a novel concept for stochastic priors, which is complemented by two important theoretical assertions (see summary). The assumptions can be regarded as mild. Both results are highly relevant for numerical experiments. 
The paper remarkably advances SOTA for CS-MRI and is on par with competing methods in the case of SISR.

### Weaknesses
The presentation of the paper could be improved in some places. In particular, I am missing a more concise introduction to the motivation behind the actual concept of ShaRP in Section 4.
The parameter $b$ representing the number of restoration problems is essential for the numerical performance. However, I am lacking a substantial discussion of the role of this parameter. In particular, no details about the role of $\alpha$ (see Section 5.1) and its impact on the results were provided.

### Questions
1. Can you please provide more details about the impact of the choice of $b$ on the numerical results? An ablation study might help here. Likewise, you could numerically evaluate the impact of $\alpha$.
2. What happens if you modify the number of restoration priors in B.1.1?
3. Please rewrite the motivation in Section 4. In particular, I am missing a good motivation and reasoning for the actual definition in equation (6), which might help the reader to better understand ShaRP. In addition, the object $G_\sigma(s-Hx)$ could be better motivated from a mathematical point of view.
4. The results in Table 4 do not show a clear tendency for DiffPIR, DRP, or ShaRP. Is there a reason for this available? Are there specific conditions under which each method performs best?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes a new approach for learning a regularizer for linear imaging inverse problems. While Gaussian denoisers have been successfully used as effective image priors, the authors extend the idea by using more general restoration operators as image priors. In the numerical experiments, the authors show the reconstruction performance with their new “ShaRP” regularizers for two important inverse problems (compressive MRI and single-image super-resolution). The numerical experiments are performed with ShaRP regularizers trained in supervised and self-supervised manners.

### Strengths
1. Using general restoration operators instead of simple Gaussian denoisers is interesting and novel. Theorem 1 also puts this idea on a solid theoretical foundation. 

2. The experimental results are compelling. Comparison of ShaRP with different baseline methods, especially PnP algorithms that utilize Gaussian denoisers, provides evidence for the claim made by the authors in the abstract.

### Weaknesses
1. I am not particularly convinced about the motivation for using a general restoration prior, that arises out of training a deep reconstruction operator for linear inverse problems (with several degradation operators), for solving another linear inverse problem. While learning Gaussian denoisers is a problem that is fundamentally easier than solving a general inverse problem (with an ill-posed operator), this is not the case here. 

2. The novelty in the convergence analysis (namely, the proof of Theorem 2) is unclear to me. To my understanding, the assumptions and the result are along the same lines as in the paper entitled “A Guide Through the Zoo of Biased SGD” by Demidovich et al. (which the authors have cited). 

3. Although the authors show the performance of shaRP trained in an unsupervised manner, I don’t think the training loss used in the unsupervised case (e.g., the loss in Algorithm 3) corresponds to a restoration operator that approximates the conditional expectation $E[x|s, H]$. Consequently, the theoretical analysis does not explain the experimental results with a restoration operator trained in an unsupervised manner.

### Questions
1. Page 1: “...Tweedie’s formula (Robbins, 1956; Efron, 2011) seemingly implies that Gaussian denoising alone might be sufficient for learning priors,...”: That is indeed true and this work does not disprove this fact. 

2. Page 2: “...ShaRP provides a richer and more flexible representation of image priors…”: At this stage, it is rather vague as to what “richer and more flexible” means. 
3. Page 2: “Unlike Gaussian denoisers, the restoration models in ShaRP can often be directly
trained in a self-supervised manner”: Even Gaussian denoisers can be trained in an unsupervised manner using only noisy images (e.g., using a SURE loss). 

4. It might be good to put some of the math background in the intro, effectively shortening the material in the first three pages. 

5. Page 3: “We introduce a novel regularization concept for inverse problems that encourages solutions that produce degraded versions closely resembling real degraded images.”: Firstly, this would be the case if the degradation is the same for which the restoration network is trained, not the forward operator “A” for which you want to solve the inverse problem. Secondly, can one not promote this property using a simple Gaussian denoiser together with a data-consistency loss? 

6. Theorem 1: “Assume that the prior density $p_x$ is non-degenerate over $\mathbb{R}^n$”: This is almost always not true. 

7. Assumption 3: Define what “$b(x)$” is. 

Requested changes:

1. Provide a comprehensive review of the convergence analysis of biased SGD in terms of the assumptions and results to put your theoretical contributions in perspective. 

2. Clarify whether one can approximate the conditional expectation $E[x|s, H]$ using a restoration operator trained on a self-supervised loss. If not, the interpretation of ShaRP as biased SGD does not hold in this case.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper extends the work of [Hu et al., 2024c] for the regularization of inverse problems using a deep restoration prior trained on multiple restoration tasks. The authors theoretically analyze the induced objective function and the convergence of the proposed algorithm. Empirical results demonstrate that the framework outperforms Gaussian denoiser priors in image reconstruction tasks, including MRI and super-resolution.

### Strengths
1. The experiments show that training a deep restoration network on multiple tasks can improve its regularization capability.

2. Theoretical analysis is also provided.

### Weaknesses
1. The paper seems limited in novelty and closely resembles [Hu et al., 2024c]. The main contribution is the training of a restoration prior across the same task with varying levels of ill-posedness. The authors should clearly explain how the proposed methodology differs from [Hu et al., 2024c].

2. The authors claim that the proposed prior trained on a general restoration task outperforms Gaussian denoisers; however, there is a lack of sufficient experiments to support this claim. Gaussian denoisers can be used for solving general inverse problems, but the experiments presented here are confined to the same inverse problem and forward operator on which the prior was trained, although with different levels of ill-posedness. For example, in Section 5.1, the restoration prior is trained on MRI with an x8 acceleration and applied to the same problem with x4 and x6 acceleration factors. Although Algorithm 1 suggests that the proposed prior can be used to solve any other inverse problems, the experiments are limited to the same problem type. To validate the claimed superiority over Gaussian denoisers, the authors are requested to include experiments where the trained prior is applied to a different inverse problem from the one it was originally trained on. For example, training the prior on super-resolution and using it for solving MRI reconstruction. You can also train your prior on multiple distinct tasks as suggested by Algorithm I, for example training the prior on super-resolution and image in-painting tasks, and use it for solving a different task like MRI. This comparison is crucial as Gaussian denoisers can solve any other different inverse problems once trained.

Hu, Yuyang, et al. "A Restoration Network as an Implicit Prior." ICLR 2024.

### Questions
Please take a look at the comments under weaknesses

### Soundness
3

### Presentation
3

### Contribution
1
