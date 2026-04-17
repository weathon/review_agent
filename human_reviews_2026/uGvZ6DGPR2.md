# Likelihood Matching for Diffusion Models

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
We propose a Likelihood Matching approach for training diffusion models by first establishing an equivalence between the likelihood of the target data distribution and a likelihood along the sample path of the reverse diffusion. To efficiently compute the reverse sample likelihood, the equivalence, a quasi-likelihood is considered to approximate each reverse transition density by a Gaussian distribution with matched conditional mean and covariance, respectively. The score and Hessian functions for the diffusion generation are estimated by maximizing the quasi-likelihood, ensuring a consistent matching of both the first two transition moments between every two time points. A stochastic sampler is introduced to facilitate the computation that leverages on both the estimated score and Hessian information. We establish consistency of the quasi-maximum likelihood estimation, and provide non-asymptotic convergence guarantees for the proposed sampler, quantifying the rates of the approximation errors due to score and Hessian estimation, dimensionality, and the number of diffusion steps. Empirical and simulation evaluations demonstrate the effectiveness of the proposed Likelihood Matching and validate the theoretical results.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a likelihood-based training framework for diffusion models that incorporates both first- and second-order score supervision. By matching the mean and covariance of reverse transitions, the method extends standard score matching. Empirical results show improved performance over vanilla score matching, and theoretical analysis provides non-asymptotic convergence guarantees for the resulting sampler, quantifying the impact of score and Hessian estimation error.

### Strengths
The paper is well-written and easy to follow. The theoretical analysis is thorough and well-supported by experiments on both synthetic and real datasets.

### Weaknesses
I am very willing to raise my score if the following concerns are addressed:

1. Ambiguity of N: The notation N is sometimes unclear, it's not always evident whether it refers to the total number of sampling steps, the number of selected time steps during training, or something else. Could the authors provide a unified and consistent definition of N throughout the paper?

2. Motivation for Second-Order Information: The motivation for incorporating the Hessian (second-order score) remains insufficiently convincing. Why is second-order information important for diffusion models? How does it improve performance? Rather than only reporting improved metrics, the authors are encouraged to provide deeper empirical or theoretical insights. For example, since the Mixture of Gaussians (MoG) has an analytical score function, comparing oracle trajectories (e.g., deterministic ODE sampling) with score-only trajectories could be highly illustrative. If this is computationally intensive, a compelling empirical explanation would also be acceptable.

3. Low-Rank Approximation Hyperparameter: A minor but relevant limitation is that the low-rank Hessian approximation rank r is a hyperparameter requiring tuning. While not a critical issue, it would be helpful if the authors could offer empirical guidance on how to set or adapt this parameter.

4. Connection to Prior Work: The paper would benefit from a brief discussion on how its covariance modeling relates to prior DDPM variants that also learn covariance matrices, as well as to score matching with weighting schemes. Citing one or two representative works would be sufficient.

5. Time Sampling Strategy: A minor but interesting point for discussion: some prior works suggest that sampling t from a non-uniform, pre-defined distribution can improve performance. In contrast, this paper adopts uniform sampling as dictated by the likelihood objective. Could the authors discuss the pros and cons of these time sampling strategies?

### Questions
See weakness above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes learning a Gaussian denoising distribution with a learned mean and covariance in a diffusion model, where the covariance is learned using a low-rank approximation.

### Strengths
There are some theoretical analyses that make the paper look okay.

### Weaknesses
1. The covariance estimation of the Gaussian denoising diffusion model is a solved problem. I am not sure why this paper didn't discuss any related work on this. I will give a brief introduction to this line of research and show **why learning the covariance under quasi-MLE is unnecessary**. All the papers I list below use a Gaussian variational distribution to approximate the denoising distribution under forward KL divergence, which is the same as the quasi-MLE terminology that this paper mentioned. Unfortunately, none of these papers is cited or mentioned.

For example:
1. iDDPM (https://arxiv.org/abs/2102.09672) is the first paper to learn the covariance under ELBO, and showsthat  learning the covariance of a Gaussian can accelerate the denoising sampling.
2. Analytic-dpm (https://arxiv.org/abs/2201.06503) shows that the optimal state-independent isotropic Gaussian covariance can be analytically derived with the learned mean function.
3. OCM-DDPM (https://arxiv.org/pdf/2406.10808) shows that the optimal covariance only depends on the learned mean function, which means you can just learn the desnoing mean, and the full covariance can be analytically derived, which means learning the covariance is not necessary. The optimal here is all under the KL divergence.

There are also NPR-DDPM and SN-DDPM (https://arxiv.org/abs/2206.07309) are all learning the covariance.


2. The experiments and comparisons are not convincing. 
The experiments need to reach the level of one of the papers above. The current experiments are too toy-like and have no comparisons.

I suggest the author do a literature review before starting a research project. The Gaussian approximation of the denoising distribution is almost finished; there are also non-Gaussian approximations of the denoising distribution, e.g. https://arxiv.org/abs/2502.02483.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a Likelihood Matching approach for training diffusion models by first establishing an equivalence between the likelihood of the target data distribution and the likelihood along the sample path of the reverse diffusion. By assuming that the reverse process is roughly Gaussian for small step sizes, it proposed to parametrize both the mean and the variance of such a Gaussian by parametrizing the score and the Hessian with neural networks. Further, it provides a bound on the total variation between the learned and the true distribution, and proves convergence in probability to the true parameters of the target distribution. The resulting method 'likelihood matching' (LM) is then tested on synthetic and real data.

### Strengths
Overall the paper has some interesting results. It provides non-asymptotic convergence guarantees for the proposed sampler in total variation, characterizing the errors in terms of score and Hessian estimation error, dimension d, and diffusion steps T.

It theoretically demonstrate the consistency of the proposed quasi-maximum likelihood diffusion training under reverse quasi-likelihood objectives.

Multiple simplifications are made that make this approach implementable in practice, albeit it still remains much more expensive than direct score matching.

### Weaknesses
**Main Weaknesses**

*W1* It is unclear why the Hessian is necessary theoretically. The reverse diffusion generates precisely the same distributions as the forward one, and the only unknown term therein is the score. In this sense, the score is a sufficient statistic to go backwards. The Fokker-Planck equation implies the same conclusion, as by formulating the backward probability evolution via an ODE, then the target distribution is perfectly modeled if the score has been perfectly learned. These points are also supported by Theorem 2 in [1], as that theorem implies that the KL between the modeled and target distribution becomes 0, for perfectly learned scores.

*W2* The modeling of the Hessian causes an increase of more than 2x of training time and memory usage (for Cifar). Given the modest improvements in terms of FID, it is not clear why this approach should be adopted. In addition, the compute results are only given for Cifar10. I suspect the 2x gap would increase even further for higher dimensional distributions. Could the authors provide the difference in training/memory time in the case of high-resolution Imagenet?

*W3* As stated in the paper, several previous works do adopt elements from the Hessian, or try to model the variance with an isotropic Gaussian among other approaches. In [2] the reverse step probability is modeled as a mixture of Gaussians. Given that the reverse probabilities are intractable, the approximation with a Gaussian is not fully justified, in particular for larger steps. Does the use of the Hessian give a better estimate for larger step sizes (i.e. reduce the number of steps)?

*W4* The writing quality of the paper should be improved. There are numerous grammatical errors and typographical mistakes throughout. A few examples, among many, include the very first sentence of the introduction, line 204, lines 300–301, and even the text within figures (lines 1248–1250).

[1] Song et al. Maximum Likelihood Training of Score-Based Diffusion Models. Neurips 2021

[2] Guo et al. Gaussian Mixture Solvers for Diffusion Models. Neurips 2023.

### Questions
Q1. How was the NLL computed? Using the ODE formulation or through the likelihood lower bound?

Q2. What is the difference in performance when time-steps are predefined (Algorithm 1), vs when they are randomly sampled (Algorithm 2)?

### Soundness
2

### Presentation
1

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
This paper proposes Likelihood Matching (LM), a training framework for diffusion models that directly approximates the data likelihood via a quasi-likelihood over the reverse diffusion path.

It models both the score and Hessian to match conditional mean and covariance, leading to a unified objective combining score and covariance matching.

The authors derive theoretical guarantees (consistency of quasi-MLE, non-asymptotic sampler bound) and show empirical gains on MNIST, CIFAR-10, and CelebA.

### Strengths
* Conceptually elegant: connects data likelihood with path likelihood of reverse diffusion: an underexplored yet fundamental viewpoint.
* Introduces a principled QMLE formulation, integrating first- and second-order information beyond prior Hessian-regularized SM methods.
* Solid theoretical results with practical, scalable implementation (low-rank Hessian, SMW updates).
* Consistent FID/NLL improvements and faster convergence in sampling.

### Weaknesses
* Novelty could be more clearly contrasted with prior MLE-based diffusion ODE works [1,2,3]
* Experiments remain small-scale.

[1] Song, Yang, et al. "Maximum likelihood training of score-based diffusion models." Advances in neural information processing systems 34 (2021): 1415-1428.

[2] Lu, Cheng, et al. "Maximum likelihood training for score-based diffusion odes by high order denoising score matching." International conference on machine learning. PMLR, 2022.

[3] Zheng, Kaiwen, et al. "Improved techniques for maximum likelihood estimation for diffusion odes." International Conference on Machine Learning. PMLR, 2023.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
