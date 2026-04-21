# Free Hunch: Denoiser Covariance Estimation for Diffusion Models Without Extra Costs

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6, 6

## Abstract
The covariance for clean data given a noisy observation is an important quantity in many training-free guided generation methods for diffusion models. Current methods require heavy test-time computation, altering the standard diffusion training process or denoiser architecture, or making heavy approximations. We propose a new framework that sidesteps these issues by using covariance information that is available for free from training data and the curvature of the generative trajectory, which is linked to the covariance through the second-order Tweedie's formula. We integrate these sources of information using (i) a novel method to transfer covariance estimates across noise levels and (ii) low-rank updates in a given noise level. We validate the method on linear inverse problems, where it outperforms recent baselines, especially with fewer diffusion steps.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduce a L-BFGS-like method to maintain the covariance information of denoiser along the inference path to better solve the linear inverse problems via diffusion models.

### Strengths
1. It is novel to leverage L-BFGS way to approximate the covariance in a low-dim manner in DMs
2. the mathematical derivation is rigorous, though it is quite similar to the math of BFGS optimization
3. The design of initial covariance to be the data covariance is reasonable.

### Weaknesses
1. The presentation of the paper could be improved for clarity. The term 'conditional generation' is often associated with 'text-guided' or 'label-guided' generation. However, in this paper, it refers to conditions based on partially corrupted samples. While this interpretation is not incorrect and indeed falls under the classifier-guided framework, it might be confusing for readers at first glance. 

2. There are typographical errors present in the paper, for instance, in equation (98). These need to be corrected for accuracy and better understanding.

3. The paper lacks a theoretical bound on the approximation error of the covariance using the proposed method. Both Equation 6 and the low-rank approximation involve approximation errors.

### Questions
see Weaknesses.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a new framework for estimating covariance in diffusion models, addressing issues in current methods such as high test-time computation and heavy approximations. The proposed method utilizes readily available covariance information from training data and the curvature of the generative trajectory. The authors also present a method for transferring covariance estimates across noise levels.

### Strengths
1. It is novel and reasonable to design low-dim expression and update rules for covariance approximation.
2. There are abundant math derivations to support the methods

### Weaknesses
1.  To handle linear inverse problems using diffusion models, there are a lot of classes of methods [1]. The experiments in this work only compare some of them, arthors should at least mension why take them out of comparison.
2. There lack of a experiment to back up whether the low-dim covariance of the proposed method can indeed match the underlying true one. I understand this cannot be done under high dimensional case, mayba a 2D or 10D suffice.
3. There is other applications of diffusion models would need covariance like causual reasoning [2], likelihood-evaluation [3][4] and adjoint guidance[5], maybe you could add a discussion to these.


[1] Daras, Giannis, et al. "A survey on diffusion models for inverse problems." arXiv preprint arXiv:2410.00083 (2024).

[2] Sanchez P, Liu X, O'Neil A Q, et al. Diffusion models for causal discovery via topological ordering[J]. arXiv preprint arXiv:2210.06201, 2022.

[3] Lu C, Zheng K, Bao F, et al. Maximum likelihood training for score-based diffusion odes by high order denoising score matching[C]//International Conference on Machine Learning. PMLR, 2022: 14429-14460.

[4] Anonymous. Gradient-Free Analytical Fisher Information of Diffused Distributions.

[5] Song K, Lai H. Fisher Information Improved Training-Free Conditional Diffusion Model[J]. arXiv preprint arXiv:2404.18252, 2024.

### Questions
1. refer to weaknesses.
2. in Figure 6, these setting of y is what? inpating transform or deblurring? Does this guidance meaningful across different y|x_0 and dataset?

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
4

### Summary
This paper introduces a new method to estimate the covariance during the denoising process in diffusion models. The key idea is using Tweedie's formula to relate the covariance of the denoiser with the Hessian of $\log(p(x_t))$, then using the gradient of $\log(p(x_t))$ (i.e. the score function) evaluated during each denoising step to estimate this Hessian. Starting with an estimate of the covariance of the data distribution, the algorithm performs low-rank updates to this covariance matrix, inspired by quasi-Newton methods, using the score function estimates given by the neural network. This method is then applied to fast sampling for solving inverse problems, showing improvements over other methods.

### Strengths
1. It is difficult to efficiently estimate the covariance of the denoiser due to its high-dimensionality, and methods that attempt to estimate it has have to use reduced-complexity representations. Using the DCT-diagonal basis to estimate the initial data covariance and a low-rank update inspired by quasi-Newton methods is a novel approach.

2. Because this method does not require training an additional neural network, it can be applied to existing pre-trained diffusion models to be used for solving inverse problems.

3. The experimental results seem promising and improves on previous methods on both qualitative examples and quantitative metrics.

### Weaknesses
1. There needs to be more theoretical or empirical justification for some of the algorithmic choices made, such as the choice of quasi-Newton method or the DCT basis used to approximate the data covariance.

2. The notation in the paper is inconsistent and the definition of some quantities are not clear, for example are $p(x, t)$, $p(x_t)$, $p_t(x)$ referring to the same quantity? Also, it is not clear what exactly are the full algorithms used for solving linear inverse problems; Algorithms 1 and 2 in the paper only updates the covariance estimate.

3. The experimental improvements for solving linear inverse problems seems to be only for the few-sample-step regime. It is unclear if this covariance estimation method can be helpful in other applications such as unconditional sampling, or with non-linear guidance terms.

### Questions
1. Since the data covariance is estimated in the DCT basis where the covariance is approximately diagonal, have you considered training a model to directly estimate the denoiser covariance in this basis, or with a diagonal + low-rank representation?

2. The numerical results in Table 1 seems to be worse than that in the DDNM paper, is this because the number of sampling steps is limited?

3. What is the computational overhead for estimating the data covariance and doing the low-rank updates, compared to the cost of evaluating the neural network?

4. What is the variance exploding case first mentioned in Line 213? This doesn't seem to be explicitly defined in the paper.

5. What is the difference between $\mu_{0\mid t}(x)$ and $\mu_{0 \mid t}(x_t)$, as well as $\sigma(t)$ and $\sigma_t$?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors proposed a new methodological framework for estimating covariance in the generative methods of diffusion models.  The estimation of the covariance in the reverse diffusion process is purely based on the existing samples through a two-step updating scheme. The examples show the proposed method outperforms other existing methods in denoising.

### Strengths
Originality and significance: The proposed method is relevant to the diffusion models and is novel in that it provides a fast and efficient algorithm for estimating a broader class of covariance structures. 

Quality and clarity: The paper clearly connects the proposed method to existing literature on similar problems.

### Weaknesses
The exposition can be improved as the current manuscript contains many inconsistent notations, undefined variables, and many typos. I would say it is a lovely work with appealing results, but unfortunately, the manuscript should be significantly revised and proofread to be easier to follow.

### Questions
1. In the image data results, adding the online update steps to either identity covariance or the FH covariance does not always improve the performance. Is there any explanation or further investigation on why it could happen? 
2. Table 1 shows that in most cases, FH+online outperforms FH. But in Figure 7, only the denoised image from FH is demonstrated. Is there any vision difference between FH and FH+online in terms of the images in Figure 7?
3. Notational /exposition issues:
    1. In Eq. (1), the "noising forward process" should be defined in the joint distribution of the stochastic paths or a stochastic differential equation. Eq. (1) gives the margin distribution of the state at time t, which cannot formally "define" a process.
    2. In Eqs. (2) and (3), $\dot\sigma$ and $\omega_t$ are undefined. 
    3. Above Eq. (7), $p_{0|t}$ is undefined. 
    4. In Eq. (11), add a bracket to make the inverse clear. I.e., $[\nabla_{\mathbf{x}}^2\log p(\mathbf{x},t)]^{-1}$.
    5. The last few sentences in Sec. 3.1. confused me. The integral $\int p(\mathbf{x}')\mathcal N(\mathbf{x}'\mid \mathbf{0}, \sigma^2\mathbf{I})d\mathbf{x}'$ is not a convolution. 
    6. In Eq. (18), should the right-hand side be $\left[\mathbf{\Sigma}_{0\mid t}(\mathbf{x}+\Delta\mathbf{x})\right]\Delta\mathbf{x}$.
    7. In Eq. (23), use either $\nabla_{\mathbf{x}}$ or $\nabla_{\mathbf{x_t}}$ consistently throughout the paper.
    8. In Eqs. (23) and (24), $\mathbf{A}$ is undefined.
    9. Above Eqs. (25), $\mu_{0\mid t}$ should be in boldface. (and many other occurrences throughout the paper).
    10. In Sec. 4.2., "we noticed that the guidance scale is overestimated the larger the dimensionality is, leading to overconfidence." does not read well --- the authors probably left over some words from a previous edition. 
    11. Throughout the paper and the appendix, please only label the equations you refer to later.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors introduce a new covariance estimation method which makes use of covariance information in denoiser and the trajectory curvature without introducing significant additional compute. Moreover, to make the approach suitable for high-dimensional data (e.g. covariance matrices storage issue), the low-rank updates is proposed.  The authors validate their approach on linear inverse problems, demonstrating its effectiveness compared to baselines under four metrics.

### Strengths
- Novel method, proposed a covariance estimate method via separately updating along time and position/space. Also provide a practical implementation for the proposed method. 
- The paper provides theoretical insights into why accurate covariance estimation is crucial for unbiased conditional generation, which is supported by experimental results on inverse problems.

### Weaknesses
- The experiment section is not as strong as methodology section: only one dataset and one ODE denoiser is tested under low NFE setting. The design of the experiment and the corresponding results analysis can be improved. 
- Only one ODE sampler (Heun) with steps=15, 30 is tested. While indeed results show the importance of accurate covariance estimate, it’s hard to conclude that this is applicable to all standard diffusion models.

### Questions
- The authors assume that the observation model p(y|x0) is linear gaussian. Is there any other real world problems can be expressed in this form? 
- Continue with weakness section: Can you provide more experimental results? 
For example, does the performance changes when using more Heun steps? does the improvement also happen to other type of discretization methods, e.g. Euler? Also there are many other ODE samplers and SDE samplers, if you change the ODE samplers with SDE ones, does the introduced stochasticity make any difference when using your covariance estimate method?

### Soundness
3

### Presentation
2

### Contribution
2
