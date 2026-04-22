# Continuous Diffusion Models with Explicit Score Matching for Highly Efficient Anomaly Detection

- Avg Score: 3.00
- Decision: Reject
- Scores: 0, 4, 8, 0

## Abstract
Diffusion models have proven to be highly effective in generating high-quality reconstructed images, making them ideal for the rigorous requirements of reconstruction-based anomaly detection systems. While continuous diffusion mod- els unify discrete implementations, existing methods predom- inantly rely on denoising score matching (DSM), as directly acquiring explicit scores remains challenging. In this study, we introduce a novel diffusion framework utilizing explicit score matching (ESM) via a dual-stream neural network, trained by maximum likelihood estimation. Based on the first systematic comparison between DSM and ESM paradigms, variance-guided diffusion process is developed to further im- prove the performance. Comprehensive experimental evalua- tions confirm the superior anomaly detection capabilities and computational efficiency of the proposed system. The frame- work’s flexibility allows seamless integration with existing diffusion models, offering a potential pathway for broader ap- plications in generative tasks.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes a diffusion-based anomaly detection framework which the authors refer to as a ``continuous-time explicit score matching (ESM)’’ model, but which is in practice implemented as a discretized DDPM-style denoising network with dual prediction heads for the mean and variance of a Gaussian noise model. The idea is to train these heads by maximum likelihood so that the model can compute an “explicit” marginal score, contrasting this with the denoising score used in standard continuous diffusion training. The authors argue theoretically that DSM coincides with ESM only for the squared loss case, to motivate their design and a variance-guided diffusion procedure that replaces the fixed schedule with the predicted variance for greater stability and fewer steps. At inference, the system reconstructs images and aggregates several residual- and feature-based anomaly scores, reporting strong accuracy with smaller models and reduced sampling. The paper also compares against other methods on MVTec-AD, VisA, and MPDD.

### Strengths
- Good industrial anomaly detection motivation and use of standard datasets. 
- Simple dual-head design ($\mu$, $\sigma^2$) trained via negative log-likelihood (NLL) is easy to implement.
- The paper runs broad experiments and explores multiple anomaly scores showing good performance.

### Weaknesses
- Continuous vs discrete diffusion usually refers to the type of data used in training, the authors are more specifically talking about discrete-time or continuous-time diffusion model. This should be fixed in the title and throughout the paper. The paper writes the correct reverse SDE (Eq. 2) and then “omits the random noise” to present an ODE version (Eq. 5). There is no justification or reference for this removal; it changes the generative semantics and is not the standard Song et al. reverse-time SDE/ODE derivation. This does not make the method a continuous-time diffusion model and the actual method is a discretized DDPM-style sampler with fixed $T_d$, thus, it is not a continuous-time score-based model.

- The paper equates explicit score matching with the identity $\nabla_{x}\log p_t(x)=-(x-\mu_t)/\sigma_t^2$ and calls this “the formula of ESM” (Eq. 3). But this is simply the Gaussian score; it only holds if the marginal p_t(x) is Gaussian, which is not true in general for diffusion marginals (they are data distributions convolved with Gaussians). The expression corresponds instead to the conditional Gaussian score used in denoising score matching (DSM) under the Variance Preserving (VP) diffusion process. True ESM (Hyvärinen, 2005) involves matching the marginal score using the divergence-based objective.

- Eq. (6) defines ESM as matching $s_\theta$ to the marginal score $\nabla\log p_t(x)$ inside an $\ell^l$ loss. This assumes pointwise access to the true marginal score, which is precisely what Hyvärinen’s ESM avoids via the divergence trick. The paper later cites a Vincent-style equivalence for l=2 (Eq. 12), but the preceding derivation is still built on an intractable oracle quantity. 

- The paper “expands” $\|s_\theta-\nabla\log p_t\|^l$ using binomial-style sums with inner products of exponentiated vectors (e.g., $\langle s^{l-k}$, ($\nabla\log p)^k\rangle$). That expansion is not a valid identity for vector norms; it implicitly treats the vector norm to a power as if it were a scalar binomial, which is wrong. These steps underlie Eq. (8) and Eq. (9), so the subsequent comparison built on them is unreliable. 

- Replacing $\beta(t)$ with a predicted, input-dependent $\sigma^2(x,t)$ (Eqs. 16–17) breaks the standard SDE setup where drift/diffusion are time-dependent (or parameter-dependent) but not input-dependent in that way. It’s unclear what distribution is simulated, and arguments are missing.

- Implementation details admit $T_d$ is “manually set for each class,” with tables of hand-picked $T_d$ for each dataset category. This amounts to selecting hyperparameters directly on the test set, effectively overfitting and undermining the validity of their comparisons to baselines that use fixed or globally defined schedules.

- A substantial part of the final score mixes four detectors (cosine/DINO, uncertainty-weighted residual, KL, etc.). It’s unclear how much of the improvement comes from the dual-stream diffusion versus these add-ons. An ablation (Table 5) exists but not every component is individually analyzed. Would a simple diffusion model work as well using this assemble of four detectors?

- The writing quality is poor and often confusing; the flow between sentences is abrupt and lacks coherence. Informal phrasing (e.g. “actually”) appears throughout, which undermines academic tone. Several citations are incorrectly formatted, missing parentheses or proper spacing before the reference marker. Conclusions are often stated without logical grounding in the preceding arguments, creating a sense of unjustified claims rather than reasoned progression.

- The provided supplemental code is poorly structured and difficult to follow; its quality does not meet common research reproducibility or clarity standards.

In summary, the method used is still a discrete-time diffusion model using DSM, which goes against the whole argument of the paper.

### Questions
- $T_d$ was set per class. How was this chosen without test leakage?
- As you claimed an efficient method, what are the gains in training and inference speed compared to other methods beyond the number of timesteps and model size?
- How exactly are the images reconstructed? Is there a forward noising process first? I see it is the case in the code, but it is not explained in the paper.
- What happens if the variance predicted by the model is always zero (or close to zero)? It looks like the model would become a standard DDPM. I suspect this is the case.

Please see weaknesses.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work aims to improve diffusion-based anomaly detection by explicitly learning the score function via a maximum-likelihood dual-stream network. The proposed approach replaces the denoising score with a direct estimate of the log-likelihood gradient and integrates a variance-guided diffusion process to enhance efficiency.

### Strengths
- Clear motivation to bridge DSM and ESM formulations in diffusion modeling.
- Extensive experimental evaluation across multiple datasets.

### Weaknesses
- The ESM vs. DSM comparison is largely restating prior work (Vincent, 2011) and does not offer new theoretical insights.
- Improvements over baselines are marginal (1–2%) and within variance; the claim of “SOTA” performance is somehow overstated.
- No systematic exploration of the role of the number of diffusion steps, or parameter choices.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the use of explicit score matching (ESM) as an alternative to the widely used denoising score matching (DSM) in the context of anomaly detection.

The authors consider the anomaly detection task by computing an anomaly score S via reconstruction (i.e. comparing the input image with its projection into the “normal” domain as by some model) and some additional terms. They first rederive the cases where ESM and DSM are or are not equivalent and motivate the proposed estimation of the variance and the corresponding loss function for the ESM. For the experiments they use standard benchmark datasets like MVTec-AD etc, and show  the performance of the proposed model, but also relevant baselines.

- introduce explicit score matching vs classical Denoising Score Matching
- “Also, the difference between ESM and DSM is investigated, and the experiment results show that there would exist risk of performance downgrade in both accuracy and efficiency by using DSM instead of ESM, which unfortunately is the most common way for recent diffusion models.”
- “However, some fundamental issues remain under-explored, one of which is the usage of DSM(denoising score match) instead of ESM(explicit score matching), as ESM could not be obtained by modern diffusion model”

### Strengths
The authors show that their proposed continuous diffusion model with variance guided ESM outperform the various baselines and that ESM with variance guidance is worth investigating further. They show the performance across multiple standard benchmark datasets where their proposed method shows an impressive performance. The authors promised to publish their code, which is necessary for this work to be reproducible. The authors also show that the proposed method is also more efficient in terms of model size and compute, compared to the baselines.

### Weaknesses
In the method section (3.1) the authors show that for l>2 the terms for ESM and DSM are not equivalent, however there is no motivation why we would even consider anything other than l=2 in the first place. This would in my opinion be necessary to motivate the method, and it should also be reported what “l” is being used in the end.

The anomaly score is composed of various terms, among them the cosine similarity of DINO features, which do not directly seem to have any connection to the proposed method. It would be nice to also show the same ablation but without these, to see what part contributes how much.

In the method section the authors rederive the difference/equivalence between ESM and DSM which follows the work of Vincent as cited. While this context is appreciated for the motivation, it contains (along with the rest of the paper) some typos and inconsistencies that make it hard to follow. For instance, the jump from Eq8/9 to Eq 10/11 does not explain why the absolute value within the expectation can just be ignored. While true for even “l”, it is unclear in general.

### Questions
In the method section (3.1) the authors show that for l>2 the terms for ESM and DSM are not equivalent, however there is no motivation why we would even consider anything other than l=2 in the first place. This would in my opinion be necessary to motivate the method, and it should also be reported what “l” is being used in the end.

The anomaly score is composed of various terms, among them the cosine similarity of DINO features, which do not directly seem to have any connection to the proposed method. It would be nice to also show the same ablation but without these, to see what part contributes how much.

In the method section the authors rederive the difference/equivalence between ESM and DSM which follows the work of Vincent as cited. While this context is appreciated for the motivation, it contains (along with the rest of the paper) some typos and inconsistencies that make it hard to follow. For instance, the jump from Eq8/9 to Eq 10/11 does not explain why the absolute value within the expectation can just be ignored. While true for even “l”, it is unclear in general.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The goal of this paper is to introduce a diffusion model by explicit score matching. This is claimed to be achieved by using a dual stream encoder-decoder to calculate the expected value and variance of the noise at each time-step, thereby directly estimating the noise density without the need to calculate the conditional score via denoising score matching. However, the paper is very poorly written and there are many flaws in the logic, ambiguities in the theoretical framework, and lack of proper experimental validation that render the paper unsuitable for acceptance.

### Strengths
The paper presents an interesting approach in score matching for diffusion models, particularly the denoising (backward) process.

### Weaknesses
The paper has many flaws that need to be addressed for a proper evaluation of the proposed framework. These include improvements in the theory, originality, significance, clarity, and experimental design. Please see more details below:

- From a theoretical framework, the purported claim that current diffusion models have to use the conditional $p(\hat{x}|x)$ because they cannot directly find the standard deviation of the noise probability distribution is not quite true. The real reason for using the conditional is the fact that estimating the partition function (z) does not have a tractable solution. This fact alone calls into question the soundness of the proposed method. Simply reducing the problem into estimating the standard deviation, completely ignores issue with the tractability of the partition function (z). 

- The purported mathematical derivations are also quite unclear. The authors should clearly present the core theoretical problem with clear and concise formulations, prove the effectiveness of their proposed method, and draw clear theoretical conclusions. Most of the math shown in the paper is rather straightforward, without touching on the core issue for the use of conditional probabilities in the denoising process. 

- Even if the theoretical background of the proposed work is sound, the proposed solution is rather incremental. The authors simply replace the encoder-decoder model of the DSM approach by two encoder-decoders and claim that it solves the problem. It is not clear what the original idea is, beyond simply estimating $\sigma$ rather than $\alpha$ in the original DSM method.

- There is also a lack of discussion on the significance of this approach, especially with respect to the contribution is the domain of diffusion models. Beyond very little improvements in accuracy of anomaly detection tasks, what is the major shortcoming of DSM that cannot be achieved by better training, and why this proposed method is needed. There are no empirical or experimental examinations to demonstrate the superiority of this approach over established diffusion models.

- The experimental design of the paper is also severely lacking in various aspects. Firstly, the experiments solely focus on anomaly detection in images. It is not clear if the performance shown in the experiments are directly related to the proposed score estimation or related to the anomaly detection method used. The paper also compares with three other diffusion models without clearly explaining how the other methods are utilized for the anomaly detection tasks. For example, did the authors replace the diffusion process in GLAD or ADSPR with their diffusion process, or are these results the direct end-to-end comparisons in anomaly detection. In addition, several of the reported AUROC values are 100% which hints at potential overfitting. There is no discussion on this issue.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2
