# Observation-Guided Diffusion Probabilistic Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5

## Abstract
We propose a novel diffusion model called observation-guided diffusion probabilistic model (OGDM), which effectively addresses the trade-off between quality control and fast sampling. Our approach reestablishes the training objective by integrating the guidance of the observation process with the Markov chain in a principled way. This is achieved by introducing an additional loss term derived from the observation based on the conditional discriminator on noise level, which employs Bernoulli distribution indicating whether its input lies on the (noisy) real manifold or not. This strategy allows us to optimize the more accurate negative log-likelihood induced in the inference stage especially when the number of function evaluations is limited. The proposed training method is also advantageous even when incorporated only into the fine-tuning process, and it is compatible with various fast inference strategies since our method yields better denoising networks using the exactly same inference procedure without incurring extra computational cost. We demonstrate the effectiveness of the proposed training algorithm using diverse inference methods on strong diffusion model baselines.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes observation-guided diffusion probabilistic model (OGDM), which essentially incorporates an additional GAN component that tries to further match the simulated (via ODE) distribution and true (noise injection) distribution. The proposed method accelerates inference speed while maintaining high sample quality by altering only the training procedures of diffusion models. Empirically, it demonstrates benefits across datasets and diffusion baselines.

### Strengths
* The authors' proposed method OGDM is novel to my knowledge. There were existing works that also explored utilizing a discriminator to help diffusion model training/sampling, but this paper's proposal is different from those.
* Empirically, the proposed method demonstrates nontrivial benefits when incorporated into various baselines, especially with few NFEs.

### Weaknesses
1. There is no empirical comparison between the proposed method and the existing diffusion works that utilize a discriminator. Currently, the paper only compares with the "vanilla" diffusion training baselines. As we have already seen from prior works, incorporating a GAN component into diffusion models could improve the empirical results by a lot, I would highly suggest the authors do so, e.g. [1], [2], to give a clear picture of where this method stands against its peers.
2. The presentation of the paper feels unnecessarily complicated at times. In my opinion, for example, Fig.2, Sec.3.1, and 3.2 are quite distant from the presented method in Sec.3.4. $y$ which at first is defined to be some observation of the underlying diffusion process, but only to materialize as true or fake label. I suggest the authors make the story more straightforward, like the related work [1] and [2].
3. I find Sec.3.5 lacks motivation and clarity, and does not contribute to the overall paper. Essentially the whole subsection is built upon an unreal assumption: for any $\beta$, $p_{u|v}$ can be approximated by some weighted geometric mean of two distributions. The authors claim to provide validity of the approximation, but only to analyze a toy example where the base distribution is Gaussian.


[1] Xiao et al. "Tackling the generative learning trilemma with denoising diffusion gans." ICLR 2022.

[2] Kim et al. "Refining generative process with discriminator guidance in score-based diffusion models." ICML 2023.

### Questions
1. In the introduction, the paper talks about diffusion models with DDPM's formulation with "thousands of NFEs" and "reverse Gaussian assumption", but only presents experiments with fast deterministic samplers, where these issues do not arise. Related to this: in Sec.3.5, are you suggesting you are working with a stochastic sampling process that is fast? (You mentioned fast sampling, and also each reverse step is defined as a Gaussian so I assume so.) This is not standard practice nor what is considered in the experiments.
2. In Eq.18, how do you get $x_{t-s}$? Just add a smaller noise to data, which corresponds to one with time step $t-s$?
3. For Eq.25 and the associated explanation, I do not think your method considers the second KL term, but rather it considers the KL between the marginals of $p$ and $q$ at time $\tau_{i-1}$. In other words, the discriminator is not aware of $x_{\tau_i}$ in your formulation.
4. Could you explain why the discriminator needs both $t$ and $s$ as input, rather than just $t-s$? If $t_1-s_1 = t_2-s_2$, then $q(x_{t_1-s_1})$ and $q(x_{t_2-s_2})$ should have the same marginal, no?
5. I suggest the authors provide training overhead (if trained from scratch) of the proposed method.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new training objective for diffusion models. Compared to the original DDPM framework, an additional term is added into the loss function, with the goal of fooling a discriminator to predict the image after one-step denoising as real image. The proposed objective is grounded in a framework where a sequence of observations are added into the original forward and backward process of DDPM. And each observed variable $y_t$ is defined as whether the associated image $x_t$ is from the real image distribution or not. Experiments show that models trained with the proposed objective (both trained from scratch and fine-tuned from pre-trained diffusion models) outperform the baselines in terms of FID and recall, especially when the number of denoising steps is small.

### Strengths
1. The paper proposes a new training objective for diffusion models that is theoretically grounded.
2. The paper did comprehensive experiments on three datasets of various resolutions, using various sampling algorithms.

### Weaknesses
1. The proposed training pipeline is coupled with a specific sampling algorithm. At inference time, when the sampling algorithm is changed to another one that is different from the one used during training, the proposed method has limited improvements compared to baselines, as demonstrated in Table 3. This limits the applicability of the method, since if new sampling algorithm is proposed, the diffusion model also needs to be re-trained.
2. Comparison with important baselines are missing. Specifically, the discriminator guidance [1] should also be compared, since it also utilizes a discriminator, and it can be applied at inference time without re-training diffusion models. In particular, the reported performance in Table 2 seems to be worse than the number reported in [1].
3. The cost for training diffusion models should also be reported.

[1] Kim et al., 2023. Refining generative process with discriminator guidance in score-based diffusion models.

### Questions
1. Why is the projection function $f(\cdot)$ defined as the one-step denoising of the diffusion model?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new training method for the diffusion models, where a discriminative loss is designed to reduce the sampling steps for inference. Experiments on several datasets show the proposed method performs better than baselines.

### Strengths
1. The paper is well written and clear.
2. The paper presents detailed theoretic analysis for the proposed method.
3. Experiments on several datasets show the effectiveness of the proposed method.

### Weaknesses
1. The paper introduces additional cost for training, but there is no additional training cost analysis.
2. The advantage of diffusion models compared to GAN is the training stability, it introduce GAN training again, which may harm the training stability.
3. The experiments are conducted on unconditional generation, leaving its performance on the mainstream text-to-image generation models unclear.

### Questions
Please provide the training memory and time cost, and better to provide its performance for Pretrained conditional text-to-image models such as Stable Diffusion.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
