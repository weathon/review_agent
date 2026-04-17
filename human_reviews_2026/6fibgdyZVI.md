# One-shot Conditional Sampling: MMD meets Nearest Neighbors

- Decision: Reject
- Scores: 4, 8, 6, 4

## Abstract
How can we generate samples from a conditional distribution that we never fully observe? This question arises across a broad range of applications in both modern machine learning and classical statistics, including image post-processing in computer vision, approximate posterior sampling in simulation-based inference, and conditional distribution modeling in complex data settings. In such settings, compared with unconditional sampling, additional feature information can be leveraged to enable more adaptive and efficient sampling. Building on this, we introduce Conditional Generator using MMD (CGMMD), a novel framework for conditional sampling. Unlike many contemporary approaches, our method frames the training objective as a simple, adversary-free direct minimization problem. A key feature of CGMMD is its ability to produce conditional samples in a single forward pass of the generator, enabling practical one-shot sampling with low test-time complexity. We establish rigorous theoretical bounds on the loss incurred when sampling from the CGMMD sampler, and prove convergence of the estimated distribution to the true conditional distribution. In the process, we also develop a uniform concentration result for nearest-neighbor based functionals, which may be of independent interest. Finally, we show that CGMMD performs competitively on synthetic tasks involving complex conditional densities, as well as on practical applications such as image denoising and image super-resolution.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a conditional sampling method based on maximum mean discrepancies. To this end, the authors introduce a conditioning variable into MMD-GANs and minimize the expected MMD of the approximated and true posterior distribution. Here, the MMD is discretized based on a nearest neighbor search. From a theoretical side the paper proves a bound how the restriction of the generator to be a neural network effects the minimal objective value and shows some convergence result of the discrete estimator. Numerical examples are performed on MNIST and CelebA for superresolution and denoising.

### Strengths
The paper is generally sound in the sense that the generative model actually minimizes a distance to the target which can generally be evaluated. Also it considers the approximation class of neural networks instead of just assuming that neural networks are universal function approximators.

### Weaknesses
Major:

- Literature: Conditional sampling or generative modeling by directly minimizing the MMD have previously been considered in the context of gradient flows in [1,2,3,4]. In particular [2] considers the exact same problem and proves corresponding error bounds. The authors should rework the literature part and maybe compare to [2].

- Generally the optimal sampler $\bar g$ is neither unique nor regular. Plenty of works show that the Lipschitz constant (and thus also the modulus of continuity) must explode whenever we approximate multimodal or heavy-tailed distributions, see e.g. [5,6]. In practice, the bound from Thm 4.1 will be very very loose. Also, the notation is a bit confusing as the definition of $\omega_{\bar g}$ skips one of the arguments of $\bar g$ (I guess it is the conditioning which is omitted here).

- There is a significant mismatch between minimizing the MMD and its empirical estimators, in the sense that one needs a very large batch size in order to approximate the MMD precisely. Using a minibatch approximation usually yields blurred results which can be seen very well in Figure 2 where the CGMMD produces plenty of samples outside of the support of the target distribution compared to the baselines. A discussion on this topic is missing.

- Experiments: Compared to the literature, the paper only considers quite small experiments. I am not sure, whether denoising or superresolution tasks on MNIST really make sense as it is a really simplistic dataset. The quality of the results seems to be below classical image reconstruction methods like total variation. Also, for this kind of task the evaluation of FID and inception score does not make sense, as they have a predefined ground truth and metrics like PSNR, SSIM or LPIPS can cover them much better (I noted that PSNR and SSIM are reported, but my point is that it doesn't really make sense to report FID and IS in this case). Given that the method matches the distributions of a generator with a data distribution, the authors should evaluate how much generative power their model has. Suitable tasks could be class conditional sampling or box-inpainting.

Minor: 

- Typo: In Thm 4.1 in the definition of $\omega_{\bar g}$ there is missing the bar within the sup.

- I would prefer if the authors sharpened their Contributions section a bit to showcase what is new and what is taken from the literature. If I understand it correctly, the definition of the ECMMD and its empirical computation via neighborhood graphs comes from the literature, while the contribution in this paper is the construction of a conditional MMD GAN.

- Please state some relevant information about the experiments in Section 5. E.g. which kernel is used, which batch size etc.

- Citations are not always accurate. For instance the fact that $P_{Y|X}=P_{Z|X}$ almost surely iff $P_{Y,X}=P_{Z,X}$ (line 97 to 101 and Lemma 2.1) is simply Bayes theorem and that the corresponding metrics are zero is an immediate consequence and existance of the mapping $\bar g$ already follows from Breniers theorem; the citations in the paper are far away from being the origin of these results. Similarly, the form of the MMD in (2.2) exists for decades in the literature and attributing it to Chatterjee et al. is inaccurate.

---

[1] Maximum mean discrepancy gradient flow, NeurIPS 2019

[2] Posterior Sampling Based on Gradient Flows of the MMD with Negative Distance Kernel, ICLR 2024

[3] Generative Sliced MMD Flows with Riesz Kernels, ICLR 2024

[4] Deep MMD gradient flow without adversarial training, ICLR 2025

[5] Tails of Lipschitz triangular flows, ICML 2020

[6] Can push-forward generative models fit multimodal distributions?, NeurIPS 2022

### Questions
How much computational power is required for training (which GPUs, how long does it take, how much GPU memory is required etc.)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Conditional Generator using Maximum Mean Discrepancy (CGMMD), a new adversary-free framework for one-shot conditional sampling. The method trains a conditional generator $g(\eta,X)$ by minimizing the expected MMD between the conditional distribution and the model distribution.

The authors prove both the non-asymptotic finite-sample error bounds and asymptotic convergence of the learned conditional distribution to the true one in expected MMD.

Last, the authors provide empirical results on synthetic data and image tasks (denoising and super-resolution) demonstrate good performance and speedups over conditional diffusion models.

### Strengths
1. **Bridging theory and practice**: the paper not only demonstrate the performance of the proposed CGMMD empirically, but also provide theoretical analysis on the loss convergence, deepening the theoretical understanding of the method.

2. **Broader relevance**: The analysis of nearest-neighbor-based concentration inequalities could be useful beyond this work, potentially benefiting other nonparametric or kernel-based generative models.

### Weaknesses
1. **Limited comparison to diffusion-based models**: diffusion with CFG is one of the most popular conditional generation methods nowadays. I feel more comparison to it should be provided, especially on the generation quality and generation time on other larger-scale datasets.

2. **Lack of illustration on parameter setting in numerical experiments**: it is not clear how the parameters/kernels/$k_n$ are chosen in the experiments.

### Questions
1. How does different kernels and $k_n$ affect the performance of CGMMD?

2. In the synthetic data experiments (Figure 2), CGMMD seems to tend to underestimate the variance and concentrate on the center. Does similar phenomenon happen in other experiments? Is there any intuition behind this kind of generation bias for CGMMD?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents an alternative to diffusion and GANs for a generative model. Their method is based on MMD, although using MMD is a well-known method, they consider the conditional setting where they employ a squared MMD distance. The authors demonstrate that by doing this, they can leverage a kernel $K$ and then use a nearest neighbors algorithm to efficiently estimate the squared MMD. Notably, their method produces a one shot generator and the authors demonstrate their effectiveness on synthetic datasets as well as inverse problems like super-resolution and image denoising.

### Strengths
- The paper presents a clear method for one-shot generative models based on mathematical foundations
- The paper presents a simple way of approximating their objective with a k-nearest neighbor approach
- The paper has good theoretical results to justify the convergence of their method

### Weaknesses
- The presentation could be improved, for instance, currently the introduction already contains part of the background, which makes it harder to understand the main methodology
- There are no experiments on generative modeling beyond MNIST, which makes it hard to assess its utility in such a setting
- Although there is a comparison with Diffusion models, demonstrating that the existing method can be significantly faster. I believe a more interesting comparison would be comparing against distilled diffusion models such as consistency models, consistency-trajectory models or shortcut models. As they can significantly accelerate the sampling

### Questions
- Can the authors provide a result for a pure generative model outside of MNIST, which is currently considered to be too simple of a task?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes CGMMD, a conditional generator trained by directly minimizing an Expected Conditional MMD (ECMMD) loss estimated with a k-nearest-neighbor graph over the conditioning variables. This yields a one-shot conditional sampler: draw noise $\eta$ and evaluate $\widehat{g}(\eta, x)$ in a single forward pass, avoiding adversarial training and diffusion’s iterative denoising. The authors give a finite-sample generalization bound for the learned sampler and a convergence result (in MMD and via characteristic functions). Experiments span synthetic helix-shaped conditionals and image tasks (MNIST super-resolution/denoising, CelebHQ denoising); compared with a conditional diffusion baseline, CGMMD is ~100× faster per image but with lower perceptual quality. Algorithmic details and the kNN ECMMD estimator are spelled out.

### Strengths
- Clear objective, no min–max: Framing training as direct ECMMD minimization sidesteps GAN instability and mode collapse while preserving flexibility of neural generators.

- One-shot conditional sampling: Practical speed advantage versus diffusion; single forward pass at test time. Table shows orders-of-magnitude faster generation.

- Theory with modern tools: Non-asymptotic error bound and convergence guarantee; analysis combines uniform concentration for kNN functionals and NN approximation theory.

- Sound choice of discrepancy: ECMMD is a strict scoring rule for conditionals, grounding the objective in kernel mean embeddings. This aligns with prior work formalizing ECMMD. 

- Positioned in the literature: Connects thoughtfully to MMD foundations and conditional generative approaches (GAN/OT/diffusion).

### Weaknesses
- Assumption load and scope: Theory requires sub-Gaussian X, uniqueness in kNN graphs, bounded Lipschitz characteristic kernels, and a nontrivial regularity condition on conditional mean embeddings. This is reasonable but limiting in heavy-tailed or structured-X regimes.

- Batching. How is the data being generated? In each epoch, do we get a fresh sample of $y$s? And how did you get kNNs to work on mini-batches?

- Estimator design choices left open: Performance will depend on kernel, bandwidth, and k; guidance for tuning/robustness and sensitivity analyses are minimal. (MMD performance is known to be kernel-dependent.) 

- Empirics are narrow: Image tasks are small-scale; diffusion outperforms on PSNR/SSIM/FID despite CGMMD’s speed, and baselines omit strong conditional flows/normalizing flows which are fairly common. 

- Scalability details: kNN graphs each batch add overhead and may complicate very large-scale or high-dimensional X; no discussion of approximate NN structures etc to accelerate ECMMD.

- Comparative framing: The diffusion baseline uses classifier-free guidance but lacks careful parity in compute/architecture; more apples-to-apples studies (e.g., distilled diffusion or faster samplers) would help.

### Questions
If you can address the questions I put in the weaknesses section, I'd appreciate it.

Overall, I think the main weakness of the paper is that there isn't a real apples-to-apples experimental result. You've only compared to the original diffusion algorithm (which is known to be slow) and in cases where generating conditional samples of $y|x$ is easy. There's been so much work on consistency models, progressive distillation, faster samplers, and it's largely been ignored. The low resolution data is also not convincing.

### Soundness
3

### Presentation
2

### Contribution
2
