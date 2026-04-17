# Generative Modeling with Bayesian Sample Inference

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
We derive a novel generative model from iterative Gaussian posterior inference. By treating the generated sample as an unknown variable, we can formulate the sampling process in the language of Bayesian probability. Our model uses a sequence of prediction and posterior update steps to iteratively narrow down the unknown sample starting from a broad initial belief. In addition to a rigorous theoretical analysis, we establish a connection between our model and diffusion models and show that it includes Bayesian Flow Networks (BFNs) as a special case. In our experiments, we demonstrate that our model improves sample quality on ImageNet32 over both BFNs and the closely related Variational Diffusion Models, while achieving equivalent log-likelihoods on ImageNet32 and CIFAR10.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors present a generative algorithm that can be framed as iterative Bayesian inference. They derive an ELBO objective for the model, and show in a series of numerical experiments that they perform competitively with related models such as Bayesian Flow Networks, or Variational Diffusion models, on ImageNet and CIFAR datasets. The appendix carefully discusses the differences with other frameworks, notably diffusion models.

### Strengths
The paper is well written, and technical statements are satisfyingly accompanied with explanatory discussion. The interpretation of the generative model in terms of iterative Bayesian inference is interesting, and the numerical experiments seem convincing, although I have limited expertise in evaluating the latter.

### Weaknesses
My main concern revolves around the novelty of the theoretical generative framework, and how it relates to existing works. The authors already provide a detailed discussion in Appendix A.2 on the differences with standard diffusion models, highlighting how the noising process in their framework is non-Markovian. However, the closely related stochastic interpolant models (see e.g. Albergo et al, Stochastic Interpolants: A Unifying Framework for Flows and Diffusions, 2023) is to the best of my reading not discussed at all. The noising process in the latter seem to the best of my understanding very closely related to that considered by the authors. Notably, the denoising process (1) that involves the posterior denoiser of $x$ given the noisy observation $y$ resembles e.g. (4.3) in the provided reference, notably in the continuous limit of l.170. It is not impossible I have misunderstood some key contributions of the submission, and I am happy to raise my score after clarifications of the authors. However, I cannot be in favor of acceptance unless a detailed discussion is included to clarify differences to this line of works.

### Questions
Please see the weakness section.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Bayesian Sample Inference (BSI), a novel generative model that frames the generation process as iterative Bayesian inference. The core idea is to treat a data sample as an unknown variable and to iteratively refine a belief about it, starting from a broad prior and updating it based on noisy measurements of model predictions. The authors derive an Evidence Lower Bound (ELBO) for both finite and infinite-step versions of this process, enabling likelihood-based training. Through this framework, they show that Bayesian Flow Networks (BFNs) are a special case of BSI, providing a new perspective on them. Experiments on ImageNet32 and CIFAR10 demonstrate that BSI achieves sample quality (FID) superior to both Variational Diffusion Models (VDM) and BFNs, while maintaining equivalent log-likelihoods (BPD). The significance of this work lies in its introduction of a conceptually simple and elegant Bayesian perspective on generative modeling that unifies existing models and proves to be empirically effective.

### Strengths
*   **Novelty:** The primary strength of the paper is its novel framing of generative modeling. Viewing sample generation as the process of inferring a fixed but unknown variable via iterative posterior updates is a conceptually clean and powerful idea. This Bayesian perspective is a welcome contribution, offering a different way to think about the gradual refinement process that characterizes modern generative models.

*   **Theoretical Soundness:** The paper provides a rigorous theoretical foundation for BSI. The derivation of the ELBO (Theorems 3.1 and 3.2) is solid and connects the model to the established practice of variational inference. A key contribution is showing that BFNs emerge as a special case of BSI when the initial prior is deterministic (infinite precision). This not only unifies BFNs within a more general framework but also simplifies and clarifies their connection to diffusion models. The analysis of the "noising" process (Section A.2) and how BSI's non-Markovian forward process differs from BFN's Markovian one is insightful.

*   **Strong Empirical Results:** The experimental validation, though limited in scope, is compelling within its domain. The authors compare BSI against two highly relevant and strong baselines: VDM and BFN. On the ImageNet32 benchmark, BSI achieves a statistically significant improvement in sample quality (FID of 8.9 vs. 9.9 for VDM and 11.0 for BFN) while matching their performance on log-likelihood. This is a strong result that demonstrates a clear advantage of the proposed formulation.

### Weaknesses
Despite its elegant formulation and promising results I still have doubts on the methodology and experimental validation, which raise several critical questions that weaken its overall contribution. The claims of novelty and superiority feel overstated upon closer inspection of the underlying algorithm and the lack of crucial ablation studies.

1.  **Conflation of Re-framing with Algorithmic Novelty:** The core training algorithm (Algorithm 2) is structurally almost identical to standard denoising diffusion models. It samples a noise level (`λ`), creates a noisy input (`μ_λ`), and trains a network `f_θ` to predict the original sample `x` from `μ_λ`. While the Bayesian derivation is new, the final algorithm is not. This raises the question of whether BSI is a fundamentally new *class* of models or a well-motivated re-derivation of a specific noise schedule and prior for existing models.

2.  **Lack of Ablation Studies:** The paper's central claim is that BSI's performance advantage over BFN comes from its non-deterministic prior (`p(μ₀)` with `λ₀⁻¹ > 0`). This is a critical claim that is not substantiated with sufficient evidence. The paper only compares the two extremes: BSI (`λ₀⁻¹ = 100`) and BFN (`λ₀⁻¹ = 0`). A proper scientific validation would require an ablation study showing how FID and BPD vary across a range of `λ₀⁻¹` values to demonstrate a clear trend and justify the chosen hyperparameter. The same applies to other key design choices, such as the `p(λ)` proposal distribution.

3.  **Questionable Experimental Comparison:** The main empirical comparison is between BSI, BFN, and VDM using a DiT-L-2 backbone. Figure 2 clearly shows that the input distributions for these models at the highest noise levels are qualitatively different. BSI and BFN start from (near) pure noise, whereas VDM's noisiest input mean (`Np(0.08x, 1)`) still contains significant structural information about the image. The DiT architecture, with its global self-attention mechanism, is particularly well-suited to reconstructing images from unstructured noise. This setup may be unfairly biased towards BSI/BFN and confounding the architectural strengths with the merits of the generative formulation.

4.  **Disconnect Between Training Objective and Sampling:** The model is trained to optimize an "infinite-step" ELBO, which is a continuous integral over all precision levels `λ`. However, sampling is always performed in a finite, discrete number of steps. The paper does not provide a theoretical justification for why optimizing the continuous objective is ideal for the discrete sampling task. This is particularly relevant for the low-step sampling regime, which is crucial for practical efficiency but not explored in the paper.

### Questions
*   **Question 1:** The final training objective (optimizing `L_M`) amounts to a noise-conditioned denoising task, structurally similar to other diffusion models. Beyond the novel derivation, what is the fundamental algorithmic difference between BSI and a VDM/BFN where the noise schedule is designed to match BSI's effective `λ` distribution and a non-zero initial noise level is used?

*   **Question 2:** The choice of a log-uniform proposal distribution for importance sampling is motivated by an approximation where `f_θ(μ, λ) ≈ μ`. How does the validity of this approximation vary across precision levels `λ`, and have you investigated the sensitivity of training stability and final performance to this specific choice of `p(λ)`? Could a poorly chosen `p(λ)` mask or artificially inflate the model's true performance?

*   **Question 3:** The comparison with VDM is positioned as a key result. However, Figure 2 shows that the nature of the noisiest inputs for VDM and BSI are qualitatively different. Could the superior FID of BSI be an artifact of the DiT architecture being better suited to BSI's noise distribution and prior, rather than an inherent advantage of the BSI framework itself? How would the models compare with an architecture less biased towards global attention, like a pure U-Net on the same task?

*   **Question 4:** The central claim for BSI's improved FID over BFN is the use of a non-deterministic initial belief (`λ₀ < ∞`). However, this is only validated by comparing the two extremes. The paper lacks a proper ablation study on the effect of the initial noise variance `λ₀⁻¹`. How does the FID score change as `λ₀⁻¹` is varied between 0 (BFN) and 100 (BSI)?

*   **Question 5:** Training is performed by optimizing the infinite-step ELBO (`L_M`), which involves a continuous integral over precision `λ`. Sampling, however, is a discrete, finite-step process. What is the theoretical justification for this being the optimal training strategy for finite-step sampling, and how does this approach affect performance in the very low-step sampling regime (e.g., < 20 steps)?

*   **Question 6:** The training objective, which minimizes the squared error between the true sample `x` and the model's prediction `f_θ(μ_λ, λ)`, bears a strong resemblance to the denoising score-matching objective. Can the BSI objective be formally interpreted as a form of score matching? Specifically, how does the BSI update rule relate to the updates in score-based generative modeling, and could this connection provide further insights into the model?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Bayesian Sample Inference (BSI), a generative modeling framework that treats generation as iterative posterior inference on an unknown sample using a sequence of prediction and Bayesian updates . The authors derive an ELBO in both finite-step and continuous limits, enabling likelihood-based training with a variance-reduced estimator via log-uniform importance sampling. They analyze relationships to Variational Diffusion Models (VDM) and Bayesian Flow Networks (BFN), showing BFN as a special case and connecting BSI to diffusion via a (generally) non-Markov forward process . Empirically, BSI matches likelihoods and improves FID on ImageNet32 vs. VDM/BFN using the same DiT backbone, with additional CIFAR-10 likelihood comparisons using a shared U-Net .

### Strengths
- Clear probabilistic formulation with closed-form update and a principled ELBO; continuous-limit bound is neat and intuitive.

- Concrete connection to VDM/BFN; proof that BFN is a limit case and discussion of Markov vs. non-Markov forward processes add theoretical clarity.

- Variance reduction via log-uniform sampling is motivated analytically and validated empirically.

- Competitive or better results on ImageNet32 (lower FID at matched likelihoods) and matched BPD on CIFAR-10 with controlled architectures/seeds.

### Weaknesses
- Empirical scope is narrow: small-resolution datasets (ImageNet32, CIFAR-10); no high-res, text-conditioned, or scaling-law study to substantiate broader quality claims.

- The theoretical comparison to VDM emphasizes simplicity of the BSI/BFN update, but practical stability/accuracy trade-offs vs. VDM’s log-space parameterization are not quantified.

### Questions
- The proposed Bayesian perspective vs. VDM/BFN: what concrete theoretical advantages does BSI deliver (e.g., tighter/cleaner ELBO, better conditioning, robustness) beyond interpretability? Please make the comparison formal.

- It would be important to support the argument that BSI has higher generation quality than VDM and BFN if there are larger scales such as higher resolutions(e.g., ImageNet-256 and ImageNet-512) or experiments on scaling laws. Please provide results generated with larger resolution or text conditions, and performance trends as the model/data/number of steps increases.

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
4

### Summary
The paper introduces Bayesian Sample Inference (BSI), a generative framework that reformulates sample generation as iterative Gaussian posterior updates. The proposed model uses a sequence of prediction and posterior update steps to iteratively narrow down the unknown sample starting from a broad initial belief. The paper include an extensive theoretical analysis along with empirical evaluation on the ImageNet 32x32 and CIFAR-10 datasets.

### Strengths
The paper provides a novel perspective on generative modeling that frames sample generation as iterative Bayesian inference.
• The proposed BSI framework is quite general, and the paper shows that BSI includes BFN (“Bayesian Flow Networks”) as a special case.
• The includes a succinct and easy to understand comparison between the BSI and BFN / VDM frameworks.

### Weaknesses
• The proposed BSI only leads to minor improvements over the BFN and VDM models. In fact, the loglikelihood BPD metric is nearly identical compared to the BFN and VDM models on both the ImageNet 32x32 and CIFAR-10 datasets.
• The paper only considers a small set of baselines – BFN and VDM – on ImageNet 32x32 and CIFAR-10. The paper should also compare to state of the art models such as “Improved Denoising Diffusion Probabilistic Models, arXiv 2025”, “Neural Diffusion Models, arXiv 2024”.
• The paper only considers the low-resolution ImageNet 32x32 and CIFAR-10 datasets. The paper should provide results on the higher resolution ImageNet 64x64 dataset, as provided in the VDM (“Variational Diffusion Models”) paper.
• The paper does not provide any analysis of the number of steps required for convergence by the proposed BSI model compared to the BFN and VDM models. In Figure 2, the proposed BSI model requires more steps compared to the BFN and VDM models to converge.
• The difference between the proposed BSI model compared to the BFN model is minor. The effective difference is only in the choice of the non-deterministic hyper-prior $p(\mu_0)$ in BSI.

### Questions
• The advantage of the proposed BSI framework over the BFN/ VDM frameworks should be described in more detail.
• The choice of baselines should be motivated in more detail.
• The paper should include more results on higher resolution image datasets, e.g., ImageNet 64 x 64.

### Soundness
3

### Presentation
3

### Contribution
3
