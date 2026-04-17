# Continuously Augmented Discrete Diffusion model for Categorical Generative Modeling

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Standard discrete diffusion models treat all unobserved states the same way, typically mapping them to an absorbing [MASK] token. This creates an "information void" where global semantic information that may be inferred for the masked tokens from the unmasked tokens is not directly passed from one denoising step to another.  We introduce **Continuously Augmented Discrete Diffusion (CADD)**, a framework that augments the discrete state space with a paired diffusion in a continuous latent space. This yields graded, gradually corrupted states in which masked tokens are represented by noisy yet informative latent vectors rather than information voids. At each reverse step, CADD uses the continuous latent as a semantic hint to guide discrete denoising. The design is clean and compatible with existing discrete diffusion training. At sampling time, the strength and estimator of the continuous latent vector enables a controlled trade-off between mode-coverage (diversity-oriented) and mode-seeking (context-localization-oriented). Empirically, we demonstrate CADD improves generative quality over mask-based diffusion across text generation, image synthesis, and code modeling, with consistent gains on both qualitative and quantitative metrics against strong discrete baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper tackles a core problem in discrete diffusion models: all states collapse to a single absorbing state without retaining meaningful information. Inspired by the continuous diffusion model, the authors propose continuously augmented discrete diffusion (CADD), a hybrid framework that augments the standard masking process paired diffusion process in a continuous latent space. CADD utilizes a noisy continuous vector as a semantic condition to guide the discrete desnoising step. Experimental results across three distinct modalities demonstrate the effectiveness and versatility of CADD.

### Strengths
- The idea that combines the discrete diffusion process with the continuous semantic latent to resolve the information void problem is novel and seamlessly integrated into the training and sampling pipeline. Furthermore, the lower bound of the objective is well-justified.

- The paper's motivation and core concept are clearly presented, and the visualizations in Figures 1 and 2 are also clear.

- The effectiveness of CADD was validated across three distinct tasks, consistently demonstrating improvement over conventional discrete diffusion models. The paired continuous latent vector effectively resolves the 'information void' problem, which is clearly reflected in the scalable generation quality as sampling steps increase.

### Weaknesses
- While compatible with MDM training, the CADD framework is inherently more complex. It requires managing two parallel diffusion samplers and fusing their representations at every step.

- The authors opt to use only the standard discrete cross-entropy loss for simplicity, omitting the continuous MSE loss that would naturally arise from the ELBO. While this works well empirically, the paper could be strengthened by either (a) showing that adding the MSE loss provides no benefit (justifying its omission) or (b) exploring the full loss term, which might lead to even better performance.

- While CADD shows superior capabilities with large sampling steps, the performance seems to degrade sharply with a few-step sampling, probably due to the mode covering problem of the continuous semantic latent. This could amplify the inefficiency of the discrete diffusion models, which already suffer from large NFEs.

### Questions
- Modern visual generative models mostly utilize vector-quantized (VQ) visual tokens for image synthesis with complex conditioning (such as text prompts). As this paper primarily searches for unconditional visual generation with discrete pixel space, could CADD be impacted by the complex VQ tokens and conditioning?

- Masked Generative Models are a family of discrete diffusion models with an absorbing state, while they show superior efficiency [1][2]. Can the CADD framework be applied as a drop-in augmentation to MaskGIT-style models? If so, what is the expected trade-off in terms of sample quality/diversity?

[1] Maskgit: Masked generative image transformer
[2] An Image is Worth 32 Tokens for Reconstruction and Generation

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper is motivated by the information loss when transforming clean tokens into mask tokens in MDMs. To address this issue, CADD augments each token with an informative continuous latent vector, which can provide more information on the original tokens. I find this idea is quite novel and can be viewed as the most important contribution of this paper.
The authors discuss the training and sampling method of CADD and validate it on image, text, and code generation tasks.

### Strengths
- I find the idea of augmenting MDMs with continuous latent vectors to provide semantic information is both novel and well-motivated.
- The empirical evidence is strong to support the effectiveness of the CADD method.
- The paper presents a thorough derivation for training objective (ELBO) and sampling method, making the methodology rather practical.

### Weaknesses
- The authors use a learnable network $w_\theta$ which takes sequence $x$ as input and outputs the latent vector $z$. However, in my opinion, a more straightforward and robust way is directly assigning a learnable latent vector for each token, and all the parameters forms a "codebook". Will this parameterization works in practice?
- In line 5 of Algorithm 1, the $w_\theta$ takes the entire $x$ as input, and in line 10 of Algorithm 2, the $w_\theta$ takes a single token $x^i$ as input. Which one is the correct expression?
- The authors should discuss how they implement the  $w_\theta$ in practice. Moreover, as they introduce an additional model, a comparison on memory and computation time should be provided.
- I suggest that the authors to provide more experimental results. For example, the test perplexity and zero-shot perplexity can be a more  common benchmark for unconditional text generation. I am also interested in how the FID score on CIFAR10 scales on different choices of NFEs?

### Questions
- The current methodology of discrete diffusion almost converges to masked diffusion models. However, regarding the information void issue of MDMs, can this issue be alleviated by discrete diffusion with a uniform distribution noise?
- How do you compute your NFE? (e.g. in Algorithm 2)

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents Continuously Augmented Discrete Diffusion (CADD), a framework which combines the strengths of Discrete Diffusion Models and Continuous Diffusion Models. CADD retains the discrete masking strategy and uses continuous latent vectors as the tokens get masked, to maintain semantic information. In essence, two diffusion processes happen at the same time. Experimental results show that CADD models consistently generate higher quality samples over the baselines in three tasks: text, image and code generation.

### Strengths
* The paper is well written, complete and easy to follow.
* It bridges a known gap and is conceptually novel. This specific combination of discrete diffusion with continuous latent vectors when the tokens are masked had not been previously explored.
* The idea is technically sound and the formulation of CADD is coherent. The factorization q(x_t​,z_t​ | x_0​)=q(x_t ​|x_0​)q(z_t​|x_t​,x_0​) provides a clear way to combine the discrete and continuous variables. 
* There is strong empirical validation. Results are presented for 3 different tasks, and CADD offers significant improvement over the baselines.

### Weaknesses
* The discrete and continuous reverse process are derived in Section 4, but in the experiments the continuous MSE term is not used. This means that the continuous part is never directly trained to match the distribution, just guided. Although z_t appears to carry meaningful information, this makes me question how including the MSE would impact z_t and overall performance.
* CADD maintains an extra latent vector per masked token, when compared to standard Discrete Diffusion Models. The authors should discuss whether this results in extra computational or memory cost.
* In the text generation experiments, the authors do not compare CADD against any continuous/hybrid diffusion model, such as Duo.

### Questions
1. What happens if you include the MSE loss? How does this affect z_t and performance?
2. Does CADD incur extra computational cost vs discrete diffusion models? If so, is it significant?
3. Did the authors test any alternative schedules for the continuous process?
4. In the main experiments for text generation, why isn’t Duo a baseline?
5. It could be interesting to include interpretability analysis regarding z_t, to visualize examples where the latent hints help disambiguate the tokens.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Continuously Augmented Discrete Diffusion (CADD), a framework that combines Masked Discrete Diffusion (MDM) with a parallel continuous diffusion process in the embedding space. The key innovation is to replace the "information void" of a [MASK] token with a noisy but semantically informative continuous latent vector. This provides "soft hints" during the denoising process, leading to more coherent and higher-quality generation across text, image, and code modalities. The method is shown to be effective, scalable, and simple to implement.

### Strengths
The paper target at solving a limitation of existing MDMs, which is the "information void" created by the absorbing [MASK] state. The idea of augmenting the discrete space with a continuous one to preserve graded semantic information is both intuitive and powerful.

The paper provides extensive experiments across multiple modalities (text, image, code) and scales (from 168M to 7B parameters). Consistent gains are observed over baselines.

The paper is well-written.

### Weaknesses
Time cost of model running can be analyzed. For example, a time cost and FLOPs comparison between CADD and the baselines can be helpful for the readers to understand the performance-efficiency trade-off.

The experiment comparison is presented mainly with statistical result. It lacks illustrative example to show how the latent space parameter helps improve the generation performance.

Code is not released for reproducibility checking.

### Questions
What is the inference-time latency of CADD (with K=1) compared to a standard MDM baseline when generating a sequence of fixed length? Is the overhead primarily from the extra continuous state operations or from the architectural changes?

The continuous latent z_t is initialized from a standard Gaussian prior at step T. Have you experimented with more informative priors, and if so, what was the effect?

### Soundness
3

### Presentation
3

### Contribution
3
