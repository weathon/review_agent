# Contrastive CFG: Guiding Diffusion Sampling by Contrasting Positive and Negative Concepts

- Decision: Reject
- Scores: 0, 4, 6, 4

## Abstract
As Classifier-Free Guidance (CFG) has proven effective in conditional diffusion model sampling for improved condition alignment, many applications use a negated CFG term as a Negative Prompting (NP) to filter out unwanted features from samples. However, simply negating CFG guidance creates an inverted probability distribution, often distorting samples away from the marginal distribution. Inspired by recent advances in conditional diffusion models for inverse problems, here we present a novel method to achieve guidance toward the given condition using contrastive loss. Specifically, our guidance term aligns or repels the denoising direction based on the given condition through contrastive loss, achieving a similar guiding effect to traditional CFG for positive conditions while overcoming the limitations of existing negative guidance methods. Experimental results demonstrate that our approach effectively injects or removes the given concepts while maintaining sample quality across diverse scenarios, from simple class conditions to complex and overlapping text prompts.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper studies the phenomenon that negative prompts (NP) in CFG might push samples away from the negative conditions unboundedly, which severely harms the fidelity of the ground-truth conditional distribution and CFG with positive conditions. To this end, the authors proposes to reformulate the guidance weight through NCE loss. Such an adaptive methodology could alleviate the overemphasis of negative prompts in a milder and more flexible way.

### Strengths
- The design of adaptive guidance weights based on contrastive learning enhance the faithfulness of guided sampling both theoretically and experimentally.
- The whole pipeline is training-free and efficient, which needs almost no additional time cost.
- The experimental results confirms its efficacy compared with NP method.

### Weaknesses
- From theoretical perspective, contrastive learning forces the sample to appear similar to positive ones and different with negative ones, which is exactly the motivation of CFG from Bayesian theory using score functions and classification probabilities. The authors simply use a new probability function $p(x_{t-1}|x_t,c)$ instead of traditional $p(x_{t-1}|c)$ in CFG. By replacing the proposed function with what I raised, we have
  $$l^+=\log(1+\frac{p(x_{t-1}|\phi)}{p(x_{t-1}|c)})\approx\frac{p(x_{t-1}|\phi)}{p(x_{t-1}|c)}$$
  $$\nabla\frac{p(x_{t-1}|\phi)}{p(x_{t-1}|c)}=\frac{p(x_{t-1}|c)\nabla p(x_{t-1}|\phi)-p(x_{t-1}|\phi)\nabla p(x_{t-1}|c)}{p(x_{t-1}|c)^2}=\frac{p(x_{t-1}|c)p(x_{t-1}|\phi)(\nabla\log p(x_{t-1}|\phi)-\nabla\log p(x_{t-1}|c))}{p(x_{t-1}|c)^2}=\frac{p(x_{t-1}|\phi)}{p(x_{t-1}|c)}(\nabla\log p(x_{t-1}|\phi)-\nabla\log p(x_{t-1}|c))$$
  $$\epsilon_\phi-\gamma_t\nabla\frac{p(x_{t-1}|\phi)}{p(x_{t-1}|c)}=\epsilon_\phi+\gamma_t'(\nabla\log p(x_{t-1}|c)-\nabla\log p(x_{t-1}|\phi)),\quad\gamma_t'=\gamma_t\frac{p(x_{t-1}|\phi)}{p(x_{t-1}|c)}$$
This is almost the traditional CFG with some scaling term. Derivation of $l^-$ is similar. **So there is nothing new to involve contrastive learning than native CFG with Bayesian**. This is still correct without the first approximation, since we have
  $$\nabla\log(1+\frac{p(x_{t-1}|\phi)}{p(x_{t-1}|c)})=\frac{p(x_{t-1}|c)}{p(x_{t-1}|c)+p(x_{t-1}|\phi)}\nabla\frac{p(x_{t-1}|\phi)}{p(x_{t-1}|c)}$$
  $$\epsilon_\phi-\gamma_t\nabla l^+=\epsilon_\phi+\gamma_t''(\nabla\log p(x_{t-1}|c)-\nabla\log p(x_{t-1}|\phi)),\quad\gamma_t''=\gamma_t\frac{p(x_{t-1}|\phi)}{p(x_{t-1}|c)+p(x_{t-1}|\phi)}$$
- In spite of introducing contrastive learning regime in guidance weights, the proposed method is technically some **linear scaling** trick instead of previous fixed and identical guidance weights which are empirically set, which is of poor contribution. To be more detailed, native CFG use a fixed $\gamma$ for positive conditions and $-\gamma$ for negative ones, as is claimed in Eq. (9). What the authors propose is to use a new coefficient larger than 1 multiplying the positive weights ($\frac{2}{1+\exp{(-\tau\|\epsilon_{\phi}-\epsilon_c\|)}}>1$) and a new one smaller than 1 multiplying the negative weights ($\frac{2\exp{(-\tau\|\epsilon_{\phi}-\epsilon_c\|)}}{1+\exp{(-\tau\|\epsilon_{\phi}-\epsilon_c\|)}}<1$). This strengthens the positive signal and weakens the negative one, somewhat provides a more compact bound for negative guidance.
- What further confirms my opinion is Fig. 3. The proposed positive weights almost coincide with the native fixed ones, and the negative weights remain extremely small (smaller than 1). Therefore the proposed method is almost equivalent to introducing no negative prompts in CFG, considering that traditional CFG in text-to-image generation involves guidance weight greater than 5.
- The experiments are unfair and marginal. First, no significant improvements are addressed in Fig. 4, Tabs. 1-4. Second, no native CFG results are provided in Fig. 4-5, Tab. 2, and Tab. 4 (CLIP Score metric is missing). Third, no FID or FD_DINOv2 is reported on ImageNet but only classification.

### Questions
Please carefully discuss the relation between the proposed method and what I raised in Weaknesses, especially the behavior of traditional CFG with scaled weights. Besides, please add all missed quantitative results including CLIP Score, FID, and FD_DINOv2.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Contrastive Classifier-Free Guidance (CCFG), a novel diffusion sampling mechanism that reformulates conditional guidance as an optimization problem based on a contrastive loss framework. CCFG successfully addresses the limitations of traditional Negative Prompting (NP), which suffers from sample distortion and quality degradation due to unbounded inverse probability distributions. The core innovation is the negative guidance term whose weight automatically approaches zero when the sample is irrelevant to the negative concept, preventing unnecessary pushing into low-density areas and enhancing sampling stability. CCFG achieves superior performance in handling both positive and negative conditions. Experiments demonstrate that, in both class-conditional generation and text-to-image tasks, CCFG more reliably and thoroughly excludes unwanted concepts, improving alignment with positive prompts while maintaining perceptual sample quality.

### Strengths
1. This paper proposes CCFG, which offers a theoretically sound mechanism by reformulating conditional guidance using a contrastive loss framework.
2. The proposed CCFG resolves the critical issue of distribution distortion and sample quality degradation inherent in standard Negative Prompting.
3. Quantitative and qualitative experiments show the proposed method’s effectiveness.

### Weaknesses
1.	The paper lacks a detailed sensitivity analysis on the crucial hyper-parameters τ_t and ω_t, hindering reproducibility and generalizability.
2.	The authors should compare the proposed CCFG with more recent Classifier-Free Guidance methods, such as CFG++[1], Autoguidance[2], Safree[3] on SD1.5/SDXL and also CFG-zero[4] on flow-matching model, for a fairer assessment.
3.	The advantages of CCFG over comparing methods on quantitative experiments are relatively modest. From Fig.4 and Tab.2, the CCFG’s scores are often only slightly higher or sometimes lower than other methods, raising the question about whether the theoretical analysis lead to substantial and stable improvement.
4.	The qualitative comparisons in Fig.5 is difficult to clearly see the advantages of CCFG over other methods. It is recommended to highlight the differences for better readability.
5.	The paper does not provide a comparison of the inference time or computational overhead introduced by CCFG compared with standard CFG or other complex guidance methods.

[1] Chung H, Kim J, Park G Y, et al. Cfg++: Manifold-constrained classifier free guidance for diffusion models[J]. arXiv preprint arXiv:2406.08070, 2024.

[2] Karras T, Aittala M, Kynkäänniemi T, et al. Guiding a diffusion model with a bad version of itself[J]. Advances in Neural Information Processing Systems, 2024, 37: 52996-53021.

[3] Yoon J, Yu S, Patil V, et al. Safree: Training-free and adaptive guard for safe text-to-image and video generation[J]. arXiv preprint arXiv:2410.12761, 2024.

[4] Fan W, Zheng A Y, Yeh R A, et al. Cfg-zero*: Improved classifier-free guidance for flow matching models[J]. arXiv preprint arXiv:2503.18886, 2025.

### Questions
Please refer to the questions and suggestions in the “Weaknesses” part.

### Soundness
2

### Presentation
3

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
The paper proposes Contrastive CFG, a guidance method for diffusion sampling that handles positive and negative concepts via an NCE perspective. It targets a known issue with “negative prompts” implemented as naïvely negated CFG terms, which can invert the distribution and degrade image quality. The method is claimed to both inject and remove specified concepts while preserving fidelity, across simple class labels and more complex text prompts.

### Strengths
1. **Conceptually novel and well-motivated.** The paper connects the behavior of negative prompts to an NCE objective and proposes a principled fix, achieving strong reported results.  
2. **Broad and diverse evaluation.** Beyond standard class-conditional generation, the authors introduce a more realistic benchmark with contrastive prompts.  
3. **Clear intuition.** Figure 2(b) and its accompanying experiment are intuitive and effectively illustrate the core idea.

### Weaknesses
1. **Generalization concerns**
   - **Sampler coverage.** The paper focuses on DDIM (deterministic). Please evaluate a stochastic sampler (e.g., DDPM). Since the method acts directly on the sampling process, stochasticity might reveal interesting behavior.  
   - **Backbone diversity.** Can the approach generalize to ViT-based diffusion backbones? In paper, authors only consider SD1.x and SDXL.

2. **Missing detail in benchmarks and setup**
   - **Class-pair construction.** Please elaborate on how the 100 class pairs with strong visual proximity are selected, and include more examples in the appendix.  
   - **PIE-Bench-Neg clarity.** The construction is currently vague. Provide concrete prompt examples.
   - **Sampling/config details (Tables 1 &3 & 4).** Please specify how CFG and CCFG are configured (e.g., guidance scales, sampling steps, schedulers) to enable exact reproducibility.

3. **Limited ablation**
   - **Guidance strength.** Include ablations across guidance weights (and any temperature/margin in the contrastive objective) to characterize stability and trade-offs.

### Questions
1. **About PIE-Bench-Neg.** The appendix mentions four categories. Can you report per-category results to reveal where the method helps most. Also analyze whether negative prompts being **adjectives** (style/attribute changes) versus **nouns** (object removal) affects performance.  It is very interesting.
2. **Failure cases.** Can you visualize typical failure modes and give some explanation?  

Overall, the paper is **novel and effective**, with a compelling motivation and promising results. However, several points remain underspecified. If you can address my concerns, I will consider raising my score.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a contrastive guidance method for conditional diffusion models that improves upon traditional Classifier-Free Guidance (CFG) and Negative Prompting (NP). The method uses a contrastive loss to align or repel the denoising direction based on the given condition. This approach achieves effective concept injection or removal while preserving image quality across various conditions, from simple classes to complex text prompts.

### Strengths
- This paper proposes a framework for designing the coefficients of positive and negative guidance in CFG.  
- The method shows only marginal improvement.

### Weaknesses
* The method needs one more forward times compared with classic positive minus negative way.

*  Still lack the analytic closed-form sampling distribution.

### Questions
- Can the framework provide design choices or strategies for scenarios where only two inference steps are allowed for each step?

### Soundness
2

### Presentation
2

### Contribution
2
