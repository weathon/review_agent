# Diffusion-Inspired Reconfiguration of Transformers for Uncertainty Quantification

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Uncertainty quantification in pre-trained transformers is critical for their reliable deployment in risk-sensitive applications. Yet, most existing pre-trained transformers do not have a principled mechanism for uncertainty propagation through their feature transformation stack. In this work, we propose a diffusion-inspired reconfiguration of transformers in which each feature transformation block is modeled as a probabilistic mapping. Composing these probabilistic mappings reveals a probability path that mimics the structure of a diffusion process, transporting data mass from the input distribution to the pre-trained feature distribution. This probability path can then be recompiled on a diffusion process with a unified transition model to enable principled propagation of representation uncertainty throughout the pre-trained model’s architecture while maintaining its original predictive performance. Empirical results across a variety of vision and language benchmarks demonstrate that our method achieves superior calibration and predictive accuracy compared to existing uncertainty-aware transformers.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper follows a recent line of research that views transformers as diffusion processes in the embedding space. The paper proposes to record the embedding state after each multi-head attention block and fit a diffusion model to fit the trajectory of embeddings from input (t=T) to final-layer embedding (t=0). This is a subtle change in how the diffusion process is modeled and constructed from the pretrained transformer in previous works. The authors report to match or outperform the transformers they distill from across the line. The experiments are on computer vision and NLP on a small ~3M scale.

### Strengths
In order of magnitude:

1. The relevance and novelty of this line of research is intruiging. However, I cannot judge the exact novelty over the previous three papers from Chen et al., which it seems to iterate on. I hope that the other reviewers have read these previous papers and can comment on how strong the novelty of the proposed changes is.
2. The proposed method seems to match or outperform the baselines in terms of accuracy while improving ECE/NLL/Brier score.
3. The evaluation suite is wide in terms of metrics and baseline (variational) methods and consistently reports standard errors.
4. The code is provided and gives all exact commands to reproduce the results of the paper.
5. The writing is very clear and the mathematical notation makes it easy to follow the details.

### Weaknesses
In order of magnitude:

1. I am somewhat concerned with the stability / robustness to deeper transformers. Currently, all experiments are on very small (<= 6M) and very shallow (5 to 7 layer) transformers, whereas more modern models tend to have 4x the layers and about 1000x the parameters. Besides the transfer to 1000x larger models (which often fails), I am most concerned that the probability trajectories might not be distilled as well on deeper architectures (= more diffusion steps). It would be great if you could give at least single points of evidence here for NLP transformers, like a small Qwen 2.5 model (on top of the ViT-B-16 experiment in Appendix A.6.1)
2. Compute permitting (duly noted that “all experiments are conducted on a single NVIDIA L40 GPU”), it would be great to evaluate on bigger test datasets. Currently, the analysis rests on 2x 10k test samples in computer vision and 25k + 2k test samples in NLP. E.g., in computer vision, there is ImageNet instead of CIFAR, and maybe DINO or AIM-like setups, and in language there is the lm eval harness.
3. The baseline transformers appear to have been trained on very little compute. I am not sure if the performance is converged, particularly because the standard error is within the performance delta (and so the standard deviation is even larger). It would be great to distill from pretrained models from the literature (to make sure they are converged). It would also be great to detail your train pipeline, i.e., what has been distilled from what for how many epochs, etc. This can prevent seeing improved results simply because the distilled method has effectively been trained for longer.
4. It would be nice, but not high priority, to add a distillation baseline that distills a student transformer from the teacher transformer, using the same pipeline (= epochs, data, ...) you use for your method. This can make sure that the gains you observe are not just due to soft-label distillation, longer training, or any other maybe hidden difference that the distillation pipeline introduces.

### Smaller weaknesses that don’t influence my score and don’t need to be rebuttled, but would be nice to fix in the revised version

* It would be great to bring your deep ensemble comparisons from the appendix to the main paper in Table 2, especially because you claim to set a new SOTA on uncertainty quantification.
* I am not sure how you structured your experiments with the random seeds. If you pretrained five models, and then distilled them each, then you could do coupled comparisons to minimize the estimation variance of the delta your method brings.
* I would suggest to move the related works to the front, and to use it to elaborate a bit more on distilling diffusion trajectories from transformers, so basically the Chen et al. line of research

## Justification for the overall score

The paper is refreshing and works on bridging two important current techniques with attention to detail. I think it offers a nice perspective to the conference and should be accepted. However, I remain cautious of a too enthusiastic recommendation since all experiments are on very small scale and may not be relevant for larger modern models. I am happy to increase my score if the authors can resolve this doubt in the rebuttal period.

### Questions
1. Can you elaborate on which parts of your perspective come from the previous Chen et al. research line and where you differ? From lines 127-133 it seems like the main novelty is that you stabilize training by recording embeddings after each attention block, treating MLP and LN as part of what function the next-step GP models (as opposed to something different before)?
2. How are the baselines pretrained? E.g., how many epochs do you use, and have you considered using pretrained models from other researchers (which have the benefit of likely having converged in training)
3. Is there any benefits that the GPs give you in terms of disentangling aleatoric and epistemic uncertainty? Maybe you could launch a simple experiment, but this has lower priority compared to the other experiments above.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes DIRECTOR, a framework for UQ in pre-trained transformer models. The key idea is to interpret transformer blocks as probabilistic transitions between latent representations, allowing the network to be modeled as a reverse diffusion process. Experiments on vision and language tasks (e.g., CIFAR-10/100, IMDB, CoLA) show that DIRECTOR results on calibration and out-of-distribution robustness compared to standard deterministic transformers and prior UQ methods.

### Strengths
1. The paper addresses an important and timely topic in modern machine learning.

2. The experimental evaluation is thorough in the sense that it includes comparisons against numerous baselines across multiple datasets.

### Weaknesses
1. The paper treats all forms of uncertainty as a single quantity, without distinguishing between aleatoric and epistemic components. If my understanding is correct, the DIRECTOR framework estimates only the total uncertainty.

2. DIRECTOR requires training a separate diffusion model on top of a pre-trained transformer. This introduces additional computational cost for uncertainty estimation, whereas several existing approaches can extract uncertainty directly from the predictive model without such retraining.

3. It is unclear whether the reported improvements are statistically significant. The paper would benefit from significance testing or error bars to support the claimed performance gains.

4. Minor Issues:
   - The definition of \( U_t \) is missing in the main text.  
   - The acronym DIRECTOR feels somewhat forced and unintuitive.  
   - The bold formatting at the beginning of paragraphs is inconsistent and at times unnecessary.  
   - The paper would benefit from additional background on prior methods, such as the Gaussian Process interpretation of transformers and the KEP framework. This material could be included in the appendix.

### Questions
1. What exactly is used as the estimate of uncertainty? It is unclear to me how the paper defines or computes the uncertainty measure.

2. When training the diffusion model, could you incorporate existing methods that already estimate uncertainty in diffusion models [1,2]?

3. What is the statistical significance of the results reported in Tables 1–5?

4. Have you evaluated your method for hallucination detection?

[1] Berry, Lucas, Axel Brando, and David Meger. "Shedding light on large generative networks: Estimating epistemic uncertainty in diffusion models." The 40th Conference on Uncertainty in Artificial Intelligence. 2024.

[2] Berry, Lucas, et al. "Seeing the Unseen: How EMoE Unveils Bias in Text-to-Image Diffusion Models." arXiv preprint arXiv:2505.13273 (2025).

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces **DIRECTOR**, a method that reinterprets Transformer architectures as **reverse-time diffusion processes** in feature space. By reconfiguring each block to end with a Multi-Head Self-Attention (MHSA) module, the model treats inter-block representations as Gaussian transitions. A unified spatiotemporal kernel then models uncertainty propagation across layers.

The approach is formalized through a **variational upper bound** on the likelihood, yielding a practical objective that combines KL-matching between per-step transitions and a learnable diffusion model, plus a performance-aware regularizer.

Experiments on CIFAR-10, CIFAR-10-C, and CoLA demonstrate improvements in calibration (ECE, NLL), robustness, and OOD detection while maintaining accuracy. The contribution aims to make uncertainty quantification in Transformers more coherent and computationally efficient compared to per-block Gaussian process reparameterizations.

### Strengths
- **Elegant conceptual reformulation:** The idea of treating Transformer layers as a unified stochastic diffusion process is creative and intuitively appealing.
- **Mathematically consistent derivation:** The variational objective is well explained and easy to reproduce.
- **Trustworthiness impact:** DIRECTOR provides an interpretable probabilistic mechanism for uncertainty propagation, aligning with responsible AI principles.
- **Consistent empirical results:** Improvements in calibration and robustness are observed across different datasets.
- **Potential for extensibility:** The framework could inspire future work on probabilistic Transformers and diffusion-based calibration.

### Weaknesses
1. **Limited novelty:** The work combines existing paradigms (diffusion models, GP interpretations, variational inference) without introducing fundamentally new theory.
2. **Empirical scope:** Experiments are limited to moderate-sized benchmarks; scalability to large models remains untested.
3. **Simplified covariance modeling:** The diagonal and Cholesky-like approximations likely underestimate structured uncertainty.
4. **No runtime comparison:** Computational overhead relative to KEP, SWAG, or ensembles is not reported.
5. **Lack of theoretical guarantees:** The paper does not formally explain why the unified diffusion kernel improves calibration.
6. **Ablation limitations:** The sensitivity of results to loss weight choices and kernel design is not fully explored.

### Questions
1. How is the unified kernel \(q_\theta(X_{t-1}|X_t)\) parameterized? Are its parameters shared across time steps or conditioned via embeddings?
2. Have you tested low-rank or structured covariance forms, and if so, how do they affect calibration and runtime?
3. What is the computational and memory overhead of DIRECTOR compared to KEP-all, KEP-last, and deep ensembles?
4. Could you report comparisons with simpler calibration baselines such as temperature scaling or MC dropout?
5. Is there a theoretical link between diffusion-step consistency and improved uncertainty calibration that could explain the observed robustness gains?
6. How might the approach generalize to large models like ViT-B or language models (e.g., GPT architectures)?

### Soundness
3

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
The paper proposes DIRECTOR, which reconfigures pretrained transformers into a diffusion-like probabilistic model for uncertainty quantification (UQ). Each transformer block is treated as a Gaussian transition, forming a continuous stochastic process that captures correlations across blocks. The learned diffusion kernel then enables principled uncertainty propagation across the network. Experiments span vision (CIFAR-10/100, CIFAR-10-C) and language (IMDB, CoLA) benchmarks, comparing DIRECTOR with ViT, transformers, and other uncertainty-aware models like KEP-SVGP, MC Dropout, and KFLLA. Results show lower calibration error (ECE, NLL, Brier) and competitive or improved accuracy, with added robustness to distribution shifts and OOD detection

### Strengths
+ Better or competitive accuracy and calibration performance across CV and NLP, in-dist, OOD, and corruption datasets.

### Weaknesses
- The paper assumes readers already know the emerging approaches with the terms "reparameterize the attention outputs", " probabilistic chain mapping", etc, under-explained. It is hard to get the real problem of these emerging approaches, thus weakening the motivation of the proposed method. The core motivation, "uncertainty doesn’t propagate properly when each block is reparameterized separately", is supported only by a small, hand-picked comparison that could be confounded. 
- In Fig. 1, if cross-block correlation is the culprit, the paper should show stronger diagnostics, e.g., measuring inter-block covariance of features/uncertainty, or ablating correlations explicitly, rather than relying on a single "all-blocks vs last-block" contrast.
- Prior work is said to already "recast the pre-trained transformer as a probabilistic chain", but Contribution 1 then "reinterprets" step-wise transformations as a probability path, which sounds overlapping, and the paper doesn’t draw a crisp boundary between prior "probabilistic chain" framing and this paper’s stated reinterpretation.

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2
