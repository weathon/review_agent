# Intriguing Bias-Variance Tradeoff in Diffusion Models

- Avg Score: 4.67
- Decision: Reject
- Scores: 8, 4, 2

## Abstract
Despite the remarkable performance of generative Diffusion Models (DMs), their internal working is still not well understood, which is potentially problematic. This paper focuses on exploring the important notion of bias-variance tradeoff in diffusion models.
Providing a systematic foundation for this exploration, it establishes that at one extreme, the diffusion models may amplify the inherent bias in the training data, and on the other, they may compromise the presumed privacy of the training samples. Our exploration aligns with the memorization-generalization understanding of the generative models, but it also expands further along this spectrum beyond "generalization", revealing the risk of bias amplification in deeper models. Our claims are validated both theoretically and empirically.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper studies the “bias–variance tradeoff” in diffusion models, inspired by the k-NN analogy, showing that the abstraction level of latent representations controls the smoothness of the learned energy landscape.
Models operating at higher abstraction levels yield smoother landscapes that amplify dataset bias, whereas models relying on lower-level, less abstract features tend to memorize training samples and risk privacy leakage.
Both theoretical proofs and experiments on MNIST, CelebA, and Stable Diffusion v1.5 empirically support this connection.

### Strengths
It offers a novel and unified theoretical framework that connects bias amplification and memorization in diffusion models through the bias–variance tradeoff.

### Weaknesses
1. The experimental validation is somewhat limited — results are based on small-scale datasets and qualitative observations on Stable Diffusion v1.5,  and only train on the unconditional model. If you 
2. The definition of the “abstraction level” is not clearly established. The paper implicitly equates abstraction with the number of encoder blocks or latent downsampling, but this connection is heuristic rather than theoretically grounded

### Questions
1. Have you examined whether similar bias–variance phenomena appear in Rectified Flow models ?
2. In DiT-style architectures without a clear encoder–decoder hierarchy, how can the notion of “abstraction level” be rigorously defined or measured?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work uncovers a general bias-variance trade-off phenomenon, based on the landscape analysis of hierarchy representations. The basic  insight is: (i) deep models would emphasize majority sample classes and generate more diverse images; (ii) shallow models are more sensitive and hence show privacy risks.

### Strengths
1. The paper is clearly organized with many illustrations. 
2. Both theoretical analysis and multiple numerical verifications are provided to support main findings. 
3. Main findings are new and interesting.

### Weaknesses
1. It seems that the whole framework (e.g. Eq. (3, 4, 5)), analysis and results (e.g. Thm. 1) work for general NNs/en-decoders. Why do authors only discuss diffusion models, and are there any new insights specifically tailored to diffusion models? 
2. Following 1, it would be better to provide self-contained formulations of e.g. latent spaces and energy landscapes, particularly for diffusion models, and formulate the exact definition of "abstraction" repeatedly used in this paper. 
3. For Thm. 1: 
- The notations $E^a$ or $E^{(a)}$ should be consistent, and in fact, authors do not define $E^{(a)}$ in Eq. (6). 
- The transformation $\phi^{(a)}$: 
    - $\phi^{(a)}$ is assumed as a contractive mapping, but why does it hold particularly for diffusion models? It seems that this is due to the statement "As increasing abstraction decreases sensitivity to perturbations" in the proof, but why does this hold? 
    - $\phi^{(a)}$ is also assumed as a invertible mapping in Eq. (6). Why does it hold particularly for diffusion models?
4. This work aims to justify both bias and variance in diffusion models, and the bias aspect is mathematically justified in Thm. 2. However, this is not the case for the variance aspect. Can authors give more quantitative analysis of these high-variance cases? In addition, it is not clear how these theoretical/numerical justifications of bias and variance relate to their definitions Eq. (4, 5). 
5. As shown in e.g. Fig. 4 and Tab. 3, deep models appear both high biases and diversity. However, biased models prefer majority classes, which reduce the overall diversity in principle. Can authors clearly explain the definition of diversity used and evaluated in this work, otherwise the implications would be contradictory. 
6. As discussed in this work, deep models show high biases on majority classes, and shallow ones have high variance and hence privacy risks. Then, as required by the paper title, how can we trade-off the depth in practice, particularly for diffusion models?

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the bias-variance tradeoff in diffusion models. They first theoretically demonstrate that a energy model with higher abstraction level tends to have more bias, i.e., the majority class will be amplified. On the other hand, low abstraction leads to local minimums close to the individual data, hence suffer from high variance. They validate these conceptions in diffusion models. They find that diffusion models with more encoding blocks (higher abstraction level) exhibits bias, generating more from majority class, whereas models with less encoding blocks (lower abstraction level) exhibits high variance, generating samples that are more similar to the training data.

### Strengths
The empirical observations on the bias-variance tradeoff in diffusion models are interesting, which can inspire further investigation.

### Weaknesses
1. The energy model and KNN perspective (figure 2, table 1) of diffusion models are mostly conceptual, with insufficient empirical or theoretical support. Specifically, the theory has nothing to do with diffusion models, while I understand the conceptual similarity, it still feels kind of vague. Overall, I feel this work is not rigorous enough, and not very convincing.

2. The experiment design could be better. For example, it is unclear whether when increasing the encoder block, the total number of model parameters is fixed or not. It is well-known that larger models will be more prone to memorization, i.e., higher variance. How to isolate the effect of abstraction (number of encodings blocks) from the effects of total model parameters? I understand one related experiment is provided, but the setting in others remain unclear.

3. The experiment is limited to simple dataset such as MNIST and face dataset. The number of classes are limited. To support the arguments in the paper, more experiments should be performed on dataset with large number of classes such as ImageNet.

### Questions
1. Why more encoding blocks leads to higher abstraction level? Imagine given enough model parameters, one encoding block can learn the same function as multiple encoding blocks.

2. Why more encoding blocks lead to less variance, which increase the model parameters? This seems to contradict with prior works that demonstrate a larger model tends to memorize more (have higher variance).

3. No matter how many encodings blocks a model have, when trained with denoising score matching loss, they are always approximating the data distribution, which in theory should have no bias and variance issue. It is unclear why the observed bias-variance tradeoff emerges. 

4. Can you do more experiments on dataset with more classes?

### Soundness
2

### Presentation
3

### Contribution
2
