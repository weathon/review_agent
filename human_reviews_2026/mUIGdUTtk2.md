# Cross-Domain Lossy Compression via Rate- and Classification-Constrained Optimal Transport

- Decision: Accept (Oral)
- Scores: 2, 6, 6, 10

## Abstract
We study cross-domain lossy compression, where the encoder observes a degraded source while the decoder reconstructs samples from a distinct target distribution. The problem is formulated as constrained optimal transport with two constraints on compression rate and classification loss. With shared common randomness, the one-shot setting reduces to a deterministic transport plan, and we derive closed-form distortion-rate-classification (DRC) and rate-distortion-classification (RDC) tradeoffs for Bernoulli sources under Hamming distortion. In the asymptotic regime, we establish analytic DRC/RDC expressions for Gaussian models under mean-squared error. The framework is further extended to incorporate perception divergences (Kullback-Leibler and squared Wasserstein), yielding closed-form distortion-rate-perception-classification (DRPC) functions. To validate the theory, we develop deep end-to-end compression models for super-resolution (MNIST), denoising (SVHN, CIFAR-10, ImageNet, KODAK), and inpainting (SVHN) problems, demonstrating the consistency between the theoretical results and empirical performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
## Summary
* This paper studies the rate distortion characteristic of a problem named "cross-domain lossy compression". In this problem, the authors considers joint optimization of distortion loss, classification uncertainty and even perception divergence.
* The authors derive the closed formed solution to their function using constrained pptimal transport for Bernouli and Gaussian source.
* The authors validate their theory through a somewhat different experimental setup using CE loss instead of classification uncertainty.

### Strengths
## Strength
* This paper is well written and the study of the problem is comprehensisve. Both one-shot and asymptotic cases are considered.
* The authors provide detailed example for both discrete and continous source (Bernouli and Gaussian).
* The experimental design covers different degradation and different image dataset.

### Weaknesses
## Weakness
* First, the mathematical formulation of the proposed approach is deviated from its objective.
  * More precisely, the authors claim that the formulation considers the "Classification utility" and can be useful in practical scenario. However, the constraint imposed for the "classification" is $H(S|Y)$, the entropy of posterior distribution of class label conditioned on reconstruction. This $H(S|Y)$ has really nothing to do with classification utility as the classification in rate-distortion-classification (RDC) or rate-distortion-perception-classification
(RDPC) [Task-oriented lossy compression with data, perception, and classification constraints].  In general, the constraint $H(S|Y)$ seems to be close to a measure of "Inception Score" [Improved Techniques for Training GANs], which is only a measure of how the reconstruction is close to any label, instead of how the reconstruction is close to the label of the source. 
  * For a MNIST example, we can let the decoder to constantly output random images with certainly recognizable digits. In that case, $\forall C, H(S|Y)=0\le C$. However, the reconstruction really has nothing todo with the input. And the $H(S|Y)$ really has no correlation with classification accuracy.
* Second, the mathematical formulation of the proposed approach is also deviated from experimental results. Despite the $H(S|Y)$ in their theory is upperbounded by the CE loss they use in experimental result, the nature of $H(S|Y)$ and CE is very different. As I have explained previously, $H(S|Y)$ only measures how certain the model is on any label, given the reconstruction. However, CE is really about the classification accuracy of given label. I doubt that the authors prefe CE loss instead of $H(S|Y)$ in their experimental results just to get a better result. I can not see why it is necessay to use CE over $H(S|Y)$, as $H(S|Y)$ is really not hard to implement.
* Overall, I do appreciate the effort the authors have spend in solving the formulated problem. However, the current problem formulation using $H(S|Y)$ is too different from the CE loss they use in experiments. Besides, the current problem formulation can not achieve the "joint restoration + classification" target that the authors have claimed. I recommend to reject this paper, as its theory deviates too much from their claim, and is too different from their empirical results.

### Questions
## Questions
* Is that possible to derive the theory with proper classification loss, such as CE?
* What is the experimental result like using the real $H(S|Y)$ loss instead of CE? This is really not difficult to implement and the authors should present the honest result using the real loss following their theory.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper models "cross-domain lossy compression" as a Constrained Optimal Transport problem. It derives closed-form expressions for the tradeoffs between Distortion-Rate-Classification (DRC) and Rate-Distortion-Classification (RDC).

### Strengths
1. The paper provides detailed and rigorous theoretical derivations.

2. The paper addresses "cross-domain lossy compression," which is more aligned with real-world application scenarios.

### Weaknesses
1. The core theoretical contribution of the paper lies in deriving closed-form DRC/DRPC tradeoffs for Bernoulli and Gaussian sources. However, the distribution of real-world natural images is far more complex than simple Bernoulli or Gaussian distributions. This may create a gap between the theory and practical application.

2. The paper does not provide a quantitative benchmark comparison against current state-of-the-art (SOTA) image restoration (e.g., denoising, super-resolution) or compression-de-compression (codec) methods. This makes it difficult for readers to judge whether the proposed method is superior in practical performance or merely more theoretically complete.

3. The paper's "one-shot" theory and experimental implementation both rely on shared common randomness ($U$). Although the paper mentions this can be implemented via a shared random seed, guaranteeing such randomness may be difficult in certain compression scenarios (e.g., write-once systems, broadcasting, or systems without two-way communication).

### Questions
see weakness.

### Soundness
3

### Presentation
2

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
The paper introduces a theoretical and algorithmic framework for cross-domain lossy compression, where the encoder observes degraded inputs while the decoder reconstructs samples from a distinct target distribution. The authors formalize this setting as a rate- and classification-constrained optimal transport (OT) problem, generalizing classical rate–distortion (RD) theory to cross-domain and task-aware scenarios. Closed-form solutions for Bernoulli and Gaussian models are derived, establishing distortion–rate–classification (DRC) and distortion–rate–perception–classification (DRPC) tradeoffs. The paper further bridges theory and practice through deep learning–based implementations, validated on tasks such as super-resolution, and denoising.

### Strengths
This paper makes a conceptually original and mathematically rigorous contribution by integrating optimal transport, rate-distortion theory, and classification constraints within a single unified framework. 

The idea of cross-domain compression-where the reconstruction belongs to a different distribution than the source, is a clear and meaningful generalization beyond standard RD or perception-aware setups. 

The theoretical development is of high quality: the paper provides single-letter characterizations, leverages shared randomness to achieve deterministic transport plans, and derives closed-form DRC expressions.

The mathematical derivations are supported by intuitive illustrations. 

The experimental section, although concise, demonstrates consistency between theory and deep generative implementations, validating the theoretical tradeoffs with empirical data.

### Weaknesses
While they derived closed from solution of the RDC tradeoff, for Bernoulli and Gaussian sources, for practical distributions like images, their evaluation is short.

In terms of clarity, the derivations are mathematically thorough but dense, which may reduce accessibility for readers outside the information theory community. 

The experimental validation, while diverse in datasets, remains relatively superficial compared to the theoretical derivations.

Discussion of related work could be more critical, especially regarding how prior task-aware generative compression frameworks differ in assumptions, experimental scope, or real-world feasibility.

### Questions
Could the authors create a heatmap version of Figure 6? The 2 constraints (rate and classification) could be the x and y axes, and distortion is the color of the pixel (or something similar to Figure 4a in [1]).

Could the authors create a 2D grid of generated samples from Figure 6 (similar to Figure 6 in [1])?

Could the authors provide quantitative comparisons between the theoretically predicted DRC curves and the empirical results on the Bernoulli or Gauss distribution?

Can the framework be extended to perception metrics (e.g., LPIPS or FID) rather than Wasserstein divergences?

[1] Blau, Yochai, and Tomer Michaeli. "Rethinking lossy compression: The rate-distortion-perception tradeoff." International Conference on Machine Learning. PMLR, 2019.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper presents a comprehensive theoretical and empirical framework for cross-domain lossy compression, where the encoder observes a degraded source while the decoder reconstructs samples from a distinct target distribution. The authors formulate this as a constrained optimal transport problem with dual constraints on compression rate and classification loss. The key theoretical contributions include closed-form characterizations of distortion-rate-classification (DRC) and rate-distortion-classification (RDC) tradeoffs for both Bernoulli sources under Hamming distortion and Gaussian models under mean-squared error. The framework is further extended to incorporate perception constraints using Kullback-Leibler and squared Wasserstein divergences. The theoretical findings are validated through extensive experiments on super-resolution (MNIST), denoising (SVHN, CIFAR-10, ImageNet, KODAK), and inpainting (SVHN) tasks, demonstrating strong consistency between theoretical predictions and empirical performance.

### Strengths
​Theoretical Novelty and Rigor:​​ The paper makes significant theoretical contributions by unifying optimal transport with rate-distortion theory and classification constraints. The derivation of closed-form solutions for both one-shot and asymptotic regimes represents substantial mathematical advancement in the field.
​​Comprehensive Framework:​​ The authors successfully integrate multiple constraints—compression rate, distortion, perception quality, and classification performance—into a unified framework. The extension to distortion-rate-perception-classification (DRPC) functions with explicit perception constraints is particularly noteworthy.
​​Strong Experimental Validation:​​ The paper provides extensive experimental validation across multiple datasets and tasks. The implementation of deep end-to-end compression models with quantization, entropy modeling, adversarial training, and classification components demonstrates practical applicability.
​​Practical Relevance:​​ The problem formulation addresses real-world scenarios where compressed representations must serve multiple purposes simultaneously—maintaining perceptual quality, supporting downstream tasks, and operating under rate constraints.

### Weaknesses
​​Limited Real-World Application Scope:​​ Although the experiments cover multiple datasets, they primarily focus on standard academic benchmarks. Validation on more practical, real-world compression scenarios would strengthen the paper's impact.
​​Computational Complexity:​​ The paper does not sufficiently address the computational requirements of the proposed approach, particularly for the adversarial training components and their scalability to very high-resolution images.
​​Comparison with State-of-the-Art:​​ While the paper includes comparisons with baseline methods, more extensive comparisons with recent state-of-the-art learned compression methods would provide better context for the proposed approach's advantages.

### Questions
​Scalability and Generalization:​​ The theoretical results are derived for Bernoulli and Gaussian sources. How well do these theoretical predictions hold for more complex, multi-modal distributions encountered in real-world data?
​​Optimization Tradeoffs:​​ The paper presents multiple constraints (rate, distortion, perception, classification). In practical applications, how should practitioners balance these competing objectives, and are there guidelines for setting the constraint parameters?

### Soundness
4

### Presentation
4

### Contribution
4
