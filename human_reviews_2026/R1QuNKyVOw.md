# MUSE: Model-Agnostic Tabular Watermarking via Multi-Sample Selection

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 2, 6, 6

## Abstract
We introduce MUSE, a novel watermarking paradigm for tabular generative models. Existing approaches often exploit DDIM invertibility to watermark tabular diffusion models, but tabular diffusion models suffer from poor invertibility, leading to degraded performance. To overcome this limitation, we leverage the computational efficiency of tabular generative models and propose a multi-sample selection paradigm, where watermarks are embedded by generating multiple candidate samples and selecting one according to a specialized scoring function.
    The key advantages of MUSE include (1) Model-agnostic: compatible with any tabular generative model that supports repeated sampling; (2) Flexible: offers flexible designs to navigate the trade-off between generation quality, detectability, and robustness; (3) Calibratable: theoretical analysis provides principled calibration of watermarking strength, ensuring minimal distortion to the original data distribution.
    Extensive experiments on five datasets demonstrate that MUSE substantially outperforms existing methods. Notably, it reduces the distortion rates by 84-88% for fidelity metrics compared with the best performing baselines, while achieving 1.0 TPR@0.1%FPR detection rate.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a tabular watermark method that assigns a pseudorandom score to each row of the table based on the hash of selected column values and a secret key. During watermark insertion, the model repeatedly samples multiple candidate rows to select the one with the highest score. During detection, the average score of a given table is compared against a threshold to determine whether the table contains the watermark.

### Strengths
1. The paper is generally well written and easy to follow.
2. The method 's detectability and fidelity guarantee are supported by mathematical theorems (Theorem 4.1 and Theorem 4.3).
3. The method is also supported by experiments in real world dataset (Adult, Default, Shoppers and Beijing).

### Weaknesses
1. The idea seems not novel. https://arxiv.org/abs/2410.02099 and https://arxiv.org/pdf/2403.04808 have almost the same idea as your work though they focus on watermarking large language models. 

2. The paper does not consider additive noise attacks in its perturbation experiments. However, such attacks are an important robustness benchmark that has been widely considered in many prior works cited by the authors (https://dl.acm.org/doi/10.1145/3658644.3690373; https://openreview.net/forum?id=71pur4y8gs)

3. It is unclear how to determine a predefined threshold for the detection. From Theorem 4.1, we can see the False Positive Rate control depends on the gap between  the expected score of an unwatermarked sample and the expected score of a watermarked sample obtained via repeated samples. However, in practice this gap is usually unknown since the unwatermarked sample is not necessary from the detector. Also, it does not make sense to consider the "optimal distribution" in Theorem 4.1 since this is in fact the best case. However in practice the unwatermarked table does not always have such nice property.

### Questions
NA

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a novel approach to watermarking synthetic tabular data agnostic to the underlying generative model while maintaining high fidelity and robustness to attacks. Experiments are shown on a wide variety of datasets and generators to showcase the efficacy of the approach.

### Strengths
(a) The watermarking approach is agnostic to the generative model and simply uses multiple samples and picks the highest scoring one, with the scoring function appropriately chosen. 
(b) Theoretical results are shown for detectability for a certain false positive rate of the watermarking approach and shown to be between 2 and 4 for a couple hundred samples@0.01. 
(c) Watermarking both at a single (few) column-level and full set of columns are provided trading off robustness versus distortion trade-off. 
(d) The computation time of generation and detection is better than the previous state-of-the-art approaches.

### Weaknesses
(i) The quantile rank is proposed as a method to thwart adversaries but it is unclear if this can not be reverse engineered by adversaries. 
(ii) Similarly, the question of breaking the current approach for watermarking is not fully addressed in the paper.
(iii) The complexity of having both categorical and continuous features in the dataset is not discussed in detail.

### Questions
(1) If the dataset is complete categorical and say binary, would it pose a problem to the approach? More generally, can this be applied to video or image watermarking and what differs?
(2) The quantile rank is proposed but is deterministic which may make it vulnerable to attackers. Would selection by randomizing according to the rank help?
(3) How does the approach  work with respect to permutation attacks where the column order is completely randomized?

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
4

### Summary
The paper proposes MUSE, a model-agnostic watermarking framework for tabular generative models that eliminates the need for fragile DDIM inversion used in prior methods such as TabWak[1]. MUSE performs multi-sample selection. It samples multiple candidates ad selects the one with the highest key-dependent watermark score. Two watermarking scoring functions are introduced, one is JV Hashing, which hashes for low-distortion fidelity, the other is PC hashing, used for strong robustness. Theoretical analysis provides calibration of the false positive rate and provable distribution preservation.


[1] Zhu, C., Tang, J., Galjaard, J. M., Chen, P. Y., Birke, R., Bos, C., & Chen, L. Y. (2025, January). TabWak: A Watermark for Tabular Diffusion Models. In ICLR.

### Strengths
1. The approach departs from the inversion-based paradigm dominant in diffusion watermarking, meaning MUSE is compatible with any generative model that supports repeated sampling, including diffusion, autoregressive, and masked models.
2. Theoretical analysis on the detectability and distribution-preservation is provided. Though I didn't take a thorough look to the proof.

### Weaknesses
1. The choice of pseudorandom function $f$ and hash function $H$ is abstracted away but critical to real-world detectability and security. The paper doesn't provide implementation sensitivity analyses, e.g., whether certain hash or key spaces degrade performance.
2. No ablation on hyperparameters $m$ beyond theory. While theorem 4.1 gives calibration, the experiments mostly fix $m=2$. There is no systematic analysis of how varying $m$ or $N$ affects trade-offs between computation, detectability, and fidelity.
3. The paper demonstrates quantitative performance but provides little visualization of what kind of statistical signal the watermark introduces in the tabular domain.

### Questions
1. The field of tabular data watermarking is relatively new. Some baselines (Tree-Ring, Gaussian Shading) are from image watermarking. Could the author explain how these two methods are used in the tabular diffusion model?
2. The detector relies on a simple mean-score threshold; it’s unclear how well this scales to more complex scenarios (e.g., mixed distributions, partial key leakage, or federated settings).

### Soundness
3

### Presentation
2

### Contribution
3
