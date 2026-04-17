# Matched Data, Better Models: Target Aligned Data Filtering with Sparse Autoencoders

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Data filtering plays a central role in improving model performance, particularly for vision language models that are pretrained on large, noisy, and redundant image-caption datasets. Existing filtering techniques assess every sample individually and retain those that exceed a certain quality threshold, but such strategies fail to capture higher-order interactions. In this work, we propose a novel submodular framework for data selection that addresses this limitation. Our method, Submodular Distribution Matching (SDM), selects a subset by: (1) training a type of sparse autoencoder to learn disentangled and \emph{monotone} features; (2) estimating a target feature distribution from a target dataset; and (3) selecting a subset of samples whose feature distribution closely matches the target via submodular maximization. Given the DataComp-medium training set and no external models, SDM achieves state-of-the-art accuracy on both ImageNet-1K and average performance across 38 downstream tasks. On the full DataComp-medium benchmark, SDM delivers performance within 1\% of the state-of-the-art results while using over \textbf{\emph{5×}} fewer GPU hours than the leading approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the problem of filtering large-scale image-text datasets (like those scraped from the web) to improve model training. Such datasets are often noisy (containing irrelevant or low-quality data) and redundant (many very similar examples), which can hurt model performance. Traditional filtering methods usually score each data sample for quality and drop the low-scoring ones, but fail to capture the 'high-level feature. This paper proposed SDM to capture the high-level features and estimate the target distribution from ImageNet as a standard. Follow the standard, SDM can select high-quality samples from new datasets that follow the standard distribution at a high conceptual level.

### Strengths
1. This paper proposed SDM to capture high-level interactions (features), which go beyond single-sample selection. 
2. SDM uses high-quality datasets as standard to estimate the target distribution of high-level features
3. This data selection strategy is verified in rich downstream tasks.

### Weaknesses
1. It highly relies on the high-quality dataset, the high-level features learned from it are treated as the target distribution.
2. The interpretability of the high-level features is a big concern as I listed in the questions

### Questions
1.  In line 154, why the sparse dimension $d_{sparse} >> d_{in}$. For the learned sparse features, how can we verify what the features are? Why are they informative to represent the information shown in the Image or text?
2. For estimating the target distribution of 'high-level' features, this paper uses ImageNet as the standard. If all datasets share the same 'high-level' features? 
3. Although ImageNet is comprehensive in images, different datasets may have different features. This paper uses the same sparse autoencoder to align the learned high-level features. What if there are new important features in new datasets? 
4. Moreover, the proportion of different classes of images may affect the target distribution. If the gained performance is because the autoencoder is pre-trained on ImageNet, the data selection process uses ImageNet as a standard, which makes the selected data fit the model. 
5. For the comparison of different model sizes, does the best baseline have the same results across the different sizes?

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
3

### Summary
This paper addresses data filtering for vision-language models pretrained on large, noisy datasets. The authors propose Submodular Distribution Matching (SDM), a submodular framework that selects samples by matching feature distributions learned via sparse autoencoders. Experiments on DataComp-medium demonstrate that SDM achieves state-of-the-art or near state-of-the-art accuracy across ImageNet-1K and 38 downstream tasks, while significantly reducing computational cost.

### Strengths
1. The proposed Submodular Distribution Matching (SDM) presents a novel and promising approach to data filtering. Extensive experiments and superior performance compared with baseline methods convincingly demonstrate its effectiveness.

2. The theoretical analysis linking the designed submodular maximization objective to the distribution-matching target strengthens the rationale behind the proposed method.

3. The paper is well organized and clearly written, making it easy to follow and understand.

### Weaknesses
From a model training perspective, selecting a data subset that precisely matches or aligns with the target distribution may critically influence the model’s out-of-distribution (OOD) generalization capability. Providing additional empirical evaluation of OOD performance using the filtered data would help clarify the practical impact of the proposed method and further highlight its contribution to improving generalization beyond the training distribution.

### Questions
1. My main concern is that selecting a data subset that exactly matches or aligns with the target distribution may critically affect the model’s out-of-distribution generalization. How to balance the goal of distribution matching with maintaining data diversity during filtering?

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
The paper proposes SDM (Submodular Distribution Matching), a data filtering framework for vision-language models that combines sparse autoencoders (SAEs) with submodular optimization. The method learns disentangled and monotone features through SAEs with a novel monotonicity loss, then selects data subsets that match a target distribution while considering sample quality. The authors claim state-of-the-art results on DataComp-medium benchmark.

### Strengths
1. Novel framework integration: First to combine SAEs with submodular optimization for data selection, providing both interpretability and theoretical guarantees
2. Theoretical contribution: Establishes connection between KL divergence minimization and submodular maximization (Theorem 2.3), enabling efficient algorithms
3. Practical impact: Achieves competitive performance on DataComp-medium with reasonable computational budget compared to alternatives
4. Comprehensive evaluation: Tests across 38 downstream tasks, showing consistent (if modest) improvements
5. Monotonicity loss innovation: Novel loss term (Eq. 3) for encouraging monotone features in SAEs could be valuable for interpretability community

### Weaknesses
Major
1 Mathematical Soundness: 
- The objective \log m_i(A) is undefined when mi​(A)=0. Add explicit ε-smoothing: log(mi​(A)+ϵ) with sensitivity analysis.
- Unverified bound: The proof of Lemma 2.4 relies on∥h∥∞​≤β which the SAE architecture doesn't guarantee.
2. Statistical Validation: All results lack error bars. Re-run with ≥3 seeds, report mean±std for all tables, and provide significance tests.
3. Computational reporting: Should clarify total pipeline costs including encoding time


Minor

1. No ablation separating component contributions
2. Provide more details on Algorithm 1 (distance metrics, buffer size)
3. Explain the 5-run intersection choice

### Questions
1. How sensitive are results to ε-smoothing value? Please provide ablation.
2. Can you guarantee the β bound in practice? What's the actual max activation value observed?
3. Why take intersection of 5 greedy runs rather than union or single run?
4. What's the breakdown of improvements from SAE features vs submodular selection?
5. How does performance vary with different target distributions?

This paper makes a solid contribution to data selection for large-scale training. The idea of using SAEs to obtain interpretable features for submodular selection is clever and well-executed. While there are technical details to clarify, the core contribution is valuable and the experimental results support the claims.

The 0.7% average improvement may seem modest, but in the context of large-scale training where compute costs are substantial, even small improvements are valuable. The framework is also general and could be applied to other domains beyond vision-language models.

### Soundness
2

### Presentation
3

### Contribution
2
