# Dataset Distillation via Committee Voting

- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Dataset distillation aims to synthesize a smaller, representative dataset that preserves the essential properties of the original data, enabling efficient model training with reduced computational resources. Prior work has primarily focused on improving the alignment or matching process between original and synthetic data, or on enhancing the efficiency of distilling large datasets. In this work, we introduce $\bf C$ommittee $\bf V$otingfor $\bf D$ataset $\bf D$istillation (CV-DD), a novel and orthogonal approach that leverages the collective wisdom of multiple models or experts to create high-quality distilled datasets. We start by showing how to establish a strong baseline that already achieves state-of-the-art accuracy through leveraging recent advancements and thoughtful adjustments in model design and optimization processes. By integrating distributions and predictions from a committee of models while generating accurate soft labels, our method captures a wider spectrum of data features, reduces model-specific biases and mitigates distributional shifts between synthetic data and original data. This voting-based strategy not only promotes diversity and robustness within the distilled dataset but also significantly reduces overfitting, resulting in improved performance on post-eval tasks. Extensive experiments across various datasets and IPCs (images per class) demonstrate that Committee Voting leads to more reliable and adaptable distilled data compared to single/multi-model distillation methods, demonstrating its potential for efficient and accurate dataset distillation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes CV-DD (Dataset Distillation via Committee Voting), a novel framework that improves dataset distillation (DD) by integrating knowledge from multiple models instead of relying on a single teacher backbone.
The key insight is that existing decoupled DD methods suffer from *single-model bias* and limited diversity, as the synthetic data inherits the inductive bias of one architecture.
CV-DD introduces a committee voting mechanism, where several heterogeneous teacher models (e.g., ResNet18/50, MobileNetV2, ShuffleNetV2, DenseNet121) contribute to the synthesis process.
Each teacher’s contribution is weighted by its *prior performance* (estimated on real data), and gradients are aggregated via a softmax-weighted voting rule.
Theoretical analyses show that model diversity encourages more diverse synthetic data and that prior-guided voting aligns the update direction with better generalization.

### Strengths
* **Clear motivation and insight:**
  The paper addresses a genuine and underexplored issue in dataset distillation — the *bias introduced by single-teacher distillation*.
  Framing the solution as a “committee voting” problem is intuitive yet novel.
* **Solid technical design:**
  The combination of prior-guided voting and batch-specific soft labeling is well-justified and effectively integrated into the distillation loop.
  The enhancements (SRe²L++ baseline, real initialization, smoothed learning rate) are reasonable and necessary for a fair comparison.
* **Theoretical support:**
  The paper provides formal analysis showing that committee diversity contributes to broader gradient coverage and improved generalization.
  This strengthens the conceptual grounding of the method.
* **Empirical improvements:**
  Consistent gains across multiple benchmarks, including large-scale ImageNet-1K, demonstrate practical utility and scalability.
  The results are competitive and in several cases surpass existing state-of-the-art methods.

### Weaknesses
* **Computation cost and scalability:**
  The committee-based design involves multiple teachers and prior evaluation steps, which may substantially increase computation.
  The paper should report training time, GPU hours, and discuss trade-offs between accuracy and cost more explicitly.
* **Limited domain diversity:**
  The framework is validated only on image classification datasets.
  It would be valuable to explore its applicability to other modalities (e.g., graph or multimodal data) or to synthetic-to-real transfer tasks.

### Questions
1. How does CV-DD’s computational cost scale with the number of committee members? Can it be reduced by low-rank or partial gradient aggregation?
2. How robust is the prior-guided voting to noisy or overfitted teacher models? Would dynamic online re-weighting (instead of fixed priors) help?
3. Have you tested CV-DD on other domains (e.g., graph or cross-modal distillation) to confirm generality?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes CV-DD, a dataset distillation framework that introduces a Prior-based Voting Strategy on top of SRe²L. The authors also incorporate several training refinements, such as real-image initialization, data augmentation, and smoothed learning rate scheduling, claiming SOTA performance under multiple IPC settings. However, the novelty appears marginal, and the fairness of the experiments is questionable.

### Strengths
The overall presentation quality is good. The figures are visually clear and help the reader understand the methodology.

The motivation of performance-guided voting is intuitively reasonable in principle.

### Weaknesses
The proposed method appears to be a marginal modification over SRe²L. The core contribution, i.e., “Prior-based Voting”, is computationally expensive yet results in only marginal improvements, as shown in Table 4 (middle).

Several enhanced training tricks are adopted only for the proposed method and SRe²L++, like "Smoothed Learning Rate & Smaller Batch Size", which is not applied to other competing methods, leading to potentially unfair comparisons.

The comparison methods are limited. The main results only include two valid baselines, RDED and SRe²L, while another relevant method, CDA, which also follows a similar paradigm as SRe²L, is missing under most experimental settings. In addition, several comparison methods are relatively outdated.

The reported results are inappropriately cited. For instance, the MTT baseline is evaluated under an altered setting that deviates from its official configuration and appears to be intentionally adjusted in favor of the proposed method.

There are typos, e.g., in line 323, “Large scale dataset” should be written in lowercase.

### Questions
- What is the architecture used in Fig. 1? Please specify it clearly.

- Regarding the Prior-based Voting in Section 3.4 and Algorithm 1, it appears that distillation and evaluation must be conducted separately for each model in the committee set, which will incur substantial computational overhead. In contrast, SRe2L also relies on pretrained models, but those are standard models that can often be assumed to be readily available. 

- Moreover, the MTT results in Table 2 are substantially lower than the original reported performance. What accounts for this degradation? Was it re-implemented using an ensemble strategy? If so, this deviates from the original setting and may distort the comparison, especially since the reported results are significantly lower than those reported in the original MTT paper.

- The ablation study is incomplete and fails to isolate the contribution of key components. Specifically, the impact of the proposed Prior-based Voting is not evaluated independently. Moreover, several additional enhancements, such as Real Image Initialization, Data Augmentation, and Smoothed Learning Rate with Smaller Batch Size, are grouped together without showing their individual effects on performance. If these techniques are treated as default settings, they should be consistently applied to all compared methods (not only to SRe²L) to ensure a fair and unbiased comparison.

- The reported results are very selective. Why is there no cross-architecture performance reported for other methods in Table 2?

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
3

### Summary
This paper presents Committee Voting for Dataset Distillation (CV-DD), a framework designed to generate diverse and representative distilled datasets by aggregating knowledge from multiple heterogeneous models. The method incorporates a Prior Performance Guided Voting Strategy to adaptively weight model contributions and a Batch-Specific Soft Labeling (BSSL) mechanism to mitigate distribution shifts between synthetic and real data. Experiments on CIFAR-10/100, Tiny-ImageNet, and ImageNet-1K indicate that CV-DD achieves competitive performance across various IPC settings and demonstrates strong cross-architecture generalization.

### Strengths
1. Committee voting is a novel approach for dataset distillation that improves data representativeness through prior performance–based weighting.
2. CV-DD achieves state-of-the-art results across datasets, e.g., 59.5% on ImageNet-1K (IPC=50) versus 56.5% for RDED, with strong cross-architecture generalization.
3. Ablations confirm the effectiveness of voting temperature, prior-guided voting, and BSSL in reducing distribution shift.

### Weaknesses
**Major:**

1. A large portion of the paper focuses on building a strong baseline SRe2L++, which already incorporates several optimizations from recent SOTA methods such as EDC and RDED. Since SRe2L++ itself performs at a very high level, the additional gain from the committee voting mechanism appears modest, reducing the overall sense of novelty.
2. Although the method reports better per-iteration efficiency than G-VBSM, the overall training pipeline is complex. It requires pretraining all committee models, running a time-consuming distillation–evaluation loop to assess prior performance, and maintaining multiple teacher models during training for gradient weighting. The high pretraining cost and runtime memory usage may limit its practicality on large-scale datasets such as ImageNet-1K.
3. The ablation study shows that increasing the number of experts N from 2 to 3 leads to performance degradation. This contradicts the intuition that more models should improve robustness, suggesting that the committee voting mechanism may have an upper limit or introduce redundancy, which restricts its scalability.

**Minor:**

1. The citation related to DD in Line 122 is incorrect.
2. The CIFAR-100 IPC-10/50 results in Table 2 are inconsistent with those in Table 1.
3. Figure 9 appears slightly blurred.

### Questions
1. Please include a clearer ablation study to isolate the contribution of the prior performance guided voting strategy over the SRe2L++ baseline. Specifically, compare (1) the original SRe2L++ (single model), (2) SRe2L++ with an equally weighted committee plus BSSL, and (3) the full CV-DD with prior-guided voting plus BSSL.
2. Beyond the per-iteration time reported in Table 3, provide a more detailed analysis of computational cost, including total pretraining time, prior performance evaluation time, and peak GPU memory usage, especially on ImageNet-1K. This would help readers assess the overall efficiency compared with single-model methods such as SRe2L++ or RDED.
3. Regarding the performance drop with larger N, please provide further analysis on model diversity and the voting mechanism. It would be useful to discuss whether two experts already provide sufficient diversity and whether adding a third causes gradient conflicts or unstable weighting.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Committee Voting for Dataset Distillation (CV-DD), a novel framework that enhances dataset distillation by combining predictions and feature distributions from multiple models (committee members), rather than depending on a single model. The authors also establish a stronger baseline, SRe2L++, and propose an improved criterion called Batch-Specific Soft Labeling (BSSL), which better aligns the distributions of real and synthetic images. The objective of CV-DD is to create smaller, high-quality synthetic datasets that preserve the essential characteristics of large real datasets while minimizing overfitting and reducing model-specific bias.

### Strengths
1. Ensemble-based methods in dataset distillation remain an emerging area of research. 

2. The performance improvements reported in this paper are significant.

### Weaknesses
1. Although the proposed CV-DD framework enhances cross-model generalization by leveraging diverse architectures, its Batch-Specific Soft Labeling (BSSL) mechanism constrains the method to architectures that include Batch Normalization (BN) layers. This dependency limits the generalization and versatility of the approach when applied to models without BN components.

2. The visualization in Figure 6 suffers from poor distinguishability between curves due to the use of similar colors and marker styles for different data series. This makes it difficult for readers to clearly identify and interpret the individual trends. Additionally, the legend overlaps with the plotted data, further obscuring important details. It is recommended to improve the figure’s clarity by using more distinct color schemes or line styles and repositioning the legend to avoid overlapping with the data.

### Questions
1. **Comparison with SRe2L:**
 Could the authors include the performance results of SRe2L in the experiments? The reviewer is particularly interested in quantifying the improvement achieved by the enhanced SRe2L++ baseline over the original SRe2L.


2. **Ablation on Committee Composition:**
 The proposed method heavily depends on the quality and diversity of the committee members. However, the manuscript lacks a detailed discussion or analysis regarding this aspect. Could the authors include additional experiments that vary the composition or number of committee members to demonstrate the sensitivity and robustness of CV-DD to these factors?


3. **Dependence on Batch Normalization:**
 While CV-DD leverages the diversity of committee members, the proposed Batch-Specific Soft Labeling (BSSL) relies on architectures containing Batch Normalization (BN) layers. This dependence appears to limit the applicability of CV-DD to models without BN layers, such as Vision Transformers (ViTs) [1], which are widely adopted in modern computer vision. Could the authors discuss or experiment with how CV-DD might be extended to such architectures?


4. **Scalability with IPC:**
 How does the performance of CV-DD scale when the number of images per class (IPC) exceeds 50? Additional results or analysis in higher IPC regimes would help illustrate the scalability and practical limits of the proposed approach.


5. **Integration with Other Frameworks:**
 The proposed CV-DD presents an orthogonal contribution to existing literature and demonstrates strong potential for integration with other dataset distillation frameworks, such as RDED [2] and IGD [3]. Could the authors discuss or explore the feasibility and potential benefits of combining CV-DD with these frameworks?


[1] Alexey Dosovitskiy et al., An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, ICLR 2021

[2] Peng Sun et al., On the Diversity and Realism of Distilled Dataset: An Efficient Dataset Distillation Paradigm, CVPR 2024

[3] Mingyang Chen et al., Influence-Guided Diffusion for Dataset Distillation, ICLR 2025

### Soundness
2

### Presentation
2

### Contribution
2
