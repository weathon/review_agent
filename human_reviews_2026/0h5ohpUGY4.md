# Understanding Dataset Distillation via Spectral Filtering

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Dataset distillation (DD) has emerged as a promising approach to compress datasets and speed up model training. However, the underlying connections among various DD methods remain largely unexplored. In this paper, we introduce UniDD, a spectral filtering framework that unifies diverse DD objectives. UniDD interprets each DD objective as a specific filter function applied to the eigenvalues of the feature-feature correlation (FFC) matrix to extract certain frequency information of the feature-label correlation (FLC) matrix. In this way, UniDD reveals that the essence of DD fundamentally lies in matching frequency-specific features. Moreover, we characterize the roles of different filters. For example, low-pass filters, \eg, DM and DC, capture blurred patches, while high-pass filters, \eg, MTT and FrePo, prefer to synthesize fine-grained textures and have better diversity. However, existing methods can only learn the sole frequency information as they rely on fixed filter functions throughout distillation. To address this limitation, we further propose Curriculum Frequency Matching (CFM), which gradually adjusts the filter parameter to cover both low- and high-frequency information of the FFC and FLC matrices. Extensive experiments on small-scale datasets, such as CIFAR-10/100, and large-scale ImageNet-1K, demonstrate the superior performance of CFM over existing baselines and validate the practicality of UniDD.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel framework, UniDD, that unifies existing Dataset Distillation (DD) methods from the perspective of spectral filtering. The authors interpret different distillation objective functions as filtering functions applied to the feature-feature correlation matrix (FFC) and the feature-label correlation matrix (FLC), revealing that the essence of dataset distillation is "matching frequency-specific features." To address the limitations that existing methods use fixed filters and thus cannot simultaneously learn across multiple frequency bands, the authors further propose Curriculum Frequency Matching (CFM). This method dynamically adjusts filter parameters, gradually covering from low to high frequencies during training, thereby achieving a balance between consistency and diversity in the synthetic data.

### Strengths
The presentation is clear and easy to follow.

### Weaknesses
1.	The paper lacks sufficient rigorous theoretical justification, and UniDD's unified analysis relies on idealized assumptions. For example, the derivation of the DC method considers only linear classifiers, while the entire network is actually considered, which reduces readability and persuasiveness.
2.	CFM requires computing and matching FFC/FLC on each feature layer and applying dynamic filtering, which inevitably incurs significant computational overhead when the network is deep.

### Questions
1.	What are the advantages of the proposed “spectral filtering” compared to transforming an image from the spatial domain to the frequency domain?
2.	Can the authors directly demonstrate the effects of what they define as low-frequency matching and high-frequency matching, rather than only visualizing them via filtered images as in Fig. 1?
3.	Please provide a comparison of computational complexity metrics.
4.	Can the ablation study include comparisons against baseline methods? Moreover, in Table 5, the performance improvement of the designed loss over the basic classification loss seems relatively weak.
5.	In Table 2, CFM achieves very strong performance on CIFAR-100. Does it surpass training on the full real dataset (Whole Dataset), or how close does it get?

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
This work explores dataset distillation from a spectral perspective and introduces a new method called Class-wise Frequency Matching (CFM), which aligns the frequency characteristics of real and distilled data. The authors argue that dataset distillation naturally acts like a frequency filter, keeping the important, discriminative details while removing redundant information. Experiments on CIFAR-10 and ImageNet-100 support this idea and show that CFM can improve generalization performance, lending further evidence to their spectral interpretation. To the best of my knowledge, this is the first work that attempts to unify the theoretical understanding and methodological design of dataset distillation under a single spectral framework.

### Strengths
1. The paper presents, to the best of my knowledge, the first theoretical framework that systematically analyzes dataset distillation. The spectral analysis connects data condensation with the preservation of high-frequency, discriminative components. The derivations are careful and mathematically consistent, giving a logically sound and interpretable explanation of how synthetic data behave as spectral filters during training.

2. The proposed CFM algorithm makes the theoretical idea practical. CFM matches the class-wise frequency distributions between real and distilled datasets to maintain frequency balance during training (e.g., Eq. 10). This turns the paper’s insight into a concrete algorithm, and the results show steady accuracy gains across different baselines, suggesting that frequency regularization can help dataset distillation generalize better.

3. The writing is clear and the work is easy to reproduce. The mathematical reasoning is well organized, and the authors describe the implementation details for spectral energy computation in sufficient depth. They also release source code for both the analysis and CFM method.

### Weaknesses
The experiments are mainly conducted on smaller datasets like CIFAR-10 and ImageNet-100, so it remains uncertain whether the same spectral behavior and CFM improvements would hold on larger datasets such as ImageNet-21K. Adding results on a larger benchmark would make the conclusions more broadly convincing and show the robustness of the proposed approach. In addition, while CFM brings consistent improvements, its performance is still slightly behind the strongest existing methods. As shown in Table 3, the gains are modest and sometimes below strong baselines like RDED or FRePo, suggesting that there is still room for further refinement and optimization of the spectral alignment idea.

### Questions
Please check Weaknesses for details.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposed the combination technique, which produce both blurring (low-pass filter) and fine-grained (high-pass filter) image for dataset distillation. Experiments across various benchmarks demonstrate its superior performance over existing baselines.

### Strengths
1. The paper is well written and easy to follow.
2. The main idea is clear, and it is easy to understand how it can lead to performance improvement.

### Weaknesses
1. The idea of using both blurring and fine-grained techniques for DD is not novel, as it has already been explored in [1].
2. The proposed method only works under high IPC settings (greater than 1 IPC).
3. The paper lacks comparisons with several state-of-the-art methods, such as [1] and [2].

[1] Enhancing Dataset Distillation via Non-Critical Region Refinement. CVPR 2025
[2] Dataset Distillation via the Wasserstein Metric. ICCV 2025

### Questions
See weaknesses section!

### Soundness
2

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
4

### Summary
This study points out that existing dataset distillation objective functions have developed in a fragmented manner, and their relationships have not been investigated. This study proposes a new framework, UniDD, utilizing spectral filtering. Furthermore, UniDD enables to categorize previous researches based on the frequency bands the filter function focuses upon. Additionally, this work proposes Curriculum frequency matching (CFM), enabling the learning of both low and high-frequency information, thereby mitigating the limitations of prior research that only learned information from specific frequency bands.

### Strengths
1. It is both highly impressive and fascinating that reinterpreting the objective function of existing dataset distillation from a spectral filtering perspective and proposing a unified framework.

2. The proposed UniDD is a simple form, and the logical flow of expressing prior research using UniDD is also readily comprehensible.

3. The motivation and core idea of CFM, which adaptively learns multiple frequency information based on UniDD, is natural and intuitive.

### Weaknesses
1. UniDD was proposed under assumptions that are prone to violation. Overall, this work develops UniDD’s logic under the assumption that the parameters of the feature extractor $\phi$ are fixed and not considered. Consequently, both gradient matching and trajectory matching only considered the parameters of a single layer following the feature extractor. However, in dataset distillation, the parameters of the feature extractor are typically not fixed, meaning the UniDD framework may not readily satisfy these conditions. This suggests that UniDD has limitations in terms of its applicability.

2. The objective functions for gradient matching and trajectory matching represent lower bounds for UniDD. This implies that achieving gradient matching and trajectory matching can be accomplished through low-frequency matching and high-frequency matching respectively; however, it is difficult to accept that the converse holds true. Therefore, the claim that gradient matching and trajectory matching correspond to low-frequency matching and high-frequency matching requires further clarification.

3. The theoretical analysis and experimental design in this study do not align well. UniDD integrates statistical matching, gradient matching, trajectory matching, and KRR from a spectral filtering perspective, and proposes CFM to learn diverse frequency information. However, the baseline used in all experimental results except Table 2 is the decoupled method, which is difficult to consider as included in UniDD. To demonstrate the superiority of CFM, I believe that it is necessary to conduct diverse performance comparison experiments against the recent works for each component included within UniDD.

4. Key previous literature is omitted. FreD[1], NSD[2], and NCFM[3] are prior studies addressing the frequency domain in dataset distillation; therefore, the relevance of the proposed idea must be discussed in the related works section.

[1] Frequency Domain-based Dataset Distillation

[2] Neural Spectral Decomposition for Dataset Distillation

[3] Dataset Distillation with Neural Characteristic Function: A Minmax Perspective

### Questions
1. To support the claim that DM and DC are low-frequency matching methods while MTT and KRR are high-frequency matching methods, I suggest plotting the spectral density for the synthetic images generated by each methodology.

2. I believe the synthetic label $Y_s$ need not necessarily be a one-hot label; treating it as an optimizable parameter does not alter the overall logic. Furthermore, in the experiments, soft labels based on fast knowledge distillation (FKD) were employed. I would be interested whether CFM remains effective when using optimized synthetic soft-labels.

3. When using soft labels as synthetic labels, more memory is required than when storing only synthetic inputs, as the labels themselves must also be saved. Furthermore, the memory usage of synthetic labels increases significantly as the number of classes grows, to a point where it becomes non-negligible[4]. Therefore, I am curious how the utilized memory budget was configured to ensure a fair comparison with the baseline.

4. I am interested in ablation studies concerning various frequency curriculum scheduling approaches. I am also curious whether this can be applied to parameterization methodologies such as FreD[1] and NSD[2], which only involve fixed frequency components in learning.

5. I would be interested whether CFM exhibits high performance even when IPC=1.

[4] Are Large-scale Soft Labels Necessary for Large-scale Dataset Distillation?

### Soundness
3

### Presentation
3

### Contribution
3
