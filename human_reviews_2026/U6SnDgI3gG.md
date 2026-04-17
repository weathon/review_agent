# On the Spectral Differences Between NTK and CNTK and Their Implications for Point Cloud Recognition

- Decision: Accept (Poster)
- Scores: 6, 4, 8

## Abstract
The Convolutional Neural Tangent Kernel (CNTK) offers a principled framework for understanding convolutional architectures in the infinite-width regime. However, a comprehensive spectral comparison between CNTK and the classical Neural Tangent Kernel (NTK) remains underexplored. In this work, we present a detailed analysis of the spectral properties of CNTK and NTK, revealing that point cloud data exhibits a stronger alignment with the spectral bias of CNTK than images. This finding suggests that convolutional structures are inherently more suited to such geometric and irregular data formats. Based on this insight, we implement CNTK-based kernel regression for point cloud recognition tasks and demonstrate that it significantly outperforms NTK and other kernel baselines, especially in low-data settings. Furthermore, we derive a closed-form expression that connects CNTK with NTK in hybrid architectures. In addition, we introduce a closed-form of CNTK followed by NTK, while not the main focus, achieves strong empirical performance when applied to point-cloud tasks. Our study not only provides new theoretical understanding of spectral behaviors in neural tangent kernels but also shows that these insights can guide the practical design of CNTK-based regression for structured data such as point clouds.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper finds that CNTK is more suitable for point cloud data than image data, and thus proposes a CNTK-based kernel regression method for point cloud recognition tasks. Experimental results demonstrate its effectiveness, while providing a new theoretical explanation for the spectral characteristics of NTK.

### Strengths
1.This paper explains the spectral characteristics of NTK from a new perspective and finds that CNTK is more suitable for point cloud data. The formulas are rigorous and correct, and the experimental results also demonstrate the rationality of this theory.

### Weaknesses
1.This paper only conducts training on the ModelNet dataset. Experiments on ScanObjectNN, which is more in line with real-world scenarios and more challenging, should be added to demonstrate the practicality of the method.

### Questions
1.The relevant work done in this paper on shared MLP should be extendable to more advanced point cloud models, such as PointMLP. Have the authors made any relevant attempts?

### Soundness
3

### Presentation
3

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
The paper analyzes the spectral comparison between Convolutional Neural Tangent Kernel (CNTK) and Neural Tangent Kernel (NTK). Through a series of theoretical analysis and validation via synthetic data, the paper concludes that convolutional structures are inherently more suited to irregular point cloud data.

### Strengths
The theoretical comparison between CNTK and NTK can potentially guide architectural search of point cloud tasks.

### Weaknesses
* The writing of the paper is hard to follow, hard to understand. Though it is a theory paper, the presentation can be made much more accessible by explaining the intuition behind and visualization. Figure 1 seems to have such attempt, but it is not explained well.
* The conclusion reached by the paper is a well-known fact from empirical experience. The paper only decorates it with some theoretical proof.
* The conclusion that “convolutional structures are more suited to irregular point cloud data” is supported by Figure 2. But it only has one point cloud dataset and two image datasets, which cannot represent “point cloud” and “image”.
* In the experiments, the result shows PointNet performs better than PointNTK under all settings. So it is hard to understand what is the practical implication of the paper.

### Questions
What can be some practical implication of the paper given PointNTK performs worse than PointNet?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents the first systematic comparison of the spectral properties of the Neural Tangent Kernel (NTK) and the Convolutional NTK (CNTK). The authors formally prove that for data with a tensor structure, CNTK consistently exhibits a broader eigenvalue spectrum and a smaller mean eigenvalue compared to NTK. Based on this insight, they propose that CNTK's spectral bias is inherently better suited for geometric data like point clouds. This hypothesis is validated by introducing a metric for "Convolutional Suitability" and demonstrating experimentally that point clouds align more strongly with CNTK's properties. Finally, the authors propose PointNTK, a CNTK-based kernel regression method, which achieves strong performance on point cloud recognition, particularly outperforming training-based baselines in low-data settings.

### Strengths
- Novel and Insightful Theoretical Contribution: 

    The paper provides the first systematic spectral comparison of NTK and CNTK, offering a novel theoretical lens to understand the inductive biases of convolutional architectures. The introduction of mK and βK as quantitative metrics and the concept of "Convolutional Suitability" are significant contributions that provide theoretical guidance on how to choose the right network architecture for a given type of data.
﻿

- Combining theory with practice: 

    A key strength of this work is that it clearly connects theory with practice. The authors start from formal proofs to formulate a verifiable hypothesis—that point clouds are more "convolutionally suitable" than images—and then compellingly validate this hypothesis through well-designed experiments.
﻿

- Explanation a Fundamental Question in Point Cloud Processing: 

    This paper tackles a highly relevant question in the 3D vision community: why are convolutional-like structures (such as the shared-MLPs in PointNet) so effective for point cloud data. By approaching this from a theoretical kernel perspective, the work provides a novel explanation.

### Weaknesses
- [Minor] Confusing performance:
﻿
    The results in Table 1 present a slightly confusing result. For the ModelNet10_6 dataset, the vanilla 1dCNTK (91.96%) outperforms the more complex PointNTK (91.19%). This seems to contradict the paper's motivation for adding MLP layers . A discussion on why this might occur would be beneficial.
﻿
- [Minor] Potential Discrepancy Between Ablation Results and the Unorderedness Argument:
﻿
    The paper makes a strong argument that a kernel size greater than 1 is detrimental due to the unordered nature of point clouds. Following this logic, one would expect a sharp performance drop when the kernel size increases from 1 to 2. However, the experimental results (Figure 4) show a gradual decline rather than a steep fall. This gentle degradation seems not entirely consistent with the theoretical expectation that aggregating unordered points would cause severe disruption. The paper would be strengthened by a discussion or explanation of this phenomenon . 
﻿
- [Minor] Practical Limitation of the β_K Metric:

    The use of the initial layer's metric, β_K(0), as a proxy for "Convolutional Suitability" is a practical simplification but also a limitation. As Figure 1 shows, β_K evolves with depth. The paper would be stronger if it discussed this evolution or provided bounds on its variation, which would enhance the completeness of the theory.
﻿
- [Minor] The Core Concept of "Convolutional Suitability" Lacks a Formal Definition

    The paper introduces "Convolutional Suitability" as a key concept but fails to provide a formal definition. It is only mentioned that 1 - β_NTK(0) can be "interpreted as" this metric. For clarity and to facilitate future work, this concept should be formally defined when it is first introduced.

### Questions
- On the Performance of the PointNTK Model: 
    I noticed in Table 1 that for the ModelNet10_6 dataset, the simpler 1dCNTK model slightly outperforms the PointNTK hybrid model. This is counter-intuitive given that the MLP layers are introduced to enhance performance. What is your interpretation of this result? Is it related to dataset-specific properties?
﻿
- On the Smooth Decline in the Kernel Size Ablation: 
    Your argument about point cloud unorderedness suggests that a kernel size k > 1 should be highly detrimental. I would have expected a sharp performance drop when moving from k=1 to k=2. Instead, Figure 4 shows a smooth, gradual decline. What is your intuition for this gentle degradation? 
﻿
- On β_K(0) as a Proxy for Suitability: 
    You use β_K(0) as a practical proxy for "Convolutional Suitability" across the entire network. While I understand the computational challenge of evaluating β_K(L) for deep layers, do you have any evidence or theoretical argument suggesting that β_K(0) is a reliable representative? An answer here would strengthen the claim that this initial-layer metric is sufficient.

### Soundness
3

### Presentation
3

### Contribution
3
