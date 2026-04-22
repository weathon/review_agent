# Cross-Modal Feature Disentanglement with Contrastive Task Alignment for Multi-Modal Image Fusion

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Multi-modal image fusion suffers from feature entanglement, where modality-specific, content-specific, and task-specific information becomes conflated in unified representation spaces, leading to suboptimal fusion quality and limited generalization. This paper proposes Cross-Modal Feature Disentanglement with Contrastive Task Alignment (CMD-CTA), a framework that addresses this fundamental challenge through mathematically motivated feature separation and semantic alignment, supported by both theoretical analysis under idealized assumptions and empirical evidence on real-world fusion benchmarks. The approach introduces two key innovations: (1) differentiable orthogonal feature decomposition that encourages separation into content, modality, and task subspaces under information-theoretic sufficiency constraints; and (2) contrastive task alignment that establishes semantic bridges through learnable prototypes and multi-granularity contrastive learning. We further adopt hybrid Vision Mamba–Swin backbone to couple linear-complexity long-range modeling with windowed locality, thereby reducing parameters while preserving context. Extensive experiments across six fusion tasks and downstream object detection demonstrate 5.8--7.3\% improvements over state-of-the-art methods, 6.1\% higher mAP\@0.5, and 15.7$\times$ parameter efficiency. This empirically validated framework for representation learning in multi-modal fusion has broad implications for computer vision and autonomous systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a Cross-Modal feature Disentanglement and Contrastive Task Alignment  framework, which disentangles features based on mathematically formulated principles of feature separation and semantic alignment. The work is motivated by the observation that standard deep learning architectures tend to share representations, causing essentially different semantic components to become mixed due to differences in information content. Across extensive experiments on six fusion tasks, the proposed algorithm achieves state-of-the-art performance.

### Strengths
The method effectively performs feature disentanglement.

The proposed algorithm improves both efficiency and performance.

### Weaknesses
- The algorithmic section is confusing. Although the underlying theory is mathematically derived, the overall algorithmic framework is not introduced, and many module pipelines in Figure 2 are not adequately explained, leaving the paper incomplete.

- Orthogonalization can only ensure that three sets of orthogonal bases do not interfere with each other; it cannot guarantee which set corresponds to content/modality/task. The authors should strengthen the proof for this part.

- The paper formulates prototype evolution as an SDE with Brownian noise but does not implement the key parts of the SDE. For example, there is no definition of L_"prototype" ; Equation (6) lacks a noise term and does not use β anywhere.

- The use of hard negatives in Equation (6) is incorrect: it adds the mean of the hard negatives H_kto the prototype, even though H_kdenotes difficult samples that are far from the prototype. This definition is puzzling.

- Some symbols in Equation (8) are undefined—what is Y_N? The “symmetric InfoNCE formulation” cited for Equation (9) is also puzzling, as Eq. (9) only writes a single direction (from z_1to z_2).

- The experimental section lacks necessary explanations and citations for the baseline (comparison) methods, and the dataset sources are insufficiently explained. Table 1 is rather rough. The content of Table 2 is also puzzling, lacking explanations of terminology and containing errors.

- Lack of  training details description.

Overall, the paper is rough from algorithm to experiments: definitions are lacking, the logic is not smooth, and the experiments are simplistic and crude. In my view, it does not meet the standard of a qualified paper. Therefore, I am unlikely to raise my score for this paper.

### Questions
In traditional spatial-domain image fusion methods, one line of work computes feature saliency to obtain a mask that selects or allocates modality features. In essence, how does this kind of “disentanglement” differ from yours?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CMD-CTA, a novel framework that tackles the core problem of feature entanglement in multi-modal image fusion. It achieves this through two key innovations: a differentiable orthogonal decomposition module that mathematically separates features into content, modality, and task-specific subspaces, and a contrastive task alignment strategy that uses dynamically evolving prototypes to semantically align these features.

### Strengths
1.Theoretically-Grounded Disentanglement: Introduces a principled, mathematically sound framework for feature separation via orthogonal decomposition, directly addressing the core issue of feature entanglement with theoretical guarantees.

2.Unified yet Specialized Performance: Achieves state-of-the-art results across six diverse fusion tasks with a single model, eliminating the need for task-specific architectures while maintaining specialized performance.

### Weaknesses
1.The framework combines multiple loss components and novel mechanisms, making it potentially sensitive to hyperparameter tuning for optimal performance across different tasks.

2.The contrastive alignment relies on predefined task prototypes, which may limit flexibility for entirely new tasks without retraining or prototype redesign.

### Questions
1.How is the initial set of task prototypes defined, and to what extent does the framework's performance depend on this initialization? Could it adapt to a new downstream task without manually designing a new prototype?

2.While orthogonality minimizes mutual information for Gaussian features, to what extent does this hold for the actual, highly non-Gaussian features learned by the deep network? Is the empirical success driven more by the mathematical constraint or the overall architecture?

3.The final fused image is generated by the content feature 'selectively aggregating' modality-specific features through a decoder. What is the specific mechanism and criterion for this 'selective aggregation'?

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
3

### Summary
The paper tackles feature entanglement in multi-modal image fusion, where modality-specific, content-specific, and task-specific information become mixed and harm generalization. It proposes CMD-CTA, which performs differentiable orthogonal decomposition to split features into content, modality, and task subspaces, with an information-theoretic motivation for reducing mutual information, and aligns task semantics via contrastive learning using dynamically evolving prototypes across multiple granularities. Extended experiments on tasks report consistent gains over prior work, strong parameter efficiency, and improved downstream detection performance.

### Strengths
- The paper identifies feature entanglement as a key factor that degrades fusion quality and generalization, which is insightful.
- CMD-CTA improves across metrics supporting generalization claims and its efficiency is compelling relative to heavy baselines.
- The objective differs from traditional fusion methods and is interesting, leading to superior visual performance.

### Weaknesses
- The phrase “the semantic gap between pixel-level fusion requirements and high-level representations” in line 089 lacks a detailed explanation and theoretical grounding.
- The paper’s task formulation is ambiguous: it appears to conflate multi-task fusion objectives (e.g., MEF, MFF) with downstream tasks (e.g., object detection, segmentation). The different meanings and goals of these task types are not disentangled, which undermines the motivation.
- The claim in lines 191–193—"from an information-theoretic perspective …"—appears inconsistent with the principle that a fused image should retain comprehensive information; minimizing mutual information seems to contradict that goal.
- The MI bound in Eq. (3) relies on Gaussianity and zero cross-covariance, yet the paper provides no empirical MI estimates to verify that MI is actually reduced. Given that many image-fusion methods increase MI [1–3], please report MI quantitatively (e.g., before/after the proposed decomposition and against baselines) to validate the claimed “theoretical guarantees for mutual-information minimization” and to enable fair comparison
- The stochastic differential equation in Eq. 5 is conceptually appealing, but its discrete implementation specifics (e.g., step size, noise term) are not quantified. 
- The description “weights $w_g$ are determined by mutual information contribution” in Eq.15 is not operationalized with an explicit estimation procedure.

[1] Zhao Z, Xu S, Zhang C, et al. DIDFuse: Deep image decomposition for infrared and visible image fusion[J]. arXiv preprint arXiv:2003.09210, 2020.

[2] Zhang Y, Liu Y, Sun P, et al. IFCNN: A general image fusion framework based on convolutional neural network[J]. Information Fusion, 2020, 54: 99-118.

[3] Zhu P, Sun Y, Cao B, et al. Task-customized mixture of adapters for general image fusion[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2024: 7099-7108.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a new framework, CMD-CTA, for multi-modal image fusion. The core problem it addresses is "feature entanglement," where modality, content, and task-specific information are conflated in a shared representation space. The authors introduce two main technical contributions: 1) a differentiable orthogonal feature decomposition method to enforce separation between feature subspaces (content, modality, task), and 2) a contrastive task alignment mechanism with dynamic prototypes to build semantic connections across tasks. The method is evaluated on six different fusion tasks and demonstrates state-of-the-art performance in terms of quantitative metrics and visual quality, while also claiming high parameter efficiency.

### Strengths
1. The paper clearly identifies and articulates the problem of feature entanglement in multi-modal fusion, which is a significant and practical challenge.
2. The authors have conducted experiments across an impressive range of six distinct fusion tasks (IVIF, MFIF, MEIF, etc.). This comprehensive evaluation strongly supports the claim that the proposed method is a generalized framework rather than a task-specific solution.
3. The proposed CMD-CTA framework consistently outperforms previous state-of-the-art methods across all evaluated tasks, as shown in Table 1.

### Weaknesses
1. The paper's novelty appears limited. While the framework is new, its core components, such as feature orthogonalization and multi-granularity contrastive learning, are well-established techniques. Furthermore, the architecture seems to be an assemblage of existing modules (e.g., Mamba, Swin) without sufficient justification. The manuscript lacks a clear rationale for these specific architectural choices. For instance, what is the precise benefit of Mamba's long-range dependency modeling for this task? Why is a hybrid Mamba-Swin architecture superior to simpler or more conventional backbones like standard CNNs or ViTs?
2. The ablation study is unconvincing as it only assesses the contributions of the orthogonalization and contrastive learning components. Crucially, it omits any ablation on the core architectural modules. To validate the design choices, experiments demonstrating the necessity and effectiveness of Mamba, GNN, and other key structural components are essential.
3. here is an apparent contradiction regarding the model's efficiency. The authors claim a highly lightweight model (e.g., only 3.15M parameters), yet the architecture incorporates several complex modules typically associated with a large parameter count. The manuscript should provide a detailed breakdown of the parameter distribution across modules to clarify how this efficiency is achieved and substantiate the claim.
4. The manuscript fails to provide any details or sensitivity analysis for the hyperparameters (α, β, and γ) in the loss function. It is unclear how these values were chosen or whether they are consistent across different tasks. For reproducibility, these values should be explicitly stated and their selection process justified.
5. The set of quantitative evaluation metrics is incomplete. For a more comprehensive analysis, it is advisable to include information-theoretic metrics, such as Entropy (EN), which are commonly used and highly relevant in this research area.
6. There are critical issues in Table 1. First, the SCD score for MUFusion in sub-table E is reported as a negative value. This is highly unusual and requires clarification—is it a typo, an evaluation error, or an actual result? If the performance is indeed that low, a more competitive baseline should be included. Second, the rightmost sub-table is cluttered and poorly formatted, which severely impairs its readability.
7. The bibliography requires substantial revision. It is replete with inaccuracies, including frequent discrepancies in author names and publication years that do not match the original sources.

### Questions
Please see Weaknesses section

### Soundness
2

### Presentation
2

### Contribution
2
