# SF-Mamba: Rethinking State Space Model for Vision

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
The realm of Mamba for vision has been advanced in recent years to strike for the alternatives of Vision Transformers (ViTs) that suffer from the quadratic complexity.
While the recurrent scanning mechanism of Mamba offers computational efficiency, it inherently limits non-causal interactions between image patches.
Prior works have attempted to address this limitation through various multi-scan strategies; however, these approaches suffer from inefficiencies due to suboptimal scan designs and frequent data rearrangement. Moreover, Mamba exhibits relatively slow computational speed under short token lengths, commonly used in visual tasks.
In pursuit of a truly efficient vision encoder, we rethink the scan operation for vision and the computational efficiency of Mamba.
To this end, we propose SF-Mamba, a novel visual Mamba with two key proposals: auxiliary patch swapping for encoding bidirectional information flow under an unidirectional scan and batch folding with periodic state reset for advanced GPU parallelism.
Extensive experiments on image classification, object detection, and instance and semantic segmentation consistently demonstrate that our proposed SF-Mamba significantly outperforms state-of-the-art baselines while improving throughput across different model sizes.
We will release the source code after publication.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces SF-Mamba, a vision model designed to improve the efficiency and performance of State Space Models (SSMs) for visual tasks. It proposes two main contributions: Auxiliary Patch Swapping to enable bidirectional information flow in a unidirectional scan, and Batch Folding with Periodic State Reset to improve GPU parallelism for short sequences common in vision.

### Strengths
The authors correctly identify that visual Mamba models are often slow not just due to scan strategies but because of suboptimal GPU utilization on the short sequences common in vision tasks. The "Batch Folding" technique is a clever, hardware-aware solution that provides significant speedups.

### Weaknesses
**Limited Novelty**

Overlap with Adventurer [1]: The core idea of swapping auxiliary tokens to enable bidirectional information flow is conceptually almost identical to the flip operation between consecutive blocks proposed in Adventurer (CVPR 2025). Adventurer also uses this technique to address the causality constraint in unidirectional visual models. The absence of any discussion, citation, or empirical comparison to this highly relevant prior work is a major oversight and significantly weakens the paper's claim to novelty.

Batch Folding as an Engineering Optimization: While the Batch Folding technique is shown to be effective, it functions more as a low-level implementation optimization or an "engineering trick" rather than a novel academic contribution. Its impact is valuable for performance but its conceptual depth is limited for a top-tier conference paper.


**Questionable Motivation**

The paper's motivation rests on the claim that multi-directional scans are inherently slow, but this is not convincingly proven and contradicts previous findings.

Contradiction with VMamba: The authors claim inefficiency in multi-directional scans, but this conclusion is challenged by the original VMamba paper, which demonstrated that a multi-directional scan could achieve comparable throughput to a single-directional one. This discrepancy is not addressed.


Unfair Ablation Study (Table 3): The ablation study in Table 3, which aims to show the slowness of multi-scan methods, is methodologically flawed. The comparisons do not keep the parameters and MACs constant across different scan methods. For example, a "parallel bi-scan" would naturally have a higher computational cost unless channel dimensions are halved to ensure a fair comparison, which does not appear to be the case here. This invalidates the conclusion that the proposed uni-scan with swapping is inherently faster than a properly configured multi-scan architecture.


**Writing and Paper Structure**

An extensive amount of space in the main paper is used to describe the MambaVision macro-architecture. This detailed background could be moved to the appendix. This would free up valuable space to provide a more thorough analysis of the results on downstream tasks like COCO and ADE20K, which currently feel rushed and lack sufficient detail.


**Marginal Performance on Downstream Tasks**

While SF-Mamba excels on ImageNet classification, its advantages diminish significantly on more complex downstream tasks. The performance improvements over the MambaVision baseline are minimal for semantic segmentation on ADE20K.
More critically, on the MS COCO object detection and instance segmentation tasks, some SF-Mamba configurations underperform the MambaVision baseline they are supposed to improve upon.


[1]. Causal Image Modeling for Efficient Visual Understanding. CVPR2025.

### Questions
Please check weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents SF-Mamba, an improved visual Mamba architecture for vision tasks. SF-Mamba introduces several innovations: auxiliary patch swapping with an extra token, which enables bidirectional information flow during a unidirectional scan, and batch folding with periodic state reset, which enhances GPU parallelism. Extensive experiments demonstrate that SF-Mamba significantly outperforms existing models in terms of accuracy and throughput across image classification and segmentation tasks.

### Strengths
1. The motivation is good. Previous methods addressed the problem from the perspective of multiple scans, whereas this paper innovatively addresses the Mamba architecture's issue of sequential reasoning by focusing on a single-scan approach for causal information swapping.

2. By restructuring the tensor dimensions of batch data, the paper improves parallelism from a GPU computation perspective, which benefits the acceleration of large-scale training.

3. The experiments are comprehensive, covering CNN architectures, transformer architectures, and hybrid architectures. Detailed ablation studies and supplementary materials provide a thorough exploration of the proposed method.

### Weaknesses
1. The performance improvement of the model is not significant. The accuracy improvement is not obvious compared to the baseline Mamvbavision. Moreover, the gain in speed is not so significant.

2. The comparison models do not include Fast R-CNN or Faster R-CNN for the object detection experiment, and the comparison in the validation set is insufficient.

3. The method of exchanging token positions to achieve contextual structure interaction may not be the optimal approach. The current ablation studies do not directly prove that the performance improvement is caused by the position swapping.

4. What about other hyperparameters designed in the results in Figure 4? Does it have an optimal design for performance and speed?

### Questions
Please see the weakness. I would raise the ratings if the rebuttal addressed the questions.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes SF-Mamba, a novel visual encoder addressing limitations of existing Vision Transformers and visual Mamba models. ViTs suffer from quadratic complexity, while visual Mamba faces non-causal information flow constraints and inefficiency with short tokens. SF-Mamba introduces two core innovations: auxiliary patch swapping for bidirectional information flow under unidirectional scanning, and batch folding with periodic state reset to enhance GPU parallelism. Extensive experiments on image classification, object detection, and segmentation demonstrate superior accuracy-throughput trade-offs compared to SOTA baselines.

### Strengths
1. This paper targets critical pain points of visual Mamba (causality constraint, short-sequence inefficiency) with lightweight, non-intrusive solutions.
2. Comprehensive validation across three fundamental vision tasks (classification, detection, segmentation) with consistent performance gains.
3. Practical optimizations (e.g., adaptive $\(B_1\)$, Triton kernel for swapping) enhance real-world applicability, with code release planned.

### Weaknesses
1. The macro-architecture heavily relies on MambaVision’s hybrid (Mamba+Attention) design, lacking significant innovations in overall network structure.
2. Ablation studies on auxiliary token initialization and discard timing are limited; deeper analysis of their impact on different tasks is needed.
3. No discussion on generalization to ultra-high-resolution images or low-resource devices (e.g., edge GPUs), restricting scope insights.

### Questions
Please refer to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This manuscript describes a modification to vision state space models. To improve the processing speed of mamba-based vision models, the authors proposed to 1) auxilliary patch swapping for bidirectional information flow, and 2) batch folding with periodic state reset. Experiments shows positive results on these components.

### Strengths
+ The manuscript is well-presented and easy to follow.
+ It is good to see analysis on the inference speed of the mamba based model.
+ The reset trick is interesting.

### Weaknesses
+ The proposed method is only testifed on MambaVision. It should be applied to more mamba-based models to support the claim of "Rethinking State Space Model for Vision".
+ Swapping last token is not equivant to bi-directional scan, and the author failed to prove the superiority of swapping last token. As in table 3, if the attention is removed, swapping last token worse than parallel bi-scan and even series bi-scan. Since attention itself is a undirectional operation, this seems that switching to swapping last token is not working but attention works.

### Questions
+ In the manuscript of MambaVision, they utilized the same hardware, but they reported a throughput of  3670 img/s for MambaVision-B. However, in this manuscript, the speed is downgraded to 2974 img/s. Why? If jittering exists, please report mean and std over multiple test runs.
+  Why Mamba kernel requires 32 parallel threads? If current mamba kernel is not suitable for short sequence of image classification, why do we need parallel scan?  Why do we need mamba? The performance drops severly in Tab.3 if attention are removed.
+ In Table 3, why the inference troughput drop when attention is removed? If that so, the inference scheme may not be appropriate. Please report performance under large batch size, say, 2048 as in SHViT and EfficientViT.
+  Can the proposed method be applied to other mamba-based model, say, Vim, VMamba and their variants?

### Soundness
2

### Presentation
3

### Contribution
2
