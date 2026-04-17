# Dual-Kernel Adapter: Expanding Spatial Horizons for Data-Constrained Medical Image Analysis

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Adapters have become a widely adopted strategy for efficient fine-tuning of foundation models, particularly in resource-constrained settings. However, their performance under extreme data scarcity—common in medical imaging due to high annotation costs, privacy regulations, and fragmented datasets—remains underexplored. In this work, we present the first comprehensive study of adapter-based fine-tuning for vision foundation models in low-data medical imaging scenarios. We find that, contrary to their promise, conventional Adapters can degrade performance under severe data constraints, performing even worse than simple linear probing when trained on less than 1\% of the corresponding training data. Through systematic analysis, we identify a sharp reduction in Effective Receptive Field (ERF) as a key factor behind this degradation. Motivated by these findings, we propose the Dual-Kernel Adapter (DKA), a lightweight module that expands spatial context via large-kernel convolutions while preserving local detail with small-kernel counterparts. Extensive experiments across diverse classification and segmentation benchmarks show that DKA significantly outperforms existing Adapter methods, establishing new leading results in both data-constrained and data-rich regimes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
1.The paper addresses the challenge of applying parameter-efficient fine-tuning (PEFT) methods, such as Adapters, in medical imaging where extreme data scarcity is common. The authors argue that conventional Adapters underperform in these low-data settings. To mitigate this, they propose the Dual-Kernel Adapter (DKA), a novel module that incorporates dual-path depthwise convolutions—one with a large kernel and one with a small kernel. This design is intended to allow the adapter to effectively capture both local fine-grained details and global contextual information. The results show that DKA significantly outperforms conventional Adapters, LoRA, and even full fine-tuning, especially under extreme data scarcity (1% of training data), achieving state-of-the-art results on multiple 2D and 3D medical image segmentation benchmarks.

### Strengths
1.This work is presented as the first comprehensive study of adapter-based fine-tuning specifically tailored for low-data medical imaging scenarios.
2.The Dual-Kernel Adapter (DKA) is a novel architectural design within the PEFT space. Explicitly integrating multi-scale spatial feature extraction (via dual-path convolutions with different kernel sizes) into the adapter structure is a unique and effective contribution to improving PEFT performance in a difficult data regime.
3.The technical claims are robustly supported by strong experimental evidence, showing DKA's significant outperformance against well-established baselines like conventional Adapters, LoRA, and full fine-tuning.
4.The results are highly significant for the medical imaging and broader computer vision communities. A PEFT method that achieves state-of-the-art results with as little as 1% of the training data is an immense breakthrough for high-cost, high-stakes domains like healthcare.

### Weaknesses
1. The choice of 51×51 as the large kernel seems somewhat arbitrary. While ablations justify it empirically, a more principled analysis (e.g., based on image resolution or pathology size) would be valuable. And is this optimal across all medical imaging modalities?
2.Given the added complexity of the dual-kernel design, please provide a comprehensive analysis of the latency (inference time) and FLOPs of DKA relative to LoRA and the conventional Adapter, across the same backbone and hardware.
3.Although the paper demonstrates DKA’s superiority across multiple datasets and settings, it does not systematically investigate scenarios where DKA might fail. For example: Could the large kernel introduce noise in tasks where pathological structures are extremely small or sparsely distributed? Is DKA still robust under extreme class imbalance (e.g., in ISIC-2019, where some classes have very few samples)? Are there specific backbone architectures (e.g., pure CNNs) for which DKA provides no benefit or even degrades performance?
4.As noted in Appendix B.2, segmentation experiments use a batch size of 1 due to GPU memory constraints. This may lead to highly noisy gradient estimates, especially under low-data regimes and introduce optimization instability that confounds fair comparison across methods. In real-world deployment scenarios, larger batch sizes are typically used.
5.While the paper uses ERF visualizations and frequency-domain analysis to argue that large kernels expand the receptive field, the authors fails to establish a theoretical or information-theoretic link between large kernels, improved generalization, and robustness under data scarcity;

### Questions
1.The evaluation focuses exclusively on 2D and 3D segmentation. Does the DKA demonstrate similar compelling performance improvements when applied to other common medical imaging tasks, such as disease classification or object detection, under the same low-data constraints?
2.Is the 51×51 large kernel in DKA feasible for 3D medical imaging (e.g., BRATS with 3D MRI volumes)? A direct extension to a 51×51×51 kernel would lead to prohibitive parameter and computational costs.
3.While Appendix F.10 shows minimal overhead, could you explain on how DKA scales with image resolution? 
4.The Effective Receptive Field (ERF) is computed based on the magnitude of input-output gradients, which reflects spatial influence but not semantic relevance. Have the authors considered complementary analyses to verify that the expanded ERF indeed captures meaningful contextual information rather than just broader but irrelevant activations?
5.Are there any datasets or settings where DKA underperforms standard adapters?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors conduct a thorough study on the effect of adapters in fine tuning large-scale pretrained models in the low data regime prevalent in the medical image segmentation and classification. The authors notice a sizable reduction in performance in limited data scenarios and attribute it to a reduced effective receptive field (ERF). Accordingly, they use known methods from literature to build an adapter based on large kernels convolutions to counter this reduction in ERF. They demonstrate that by doing so, performance improves on multiple pretrained networks against 10 baselines in tasks in classification as well as segmentation. They confirm that the improvement is owing to kernel size and not parameter count and include useful ablation experiments.

The method proposed in the paper is well motivated with a clear analytical section in the first part of the paper with a clear takeaway on ERF reduction in data sparsity. They propose a simple albeit effective method to address this problem and demonstrate that the problem is addressed on a number of pretrained models, datasets and tasks. Their ablation experiments are also useful takeaways in terms of training strategies and architecture design. Overall, the paper forms a positive impression and should be accepted.

### Strengths
1. The work is well motivated with the analysis demonstrating the reduction in performance and tying that to a reduction in the effective receptive field with low data settings. To address this problem a simple large kernel convolution based dual kernel adapter is proposed.
2. The results use multiple pretrained models, a large number of baseline methods and clearly demonstrates that the proposed method works well.
3. The proposed method itself is extremely simple but the novelty lies in it uniquely solving the problem as motivated by the analysis in the first part of the paper.
4. The work is well written and well illustrated with a very efficient use of space to convey the key messages of the paper.

### Weaknesses
1. There is seemingly the idea of reordering the tokens back into the spatial domain that is a part of the method design. However, the authors make no mention of this in the methods section.
2. The work bears similarities to another large kernel adapter method [1] and seems to differ methodologically owing solely to a dual path convolution and the analysis in the first part of the paper. In fact, there also is an extremely similar analysis in [1] titled "Large Kernel Matters Instead of #Trainable Parameters" bearing similarities to the section 4.3 LARGE KERNEL MATTERS INSTEAD OF TRAINABLE PARAMETERS. While the similarities are unfortunate, it does significantly dampen the novelty of this work and positions it more as a derivative of existing work - while being aided by significantly denser analysis compared to [1]. However, this paper should be cited in the section for the single branch comparisons for proper referencing.

References:

[1] Zhu, Ziquan, et al. "LKA: Large Kernel Adapter for Enhanced Medical Image Classification." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2025.

### Questions
1. Does the Dual-Kernel Adapter require a step where the features move from token space into the spatial domain and back again?
2. Can the authors offer any insight into whether their findings will extend into the 3D medical image segmentation space? Are there any restrictions towards the methods being used in such a setting as datasets like BraTS (Table 2) for example are also popular in the 3D segmentation space?
3. Are the authors restricted to the token space in their adaptation? As in [1], should the authors not also be able to target ConvNeXt blocks?

References:

[1] Zhu, Ziquan, et al. "LKA: Large Kernel Adapter for Enhanced Medical Image Classification." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2025.

### Soundness
4

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
4

### Summary
This paper introduces Dual-Kernel Adapter (DKA) — a lightweight, parameter-efficient fine-tuning module designed for medical imaging tasks under extreme data scarcity. The authors first identify that conventional Adapters unexpectedly degrade performance when trained with < 1% of data, performing worse than linear probing due to reduced Effective Receptive Field (ERF).

Comprehensive experiments across multiple classification (COVID, BUSI, ISIC-2019) and segmentation (BRATS, BUSI, ISIC-2018) datasets show DKA consistently outperforms full fine-tuning and other PEFT baselines (AdapterFormer, LoRA, Convpass, AIM).

The paper also presents detailed ERF visualizations, ablations (kernel size, branch design, middle dimension, learning rates), and evidence that performance improvements stem from large-kernel receptive-field expansion rather than parameter count

### Strengths
The empirical findings are compelling and well-supported: quantitative metrics (ACC, mIoU, Dice) improve across all data scales, especially ≤ 1.25%.

The ERF analysis provides strong evidence linking reduced receptive field to Adapter degradation and DKA’s advantage.

The controlled parameter-count experiment convincingly isolates kernel size as the key factor.

### Weaknesses
The theoretical reasoning behind ERF–generalization linkage could be formalized further.

Computational overhead of large-kernel depthwise convolutions is not fully quantified.

Limited theoretical depth: lacks analytical characterization of why ERF → generalization scaling behaves linearly with data size.

Compute trade-off: large-kernel (51×51) convolutions increase FLOPs; energy/memory costs are not discussed.

### Questions
How does DKA scale computationally for high-resolution 3-D MRI or CT volumes?

Can large kernels be replaced by dilated or deformable convolutions to reduce cost?

Have you tested DKA in few-shot fine-tuning (< 10 samples/class) or cross-institutional domain adaptation?

Could self-supervised initialization (e.g., SimCLR) further amplify DKA gains under extreme scarcity?

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
2

### Summary
This paper studies adapter-based fine-tuning of large pretrained models under extreme data scarcity common in medical imaging. It reveals a surprising degradation of conventional adapters below 1% training data, where they perform worse than simple linear probing. The authors attribute this to a sharp reduction in Effective Receptive Field (ERF) under limited supervision. To overcome this, they propose the Dual-Kernel Adapter (DKA), a lightweight module that combines large-kernel (51×51) and small-kernel (5×5) depthwise convolutions in parallel to expand spatial context while preserving local detail. A comprehensive experimental evaluation across six medical imaging datasets (classification and segmentation), multiple backbones, and data regimes confirms that DKA consistently outperforms standard adapters and other parameter-efficient fine-tuning (PEFT) methods, especially in low-data settings.

### Strengths
- Important Problem: Addresses the critical challenge of limited labeled data in medical imaging
- Comprehensive Evaluation: Thorough experiments across 6 datasets, multiple backbones, and various data scales
- Surprising Finding: The observation that adapters can hurt performance under extreme data scarcity is counter-intuitive and valuable
- Strong Empirical Results: DKA shows consistent improvements, particularly in low-data settings
- Extensive Ablations: Thorough analysis of design choices including kernel sizes, learning rates, and architectural variations
- Practical Impact: The method is parameter-efficient and shows minimal computational overhead

### Weaknesses
* Limited Technical Novelty: The solution essentially adds large-kernel convolutions to adapters - this is a straightforward extension rather than a fundamental innovation
* ERF Analysis Concerns:
   * The causal relationship between ERF and performance is assumed but not proven
   * ERF computation methodology needs clarification
   * Alternative explanations (e.g., optimization difficulties, overfitting) are not thoroughly explored
* Kernel Size Selection:
   * The choice of 51×51 kernels seems arbitrary and extreme
   * Limited theoretical justification for why such large kernels are necessary
   * Computational implications of such large kernels in 3D medical imaging are not discussed
* Missing Comparisons:
   * No comparison with recent vision-specific PEFT methods (e.g., Visual Prompt Tuning variants)
   * Limited exploration of other architectural modifications that could expand receptive fields
* Experimental Concerns:
   * Batch size of 1 for segmentation might affect optimization dynamics
   * The asynchronous learning rate finding (1e-3 for adapter, 1e-4 for head) seems important but is relegated to ablations
* Dataset Size Discrepancy Makes Comparisons Misleading:
   * Critical Issue: Figure 1 compares performance across datasets using percentages (0.63%, 1.25%, etc.) but these datasets have vastly different absolute sizes. For example, Tiny ImageNet has 100,000 training images while COVID has only 3,600. This means 0.63% of Tiny ImageNet (630 images) could be larger than 25% of COVID (~900 images).
   * This makes the comparison fundamentally flawed - the paper is comparing different absolute data quantities while claiming to study "low-data" regimes
   * The authors should report absolute sample numbers and normalize comparisons appropriately
* Missing Critical Implementation Details:
   * LoRA Rank: The paper doesn't specify what rank was used for LoRA, which is crucial as rank fundamentally determines LoRA's capacity and parameter count. A rank-1 LoRA vs rank-64 LoRA are completely different methods.
   * Without this information, the comparison with LoRA is meaningless
* Unfair Parameter Comparison:
   * While the paper claims DKA adds <2% parameters, there's no detailed comparison of parameter efficiency
   * The paper should provide:
      * Exact parameter counts for each method
      * Performance vs. parameter trade-off curves
      * Comparison at iso-parameter budgets (e.g., what if LoRA used higher rank to match DKA's parameters?)
   * Table in Figure 5 attempts this but only varies hidden dimensions, not comprehensively
* Limited Applicability to Real Medical Imaging:
   * The 51×51 kernel design is completely impractical for 3D medical imaging (MRI, CT scans)

### Questions
* Dataset Size Normalization: Can you provide a table showing the absolute number of training samples at each percentage for each dataset? How do results change when comparing at equal absolute sample sizes rather than percentages?
* LoRA Configuration: What rank was used for LoRA in all experiments? Have you tried increasing LoRA rank to match DKA's parameter count?
* Fair Comparison: Can you provide a comprehensive parameter-performance trade-off analysis where all methods are compared at multiple parameter budgets?
* ERF Causality: How can you demonstrate that ERF reduction is the causal factor rather than just correlated with poor performance?
* Optimization Dynamics: Given the batch size constraints (especially batch=1 for segmentation), how much of the performance difference could be attributed to optimization difficulties?
* Alternative Architectures: Have you explored other ways to increase receptive fields, such as self-attention modules within adapters or hierarchical designs?
* Theoretical Understanding: Can you provide more theoretical insight into why standard adapters fail in low-data regimes beyond the ERF hypothesis?
* Hyperparameter Sensitivity: How sensitive is DKA to the specific kernel size choices? Would adaptive or learnable kernel sizes be beneficial?
* Computational Cost: What is the actual inference time and memory overhead for 51×51 convolutions on high-resolution medical images?

### Soundness
2

### Presentation
2

### Contribution
2
