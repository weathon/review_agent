# SpikeNet: Sparse Spike-Driven Mask Vector Transformer for Energy-Efficient and Stable Spiking Point Cloud Processing

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
The unordered nature of point cloud data poses significant challenges to %traditional analysis tasks. conventional architectures primarily designed for structured data. Spiking neural networks (SNN), by virtue of their inherent sparsity and dynamics, are particularly well-suited for processing point clouds to effectively extract meaningful features.We propose SpikeNet, a novel spiking neural network architecture for energy-efficient and robust point cloud analysis. We introduce spiking-driven sparse attention mechanism coined the Spiking Vector Mask Transformer (SVMT). By dynamically aligning the sparsity of point cloud data through binary spiking masks, SVMT eliminates the need for softmax and multiplication operations, significantly improving computational efficiency. 
We also propose a Dynamic Sparse Spiking Residual (DSSR) structure and integrate it with SVMT to form the Spiking Neural Network (SpikeNet) for point cloud classification and segmentation. SpikeNet overcomes the trade-off between accuracy and efficiency in previous SNN methods, achieving collaborative optimization of performance and energy-efficiency. Experiments on benchmark datasets show that SpikeNet achieves state-of-the-art performance in shape classification and part segmentation tasks, comparable to artificial neural network (ANN) based methods. Our source code is in supplementary material and will be made publicly available

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies the trade-off between accuracy and efficiency in existing Spiking Neural Networks (SNNs) for point cloud processing. To address this, the authors propose SpikeNet, a novel SNN architecture featuring a Spiking Vector Mask Transformer (SVMT) that uses binary spike masks to eliminate costly Softmax and multiplication operations, and a Dynamic Sparse Spiking Residual (DSSR) structure to stabilize training. Experiments show that SpikeNet achieves state-of-the-art performance among SNNs on classification and segmentation tasks, competitive with conventional Artificial Neural Networks (ANNs).

### Strengths
1. The proposed SVMT module is well-justified, as it cleverly leverages binary spiking masks to dynamically align with the native sparsity of point cloud data, providing a natural and efficient alternative to standard attention.

2. This design provides a significant energy-saving advantage by fundamentally eliminating the need for the computationally intensive softmax and multiplication operations that are typical in transformer-based models.

### Weaknesses
The main weakness of this paper is its unclear application value.

1. The authors mainly compare the energy consumption in their experiments. However, in most real-world application, inference latency and memory consumption are more important. The authors should provide these results.

2. How does the energy calculated? Any hardware experiments to validate its impact?

3. The authors only conduct experiments on shape-level 3D analysis, which is too toy to validate the real-world application value. Shape analysis is very different from scene-level perception, so at least experiments on scene segmentation should be provided.

### Questions
See weakness. Shape-level pointcloud analysis is a very toy experimental setting. For pointcloud analysis, scene-level perception is much closer to real application.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes SpikeNet, a spiking neural backbone for point-cloud classification and part segmentation that couples a Dynamic Sparse Spiking Residual (DSSR) module with a Spiking Vector Mask Transformer (SVMT). DSSR is designed to stabilize direct SNN training via bounded surrogate gradients and dynamic sparsity, leading to a provable attenuation of gradient amplification. SVMT replaces dot-product attention with a local per-channel Q–K difference gate and a spike-driven mask, which removes softmax and quadratic token interactions; the resulting attention scales linearly in time–channel–point dimensions and is amenable to neuromorphic implementations.

### Strengths
1. The proposed Spiking Vector Mask Transformer (SVMT) is not a trivial substitution of ReLU with spikes. It redefines attention itself in spike terms: attention weights are derived from spike-domain channelwise differences $\tau(Q,K)=Q-K$ instead of dense dot products and softmax, and sparsification is enforced through a Spike Point Masker that gates either channels or points using binary spike statistics. 
2. The model is benchmarked on ModelNet40, ScanObjectNN, and ShapeNetPart, which together cover CAD-like clean geometry, noisy real-world scans, and fine-grained part-level segmentation. 
3. The ablations are not superficial. They analyze alternative attention operators (sum, product, etc.), masking strategies (channel vs point), and embedding dimensionality. These studies make a credible case that the final architecture is not arbitrary.

### Weaknesses
1. DSSR’s stability argument is persuasive but qualitative. The paper does not report gradient-norm distributions across depth, training-loss curves comparing with/without DSSR, or any theorem giving a global bound across layers and timesteps. Adding a small empirical study here would turn into strong evidence.
2. The Spike Point Masker is described in both pointwise and channelwise forms, but notation occasionally glosses over which axes are being summed (over points $N$, channels $C$, timesteps $T$). This matters because those masks drive sparsity and thus energy savings. 
3. The model has not yet been demonstrated on full outdoor LiDAR scenes, nor is there an ablation of memory scaling in that regime. S3DIS shows meaningful indoor semantics, but the overall mIoU is still behind the best ANN scene-segmentation models.
4. A few phrases imply that SpikeNet “removes multiplications” or is “entirely neuromorphic.” The appendices clarify the nuance: SVMT avoids $QK^\top$ dot-product attention and softmax, but linear projections and masked elementwise products still exist; the encoder/decoder are spike-friendly, but the input embedding and final FC head are still float MAC. These nuances should be stated up front.

### Questions
1. Can you provide per-sample inference latency and measured runtime energy or power traces on any available hardware?
2. Can you share gradient-norm vs depth, or loss-vs-epoch curves, comparing SpikeNet with and without DSSR? This would convert the qualitative stability argument into quantitative evidence.
3. Spike Point Masker details: Please precisely state tensor layout ($T\times C\times N$ or equivalent) and specify along which axes you sum to get channel masks vs point masks. A short pseudocode snippet in the appendix would remove any ambiguity.
4. You state that the encoder/decoder can in principle be mapped to spike-friendly AC-heavy hardware, while the input embedding stem and FC head remain MAC-dominated. Is the intended deployment strategy a hybrid (MAC front/back end around a spike core), or do you envision eventually replacing the stem with spike-native modules as well?

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
4

### Summary
This paper proposes a spiking neural network for point cloud data, named as SpikeNet, It primarily designs a spike attention mechanism on the Spiking Vector Mask Transformer (SVMT) that eliminates the need for multiplication and softmax operations, and introduces a Dynamic Sparse Spiking Residual (DSSR) structure. Experiments demonstrate the superior performance of the proposed model.

### Strengths
1. The paper is well-organized and provides a detailed presentation of the methodology.  
2. The proposed model demonstrates superior performance on both point cloud classification and segmentation tasks.  
3. The models significantly reduce parameter count and energy consumption. The authors also provide code in the supplementary material.

### Weaknesses
1. This work lacks stronger motivation and fails to provide valuable insights, such as those on the proposed spiking attention and DSSR module.

2. In Equation 7, element-wise subtraction is employed to aggregate features S‘^{(l-1)} and E_{va}. I do not understand the rationale for using subtraction here. Both are spike features, and passing their difference through an SNN layer will result in the complete loss of information from E_{va}.

3. Please explain the energy consumption calculation approach. In Table 1. SpikePointNet (ICCV23) exhibits an energy consumption of 1.6 mJ at 0.1 FLOPs, while SpikeNet-N/C consumes only 1.8 mJ at 1.9 FLOPs. This suggests potential inconsistencies in the calculation approaches.

### Questions
The author needs to respond to the Weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SpikeNet, a spiking-driven architecture for point cloud analysis. The authors designed two key modules: (1) Spiking Vector Mask Transformer (SVMT), which uses sparse spike-based queries, keys, and values to dynamically align point cloud data with a binary spike mask; (2) Dynamic Sparse Spiking Residual (DSSR), which used to stabilize SSN training while exploiting temporal sparsity. Extensive experiments on ModelNet40, ScanObjectNN and ShapeNetPart demonstrate the proposed architecture’s superior performance compared to previous SOTA.

### Strengths
1. The paper provides clear motivation and is well organized; the figures are readable and understandable. 

2. The experimental results are quite complete and adequate, and support the author's claims. 

3. The proposed approach is logical and technically sound.

### Weaknesses
1. Energy advantage lacks real hardware validation. The energy efficiency results provided by the authors are estimated through operation models rather than measured on GPU or neuromorphic hardware, which weakens the “energy-efficiency” claim. 

2. ANN baselines could be stronger. In Table1, the authors compare with serveral ANNs, but not include strong recent baselines such as Point Transformer V3/Point-Bert/PointNeXt under the unified training protocol. 

3. Segmentation still cannot beat top ANN SOTAs, such as PointMamba and PCM on ShapeNet, and PointRWKV on  S3DIS dataset. For example, On the S3DIS dataset, although SNN variants are competitive in some classes, mIoU results are below best ANN methods reported in the table. 

4. It seems the authors manually wrote the citation, and the format fails to comply with ICLR’s citation guidance. Please review the guidance.

### Questions
1. Could you clarify how the estimated energy correlates with the used GPU(A6000) or neuromorphic hardware measurements? Have you conducted any experiments on real devices to validate the energy advantage of SpikeNet? 
2. The authors have compared SpikeNet with recent strong SNNs and several Mamba-based models as shown in Table 1 and 2. But, not very recent ANNs such as Point Transformer V3 and PintNeXt are considered. Could the authors provide comparisons to them? 

3. The vector-difference operator τ plays an important role in the SVMT module. Does it capture relative geometric offsets or spike-timing discrepancies?

### Soundness
3

### Presentation
3

### Contribution
3
