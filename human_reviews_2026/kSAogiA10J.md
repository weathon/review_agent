# Towards Differential Handling of Various Blur Regions for Accurate Image Deblurring

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Image deblurring aims to restore high-quality images by removing undesired degradation. Although existing methods have yielded promising results, they either overlook the varying degrees of degradation across different regions of the blurred image. In this paper, we propose a differential handling network (DHNet) to perform differential processing for different blur regions. Specifically, we design a Volterra block (VBlock) to incorporate nonlinear characteristics into the deblurring network, enabling it to map complex input-output relationships without relying on nonlinear activation functions. To enable the model to adaptively address varying degradation degrees in blurred regions, we devise the degradation degree recognition expert module (DDRE). This module initially incorporates prior knowledge from a well-trained model to estimate spatially variable blur information. Consequently, the router can map the learned degradation representation and allocate weights to experts according to both the degree of degradation and the size of the regions. Comprehensive experimental results show that DHNet effectively surpasses state-of-the-art (SOTA) methods on both synthetic and real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a differential handling network (DHNet) for image deblurring. The authors design a Volterra block (VBlock) to integrate the non-linear characteristics into the deblurring network without relying on stacking the number of nonlinear activation functions. In addition, the authors propose the degradation degree recognition expert module (DDRE) that initially incorporates prior knowledge from a pre-trained deblurring network and then utilizes several learnable routers and experts to estimate spatial variable blur information. The proposed DHNet achieves SOTA performance while requiring less computation cost in MACs and inference time compared to the existing methods.

### Strengths
1. The proposed DHNet provides an efficient solution for image deblurring. Unlike previous methods that stacke multiple non-liner layers, DHNet utilizes the Volterra block (VBlock) to incorporate non-linear characteristics directly into the network. In addition, DHNet proposes degree recognition expert module (DDRE) that utilizes learnable routers and experts to estimate spatial variable blur information. Experimental results demonstrate that these two components significantly enhance deblurring performance.

2. The proposed DHNet achieve SOTA results on four datasets, including GoPro, HIDE, RealBlur-J, and RealBlur-R, while requiring fewer computational resources in terms of MACs.

### Weaknesses
1. The concept of the Volterra block (VBlock) is almost identical to the MR-VNN block in MR-VNet [A]. Furthermore, the introduction section of the MR-VNet paper states, "We also showcase that the recently proposed Non-linear Activation Free Networks (NAFNet) [B] are a special case of the Volterra Formulation." This indicates that MR-VNN has already demonstrated the advantages of the Volterra Formulation over traditional non-linear networks. Consequently, the proposed Volterra block in this paper lacks novelty.

2. Apart from MACs and inference time, the authors should also compare the number of parameters in Table 8.

3. As shown in Table 1, DHNet-B that uses UFPNet significantly outperform DHNet. This demonstrates that the main performance gain is from the pre-trained UFPNet.


[A] MR-VNet: Media Restoration using Volterra Networks. In CVPR 2024.
[B] Simple Baselines for Image Restoration. In ECCV2022.

### Questions
1. The proposed Volterra block (VBlock) appears to be identical to the MR-VNN block in MR-VNet [A]. It seems that the authors may have directly adopted the MR-VNN block as the VBlock. Could you provide a detailed clarification on this?

2. Apart from MACs and inference time, the authors should also compare the number of parameters in Table 8.

3. The proposed DHNet-B heavily relies on the pre-trained UFPNet. However, UFPNet is a large model with a substantial number of parameters. Is there any alternative pre-trained model or a more lightweight approach that could be used instead?

[A] MR-VNet: Media Restoration using Volterra Networks. In CVPR 2024.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper “MBMamba: When Memory Buffer Meets Mamba for Structure-Aware Image Deblurring” aims to enhance the spatial structure modeling capability of Mamba-based image deblurring frameworks. The authors point out that the traditional Mamba’s “flatten-and-scan” strategy leads to local pixel forgetting and channel redundancy, thereby weakening spatial feature aggregation. To address this issue, they propose the MBMamba framework, whose core component is the Memory Buffering Mechanism (MemVSSM). This module preserves historical feature information through a memory buffer and fuses it via a Feature Cross-Attention Module (FCAM), strengthening local context modeling. In addition, an Ising-inspired regularization loss is designed to simulate pixel-level “mutual attraction” energy minimization, thereby maintaining structural consistency.

### Strengths
The MemVSSM module is designed to enhance local context preservation via feature partitioning and temporal buffering. The physically inspired Ising Loss is incorporated into the deblurring loss function, enforcing spatial coherence through energy minimization. Notably, this module extends the spatial-awareness capability of the Mamba backbone without altering its main architecture, achieving both efficiency and accuracy.

The proposed method provides a new perspective on memory-based state-space modeling in visual tasks. It achieves superior results across four major benchmarks — GoPro, HIDE, RealBlur-R, and RealBlur-J — while maintaining significantly lower FLOPs and inference latency compared to Transformer-based methods.

### Weaknesses
Furthermore, the title and abstract also stress “Various Blur Regions”, but the experiments do not demonstrate how this claim is achieved. There is no explanation of how the network distinguishes between different blur regions, nor how it dynamically adapts processing across regions with varying blur intensities.

Although VBlock refines the Volterra structure from MR-VNet, the core idea of using higher-order kernels to model nonlinearities is not novel. The paper does not sufficiently explain why DHNet achieves better stability or generalization compared to prior Volterra-based networks.

While the paper presents multiple theorems (Theorem 1–4) demonstrating that the Volterra kernel can approximate any continuous nonlinear function, it lacks experimental or visual evidence quantifying these properties—for instance, comparisons of Volterra versus ReLU in terms of convergence speed or stability. As a result, the theoretical claims are not tightly connected to empirical performance.

The DDRE module involves multiple routers and experts, yet only a “single-router case” is visualized. The overall routing mechanism and computational process are not clearly illustrated. Moreover, the weight assignment among experts lacks interpretability and visualization, making the internal logic of this module difficult to fully grasp.

Finally, the paper mainly compares DHNet with CNN-based and a few Transformer-based methods, but does not discuss whether DHNet can truly replace Transformers in capturing global dependencies. For instance, no visualization of global modeling capability is provided. Since DHNet remains primarily convolution-based, its advantage in handling complex dynamic blur still requires further empirical verification.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
I have reviewed this paper in a previous venue. The submission appears to be largely identical to that version. Below, I include my previous review (with possible minor edits for clarity).

This paper proposes DHNet, a new image deblurring model that consists of 1) the VBlock module for incorporating nonlinear characteristics and 2) the DDRE module for dealing with varying degradation degrees in blurred images. The deblurring performance is evaluated across several image deblurring benchmark datasets.

### Strengths
- Experimental results demonstrate that DHNet achieves the SOTA performance with higher efficiency. The deblurring results, both visualization and PSNR/SSIM metrics on commonly used benchmarks like GOPRO and HIDE, are convincing.
- The paper is well-written and easy to follow.

### Weaknesses
- The novelty is limited. 1) The VBlock is borrowed from previous research[1]. Meanwhile, it has already been applied to the image deblurring task by [2]. Clarifying the novelty of VBlock compared with these works is expected. 2) Dealing with varying degradation degrees in image deblurring has been discussed by prior works like [3]. However, these works and the differences are not discussed in the paper.
- In Fig. 2, the x-axis lacks necessary labels, making it difficult to assess the improvement in computational efficiency of the model.
- The ablation study and analysis of the DDRE module are insufficient. Fig. 6 does not convincingly demonstrate that the Experts focus on regions with different degradation degrees (since visualizations of different channels from the output features of a single network layer could also produce similar effects). I recommend that the authors adopt more rigorous quantitative metrics to verify that the focusing regions of different Experts indeed correspond to different degradation degrees.

[1] VolterraNet: A Higher Order Convolutional Network With Group Equivariance for Homogeneous Manifolds. TPAMI 2022.

[2] MR-VNet: Media Restoration using Volterra Networks. CVPR 2024.

[3] Motion deblurring algorithm for wind power inspection images based on Ghostnet and SE attention mechanism. TIP 2023.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1
