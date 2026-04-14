# BAME: Block-Aware Mask Evolution for Efficient N:M Sparse Training

- Decision: Reject
- Scores: 5, 5, 5, 5

## Abstract
N:M sparsity stands as a progressively important tool for DNN compression, achieving practical speedups by stipulating at most N non-zero components within M sequential weights. Unfortunately, most existing works identify the N:M sparse mask through dense backward propagation to update all weights, which incurs exorbitant training costs. In this paper, we introduce BAME, a method that maintains consistent sparsity throughout the N:M sparse training process. BAME perpetually keeps both sparse forward and backward propagation, while iteratively performing weight pruning-and-regrowing within designated weight blocks to tailor the N:M mask. These blocks are selected through a joint assessment based on accumulated mask oscillation frequency and expected loss reduction of mask adaptation, thereby ensuring stable and efficient identification of the optimal N:M mask. Our empirical results substantiate the effectiveness of BAME, illustrating it performs comparably to or better than previous works that fully maintaining dense backward propagation during training. For instance, BAME attains a 72.0% top-1 accuracy while training a 1:16 sparse ResNet-50 on ImageNet, eclipsing SR-STE by 0.5%, despite achieving 2.37$\times$ training FLOPs reduction. Code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This work proposes a new N:M sparse training method called BAME. Since previous methods identify the N:M sparse mask through dense backward propagation to update all weights, which incurs exorbitant training costs, BAME proposes a Block-Aware Mask Evolution  technique with two important components to reduce the training cost while preserving the performance. First, it introduces Loss-aware Mask Adaption (LMA) to compute the N:M sparse mask during training. Then, it utilizes Oscillation-aware Block Freezing (OBF) to reduce the changing frequency of mask and stabilizes the training process. Extensive experiments are conducted to verify the effectiveness of the proposed method.

### Strengths
The strengths are listed as below:
(1) This work is easy to follow and the idea is clearly stated.
(2) Extensive experiments are conducted. And the experimental results are good.
(3) Mathematical formulations are detailed.

### Weaknesses
The weaknesses are listed as below:
(1) Several important SOTAs are missed during performance comparisons. For instance, MaxQ seems to outperform BAME but it needs more training time.
(2) The visualizations specifically the figures can be further improved.
(3) There are several typos: 1. in the caption of figure 1, "Loss-Aware Mask adaption (LMA)" should be "Loss-aware Mask Adaption (LMA)"; 2. in line 290, "c+1" should be "c-1" to be consistent with the equations.

### Questions
1. Does BAME first propose loss-aware mask computation? Or other methods have done in the similar way.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper studies dynamic sparse training with N: M sparsity. Current methods still require computing dense gradients or dense back-propagation. To address this issue, this paper proposes BAME, which performs loss-aware mask adaptation to update weights in pruning-and-growing paradigm. Besides, this paper proposes oscillation-aware block freezing to further stabilize the sparse training process. Experiments on CIFAR and ImageNet with diverse networks demonstrate the effectiveness of the proposed method with much less training FLOPs.

### Strengths
1. This paper is well-motivated. Existing dynamic sparse training with N:M sparsity needs to  calculate dense gradients and back-propagation. The proposed method aims to accelerate both forward and backward pass. 

2. The mathematical analysis behind loss-aware mask adaption is solid.

### Weaknesses
1. My biggest concern of this paper is the efficiency metrics. While this paper reports training FLOPs to demonstrate the efficiency of the proposed model, presenting the actual acceleration rate on hardware would provide stronger evidence of its effectiveness. As discussed, N:M sparsity can balance the dual goals of acceleration and performance. However, reporting only FLOPs does not highlight the advantage of N:M sparsity, as unstructured sparsity could potentially achieve greater FLOP reductions.


2. Given a linear forward pass $Y=WX$, the backward pass requires using the transposed matrix $W^T$ [1], and even kernel-wise transpose in the backward of CNN [2] . This paper claims that "BAME perpetually keeps both sparse forward and backward propagation" (Line 017), however, without the transposable constraints in [1], I am curious how does the proposed method ensure the identified N:M mask from $W$ with BAME in forward pass can be applied to accelerate backward pass. 



[1] Hubara, Itay, et al. "Accelerated sparse neural training: A provable and efficient method to find n: m transposable masks." Advances in neural information processing systems 34 (2021): 21099-21111.

[2] Ma, Haoyu, et al. "HRBP: Hardware-friendly Regrouping towards Block-based Pruning for Sparse CNN Training." Conference on Parsimony and Learning. PMLR, 2024.

### Questions
Please address the above weaknesses in the author response.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The article introduces a novel method, BAME, which maintains consistent sparsity throughout the N:M sparse training process. In detail, BAME perpetually keeps both sparse forward and backward propagation, reducing the need for dense backward propagation that incurs high training costs. This results in a more efficient training process.

### Strengths
1.	The method ensures stable and efficient selection of the optimal N:M mask by choosing blocks based on mask oscillation frequency and expected loss reduction.
2.	Strong empirical evidence supports the effectiveness of the BAME method, as demonstrated in the paper's results.
3.	The paper is well-written and easy to comprehend, enhancing its accessibility and clarity for readers.

### Weaknesses
1.	The paper's innovation is limited, as the proposed Loss-Aware Mask Adaption and Oscillation-Aware Block Freezing methods appear somewhat naive, and the overall algorithmic process is relatively straightforward.
2.	The experimental comparisons in the paper are not sufficiently comprehensive, lacking comparisons with the latest SOTA methods, like MaxQ [1]. Additionally, the performance improvements are modest, with the MaxQ method achieving a 74.0% top-1 accuracy while training a 1:16 sparse ResNet-50 on ImageNet.
3.	The paper lacks an adequate number of figures, and the framework diagrams are not very clear, which may impact the reader's understanding of the content.

Reference: 
[1] Xiang J, Li S, Chen J, et al. MaxQ: Multi-Axis Query for N: M Sparsity Network[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 15845-15854.

### Questions
1.	Given that N:M Sparse Training of Large Language Models is time-consuming and more resource-intensive compared to traditional DNN networks, could the effectiveness of the proposed method be validated on LLM models, such as LLAVA?
2.	The experiments in Table 3 demonstrate that relying solely on the BAME method with Loss-Aware Mask Adaption over 300 epochs does not yield optimal performance. It still requires initial tuning using the most basic gradient-based pruning method. Does this suggest that the dynamic adjustment of masks based on loss-awareness is highly unstable?
3.	The paper primarily focuses on comparing the reduction in GFLOPS during the training phase. However, the numerous mask adjustment operations required in the BAME process—won't these affect the model's training speed? Since the core objective of the paper is to accelerate the N:M sparse training process, should the acceleration effect on actual hardware also be compared and evaluated?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper introduces BAME (Block-Aware Mask Evolution), a novel approach designed to enhance the efficiency of training Deep Neural Networks (DNNs) under N:M sparsity constraints. This method aims to reduce the computational burden associated with traditional dense backward propagation techniques while maintaining or improving model performance.

There are some key contributions of the BAME paper:
- Consistent Sparsity Maintenance: BAME ensures that both forward and backward propagation processes maintain consistent sparsity throughout training. This is achieved by employing a block-aware pruning and regrowing strategy that adapts the N:M mask iteratively.
- Loss-Aware Mask Adaptation (LMA): The paper proposes a loss-aware mechanism for adapting sparse masks, which selectively prunes and revives weights based on their impact on the training loss. This adaptation occurs infrequently, reducing the need for dense gradient calculations and thus lowering training costs.
- Oscillation-Aware Block Freezing (OBF): To stabilize training, BAME incorporates a mechanism that identifies blocks of weights exhibiting high-frequency mask oscillations and restricts updates to these blocks. This helps in maintaining a stable optimization landscape, which is crucial for effective training.

### Strengths
1. The paper is well-structured and clearly articulates its objectives, methodologies, and findings. The introduction effectively sets the stage for the problem of N:M sparsity and the limitations of current approaches, leading logically into the proposed solution with BAME.

2. The BAME method introduces a novel approach to maintaining consistent sparsity during both forward and backward propagation in neural network training. This contrasts sharply with existing methods that rely on dense backward propagation for weight updates, which can be computationally expensive. By employing block-aware mask evolution and loss-aware mask adaptation, BAME innovatively provides the potential to reduce the training burden while optimizing model performance.

### Weaknesses
1. The empirical results primarily focus on specific architectures like ResResNet-32, ResNet-50, MobileNet-V2 and DeiT-Small. This raises concerns about the generalizability of BAME across different neural network architectures, for example recurrent neural networks, mamba, or more complex transformer structure other then DeiT-Small. Another concern is about how BAME's effectiveness scales within a model family. For example, providing the BAME results from DeiT-tiny/small/base or Swin-tiny/small/base/large will be more convincing.

2. The paper does not provide a detailed analysis of how BAME performs across various components of the model structure. While it demonstrates overall effectiveness, it lacks granularity in assessing how different layers or types of operations (e.g., convolutional vs. fully connected layers, residual connection, attention, etc.) respond to the BAME approach.

3. This paper also suffers from the limited results on single classification task. I would suggest benchmarking against more complex tasks can provide insights into how well BAME scales with increasing task complexity.

4. The dynamics and stability of mask evolution are mentioned but not deeply analyzed. Understanding how different parameters (e.g., block size, pruning thresholds) affect performance could provide valuable insights for practitioners.

5. The proposed method involves complex mechanisms such as oscillation-aware block freezing and loss-aware mask adaptation, which may pose challenges for practical implementation to use the Sparse Tensor Core for training acceleration.

### Questions
1. How does BAME scale with larger models or datasets? Are there any observed limitations when applying it to more complex architectures or tasks?

2. How does BAME perform across different layers of the neural network? Are there specific layers where BAME is more or less effective?

3. Can you elaborate on the criteria used for selecting blocks for mask evolution? How do you determine the optimal balance between pruning and reviving weights?

### Soundness
2

### Presentation
3

### Contribution
2
