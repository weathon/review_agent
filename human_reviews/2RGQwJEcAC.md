# Visual Transformer with Differentiable Channel Selection: An Information Bottleneck Inspired Approach

- Decision: Reject
- Scores: 5, 6, 6

## Abstract
Self-attention and transformers have been widely used in deep learning. Recent efforts have been devoted to incorporating transformer blocks into different types of neural architectures, including those with convolutions, leading to various visual transformers for computer vision tasks. In this paper, we propose a novel and compact transformer block, Transformer with Differentiable Channel Selection, or DCS-Transformer. DCS-Transformer features channel selection in the computation of the attention weights and the input/output features of the MLP in the transformer block. Our DCS-Transformer is compatible with many popular and compact transformer networks, such as MobileViT and EfficientViT, and it reduces the FLOPs of the visual transformers while maintaining or even improving the prediction accuracy. In the experiments, we replace all the transformer blocks in MobileViT and EfficientViT with DCS-Transformer blocks, leading to DCS-Transformer networks with different backbones. The DCS-Transformer is motivated by reduction of Information Bottleneck, and a novel upper bound for the IB which can be optimized by SGD is derived and incorporated into the training loss of the network with DCS-Transformer. Extensive results on image classification and object detection evidence that DCS-Transformer renders compact and efficient visual transformers with comparable or much better prediction accuracy than the original visual transformers. The code of DCS-Transformer is available at \url{https://anonymous.4open.science/r/IB-DCS-ViT-273C/}.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The study introduces an innovative and streamlined transformer block named DCS-Transformer, which facilitates channel selection for both attention weights and attention outputs. The inspiration for this channel selection arises from the information bottleneck (IB) principle. This principle strives to diminish the mutual information between the transformer block's input and output, all the while maintaining the mutual information between the output and the task label. This is chiefly realized through the use of a Gumbel-softmax and a channel-pruning loss.

The overall framework mirrors the structure presented in the Neural Architecture Search (NAS), encompassing both a search phase and a training phase. In the quest for the optimal weights for channel selection, the authors put forth an IB-associated loss for the search process. The proficiency of the presented approach is corroborated by experimental findings on ImageNet-1k and COCO.

### Strengths
1. The paper is well written.

2. The introduction of IB loss into Vision Transformers sounds novel.

### Weaknesses
1. The rationale for utilizing an Information Bottleneck loss appears to be somewhat rigid and unclear to me.
- The authors explain that the reason for employing this loss is that


> IB prompts the network to learn features that are more strongly correlated with class labels while decreasing their correlation with the input.


However, it remains unclear to me why a traditional Softmax cross-entropy loss wouldn't be adequate to address this issue effectively.

2. The primary contribution of this paper doesn't seem to be particularly effective.

As per Table 6 in the appendix, the implementation of the proposed IB loss results in only a 0.3% improvement when used on the same backbone. Such a minor improvement could also be attained simply by using a more favourable random seed, which might be too trivial to serve as the main contribution of a ICLR paper.

3. Some of the ablation studies on hyper-parameter search are missing.

I'm intrigued by the roles of the hyper-parameters $\eta$ and $\lambda$ in the suggested approach. It appears that if $\eta$ is not set to a small value, the outcomes could be inferior to the baseline. Upon examining the code, I noticed that $\eta$ is defaulted to 0.1, which contrasts with the paper's assertion that $\eta$ is set to 50 for ImageNet. This discrepancy could potentially be confusing for many readers.

4. The discussion on related works is not comprehensive enough.

This paper introduces some techniques, e.g. channel selection with Gumbel SoftMax, and entropy minimization for architecture search, that were first applied in the field of Neural Architecture Search (NAS) and network pruning. However, the section of related work does not include a subsection in this direction, which is inappropriate from my point of view. Some seminal works like [1, 2, 3] should be included and carefully discussed.

### Reference

[1] Xie S, Zheng H, Liu C, et al. SNAS: stochastic neural architecture search.ICLR 2019.

[2] Herrmann C, Bowen R S, Zabih R. Channel selection using gumbel softmax. ECCV 2020.

[3] Liu H, Simonyan K, Yang Y. Darts: Differentiable architecture search. ICLR 2019

### Questions
see weaknesses

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces a compact transformer architecture exploring the differentiable channel selection. There are two types of channel selection, which are channel selection for attention weights and channel selection for attention outputs. In addition,  the IB loss is employed to boost the performance of the proposed framework. Extensive experiments on image classification and object detection verifies the effectiveness of the proposed method.

### Strengths
1. The paper is well-written and well-motivated. 

2. The design of the channel selection for attention weights and attention outputs make sense, corresponding to the matrix multiplication for attention and MLP. The IB loss is further considered to improve the performance.

3. The comparison with the SOTA pruning methods and compact models show that the proposed method is effective on the mobile devices.

### Weaknesses
1. There are only comparisons of parameters and FLOPs, I wonder the actual inference time of the proposed method.

2. In Figure 2, there are two points for EfficientViT while only one point for DCS-EfficientViT. What's the performance of another point? In another word, is the proposed method still valuable for a larger model?

3. The hyper-parameters are carefully designed such as the temperature etc. I am doubt about the generalization of the proposed method.

### Questions
See weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a DCS mechanism, which achieves network pruning via differentiable channel selection. Specifically, two channel selection strategies have been proposed, that is, channel selection for attention weights and channel selection for attention outputs. To ensure that only informative channels have been selected, the authors incorporate IB loss, which is inspired by the information bottleneck theory. Experiments on image classification and object detection have demonstrated the effectiveness of the proposed method, as well as its generalization on multiple Transformer architectures, including EfficientViT and MobileViT.

### Strengths
Basically, the main contribution of the proposed method is two-fold: a straightforward channel selection mechanism, and the intuitive incorporation of information bottleneck theory. Although both ideas have long been proposed, their combination and utilization in Transformer pruning may still be inspiring, especially to researchers in this specific field. The authors' claims have also been well-supported by the extensive experimental results. The manuscript is generally well-written and the English usage is satisfactory.

### Weaknesses
There are several aspects of this work that could be further improved:

1. The authors may consider focusing more on illustrating their motivation and ideas instead of explaining the technical details. For example, the usage of information bottleneck in the proposed method is not well-introduced. I was expecting to see how the information bottleneck theory is integrated into the proposed architecture and the rationale behind it, but only to fine detailed derivation of the variational upper bound for the IB loss.

2. Is it possible that the propose module be applied to more classical architecture of ViT, e.g., the vanilla ViT or Swin? And what would be performance if DCS is used in semantic segmentation tasks. More experimental results would make the paper more convincing.

3. It seems that the authors fail to compare their method to other pruning techniques, but only show that DCS is effective as it successfully reduce the number of parameters without sacrificing the performance. I also expect a comprehensive comparison against benchmark pruning methods in term of the overall computational cost.

After rebuttal:
I appreciate the detailed response provided by the authors, which have solved much of my concern and lead to the increase of overall rating. I would recommend the authors to integrate the supplementary results they provide in the rebuttal phase into their manuscript, so as to make it more intuitive and convincing.

### Questions
Please refer to the weakness. My concern mainly lies in the experiment section.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
