# Review

## Summary
The paper addresses modality imbalance in multimodal learning, where different modalities contribute unequally, leading to suppressed performance in weaker modalities. To tackle this, the authors propose a two-stage Classifier-Constrained Alternating Training (CCAT) framework. First, a shared classifier is pretrained with cross-attention and regularization to reduce modality bias. This classifier is then frozen to stabilize training, while modality-specific LoRA modules are added to allow adaptation for each modality. Additionally, a sample-level imbalance detection optimizes imbalanced samples to boost weaker modalities. Experiments show that CCAT outperforms existing methods on benchmarks, enhancing robustness in multimodal representation learning.

## Soundness
2

## Presentation
2

## Contribution
2

## Strengths
1. The paper is well-written and easy to understand.

2. The paper addresses the issue of modality imbalance, which is a crucial problem in multimodal learning.

3. The proposed method is simple and effective, making it easy to implement.

## Weaknesses
1. The related work section lacks a summary of recent papers on multimodal imbalance learning, such as [1,2,3]. Additionally, the comparison methods in the experiments are not sufficiently up-to-date; it is recommended to include comparisons with the latest papers from 2024.

2. The paper states that "dominant modalities converge faster, steering classifier parameters toward their feature space early in training," but this claim lacks evidence. The authors should provide experimental results or citations to support this assertion.

3. The method proposed by the authors is quite simple and lacks innovation. The two-stage training approach has already been widely used in previous papers, such as [4]. The authors should further elaborate on the novelty of their method.

4. The paper does not provide a detailed explanation of how the modality contribution–oriented regularization mechanism works. The authors should clarify the role of the mutual information (MI) and how it helps mitigate the model’s bias toward specific modalities.

5. The authors should conduct more ablation experiments to verify the effectiveness of the proposed method. For example, they could test the impact of different values of the hyperparameter $\lambda$ in Equation 8.

[1] Pmr: Prototypical modal rebalance for multimodal learning.

[2] Enhancing multimodal cooperation via sample-level modality valuation.

[3] Learning to Rebalance Multimodal Optimization by Adaptively Masking Subnetworks.

[4] Calibrating multimodal learning.

## Questions
See weaknesses.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
5