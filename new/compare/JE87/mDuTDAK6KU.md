# Review

## Summary
The paper introduces KOALA (KL–Lo Adversarial detection via Label Agreement), a novel, semantics-free adversarial detector designed to identify adversarial attacks on deep neural networks without requiring architectural changes or adversarial retraining. KOALA operates by detecting discrepancies between class predictions derived from two complementary similarity metrics: KL divergence and an L0-based similarity. The KL divergence metric is sensitive to dense, low-amplitude shifts, while the L0-based similarity is designed for sparse, high-impact changes. The authors provide a formal proof of correctness for their approach and demonstrate KOALA's effectiveness through extensive experiments on ResNet/CIFAR-10 and CLIP/Tiny-ImageNet, achieving precision and recall scores that validate its theoretical claims.

## Soundness
2

## Presentation
3

## Contribution
2

## Strengths
1. The paper presents a novel approach to adversarial detection that leverages the disagreement between KL divergence and L0-based predictions, which is a unique and innovative method in the field.

2. The authors provide a theoretical proof of correctness for their approach, defining the explicit conditions under which the disagreement between the two metrics is guaranteed to occur. This adds a strong mathematical foundation to the proposed method.

3. The training process for KOALA is lightweight and does not require architectural changes or adversarial examples, making it easy to integrate into existing models and data modalities.

4. The experimental results demonstrate that KOALA consistently and effectively detects adversarial examples on the ResNet/CIFAR-10 and CLIP/Tiny-ImageNet datasets, with high precision and recall scores.

## Weaknesses
1. The paper only evaluates KOALA on two specific datasets and architectures (ResNet/CIFAR-10 and CLIP/Tiny-ImageNet). It would be beneficial to see how the method performs on a wider range of datasets and model architectures to assess its generalizability.

2. The paper does not extensively explore the potential for adaptive attacks that specifically target the KOALA detector. It would be interesting to see how KOALA performs against attacks that are designed to circumvent its detection mechanism.

3. While the paper claims that KOALA is a lightweight solution, it would be helpful to see a more detailed analysis of the computational overhead introduced by the detector, especially in comparison to other adversarial detection methods.

4. The paper does not provide a thorough comparison with existing state-of-the-art adversarial detection methods in terms of both performance and computational efficiency. Including such a comparison would help to better position KOALA within the current landscape of adversarial detection techniques.

## Questions
See Weaknesses

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4