# Review

## Summary
This paper presents a method for AI-generated image detection that integrates structural semantic information into existing frameworks. It uses cuboidal partitioning to divide images into sub-regions, extracting statistical differences at each level. These features are combined with those from the AIDE model, enhancing its performance. The approach achieves state-of-the-art results on the GenImage benchmark and demonstrates strong generalization across diverse datasets, underscoring the importance of structural semantics in building robust AIGC detectors.

## Soundness
3

## Presentation
2

## Contribution
2

## Strengths
1. The paper introduces cuboidal partitioning for structural analysis in AI-generated image detection, offering a new perspective in the field.

2. The method demonstrates state-of-the-art performance on the GenImage benchmark, indicating its effectiveness in detecting artifacts from modern diffusion models.

## Weaknesses
1. The approach relies on a combination of the proposed method with the AIDE model, which may limit its applicability to other frameworks or models, potentially constraining its versatility.

2. The method shows weaker performance on the AIGCDetect benchmark and the Chameleon dataset, suggesting that its structural features may not be universally effective across all types of AI-generated images.

3. The paper does not provide a detailed analysis of the computational cost associated with the proposed method, which could be an important consideration for practical applications.

## Questions
1. Can the proposed structural semantic features be integrated with other AI-generated image detection models besides AIDE? If so, what adaptations would be necessary?

2. How does the computational complexity of the cuboidal partitioning and feature extraction compare to other state-of-the-art methods, particularly in large-scale applications?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4