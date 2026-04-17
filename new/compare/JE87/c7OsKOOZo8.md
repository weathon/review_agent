# Review

## Summary
This paper presents an end-to-end framework for multi-view diabetic retinopathy (DR) grading that reduces dependency on external annotations by generating lesion proposals internally. The authors introduce two modules: the Grade-Activated Lesion Proposal (GALP) module, which produces grade-conditioned evidence maps and selects high-evidence regions as lesion proposals, and the Cross-View Lesion Expert-Guided Regional Fusion (LGRF) module, which activates relevant feature extractors for each view’s proposals based on cross-view context. The method achieves state-of-the-art performance on two multi-view DR datasets without requiring external annotations, demonstrating the effectiveness of self-generated proposals in improving grading accuracy.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The proposed end-to-end framework for multi-view DR grading that generates lesion proposals internally, reducing dependence on costly external annotations.
2. The proposed Grade-Activated Lesion Proposal (GALP) module enhances feature discriminability and produces lesion proposals without external annotations.
3. The proposed Cross-View Lesion Expert-Guided Regional Fusion (LGRF) module enables selective, context-aware cross-view fusion, improving the integration of lesion proposals.
4. The method achieves state-of-the-art performance on two multi-view DR datasets, demonstrating its effectiveness and robustness without relying on external annotations.

## Weaknesses
1. The proposed method seems complex, which may hinder its adoption in practical applications.
2. The method’s performance is sensitive to the selection of hyperparameters, such as the number of lesion proposals and the number of experts, which may require careful tuning for different datasets.
3. The method has been evaluated primarily on multi-view DR datasets. Its generalizability to other ophthalmic diseases or datasets with different characteristics remains to be demonstrated.

## Questions
1. How does the proposed method handle variations in image quality or resolution across different datasets?
2. What is the computational cost of the proposed method compared to existing approaches, particularly for real-time applications?
3. How does the method perform on datasets with different retinal image characteristics or disease prevalence?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4