# Review

## Summary
This paper proposes VQ-Transplant, a framework that enables plug-and-play integration of new VQ modules into frozen, pre-trained tokenizers by replacing their native VQ modules. The proposed transplantation process preserves all encoder-decoder parameters, obviating the need for costly end-to-end retraining when modifying the quantization method. To mitigate decoder-quantization mismatch, this paper introduces a lightweight decoder adaptation strategy (trained for only 5 epochs on ImageNet-1k) to align feature priors with the new quantization space. The paper shows that VQ-Transplant allows obtaining near state-of-the-art reconstruction fidelity for industry-level models like VAR while reducing the training cost by 95%.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
- The paper is well-written and easy to follow.
- The paper proposes a novel framework, VQ-Transplant, which enables plug-and-play integration of new VQ modules into frozen, pre-trained tokenizers. This framework addresses the limitation of requiring significant computational resources for training quantization modules in VQ-based models.
- The paper introduces MMD VQ, a novel VQ method that leverages maximum mean discrepancy to achieve distributional alignment. MMD VQ demonstrates superior reconstruction fidelity compared to the vanilla VAR approach.
- The paper demonstrates the effectiveness of VQ-Transplant by evaluating multiple VQ approaches for visual tokenization tasks. The results show that VQ-Transplant achieves superior reconstruction fidelity while being significantly faster than training vanilla VAR.

## Weaknesses
- The paper primarily evaluates VQ-Transplant on a limited number of pre-trained tokenizers, such as VAR. It would be beneficial to evaluate VQ-Transplant on a broader range of pre-trained tokenizers to demonstrate its generalizability.
- The paper focuses on image reconstruction tasks. It would be interesting to explore the performance of VQ-Transplant on other tasks such as image generation, classification, or segmentation.

## Questions
- How does the performance of VQ-Transplant vary with different pre-trained tokenizers? Can VQ-Transplant be applied to a wider range of pre-trained tokenizers beyond VAR?
- How does the performance of VQ-Transplant vary with different VQ modules? Can VQ-Transplant be used to integrate VQ modules with different architectures or algorithms?
- How does the performance of VQ-Transplant vary with different datasets? Can VQ-Transplant be used to integrate pre-trained tokenizers trained on different datasets?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
4