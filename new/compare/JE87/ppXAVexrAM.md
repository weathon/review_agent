# Review

## Summary
This paper proposes a novel framework for novel view synthesis from a single image by leveraging a GPT-style decoder-only autoregressive model. The proposed method employs a video tokenizer to map continuous image sequences into discrete tokens and a camera encoder that converts camera trajectories into 3D positional guidance. The author claims that the proposed method achieves overall comparable to state-of-the-art view synthesis approaches based on diffusion models.

## Soundness
2

## Presentation
3

## Contribution
2

## Strengths
1. The paper is well-written and easy to follow.
2. The proposed method is well-motivated and the author provides a clear explanation of the proposed method.
3. The author provides a comprehensive evaluation of the proposed method on several public datasets and achieves comparable results to state-of-the-art methods.

## Weaknesses
1. The proposed method is a simple application of autoregressive image generation models to novel view synthesis, which lacks novelty. The video tokenizer and camera encoder are all from existing works and the autoregressive transformer module is a standard transformer with causal attention masks.
2. The proposed method is trained on RealEstate10K and ACID datasets, while the compared diffusion-based methods are mostly trained on large-scale datasets with much more data, such as MipNeRF360 and InfiniteNature. Therefore, the comparison is unfair and the performance of the proposed method is not convincing.
3. The author claims that the proposed method achieves overall comparable to state-of-the-art view synthesis approaches based on diffusion models. However, according to the quantitative results in Table 1, the proposed method is only slightly better or worse than the compared methods. And from the qualitative results in Figure 3, the generated images by the proposed method are still blurry and contain artifacts, which is worse than the compared methods.

## Questions
See weaknesses.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4